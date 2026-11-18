"""Card detection and perspective rectification."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CARD_W_MM = 63.5
CARD_H_MM = 88.9
WORK_MAX = 2000


@dataclass
class Detection:
    ok: bool
    reason: str = ""
    quad: np.ndarray | None = None          # original-image px, tl,tr,br,bl, portrait
    quad_work: np.ndarray | None = None     # same corners in work-image px
    mask_work: np.ndarray | None = None     # filled card silhouette, work scale, 0/255
    work_scale: float = 1.0
    card_width_px: float = 0.0
    card_height_px: float = 0.0
    area_fraction: float = 0.0
    rectangularity: float = 0.0
    warnings: list = field(default_factory=list)
    corners_refined: bool = True


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def order_points(pts) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def _foreground(work: np.ndarray) -> np.ndarray:
    h, w = work.shape[:2]
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, int(0.03 * min(h, w)))
    ring = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    diff = np.linalg.norm(lab - bg, axis=2)
    diff8 = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    diff8 = cv2.GaussianBlur(diff8, (0, 0), 1.5)
    thr, _ = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (diff8 > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _line_through(points: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(p1, d1, p2, d2):
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], float)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, (p2 - p1))
    return p1 + t[0] * d1


def _fit_sides(contour_pts: np.ndarray, quad: np.ndarray, tol: float, t_lo=0.2, t_hi=0.8):
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        seg = (b - a).astype(float)
        length = float(np.linalg.norm(seg))
        if length < 1:
            return None
        u = seg / length
        n = np.array([-u[1], u[0]])
        rel = contour_pts - a
        t = rel @ u / length
        dist = np.abs(rel @ n)
        sel = (t > t_lo) & (t < t_hi) & (dist < tol)
        if sel.sum() < 10:
            return None
        lines.append(_line_through(contour_pts[sel]))
    out = []
    for i in range(4):
        p = _intersect(*lines[i - 1], *lines[i])
        if p is None:
            return None
        out.append(p)
    return np.array(out, np.float32)


def refine_quad(contour_pts: np.ndarray, quad: np.ndarray):
    """Fit a line to each straight side and intersect them, so rounded or damaged die-cut corners do not pull
    the corner estimate inward. Two passes: a wide capture band, then a tight one around the first fit.
    Returns (quad, ok); ok=False means the fit failed and the input quad is returned unchanged."""
    diag = float(np.linalg.norm(quad[2] - quad[0]))
    first = _fit_sides(contour_pts, quad, 0.03 * diag)
    if first is None:
        return quad, False
    second = _fit_sides(contour_pts, first, 0.008 * diag)
    out = second if second is not None else first
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.15 * diag:
        return quad, False
    return out, True


def detect_card(img: np.ndarray) -> Detection:
    h, w = img.shape[:2]
    ws = min(1.0, WORK_MAX / max(h, w))
    work = cv2.resize(img, (int(round(w * ws)), int(round(h * ws))), interpolation=cv2.INTER_AREA) if ws < 1 else img
    wh, ww = work.shape[:2]
    mask = _foreground(work)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return Detection(False, "No object found against the background. Use a plain, contrasting background.")
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    frac = area / float(wh * ww)
    if frac < 0.06:
        return Detection(False, "The card fills too little of the frame (under 6 percent). Move closer or crop.")
    if frac > 0.97:
        return Detection(False, "The card fills the whole frame, so its edges cannot be separated from the background. "
                                "Leave a visible margin of contrasting background around the card.")
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    cpts = cnt.reshape(-1, 2).astype(np.float32)
    inits = []
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            inits.append(order_points(approx.reshape(4, 2).astype(np.float32)))
            break
    inits.append(order_points(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)))
    quad, refined = None, False
    for q0 in inits:
        q, ok = refine_quad(cpts, q0)
        if ok:
            quad, refined = q, True
            break
    if quad is None:
        quad = inits[0]

    top = np.linalg.norm(quad[1] - quad[0]); bot = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0]); right = np.linalg.norm(quad[2] - quad[1])
    if (top + bot) > (left + right):          # landscape: rotate so the long side is vertical
        quad = np.roll(quad, 1, axis=0)
        top, bot, left, right = left, right, bot, top
    qwidth = (top + bot) / 2.0
    qheight = (left + right) / 2.0
    quad_area = cv2.contourArea(quad.astype(np.float32))
    rect_ratio = float(area / quad_area) if quad_area > 0 else 0.0
    warnings = []
    if not refined:
        warnings.append("Card corners could not be located precisely (damaged or unusual outline, or a low-contrast "
                        "background); corner geometry measurements are unreliable for this photo.")
    aspect = qwidth / qheight if qheight else 0
    if abs(aspect - CARD_W_MM / CARD_H_MM) > 0.06:
        warnings.append(f"Detected shape has aspect ratio {aspect:.2f}; a standard card is 0.71. "
                        "The photo may be cropped, the card may be in a sleeve or holder, or detection may be off.")
    if rect_ratio < 0.93:
        warnings.append("The outline is not close to a clean rectangle (shadows, clutter, or a low-contrast edge).")

    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, [cnt], -1, 255, -1)
    quad_orig = quad / ws
    return Detection(
        ok=True, quad=quad_orig.astype(np.float32), quad_work=quad.astype(np.float32),
        mask_work=mask_filled, work_scale=ws,
        card_width_px=float(qwidth / ws), card_height_px=float(qheight / ws),
        area_fraction=float(frac), rectangularity=rect_ratio, warnings=warnings, corners_refined=refined)


def choose_canvas_width(det: Detection) -> int:
    return int(np.clip(round(det.card_width_px), 750, 2000))


def rectify(img: np.ndarray, det: Detection, canvas_w: int, with_transform: bool = False):
    """Returns (rect, mask), or (rect, mask, M) when with_transform: M maps original-photo px to canvas px."""
    canvas_h = int(round(canvas_w * CARD_H_MM / CARD_W_MM))
    dst = np.array([[0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(det.quad.astype(np.float32), dst)
    rect = cv2.warpPerspective(img, m, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mw = cv2.getPerspectiveTransform(det.quad_work.astype(np.float32), dst)
    mask = cv2.warpPerspective(det.mask_work, mw, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if with_transform:
        return rect, mask, m.astype(np.float64)
    return rect, mask
