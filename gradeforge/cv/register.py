"""Image registration across photographs of the same card.

Lighting variants are already rectified to the same canvas from their own card outlines; ECC then removes the
remaining few-pixel misalignment. Macro photographs (different scale and field of view) use ORB feature
matching with RANSAC and are only accepted when enough consistent matches exist.
"""
from __future__ import annotations

import cv2
import numpy as np

ECC_MIN_CC = 0.55
ECC_MAX_RESIDUAL_PX = 2.0
ORB_MIN_INLIERS = 30


def _grad(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)


def register_ecc(template: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> dict:
    """Align `moving` (BGR canvas) to `template` (BGR canvas). Returns a dict with ok, cc, residual_shift_px,
    warp3 (maps template-canvas coords to moving-canvas coords), aligned (moving resampled onto the template)."""
    h, w = template.shape[:2]
    s = min(1.0, 900.0 / w)
    size = (int(w * s), int(h * s))
    gt = _grad(cv2.resize(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32))
    gm = _grad(cv2.resize(cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32))
    msk = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    msk = cv2.erode((msk > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(gt, gm, warp, cv2.MOTION_AFFINE, crit, msk, 5)
    except cv2.error as exc:
        return {"ok": False, "reason": f"ECC did not converge ({str(exc).splitlines()[0][:80]})", "cc": 0.0}
    full = warp.astype(np.float64).copy()
    full[:, 2] /= s
    aligned = cv2.warpAffine(moving, full, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), full, (w, h),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ga = _grad(cv2.resize(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32))
    (dx, dy), response = cv2.phaseCorrelate(gt.astype(np.float64), ga.astype(np.float64))
    residual = float(np.hypot(dx, dy) / s)
    warp3 = np.vstack([full, [0, 0, 1]])
    ok = bool(cc >= ECC_MIN_CC and residual <= ECC_MAX_RESIDUAL_PX)
    reason = "" if ok else (f"registration quality too low (cc={cc:.2f}, residual {residual:.1f} px)")
    return {"ok": ok, "reason": reason, "cc": float(cc), "residual_shift_px": residual,
            "warp3": warp3, "aligned": aligned, "valid": valid,
            "to_template": np.linalg.inv(warp3)}


def register_orb(template: np.ndarray, moving: np.ndarray, scales=(1.0, 0.5, 0.33, 0.25)) -> dict:
    """Homography mapping `moving` pixels to `template` pixels via ORB + RANSAC; tries several moving-image scales."""
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gt = clahe.apply(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
    kt, dt = orb.detectAndCompute(gt, None)
    best = {"ok": False, "inliers": 0, "reason": "no consistent feature matches (needs textured artwork in view)"}
    if dt is None or len(kt) < 30:
        return best
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for s in scales:
        m = cv2.resize(moving, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else moving
        gm = clahe.apply(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
        km, dm = orb.detectAndCompute(gm, None)
        if dm is None or len(km) < 30:
            continue
        matches = sorted(bf.match(dm, dt), key=lambda x: x.distance)[:400]
        if len(matches) < ORB_MIN_INLIERS:
            continue
        src = np.float32([km[x.queryIdx].pt for x in matches]).reshape(-1, 1, 2)
        dst = np.float32([kt[x.trainIdx].pt for x in matches]).reshape(-1, 1, 2)
        hom, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if hom is None:
            continue
        n = int(inl.sum())
        if n > best["inliers"]:
            scale_m = np.diag([s, s, 1.0])
            best = {"ok": n >= ORB_MIN_INLIERS, "inliers": n, "scale_tried": s,
                    "to_template": hom @ scale_m,
                    "reason": "" if n >= ORB_MIN_INLIERS else f"only {n} consistent matches"}
    return best
