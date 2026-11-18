"""Synthetic card photographs with KNOWN ground truth, for testing the CV pipeline.

These are test fixtures, not evidence about real PSA outcomes.
"""
from __future__ import annotations

import cv2
import numpy as np

CW, CH = 1000, 1400
BG = (120, 140, 110)


def rounded_mask(w=CW, h=CH, r=50, radii=None):
    """Card silhouette. radii=(tl, tr, br, bl) overrides the uniform radius r per corner."""
    tl, tr, br, bl = radii if radii else (r, r, r, r)
    m = np.full((h, w), 255, np.uint8)
    for (rad, cx, cy, x0, y0, x1, y1) in [
            (tl, tl, tl, 0, 0, tl, tl), (tr, w - 1 - tr, tr, w - 1 - tr, 0, w, tr),
            (br, w - 1 - br, h - 1 - br, w - 1 - br, h - 1 - br, w, h),
            (bl, bl, h - 1 - bl, 0, h - 1 - bl, bl, h)]:
        corner = np.zeros((h, w), np.uint8)
        cv2.circle(corner, (cx, cy), rad, 255, -1)
        region = np.zeros((h, w), bool)
        region[y0:y1, x0:x1] = True
        m[region & (corner == 0)] = 0
    return m


def make_card(lr=(60, 60), tb=(60, 60), border=(30, 30, 30), radius=50, seed=0, radii=None):
    """Returns (card_bgr 1000x1400, mask). lr/tb are border widths in canvas px."""
    rng = np.random.default_rng(seed)
    card = np.zeros((CH, CW, 3), np.uint8)
    card[:] = border
    l, r = lr
    t, b = tb
    art = card[t:CH - b, l:CW - r]
    ah, aw = art.shape[:2]
    gx = np.linspace(0, 1, aw)[None, :, None]
    gy = np.linspace(0, 1, ah)[:, None, None]
    art[:] = (np.clip(60 + 150 * gx + 30 * gy, 0, 255) * np.array([1.0, 0.7, 0.4])[None, None, :]).astype(np.uint8)
    for _ in range(25):
        x0, y0 = int(rng.integers(0, aw - 80)), int(rng.integers(0, ah - 80))
        cv2.rectangle(art, (x0, y0), (x0 + int(rng.integers(30, 80)), y0 + int(rng.integers(30, 80))),
                      tuple(int(v) for v in rng.integers(40, 250, 3)), -1)
    mask = rounded_mask(r=radius, radii=radii)
    return card, mask


def whiten(card, mask, region, depth=10, value=235):
    """Paint card pixels within `depth` px of the silhouette edge, inside region=(y0,y1,x0,x1), near-white."""
    padded = cv2.copyMakeBorder((mask > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    y0, y1, x0, x1 = region
    sel = np.zeros(mask.shape, bool)
    sel[y0:y1, x0:x1] = True
    sel &= (dist <= depth) & (mask > 0)
    card[sel] = value
    return card


def draw_scratch(card, p0, p1, value=(235, 235, 235), thickness=2):
    cv2.line(card, p0, p1, value, thickness, cv2.LINE_AA)
    return card


def photograph(card, mask, photo_size=(1600, 2100), angle=3.0, tilt=0.0, scale=1.1, noise=1.5, blur=0.8,
               seed=1, glare_blob=None):
    """Place the card on a background with rotation/perspective, add noise/blur, optional glare ellipse."""
    rng = np.random.default_rng(seed)
    pw, ph = photo_size
    cw, chh = card.shape[1], card.shape[0]
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, chh - 1], [0, chh - 1]], np.float32)
    cx, cy = pw / 2, ph / 2
    sw, sh = cw * scale / 2, chh * scale / 2
    dst = np.array([[-sw, -sh], [sw, -sh], [sw * (1 - tilt), sh], [-sw * (1 - tilt), sh]], np.float32)
    ang = np.deg2rad(angle)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = dst @ rot.T + np.array([cx, cy], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    bg = np.zeros((ph, pw, 3), np.uint8)
    bg[:] = BG
    warped = cv2.warpPerspective(card, m, (pw, ph), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpPerspective(mask, m, (pw, ph), flags=cv2.INTER_LINEAR)
    a = (wmask.astype(np.float32) / 255.0)[..., None]
    photo = (warped * a + bg * (1 - a)).astype(np.float32)
    if glare_blob:
        gx, gy, ax, ay = glare_blob
        layer = np.zeros((ph, pw), np.float32)
        cv2.ellipse(layer, (gx, gy), (ax, ay), 0, 0, 360, 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 6)
        photo = photo * (1 - layer[..., None]) + 255.0 * layer[..., None]
    if blur:
        photo = cv2.GaussianBlur(photo, (0, 0), blur)
    photo += rng.normal(0, noise, photo.shape)
    return np.clip(photo, 0, 255).astype(np.uint8)


def encode(img, quality=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------------------------
# Lighting-variant scenes. Physical model: albedo features (artwork, stains, ink) are identical in every
# light; relief features respond to light angle (scratch = specular line with per-light gain, dent = shading
# whose sign follows the light direction); glare is a saturated blob that moves between photos.
# ---------------------------------------------------------------------------------------------

def add_stain(card, cx, cy, rx, ry, color=(60, 120, 150), strength=0.6):
    """Albedo stain: soft ellipse blended toward a brown/yellow tint (same in every light)."""
    h, w = card.shape[:2]
    layer = np.zeros((h, w), np.float32)
    cv2.ellipse(layer, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(2.0, min(rx, ry) / 4)) * strength
    tint = np.array(color, np.float32)[None, None, :]
    out = card.astype(np.float32) * (1 - layer[..., None]) + tint * layer[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def lit(card, ramp=(0.0, 0.0), scratch=None, dent=None, gloss=None):
    """Apply per-light effects to the albedo card.
    scratch=(p0, p1, gain, thickness): specular line, brightness gain `gain` (can be 0 or negative).
    dent=(cx, cy, sigma, amplitude, direction): Gaussian bump; shading = direction * amplitude * d(bump)/dx.
    gloss=(cx, cy, rx, ry, gain): broad specular patch."""
    h, w = card.shape[:2]
    img = card.astype(np.float32)
    xs = np.linspace(-0.5, 0.5, w)[None, :, None]
    ys = np.linspace(-0.5, 0.5, h)[:, None, None]
    img = img * (1 + ramp[0] * xs + ramp[1] * ys)
    add = np.zeros((h, w), np.float32)
    if scratch:
        p0, p1, gain, th = scratch
        layer = np.zeros((h, w), np.float32)
        cv2.line(layer, p0, p1, 1.0, th, cv2.LINE_AA)
        add += gain * layer
    if dent:
        cx, cy, sg, amp, direction = dent
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        bump = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))
        add += direction * amp * (-(xx - cx) / sg ** 2) * bump * sg
    if gloss:
        cx, cy, rx, ry, gain = gloss
        layer = np.zeros((h, w), np.float32)
        cv2.ellipse(layer, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        add += gain * cv2.GaussianBlur(layer, (0, 0), max(3.0, min(rx, ry) / 3))
    img = img + add[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def roughen_edge(mask, side="top", x0=200, x1=800, amp=6.0, seed=3, corr=5.0):
    """Cut a spatially correlated rough outline (random walk, not per-column noise) into one straight edge."""
    rng = np.random.default_rng(seed)
    n = x1 - x0
    walk = cv2.GaussianBlur(rng.normal(0, 1, (1, n)).astype(np.float32), (0, 0), corr)[0]
    walk = (walk - walk.min()) / (walk.max() - walk.min() + 1e-9) * amp
    for i, d in enumerate(walk.astype(int)):
        if side == "top":
            mask[0:d, x0 + i] = 0
        elif side == "bottom":
            mask[mask.shape[0] - d:, x0 + i] = 0
        elif side == "left":
            mask[x0 + i, 0:d] = 0
    return mask
