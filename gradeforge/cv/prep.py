"""Preprocessing: illumination normalization, clipping masks, native-resolution ROI extraction."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def luminance(rect: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rect, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


def flatfield(lum: np.ndarray, mask: np.ndarray, sigma: float):
    """Mask-aware illumination flattening. Returns (flattened luminance, smooth background).
    Large sigma keeps local features and removes lighting falloff."""
    m = (mask > 0).astype(np.float32)
    num = cv2.GaussianBlur(lum * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
    bg = num / den
    ref = float(bg[m > 0].mean()) if (m > 0).any() else 1.0
    gain = ref / np.maximum(bg, 8.0)
    return lum * gain, bg


def clipped(rect: np.ndarray, thr: int = 250, dilate_px: int = 0) -> np.ndarray:
    c = (rect.min(axis=2) >= thr).astype(np.uint8)
    if dilate_px > 0:
        c = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return c


def jpeg_blockiness(gray: np.ndarray) -> float:
    """Ratio of mean absolute difference across 8-px block boundaries to inside blocks (1.0 = no grid)."""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
    idx = np.arange(dx.size)
    on = dx[(idx % 8) == 7].mean()
    off = dx[(idx % 8) != 7].mean() + 1e-6
    return float(on / off)


def robust_sigma(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6)


def roi_from_original(img: np.ndarray, to_canvas: np.ndarray, box, scale: float, max_side: int = 1400):
    """Crop the canvas-coordinate `box` out of the ORIGINAL photo at its native pixel density.

    to_canvas: 3x3 homography mapping original-photo pixels to canvas pixels.
    scale: output pixels per canvas pixel (source resolution / canvas resolution, at least 1).
    Returns a BGR crop, or None if the box falls outside the photo."""
    x, y, bw, bh = box
    k = float(max(1.0, scale))
    ow, oh = int(np.ceil(bw * k)), int(np.ceil(bh * k))
    if max(ow, oh) > max_side:
        k *= max_side / max(ow, oh)
        ow, oh = int(np.ceil(bw * k)), int(np.ceil(bh * k))
    tr = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], np.float64)
    sc = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]], np.float64)
    t = sc @ tr @ np.asarray(to_canvas, np.float64)
    out = cv2.warpPerspective(img, t, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out


def png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("png encode failed")
    return buf.tobytes()
