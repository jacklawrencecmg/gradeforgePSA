"""Shared geometry and resolution helpers."""
from __future__ import annotations

import cv2
import numpy as np

CARD_W_MM = 63.5
CARD_H_MM = 88.9
MIN_FEATURE_SRC_PX = 2.0      # a feature narrower than this in the SOURCE photo is not resolvable


def inside_distance(inside: np.ndarray) -> np.ndarray:
    """Distance from each card pixel to the silhouette edge. The canvas border is the card edge on
    straight sides, so the mask is zero-padded first (cv2.distanceTransform ignores the array border)."""
    padded = cv2.copyMakeBorder(inside, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]


def clip_box(box, w, h):
    x, y, bw, bh = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def expand_box(box, w, h, frac=0.5, min_px=200, max_px=900):
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    nw = float(np.clip(bw * (1 + 2 * frac), min_px, max_px))
    nh = float(np.clip(bh * (1 + 2 * frac), min_px, max_px))
    return clip_box([cx - nw / 2, cy - nh / 2, nw, nh], w, h)


def resolvability(width_px_canvas: float, src_ppm: float, canvas_ppm: float) -> dict:
    """How well a feature of this width can be resolved in the SOURCE photo."""
    width_src = float(width_px_canvas) * (src_ppm / canvas_ppm)
    return {
        "source_px_per_mm": round(src_ppm, 2),
        "feature_width_source_px": round(width_src, 2),
        "feature_width_mm": round(float(width_px_canvas) / canvas_ppm, 3),
        "at_resolution_limit": bool(width_src < MIN_FEATURE_SRC_PX),
        "min_resolvable_width_mm": round(MIN_FEATURE_SRC_PX / src_ppm, 3),
    }


def box_of_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
