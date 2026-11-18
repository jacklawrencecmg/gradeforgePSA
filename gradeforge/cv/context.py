"""Per-side analysis context shared by detectors."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geom import inside_distance

LIGHT_BORDER_L = 190.0     # OpenCV 8-bit Lab L; whitening cannot be separated from a border this light


@dataclass
class Ctx:
    rect: np.ndarray
    mask: np.ndarray
    inside: np.ndarray
    dist_in: np.ndarray
    lab: np.ndarray
    ppm: float            # canvas pixels per mm
    src_ppm: float        # source-photo pixels per mm (never more than the real resolution)
    border_l: float
    light_border: bool
    border_measured: bool
    w: int
    h: int
    win: int              # corner window size in px

    @property
    def whitish(self) -> np.ndarray:
        chroma = np.hypot(self.lab[..., 1] - 128.0, self.lab[..., 2] - 128.0)
        return (self.lab[..., 0] > self.border_l + 45.0) & (chroma < 16.0)


def make_ctx(rect, mask, border_lab, ppm: float, src_ppm: float, border_measured: bool) -> Ctx:
    h, w = mask.shape
    inside = (mask > 0).astype(np.uint8)
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_l = float(border_lab[0]) if border_measured else 128.0
    return Ctx(rect=rect, mask=mask, inside=inside, dist_in=inside_distance(inside), lab=lab, ppm=ppm,
               src_ppm=min(src_ppm, ppm * 1.0001) if src_ppm else ppm, border_l=border_l,
               light_border=(not border_measured) or border_l >= LIGHT_BORDER_L,
               border_measured=border_measured, w=w, h=h, win=int(round(0.09 * w)))
