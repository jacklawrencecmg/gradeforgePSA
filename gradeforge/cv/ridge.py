"""Multi-scale line (ridge) detection with noise-normalized responses and hysteresis."""
from __future__ import annotations

import cv2
import numpy as np

from .prep import robust_sigma


def ridge_z(img: np.ndarray, region: np.ndarray, sigmas=(1.0, 1.6, 2.5)):
    """Z-scored ridge strength for bright (zb) and dark (zd) thin structures, max over scales.
    Each scale is normalized by a robust noise estimate taken from `region`, so thresholds are in noise units."""
    img = img.astype(np.float32)
    reg = region > 0
    zb = np.zeros(img.shape, np.float32)
    zd = np.zeros(img.shape, np.float32)
    wid = np.zeros(img.shape, np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(img, (0, 0), s)
        lxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        lyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        lxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        tmp = np.sqrt(((lxx - lyy) / 2.0) ** 2 + lxy ** 2)
        l1 = (lxx + lyy) / 2.0 + tmp          # larger eigenvalue
        l2 = (lxx + lyy) / 2.0 - tmp          # smaller eigenvalue
        sig_n = robust_sigma(l2[reg]) if reg.any() else 1.0
        sig_p = robust_sigma(l1[reg]) if reg.any() else 1.0
        bright = np.where(l1 > -0.5 * np.abs(l2), np.maximum(-l2, 0), 0) / sig_n
        dark = np.where(l2 < 0.5 * np.abs(l1), np.maximum(l1, 0), 0) / sig_p
        upd = bright > zb
        wid = np.where(upd, 2.35 * s, wid)
        zb = np.maximum(zb, bright)
        zd = np.maximum(zd, dark)
    zb[~reg] = 0
    zd[~reg] = 0
    return zb, zd, wid


def hysteresis_components(z: np.ndarray, lo: float, hi: float, min_area: int = 6):
    low = (z > lo).astype(np.uint8)
    n, labels = cv2.connectedComponents(low, connectivity=8)
    if n <= 1:
        return labels, []
    seeds = np.unique(labels[z > hi])
    keep = [int(i) for i in seeds if i != 0]
    return labels, keep


def line_candidates(z: np.ndarray, wid: np.ndarray, ppm: float, lo: float = 4.0, hi: float = 8.0,
                    min_len_mm: float = 0.9, min_elong: float = 4.0, max_keep: int = 60):
    """Elongated components from a z-map. Returns list of dicts: mask_box, pts (N,2), length_mm, width_px, z."""
    labels, keep = hysteresis_components(z, lo, hi)
    out = []
    if not keep:
        return out
    # per-label bounding boxes via stats on the kept subset
    keep_mask = np.isin(labels, keep).astype(np.uint8)
    n, lab2, stats, _ = cv2.connectedComponentsWithStats(keep_mask, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 6:
            continue
        ys, xs = np.nonzero(lab2[y:y + bh, x:x + bw] == i)
        pts = np.stack([xs + x, ys + y], axis=1).astype(np.float32)
        c = pts - pts.mean(axis=0)
        if len(pts) < 3:
            continue
        evals, evecs = np.linalg.eigh(np.cov(c.T))
        proj = c @ evecs[:, 1]
        length = float(proj.max() - proj.min() + 1)
        width = max(1.0, area / max(length, 1.0))
        if length / ppm < min_len_mm or length / width < min_elong:
            continue
        zz = z[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        wpx = float(np.median(wid[pts[:, 1].astype(int), pts[:, 0].astype(int)]))
        out.append({"box": [int(x), int(y), int(bw), int(bh)], "pts": pts, "length_mm": length / ppm,
                    "width_px": max(wpx, width), "z": float(zz.max()), "z_mean": float(zz.mean()), "area": int(area)})
    out.sort(key=lambda d: -d["z"])
    return out[:max_keep]
