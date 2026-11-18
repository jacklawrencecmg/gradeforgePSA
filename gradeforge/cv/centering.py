"""Border-based centering measurement on a rectified card.

Works for cards with a distinct, roughly uniform border. Borderless or full-art
cards are reported as not measurable instead of being guessed at.
"""
from __future__ import annotations

import cv2
import numpy as np


def _views(lab: np.ndarray, w: int, h: int) -> dict:
    y0, y1 = int(0.25 * h), int(0.75 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    return {
        "left": lab[y0:y1, :, :],
        "right": lab[y0:y1, ::-1, :],
        "top": lab[:, x0:x1, :].transpose(1, 0, 2),
        "bottom": lab[::-1, x0:x1, :].transpose(1, 0, 2),
    }


def _scan(view: np.ndarray, inset: int, band: int, thr: float, run: int, max_search: int):
    n = view.shape[0]
    ref = np.median(view[:, inset:inset + band, :], axis=1)
    dist = np.linalg.norm(view[:, :max_search, :] - ref[:, None, :], axis=2).astype(np.float32)
    dist = cv2.blur(dist, (3, 1))
    above = dist > thr
    start = inset + band
    kernel = np.ones(run, int)
    pos = np.full(n, np.nan)
    for i in range(n):
        seg = above[i, start:].astype(int)
        if seg.size < run:
            continue
        hits = np.convolve(seg, kernel, "valid") == run
        if hits.any():
            pos[i] = start + int(np.argmax(hits))
    return pos, ref


def measure_centering(rect: np.ndarray) -> dict:
    h, w = rect.shape[:2]
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
    inset = max(3, int(round(0.010 * w)))
    band = max(4, int(round(0.006 * w)))
    run = max(3, int(round(0.003 * w)))
    views = _views(lab, w, h)

    med, mad, refs = {}, {}, []
    for name, view in views.items():
        length = view.shape[1]
        pos, ref = _scan(view, inset, band, 12.0, run, int(0.30 * length))
        valid = pos[~np.isnan(pos)]
        if valid.size < 0.6 * pos.size:
            return {"measurable": False,
                    "reason": f"No clear border-to-artwork transition found on the {name} side "
                              "(borderless or full-art card, or the border and artwork are too similar in color)."}
        med[name] = float(np.median(valid))
        mad[name] = float(1.4826 * np.median(np.abs(valid - np.median(valid))))
        refs.append(ref)

    floor = inset + band + run + 1
    if min(med.values()) <= floor:
        return {"measurable": False,
                "reason": "Border is too thin to measure reliably (borderless or full-art design)."}
    worst_mad = max(mad.values())
    consistent = worst_mad <= 0.012 * w

    l, r, t, b = med["left"], med["right"], med["top"], med["bottom"]
    lr_share = max(l, r) / (l + r) * 100.0
    tb_share = max(t, b) / (t + b) * 100.0
    worst_axis, worst_pct = ("left-right", lr_share) if lr_share >= tb_share else ("top-bottom", tb_share)

    err_px = max(1.5, worst_mad)
    unc = float(max(100.0 * err_px / (l + r), 100.0 * err_px / (t + b)))
    border_lab = np.median(np.concatenate(refs, axis=0), axis=0)

    return {
        "measurable": True,
        "reliable": bool(consistent),
        "reason": "" if consistent else "Border boundary is irregular; treat these numbers with extra caution.",
        "left_px": l, "right_px": r, "top_px": t, "bottom_px": b,
        "lr_split": [round(100 * l / (l + r), 1), round(100 * r / (l + r), 1)],
        "tb_split": [round(100 * t / (t + b), 1), round(100 * b / (t + b), 1)],
        "lr_share_pct": round(lr_share, 2), "tb_share_pct": round(tb_share, 2),
        "worst_axis": worst_axis, "worst_pct": round(worst_pct, 2),
        "uncertainty_pct": round(unc, 2),
        "boundary_mad_px": round(worst_mad, 2),
        "border_lab": [float(x) for x in border_lab],
        "canvas": [int(w), int(h)],
    }
