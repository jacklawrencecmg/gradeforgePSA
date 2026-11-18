"""Evidence comparison across registered lighting variants.

Idea: albedo features (ink, print lines, stains, artwork) look the same under every light after normalization.
Relief and gloss features (scratches, dents, creases, roller marks, gloss patches) change with light angle.
So the per-pixel *difference* between lights suppresses artwork and keeps relief/gloss, and glare is excluded
by per-image validity masks instead of being mistaken for damage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .prep import clipped, flatfield, luminance, robust_sigma


@dataclass
class LightingStack:
    count: int = 1                                   # images used (primary + registered lights)
    roles: list = field(default_factory=list)
    lum: list = field(default_factory=list)          # flat-fielded luminance per image (float32)
    valid: list = field(default_factory=list)        # uint8 validity per image (card interior, not glare, registered)
    clip: list = field(default_factory=list)         # dilated clipped (glare) masks per image
    reg: list = field(default_factory=list)          # registration info per image
    dev: list = field(default_factory=list)          # signed deviation from per-pixel median of valid images
    spec: np.ndarray | None = None                   # max - min across valid images (>=2 valid), else 0
    cover: np.ndarray | None = None                  # number of valid images per pixel
    noise_sigma: float = 1.0
    diversity_ok: bool = False
    diversity_ratio: float = 0.0
    max_residual_px: float = 0.0


def build_stack(primary: np.ndarray, mask: np.ndarray, lights: list[dict], ppm: float) -> LightingStack:
    """lights: dicts from register_ecc with keys ok, aligned, valid, residual_shift_px, plus 'role'."""
    h, w = mask.shape
    stack = LightingStack()
    inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    sigma = 0.12 * w
    dil = max(3, int(round(0.7 * ppm)))
    imgs = [("primary", primary, np.full((h, w), 255, np.uint8), {"ok": True, "residual_shift_px": 0.0, "cc": 1.0})]
    for l in lights:
        if l.get("ok"):
            imgs.append((l["role"], l["aligned"], l["valid"], l))
        else:
            stack.reg.append({"role": l.get("role"), "ok": False, "reason": l.get("reason", "")})
    for role, img, valid_reg, info in imgs:
        lum = luminance(img)
        ff, _ = flatfield(lum, mask, sigma)
        clip = clipped(img, 250, dil)
        valid = (inner > 0) & (clip == 0) & (valid_reg > 0)
        stack.roles.append(role)
        stack.lum.append(ff)
        stack.clip.append(clip)
        stack.valid.append(valid.astype(np.uint8))
        stack.reg.append({"role": role, "ok": True, "cc": round(float(info.get("cc", 1.0)), 3),
                          "residual_shift_px": round(float(info.get("residual_shift_px", 0.0)), 2)})
    stack.count = len(stack.lum)
    stack.max_residual_px = max([r.get("residual_shift_px", 0.0) for r in stack.reg if r.get("ok")] or [0.0])
    stack.cover = np.sum(np.stack(stack.valid), axis=0).astype(np.uint8)
    if stack.count >= 2:
        arr = np.stack(stack.lum).astype(np.float32)
        vmask = np.stack(stack.valid) > 0
        arr_nan = np.where(vmask, arr, np.nan)
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(arr_nan, axis=0)
            mx = np.nanmax(arr_nan, axis=0)
            mn = np.nanmin(arr_nan, axis=0)
        enough = stack.cover >= 2
        stack.spec = np.where(enough, mx - mn, 0).astype(np.float32)
        for k in range(stack.count):
            d = np.where(vmask[k] & enough, arr[k] - med, 0).astype(np.float32)
            stack.dev.append(d)
        hf = stack.spec - cv2.GaussianBlur(stack.spec, (0, 0), 1.5)
        region = enough & (inner > 0)
        stack.noise_sigma = robust_sigma(hf[region]) if region.any() else 1.0
        p90 = float(np.percentile(stack.spec[region], 90)) if region.any() else 0.0
        stack.diversity_ratio = p90 / max(stack.noise_sigma, 1e-3)
        stack.diversity_ok = bool(stack.diversity_ratio >= 3.5)
    return stack
