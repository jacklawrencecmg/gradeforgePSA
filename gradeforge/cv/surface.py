"""Surface inspection.

Two evidence paths, kept separate on purpose:
  * Border ring (no artwork): marks stand out against a uniform background. Works with one photo, but a single
    photo can never separate physical damage from artifacts, so those findings stay 'indeterminate'.
  * Lighting differential (needs >= 2 registered photos): subtracting registered lighting variants cancels
    artwork and keeps whatever changes with light angle (scratches, creases, dents, gloss). This is the only
    path that can say anything about the artwork area.
Print lines, staining and foil damage inside the artwork are NOT assessed: they are indistinguishable from
artwork content in a flat photograph.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..findings import new_finding
from .artifacts import classify_candidate
from .context import Ctx
from .geom import inside_distance, resolvability
from .lighting import LightingStack
from .prep import robust_sigma
from .ridge import line_candidates, ridge_z


def build_ring(mask: np.ndarray, cen: dict, ppm: float) -> np.ndarray:
    h, w = mask.shape
    inside = (mask > 0).astype(np.uint8)
    dist_in = inside_distance(inside)
    m = int(round(0.19 * ppm))
    l, r, t, b = (int(round(cen[k])) for k in ("left_px", "right_px", "top_px", "bottom_px"))
    art = np.zeros((h, w), np.uint8)
    art[max(0, t - m):max(0, h - b + m), max(0, l - m):max(0, w - r + m)] = 1
    return ((dist_in >= 1.1 * ppm) & (art == 0)).astype(np.uint8)


def glare_report(rect: np.ndarray, mask: np.ndarray, ring: np.ndarray | None, px_per_mm: float) -> dict:
    card = mask > 0
    clipped_px = (rect.min(axis=2) >= 250) & card
    frac = float(clipped_px.sum() / max(1, card.sum()))
    overexposed = False
    if ring is not None and ring.sum() > 500:
        overexposed = bool(np.median(rect.min(axis=2)[ring > 0]) >= 248)
    regions = []
    if not overexposed:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(clipped_px.astype(np.uint8), connectivity=8)
        min_area = 12.0 * px_per_mm ** 2
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area >= min_area:
                regions.append({"box": [int(x), int(y), int(bw), int(bh)], "area_mm2": float(area / px_per_mm ** 2)})
    return {"clipped_fraction": frac, "border_overexposed": overexposed, "regions": regions}


# ----------------------------------------------------------------------------------------------
def _ring_resid(lab: np.ndarray, ring: np.ndarray, ppm: float):
    inring = ring > 0
    k = 2 * int(round(0.8 * ppm)) + 1

    def resid(ch):
        med = float(np.median(ch[inring]))
        filled = np.clip(np.where(inring, ch, med), 0, 255).astype(np.uint8)
        return ch - cv2.medianBlur(filled, k).astype(np.float32)

    rl_raw = resid(lab[..., 0])
    rc_raw = np.hypot(resid(lab[..., 1]), resid(lab[..., 2]))
    rl_raw[~inring] = 0
    rc_raw[~inring] = 0
    rl = cv2.GaussianBlur(rl_raw, (0, 0), 0.8)
    rc = cv2.GaussianBlur(rc_raw, (0, 0), 0.8)
    sl, sc = robust_sigma(rl[inring]), robust_sigma(rc[inring])
    return {"rl_raw": rl_raw, "rl": rl, "rc": rc, "thr_l": max(8.0, 6.0 * sl), "thr_c": max(6.0, 6.0 * sc),
            "sigma_l": sl, "sigma_c": sc}


def _per_image(pts_idx, stack: LightingStack, resid_list, ring_boxes_clip):
    out = []
    ys, xs = pts_idx
    for k, res in enumerate(resid_list):
        valid_frac = float(stack.valid[k][ys, xs].mean()) if len(ys) else 0.0
        vl, vc = np.abs(res["rl"][ys, xs]), res["rc"][ys, xs]
        pl, pc = float(np.percentile(vl, 90)), float(np.percentile(vc, 90))
        present = pl >= 0.8 * res["thr_l"] or pc >= 0.8 * res["thr_c"]
        signed = float(np.median(res["rl"][ys, xs]))
        if abs(signed) < 1.0:
            signed = float(np.median(vc)) * (1 if pc > pl else 0.0) or signed
        out.append({"role": stack.roles[k], "valid": valid_frac >= 0.7, "present": bool(present),
                    "signed": round(signed, 2), "p90_L": round(pl, 1), "p90_C": round(pc, 1)})
    return out


def _near_clip(stack: LightingStack, k: int, box, ppm: float) -> bool:
    x, y, bw, bh = box
    m = int(2.0 * ppm)
    h, w = stack.clip[k].shape
    return bool(stack.clip[k][max(0, y - m):min(h, y + bh + m), max(0, x - m):min(w, x + bw + m)].any())


def _make(side, ctx, stack, category, target, box, meas, sev, cls, reasons, detector, summary, per_image, width_px):
    res = resolvability(width_px, ctx.src_ppm, ctx.ppm)
    if res["at_resolution_limit"] and cls.startswith("physical"):
        cls = "indeterminate_resolution"
        reasons = reasons + ["The feature is at or below what the photo can resolve; it cannot be characterized."]
    f = new_finding(side, category, target, box, meas, sev, cls, reasons, detector, summary,
                    support={"images_total": stack.count, "images_valid": sum(1 for e in per_image if e["valid"]),
                             "images_present": sum(1 for e in per_image if e["present"]), "per_image": per_image},
                    resolvability=res)
    return f


def _side_of(cx, cy, w, h):
    return "left" if cx < w * 0.25 else "right" if cx > w * 0.75 else "top" if cy < h / 2 else "bottom"


def detect_ring(ctx: Ctx, ring: np.ndarray, stack: LightingStack, side: str, resid_all: list) -> tuple[list, list]:
    """Returns (findings, masks_of_candidates)."""
    ppm = ctx.ppm
    prim = resid_all[0]
    inring = ring > 0
    findings, masks = [], []
    zb, zd, wid = ridge_z(prim["rl_raw"], cv2.erode(ring, np.ones((3, 3), np.uint8)))
    lines = line_candidates(np.maximum(zb, zd), wid, ppm, lo=4.0, hi=8.0)
    taken = np.zeros(ring.shape, np.uint8)

    def emit(kind_cat, pts, box, meas, sev, summary, thin, width_px):
        ys, xs = pts[:, 1].astype(int), pts[:, 0].astype(int)
        per = _per_image((ys, xs), stack, resid_all, None)
        near = any(e["present"] and _near_clip(stack, k, box, ppm) for k, e in enumerate(per))
        cls, reasons = classify_candidate(per, thin, near)
        f = _make(side, ctx, stack, kind_cat, f"ring_{_side_of(box[0] + box[2] / 2, box[1] + box[3] / 2, ctx.w, ctx.h)}",
                  box, meas, sev, cls, reasons, f"surface.{kind_cat}", summary, per, width_px)
        findings.append(f)
        taken[ys, xs] = 1

    for c in lines:
        ys, xs = c["pts"][:, 1].astype(int), c["pts"][:, 0].astype(int)
        pol = "bright" if prim["rl_raw"][ys, xs].mean() > 0 else "dark"
        emit("surface_linear_mark", c["pts"], c["box"],
             {"length_mm": round(c["length_mm"], 2), "width_px": round(c["width_px"], 2), "polarity": pol,
              "z": round(c["z"], 1), "strength": c["z"]},
             "moderate" if c["length_mm"] >= 5.0 else "minor",
             f"{pol.capitalize()} line, {c['length_mm']:.1f} mm long, about {c['width_px']:.1f} px wide (border).",
             True, c["width_px"])
    # blobs: spots and stains not already explained by a line
    mask = (((np.abs(prim["rl"]) > prim["thr_l"]) | (prim["rc"] > prim["thr_c"])) & inring).astype(np.uint8)
    mask[cv2.dilate(taken, np.ones((5, 5), np.uint8)) > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(12, int(0.08 * ppm ** 2))
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
        pts = np.stack([xs + x, ys + y], axis=1).astype(np.float32)
        blobs.append((area, pts, [int(x), int(y), int(bw), int(bh)]))
    blobs.sort(key=lambda t: -t[0])
    for area, pts, box in blobs[:30]:
        area_mm2 = area / ppm ** 2
        ys, xs = pts[:, 1].astype(int), pts[:, 0].astype(int)
        dl = float(prim["rl"][ys, xs].mean())
        dc = float(prim["rc"][ys, xs].mean())
        stain = area_mm2 >= 1.0
        cat = "surface_stain" if stain else "surface_spot"
        emit(cat, pts, box, {"area_mm2": round(area_mm2, 2), "mean_dL": round(dl, 1), "mean_dChroma": round(dc, 1),
                             "strength": max(abs(dl) / prim["thr_l"], dc / prim["thr_c"])},
             "moderate" if (stain and area_mm2 >= 4.0) else "minor",
             f"{'Discoloration' if stain else 'Spot'} of {area_mm2:.1f} mm² on the border "
             f"(luminance {dl:+.0f}, chroma {dc:+.0f}).", not stain, float(np.sqrt(area)))
    return findings, [taken]


def _reg_sensitive(stack: LightingStack, ppm: float) -> np.ndarray:
    """Pixels on strong structure that is the SAME in every image (artwork edges): gradient vectors agree across
    images. Relief changes sign/strength with light, so it does not agree and is not masked."""
    gv = []
    for lum in stack.lum:
        gv.append(np.stack([cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)]))
    mags = [np.hypot(g[0], g[1]) for g in gv]
    gmin = np.minimum.reduce(mags)
    diff = np.zeros_like(gmin)
    for i in range(len(gv)):
        for j in range(i + 1, len(gv)):
            diff = np.maximum(diff, np.hypot(gv[i][0] - gv[j][0], gv[i][1] - gv[j][1]))
    ok = stack.cover >= 2
    thr = np.percentile(gmin[ok], 85) if ok.any() else 1e9
    edge = (gmin > thr) & (diff < 0.5 * gmin)
    r = int(np.ceil(stack.max_residual_px)) + 3
    return cv2.dilate(edge.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8))


def _fit_poly2(img: np.ndarray, region: np.ndarray, step: int = 8):
    h, w = img.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    sel = region[yy, xx] > 0
    xn, yn = xx[sel] / w - 0.5, yy[sel] / h - 0.5
    a = np.c_[np.ones_like(xn), xn, yn, xn * yn, xn ** 2, yn ** 2]
    v = img[yy, xx][sel]
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    for _ in range(2):
        r = v - a @ coef
        mad = robust_sigma(r)
        keep = np.abs(r) < 3 * mad
        if keep.sum() > 20:
            coef, *_ = np.linalg.lstsq(a[keep], v[keep], rcond=None)
    ys, xs = np.mgrid[0:h, 0:w]
    xn, yn = xs / w - 0.5, ys / h - 0.5
    return (coef[0] + coef[1] * xn + coef[2] * yn + coef[3] * xn * yn + coef[4] * xn ** 2 + coef[5] * yn ** 2).astype(np.float32)


def detect_differential(ctx: Ctx, stack: LightingStack, side: str, ring_taken: np.ndarray) -> list:
    if stack.count < 2 or stack.spec is None:
        return []
    ppm = ctx.ppm
    region = ((stack.cover >= 2) & (ctx.inside > 0)).astype(np.uint8)
    region = cv2.erode(region, np.ones((int(0.6 * ppm) * 2 + 1,) * 2, np.uint8))
    if region.sum() < 1000:
        return []
    findings = []
    reg_sens = _reg_sensitive(stack, ppm)
    dev0 = stack.dev[0]
    noise = max(stack.noise_sigma, 0.3)
    claimed = cv2.dilate(ring_taken, np.ones((7, 7), np.uint8))

    def per_image_dev(ys, xs):
        out = []
        for k in range(stack.count):
            v = float(np.median(stack.dev[k][ys, xs]))
            out.append({"role": stack.roles[k], "valid": float(stack.valid[k][ys, xs].mean()) >= 0.7,
                        "present": abs(v) >= 3 * noise, "signed": round(v, 2)})
        return out

    def add(cat, pts, box, meas, sev, summary, width_px, reasons_extra=None):
        ys, xs = pts[:, 1].astype(int), pts[:, 0].astype(int)
        if claimed[ys, xs].mean() > 0.5:
            return False
        sens = float(reg_sens[ys, xs].mean())
        per = per_image_dev(ys, xs)
        cls, reasons = classify_candidate(per, True, False, sens, stack.max_residual_px, differential=True)
        reasons = reasons + (reasons_extra or [])
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        tgt = "artwork"
        for name, (px, py) in {"tl": (0, 0), "tr": (ctx.w, 0), "br": (ctx.w, ctx.h), "bl": (0, ctx.h)}.items():
            if abs(cx - px) < ctx.win and abs(cy - py) < ctx.win:
                tgt, cat = f"corner_{name}", "corner_relief"
        f = _make(side, ctx, stack, cat, tgt, box, meas, sev, cls, reasons, f"surface.{cat}", summary, per, width_px)
        f["support"]["registration_residual_px"] = round(stack.max_residual_px, 2)
        findings.append(f)
        claimed[ys, xs] = 1
        return True

    # ---- relief lines (scratch / crease candidates) from the lighting-difference magnitude ----
    # Strong artwork edges are excluded: a gain difference between photos scales edge contrast and mimics relief
    # there, so relief cannot be assessed on those pixels from lighting differences alone.
    # Bloom around a glare spot also differs between photos, so a 3 mm margin around any clipped area is not judged.
    any_clip = np.maximum.reduce([c for c in stack.clip]).astype(np.uint8)
    halo = cv2.dilate(any_clip, np.ones((int(3.0 * ppm) | 1,) * 2, np.uint8))
    region = (region & (halo == 0)).astype(np.uint8)
    region_ne = (region & (reg_sens == 0)).astype(np.uint8)
    # ---- dipoles (dent / indentation candidates), checked in every image's deviation map ----
    sg = max(1.0, 0.3 * ppm)
    for kk in range(stack.count):
        ds = cv2.GaussianBlur(stack.dev[kk], (0, 0), sg)
        sd = robust_sigma(ds[region > 0])
        pos, neg = (ds > 5 * sd) & (region_ne > 0), (ds < -5 * sd) & (region_ne > 0)
        both = cv2.morphologyEx((pos | neg).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(0.6 * ppm) | 1,) * 2, np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(both, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.5 * ppm ** 2:
                continue
            sub = labels[y:y + bh, x:x + bw] == i
            pa, na = int((pos[y:y + bh, x:x + bw] & sub).sum()), int((neg[y:y + bh, x:x + bw] & sub).sum())
            if min(pa, na) < 0.25 * (pa + na):
                continue
            ys, xs = np.nonzero(sub)
            pts = np.stack([xs + x, ys + y], axis=1).astype(np.float32)
            add("surface_relief_dipole", pts, [int(x), int(y), int(bw), int(bh)],
                {"area_mm2": round(area / ppm ** 2, 2), "positive_area_px": pa, "negative_area_px": na,
                 "peak_deviation": round(float(np.abs(ds[y:y + bh, x:x + bw][sub]).max()), 1),
                 "strength": float(np.abs(ds[y:y + bh, x:x + bw][sub]).max() / max(sd, 1e-3))},
                "moderate" if area / ppm ** 2 >= 2.0 else "minor",
                f"Paired bright/dark lobes ({area / ppm ** 2:.1f} mm²) that swap with light direction (dent or indentation candidate).",
                float(np.sqrt(area)))
    zb, _, wid = ridge_z(stack.spec, region_ne)
    for c in line_candidates(zb, wid, ppm, lo=5.0, hi=12.0, min_len_mm=2.0):
        add("surface_relief_line", c["pts"], c["box"],
            {"length_mm": round(c["length_mm"], 2), "width_px": round(c["width_px"], 2), "z": round(c["z"], 1), "strength": c["z"]},
            "moderate" if c["length_mm"] >= 5.0 else "minor",
            f"Line {c['length_mm']:.1f} mm long that changes with lighting angle (scratch or crease candidate).", c["width_px"])
    # ---- gloss inconsistency ----
    gl = cv2.GaussianBlur(dev0, (0, 0), max(2.0, 2.0 * ppm))
    try:
        surf = _fit_poly2(gl, region)
    except np.linalg.LinAlgError:
        surf = np.zeros_like(gl)
    gr = gl - surf
    sgl = robust_sigma(gr[region > 0])
    near = cv2.dilate(claimed, np.ones((int(3 * ppm) | 1,) * 2, np.uint8))
    gm = ((np.abs(gr) > 6 * sgl) & (region > 0) & (near == 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gm, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 6.0 * ppm ** 2:
            continue
        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
        pts = np.stack([xs + x, ys + y], axis=1).astype(np.float32)
        if abs(float(gr[y:y + bh, x:x + bw][labels[y:y + bh, x:x + bw] == i].mean())) < 2.0:
            continue
        add("surface_gloss_variation", pts, [int(x), int(y), int(bw), int(bh)],
            {"area_mm2": round(area / ppm ** 2, 1), "mean_deviation": round(float(gr[y:y + bh, x:x + bw][labels[y:y + bh, x:x + bw] == i].mean()), 1),
             "strength": float(np.abs(gr[y:y + bh, x:x + bw]).max() / max(sgl, 1e-3))},
            "minor", f"Broad patch ({area / ppm ** 2:.0f} mm²) whose lighting response differs from the rest of the card.",
            float(np.sqrt(area)), ["Illumination falloff was removed with a quadratic fit; strong local lighting differences can still leak through."])
    # ---- periodic banding ----
    findings += _banding(ctx, stack, side, dev0, region, noise)
    return findings


def _banding(ctx, stack, side, dev0, region, noise):
    out = []
    ppm = ctx.ppm
    for axis, name in ((0, "horizontal bands"), (1, "vertical bands")):
        wts = region.sum(axis=axis).astype(np.float64)
        prof = (dev0 * region).sum(axis=axis) / np.maximum(wts, 1)
        good = wts > 0.5 * wts.max()
        if good.sum() < 200:
            continue
        idx = np.nonzero(good)[0]
        p = prof[idx[0]:idx[-1] + 1]
        win = max(5, int(5.0 * ppm)) | 1
        p = p - np.convolve(p, np.ones(win) / win, mode="same")
        p = p[win:-win] if len(p) > 3 * win else p
        n = len(p)
        if n < 200:
            continue
        spec = np.abs(np.fft.rfft(p * np.hanning(n)))
        freqs = np.fft.rfftfreq(n)             # cycles per pixel
        period_mm = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9) / ppm, np.inf)
        band = (period_mm >= 1.5) & (period_mm <= 25.0)
        if not band.any():
            continue
        k = int(np.argmax(np.where(band, spec, 0)))
        floor = float(np.median(spec[band]) + 1e-9)
        ratio = float(spec[k] / floor)
        amp = float(2 * spec[k] / (n * 0.5))
        if ratio >= 15.0 and amp >= max(0.8, 8 * noise / np.sqrt(max(region.sum(axis=axis).max(), 1))):
            w, h = ctx.w, ctx.h
            per = [{"role": r, "valid": True, "present": True, "signed": 0.0} for r in stack.roles]
            f = _make(side, ctx, stack, "surface_periodic_banding", "artwork", [0, 0, w, h],
                      {"period_mm": round(float(period_mm[k]), 2), "amplitude_L": round(amp, 2), "orientation": name,
                       "spectral_peak_ratio": round(ratio, 1), "strength": ratio},
                      "minor", "physical_probable_relief", [f"Regular {name} with {period_mm[k]:.1f} mm spacing appear in the lighting difference."],
                      "surface.periodic_banding",
                      f"Regular {name} every {period_mm[k]:.1f} mm in the lighting response (possible roller marks).", per, 4.0)
            out.append(f)
    return out


def glare_findings(ctx: Ctx, stack: LightingStack, rects: list, side: str) -> list:
    out = []
    for k, img in enumerate(rects):
        g = glare_report(img, ctx.mask, None, ctx.ppm)
        for reg in g["regions"][:6]:
            per = [{"role": r, "valid": True, "present": i == k, "signed": 0.0} for i, r in enumerate(stack.roles)]
            f = new_finding(side, "glare_region", f"glare:{stack.roles[k]}", reg["box"],
                            {"area_mm2": round(reg["area_mm2"], 1), "image": stack.roles[k], "strength": reg["area_mm2"]},
                            "info", "artifact_glare", ["Saturated highlight; the area underneath cannot be inspected in this image."],
                            "surface.glare", f"Glare/clipped region of {reg['area_mm2']:.0f} mm² in {stack.roles[k]}.",
                            support={"images_total": stack.count, "per_image": per})
            out.append(f)
    return out
