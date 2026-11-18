"""Edge inspection: whitening / exposed cardstock, rough cuts, notches, layer anomalies.

Layer anomalies (possible peeling or delamination) are image-intensity irregularities near the edge. A flat
photograph cannot show layer separation directly, so these are always reported as unconfirmed candidates.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..findings import new_finding
from .context import Ctx
from .geom import resolvability
from .prep import robust_sigma

EDGES = ("top", "right", "bottom", "left")


def _orient(arr, name):
    if name == "top":
        return arr
    if name == "bottom":
        return arr[::-1]
    if name == "left":
        return arr.T
    return arr.T[::-1]


def _box(name, a0, a1, d, w, h):
    a0, a1 = int(a0), int(a1)
    if name == "top":
        return [a0, 0, a1 - a0, d]
    if name == "bottom":
        return [a0, h - d, a1 - a0, d]
    if name == "left":
        return [0, a0, d, a1 - a0]
    return [w - d, a0, d, a1 - a0]


def find_runs(flags: np.ndarray, min_len: int, merge_gap: int = 0):
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_len]


_ENDS = {"top": ("tl", "tr"), "right": ("tr", "br"), "bottom": ("bl", "br"), "left": ("tl", "bl")}


def analyze_edges(ctx: Ctx, side: str, lights: list[np.ndarray] | None = None, corner_excl: dict | None = None):
    """corner_excl: pixels to skip at each corner (grown when a corner is badly damaged) so corner damage is
    reported once, on the corner, rather than again as phantom edge notches."""
    w, h, win, ppm = ctx.w, ctx.h, ctx.win, ctx.ppm
    corner_excl = corner_excl or {}
    lights = lights or []
    band = (ctx.dist_in >= 0.25 * ppm) & (ctx.dist_in <= 1.0 * ppm)
    lum = ctx.lab[..., 0]
    light_wh = []
    for img in lights:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        light_wh.append((lab[..., 0] > ctx.border_l + 45.0) & (chroma < 16.0))
    d = int(round(1.0 * ppm)) + 8
    wh_all = ctx.whitish

    prof = {}
    for name in EDGES:
        ins = _orient(ctx.inside, name)
        e0, e1 = (max(win, int(corner_excl.get(c, win))) for c in _ENDS[name])
        sl = slice(e0, ins.shape[1] - e1)
        ins_o = ins[:d, sl]
        has = ins_o.any(axis=0)
        first = np.argmax(ins_o > 0, axis=0).astype(float)
        idx = np.arange(first.size)[has]
        rms = 0.0
        resid_full = np.zeros(first.size)
        if idx.size > 20:
            coef = np.polyfit(idx, first[has], 1)
            resid = first[has] - np.polyval(coef, idx)
            resid_full[has] = resid
            rms = float(np.sqrt(np.mean(resid ** 2)) / ppm)
        prof[name] = {"rms": rms, "resid": resid_full, "sl": sl}
    rms_all = {n: p["rms"] for n, p in prof.items()}

    edges, findings = [], []
    for name in EDGES:
        sl = prof[name]["sl"]
        a0 = sl.start
        band_o = _orient(band, name)[:d, sl]
        wh_o = _orient(wh_all, name)[:d, sl] & band_o
        dist_o = _orient(ctx.dist_in, name)[:d, sl]
        n = int(band_o.sum())
        frac = float(wh_o.sum() / n) if (n and not ctx.light_border) else None
        any_cols = wh_o.any(axis=0) if frac is not None else np.zeros(band_o.shape[1], bool)
        runs = find_runs(any_cols, max(2, int(0.5 * ppm)), int(0.2 * ppm))
        longest = max([(b - a) for a, b in runs] or [0]) / ppm
        longest_raw = 0
        cur = 0
        for f in any_cols:
            cur = cur + 1 if f else 0
            longest_raw = max(longest_raw, cur)
        run_mm = longest_raw / ppm
        light_fr = []
        for lw_ in light_wh:
            lo = _orient(lw_, name)[:d, sl] & band_o
            light_fr.append(float(lo.sum() / max(1, n)))

        level = 0
        e = {"id": name, "whitening_fraction": frac, "longest_whitening_run_mm": run_mm,
             "outline_irregular": False, "max_notch_mm": 0.0, "roughness_rms_mm": prof[name]["rms"],
             "layer_anomaly_windows": 0, "status": "not_measurable", "notes": [],
             "box": _box(name, a0, sl.stop, d, w, h)}

        def emit(cat, sev, meas, summary, box, cls, width_px, reasons=None):
            res = resolvability(width_px, ctx.src_ppm, ppm)
            if res["at_resolution_limit"]:
                cls = "indeterminate_resolution"
            findings.append(new_finding(side, cat, f"edge_{name}", box, meas, sev, cls, reasons or [],
                                        f"edge.{cat}", summary, support={"images_total": 1 + len(lights)},
                                        resolvability=res))

        # ---- whitening / exposed cardstock ----
        wl_cols = set()
        if frac is not None:
            for (r0, r1) in runs:
                seg = wh_o[:, r0:r1]
                seg_band = band_o[:, r0:r1]
                seg_frac = float(seg.sum() / max(1, seg_band.sum()))
                depth_mm = float(np.percentile(dist_o[:, r0:r1][seg], 90) / ppm) if seg.any() else 0.0
                length_mm = (r1 - r0) / ppm
                lvl = 2 if (length_mm >= 3.0 or seg_frac >= 0.2) else 1
                level = max(level, lvl)
                wl_cols.update(range(r0, r1))
                if not lights:
                    cls, why = "indeterminate_single_image", ["Only one lighting image; edge brightness cannot be separated from glare."]
                else:
                    fr_l = []
                    for lw_ in light_wh:
                        lo = (_orient(lw_, name)[:d, sl] & band_o)[:, r0:r1]
                        fr_l.append(float(lo.sum() / max(1, seg_band.sum())))
                    if all(x >= 0.5 * seg_frac for x in fr_l):
                        cls, why = "physical_probable_albedo", ["Near-white edge pixels present in every registered image."]
                    else:
                        cls, why = "indeterminate_angle_dependent", ["Near-white edge pixels are not present in every lighting image."]
                exposed = depth_mm >= 0.2 and length_mm >= 1.0
                emit("edge_whitening", "moderate" if lvl == 2 else "minor",
                     {"length_mm": round(length_mm, 2), "depth_mm": round(depth_mm, 3), "band_fraction": round(seg_frac, 3),
                      "exposed_cardstock_like": bool(exposed), "strength": length_mm},
                     f"{length_mm:.1f} mm run of near-white pixels, about {depth_mm:.2f} mm deep"
                     + (" (exposed-cardstock-like)." if exposed else "."),
                     _box(name, a0 + r0, a0 + r1, d, w, h), cls, max(2.0, depth_mm * ppm), why)
            if run_mm < 0.5 and frac >= 0.02:
                level = max(level, 1)
        # ---- notches / rough cut ----
        resid = prof[name]["resid"]
        flag = resid > max(0.2 * ppm, 2.5)
        notch_runs = find_runs(flag, max(2, int(0.4 * ppm)), int(0.1 * ppm))
        others = [v for k, v in rms_all.items() if k != name]
        med_o = float(np.median(others))
        rough_thr = max(0.06, 1.0 / ppm)
        rough_by_rms = prof[name]["rms"] >= rough_thr and prof[name]["rms"] >= 2.0 * med_o + 0.02
        if len(notch_runs) >= 3 or (rough_by_rms and not notch_runs):
            level = max(level, 1)
            e["outline_irregular"] = True
            e["max_notch_mm"] = float(resid.max() / ppm)
            emit("edge_rough_cut", "minor",
                 {"roughness_rms_mm": round(prof[name]["rms"], 3), "median_other_edges_mm": round(med_o, 3),
                  "notch_count": len(notch_runs), "max_notch_mm": round(float(resid.max() / ppm), 3),
                  "strength": prof[name]["rms"]},
                 f"Edge outline is irregular: {prof[name]['rms']:.2f} mm RMS vs {med_o:.2f} mm on the other edges"
                 + (f", with {len(notch_runs)} notches." if notch_runs else "."),
                 e["box"], "geometry_measurement", max(prof[name]["rms"], 0.5 / ppm) * ppm,
                 ["Relative to this card's other edges; cutter variance between sheets is not modeled."])
        else:
            for (r0, r1) in notch_runs:
                depth_mm = float(resid[r0:r1].max() / ppm)
                e["outline_irregular"] = True
                e["max_notch_mm"] = max(e["max_notch_mm"], depth_mm)
                level = max(level, 1)
                emit("edge_notch", "moderate" if depth_mm >= 0.4 else "minor",
                     {"length_mm": round((r1 - r0) / ppm, 2), "depth_mm": round(depth_mm, 3), "strength": depth_mm},
                     f"Edge outline is notched {depth_mm:.2f} mm deep over {(r1 - r0) / ppm:.1f} mm.",
                     _box(name, a0 + r0, a0 + r1, d, w, h), "geometry_measurement", depth_mm * ppm)
        # ---- layer anomaly ----
        d0, d1 = int(0.3 * ppm), int(1.2 * ppm)
        reg = _orient(lum, name)[d0:d1, sl]
        full_in = _orient(ctx.inside, name)[d0:d1, sl].all(axis=0)
        wl = max(3, int(ppm))
        nwin = reg.shape[1] // wl
        if nwin >= 6:
            pm = np.stack([reg[:, j * wl:(j + 1) * wl].mean(axis=1) for j in range(nwin)])
            ok_w = np.array([full_in[j * wl:(j + 1) * wl].all() and not any(c in wl_cols for c in range(j * wl, (j + 1) * wl, max(1, wl // 3)))
                             for j in range(nwin)])
            dv = pm - np.median(pm[ok_w], axis=0) if ok_w.sum() >= 4 else pm * 0
            score = np.abs(dv).max(axis=1)
            sig = robust_sigma(dv[ok_w].ravel()) if ok_w.sum() >= 4 else 1.0
            bad = ok_w & (score > max(12.0, 6.0 * sig))
            for (r0, r1) in find_runs(bad, 2, 1):
                level = max(level, 1)
                e["layer_anomaly_windows"] += (r1 - r0)
                emit("edge_layer_anomaly", "minor",
                     {"length_mm": round((r1 - r0) * wl / ppm, 2), "max_luminance_deviation": round(float(score[r0:r1].max()), 1),
                      "strength": float(score[r0:r1].max())},
                     f"Edge-band brightness differs from the rest of this edge over {(r1 - r0) * wl / ppm:.1f} mm.",
                     _box(name, a0 + r0 * wl, a0 + r1 * wl, d, w, h), "indeterminate_single_image" if not lights else "indeterminate_angle_dependent",
                     1.5 * wl / 4,
                     ["A flat photo cannot show layer separation; this is an intensity irregularity only. A macro photo at a raking angle is needed."])
        if ctx.light_border:
            e["notes"].append("Border is light or unmeasured: whitening cannot be separated from the border by color.")
            e["status"] = "minor_candidate" if level else "not_measurable"
        else:
            e["status"] = ("clean_observed", "minor_candidate", "visible_candidate")[level]
        if e["outline_irregular"]:
            e["notes"].append(f"Edge outline has a notch up to {e['max_notch_mm']:.2f} mm deep (possible chip or die-cut irregularity).")
        edges.append(e)
    return edges, findings
