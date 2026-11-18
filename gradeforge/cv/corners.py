"""Corner inspection: whitening, rounding, chips, fraying, missing stock, outline softness.

All measurements come from the rectified canvas and its silhouette mask. Precision is limited by the pixel
density of the source photo; findings smaller than ~2 source pixels are marked at the resolution limit.
Thresholds are GradeForge model assumptions, not PSA standards.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..findings import new_finding
from .context import Ctx
from .geom import resolvability

CORNERS = ("tl", "tr", "br", "bl")
NOMINAL_ROUND_MM = 4.0


def _local(arr, name, win, w, h):
    if name == "tl":
        return arr[0:win, 0:win]
    if name == "tr":
        return arr[0:win, w - win:w][:, ::-1]
    if name == "br":
        return arr[h - win:h, w - win:w][::-1, ::-1]
    return arr[h - win:h, 0:win][::-1, :]


def _canvas_box(name, lx, ly, lw, lh, w, h):
    if name == "tl":
        return [lx, ly, lw, lh]
    if name == "tr":
        return [w - lx - lw, ly, lw, lh]
    if name == "br":
        return [w - lx - lw, h - ly - lh, lw, lh]
    return [lx, h - ly - lh, lw, lh]


def fit_circle(pts: np.ndarray):
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    a = np.c_[x, y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(a, -(x ** 2 + y ** 2), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    return cx, cy, float(np.sqrt(max(cx * cx + cy * cy - sol[2], 1e-9)))


def _robust_circle(pts: np.ndarray):
    cx, cy, r = fit_circle(pts)
    for _ in range(2):
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
        mad = 1.4826 * np.median(np.abs(d - np.median(d))) + 0.3
        keep = np.abs(d - np.median(d)) < 2.5 * mad
        if keep.sum() < 12:
            break
        cx, cy, r = fit_circle(pts[keep])
    return cx, cy, r


def _arc_points(inside_local: np.ndarray):
    bg = (inside_local == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    if n <= 1:
        return None, 0
    ys, xs = np.nonzero(labels > 0)
    d = xs ** 2 + ys ** 2
    lab_near = labels[ys[np.argmin(d)], xs[np.argmin(d)]]
    comp = (labels == lab_near).astype(np.uint8)
    area = int(comp.sum())
    edge = cv2.dilate(comp, np.ones((3, 3), np.uint8)) & inside_local
    py, px = np.nonzero(edge)
    return (np.c_[px, py].astype(np.float32) if px.size else None), area


def _esf_width(lum_local: np.ndarray, cx, cy, r) -> float | None:
    """10-90% edge-spread width (px) across the corner arc, sampled along outward normals."""
    ts = np.arange(-8, 8.01, 0.5, dtype=np.float32)
    profs = []
    for th in np.deg2rad(np.arange(12, 79, 3)):
        nx, ny = -np.cos(th), -np.sin(th)
        px, py = cx + r * nx, cy + r * ny
        xs = (px + ts * nx).astype(np.float32)
        ys = (py + ts * ny).astype(np.float32)
        if xs.min() < 0 or ys.min() < 0 or xs.max() > lum_local.shape[1] - 1 or ys.max() > lum_local.shape[0] - 1:
            continue
        profs.append(cv2.remap(lum_local, xs[None, :], ys[None, :], cv2.INTER_LINEAR)[0])
    if len(profs) < 6:
        return None
    p = np.mean(profs, axis=0)
    inside_lvl = float(p[ts <= -4].mean())
    outside_lvl = float(p[ts >= 4].mean())
    if abs(inside_lvl - outside_lvl) < 25:
        return None
    n = (p - inside_lvl) / (outside_lvl - inside_lvl)
    n = np.maximum.accumulate(np.clip(n, 0, 1))
    if n[0] > 0.1 or n[-1] < 0.9:
        return None
    return float(np.interp(0.9, n, ts) - np.interp(0.1, n, ts))


def analyze_corners(ctx: Ctx, side: str, lights: list[np.ndarray] | None = None):
    """Returns (corner_dicts, findings). `lights`: registered lighting canvases for whitening persistence."""
    w, h, win, ppm = ctx.w, ctx.h, ctx.win, ctx.ppm
    lights = lights or []
    band = (ctx.dist_in >= 0.25 * ppm) & (ctx.dist_in <= 1.2 * ppm)
    wh_primary = ctx.whitish
    lum = ctx.lab[..., 0]
    light_wh = []
    for img in lights:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        light_wh.append((lab[..., 0] > ctx.border_l + 45.0) & (chroma < 16.0))

    raw = {}
    for name in CORNERS:
        ins = _local(ctx.inside, name, win, w, h).copy()
        pts, bg_area = _arc_points(ins)
        info = {"id": name, "bg_area_px": bg_area}
        if pts is None or len(pts) < 25 or bg_area < 20:
            info.update(measurable=False)
            raw[name] = info
            continue
        cx, cy, r = _robust_circle(pts)
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        resid = dist - r
        inward = r - dist
        notch_thr_px = max(0.15 * ppm, 2.5)
        deep = inward > notch_thr_px
        info.update(measurable=True, cx=cx, cy=cy, r_px=r, r_mm=r / ppm,
                    rms_mm=float(np.sqrt(np.mean(resid ** 2)) / ppm),
                    notch_mm=float(np.max(inward) / ppm) if len(inward) else 0.0,
                    notch_pts=int(deep.sum()), arc_pts=len(pts),
                    gap_mm=float((np.sqrt(2) - 1) * r / ppm))
        info["esf_px"] = _esf_width(_local(lum, name, win, w, h), cx, cy, r)
        raw[name] = info

    radii = {n: v["r_mm"] for n, v in raw.items() if v.get("measurable")}
    esfs = {n: v["esf_px"] for n, v in raw.items() if v.get("esf_px") is not None}

    corners, findings = [], []
    for name in CORNERS:
        v = raw[name]
        loc_band = _local(band, name, win, w, h)
        loc_wh = _local(wh_primary, name, win, w, h) & loc_band
        n_band = int(loc_band.sum())
        frac = float(loc_wh.sum() / n_band) if (n_band and not ctx.light_border) else None
        dloc = _local(ctx.dist_in, name, win, w, h)
        depth_mm = float(np.percentile(dloc[loc_wh], 90) / ppm) if (frac is not None and loc_wh.any()) else 0.0
        frag = 0
        if frac:
            nfr, _, st, _ = cv2.connectedComponentsWithStats(loc_wh.astype(np.uint8), connectivity=8)
            frag = int((st[1:, cv2.CC_STAT_AREA] >= 4).sum())
        persist = None
        if lights and frac:
            fr_l = []
            for lw_ in light_wh:
                lw_loc = _local(lw_, name, win, w, h) & loc_band
                fr_l.append(float(lw_loc.sum() / max(1, n_band)))
            persist = [round(x, 4) for x in fr_l]

        c = {"id": name, "status": "not_measurable", "whitening_fraction": frac, "whitening_depth_mm": depth_mm,
             "whitening_fragments": frag, "gap_mm": v.get("gap_mm"), "radius_mm": v.get("r_mm"),
             "outline_rms_mm": v.get("rms_mm"), "notch_depth_mm": v.get("notch_mm"),
             "esf_width_px": v.get("esf_px"), "profile_deviation_mm": None, "profile_flag": False,
             "box": [0, 0, 0, 0], "notes": []}
        cbox = _canvas_box(name, 0, 0, win, win, w, h)
        c["box"] = cbox
        if ctx.light_border:
            c["notes"].append("Border is light or unmeasured: whitening cannot be separated from the border by color.")
        level = 0

        def emit(cat, sev, measurements, summary, box, cls="geometry_measurement", width_px=2.5, reasons=None):
            res = resolvability(width_px, ctx.src_ppm, ppm)
            if res["at_resolution_limit"]:
                cls = "indeterminate_resolution"
            f = new_finding(side, cat, f"corner_{name}", box, measurements, sev, cls,
                            reasons or [], f"corner.{cat}", summary,
                            support={"images_total": 1 + len(lights)}, resolvability=res)
            findings.append(f)

        # ---- whitening ----
        if frac is not None and frac >= 0.03:
            lvl = 2 if frac >= 0.10 else 1
            level = max(level, lvl)
            if not lights:
                cls, reasons = "indeterminate_single_image", ["Only one lighting image; whitening-like pixels cannot be separated from glare or dust."]
            elif persist and all(p >= 0.5 * frac for p in persist):
                cls, reasons = "physical_probable_albedo", ["Light neutral pixels at the same place in every registered image."]
            else:
                cls, reasons = "indeterminate_angle_dependent", ["Whitening-like pixels are not present in every lighting image."]
            lw, lh = max(6, int(win * 0.55)), max(6, int(win * 0.55))
            emit("corner_whitening", "moderate" if lvl == 2 else "minor",
                 {"whitening_fraction": round(frac, 4), "depth_mm": round(depth_mm, 3), "fragments": frag,
                  "per_light_fraction": persist, "strength": frac},
                 f"{frac * 100:.1f}% of the outer {1.2:.1f} mm band is near-white (depth about {depth_mm:.2f} mm).",
                 _canvas_box(name, 0, 0, lw, lh, w, h), cls, width_px=max(2.0, depth_mm * ppm), reasons=reasons)
        # ---- geometry ----
        if v.get("measurable"):
            others = [x for n, x in radii.items() if n != name]
            med_o = float(np.median(others)) if others else v["r_mm"]
            dr = v["r_mm"] - med_o
            c["profile_deviation_mm"] = float((np.sqrt(2) - 1) * dr)
            c["radius_deviation_mm"] = float(dr)
            thr_dr = max(0.3, 4.5 / ppm)
            ideal_r = med_o * ppm
            yy, xx = np.mgrid[0:win, 0:win].astype(np.float32)
            outside_ideal = (xx + 0.5 < ideal_r) & (yy + 0.5 < ideal_r) & (np.hypot(xx + 0.5 - ideal_r, yy + 0.5 - ideal_r) > ideal_r)
            ins_loc = _local(ctx.inside, name, win, w, h)
            missing_mm2 = float(((~outside_ideal) & (ins_loc == 0)).sum() / ppm ** 2)
            c["missing_area_mm2"] = missing_mm2
            ab = max(8, int(0.8 * ppm * 2))
            if dr > thr_dr and missing_mm2 >= 0.6:
                level = max(level, 2)
                emit("corner_missing_stock", "moderate", {"radius_mm": round(v["r_mm"], 2), "median_other_radius_mm": round(med_o, 2),
                     "missing_area_mm2": round(missing_mm2, 2), "strength": missing_mm2},
                     f"Corner removes {missing_mm2:.2f} mm² more cardstock than the other corners' median profile.",
                     _canvas_box(name, 0, 0, ab, ab, w, h), width_px=dr * ppm)
            elif dr > thr_dr:
                level = max(level, 1)
                c["profile_flag"] = True
                emit("corner_rounding", "minor", {"radius_mm": round(v["r_mm"], 2), "median_other_radius_mm": round(med_o, 2),
                     "deviation_mm": round(dr, 2), "strength": dr},
                     f"Corner radius {v['r_mm']:.2f} mm is {dr:.2f} mm larger than the median of the other corners.",
                     _canvas_box(name, 0, 0, ab, ab, w, h), width_px=dr * ppm,
                     reasons=["Comparison is relative to this card's other corners; manufacturer die-cut variance is not modeled."])
            if v["notch_pts"] >= 4:
                level = max(level, 1)
                sev = "moderate" if v["notch_mm"] >= 0.4 else "minor"
                if sev == "moderate":
                    level = 2
                emit("corner_chip", sev, {"notch_depth_mm": round(v["notch_mm"], 3), "notch_arc_points": v["notch_pts"],
                     "strength": v["notch_mm"]},
                     f"Outline dips {v['notch_mm']:.2f} mm inside the fitted corner arc over {v['notch_pts']} outline points.",
                     _canvas_box(name, 0, 0, ab, ab, w, h), width_px=v["notch_mm"] * ppm)
            fray_thr = max(0.07, 1.2 / ppm)
            if v["rms_mm"] >= fray_thr and ((frac or 0) >= 0.03 or frag >= 3):
                level = max(level, 1)
                emit("corner_fray", "moderate" if (frac or 0) >= 0.10 else "minor",
                     {"outline_rms_mm": round(v["rms_mm"], 3), "whitening_fraction": None if frac is None else round(frac, 4),
                      "fragments": frag, "strength": v["rms_mm"]},
                     f"Irregular outline (RMS {v['rms_mm']:.2f} mm) with whitening fragments.",
                     _canvas_box(name, 0, 0, ab, ab, w, h), width_px=v["rms_mm"] * ppm)
        if name in esfs and len(esfs) >= 3:
            others = [x for n, x in esfs.items() if n != name]
            med_e = float(np.median(others))
            if esfs[name] > 1.5 * med_e + 1.0:
                level = max(level, 1)
                ab = max(8, int(0.8 * ppm * 2))
                emit("corner_softness", "minor", {"esf_width_px": round(esfs[name], 2), "median_other_px": round(med_e, 2),
                     "strength": esfs[name] - med_e},
                     f"Corner edge transition is {esfs[name]:.1f} px wide vs {med_e:.1f} px median on the other corners.",
                     _canvas_box(name, 0, 0, ab, ab, w, h),
                     reasons=["Could be local defocus rather than wear; compare against a macro photo."],
                     width_px=esfs[name])
        if ctx.light_border:
            c["status"] = "minor_candidate" if level else "not_measurable"
            if level == 0 and v.get("measurable"):
                c["status"] = "not_measurable"
        else:
            c["status"] = ("clean_observed", "minor_candidate", "visible_candidate")[level]
        if not v.get("measurable"):
            c["notes"].append("Corner outline could not be fitted (sharp or occluded corner, imprecise card outline, or silhouette not found).")
            if level == 0:
                c["status"] = "not_measurable"
        corners.append(c)
    excl = {c["id"]: int(max(win, 1.4 * (c["radius_mm"] or 0) * ppm)) for c in corners}
    return corners, findings, excl
