"""Single-side inspection pipeline: bytes in -> findings, measurements, completeness, evidence out.

Stages: decode -> card detection -> rectification (+transform kept) -> centering -> multi-photo registration (ECC)
-> corner/edge/surface detectors -> artifact-vs-physical classification -> alteration indicators -> macro
registration (ORB) -> evidence crops from the original photos -> completeness.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..findings import assign_ids, new_finding
from .centering import measure_centering
from .completeness import assess_completeness
from .context import make_ctx
from .corners import analyze_corners
from .detect import CARD_W_MM, choose_canvas_width, decode_image, detect_card, rectify
from .edges import analyze_edges
from .evidence import write_evidence
from .lighting import build_stack
from .prep import sha256
from .quality import assess_quality
from .register import register_ecc, register_orb
from . import surface as surf


def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()
                if not (k in ("mask", "labels", "img", "aligned", "valid", "warp3", "to_template") and isinstance(v, np.ndarray))}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _overlays(cen, findings, glare, w, h):
    ov = []
    if cen.get("measurable"):
        l, r, t, b = cen["left_px"], cen["right_px"], cen["top_px"], cen["bottom_px"]
        ov += [{"kind": "line", "layer": "centering", "x1": l, "y1": 0, "x2": l, "y2": h, "label": "left border"},
               {"kind": "line", "layer": "centering", "x1": w - r, "y1": 0, "x2": w - r, "y2": h, "label": "right border"},
               {"kind": "line", "layer": "centering", "x1": 0, "y1": t, "x2": w, "y2": t, "label": "top border"},
               {"kind": "line", "layer": "centering", "x1": 0, "y1": h - b, "x2": w, "y2": h - b, "label": "bottom border"}]
    return ov


def _alteration_indicators(det, side, w, h):
    out = []
    ratio = det.card_height_px / max(det.card_width_px, 1e-6)
    expected = 88.9 / 63.5
    dev = (ratio - expected) / expected
    q = det.quad
    top = np.linalg.norm(q[1] - q[0]); bot = np.linalg.norm(q[2] - q[3])
    left = np.linalg.norm(q[3] - q[0]); right = np.linalg.norm(q[2] - q[1])
    skew = max(abs(top - bot) / max(top, bot), abs(left - right) / max(left, right))
    if abs(dev) > 0.025 and skew < 0.04 and not det.warnings:
        out.append(new_finding(
            side, "alteration_indicator", "card_outline", [0, 0, w, h],
            {"aspect_ratio": round(ratio, 4), "expected_ratio": round(expected, 4), "deviation_pct": round(dev * 100, 2),
             "strength": abs(dev) * 100},
            "info", "geometry_measurement",
            ["Measured from one photo; a tilted camera or a non-standard card format gives the same signal."],
            "alteration.aspect_ratio",
            f"Card outline aspect ratio is {abs(dev) * 100:.1f}% {'taller' if dev > 0 else 'wider'} than 88.9 x 63.5 mm. "
            "Possible trimming, or a different card format. Not evidence of alteration by itself."))
    return out


def analyze_side(data: bytes, side: str, lighting: list[bytes] | None = None, macros: list | None = None):
    """Returns (result_dict, artifacts). artifacts = {rect, evidence{name:bytes}, originals{role:bytes}}."""
    img = decode_image(data)
    base = {"side": side, "ok": False}
    if img is None:
        base["reason"] = "The file could not be decoded as an image."
        return base, None
    det = detect_card(img)
    if not det.ok:
        base["reason"] = det.reason
        return base, None

    cw = choose_canvas_width(det)
    rect, mask, M = rectify(img, det, cw, with_transform=True)
    h, w = rect.shape[:2]
    ppm = cw / CARD_W_MM
    src_ppm = det.card_width_px / CARD_W_MM
    src_scale = max(1.0, det.card_width_px / cw)

    cen = measure_centering(rect)
    ring = surf.build_ring(mask, cen, ppm) if cen.get("measurable") else None
    glare = surf.glare_report(rect, mask, ring, ppm)
    quality = assess_quality(det, rect, mask, cen, glare)

    # ---- additional lighting photos: rectify at the same canvas width, then register to the primary ----
    reg_info, light_dicts, evidence_lights = [], [], []
    originals = {"primary": data}
    sources = [{"role": "primary", "sha256": sha256(data)}]
    for i, lb in enumerate(lighting or []):
        role = f"light{i + 1}"
        limg = decode_image(lb)
        ldet = detect_card(limg) if limg is not None else None
        if ldet is None or not ldet.ok:
            reg_info.append({"role": role, "ok": False, "reason": "Card not found in this photo.", "sha256": sha256(lb)})
            continue
        lrect, _, Ml = rectify(limg, ldet, cw, with_transform=True)
        res = register_ecc(rect, lrect, mask)
        reg_info.append({"role": role, "ok": res["ok"], "reason": res.get("reason", ""), "cc": round(res.get("cc", 0.0), 3),
                         "residual_shift_px": round(res.get("residual_shift_px", 0.0), 2), "sha256": sha256(lb)})
        originals[role] = lb
        sources.append({"role": role, "sha256": sha256(lb)})
        if res["ok"]:
            res["role"] = role
            light_dicts.append(res)
            evidence_lights.append({"role": role, "img": limg, "to_canvas": res["to_template"] @ Ml,
                                    "scale": max(1.0, ldet.card_width_px / cw)})
    aligned = [d["aligned"] for d in light_dicts]

    cen_ok = bool(cen.get("measurable"))
    ctx = make_ctx(rect, mask, cen["border_lab"] if cen_ok else [128.0, 128.0, 128.0], ppm, src_ppm, cen_ok)

    corners, cfind, excl = analyze_corners(ctx, side, aligned)
    edges, efind = analyze_edges(ctx, side, aligned, excl)
    if not cen_ok:
        for c in corners:
            c["status"] = "not_measurable"
            c["notes"] = ["Border could not be measured, so whitening cannot be judged."]
        for e in edges:
            e["status"] = "not_measurable"
            e["notes"] = ["Border could not be measured, so whitening cannot be judged."]
    wear = {"corners": corners, "edges": edges, "border_is_light": bool(ctx.light_border)}

    stack = build_stack(rect, mask, light_dicts, ppm)
    sfind = []
    ring_assessed = False
    if ring is not None and quality["sufficient_for"]["ring_surface"]:
        ring_assessed = True
        imgs = [rect] + aligned
        resid = [surf._ring_resid(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), ring, ppm) for im in imgs]
        rf, taken = surf.detect_ring(ctx, ring, stack, side, resid)
        sfind += rf
    else:
        taken = [np.zeros((h, w), np.uint8)]
    relief_assessed = bool(stack.count >= 2 and stack.diversity_ok and quality["sufficient_for"]["corners"])
    if relief_assessed:
        sfind += surf.detect_differential(ctx, stack, side, taken[0])
    gfind = surf.glare_findings(ctx, stack, [rect] + aligned, side)
    afind = _alteration_indicators(det, side, w, h)

    findings = cfind + efind + sfind + gfind + afind
    assign_ids(findings, side)

    # ---- macro photos: registered by ORB so findings can show a closer look ----
    macro_out = []
    for i, item in enumerate(macros or []):
        slot = None
        if isinstance(item, tuple):
            name, mb, *rest = item
            slot = rest[0] if rest else None
        else:
            name, mb = f"macro{i + 1}", item
        m = {"index": i, "name": name, "slot": slot, "sha256": sha256(mb), "registered": False}
        mimg = decode_image(mb)
        if mimg is None:
            m["reason"] = "Could not decode."
        else:
            r = register_orb(rect, mimg)
            m["inliers"] = r.get("inliers", 0)
            if r.get("ok"):
                H = r["to_template"]
                mh, mw = mimg.shape[:2]
                pts = cv2.perspectiveTransform(np.float32([[[0, 0], [mw, 0], [mw, mh], [0, mh]]]), H.astype(np.float32))[0]
                x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
                scale = 1.0 / float(np.sqrt(abs(np.linalg.det(H[:2, :2])) + 1e-12))
                if (x1 - x0) * (y1 - y0) <= 1.5 * w * h and scale >= 1.0:
                    m.update(registered=True, footprint=[float(x0), float(y0), float(x1), float(y1)],
                             to_canvas=H, scale=scale, img=mimg, macro_px_per_mm=round(scale * ppm, 1))
                else:
                    m["reason"] = "Registration was geometrically implausible; not used."
            else:
                m["reason"] = r.get("reason", "")
        macro_out.append(m)
        originals[f"macro{i + 1}"] = mb
        sources.append({"role": f"macro{i + 1}", "sha256": m["sha256"]})

    evidence_files = write_evidence(findings, side, rect, img, M, src_scale, evidence_lights, stack,
                                    [m for m in macro_out if m.get("registered")], sources)

    comp = assess_completeness(quality, ctx, stack, wear, macro_out, reg_info)
    res_floor = 2.0 / src_ppm
    surface = {
        "ring": {"assessed": ring_assessed, "reason": "" if ring_assessed else "Border could not be measured or the photo is too small/overexposed."},
        "glare": glare, "lighting_images_used": len(light_dicts),
        "artwork_relief_assessed": relief_assessed,
        "artwork_print_and_stain_assessed": False,
        "artwork_note": "Artwork relief is judged only from differences between registered lighting photos. Print lines, "
                        "staining and foil damage inside the artwork cannot be separated from artwork content in photos.",
    }
    result = {
        "side": side, "ok": True,
        "detection": {"card_width_px": round(det.card_width_px), "canvas": [w, h], "px_per_mm": round(ppm, 2),
                      "source_px_per_mm": round(src_ppm, 2), "source_scale": round(src_scale, 4),
                      "transform_original_to_canvas": np.asarray(M).round(8).tolist(), "rectangularity": round(det.rectangularity, 3),
                      "warnings": det.warnings, "source_sha256": sources[0]["sha256"]},
        "resolution": {"min_resolvable_feature_mm": round(res_floor, 3),
                       "note": f"Features narrower than about {res_floor:.2f} mm are below what this photo can resolve; "
                               "none are reported as physical damage."},
        "quality": quality, "centering": cen, "wear": wear, "surface": surface,
        "registration": {"lighting": reg_info, "lighting_stack": {
            "images": stack.count, "diversity_ratio": round(float(stack.diversity_ratio), 2), "diversity_ok": stack.diversity_ok,
            "noise_sigma": round(float(stack.noise_sigma), 3), "max_residual_px": round(float(stack.max_residual_px), 2)},
            "macros": macro_out},
        "completeness": comp, "findings": findings,
        "overlays": _overlays(cen, findings, glare, w, h),
        "detector_note": "All detectors are hand-written classical image-processing rules (OpenCV/numpy). No trained "
                         "or pretrained defect model is used, and thresholds are uncalibrated until validated on labeled outcomes.",
    }
    return to_py(result), {"rect": rect, "evidence": evidence_files, "originals": originals}
