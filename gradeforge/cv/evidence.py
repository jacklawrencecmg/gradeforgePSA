"""Traceable evidence: every finding gets crops of the ORIGINAL photos (native resolution), not just processed maps."""
from __future__ import annotations

import cv2
import numpy as np

from .geom import expand_box
from .prep import png, roi_from_original, sha256


def _heat(spec: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    sub = spec[y:y + h, x:x + w]
    v = np.clip(sub / max(8.0, float(np.percentile(spec[spec > 0], 99)) if (spec > 0).any() else 8.0), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def _annotate(img: np.ndarray, box, ebox) -> np.ndarray:
    out = img.copy()
    sx = img.shape[1] / max(1, ebox[2])
    sy = img.shape[0] / max(1, ebox[3])
    x0 = int((box[0] - ebox[0]) * sx); y0 = int((box[1] - ebox[1]) * sy)
    x1 = int((box[0] + box[2] - ebox[0]) * sx); y1 = int((box[1] + box[3] - ebox[1]) * sy)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), max(1, img.shape[1] // 300))
    return out


def write_evidence(findings: list[dict], side: str, rect: np.ndarray, orig: np.ndarray, M_primary: np.ndarray,
                   src_scale: float, lights: list[dict], stack, macros: list[dict], sources: list[dict]) -> dict:
    """Fills finding['evidence'] and returns {filename: png_bytes}. lights: [{role, img, to_canvas, scale}] with
    to_canvas mapping that photo's ORIGINAL pixels to primary-canvas pixels."""
    files: dict[str, bytes] = {}
    h, w = rect.shape[:2]
    for f in findings:
        box = f["box"]
        if box[2] <= 0 or box[3] <= 0:
            continue
        eb = expand_box(box, w, h)
        ev = f["evidence"]
        ev["sources"] = sources
        ev["transform_original_to_canvas"] = np.asarray(M_primary).round(6).tolist()
        ev["crop_box_canvas"] = list(eb)
        ev["files"] = {}

        def put(key, img, label):
            if img is None or img.size == 0:
                return
            data = png(img)
            name = f"{side}_{f['id']}_{key}.png"
            files[name] = data
            ev["files"][key] = {"file": name, "sha256": sha256(data), "label": label, "px": [int(img.shape[1]), int(img.shape[0])]}

        crop = rect[eb[1]:eb[1] + eb[3], eb[0]:eb[0] + eb[2]]
        put("primary_canvas", crop, "Rectified primary photo (unannotated)")
        put("primary_annotated", _annotate(crop, box, eb), "Same crop with the finding box")
        put("primary_original", roi_from_original(orig, M_primary, eb, src_scale), "Original photo pixels (native resolution)")
        for L in lights:
            put(f"light_{L['role']}", roi_from_original(L["img"], L["to_canvas"], eb, L["scale"]),
                f"Original {L['role']} photo, same region")
        if stack is not None and stack.spec is not None and f["classification"] != "artifact_glare":
            put("lighting_difference", _heat(stack.spec, eb), "Lighting-difference map (bright = changes with light)")
        for m in macros:
            if not m.get("registered"):
                continue
            fp = m["footprint"]
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            if fp[0] <= cx <= fp[2] and fp[1] <= cy <= fp[3]:
                put(f"macro_{m['index']}", roi_from_original(m["img"], m["to_canvas"], eb, m["scale"]),
                    f"Macro photo {m['index'] + 1}, same region")
        ev["has_original_crop"] = "primary_original" in ev["files"]
    return files
