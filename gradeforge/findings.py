"""Finding schema: every detector output is a traceable, reviewable record.

A finding is a *candidate* until a person confirms it. Automatic classification (physical vs artifact) is
recorded separately from review state so neither can masquerade as the other.
"""
from __future__ import annotations

DETECTOR_VERSION = "0.2"

CATEGORIES = {
    # corners
    "corner_whitening": "Corner whitening", "corner_rounding": "Corner rounding",
    "corner_chip": "Corner chip / notch", "corner_fray": "Corner fraying",
    "corner_softness": "Soft corner outline", "corner_missing_stock": "Missing cardstock at corner",
    "corner_relief": "Corner relief anomaly (possible bend/compression)",
    # edges
    "edge_whitening": "Edge whitening / exposed cardstock", "edge_rough_cut": "Rough edge cut",
    "edge_notch": "Edge notch / chip", "edge_layer_anomaly": "Edge layer anomaly (possible peeling/delamination)",
    # surface
    "surface_linear_mark": "Linear mark (print line / hairline scratch candidate)",
    "surface_spot": "Spot / speck", "surface_stain": "Stain / discoloration",
    "surface_relief_line": "Relief line (scratch / crease candidate, light-dependent)",
    "surface_relief_dipole": "Relief lobe pair (dent / indentation / crease candidate)",
    "surface_gloss_variation": "Gloss inconsistency", "surface_periodic_banding": "Periodic banding (possible roller marks)",
    "glare_region": "Glare / clipped region", "alteration_indicator": "Possible alteration indicator",
    "manual": "Manual observation",
}

# Automatic classification classes. Only the physical_probable_* classes may influence ceilings before review.
CLASSES = {
    "physical_probable_albedo": "Present with the same appearance in every valid lighting image (ink, print, stain-like)",
    "physical_probable_relief": "Appearance changes with light angle in valid images (scratch, dent, crease, gloss-like)",
    "indeterminate_single_image": "Only one usable image: cannot separate physical damage from artifact",
    "indeterminate_angle_dependent": "Visible under one light only and not glare-like: possible raking-light relief; needs a macro photo",
    "indeterminate_resolution": "At or below what the photo can resolve: cannot be characterized",
    "artifact_glare": "Glare or clipped highlight",
    "artifact_registration": "Coincides with strong artwork edges where photo alignment is imperfect",
    "artifact_noise_jpeg": "Consistent with sensor noise or JPEG compression structure",
    "geometry_measurement": "Outline measurement from one image; physical cause not separable from die-cut variance",
    "manual_observation": "Entered by a person with attached evidence",
}

PHYSICAL = {"physical_probable_albedo", "physical_probable_relief"}


def provenance(detector_id: str) -> dict:
    return {"id": detector_id, "version": DETECTOR_VERSION, "type": "classical_cv", "trained": False,
            "validated": False,
            "note": "Hand-written image-processing rules. Not a trained model; thresholds are uncalibrated "
                    "until validated against labeled outcomes."}


def new_finding(side: str, category: str, target: str, box, measurements: dict, severity: str,
                classification: str, reasons: list, detector_id: str, summary: str,
                support: dict | None = None, resolvability: dict | None = None,
                canvas_box_extra: dict | None = None) -> dict:
    assert category in CATEGORIES, category
    assert classification in CLASSES, classification
    return {
        "id": None, "side": side, "category": category, "label": CATEGORIES[category], "target": target,
        "box": [int(round(v)) for v in box], "measurements": measurements, "severity": severity,
        "classification": classification, "classification_text": CLASSES[classification],
        "classification_reasons": reasons, "summary": summary,
        "support": support or {}, "resolvability": resolvability or {},
        "detector": provenance(detector_id), "evidence": {"files": {}, "sources": [], "transform": None},
        "status": "candidate", "review": None,
    }


_SEV = {"moderate": 0, "minor": 1, "info": 2}
_CLS = {"physical_probable_relief": 0, "physical_probable_albedo": 0, "geometry_measurement": 1,
        "indeterminate_angle_dependent": 2, "indeterminate_single_image": 2, "indeterminate_resolution": 3,
        "artifact_registration": 4, "artifact_noise_jpeg": 4, "artifact_glare": 4, "manual_observation": 0}


def priority(f: dict):
    return (_SEV.get(f["severity"], 3), _CLS.get(f["classification"], 5), -(f["measurements"].get("strength") or 0))


def assign_ids(findings: list[dict], side: str) -> list[dict]:
    findings.sort(key=priority)
    for i, f in enumerate(findings, 1):
        f["id"] = f"{side[0]}{i:03d}"
    return findings
