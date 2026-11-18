"""Separating image artifacts from physical damage.

Rules use only evidence in the supplied images. With a single usable image nothing can be called physical:
the honest class is 'indeterminate_single_image'.
"""
from __future__ import annotations


def classify_candidate(per_image: list[dict], thin: bool, near_clip_present: bool,
                       reg_sensitive_fraction: float = 0.0, registration_residual_px: float = 0.0,
                       differential: bool = False) -> tuple[str, list[str]]:
    """per_image: [{role, valid, present, signed}]. `differential` marks candidates found from the lighting
    difference map (relief/gloss by construction)."""
    valid = [e for e in per_image if e["valid"]]
    present = [e for e in valid if e["present"]]
    reasons: list[str] = []
    if differential and reg_sensitive_fraction >= 0.5 and registration_residual_px >= 0.4:
        return "artifact_registration", [
            f"{reg_sensitive_fraction * 100:.0f}% of the candidate lies on strong artwork edges and the lighting photos are "
            f"only aligned to {registration_residual_px:.1f} px."]
    if len(valid) < 2:
        why = "Only one image is usable at this location"
        if len(per_image) > 1:
            why += " (the other photos are glare-covered or not registered here)"
        return "indeterminate_single_image", [why + ": cannot separate physical damage from artifact."]
    if differential:
        reasons.append(f"Appearance differs between {len(valid)} valid lighting images.")
        return "physical_probable_relief", reasons
    if len(present) == len(valid):
        vals = [abs(e["signed"]) for e in present]
        same_sign = len({e["signed"] > 0 for e in present}) == 1
        ratio = min(vals) / max(max(vals), 1e-6)
        if same_sign and ratio >= 0.6:
            return "physical_probable_albedo", [f"Same sign and similar strength (ratio {ratio:.2f}) in all {len(valid)} valid images."]
        return "physical_probable_relief", [f"Present in all {len(valid)} valid images but strength/sign changes with light (ratio {ratio:.2f})."]
    if len(present) == 1:
        if near_clip_present or not thin:
            return "artifact_glare", ["Visible in one lighting image only, next to a clipped highlight or as a broad soft patch."]
        return "indeterminate_angle_dependent", ["Visible under one light only; thin features can be raking-light scratches, so this stays unresolved."]
    return "physical_probable_relief", [f"Present in {len(present)} of {len(valid)} valid images."]
