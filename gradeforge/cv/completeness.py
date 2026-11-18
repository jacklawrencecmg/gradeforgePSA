"""Inspection completeness: which parts of the card can actually be judged from the supplied photos."""
from __future__ import annotations

import numpy as np

CORNERS = ("tl", "tr", "br", "bl")
EDGES = ("top", "right", "bottom", "left")


def _glare_frac(clip: np.ndarray, box) -> float:
    x, y, w, h = [int(v) for v in box]
    sub = clip[max(0, y):y + h, max(0, x):x + w]
    return float(sub.mean()) if sub.size else 0.0


def assess_completeness(quality: dict, ctx, stack, wear: dict, macros: list[dict], registration: list[dict]) -> dict:
    suff = quality["sufficient_for"]
    rois = []

    def add(rid, label, state, reasons, requests):
        rois.append({"id": rid, "label": label, "inspectable": state, "reasons": reasons, "requests": requests})

    primary_clip = stack.clip[0] if stack.clip else np.zeros((ctx.h, ctx.w), np.uint8)
    n_ok = sum(1 for r in registration if r.get("ok"))
    macro_ok = [m for m in macros if m.get("registered")]

    def macro_cover(box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        return any(m.get("footprint") and m["footprint"][0] <= cx <= m["footprint"][2] and m["footprint"][1] <= cy <= m["footprint"][3]
                   for m in macro_ok)

    for kind, ids, ok_key in (("corner", CORNERS, "corners"), ("edge", EDGES, "edges")):
        items = wear["corners"] if kind == "corner" else wear["edges"]
        for it in items:
            reasons, requests, state = [], [], "yes"
            if not suff[ok_key]:
                state = "no"
                reasons.append("Photo resolution is too low for this inspection.")
                requests.append("Retake with the card at least 1200 px wide.")
            elif it["status"] == "not_measurable":
                state = "no"
                reasons.append("The outline or border could not be measured here.")
                requests.append("Retake with the whole card in view and a darker, contrasting background.")
            else:
                if ctx.light_border:
                    state = "partial"
                    reasons.append("Light border: whitening cannot be separated from the border by color; outline geometry only.")
                g = _glare_frac(primary_clip, it["box"])
                if g > 0.5:
                    state = "no"
                    reasons.append(f"{g * 100:.0f}% of this region is glare-covered in the primary photo.")
                    requests.append("Retake with diffuse light so this region is not blown out.")
                elif g > 0.1 and state == "yes":
                    state = "partial"
                    reasons.append(f"{g * 100:.0f}% of this region is glare-covered in the primary photo.")
                if state == "yes" and n_ok == 0:
                    reasons.append("One photo only: marks cannot be separated from glare/dust; findings stay unconfirmed.")
                it["macro_covered"] = macro_cover(it["box"])
            add(f"{kind}_{it['id']}", f"{kind.capitalize()} {it['id'].upper() if kind == 'corner' else it['id']}", state, reasons, requests)

    # centering
    cen_ok = suff["centering"]
    add("centering", "Centering", "yes" if cen_ok else "no",
        [] if cen_ok else ["Centering could not be measured reliably."],
        [] if cen_ok else ["Retake square-on with the full card and its border in view."])
    # border ring
    r_reasons, r_req = [], []
    if not suff["ring_surface"]:
        rs = "no"
        r_reasons.append("Border is overexposed or resolution too low for surface inspection.")
        r_req.append("Retake with less exposure and at least 1200 px card width.")
    elif n_ok == 0:
        rs = "partial"
        r_reasons.append("One photo only: border marks cannot be told apart from glare, dust or print texture.")
        r_req.append("Add a second photo with the light coming from a different direction.")
    else:
        rs = "yes"
    add("border_surface", "Border surface", rs, r_reasons, r_req)
    # artwork relief
    a_reasons, a_req = [], []
    if stack.count < 2:
        a_state = "no"
        a_reasons.append("Artwork relief (scratches, dents, creases, gloss) needs at least two registered lighting photos.")
        a_req.append("Add at least one more photo of the same side with the light from a clearly different direction.")
    elif not stack.diversity_ok:
        a_state = "no"
        a_reasons.append(f"The lighting photos look too alike (difference-to-noise {stack.diversity_ratio:.1f}, needs at least 3.5), "
                         "so relief cannot be separated from artwork.")
        a_req.append("Retake one photo with the light moved to the opposite side or lowered to a raking angle.")
    else:
        valid = ((stack.cover >= 2) & (ctx.inside > 0)).sum() / max(1, (ctx.inside > 0).sum())
        if valid < 0.7:
            a_state = "partial"
            a_reasons.append(f"Only {valid * 100:.0f}% of the card is usable in two or more photos (glare or misregistration elsewhere).")
            a_req.append("Retake the glare-covered photos with diffuse light.")
        else:
            a_state = "yes"
    add("artwork_relief", "Artwork relief / gloss", a_state, a_reasons, a_req)
    add("artwork_print_and_stain", "Artwork print lines, staining, foil damage", "no",
        ["A flat photo cannot separate these from artwork content. Human review of macro photos is required."],
        ["Add macro photos of suspect areas and confirm or reject by hand."])

    states = [r["inspectable"] for r in rois if r["id"] != "artwork_print_and_stain"]
    level = "complete" if all(s == "yes" for s in states) else "insufficient" if any(s == "no" for s in states) else "partial"
    return {"rois": rois, "level": level,
            "requests": list(dict.fromkeys(q for r in rois for q in r["requests"])),
            "lighting_images_registered": n_ok, "lighting_diversity_ratio": round(float(stack.diversity_ratio), 2),
            "macros_registered": len(macro_ok),
            "note": "'complete' covers only what this tool can assess. Print lines, staining and authenticity always need a person."}
