"""Plain-English result for the top of the screen.

Why this exists: the engine's lowest ceiling counts every UNCONFIRMED detector flag as if it were real damage. On real photos
that turns texture, glare and dust into 'possible damage' and drags the headline number down (a PSA 9 card showed as 8).
So the headline never presents that worst case as the finding. It shows the range, says how many flagged items are only
suspected, and tells the person exactly what to check or photograph to narrow it. Nothing here adds information the engine
does not have, and no percentages or probabilities are invented.
"""
from __future__ import annotations

from .engine import finding_state
from .shots import CORNER_NAMES, EDGE_NAMES

WORDS = {
    "corner_whitening": "possible whitening on the corner", "corner_rounding": "possible rounding of the corner",
    "corner_chip": "possible chip or notch at the corner", "corner_fray": "possible fraying at the corner",
    "corner_softness": "a corner that looks a little soft", "corner_missing_stock": "possible missing cardstock at the corner",
    "corner_relief": "a possible bend or compression at the corner",
    "edge_whitening": "possible whitening along the edge", "edge_rough_cut": "a possibly rough edge cut",
    "edge_notch": "a possible nick or chip in the edge", "edge_layer_anomaly": "a possible edge layer problem",
    "surface_linear_mark": "a possible line (print line or hairline scratch)", "surface_spot": "a possible speck or spot",
    "surface_stain": "a possible stain or discoloration", "surface_relief_line": "a possible scratch or crease line",
    "surface_relief_dipole": "a possible dent or indentation", "surface_gloss_variation": "possible uneven gloss",
    "surface_periodic_banding": "possible roller marks", "manual": "something you marked",
}
SIDE = {"front": "front", "back": "back"}


def _where(f: dict) -> str:
    t = f.get("target", "")
    side = SIDE.get(f.get("side"), "")
    if t.startswith("corner_"):
        return f"{side} {CORNER_NAMES.get(t[7:], t)} corner"
    if t.startswith("edge_"):
        return f"{side} {EDGE_NAMES.get(t[5:], t)}"
    if t.startswith("ring_"):
        return f"{side} border"
    return f"{side} artwork"


def _flagged(result: dict) -> tuple[list, list]:
    """(confirmed, suspected). Suspected = automatic flags nobody has confirmed or rejected."""
    conf, susp = [], []
    for side in ("front", "back"):
        s = result.get(side)
        if not (s and s.get("ok")):
            continue
        for f in s["findings"]:
            st = finding_state(f)
            if st == "confirmed":
                conf.append(f)
            elif st in ("candidate", "unresolved") and f.get("severity") != "info":
                susp.append(f)
    key = lambda f: (0 if f.get("severity") == "moderate" else 1, 0 if finding_state(f) == "candidate" else 1)
    return sorted(conf, key=key), sorted(susp, key=key)


def _item(f: dict) -> dict:
    return {"id": f["id"], "side": f["side"], "where": _where(f), "what": WORDS.get(f["category"], f["label"].lower()),
            "size": f["severity"], "confirmed": finding_state(f) == "confirmed",
            "how_sure": ("You confirmed this." if finding_state(f) == "confirmed" else
                         "The software thinks this is probably real but nobody has checked it." if finding_state(f) == "candidate" else
                         "The software cannot tell whether this is damage or a photo artifact.")}


def plain_summary(result: dict) -> dict:
    g = result.get("gem") or {}
    e = g.get("psa10_eligibility", {})
    ceil = g.get("grade_ceiling", {})
    conf, susp = _flagged(result)
    chars = {c["id"]: c for c in g.get("characteristics", [])}
    hi, lo = ceil.get("optimistic"), ceil.get("conservative")
    status = e.get("status")

    headline = {
        "ruled_out": "The photos show a problem that rules out a PSA 10.",
        "unlikely": "A PSA 10 looks unlikely from these photos.",
        "uncertain": "From these photos this is too close to call between a PSA 9 and a PSA 10.",
        "credible_candidate": "A PSA 10 is a credible possibility. Photos alone can never confirm one.",
        "insufficient_evidence": "There isn't enough photo evidence yet to say.",
    }.get(status, "Result unavailable.")

    # the grade range, in words
    if hi is None:
        grade_line = "No grade range yet: the photos could not be measured."
    elif conf:
        grade_line = f"With the problems you confirmed, the best this card could grade is a PSA {lo}."
    elif susp:
        grade_line = (f"No damage has been confirmed. The best this card could grade is a PSA {hi}, and a PSA {lo} only if every one of the "
                      f"{len(susp)} flagged item{'s' if len(susp) != 1 else ''} below turns out to be real damage. Most flags on a photo are false alarms, "
                      "so check them before reading anything into the lower number.")
    else:
        grade_line = f"No damage was found in these photos. The best this card could grade is a PSA {hi}."
    if hi is not None and lo is not None and hi == lo and not susp:
        grade_line = f"The best this card could grade, based on these photos, is a PSA {hi}."

    good, worry = [], []
    for cid in ("front_centering", "back_centering"):
        c = chars.get(cid)
        if c:
            (good if c["status"] == "clear" else worry).append({"area": c["label"], "text": c["observed"]})
    for g_id in ("corners", "edges"):
        grp = [c for c in g.get("characteristics", []) if c.get("group") == g_id and c.get("side")]
        clear = [c for c in grp if c["status"] == "clear"]
        if grp and len(clear) == len(grp):
            good.append({"area": g_id.capitalize(), "text": f"No problems found on any of the {len(grp)} {g_id} (front and back) at this photo quality."})
    if not conf and not susp and any(s and s.get("ok") for s in (result.get("front"), result.get("back"))):
        good.append({"area": "Flaws", "text": "The software found no flaws it could measure."})

    flags = [_item(f) for f in (conf + susp)[:8]]
    steps = []
    for p in sorted(g.get("additional_inspection", []), key=lambda x: x["priority"]):
        if p["type"] == "photo":
            steps.append({"kind": "photo", "title": p["what"], "where_to_upload": p.get("upload_to", []), "how": p["how"]})
        else:
            steps.append({"kind": "in_hand", "title": p["what"], "where_to_upload": [], "how": p["how"]})
        if len(steps) >= 6:
            break
    if susp:
        steps.insert(0, {"kind": "review", "title": f"Review the {len(susp)} flagged item{'s' if len(susp) != 1 else ''}",
                         "where_to_upload": [], "how": "Open the Defect map below. For each flag, look at the close-up image and mark it Real damage or Not damage. "
                                                       "Every item you mark updates the result immediately. This is the fastest way to a better answer."})
    ml = g.get("most_likely", {})
    return {
        "headline": headline, "detail": e.get("summary", ""), "grade_line": grade_line,
        "best_possible": hi, "if_all_flags_real": lo, "n_confirmed": len(conf), "n_suspected": len(susp),
        "most_likely_grade": ml.get("grade"), "most_likely_note": ml.get("note", ""),
        "looks_good": good, "worries": worry, "flagged": flags, "next_steps": steps[:7],
        "how_to_read": ("'Best this card could grade' is a ceiling set by what the photos show, not a prediction of the grade PSA will give. "
                        "PSA looks at the physical card, and photos can hide flaws. GradeForge has not yet been checked against real PSA results, "
                        "so it gives no percentage chances."),
    }
