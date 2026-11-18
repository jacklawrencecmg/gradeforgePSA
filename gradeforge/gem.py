"""Gem Mint analysis: is this card a credible PSA 10 candidate, or more likely a PSA 9 (or lower)?

Three kinds of statement are kept strictly apart, and every output item is tagged with one of them:

  official   PSA's own published wording (standards_db official_items, read from psacard.com/gradingstandards).
  secondary  Numbers/wording reported by third parties that quote PSA; not verified against PSA.
  model      GradeForge assumptions in this file (thresholds, caps, visibility heuristic). UNCALIBRATED until
             validated on real PSA outcomes (see gem_model.py).

Design rules:
  * DOMINANCE. The grade ceiling is the MINIMUM over all characteristics. There is no score that centering can
    raise, so excellent centering can never offset a defect. `dominance.offsets_applied` is always False.
  * NO FABRICATED NUMBERS. This module emits no probability. A probability appears only when a trained model has
    passed the validation criteria in gem_model.py, and then only with its measured calibration.
  * ABSENCE OF EVIDENCE IS NOT EVIDENCE OF ABSENCE. A characteristic is "clear" only where the photos had the
    coverage to see a flaw; otherwise it is "limited"/"not_assessable" and a specific extra check is requested.
  * Some PSA criteria cannot be judged from photos at all (print lines on chrome/refractor, staining inside the
    artwork, full original gloss, eye appeal, authenticity). They are listed as `human_required` and never cleared.
"""
from __future__ import annotations

from . import shots
from .engine import FINDING_CAPS, _centering_ceilings, finding_state
from .standards import Standards, load_standards

GEM_VERSION = "gem-rules-1.0 (uncalibrated)"

FINISHES = ("standard", "chrome", "refractor", "foil_holo", "matte", "unknown")
CHROMELIKE = ("chrome", "refractor", "foil_holo")
CORNERS = ("tl", "tr", "br", "bl")
EDGES = ("top", "right", "bottom", "left")
# A flaw is treated as visible without magnification above these sizes (model heuristic, uncalibrated).
VISIBLE_LENGTH_MM = 3.0
VISIBLE_AREA_MM2 = 2.0

ELIGIBILITY_TEXT = {
    "ruled_out": "The photos show evidence that a PSA 10 is not supported (see reasons).",
    "unlikely": "Visible evidence points away from a PSA 10; a PSA 9 or lower is the more likely outcome.",
    "uncertain": "No observed blocker, but open questions remain that decide between a PSA 9 and a PSA 10.",
    "credible_candidate": "Everything the photos can check is clean at adequate coverage. This supports considering the card "
                          "for a Gem Mint grade; it is not a prediction that PSA will grade it a 10.",
    "insufficient_evidence": "The photos do not allow a Gem Mint assessment.",
}
IMPACT_RANK = {"none": 0, "open": 1, "possible": 2, "likely_block": 3, "block": 4}

# ---------------------------------------------------------------------------------------------------------
# Characteristic bookkeeping
# ---------------------------------------------------------------------------------------------------------


def _visibility(f: dict) -> str:
    """'visible' (likely seen without magnification) or 'magnification' (needs a loupe). Model heuristic."""
    m = f.get("measurements", {})
    length = m.get("length_mm") or 0
    area = m.get("area_mm2") or m.get("missing_area_mm2") or 0
    if f.get("severity") == "moderate" or length >= VISIBLE_LENGTH_MM or area >= VISIBLE_AREA_MM2:
        return "visible"
    return "magnification"


def _char_for(f: dict) -> str | None:
    """Which characteristic a finding belongs to (None = not a defect claim, e.g. glare)."""
    cat, tgt = f["category"], f.get("target", "")
    if cat == "glare_region":
        return None
    if cat.startswith("corner_"):
        return tgt                                   # corner_tl ...
    if cat.startswith("edge_"):
        return tgt                                   # edge_top ...
    if cat in ("surface_linear_mark", "surface_relief_line"):
        return "linear_marks"
    if cat == "surface_relief_dipole":
        return "dents"
    if cat in ("surface_gloss_variation", "surface_periodic_banding"):
        return "finish"
    if cat in ("surface_stain", "surface_spot"):
        return "stains"
    if cat == "alteration_indicator":
        return "manufacturing"
    if cat == "manual":
        if tgt.startswith(("corner_", "edge_")):
            return tgt
        return "surface_other"
    return "surface_other"


def _roi_state(comp: dict, ids: tuple) -> str:
    states = {r["id"]: r["inspectable"] for r in comp.get("rois", [])}
    vals = [states.get(i, "no") for i in ids]
    return "no" if "no" in vals else "partial" if "partial" in vals else "yes"


def _finding_status(fs: list) -> tuple[str, str, tuple, list, list]:
    """-> (status, impact, (cons_cap, opt_cap), finding_ids, notes) from this characteristic's findings."""
    status, impact, cap, ids, notes = "clear", "none", (10, 10), [], []

    def bump(st, imp):
        nonlocal status, impact
        if IMPACT_RANK[imp] > IMPACT_RANK[impact]:
            status, impact = st, imp

    for f in fs:
        st = finding_state(f)
        if st is None or f.get("severity") == "info":
            continue
        c = FINDING_CAPS[st][f["severity"]]
        vis = _visibility(f)
        if c != (10, 10):
            cap = (min(cap[0], c[0]), min(cap[1], c[1]))
            ids.append(f["id"])
        if st == "confirmed":
            if f["severity"] == "moderate" or vis == "visible":
                bump("severe" if f["severity"] == "moderate" else "blocks_10", "block")
                notes.append(f"{f['id']}: confirmed and likely visible without magnification.")
            else:
                bump("blocks_10", "likely_block")
                notes.append(f"{f['id']}: confirmed but small (magnification-only); a PSA 10 would depend on the grader overlooking it.")
        elif st == "candidate":
            if f["severity"] == "moderate":
                bump("blocks_10", "likely_block")
                notes.append(f"{f['id']}: unconfirmed moderate candidate; review its evidence crops.")
            else:
                bump("minor_concern", "possible")
                notes.append(f"{f['id']}: unconfirmed minor candidate ({vis}); review its evidence crops.")
        elif st == "unresolved":
            bump("unresolved", "open")
            notes.append(f"{f['id']}: could not be separated from an image artifact ({f['classification'].replace('_', ' ')}).")
    return status, impact, cap, ids, notes


def _zone(pct: float, unc: float, nominal: float, outer: float | None):
    lo, hi = pct - unc, pct + unc
    if hi <= nominal:
        return "inside_nominal", lo, hi
    if outer is None:
        return ("straddles_nominal" if lo <= nominal else "beyond_nominal"), lo, hi
    if lo > outer:
        return "beyond_outer", lo, hi
    if lo > nominal:
        return ("discretion_band" if hi <= outer else "straddles_outer"), lo, hi
    return "straddles_nominal", lo, hi


# ---------------------------------------------------------------------------------------------------------
# Centering (front 55/45 vs 50/50, and reverse)
# ---------------------------------------------------------------------------------------------------------

def _centering_char(side_name: str, s: dict | None, std: Standards) -> dict:
    cid = f"{side_name}_centering"
    front = side_name == "front"
    nominal = float(std.front_limit(10) if front else std.back_limit(10))
    outer = float(std.front_discretion(10)) if front else None
    basis = "official"
    label = "Front centering" if front else "Reverse centering"
    base = {"id": cid, "label": label, "group": "centering", "side": side_name, "basis": basis,
            "human_required": False, "findings": [], "grade_cap": (10, 10)}
    if not (s and s.get("ok")):
        return {**base, "status": "not_assessable", "impact": "open", "observed": f"{side_name} photo missing or unreadable.",
                "notes": [], "detail": None}
    c = s["centering"]
    if not c.get("measurable") or s["quality"]["sufficient_for"].get("centering") is False:
        return {**base, "status": "not_assessable", "impact": "open",
                "observed": c.get("reason") or "Centering could not be measured reliably.", "notes": [], "detail": None}
    pct, unc = float(c["worst_pct"]), float(c["uncertainty_pct"])
    zone, lo, hi = _zone(pct, unc, nominal, outer)
    detail = {"worst_pct": pct, "uncertainty_pct": unc, "interval": [round(lo, 2), round(hi, 2)], "zone": zone,
              "nominal_limit_pct": nominal, "outer_limit_pct": outer, "worst_axis": c["worst_axis"],
              "lr_split": c["lr_split"], "tb_split": c["tb_split"],
              "margin_to_nominal_points": round(nominal - pct, 2),
              "resolves_50_vs_55": bool(2 * unc < 5.0)}
    notes = []
    if not detail["resolves_50_vs_55"]:
        notes.append(f"Measurement uncertainty ±{unc:.1f} points is too large to separate 50/50 from 55/45.")
    if front:
        if zone == "inside_nominal":
            status, impact = "clear", "none"
            obs = f"{pct:.1f}/{100 - pct:.1f} on {c['worst_axis']} (±{unc:.1f}); inside the 55/45 limit even at the pessimistic end."
        elif zone == "straddles_nominal":
            status, impact = "minor_concern", "possible"
            obs = f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}) straddles the nominal 55/45 limit: the photo cannot say which side it falls on."
        elif zone == "discretion_band":
            status, impact = "minor_concern", "possible"
            obs = (f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}) is above 55/45 but within PSA's stated 'approximately 55/45 to 60/40' range. "
                   "A PSA 10 here depends on grader judgment and eye appeal.")
            notes.append("Official wording: the front tolerance 'not to exceed approximately 55/45 to 60/40'. Where inside that range a card "
                         "lands is a grader judgment GradeForge cannot predict.")
        elif zone == "straddles_outer":
            status, impact = "blocks_10", "likely_block"
            obs = f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}) is at the outer end of PSA's stated range (60/40) or beyond it."
        else:
            status, impact = "blocks_10", "block"
            obs = f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}) is worse than 60/40 even at the optimistic end: outside PSA's stated PSA 10 range."
    else:
        if zone == "inside_nominal":
            status, impact = "clear", "none"
            obs = f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}); inside the 75/25 reverse limit even at the pessimistic end."
        elif zone == "straddles_nominal":
            status, impact = "minor_concern", "possible"
            obs = f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}) straddles the 75/25 reverse limit."
        else:
            status, impact = "blocks_10", "block"
            obs = f"{pct:.1f}/{100 - pct:.1f} (±{unc:.1f}) is worse than 75/25 even at the optimistic end."
    return {**base, "status": status, "impact": impact, "observed": obs, "notes": notes, "detail": detail}


# ---------------------------------------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------------------------------------

_CHAR_META = {
    "linear_marks": ("Hairline scratches / linear marks", "surface", ("border_surface", "artwork_relief"), False,
                     "PSA's PSA 10 wording requires a virtually perfect card; a scratch and a factory print line are both surface flaws."),
    "dents": ("Dents / indentations", "surface", ("artwork_relief",), False, "Surface relief seen only through lighting differences."),
    "finish": ("Surface finish / gloss irregularities", "surface", ("artwork_relief",), True,
               "PSA 10 requires full original gloss (official). Gloss needs raking-light judgment by a person."),
    "stains": ("Staining / spots", "surface", ("border_surface",), True,
               "PSA 10 must be free of staining of any kind (official). Photos can only check borders automatically."),
    "manufacturing": ("Manufacturing / print defects", "print", (), True,
                      "Ink dots, registration errors and roller marks cannot be separated from artwork content automatically."),
    "surface_other": ("Other surface observations", "surface", (), False, "Reviewer-entered observations."),
}


def analyze_gem(result: dict, std: Standards | None = None, traits: dict | None = None,
                model: dict | None = None, reliability: dict | None = None) -> dict:
    """result: pipeline output with 'front'/'back' side dicts (and optionally 'card_traits')."""
    std = std or load_standards(result.get("standards_version"))
    traits = {**(result.get("card_traits") or {}), **(traits or {})}
    finish = traits.get("finish") if traits.get("finish") in FINISHES else "unknown"
    chrome = finish in CHROMELIKE
    sides = {"front": result.get("front"), "back": result.get("back")}
    chars: list[dict] = []

    # --- centering ---
    for name in ("front", "back"):
        chars.append(_centering_char(name, sides[name], std))

    # --- per-side physical characteristics ---
    for name, s in sides.items():
        ok = bool(s and s.get("ok"))
        findings = (s or {}).get("findings", []) if ok else []
        comp = (s or {}).get("completeness", {}) if ok else {}
        by_char: dict[str, list] = {}
        for f in findings:
            if f.get("status") == "rejected":
                continue
            k = _char_for(f)
            if k:
                by_char.setdefault(k, []).append(f)

        def build(cid, label, group, roi_ids, human, basis_text, key, side_ok=ok):
            fs = by_char.get(key, [])
            st, imp, cap, ids, notes = _finding_status(fs)
            roi = _roi_state(comp, roi_ids) if (side_ok and roi_ids) else ("no" if not side_ok else "yes")
            if not side_ok:
                st, imp, notes = "not_assessable", "open", [f"{name} photo missing or unreadable."]
            elif imp == "none":
                if roi == "no":
                    st, imp = "not_assessable", "open"
                    notes = _roi_reasons(comp, roi_ids)
                elif roi == "partial":
                    st, imp = "limited", "open"
                    notes = _roi_reasons(comp, roi_ids)
                elif human:
                    st, imp = "limited", "open"
                    notes = [basis_text]
            elif imp in ("open",) and roi == "no":
                pass
            if ids:
                observed = f"{len(ids)} finding(s): " + ", ".join(ids)
            else:
                observed = {"clear": "No finding at adequate coverage.", "limited": "No finding, but coverage is incomplete.",
                            "not_assessable": "Not assessable from the supplied photos."}.get(st, "See notes.")
            states = {f["id"]: finding_state(f) for f in fs if f["id"] in ids}
            chars.append({"id": cid, "label": label, "group": group, "side": name,
                          "basis": {"corners": "official", "edges": "secondary"}.get(group, "model"),
                          "finding_states": states,
                          "human_required": human, "status": st, "impact": imp, "grade_cap": cap, "findings": ids,
                          "observed": observed, "notes": notes, "detail": None, "roi_state": roi})

        for c in CORNERS:
            build(f"{name}_corner_{c}", f"{name.capitalize()} {c.upper()} corner", "corners", (f"corner_{c}",), False,
                  "PSA 10 requires four perfectly sharp corners (official).", f"corner_{c}")
        for e in EDGES:
            build(f"{name}_edge_{e}", f"{name.capitalize()} {e} edge", "edges", (f"edge_{e}",), False,
                  "PSA 10 requires full original gloss and no chipping or whitening along edges (secondary reading of PSA).", f"edge_{e}")
        for key, (label, group, rois, human, text) in _CHAR_META.items():
            if key == "surface_other" and not by_char.get(key):
                continue
            build(f"{name}_{key}", f"{name.capitalize()} {label.lower()}", group, rois, human, text, key)
        # print lines: chrome / refractor / foil need a person
        pl_imp = "open"
        chars.append({"id": f"{name}_print_lines", "label": f"{name.capitalize()} print lines (chrome/refractor/foil)",
                      "group": "print", "side": name, "basis": "official", "human_required": True,
                      "status": "not_assessable", "impact": pl_imp, "grade_cap": (10, 10), "findings": [],
                      "observed": ("Print lines are a common factory flaw on chrome/refractor/foil finishes and PSA counts them as surface "
                                   "defects; a flat photo cannot separate them from artwork." if chrome else
                                   "Print lines cannot be separated from artwork content in a flat photo."),
                      "notes": ["Straight parallel lines found by the relief detector appear under 'linear marks' and may be print lines "
                                "or scratches; both count against a 10."], "detail": None, "priority_hint": 1 if chrome else 3})

    # --- eye appeal (informational, human) ---
    flaws = [f for k in ("front", "back") for f in ((sides[k] or {}).get("findings", []) if sides[k] and sides[k].get("ok") else [])
             if finding_state(f) in ("confirmed", "candidate") and f.get("severity") != "info"]
    vis = [f["id"] for f in flaws if _visibility(f) == "visible"]
    mag = [f["id"] for f in flaws if _visibility(f) == "magnification"]
    cen_f = next((c for c in chars if c["id"] == "front_centering"), None)
    chars.append({"id": "eye_appeal", "label": "Eye appeal", "group": "eye_appeal", "side": "both", "basis": "official",
                  "human_required": True, "status": "limited", "impact": "open", "grade_cap": (10, 10), "findings": vis + mag,
                  "observed": (f"{len(vis)} finding(s) likely visible without magnification, {len(mag)} needing a loupe. "
                               "Eye appeal itself is a PSA grader judgment."),
                  "notes": ["Official: graders may make a judgment call based on the strength or weakness of eye appeal.",
                            "Visibility split is a GradeForge heuristic (length >= 3 mm, area >= 2 mm2, or moderate severity)."]
                  + (["Front centering is in PSA's discretion range, so eye appeal decides it."]
                     if cen_f and (cen_f.get("detail") or {}).get("zone") in ("discretion_band", "straddles_nominal") else []),
                  "detail": {"visible_findings": vis, "magnification_only": mag}})

    return _assemble(result, std, sides, chars, finish, model, reliability)


def _roi_reasons(comp: dict, ids: tuple) -> list:
    out = []
    for r in comp.get("rois", []):
        if r["id"] in ids and r["inspectable"] != "yes":
            out += r["reasons"] or [f"{r['label']}: {r['inspectable']}"]
    return out or ["Coverage is incomplete."]


# ---------------------------------------------------------------------------------------------------------
# Verdict, reasons, inspection plan
# ---------------------------------------------------------------------------------------------------------

def _assemble(result, std, sides, chars, finish, model, reliability):
    chrome = finish in CHROMELIKE
    f, b = sides["front"], sides["back"]
    f_ok, b_ok = bool(f and f.get("ok")), bool(b and b.get("ok"))

    # ----- grade ceilings (dominance: minimum, never an average or a trade-off) -----
    caps_c, caps_o, binding = [10], [10], []
    for c in chars:
        cc, co = c["grade_cap"]
        caps_c.append(cc)
        caps_o.append(co)
    cen = {c["side"]: c for c in chars if c["group"] == "centering"}
    if f_ok and cen["front"].get("detail"):
        fd = cen["front"]["detail"]
        bd = (cen["back"].get("detail") or {})
        c_cons, c_opt = _centering_ceilings(std, fd["worst_pct"], bd.get("worst_pct"), fd["uncertainty_pct"], bd.get("uncertainty_pct", 0.0))
        caps_c.append(c_cons)
        caps_o.append(c_opt)
    # PSA describes a 9 as tolerating one minor flaw (secondary wording): two or more distinct flaw sites cap at 8 (model).
    def sites(include_candidates):
        n = 0
        for c in chars:
            if c["group"] == "centering" or not c["findings"]:
                continue
            want = ("confirmed", "candidate") if include_candidates else ("confirmed",)
            if any(v in want for v in c.get("finding_states", {}).values()):
                n += 1
        return n
    multi_c = sites(True) >= 2
    multi_o = sites(False) >= 2
    if multi_c:
        caps_c.append(8)
    if multi_o:
        caps_o.append(8)
    cons, opt = min(caps_c), min(caps_o)
    opt = max(opt, cons)
    for c in chars:
        if c["group"] != "centering" and c["grade_cap"][0] == cons and cons < 10:
            binding.append(c["id"])
    if cons < 10 and any(c["group"] == "centering" for c in chars) and not binding:
        binding.append("centering")
    if multi_c and cons == 8:
        binding.append("multiple_flaw_sites")

    # ----- PSA 10 eligibility -----
    blockers = [c for c in chars if c["impact"] == "block"]
    likely = [c for c in chars if c["impact"] == "likely_block"]
    possible = [c for c in chars if c["impact"] == "possible"]
    open_items = [c for c in chars if c["impact"] == "open" and not c["human_required"]]
    human_items = [c for c in chars if c["human_required"]]
    evidence_gaps = []
    if not f_ok:
        evidence_gaps.append("front photo missing or unreadable")
    elif cen["front"]["status"] == "not_assessable":
        evidence_gaps.append("front centering could not be measured")
    if not b_ok:
        evidence_gaps.append("back photo missing or unreadable")
    if blockers:
        status = "ruled_out"
    elif likely:
        status = "unlikely"
    elif evidence_gaps:
        status = "insufficient_evidence"
    elif open_items or possible:
        status = "uncertain"
    else:
        status = "credible_candidate"
    if status == "credible_candidate":
        # also require two lighting photos per side so that a clean relief result means something
        for nm, s in (("front", f), ("back", b)):
            if (s["registration"]["lighting_stack"]["images"] < 2) or not s["surface"].get("artwork_relief_assessed"):
                status = "uncertain"

    if status in ("ruled_out", "unlikely"):
        strong = [c for c in chars if IMPACT_RANK[c["impact"]] >= 3 and
                  (c["group"] == "centering" or any(v == "confirmed" for v in c.get("finding_states", {}).values()))]
        basis = "confirmed_or_measured" if strong else "unconfirmed_candidates"
    else:
        basis = None
    reasons9, reasons10 = [], []

    def add9(c, text, strength, source):
        reasons9.append({"characteristic": c["id"], "text": text, "strength": strength, "source": source, "findings": c["findings"]})

    for c in sorted(chars, key=lambda x: -IMPACT_RANK[x["impact"]]):
        if c["impact"] in ("block", "likely_block", "possible"):
            if c["group"] == "centering":
                add9(c, c["observed"], "official_limit" if c["impact"] != "possible" else "near_limit", "official")
            else:
                strength = ("confirmed_defect" if c["impact"] == "block" or any(_is_confirmed(result, fid) for fid in c["findings"])
                            else "unconfirmed_candidate")
                add9(c, f"{c['label']}: " + " ".join(c["notes"][:2]), strength, "model")
    if multi_c:
        reasons9.append({"characteristic": "multiple_flaw_sites", "text": "Two or more separate flaw sites. PSA describes a 9 as allowing one "
                         "minor flaw (secondary wording), so this pattern is capped at 8 by the model.", "strength": "model_rule",
                         "source": "secondary+model", "findings": []})

    for c in chars:
        if c["impact"] == "none" and c["status"] == "clear":
            reasons10.append({"characteristic": c["id"], "text": f"{c['label']}: {c['observed']}", "source": "official" if c["group"] in ("centering", "corners") else "model",
                              "caveat": "A clean result means no flaw was detected at this photo's resolution and coverage, not that none exists.",
                              "findings": []})
        elif c["impact"] == "possible":
            reasons10.append({"characteristic": c["id"], "text": f"{c['label']}: concern is small or unconfirmed, so a PSA 10 remains possible. "
                              + (" ".join(c["notes"][:1])), "source": "model", "caveat": "Depends on review or grader discretion.", "findings": c["findings"]})
    mag_only = [fid for c in chars for fid in c["findings"] if not _is_visible(result, fid)]
    if mag_only and not blockers:
        reasons10.append({"characteristic": "eye_appeal", "text": f"{len(mag_only)} finding(s) are small enough to need magnification; PSA's wording "
                          "allows a slight printing imperfection only if it does not impair overall appeal, and graders weigh eye appeal.",
                          "source": "official+model", "caveat": "Grader judgment; not predictable from photos.", "findings": mag_only})

    # ----- photographic limitations (what could hide a flaw) -----
    limits = _limitations(sides, chars, chrome)
    plan = _inspection_plan(sides, chars, limits, finish, status)

    # ----- evidence sufficiency and most likely grade -----
    gap_chars = [c["id"] for c in chars if c["impact"] == "open" and not c["human_required"]]
    unreviewed = [fid for c in chars for fid in c["findings"] if not _is_confirmed(result, fid)]
    sufficient = (not evidence_gaps and not gap_chars and not unreviewed)
    most_likely = {"grade": None, "basis": None, "note": ""}
    if model and model.get("validated") and model.get("prediction") and sufficient:
        p = model["prediction"]
        most_likely = {"grade": 10 if (p["p_psa10"] >= model["threshold"] and cons >= 10) else min(9, cons), "basis": "trained_model",
                       "note": "From a model that passed GradeForge's validation criteria; see reliability for its measured accuracy. "
                                "The model can never raise a grade above the rules' ceiling."}
    elif sufficient and cons < 10 and cons == opt:
        most_likely = {"grade": cons, "basis": "rules",
                       "note": "Set by reviewed, confirmed defects under GradeForge's uncalibrated rules; not a probability."}
    else:
        why = ("a defect, an open item or an unreviewed candidate remains" if not sufficient else
               "photos cannot confirm a PSA 10 and no validated model exists")
        most_likely["note"] = f"Not given: {why}."

    verdict_lines = {"ruled_out": "PSA 10 not supported by the photos.", "unlikely": "More likely a PSA 9 (or lower) than a PSA 10.",
                     "uncertain": "A PSA 10 is still plausible; open items decide it.", "credible_candidate": "Credible PSA 10 candidate on photo evidence.",
                     "insufficient_evidence": "Not enough evidence to assess."}
    return {
        "engine_version": GEM_VERSION,
        "card_traits": {"finish": finish},
        "psa10_eligibility": {"status": status, "label": verdict_lines[status], "summary": ELIGIBILITY_TEXT[status],
                              "evidence_gaps": evidence_gaps, "basis": basis},
        "grade_ceiling": {"conservative": cons, "optimistic": opt, "multi_flaw_rule_applied": multi_c},
        "most_likely": most_likely,
        "dominance": {"rule": "The grade ceiling is the minimum across all characteristics. Centering has no score that can raise it.",
                      "offsets_applied": False, "binding_characteristics": binding},
        "reasons_psa9": reasons9,
        "reasons_psa10_plausible": reasons10,
        "additional_inspection": plan,
        "photographic_limitations": limits,
        "characteristics": chars,
        "human_required": [{"id": c["id"], "label": c["label"], "why": (c["notes"] or [c["observed"]])[0]} for c in human_items],
        "evidence_sufficient": sufficient,
        "reliability": _reliability_block(reliability, model),
        "standards": {"database_version": std.version, "official_items": std.official_items,
                      "official_page_verified": std.official_page_verified,
                      "official_page_partially_verified": bool(std.data.get("official_page_partially_verified")),
                      "model_assumptions": [
                          "Defect-to-ceiling caps (engine.FINDING_CAPS) and the 'two flaw sites => 8' rule are GradeForge assumptions.",
                          f"A flaw is 'visible without magnification' at >= {VISIBLE_LENGTH_MM} mm length, >= {VISIBLE_AREA_MM2} mm2 area or moderate severity.",
                          "Unconfirmed detector candidates never rule a PSA 10 out; only confirmed defects or measured centering beyond PSA's stated range do.",
                          "PSA 9 reverse centering (90/10) and the 'one minor flaw' wording are secondary-source values."]},
    }


def _is_confirmed(result: dict, fid: str) -> bool:
    for k in ("front", "back"):
        s = result.get(k)
        if s and s.get("ok"):
            for f in s["findings"]:
                if f["id"] == fid:
                    return f.get("status") == "confirmed"
    return False


def _is_visible(result: dict, fid: str) -> bool:
    for k in ("front", "back"):
        s = result.get(k)
        if s and s.get("ok"):
            for f in s["findings"]:
                if f["id"] == fid:
                    return _visibility(f) == "visible"
    return False


def _reliability_block(reliability: dict | None, model: dict | None) -> dict:
    r = dict(reliability or {})
    r.setdefault("rules_validated", False)
    r.setdefault("model_validated", bool(model and model.get("validated")))
    r["statement"] = ("These rules are uncalibrated assumptions. No probability is reported unless a trained model has passed GradeForge's "
                      "cross-validated criteria on real PSA outcomes, and even then it is a statistical estimate, not a guarantee.")
    if model:
        r["model"] = {"version": model.get("version"), "threshold": model.get("threshold"), "validated": bool(model.get("validated"))}
    if model and model.get("validated") and model.get("prediction"):
        r["model_prediction"] = model["prediction"]
    return r


def _limitations(sides, chars, chrome) -> list:
    out = []
    for name, s in sides.items():
        if not (s and s.get("ok")):
            out.append({"side": name, "region": "whole side", "why": f"No usable {name} photo.",
                        "could_hide": ["every defect on that side"], "fix": f"Upload a sharp, square-on {name} photo."})
            continue
        floor = s["resolution"]["min_resolvable_feature_mm"]
        out.append({"side": name, "region": "fine detail", "why": f"This photo resolves features of about {floor:.2f} mm or wider; "
                    "hairline scratches and microscopic chipping are narrower than that.",
                    "could_hide": ["hairline scratches (their bright streak may still show under raking light)", "microscopic chipping", "light fraying"],
                    "fix": "Macro photos of corners/edges and two raking-light photos of the surface."})
        for r in s["completeness"]["rois"]:
            if r["id"] == "artwork_print_and_stain":
                continue
            if r["inspectable"] != "yes":
                out.append({"side": name, "region": r["label"], "why": " ".join(r["reasons"]) or "Incomplete coverage.",
                            "could_hide": _could_hide(r["id"], chrome), "fix": " ".join(r["requests"]) or "Retake this region."})
        if s["quality"]["issues"]:
            out.append({"side": name, "region": "image quality", "why": " ".join(s["quality"]["issues"][:3]),
                        "could_hide": ["soft-focus hides print and surface defects"], "fix": " ".join(s["quality"]["requests"][:2])})
        if s["surface"]["glare"]["clipped_fraction"] > 0.01:
            out.append({"side": name, "region": "glare areas", "why": f"{s['surface']['glare']['clipped_fraction'] * 100:.1f}% of the card is blown out.",
                        "could_hide": ["scratches, print lines or dents under the highlight"], "fix": "Retake with diffuse light or a polarizer."})
    return out


def _could_hide(roi_id: str, chrome: bool) -> list:
    if roi_id.startswith("corner"):
        return ["corner whitening", "softness/rounding", "chipping", "fraying"]
    if roi_id.startswith("edge"):
        return ["edge whitening", "nicks and rough cuts", "peeling or delamination"]
    if roi_id == "artwork_relief":
        return ["hairline scratches", "dents and indentations", "creases", "gloss loss", "roller marks"] + (["print lines"] if chrome else [])
    if roi_id == "border_surface":
        return ["border scratches", "spots and stains"]
    if roi_id == "centering":
        return ["centering error larger than the stated uncertainty"]
    return ["unseen flaws"]


def _inspection_plan(sides, chars, limits, finish, status) -> list:
    plan, seen = [], set()
    chrome = finish in CHROMELIKE

    def add(pid, prio, kind, what, how, resolves, upload=None):
        if pid in seen:
            return
        seen.add(pid)
        plan.append({"id": pid, "priority": prio, "type": kind, "what": what, "how": how, "resolves": resolves, "upload_to": upload or []})

    for name, s in sides.items():
        if not (s and s.get("ok")):
            add(f"{name}_photo", 1, "photo", f"{name.capitalize()} photo",
                f"Photograph the {name} square-on (camera parallel to the card), card at least 1200 px wide, plain contrasting background, diffuse light.",
                [f"{name}_centering"], [f"{name.capitalize()} \u2192 {shots.HEADERS['overall']}"])
            continue
        c = next(x for x in chars if x["id"] == f"{name}_centering")
        d = c.get("detail") or {}
        if c["status"] == "not_assessable" or (d and (not d["resolves_50_vs_55"] or d["zone"] in ("straddles_nominal", "straddles_outer", "discretion_band"))):
            add(f"{name}_centering", 1, "photo", f"Re-measure {name} centering more precisely",
                "Reshoot square-on with the whole card and its border in frame, card at least 1500 px wide, and keep the card flat. "
                "As a physical check, measure the four border widths with a ruler or loupe; the larger share over the sum of both is the percentage.",
                [f"{name}_centering"], [f"{name.capitalize()} \u2192 {shots.HEADERS['overall']}"])
        relief = [x for x in chars if x["side"] == name and x["id"].endswith(("linear_marks", "dents", "finish"))]
        if any(x["status"] in ("not_assessable", "limited") for x in relief) or s["registration"]["lighting_stack"]["images"] < 2:
            add(f"{name}_raking", 1, "photo", f"Two more {name} photos with a single light from different sides",
                "Keep the camera and card fixed. Using a single light source, take one photo with the lamp low (about 15-30 degrees above the surface) from the left and "
                "another from the right, then upload both as extra lighting angles for the same side. Avoid any blown-out highlights on the card.",
                [f"{name}_linear_marks", f"{name}_dents", f"{name}_finish"], [f"{name.capitalize()} \u2192 {shots.HEADERS['angle']}"])
        have = {m.get("slot") for m in s["registration"].get("macros", []) if m.get("registered") and m.get("slot")}
        trouble = ("not_assessable", "limited", "unresolved", "minor_concern", "blocks_10", "severe")
        need_c = [k for k in CORNERS if next(x for x in chars if x["id"] == f"{name}_corner_{k}")["status"] in trouble and f"corner_{k}" not in have]
        if need_c:
            add(f"{name}_corner_macros", 2, "photo", "Close-up photos of the " + ", ".join(shots.CORNER_NAMES[k] for k in need_c) + f" {name} corner" + ("s" if len(need_c) > 1 else ""),
                shots.HOWTO["corner"], [f"{name}_corner_{q}" for q in need_c], [shots.upload_path(name, f"corner_{k}") for k in need_c])
        need_e = [k for k in EDGES if next(x for x in chars if x["id"] == f"{name}_edge_{k}")["status"] in trouble and f"edge_{k}" not in have]
        if need_e:
            add(f"{name}_edge_macros", 2, "photo", "Close-up photos of the " + ", ".join(shots.EDGE_NAMES[k] for k in need_e) + f" ({name})",
                shots.HOWTO["edge"], [f"{name}_edge_{q}" for q in need_e], [shots.upload_path(name, f"edge_{k}") for k in need_e])
        if s["surface"]["glare"]["clipped_fraction"] > 0.01:
            add(f"{name}_glare", 2, "photo", f"Retake {name} without glare", "Use diffuse light, move the lamp, or use a polarizer so no area is blown out.",
                [f"{name}_linear_marks", f"{name}_dents"], [f"{name.capitalize()} \u2192 {shots.HEADERS['overall']}"])
    # physical checks: the things photos cannot settle
    add("loupe", 2, "physical", "10x loupe check of all corners and edges, both sides",
        "Under bright diffuse light, look for whitening, softness and nicks. Handle by the edges, clean dry hands, over a soft surface.",
        [x["id"] for x in chars if x["group"] in ("corners", "edges")])
    add("tilt", 1 if chrome else 3, "physical", "Tilt-and-rotate check for print lines and dents",
        "Hold the card under one point light and tilt slowly through several angles while watching the reflection; look for straight parallel lines "
        "(print lines), shadows that appear and vanish (dents) and dull patches (gloss loss)." +
        (" Chrome, refractor and foil finishes need this most." if chrome else ""),
        [f"{n}_print_lines" for n in ("front", "back")] + [f"{n}_finish" for n in ("front", "back")] + [f"{n}_dents" for n in ("front", "back")])
    add("back_stain", 3, "physical", "Inspect the back for stains, wax or gum marks",
        "PSA 10 must be free of staining of any kind (official). Check the back and border under bright light; photos only check borders automatically.",
        ["front_stains", "back_stains"])
    add("arm_length", 3, "physical", "Arm's-length eye-appeal check",
        "View the card at normal handling distance under bright light. Anything that catches the eye without magnification hurts eye appeal.",
        ["eye_appeal"])
    plan.sort(key=lambda p: (p["priority"], p["type"] != "photo"))
    return plan
