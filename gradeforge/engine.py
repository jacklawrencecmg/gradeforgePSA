"""Transparent grading engine.

Two layers are kept apart on purpose:
  * PSA's published standards (standards_db/*.json): centering limits per grade.
  * GradeForge's own model (this file): how measured defect candidates cap a grade.
    Those caps are UNCALIBRATED assumptions until validated on labeled PSA outcomes.

The engine produces grade CEILINGS (upper bounds), not a point estimate, because
photographs cannot confirm the absence of every flaw PSA looks for. No
probabilities are produced; confidence is described as inspection coverage.
"""
from __future__ import annotations

from .standards import Standards, load_standards

MODEL_VERSION = "gradeforge-rules-0.2 (uncalibrated)"

# (conservative cap, optimistic cap) by observation status. Model assumptions.
WEAR_CAPS = {
    "clean_observed": (10, 10),
    "minor_candidate": (9, 10),
    "visible_candidate": (8, 9),
    "not_measurable": (10, 10),
}
SURFACE_CAPS = {
    "persistent_linear": (8, 9),
    "other_candidate": (9, 10),
}


def _centering_ceilings(std: Standards, front_pct, back_pct, f_unc, b_unc):
    def ceil(fr, bk, leeway):
        for g in (10, 9, 8, 7, 6):
            lf, lb = std.front_limit(g), std.back_limit(g)
            lf = lf + (std.leeway_pct if (leeway and g >= std.leeway_min_grade) else 0)
            if fr <= lf and (bk is None or bk <= lb):
                return g
        return 5
    cons = ceil(front_pct + f_unc, None if back_pct is None else back_pct + b_unc, False)
    opt = ceil(front_pct - f_unc, None if back_pct is None else max(0.0, back_pct - b_unc), True)
    return cons, opt


def _centering_status(pct, unc, limit, leeway=0.0):
    if pct + unc <= limit:
        return "pass_observed"
    if pct - unc > limit + leeway:
        return "fail_observed"
    if pct - unc > limit:
        return "fail_nominal_leeway_possible" if leeway else "borderline"
    return "borderline"



# Per-finding model caps (conservative, optimistic) by review state and severity. UNCALIBRATED assumptions.
FINDING_CAPS = {
    "confirmed": {"minor": (9, 9), "moderate": (8, 8), "info": (10, 10)},
    "candidate": {"minor": (9, 10), "moderate": (8, 9), "info": (10, 10)},
    "unresolved": {"minor": (9, 10), "moderate": (9, 10), "info": (10, 10)},
}
_PHYS = ("physical_probable_albedo", "physical_probable_relief", "geometry_measurement", "manual_observation")


def finding_state(f: dict):
    """confirmed | candidate | unresolved | None (rejected or artifact: carries no cap)."""
    if f.get("status") == "rejected":
        return None
    if f.get("status") == "confirmed":
        return "confirmed"
    c = f.get("classification", "")
    if c in _PHYS:
        return "candidate"
    if c.startswith("indeterminate"):
        return "unresolved"
    return None


def _group_findings(findings: list) -> dict:
    groups: dict = {}
    for f in findings:
        t = f.get("target", "")
        key = (t if t.startswith(("corner_", "edge_")) else "border" if t.startswith("ring_") else
               "artwork" if t == "artwork" else "other")
        groups.setdefault(key, []).append(f)
    return groups


def _item_state(fs: list):
    best, cap, ids, notes = None, None, [], []
    order = {"confirmed": 0, "candidate": 1, "unresolved": 2}
    for f in fs:
        st = finding_state(f)
        if st is None:
            continue
        c = FINDING_CAPS[st][f.get("severity", "minor")]
        if c[0] >= 10 and c[1] >= 10:
            continue
        ids.append(f["id"])
        cap = c if cap is None else (min(cap[0], c[0]), min(cap[1], c[1]))
        if best is None or order[st] < order[best[0]]:
            best = (st, f.get("severity"))
    if best is None:
        return "pass_observed", None, [], ""
    st, sev = best
    status = "fail_confirmed" if st == "confirmed" else ("fail_candidate" if sev == "moderate" and st == "candidate" else "borderline")
    detail = {"confirmed": "Confirmed by a reviewer with attached evidence.",
              "candidate": "Unconfirmed automatic candidate: review the evidence crops to confirm or reject.",
              "unresolved": "Automatic classification could not separate damage from artifact: add photos or review manually."}[st]
    return status, cap, ids, detail


def _ready(side: dict | None):
    return bool(side and side.get("ok"))


def assess_card(front: dict | None, back: dict | None, std: Standards | None = None) -> dict:
    std = std or load_standards()
    out = {
        "model_version": MODEL_VERSION,
        "standards": std.meta(),
        "calibration": {"available": False,
                        "note": "No calibrated model. Ceilings come from PSA centering limits plus uncalibrated "
                                "defect caps. No probability is reported."},
        "probabilities_reported": False,
    }
    requests, blockers = [], []
    sides = {"front": front, "back": back}

    for name, s in sides.items():
        if s is None:
            blockers.append(f"{name} image missing")
            requests.append(f"Upload the {name} of the card. PSA grades both sides, so a prediction cannot be made without it.")
        elif not s.get("ok"):
            blockers.append(f"{name}: {s.get('reason', 'analysis failed')}")
            requests.append(f"{name.capitalize()}: {s.get('reason', 'analysis failed')}")
        else:
            q = s["quality"]
            requests += [f"{name.capitalize()}: {r}" for r in q["requests"]]
            for part in ("corners", "edges"):
                if not q["sufficient_for"][part]:
                    blockers.append(f"{name}: resolution too low to inspect {part}")
            if name == "front" and not q["sufficient_for"]["centering"]:
                blockers.append("front: centering could not be measured reliably")
                reason = s["centering"].get("reason") or "centering not measurable"
                requests.append(f"Front: {reason}")

    partial = {}
    for name, s in sides.items():
        if _ready(s) and s["centering"].get("measurable"):
            c = s["centering"]
            partial[f"{name}_centering"] = {"worst_pct": c["worst_pct"], "uncertainty_pct": c["uncertainty_pct"],
                                            "lr_split": c["lr_split"], "tb_split": c["tb_split"]}
    if blockers:
        out.update({
            "prediction_status": "insufficient_evidence",
            "reasons": blockers, "requests": list(dict.fromkeys(requests)),
            "partial_measurements": partial,
            "conservative_ceiling": None, "optimistic_ceiling": None, "most_likely": None,
            "psa10_eligibility": {"status": "indeterminate",
                                  "detail": "Not enough inspectable evidence to rule PSA 10 in or out."},
            "evidence_level": "insufficient", "checklist": [], "grade_limiting": [],
        })
        return out

    checklist, caps_cons, caps_opt, limiting = [], [], [], []

    def add(item_id, req, observed, status, detail="", cap=None):
        checklist.append({"id": item_id, "requirement": req, "observed": observed, "status": status, "detail": detail})
        if cap:
            caps_cons.append(cap[0]); caps_opt.append(cap[1])
            if cap[0] < 10 or cap[1] < 10:
                limiting.append({"id": item_id, "requirement": req, "conservative_cap": cap[0],
                                 "optimistic_cap": cap[1], "detail": detail})

    # ---- Centering (PSA-published limits, measured) ----
    fc = front["centering"]
    bc = back["centering"] if back["centering"].get("measurable") else None
    f_pct, f_unc = fc["worst_pct"], fc["uncertainty_pct"]
    b_pct = bc["worst_pct"] if bc else None
    b_unc = bc["uncertainty_pct"] if bc else 0.0
    c_cons, c_opt = _centering_ceilings(std, f_pct, b_pct, f_unc, b_unc)
    caps_cons.append(c_cons); caps_opt.append(c_opt)
    if c_cons < 10 or c_opt < 10:
        limiting.append({"id": "centering", "requirement": "PSA centering limits",
                         "conservative_cap": c_cons, "optimistic_cap": c_opt,
                         "detail": f"Front {f_pct:.1f}% ±{f_unc:.1f}" +
                                   (f", back {b_pct:.1f}% ±{b_unc:.1f}" if bc else ", back not measurable")})
    lim10f, lim10b = std.front_limit(10), std.back_limit(10)
    st_f = _centering_status(f_pct, f_unc, lim10f, std.leeway_pct)
    add("front_centering", f"Front centering {lim10f}/{100 - lim10f} or better (PSA 10)",
        f"{f_pct:.1f}/{100 - f_pct:.1f} ±{f_unc:.1f} on {fc['worst_axis']}", st_f,
        "Outside nominal limit but inside the reported 5% leeway: a 10 would depend on PSA discretion."
        if st_f == "fail_nominal_leeway_possible" else "")
    if bc:
        st_b = _centering_status(b_pct, b_unc, lim10b)
        add("back_centering", f"Back centering {lim10b}/{100 - lim10b} or better (PSA 10)",
            f"{b_pct:.1f}/{100 - b_pct:.1f} ±{b_unc:.1f} on {bc['worst_axis']}", st_b)
    else:
        add("back_centering", f"Back centering {lim10b}/{100 - lim10b} or better (PSA 10)",
            "not measurable", "unknown_not_assessed", bc and "" or back["centering"].get("reason", ""))

    # ---- Findings-driven items (model caps; uncalibrated) ----
    for sname, s in sides.items():
        comp = {r["id"]: r for r in s.get("completeness", {}).get("rois", [])}
        groups = _group_findings(s.get("findings", []))

        def item(item_id, label, roi_id, key, pass_text, unassessed_text):
            fs = groups.get(key, [])
            state, cap, ids, detail = _item_state(fs)
            roi = comp.get(roi_id, {"inspectable": "yes", "reasons": []})
            note = " ".join(roi["reasons"])
            if state in ("fail_confirmed", "fail_candidate", "borderline"):
                observed = f"{len(fs)} finding(s): " + ", ".join(f"{f['id']} {f['label'].lower()}" for f in fs[:4])
                add(item_id, label, observed, state, (detail + " " + note).strip(), cap)
                checklist[-1]["finding_ids"] = ids
            elif roi["inspectable"] == "yes":
                add(item_id, label, pass_text, "pass_observed", note)
            else:
                add(item_id, label, "not assessed" if roi["inspectable"] == "no" else "partly assessed",
                    "unknown_not_assessed", (unassessed_text + " " + note).strip())

        for c in s["wear"]["corners"]:
            item(f"{sname}_corner_{c['id']}", f"{sname} {c['id']} corner sharp (PSA 10)", f"corner_{c['id']}",
                 f"corner_{c['id']}", "no corner finding above threshold", "")
        for e in s["wear"]["edges"]:
            item(f"{sname}_edge_{e['id']}", f"{sname} {e['id']} edge clean (PSA 10)", f"edge_{e['id']}",
                 f"edge_{e['id']}", "no edge finding above threshold", "")
        item(f"{sname}_border_surface", f"{sname} border surface free of marks (PSA 10)", "border_surface",
             "border", "no border mark above threshold", s["surface"]["ring"].get("reason", ""))
        item(f"{sname}_artwork_relief", f"{sname} artwork free of scratches, dents, creases, gloss loss (lighting-based)",
             "artwork_relief", "artwork", "no relief candidate in the lighting differences", "")
        add(f"{sname}_artwork_print_and_stain", f"{sname} artwork free of print lines, staining, foil damage",
            "not assessed", "unknown_not_assessed", s["surface"]["artwork_note"])
        add(f"{sname}_gloss_focus", f"{sname} full original gloss and sharp focus",
            f"sharpness index {s['quality']['sharpness_var']}", "unknown_not_assessed",
            "Gloss needs raking-light inspection; the sharpness index is image focus, not print focus.")
        other = [f for f in groups.get("other", []) if f.get("status") == "confirmed"]
        if other:
            state, cap, ids, detail = _item_state(other)
            add(f"{sname}_other_confirmed", f"{sname} other confirmed observations", f"{len(other)} confirmed", state, detail, cap)
            checklist[-1]["finding_ids"] = ids
    add("authenticity", "No alteration, trimming, recoloring or counterfeit indicators", "not assessed",
        "unknown_not_assessed", "No automated authentication. Alteration indicators (if any) are listed as findings; "
        "have a person inspect any high-value card.")

    # ---- Combine ----
    unknown_ids = [c["id"] for c in checklist if c["status"] == "unknown_not_assessed"]
    conservative = min(caps_cons)
    optimistic = min(caps_opt)
    if unknown_ids:
        conservative = min(conservative, 9)   # a 10 needs every criterion positively verified
    optimistic = max(optimistic, conservative)

    statuses = {c["id"]: c["status"] for c in checklist}
    hard_fail = [i for i in ("front_centering", "back_centering") if statuses.get(i) == "fail_observed"]
    leeway_only = statuses.get("front_centering") == "fail_nominal_leeway_possible"
    cand_fail = [i for i, s in statuses.items() if s in ("fail_candidate", "fail_confirmed")]
    if hard_fail:
        p10 = {"status": "ruled_out", "detail": "Measured centering exceeds the PSA 10 limit even after measurement error and leeway: " + ", ".join(hard_fail) + "."}
    elif leeway_only or cand_fail:
        why = []
        if leeway_only:
            why.append("front centering is outside the nominal PSA 10 limit (leeway/discretion only)")
        if cand_fail:
            why.append("defect candidates: " + ", ".join(cand_fail))
        p10 = {"status": "unlikely", "detail": "; ".join(why) + "."}
    else:
        p10 = {"status": "not_ruled_out",
               "detail": "No measured criterion rules out PSA 10, but it cannot be confirmed from photos: "
                         f"{len(unknown_ids)} criteria were not assessed.",
               "unassessed": unknown_ids}

    assessed = sum(1 for c in checklist if c["status"] != "unknown_not_assessed")
    coverage = assessed / len(checklist)
    evidence = "low" if coverage < 0.5 else "moderate"   # never 'high': print/stain and authenticity are always unassessed

    nine_vs_ten = {
        "summary": ("PSA 10 ruled out by measurement; the ceiling is 9 or lower." if p10["status"] == "ruled_out" else
                    "PSA 10 is unlikely given the items marked failing." if p10["status"] == "unlikely" else
                    "Measured criteria allow either a 9 or a 10. Distinguishing them needs the unassessed items below, "
                    "which this tool cannot judge from photos."),
        "decisive_unknowns": unknown_ids,
        "borderline": [i for i, s in statuses.items() if s == "borderline"],
        "failing": [i for i, s in statuses.items() if s.startswith("fail")],
        "unreviewed_candidates": [c["id"] for c in checklist if c["status"] in ("fail_candidate", "borderline") and c["id"] not in ("front_centering", "back_centering")],
    }
    limiting.sort(key=lambda d: (d["conservative_cap"], d["optimistic_cap"]))
    out.update({
        "prediction_status": "ceilings_only",
        "conservative_ceiling": conservative, "optimistic_ceiling": optimistic,
        "most_likely": None,
        "most_likely_note": "No point estimate is given: ceilings are upper bounds and no calibrated model exists. "
                            "A point estimate requires labeled PSA outcomes (see validation module).",
        "psa10_eligibility": p10, "grade_limiting": limiting, "checklist": checklist,
        "nine_vs_ten": nine_vs_ten, "evidence_level": evidence,
        "inspection_coverage": {"assessed": assessed, "total": len(checklist)},
        "requests": list(dict.fromkeys(requests)),
        "reasons": [],
    })
    return out
