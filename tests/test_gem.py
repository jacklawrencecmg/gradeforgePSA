import pytest

from gradeforge.gem import analyze_gem
from gradeforge.standards import load_standards
from tests.gem_fixtures import finding, result, side

STD = load_standards()


def gem(**kw):
    return analyze_gem(result(**kw), STD)


def char(g, cid):
    return next(c for c in g["characteristics"] if c["id"] == cid)


def test_standards_record_official_vs_secondary():
    m = STD.meta()
    assert m["official_page_partially_verified"] and not m["official_page_verified"]
    ids = {i["id"] for i in m["official_items"]}
    assert {"psa10_front_centering", "psa10_back_centering", "psa9_front_centering", "front_leeway", "eye_appeal"} <= ids
    item = next(i for i in m["official_items"] if i["id"] == "psa10_front_centering")
    assert (item["min_pct"], item["max_pct"]) == (55, 60)       # the official wording is a range, not a single number


def test_clean_complete_card_is_credible_candidate_but_never_confirmed():
    g = gem()
    assert g["psa10_eligibility"]["status"] == "credible_candidate"
    assert g["most_likely"]["grade"] is None                     # photos cannot confirm a 10, no validated model
    assert g["reasons_psa10_plausible"] and all(r["caveat"] or r["source"] for r in g["reasons_psa10_plausible"])
    assert g["reasons_psa9"] == []
    # things photos can never clear stay uncleared
    for cid in ("front_print_lines", "front_finish", "front_stains", "front_manufacturing", "back_print_lines", "eye_appeal"):
        assert char(g, cid)["status"] != "clear" and char(g, cid)["human_required"]
    assert {h["id"] for h in g["human_required"]} >= {"front_print_lines", "eye_appeal"}
    assert any(p["type"] == "physical" for p in g["additional_inspection"])
    assert "probab" not in str(g["psa10_eligibility"]).lower().replace("not a prediction", "")


@pytest.mark.parametrize("pct,unc,zone,status", [
    (50.0, 0.5, "inside_nominal", "credible_candidate"),
    (54.5, 1.5, "straddles_nominal", "uncertain"),
    (57.5, 0.5, "discretion_band", "uncertain"),
    (59.5, 1.0, "straddles_outer", "unlikely"),
    (63.0, 1.0, "beyond_outer", "ruled_out"),
])
def test_front_centering_zones(pct, unc, zone, status):
    g = gem(front=side("front", pct=pct, unc=unc))
    c = char(g, "front_centering")
    assert c["detail"]["zone"] == zone
    assert g["psa10_eligibility"]["status"] == status


def test_discretion_band_cites_official_range_and_eye_appeal():
    g = gem(front=side("front", pct=57.5, unc=0.5))
    txt = " ".join(r["text"] for r in g["reasons_psa9"])
    assert "55/45 to 60/40" in txt and "eye appeal" in txt
    assert any(r["source"] == "official" for r in g["reasons_psa9"])


def test_50_vs_55_resolution_flag():
    assert char(gem(front=side("front", pct=52, unc=0.9)), "front_centering")["detail"]["resolves_50_vs_55"] is True
    c = char(gem(front=side("front", pct=52, unc=3.0)), "front_centering")
    assert c["detail"]["resolves_50_vs_55"] is False and any("50/50" in n for n in c["notes"])
    assert any(p["id"] == "front_centering" for p in gem(front=side("front", pct=52, unc=3.0))["additional_inspection"])


def test_reverse_centering_limits():
    g = gem(back=side("back", pct=80.0, unc=1.0))
    assert g["psa10_eligibility"]["status"] == "ruled_out"
    assert any(r["characteristic"] == "back_centering" for r in g["reasons_psa9"])
    g = gem(back=side("back", pct=74.0, unc=2.0))
    assert char(g, "back_centering")["detail"]["zone"] == "straddles_nominal"
    assert g["psa10_eligibility"]["status"] == "uncertain"
    g = gem(back=side("back", measurable=False))
    assert char(g, "back_centering")["status"] == "not_assessable"
    assert any(p["id"] == "back_centering" for p in g["additional_inspection"])


def test_excellent_centering_never_offsets_a_severe_defect():
    bad = finding("front", "corner_missing_stock", "corner_tl", "moderate", "geometry_measurement", status="confirmed",
                  missing_area_mm2=8.7, radius_mm=6.9)
    g = gem(front=side("front", pct=50.0, unc=0.2, findings=[bad]))
    assert g["psa10_eligibility"]["status"] == "ruled_out"
    assert g["grade_ceiling"]["conservative"] <= 8 and g["grade_ceiling"]["optimistic"] <= 8
    assert g["dominance"]["offsets_applied"] is False
    assert "front_corner_tl" in g["dominance"]["binding_characteristics"]
    # perfect 50/50 on both sides changes nothing about the ceiling versus mediocre centering
    g2 = gem(front=side("front", pct=57.0, unc=0.5, findings=[bad]), back=side("back", pct=60.0))
    assert g2["grade_ceiling"]["conservative"] == g["grade_ceiling"]["conservative"] or g2["grade_ceiling"]["conservative"] < g["grade_ceiling"]["conservative"]


def test_one_confirmed_small_flaw_gives_nine_ceiling_and_rules_most_likely():
    f = finding("front", "corner_whitening", "corner_tr", "minor", "physical_probable_albedo", status="confirmed", length_mm=0.8)
    g = gem(front=side("front", findings=[f]))
    assert g["psa10_eligibility"]["status"] == "unlikely" and g["psa10_eligibility"]["basis"] == "confirmed_or_measured"
    assert g["grade_ceiling"] == {"conservative": 9, "optimistic": 9, "multi_flaw_rule_applied": False}
    assert g["most_likely"]["grade"] == 9 and g["most_likely"]["basis"] == "rules"
    assert any(r["strength"] == "confirmed_defect" for r in g["reasons_psa9"])
    assert any("magnification" in r["text"] for r in g["reasons_psa10_plausible"])   # small flaws noted as the plausible-10 path


def test_two_confirmed_flaw_sites_cap_at_eight():
    a = finding("front", "corner_whitening", "corner_tr", "minor", status="confirmed")
    b = finding("front", "edge_whitening", "edge_top", "minor", status="confirmed", length_mm=1.0)
    g = gem(front=side("front", findings=[a, b]))
    assert g["grade_ceiling"]["conservative"] == 8 and g["grade_ceiling"]["multi_flaw_rule_applied"]


def test_unconfirmed_candidates_do_not_rule_out_and_block_a_most_likely_grade():
    c = finding("front", "surface_spot", "ring_left", "minor", status="candidate", area_mm2=0.4)
    g = gem(front=side("front", findings=[c]))
    assert g["psa10_eligibility"]["status"] == "uncertain"
    assert (g["grade_ceiling"]["conservative"], g["grade_ceiling"]["optimistic"]) == (9, 10)
    assert g["most_likely"]["grade"] is None
    assert any(r["strength"] == "unconfirmed_candidate" for r in g["reasons_psa9"])


def test_unconfirmed_moderate_candidate_is_unlikely_but_labeled_unconfirmed():
    c = finding("front", "surface_relief_line", "artwork", "moderate", "physical_probable_relief", length_mm=26.0)
    g = gem(front=side("front", findings=[c]))
    assert g["psa10_eligibility"]["status"] == "unlikely" and g["psa10_eligibility"]["basis"] == "unconfirmed_candidates"


def test_rejected_finding_is_ignored():
    c = finding("front", "corner_whitening", "corner_tr", "moderate", status="rejected")
    assert gem(front=side("front", findings=[c]))["psa10_eligibility"]["status"] == "credible_candidate"


def test_artifacts_and_unresolved_are_not_defects():
    glare = finding("front", "glare_region", "glare:light1", "info", "artifact_glare", area_mm2=30)
    unres = finding("front", "surface_relief_line", "artwork", "minor", "indeterminate_angle_dependent")
    g = gem(front=side("front", findings=[glare, unres]))
    assert g["psa10_eligibility"]["status"] == "uncertain"
    assert char(g, "front_linear_marks")["status"] == "unresolved"
    assert g["grade_ceiling"]["conservative"] == 9 and g["grade_ceiling"]["optimistic"] == 10


def test_defects_obscured_by_photographic_limits_are_reported_with_exact_fixes():
    g = gem(front=side("front", relief="no", border="partial", lights=1), back=side("back", relief="no", lights=1))
    assert g["psa10_eligibility"]["status"] == "uncertain"          # cannot be credible with relief unassessed
    for cid in ("front_linear_marks", "front_dents", "front_finish"):
        assert char(g, cid)["status"] in ("not_assessable", "limited")
    regions = {(l["side"], l["region"]) for l in g["photographic_limitations"]}
    assert ("front", "Artwork relief / gloss") in regions and ("front", "fine detail") in regions
    hide = next(l for l in g["photographic_limitations"] if l["region"] == "Artwork relief / gloss")["could_hide"]
    assert "hairline scratches" in hide and "dents and indentations" in hide
    ins = {p["id"]: p for p in g["additional_inspection"]}
    assert "single light" in ins["front_raking"]["how"] and "15-30 degrees" in ins["front_raking"]["how"]
    assert ins["front_raking"]["priority"] == 1 and "front_dents" in ins["front_raking"]["resolves"]


def test_chrome_refractor_print_lines_need_a_person_and_are_prioritised():
    chrome = gem(front=side("front"), finish="refractor")
    std = gem()
    tilt = lambda g: next(p for p in g["additional_inspection"] if p["id"] == "tilt")
    assert tilt(chrome)["priority"] == 1 and tilt(std)["priority"] == 3
    assert "print lines" in char(chrome, "front_print_lines")["observed"].lower()
    assert chrome["card_traits"]["finish"] == "refractor"
    # a line-shaped relief finding is reported as a scratch-or-print-line, both of which count against a 10
    lin = finding("front", "surface_relief_line", "artwork", "minor", "physical_probable_relief", length_mm=2.5)
    g = analyze_gem(result(front=side("front", findings=[lin]), finish="chrome"), STD)
    assert char(g, "front_linear_marks")["impact"] == "possible"
    assert any("print line" in n for n in char(g, "front_print_lines")["notes"])


def test_missing_back_is_insufficient_but_front_centering_can_still_rule_out():
    r = result()
    r["back"] = None
    g = analyze_gem(r, STD)
    assert g["psa10_eligibility"]["status"] == "insufficient_evidence"
    assert any(p["id"] == "back_photo" for p in g["additional_inspection"])
    r = result(front=side("front", pct=64.0, unc=0.5))
    r["back"] = None
    assert analyze_gem(r, STD)["psa10_eligibility"]["status"] == "ruled_out"


def test_no_probability_without_validated_model():
    g = gem()
    blob = str(g).lower()
    assert "p_psa10" not in blob
    assert g["reliability"]["model_validated"] is False and g["reliability"]["rules_validated"] is False
    assert "uncalibrated" in g["reliability"]["statement"]


def test_validated_model_prediction_only_used_when_evidence_is_sufficient():
    model = {"trained": True, "validated": True, "threshold": 0.5, "n_train": 400, "criteria_results": {}, "trained_at": 0,
             "prediction": {"p_psa10": 0.8, "interpretation": "x"}}
    g = analyze_gem(result(), STD, model=model)
    assert g["evidence_sufficient"] is True
    assert g["most_likely"]["basis"] == "trained_model" and g["most_likely"]["grade"] == 10
    assert g["reliability"]["model_prediction"]["p_psa10"] == 0.8
    # an unreviewed candidate, or a missing lighting set, means evidence is not sufficient: the model is not used
    c = finding("front", "surface_spot", "ring_left", "minor", status="candidate", area_mm2=0.4)
    g = analyze_gem(result(front=side("front", findings=[c])), STD, model=model)
    assert g["evidence_sufficient"] is False and g["most_likely"]["basis"] is None
    g = analyze_gem(result(front=side("front", relief="no", lights=1)), STD, model=model)
    assert g["most_likely"]["basis"] is None
    # an unvalidated model is never used
    g = analyze_gem(result(), STD, model={**model, "validated": False})
    assert g["most_likely"]["basis"] is None
