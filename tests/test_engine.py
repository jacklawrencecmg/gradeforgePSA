import pytest

from gradeforge.analyze import analyze_card
from gradeforge.standards import available_versions, load_standards
from tests import synth


def photo_bytes(**card_kw):
    photo_kw = {k: card_kw.pop(k) for k in list(card_kw) if k in ("angle", "scale", "photo_size")}
    card, mask = synth.make_card(**card_kw)
    return synth.encode(synth.photograph(card, mask, **photo_kw))


def test_standards_database_is_versioned_and_honest_about_verification():
    std = load_standards()
    assert std.version in available_versions()
    assert std.front_limit(10) == 55 and std.back_limit(10) == 75
    assert std.front_limit(9) == 60 and std.back_limit(9) == 90
    assert std.official_page_verified is False          # the official page could not be read
    assert std.meta()["known_conflicts"]
    with pytest.raises(ValueError):
        load_standards("9.9.9")


def test_front_only_never_forces_a_prediction():
    res, _ = analyze_card(photo_bytes(), None)
    a = res["assessment"]
    assert a["prediction_status"] == "insufficient_evidence"
    assert a["most_likely"] is None and a["conservative_ceiling"] is None
    assert any("back" in r.lower() for r in a["requests"])


def test_low_resolution_abstains_and_asks_for_better_images():
    import cv2
    card, mask = synth.make_card()
    photo = synth.photograph(card, mask)
    small = synth.encode(cv2.resize(photo, (500, 656), interpolation=cv2.INTER_AREA))
    res, _ = analyze_card(small, small)
    a = res["assessment"]
    assert a["prediction_status"] == "insufficient_evidence"
    assert a["psa10_eligibility"]["status"] == "indeterminate"
    assert any("px wide" in r for r in a["requests"])


def test_clean_well_centered_card_gets_ceilings_not_a_point_estimate():
    res, rects = analyze_card(photo_bytes(), photo_bytes(seed=5))
    a = res["assessment"]
    assert a["prediction_status"] == "ceilings_only"
    assert a["optimistic_ceiling"] == 10
    assert a["conservative_ceiling"] == 9              # a 10 cannot be confirmed: criteria remain unassessed
    assert a["most_likely"] is None
    assert a["probabilities_reported"] is False
    assert a["psa10_eligibility"]["status"] == "not_ruled_out"
    assert a["psa10_eligibility"]["unassessed"]
    assert a["evidence_level"] in ("low", "moderate")  # never 'high' in v0.1
    assert {"front", "back"} <= {k for k, v in rects.items() if v is not None}  # artifacts per side


def test_bad_front_centering_rules_out_psa10():
    # left 36 / right 84 -> larger share 70 %.
    res, _ = analyze_card(photo_bytes(lr=(36, 84)), photo_bytes())
    a = res["assessment"]
    assert a["psa10_eligibility"]["status"] == "ruled_out"
    assert a["optimistic_ceiling"] <= 8
    assert any(g["id"] == "centering" for g in a["grade_limiting"])


def test_between_nominal_and_leeway_is_unlikely_not_excluded():
    # 58.3 % front: outside 55/45, inside the 5 % leeway.
    res, _ = analyze_card(photo_bytes(lr=(50, 70)), photo_bytes())
    a = res["assessment"]
    assert a["psa10_eligibility"]["status"] == "unlikely"
    assert a["optimistic_ceiling"] == 10 and a["conservative_ceiling"] <= 9
    item = next(c for c in a["checklist"] if c["id"] == "front_centering")
    assert item["status"] == "fail_nominal_leeway_possible"


def test_excellent_centering_does_not_imply_psa10():
    res, _ = analyze_card(photo_bytes(), photo_bytes())
    a = res["assessment"]
    assert a["psa10_eligibility"]["status"] != "eligible"
    assert a["conservative_ceiling"] < 10


def test_defect_on_back_corner_caps_grade():
    card, mask = synth.make_card()
    synth.whiten(card, mask, (1310, 1400, 910, 1000), depth=12)       # bottom-right corner region
    back = synth.encode(synth.photograph(card, mask))
    res, _ = analyze_card(photo_bytes(), back)
    a = res["assessment"]
    corner = next(c for c in a["checklist"] if c["id"] == "back_corner_br")
    assert corner["status"] in ("borderline", "fail_candidate")
    assert a["conservative_ceiling"] <= 9


def test_checklist_marks_unassessed_items_explicitly():
    res, _ = analyze_card(photo_bytes(), photo_bytes())
    ids = {c["id"] for c in res["assessment"]["checklist"] if c["status"] == "unknown_not_assessed"}
    assert {"front_artwork_print_and_stain", "back_artwork_print_and_stain", "front_artwork_relief", "back_artwork_relief", "authenticity"} <= ids
