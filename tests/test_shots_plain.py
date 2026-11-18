from gradeforge import shots
from gradeforge.plain import plain_summary


def test_check_slot_corner_edge_and_wide():
    w, h = 1000, 1400
    assert shots.check_slot("corner_tl", [0, 0, 300, 300], w, h)["ok"]
    assert not shots.check_slot("corner_tl", [700, 1100, 1000, 1400], w, h)["ok"]
    assert shots.check_slot("edge_bottom", [300, 1200, 700, 1400], w, h)["ok"]
    assert not shots.check_slot("edge_bottom", [300, 0, 700, 200], w, h)["ok"]
    assert not shots.check_slot("corner_tl", [0, 0, 950, 1300], w, h)["ok"]  # nearly whole card
    assert shots.check_slot("surface", [400, 400, 600, 600], w, h)["ok"]


def test_upload_path_wording():
    assert shots.upload_path("front", "corner_tl") == "Front → Corner close-ups → Top-left corner"


def test_inventory_missing_when_nothing_uploaded():
    inv = shots.build_inventory({"front": None, "back": None}, {})
    assert inv["totals"]["uploaded"] == 0
    assert inv["sides"]["front"][0]["status"] == "missing"


def _f(i, sev, state_kw):
    return {"id": i, "side": "front", "target": "corner_tl", "category": "corner_whitening", "label": "Corner whitening",
            "severity": sev, **state_kw}


def test_plain_does_not_headline_worst_case():
    gem = {"psa10_eligibility": {"status": "uncertain", "summary": "s"}, "grade_ceiling": {"optimistic": 10, "conservative": 8},
           "characteristics": [], "additional_inspection": [], "most_likely": {}}
    r = plain_summary({"gem": gem, "front": {"ok": True, "findings": []}, "back": None})
    assert r["best_possible"] == 10 and "8" not in r["headline"]
    assert r["n_confirmed"] == 0
