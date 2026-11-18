import cv2
import numpy as np
import pytest

from gradeforge.cv.pipeline import analyze_side
from tests import synth


def side(lr=(60, 60), tb=(60, 60), **kw):
    card, mask = synth.make_card(lr=lr, tb=tb)
    photo_kw = {k: kw.pop(k) for k in list(kw) if k in ("angle", "tilt", "scale", "noise", "blur", "glare_blob")}
    return card, mask, photo_kw


def run(card, mask, **photo_kw):
    photo = synth.photograph(card, mask, **photo_kw)
    res, art = analyze_side(synth.encode(photo), "front")
    return res, (art["rect"] if art else None)


def test_detects_and_rectifies_rotated_card():
    card, mask, kw = side()
    res, rect = run(card, mask, angle=7.0, tilt=0.03)
    assert res["ok"], res.get("reason")
    assert rect.shape[1] / rect.shape[0] == pytest.approx(63.5 / 88.9, rel=0.01)


def test_centering_measured_within_tolerance():
    # left 50 / right 70 -> larger share 58.3 %; top/bottom perfectly centered.
    card, mask, kw = side(lr=(50, 70), tb=(60, 60))
    res, _ = run(card, mask, angle=2.0)
    c = res["centering"]
    assert c["measurable"]
    assert c["worst_axis"] == "left-right"
    assert c["worst_pct"] == pytest.approx(58.33, abs=2.0)
    assert c["tb_share_pct"] == pytest.approx(50.0, abs=2.0)


def test_perfect_centering_reads_near_50():
    card, mask, _ = side()
    res, _ = run(card, mask)
    assert res["centering"]["worst_pct"] == pytest.approx(50.0, abs=2.0)


def test_uncertainty_is_reported_and_positive():
    card, mask, _ = side()
    res, _ = run(card, mask)
    assert res["centering"]["uncertainty_pct"] > 0


def test_borderless_card_is_not_measurable_not_guessed():
    card, mask, _ = side(lr=(2, 2), tb=(2, 2))
    res, _ = run(card, mask)
    assert res["ok"]
    assert res["centering"]["measurable"] is False
    assert all(c["status"] == "not_measurable" for c in res["wear"]["corners"])


def test_clean_card_has_no_whitening_candidates():
    card, mask, _ = side()
    res, _ = run(card, mask)
    assert all(c["status"] == "clean_observed" for c in res["wear"]["corners"]), res["wear"]["corners"]
    assert all(e["status"] == "clean_observed" for e in res["wear"]["edges"]), res["wear"]["edges"]


def test_corner_whitening_is_flagged_on_the_right_corner_only():
    card, mask = synth.make_card()
    synth.whiten(card, mask, (0, 90, 0, 90), depth=10)            # top-left corner region
    res, _ = run(card, mask)
    status = {c["id"]: c["status"] for c in res["wear"]["corners"]}
    assert status["tl"] in ("minor_candidate", "visible_candidate"), status
    assert status["br"] == "clean_observed"


def test_edge_whitening_is_flagged_on_the_right_edge():
    card, mask = synth.make_card()
    synth.whiten(card, mask, (0, 14, 300, 700), depth=10)         # 400 px of top edge
    res, _ = run(card, mask)
    status = {e["id"]: e["status"] for e in res["wear"]["edges"]}
    assert status["top"] in ("minor_candidate", "visible_candidate"), status
    assert status["bottom"] == "clean_observed"


def test_light_border_whitening_is_declared_not_measurable():
    card, mask = synth.make_card(border=(235, 235, 235))
    res, _ = run(card, mask)
    assert res["wear"]["border_is_light"]
    assert all(e["status"] == "not_measurable" for e in res["wear"]["edges"])


def test_rounded_corner_is_flagged_against_other_corners():
    # Top-left corner die-cut (or worn) to radius 110 px vs 50 px elsewhere.
    card, mask2 = synth.make_card(radii=(110, 50, 50, 50))
    res, _ = run(card, mask2)
    tl = [f for f in res["findings"] if f["target"] == "corner_tl" and f["category"] in ("corner_rounding", "corner_missing_stock")]
    assert tl, res["findings"]
    assert tl[0]["measurements"]["radius_mm"] > 2 * tl[0]["measurements"]["median_other_radius_mm"]
    assert not [f for f in res["findings"] if f["target"] == "corner_br"]


def test_ring_scratch_found_as_linear_candidate():
    card, mask = synth.make_card(lr=(80, 80), tb=(80, 80))
    synth.draw_scratch(card, (40, 500), (40, 620), value=(235, 235, 235), thickness=2)
    res, _ = run(card, mask)
    ring = res["surface"]["ring"]
    assert ring["assessed"]
    lin = [f for f in res["findings"] if f["category"] == "surface_linear_mark"]
    assert lin, res["findings"]
    # one photo only: must never be called physical damage
    assert all(f["classification"] == "indeterminate_single_image" for f in lin)
    assert all(f["status"] == "candidate" for f in lin)


def test_clean_ring_has_no_candidates():
    card, mask = synth.make_card(lr=(80, 80), tb=(80, 80))
    res, _ = run(card, mask)
    assert [f for f in res["findings"] if f["category"].startswith("surface_")] == []


def test_glare_blob_is_reported():
    card, mask = synth.make_card()
    res, _ = run(card, mask, glare_blob=(800, 1000, 120, 90))
    assert len(res["surface"]["glare"]["regions"]) >= 1


def test_low_resolution_photo_is_flagged_insufficient():
    card, mask = synth.make_card()
    photo = synth.photograph(card, mask, photo_size=(1600, 2100), scale=1.1)
    small = cv2.resize(photo, (500, 656), interpolation=cv2.INTER_AREA)
    res, _ = analyze_side(synth.encode(small), "front")
    assert res["ok"]
    assert res["quality"]["resolution_class"] == "insufficient"
    assert res["quality"]["requests"]


def test_not_an_image_fails_cleanly():
    res, rect = analyze_side(b"this is not an image", "front")
    assert not res["ok"] and rect is None
