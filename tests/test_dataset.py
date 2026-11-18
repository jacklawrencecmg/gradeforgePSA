"""Dataset layer: provenance, partitions, leakage controls, annotation. Synthetic cards; plumbing only."""
import pytest

from gradeforge.dataset import Dataset, POLICY
from gradeforge.storage import Store
from tests.dataset_fixtures import add_card, make_analysis, new_ds, populate
from tests.gem_fixtures import finding


def test_cards_require_rights_attestation_valid_types_and_refuse_synthetic_by_default(tmp_path):
    real = Dataset(Store(tmp_path), allow_synthetic=False)
    with pytest.raises(ValueError, match="rights_attested"):
        real.create_card(card_type="tcg_holo")
    with pytest.raises(ValueError, match="card_type"):
        real.create_card(card_type="mystery", rights_attested=True)
    with pytest.raises(ValueError, match="Synthetic"):
        real.create_card(card_type="tcg_holo", rights_attested=True, origin="synthetic_test")
    c = real.create_card(card_type="tcg_holo", rights_attested=True)
    assert c["origin"] == "real_physical_card" and c["partition"] in ("train", "val", "test") and c["status"] == "registered"


def test_partition_is_deterministic_grouped_and_roughly_70_15_15(tmp_path):
    ds = new_ds(tmp_path)
    a = ds.create_card(card_type="sports_vintage", rights_attested=True, set_name="1952 Topps", year=1952, card_number="311")
    b = ds.create_card(card_type="sports_vintage", rights_attested=True, set_name="1952  topps", year=1952, card_number="311")
    assert a["partition"] == b["partition"] and a["group_key"] == b["group_key"]       # copies of one issue never straddle partitions
    assert ds.partition_for(a["group_key"]) == a["partition"]
    cards = [ds.create_card(card_type="other", rights_attested=True) for _ in range(600)]
    share = {p: sum(c["partition"] == p for c in cards) / 600 for p in ("train", "val", "test")}
    assert 0.62 < share["train"] < 0.78 and 0.08 < share["val"] < 0.22 and 0.08 < share["test"] < 0.22
    other = Dataset(Store(tmp_path / "other"), allow_synthetic=True)                      # different salt, different store
    assert other._salt() != ds._salt()


def test_image_phase_is_explicit_slab_photos_never_count_and_late_photos_are_refused(tmp_path):
    ds = new_ds(tmp_path)
    c = ds.create_card(card_type="other", rights_attested=True)["card_uid"]
    a1, _ = make_analysis(ds.store, 1, 51)
    a2, _ = make_analysis(ds.store, 2, 51)
    with pytest.raises(ValueError, match="phase"):
        ds.link_image(c, a1, "")
    with pytest.raises(ValueError, match="captured_on"):
        ds.link_image(c, a1, "pre_grading")
    ds.link_image(c, a2, "post_grading_slab")
    with pytest.raises(ValueError, match="pre-grading"):
        ds.record_result(c, 10)                                          # slab photo alone is not enough
    ds.link_image(c, a1, "pre_grading", "2026-01-01")
    ds.record_submission(c, "2026-02-01")
    ds.record_result(c, 9, cert="12345678")
    a3, _ = make_analysis(ds.store, 3, 51)
    with pytest.raises(ValueError, match="already recorded"):
        ds.link_image(c, a3, "pre_grading", "2026-01-02")                # photos after the grade cannot be 'pre-grading'
    card = ds.get_card(c)
    assert card["frozen"]["analysis_id"] == a1                           # the slab analysis is never the primary


def test_same_analysis_or_same_photo_files_cannot_join_two_cards(tmp_path):
    ds = new_ds(tmp_path)
    c1, c2 = (ds.create_card(card_type="other", rights_attested=True)["card_uid"] for _ in range(2))
    a, _ = make_analysis(ds.store, 1, 51)
    b, _ = make_analysis(ds.store, 2, 51)
    for aid in (a, b):                                                    # same bytes uploaded twice
        d = ds.store.dir / "originals" / aid
        d.mkdir(parents=True, exist_ok=True)
        (d / "front_primary.bin").write_bytes(b"identical photo bytes")
    ds.link_image(c1, a, "pre_grading", "2026-01-01")
    with pytest.raises(ValueError, match="already linked"):
        ds.link_image(c2, a, "pre_grading", "2026-01-01")
    with pytest.raises(ValueError, match="same photo"):
        ds.link_image(c2, b, "pre_grading", "2026-01-01")


def test_result_cert_validation_duplicates_freezing_and_audited_correction(tmp_path):
    ds = new_ds(tmp_path)
    u1, a1 = add_card(ds, 1, 10, verified=False)
    c2 = ds.create_card(card_type="other", rights_attested=True)["card_uid"]
    a2, _ = make_analysis(ds.store, 2, 51)
    ds.link_image(c2, a2, "pre_grading", "2026-01-01")
    with pytest.raises(ValueError, match="digits"):
        ds.record_result(c2, 9, cert="abc")
    with pytest.raises(ValueError, match="already recorded for card"):
        ds.record_result(c2, 9, cert="50000001")                         # cert of card 1
    with pytest.raises(ValueError, match="1-10"):
        ds.record_result(c2, 11)
    with pytest.raises(ValueError, match="already recorded"):
        ds.record_result(u1, 9)                                           # no silent overwrite
    ds.correct_result(u1, grade=9, reason="PSA cert page shows 9")
    log = [e["action"] for e in ds.audit(u1)]
    assert "result_corrected" in log
    assert ds.get_card(u1)["submission"]["verification_status"] == "unverified"       # a correction needs re-verification
    with pytest.raises(ValueError, match="reason"):
        ds.correct_result(u1, grade=10, reason=" ")


def test_only_verified_pre_submission_graded_cards_are_eligible(tmp_path):
    ds = new_ds(tmp_path)
    ok, _ = add_card(ds, 1, 10)
    unver, _ = add_card(ds, 2, 9, verified=False)
    late, _ = add_card(ds, 3, 9, captured="2026-03-01", submitted="2026-02-01")
    conflict, _ = add_card(ds, 4, 9, verified=False)
    ds.verify_result(conflict, verified_by="t", cert_number_seen="50000004", cert_page_grade=10)   # page says 10, recorded 9
    excl, _ = add_card(ds, 5, 9)
    ds.exclude_card(excl, "re-holdered before grading")
    nogr = ds.create_card(card_type="other", rights_attested=True)["card_uid"]
    got = {r["card_uid"] for r in ds.rows()}
    assert got == {ok}
    why = lambda u: " ".join(ds.get_card(u)["ineligible_reasons"])
    assert "not verified" in why(unver) and "not captured on or before" in why(late) and "conflicts" in why(conflict)
    assert "excluded" in why(excl) and "no PSA result" in why(nogr)
    assert ds.get_card(conflict)["submission"]["verification_status"] == "conflict"
    s = ds.summary()
    assert s["funnel"]["eligible"] == 1 and s["funnel"]["graded"] == 5 and s["funnel"]["verified"] == 3


def test_dataset_snapshot_is_content_addressed_and_frozen(tmp_path):
    ds = new_ds(tmp_path)
    populate(ds, 12)
    s1 = ds.create_snapshot("first")
    assert ds.create_snapshot("again")["id"] == s1["id"]                  # unchanged data -> same version
    add_card(ds, 99, 10)
    s2 = ds.create_snapshot("more")
    assert s2["id"] != s1["id"] and s2["n"] == s1["n"] + 1
    assert len(ds.snapshot_rows(s1["id"])) == s1["n"]


def test_leakage_audit_passes_clean_data_and_detects_tampering(tmp_path):
    ds = new_ds(tmp_path)
    uids = populate(ds, 10)
    audit = ds.leakage_audit()
    assert audit["passed"], [c for c in audit["checks"] if not c["passed"]]
    assert audit["not_checkable"]
    other = "val" if ds.get_card(uids[0])["partition"] != "val" else "test"
    with ds.store._conn() as c:                                          # someone hand-edits a partition
        c.execute("UPDATE cards SET partition=? WHERE card_uid=?", (other, uids[0]))
    bad = ds.leakage_audit()
    assert not bad["passed"] and any("never edited" in c["check"] for c in bad["checks"] if not c["passed"])
    with ds.store._conn() as c:
        c.execute("UPDATE cards SET partition=? WHERE card_uid=?", (ds.partition_for(ds.get_card(uids[0])["group_key"]), uids[0]))
        aid = ds.get_card(uids[1])["frozen"]["analysis_id"]
        c.execute("UPDATE gem_snapshots SET snapshot_json=snapshot_json || ' ' WHERE analysis_id=?", (aid,))
    again = ds.leakage_audit()
    assert not again["passed"] and any("unchanged since" in c["check"] and not c["passed"] for c in again["checks"])


def test_post_result_reviews_do_not_change_frozen_features_and_are_flagged(tmp_path):
    ds = new_ds(tmp_path)
    uid, aid = add_card(ds, 1, 10)
    before = ds.rows()[0]["features"]["n_conf_minor"]
    rec = ds.store.get(aid)
    f = finding("front", "corner_whitening", "corner_tl", "minor", status="confirmed")
    f["id"] = "f900"
    rec["result"]["front"]["findings"].append(f)
    ds.store._write_result(aid, rec["result"])
    ds.store._refresh(aid, rec["result"])
    assert ds.rows()[0]["features"]["n_conf_minor"] == before
    assert ds.leakage_audit()["passed"]


def test_annotations_blindness_and_detector_evaluation(tmp_path):
    ds = new_ds(tmp_path)
    c = ds.create_card(card_type="other", rights_attested=True)["card_uid"]
    f1 = finding("front", "corner_whitening", "corner_tr", "minor")
    f1["box"] = [100, 100, 40, 40]
    f2 = finding("front", "surface_spot", "x", "minor")
    f2["box"] = [600, 600, 30, 30]
    from gradeforge.analyze import evaluate_gem
    from tests.gem_fixtures import result, side
    r = result(front=side("front", findings=[f1, f2])); r["standards_version"] = None
    r["gem"] = evaluate_gem(r); r["assessment"] = {"prediction_status": "ceilings_only", "psa10_eligibility": {"status": "x"}}
    aid = ds.store.save("a", r, {})
    ds.link_image(c, aid, "pre_grading", "2026-01-01")
    with pytest.raises(ValueError, match="annotator"):
        ds.add_annotation(c, aid, "front", "corner_whitening", [95, 95, 50, 50], annotator="")
    a1 = ds.add_annotation(c, aid, "front", "corner_whitening", [95, 98, 45, 45], annotator="jl")
    a2 = ds.add_annotation(c, aid, "front", "hairline_or_scratch", [300, 300, 60, 6], annotator="jl")
    assert a1["blind"] and a2["blind"]
    ds.set_annotation_complete(aid, "front", "jl")
    ev = ds.evaluate_detectors()["by_kind"]
    assert ev["corner_whitening"]["recall"] == 1.0 and ev["hairline_or_scratch"]["recall"] == 0.0     # the scratch was missed
    assert ev["corner_whitening"]["precision_on_fully_annotated_sides"] == 1.0
    assert ev["stain_or_spot"]["precision_on_fully_annotated_sides"] == 0.0                           # detector spot nobody marked
    ds.record_submission(c, "2026-02-01"); ds.record_result(c, 9, cert="87654321")
    late = ds.add_annotation(c, aid, "front", "dent_or_indentation", [10, 10, 20, 20], annotator="jl")
    assert not late["blind"] and "NOT blind" in late["warning"]
    assert ds.evaluate_detectors()["n_not_blind_excluded"] == 1


def test_policy_is_declared_in_code():
    assert POLICY["require_verified_grade"] and POLICY["require_pre_grading_images"]
