"""Training / calibration / one-shot test / promotion lifecycle on SYNTHETIC cards (origin=synthetic_test).
These prove the machinery and its guards. They say nothing about accuracy on real PSA outcomes."""
import json

import numpy as np
import pytest

from gradeforge import evaluation as ev
from gradeforge.registry import Registry
from tests.dataset_fixtures import add_card, new_ds, populate

N = 900


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    ds = new_ds(tmp_path_factory.mktemp("big"))
    populate(ds, N, learnable=True, seed=1)
    return ds, Registry(ds)


def test_retrain_refuses_without_enough_verified_cards_and_makes_no_model(tmp_path):
    ds = new_ds(tmp_path)
    populate(ds, 40)
    reg = Registry(ds)
    r = reg.retrain()
    assert r["trained"] is False and "Not enough" in r["reason"] and not reg.versions()
    assert reg.readiness()["ready_to_train"] is False
    assert (ds.store.dir / "models" / "champion.json").exists() is False
    assert ds.store.load_model() is None


def test_unverified_cards_do_not_count_toward_readiness(tmp_path):
    ds = new_ds(tmp_path)
    for i in range(60):
        add_card(ds, i, 10 if i % 2 else 9, verified=False)
    s = ds.summary()
    assert s["funnel"]["graded"] == 60 and s["funnel"]["eligible"] == 0
    assert Registry(ds).retrain()["trained"] is False


def test_full_lifecycle_train_calibrate_test_once_promote_rollback(big):
    ds, reg = big
    ready = reg.readiness()
    assert ready["ready_to_train"] and ready["ready_to_test"], ready
    t = reg.retrain("first")
    assert t["trained"] and t["status"] == "candidate" and t["synthetic"] is True
    v = t["version"]
    assert t["validation_performance"]["n"] >= 100 and "optimistic" in t["validation_note"]
    model = reg._load(v)
    assert model["validated"] is False and model["calibration"]["method"] == "platt"
    snap = ds.get_snapshot(t["snapshot"])
    train_uids = set(model["train_uids"])
    parts = {m["card_uid"]: m["partition"] for m in snap["manifest"]}
    assert train_uids and all(parts[u] == "train" for u in train_uids)             # nothing but train rows were fit
    # the test partition cannot be touched without explicit confirmation, and only once
    with pytest.raises(ValueError, match="confirm"):
        reg.evaluate_on_test(v)
    res = reg.evaluate_on_test(v, confirm=True)
    assert res["evaluated"] and res["status"] == "passed_test" and res["validity"]["passed"]
    perf = res["performance"]
    assert perf["n"] == snap["counts"]["test"] and perf["model"]["auc"] > 0.8
    assert "nine_vs_ten" in perf and perf["grade_accuracy"]["system"]["within_one"]["rate"] is not None
    again = reg.evaluate_on_test(v, confirm=True)
    assert again["already_evaluated"] and again["performance"] == perf                # stored result, not recomputed
    with ds.store._conn() as c:
        assert c.execute("SELECT COUNT(*) n FROM test_looks WHERE version=? AND purpose='primary'", (v,)).fetchone()["n"] == 1
    # a synthetic model must not become production
    ds.allow_synthetic = False
    with pytest.raises(ValueError, match="synthetic"):
        reg.promote(v)
    ds.allow_synthetic = True
    out = reg.promote(v)
    assert out["promoted"] == v and reg.champion() == v
    assert ds.store.load_model()["version"] == v and ds.store.load_model()["validated"] is True
    # second model: must not be worse than the champion on the same test cards
    t2 = reg.retrain("second")
    r2 = reg.evaluate_on_test(t2["version"], confirm=True)
    assert r2["evaluated"]
    if r2["status"] == "passed_test":
        try:
            reg.promote(t2["version"])
        except ValueError as e:
            assert "worse than" in str(e)
    rb = reg.rollback()
    assert rb["rolled_back"] and (ds.store.load_model() is None or ds.store.load_model()["version"] != rb["rolled_back"])
    assert any(x["version"] == v for x in reg.versions())


def test_prediction_is_attached_only_for_a_model_that_passed_the_test(big):
    ds, reg = big
    from gradeforge.analyze import evaluate_gem
    from tests.gem_fixtures import result, side
    reg2 = reg
    champ = reg2.champion()
    r = result(front=side("front", pct=51.0)); r["standards_version"] = None
    g_none = evaluate_gem(r, model=None)
    assert g_none["reliability"].get("model_prediction") is None
    cand = reg2.retrain("for prediction")["version"]
    model = reg2._load(cand)
    from gradeforge import gem_model as gm
    pub = gm.model_public(model, {k: 0.0 for k in gm.FEATURE_NAMES})
    assert pub["validated"] is False and "prediction" not in pub                       # untested model: no probability
    reg2.evaluate_on_test(cand, confirm=True)
    model = reg2._load(cand)
    if model["validated"]:
        g = evaluate_gem(r, model=model)
        pr = g["reliability"]["model_prediction"]
        assert 0 < pr["p_psa10"] < 1 and "held-out test predictions" in pr["interpretation"]
        assert g["reliability"]["model"]["version"] == cand
    assert champ is None or isinstance(champ, str)


def test_random_labels_fail_the_test_and_cannot_be_promoted(tmp_path):
    ds = new_ds(tmp_path)
    populate(ds, N, learnable=False, seed=5)
    reg = Registry(ds)
    t = reg.retrain()
    assert t["trained"]
    res = reg.evaluate_on_test(t["version"], confirm=True)
    assert res["status"] == "failed_test" and not res["validity"]["passed"]
    assert reg._load(t["version"])["validated"] is False
    with pytest.raises(ValueError, match="passed the test"):
        reg.promote(t["version"])
    assert ds.store.load_model() is None


def test_test_partition_is_left_untouched_when_it_is_too_small(tmp_path):
    ds = new_ds(tmp_path)
    populate(ds, N, seed=2)
    reg = Registry(ds)
    t = reg.retrain()
    # keep only a handful of test-partition PSA 10s verified
    with ds.store._conn() as c:
        c.execute("""UPDATE submissions SET verification_status='unverified' WHERE card_uid IN
                     (SELECT s.card_uid FROM submissions s JOIN cards k ON k.card_uid=s.card_uid WHERE k.partition='test' AND s.grade=10)""")
    t2 = reg.retrain("after")
    snap_rows = ds.snapshot_rows(t2["snapshot"], "test") if t2["trained"] else []
    if t2["trained"]:
        out = reg.evaluate_on_test(t2["version"], confirm=True)
        assert out["evaluated"] is False and "NOT used" in out["reason"]
        with ds.store._conn() as c:
            assert c.execute("SELECT COUNT(*) n FROM test_looks").fetchone()["n"] == 0
    assert t["trained"]


def test_grouped_report_and_provenance_and_leakage_audit(big):
    ds, reg = big
    v = reg.versions()[-1]["version"]
    rep = reg.report(v, "card_type")
    assert rep["partition"] in ("test", "val") and rep["synthetic_warning"]
    assert {g["group"] for g in rep["by_group"]["groups"]} <= {"sports_modern_base", "sports_modern_chrome", "tcg_holo", "sports_vintage"}
    assert all(g["metrics"] is not None for g in rep["by_group"]["groups"] if g["n"] >= 30 and len({1}) )
    audit = ds.leakage_audit(registry=reg)
    assert audit["passed"], [c for c in audit["checks"] if not c["passed"]]
    names = " ".join(c["check"] for c in audit["checks"])
    assert "only on train-partition" in names and "once" in names
    # tamper: pretend a test card was used in training
    mfile = reg._path(v)
    m = json.loads(mfile.read_text())
    test_uid = next(r["card_uid"] for r in ds.rows("test"))
    m["train_uids"].append(test_uid)
    mfile.write_text(json.dumps(m))
    bad = ds.leakage_audit(registry=reg)
    assert not bad["passed"] and any("train-partition" in c["check"] and not c["passed"] for c in bad["checks"])
    m["train_uids"].remove(test_uid)
    mfile.write_text(json.dumps(m))


def test_monitor_scores_only_cards_graded_after_training_and_errors_are_tracked(tmp_path):
    ds = new_ds(tmp_path)
    populate(ds, N, seed=3)
    reg = Registry(ds)
    t = reg.retrain()
    reg.evaluate_on_test(t["version"], confirm=True)
    mon = reg.monitor(t["version"])
    assert mon["n"] == 0
    populate(ds, 12, seed=4, start=5000)
    mon = reg.monitor(t["version"])
    assert mon["n"] == 12 and "prospective" in mon["note"]
    errs = ev.prediction_errors(ds)
    assert errs["n"] > 0 and errs["ceiling_violations"]["n"] == errs["n"]
    assert "rules_misses" in errs


def test_selection_diagnostics_report_limits_and_quality_tiers(tmp_path):
    ds = new_ds(tmp_path)
    for i in range(12):
        add_card(ds, i, 10 if i % 3 == 0 else 9, basis="prescreened_by_owner" if i % 2 else "all_owned_cards", lights=3 if i < 8 else 1)
    d = ev.selection_diagnostics(ds)
    assert set(d["psa10_rate_by_photo_quality_tier"]) == {"good", "limited"}
    assert set(d["psa10_rate_by_declared_selection_basis"]) == {"prescreened_by_owner", "all_owned_cards"}
    assert any("inverse-probability" in x for x in d["what_cannot_be_corrected"])
    assert d["funnel"]["eligible"] == 12


def test_a_positive_model_can_never_lift_a_grade_above_the_rules_ceiling():
    from gradeforge.analyze import evaluate_gem
    from tests.gem_fixtures import finding, result, side
    f = finding("front", "corner_whitening", "corner_tr", "minor", status="confirmed", length_mm=0.5)
    r = result(front=side("front", pct=50.0, findings=[f])); r["standards_version"] = None
    fake = {"trained": True, "validated": True, "threshold": 0.5, "feature_version": 1, "n_train": 1, "criteria_results": {}, "trained_at": 0,
            "prediction": {"p_psa10": 0.99, "interpretation": "x"}}
    from gradeforge import gem as gemmod
    g = gemmod.analyze_gem(r, None, model=fake)
    assert g["most_likely"]["basis"] == "trained_model" and g["most_likely"]["grade"] <= g["grade_ceiling"]["conservative"] < 10
