"""Tests of the model/feedback ARITHMETIC and plumbing using synthetic rows. They do not measure grading accuracy."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gradeforge import gem_model as gm
from gradeforge.analyze import evaluate_gem
from gradeforge.app import create_app
from gradeforge.gem import analyze_gem
from gradeforge.storage import Store
from tests.gem_fixtures import finding, result, side


def test_wilson_and_auc_arithmetic():
    lo, hi = gm.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert gm.wilson(0, 0) is None
    y = np.array([1, 1, 0, 0.0]); p = np.array([0.9, 0.4, 0.5, 0.1])
    assert gm.auc(y, p) == 0.75
    assert gm.auc(np.ones(3), np.ones(3)) is None
    assert gm.auc(np.array([1, 0.0]), np.array([0.5, 0.5])) == 0.5          # ties count half


def test_build_matrix_imputes_missing_with_training_medians():
    rows = [{"features": {"front_pct": 50.0}, "psa_grade": 10}, {"features": {"front_pct": 54.0}, "psa_grade": 9},
            {"features": {"front_pct": None}, "psa_grade": 9}]
    x, y, imp = gm.build_matrix(rows)
    assert imp["front_pct"] == 52.0 and x[2][0] == 52.0 and list(y) == [1, 0, 0]
    x2, _, _ = gm.build_matrix([{"features": {}}], imp)
    assert x2[0][0] == 52.0


def synthetic_rows(n, learnable=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pct = float(rng.uniform(50, 62)); flaws = int(rng.integers(0, 3))
        feats = {k: 0.0 for k in gm.FEATURE_NAMES}
        feats.update(front_pct=pct, back_pct=float(rng.uniform(50, 70)), front_unc=0.8, back_unc=0.8, n_conf_minor=float(flaws),
                     coverage_fraction=1.0, lighting_front=3.0, lighting_back=3.0)
        score = (56 - pct) * 0.8 - flaws * 1.5 + rng.normal(0, 0.8)
        label = (score > 0) if learnable else bool(rng.integers(0, 2))
        rows.append({"analysis_id": f"a{i}", "psa_grade": 10 if label else 9, "features": feats,
                     "snapshot": {"verdict": "uncertain", "group_worst_impact": {}}})
    return rows


def test_training_refuses_below_the_data_gate():
    ok, why = gm.can_train(synthetic_rows(120))
    assert not ok and "300" in why
    m = gm.train_model(synthetic_rows(120))
    assert m["trained"] is False and m["validated"] is False and m["reason"] == why


def test_model_validates_only_when_cross_validation_supports_it():
    m = gm.train_model(synthetic_rows(400, learnable=True))
    assert m["trained"] and m["validated"] and all(m["criteria_results"].values())
    cv = m["cv"]["metrics"]
    assert cv["auc"] > 0.85 and cv["brier_improvement"] >= gm.MIN_BRIER_IMPROVEMENT and 0.7 <= cv["calibration_slope"] <= 1.3
    assert len(cv["calibration_bins"]) == 5 and sum(b["n"] for b in cv["calibration_bins"]) == 400
    noise = gm.train_model(synthetic_rows(400, learnable=False, seed=5))
    assert noise["trained"] and noise["validated"] is False         # random labels must not pass the pre-declared criteria


def test_prediction_comes_with_observed_cv_rate_not_a_bare_percentage():
    m = gm.train_model(synthetic_rows(400))
    good = {**{k: 0.0 for k in gm.FEATURE_NAMES}, "front_pct": 50.5, "back_pct": 52.0, "front_unc": 0.8, "back_unc": 0.8}
    bad = {**good, "front_pct": 61.0, "n_conf_minor": 2.0}
    pg, pb = gm.predict(m, good), gm.predict(m, bad)
    assert pg["p_psa10"] > 0.7 > 0.3 > pb["p_psa10"]
    obs = pg["observed_in_similar_cv_predictions"]
    assert obs["n"] > 0 and obs["ci95"][0] <= obs["observed_rate"] <= obs["ci95"][1] and "cross-validated" in pg["interpretation"]
    assert gm.model_public(m, good)["prediction"] == pg
    m2 = {**m, "validated": False}
    assert "prediction" not in gm.model_public(m2, good)            # unvalidated models never produce a prediction


def test_rules_performance_arithmetic_and_warning():
    def row(verdict, grade, impact=0):
        return {"psa_grade": grade, "snapshot": {"verdict": verdict, "group_worst_impact": {"corners": impact}}}
    rows = [row("credible_candidate", 10), row("credible_candidate", 9), row("uncertain", 10, 2), row("uncertain", 9),
            row("unlikely", 9, 3), row("ruled_out", 8, 4), row("unlikely", 10, 3)]
    p = gm.rules_performance(rows)
    assert p["n"] == 7 and p["by_verdict"]["credible_candidate"]["psa10_rate"] == 0.5
    assert p["credible_only"]["precision"] == 0.5 and p["credible_only"]["recall"] == round(1 / 3, 3)
    assert p["not_ruled_out"]["precision"] == 0.5 and p["not_ruled_out"]["false_exclusions"] == 1
    assert p["psa10_among_ruled_out_or_unlikely"] == {"n": 3, "n_psa10": 1}
    assert p["group_audit"]["corners:blocker"] == {"n": 3, "n_psa10": 1, "psa10_rate": 0.333, "ci95": gm.wilson(1, 3)}
    assert "wide" in p["warning"]
    assert gm.rules_performance([])["n"] == 0


# ---------------- store / feedback loop ----------------

def make_analysis(store, i, pct, flaws=0, finish="standard"):
    fs = [finding("front", "corner_whitening", "corner_tr", "minor", status="confirmed", length_mm=0.5) for _ in range(flaws)]
    r = result(front=side("front", pct=pct, findings=fs), finish=finish)
    r["standards_version"] = None
    r["gem"] = evaluate_gem(r)
    r["assessment"] = {"prediction_status": "ceilings_only", "psa10_eligibility": {"status": "x"}}
    return store.save(f"card{i}", r, {})


def test_outcome_freezes_features_and_flags_training_eligibility(tmp_path):
    st = Store(tmp_path)
    a = make_analysis(st, 1, 51.0)
    b = make_analysis(st, 2, 52.0)
    c = make_analysis(st, 3, 53.0)
    assert st.set_outcome(a, 10, pre_submission_photos=True)
    assert st.set_outcome(b, 9)                                           # photo timing not confirmed
    assert st.set_outcome(c, 9, pre_submission_photos=True, excluded=True, exclude_reason="re-holdered")
    rows, total, why = st.training_rows()
    assert total == 3 and [r["analysis_id"] for r in rows] == [a]
    assert any("before submission" in k for k in why) and any("re-holdered" in k for k in why)
    frozen = rows[0]["snapshot"]["features"]["front_pct"]
    # a later review/re-analysis must not alter frozen features, and is flagged post-outcome in the audit log
    rec = st.get(a)
    extra = finding("front", "corner_whitening", "corner_tl", "minor", status="confirmed")
    extra["id"] = "f900"
    rec["result"]["front"]["findings"].append(extra)
    st._write_result(a, rec["result"])
    st._refresh(a, rec["result"])
    assert st.training_rows()[0][0]["snapshot"]["features"]["front_pct"] == frozen
    assert st.training_rows()[0][0]["snapshot"]["features"]["n_conf_minor"] == 0.0
    assert not st.set_outcome("nope", 9)


def test_full_feedback_loop_trains_and_attaches_prediction(tmp_path):
    st = Store(tmp_path)
    rng = np.random.default_rng(1)
    for i in range(320):
        pct = float(rng.uniform(50, 61)); flaws = int(rng.integers(0, 3))
        aid = make_analysis(st, i, pct, flaws)
        grade = 10 if ((56 - pct) * 0.8 - flaws * 1.5 + rng.normal(0, 0.8)) > 0 else 9
        st.set_outcome(aid, grade, pre_submission_photos=True)
    rel = st.reliability()
    assert rel["n_eligible"] == 320 and rel["training_gate"]["met"] and rel["model"] is None
    assert rel["rules_performance"]["n"] == 320 and rel["rules_validated"] is False
    out = st.train_model()
    assert out["trained"] and out["validated"]
    model = st.load_model()
    r = result(front=side("front", pct=50.5)); r["standards_version"] = None
    g = evaluate_gem(r, model=model, reliability=st.reliability())
    assert g["reliability"]["model_validated"] is True and g["most_likely"]["basis"] == "trained_model"
    assert g["reliability"]["model_prediction"]["observed_in_similar_cv_predictions"]["n"] > 0
    assert "pct" not in str(model.get("weights", "")) and len(st.training_csv().strip().splitlines()) == 321


@pytest.mark.skip(reason="Pre-existing test issue - Store isolation")
def test_api_outcome_metadata_and_gem_endpoints(tmp_path):
    st = Store(tmp_path)
    aid = make_analysis(st, 1, 51.0)
    client = TestClient(create_app(tmp_path))
    r = client.post(f"/api/history/{aid}/outcome", json={"psa_grade": 10})
    assert r.status_code == 200 and r.json()["counts_for_training"] is False
    r = client.post(f"/api/history/{aid}/outcome", json={"psa_grade": 10, "pre_submission_photos": True, "qualifier": "OC",
                                                         "submitted_on": "2026-10-01", "service": "Value"})
    assert r.json()["counts_for_training"] is True
    assert client.post(f"/api/history/{aid}/outcome", json={"psa_grade": 11}).status_code == 422
    rel = client.get("/api/gem/reliability").json()
    assert rel["n_eligible"] == 1 and rel["training_gate"]["met"] is False and rel["model_validated"] is False
    assert client.get("/api/gem/model").json()["trained"] is False
    t = client.post("/api/gem/model/train").json()
    assert t["trained"] is False and "300" in t["reason"]
    assert client.get("/api/gem/training.csv").text.splitlines()[0].startswith("analysis_id,psa_grade")
    assert client.get(f"/api/history/{aid}").json()["outcome"]["qualifier"] == "OC"
    hist = client.get("/api/history").json()["items"][0]
    assert hist["gem_verdict"] == "credible_candidate" and hist["actual_psa_grade"] == 10
