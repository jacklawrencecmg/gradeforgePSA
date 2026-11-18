"""Metric arithmetic and gating. Synthetic numbers; they verify formulas, not grading accuracy."""
import numpy as np

from gradeforge import evaluation as ev


def row(grade, ceiling=10, verdict="uncertain", **kw):
    return {"psa_grade": grade, "snapshot": {"grade_ceiling": {"conservative": ceiling}, "verdict": verdict}, **kw}


def test_binary_metrics_precision_recall_fpr():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    pos = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = ev.binary_metrics(y, pos)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["psa10_precision"]["rate"] == round(2 / 3, 4) and m["psa10_recall"]["rate"] == round(2 / 3, 4)
    assert m["false_positive_rate"]["rate"] == 0.2 and m["false_positive_rate"]["n"] == 5
    assert m["psa10_precision"]["ci95"][0] < 2 / 3 < m["psa10_precision"]["ci95"][1]
    assert ev.binary_metrics(np.array([0, 0]), np.array([0, 0]))["psa10_precision"]["rate"] is None      # no positives -> no fake number


def test_grade_accuracy_exact_within_one_and_direction():
    a = ev.grade_accuracy([10, 9, 9, 8, 7], [10, 10, 9, 9, 9])
    assert a["exact"]["k"] == 2 and a["within_one"]["k"] == 4 and a["overpredicted"]["k"] == 3 and a["underpredicted"]["k"] == 0
    assert ev.grade_accuracy([9, 9], [None, None])["n"] == 0
    assert ev.grade_accuracy([9, 10], [9, None])["coverage"] == 0.5


def test_system_grade_never_exceeds_the_rules_ceiling():
    assert ev.system_grade(row(9, ceiling=8), True) == 8          # model positive but a severe defect caps it
    assert ev.system_grade(row(9, ceiling=9), True) == 9
    assert ev.system_grade(row(9, ceiling=10), True) == 10
    assert ev.system_grade(row(9, ceiling=10), False) == 9
    assert ev.system_grade(row(9, ceiling=10), None) is None


def test_platt_calibration_repairs_overconfidence_on_held_out_data():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-z))).astype(float)
    raw = 1 / (1 + np.exp(-3 * z))                               # overconfident (too extreme) scores
    cal = ev.fit_platt(raw[:2000], y[:2000])
    before = ev.calibration_report(y[2000:], raw[2000:])["slope"]
    after = ev.calibration_report(y[2000:], ev.apply_calibration(cal, raw[2000:]))["slope"]
    assert before < 0.5 and 0.85 < after < 1.15 and 0.25 < cal["b"] < 0.45
    assert ev.apply_calibration(None, raw[:3]).tolist() == raw[:3].tolist()


def test_operating_point_uses_given_data_and_reports_its_rule():
    y = np.array([0] * 50 + [1] * 50); p = np.concatenate([np.linspace(0.05, 0.6, 50), np.linspace(0.4, 0.95, 50)])
    op = ev.operating_point(y, p)
    assert 0.3 < op["threshold"] < 0.7 and "recall minus false-positive" in op["rule"]
    hi = ev.operating_point(y, p, target_precision=0.95)
    assert hi["threshold"] >= op["threshold"] and "precision" in hi["rule"]


def test_group_report_suppresses_small_groups_and_never_invents_metrics():
    rows = [row(10 if i % 2 else 9, card_type="a") for i in range(40)] + [row(10, card_type="b") for _ in range(5)]
    p = np.array([0.8 if r["psa_grade"] == 10 else 0.2 for r in rows])
    rep = ev.group_report(rows, p, 0.5, "card_type")
    a = next(g for g in rep["groups"] if g["group"] == "a"); b = next(g for g in rep["groups"] if g["group"] == "b")
    assert a["metrics"] is not None and a["metrics"]["model"]["psa10_precision"]["rate"] == 1.0
    assert b["metrics"] is None and "counts only" in b["suppressed"] and b["n"] == 5


def test_performance_nine_vs_ten_subset_excludes_lower_grades():
    rows = [row(10) for _ in range(10)] + [row(9) for _ in range(10)] + [row(5) for _ in range(10)]
    p = np.array([0.9] * 10 + [0.6] * 10 + [0.1] * 10)
    perf = ev.performance(rows, p, 0.7, baseline_rows=rows)
    nv = perf["nine_vs_ten"]
    assert nv["n"] == 20 and nv["n_psa9"] == 10 and nv["auc"] == 1.0
    assert perf["model"]["false_positive_rate"]["n"] == 20        # all non-10s
    assert nv["false_positive_rate"]["n"] == 10                    # only the 9s


def test_validity_requires_enough_of_each_class_and_the_predeclared_criteria():
    rows = [row(10) for _ in range(5)] + [row(9) for _ in range(30)]
    p = np.array([0.9] * 5 + [0.1] * 30)
    perf = ev.performance(rows, p, 0.5, baseline_rows=rows)
    v = ev.validity_checks(perf)
    assert not v["evaluable"] and "20" in v["reason"]
    rows = [row(10) for _ in range(30)] + [row(9) for _ in range(30)]
    rng = np.random.default_rng(0)
    noise = rng.random(60)
    assert not ev.validity_checks(ev.performance(rows, noise, 0.5, baseline_rows=rows))["passed"]       # random scores fail
    z = rng.normal(0, 1.6, 600)                                    # well-calibrated scores with real signal
    pr = 1 / (1 + np.exp(-z))
    yy = rng.random(600) < pr
    rows = [row(10 if t else 9) for t in yy]
    perf = ev.performance(rows, pr, 0.5, baseline_rows=rows)
    assert ev.validity_checks(perf)["passed"], ev.validity_checks(perf)
