"""Measured performance: everything here is computed from recorded outcomes, never assumed.

Conventions
* Rows are dicts from Dataset.rows(): 'psa_grade', 'snapshot' (frozen), 'partition', 'card_type', ...
* A "positive" is a prediction of PSA 10. FPR = predicted 10 among cards that did NOT receive a 10.
* Every proportion carries a Wilson 95% interval; small samples are labeled, and groups below the policy minimum
  show counts only.
"""
from __future__ import annotations

import math

import numpy as np

from . import gem_model as gm
from .dataset import POLICY
from .validation import fit_logistic


def _prop(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": None if n == 0 else round(k / n, 4), "ci95": gm.wilson(int(k), int(n))}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------- calibration
def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> dict:
    """Logistic recalibration p' = sigmoid(a + b*logit(p)). Fit on the VALIDATION partition only."""
    w = fit_logistic(logit(p_raw)[:, None], y.astype(float), l2=1e-3)
    return {"method": "platt", "a": float(w[0]), "b": float(w[1]), "n_fit": int(len(y))}


def apply_calibration(cal: dict | None, p_raw):
    p_raw = np.asarray(p_raw, float)
    if not cal:
        return p_raw
    z = cal["a"] + cal["b"] * logit(p_raw)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_report(y: np.ndarray, p: np.ndarray, bins: int = 5) -> dict:
    if len(y) == 0:
        return {"n": 0}
    b = gm.calibration_bins(y, p, bins)
    ece = sum(x["n"] * abs(x["observed_rate"] - x["mean_predicted"]) for x in b) / len(y)
    z = logit(p)[:, None]
    w = fit_logistic(z, y.astype(float), l2=1e-6)
    rng = np.random.default_rng(0)
    slopes = []
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(y):
            slopes.append(float(fit_logistic(z[idx], y[idx].astype(float), l2=1e-6)[1]))
    ci = [round(float(np.percentile(slopes, 2.5)), 3), round(float(np.percentile(slopes, 97.5)), 3)] if len(slopes) > 20 else None
    return {"n": int(len(y)), "slope": round(float(w[1]), 3), "slope_ci95": ci, "intercept_at_slope1": round(float(np.mean(y) - np.mean(p)), 4),
            "ece": round(float(ece), 4), "bins": b,
            "reading": "slope near 1 and small ECE mean stated probabilities match observed PSA 10 rates; "
                       "with few cards per bin these are rough."}


# ------------------------------------------------------------------------------------- binary metrics
def binary_metrics(y: np.ndarray, pos: np.ndarray) -> dict:
    y, pos = np.asarray(y).astype(bool), np.asarray(pos).astype(bool)
    tp, fp = int((pos & y).sum()), int((pos & ~y).sum())
    fn, tn = int((~pos & y).sum()), int((~pos & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "psa10_precision": _prop(tp, tp + fp), "psa10_recall": _prop(tp, tp + fn),
            "false_positive_rate": _prop(fp, fp + tn), "specificity": _prop(tn, fp + tn),
            "n_predicted_psa10": tp + fp,
            "accuracy_10_vs_not": _prop(tp + tn, len(y)),
            "base_rate_psa10": _prop(int(y.sum()), len(y))}


def operating_point(y: np.ndarray, p: np.ndarray, target_precision: float | None = None) -> dict:
    """Decision threshold chosen on VALIDATION data. Default: maximize Youden's J (recall - FPR). With target_precision:
    lowest threshold reaching that precision with at least 5 positives; falls back to Youden's J if none does."""
    ths = np.unique(np.round(p, 4))
    best_j, best_t = -1.0, 0.5
    prec_t = None
    for t in ths:
        pr = p >= t
        tpr = (pr & (y == 1)).sum() / max(1, (y == 1).sum())
        fpr = (pr & (y == 0)).sum() / max(1, (y == 0).sum())
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
        if target_precision is not None and prec_t is None and pr.sum() >= 5 and (pr & (y == 1)).sum() / pr.sum() >= target_precision:
            prec_t = float(t)
    return {"threshold": prec_t if prec_t is not None else best_t,
            "rule": (f"lowest threshold with validation precision >= {target_precision}" if prec_t is not None
                     else "maximum of recall minus false-positive rate on the validation partition"),
            "target_precision": target_precision, "target_met": prec_t is not None if target_precision is not None else None}


# ------------------------------------------------------------------------------------- grade accuracy
def system_grade(row: dict, positive: bool | None) -> int | None:
    """Predicted grade for exact / within-one scoring (a DERIVED assumption, not something PSA defines):
    never above the conservative rules ceiling; 10 only if the model is positive AND no ceiling is below 10;
    otherwise 9, or the ceiling when it is lower. None when there is no model decision and no ceiling below 10."""
    ceil = (row["snapshot"].get("grade_ceiling") or {}).get("conservative")
    if ceil is None:
        return None
    if ceil < 10:
        return int(min(9, ceil))
    if positive is None:
        return None
    return 10 if positive else 9


def grade_accuracy(actual: list[int], predicted: list[int | None]) -> dict:
    pairs = [(a, p) for a, p in zip(actual, predicted) if p is not None]
    n = len(pairs)
    if not n:
        return {"n": 0, "coverage": 0.0}
    diff = [p - a for a, p in pairs]
    return {"n": n, "coverage": round(n / max(1, len(actual)), 3), "exact": _prop(sum(d == 0 for d in diff), n),
            "within_one": _prop(sum(abs(d) <= 1 for d in diff), n),
            "mean_error": round(float(np.mean(diff)), 3), "mean_abs_error": round(float(np.mean(np.abs(diff))), 3),
            "overpredicted": _prop(sum(d > 0 for d in diff), n), "underpredicted": _prop(sum(d < 0 for d in diff), n)}


def majority_grade(rows: list[dict]) -> int | None:
    if not rows:
        return None
    vals = [r["psa_grade"] for r in rows]
    return max(set(vals), key=vals.count)


# ------------------------------------------------------------------------------------- full performance block
def performance(rows: list[dict], p: np.ndarray | None, threshold: float | None, *, baseline_rows: list[dict] | None = None,
                p_uncalibrated: np.ndarray | None = None) -> dict:
    """rows: evaluation rows; p: calibrated P(PSA 10) per row (None = rules only);
    baseline_rows: rows used to pick the majority-grade baseline (the TRAINING rows, never the evaluation rows)."""
    n = len(rows)
    out: dict = {"n": n}
    if n == 0:
        return {**out, "message": "No eligible cards in this set."}
    y = np.array([1 if r["psa_grade"] == 10 else 0 for r in rows])
    out["class_balance"] = {"psa10": int(y.sum()), "not_psa10": int(n - y.sum())}
    actual = [r["psa_grade"] for r in rows]
    maj = majority_grade(baseline_rows or [])
    out["grade_accuracy"] = {
        "majority_grade_baseline": {"predicted_grade": maj, **grade_accuracy(actual, [maj] * n)} if maj is not None else None,
        "conservative_ceiling_baseline": grade_accuracy(actual, [r["snapshot"]["grade_ceiling"]["conservative"] for r in rows]),
        "always_9_baseline": grade_accuracy(actual, [9] * n),
        "note": "'system' = never above the rules ceiling; 10 only when the model is positive and no ceiling is below 10; else 9 or the "
                "lower ceiling. It is a derived rule, scored beside simple baselines so you can see whether it adds anything."}
    short = np.array([r["snapshot"]["verdict"] in ("credible_candidate", "uncertain") for r in rows])
    out["rules_shortlist"] = {**binary_metrics(y, short), "definition": "positive = rules verdict credible_candidate or uncertain"}
    if p is not None and threshold is not None:
        pos = p >= threshold
        out["model"] = {"threshold": threshold, **binary_metrics(y, pos), "auc": None, "brier": None}
        a = gm.auc(y.astype(float), p)
        lo, hi = gm._bootstrap_auc_lower(y.astype(float), p) if len(set(y)) == 2 else (None, None)
        base = float(np.mean((y.mean() - y) ** 2))
        brier = float(np.mean((p - y) ** 2))
        out["model"].update({"auc": None if a is None else round(a, 4), "auc_ci95": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
                             "brier": round(brier, 4), "baseline_brier": round(base, 4), "brier_improvement": round(base - brier, 4)})
        out["calibration"] = calibration_report(y.astype(float), p)
        if p_uncalibrated is not None:
            out["calibration_before"] = calibration_report(y.astype(float), np.asarray(p_uncalibrated, float))
        out["grade_accuracy"]["system"] = grade_accuracy(actual, [system_grade(r, bool(pp)) for r, pp in zip(rows, pos)])
        # PSA 9 vs PSA 10 only (the hard, valuable question): negatives are 9s, easy low grades removed
        sub = [i for i, r in enumerate(rows) if r["psa_grade"] in (9, 10)]
        if sub:
            ys, ps = y[sub].astype(float), p[sub]
            m = binary_metrics(ys, ps >= threshold)
            a9 = gm.auc(ys, ps)
            out["nine_vs_ten"] = {"n": len(sub), "n_psa10": int(ys.sum()), "n_psa9": int(len(sub) - ys.sum()),
                                  "auc": None if a9 is None else round(a9, 4), **{k: m[k] for k in ("psa10_precision", "psa10_recall", "false_positive_rate", "tp", "fp", "fn", "tn")},
                                  "note": "Only cards that actually received a 9 or a 10. False positive = predicted 10, received 9."}
    return out


def validity_checks(perf: dict, n_min_per_class: int = 20) -> dict:
    """Pre-declared pass/fail criteria, evaluated on the TEST partition by the registry."""
    m = perf.get("model")
    if not m:
        return {"evaluable": False, "passed": False, "reason": "no model predictions"}
    cb = perf["class_balance"]
    if min(cb.values()) < n_min_per_class:
        return {"evaluable": False, "passed": False,
                "reason": f"test partition has {cb['psa10']} PSA 10s and {cb['not_psa10']} others; each class needs at least {n_min_per_class}"}
    lo = m["auc_ci95"][0]
    cal = perf["calibration"]
    sl = cal.get("slope_ci95")
    lo_s, hi_s = gm.SLOPE_RANGE
    # With a few hundred held-out cards the slope is imprecise, so a point-estimate band rejects sound models by chance.
    # Pre-declared rule: the model fails only if the 95% interval for the slope lies ENTIRELY outside the tolerated band.
    checks = {"brier_beats_base_rate_by_margin": m["brier_improvement"] >= gm.MIN_BRIER_IMPROVEMENT,
              "auc_lower_bound_ok": lo is not None and lo >= gm.MIN_AUC_LOWER_BOUND,
              "calibration_not_demonstrably_off": sl is not None and sl[1] >= lo_s and sl[0] <= hi_s}
    return {"evaluable": True, "passed": all(checks.values()), "checks": checks,
            "slope": cal["slope"], "slope_ci95": sl,
            "small_sample_note": "Passing the calibration check means the slope interval overlaps the tolerated band, i.e. the probabilities are not "
                                 "demonstrably miscalibrated. It does not prove they are exact: read the reliability bins, which show observed "
                                 "PSA 10 rates with their intervals.",
            "criteria": {"min_brier_improvement": gm.MIN_BRIER_IMPROVEMENT, "min_auc_lower_bound": gm.MIN_AUC_LOWER_BOUND,
                         "calibration_slope_range": list(gm.SLOPE_RANGE)}}


# ------------------------------------------------------------------------------------- grouped reports
GROUP_KEYS = ("card_type", "finish", "quality_tier", "selection_basis", "decade")


def _group_value(r: dict, key: str) -> str:
    if key == "decade":
        return "unknown" if not r.get("year") else f"{int(r['year']) // 10 * 10}s"
    return str(r.get(key) or "unknown")


def group_report(rows: list[dict], p: np.ndarray | None, threshold: float | None, key: str, min_n: int | None = None) -> dict:
    if key not in GROUP_KEYS:
        raise ValueError(f"group_by must be one of {GROUP_KEYS}")
    min_n = min_n or POLICY["min_group_n_for_metrics"]
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(_group_value(r, key), []).append(i)
    out = []
    for g, idx in sorted(groups.items()):
        sub = [rows[i] for i in idx]
        y = np.array([1 if r["psa_grade"] == 10 else 0 for r in sub])
        entry = {"group": g, "n": len(sub), "psa10": _prop(int(y.sum()), len(sub)),
                 "grades": {str(k): int(sum(1 for r in sub if r["psa_grade"] == k)) for k in sorted({r["psa_grade"] for r in sub})}}
        if len(sub) < min_n or len(set(y)) < 2:
            entry["metrics"] = None
            entry["suppressed"] = (f"Fewer than {min_n} cards" if len(sub) < min_n else "Only one outcome class present") + ": counts only, no accuracy claims."
        else:
            perf = performance(sub, None if p is None else p[idx], threshold)
            m = perf.get("model")
            entry["metrics"] = {"rules_shortlist_precision": perf["rules_shortlist"]["psa10_precision"],
                                "rules_shortlist_recall": perf["rules_shortlist"]["psa10_recall"],
                                "model": None if not m else {k: m[k] for k in ("psa10_precision", "psa10_recall", "false_positive_rate", "auc", "brier", "brier_improvement")},
                                "grade_exact_within_one_ceiling_baseline": {k: perf["grade_accuracy"]["conservative_ceiling_baseline"].get(k)
                                                                           for k in ("exact", "within_one")}}
        out.append(entry)
    return {"group_by": key, "min_n": min_n, "groups": out,
            "warning": "Groups are small subsets: intervals are wide, and many comparisons across groups will turn up chance differences."}


# ------------------------------------------------------------------------------------- selection and photo-quality bias
def _smd(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return None
    sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return None if sd == 0 else round(float((a.mean() - b.mean()) / sd), 3)


def selection_diagnostics(ds) -> dict:
    """What the graded dataset can and cannot tell you. No correction is attempted that the data cannot support."""
    cards = ds.list_cards()
    funnel = ds.summary()["funnel"]
    # photographed-but-never-submitted cards vs graded cards: do they look different on the measured features?
    graded_rows, unsub = ds.rows(include_ineligible=True), []
    with ds.store._conn() as c:
        for cd in c.execute("""SELECT c.card_uid FROM cards c LEFT JOIN submissions s ON s.card_uid=c.card_uid
                               WHERE s.grade IS NULL""").fetchall():
            for im in c.execute("SELECT analysis_id FROM card_images WHERE card_uid=? AND phase='pre_grading' ORDER BY linked DESC LIMIT 1",
                                (cd["card_uid"],)).fetchall():
                unsub.append(im["analysis_id"])
    feats_unsub = []
    for aid in unsub:
        rec = ds.store.get(aid)
        if rec and rec["result"].get("gem"):
            feats_unsub.append(gm.extract_features(rec["result"]["gem"], rec["result"]))
    smd = {}
    if len(feats_unsub) >= 5 and len(graded_rows) >= 5:
        for n in gm.FEATURE_NAMES:
            a = np.array([np.nan if r["features"].get(n) is None else r["features"][n] for r in graded_rows], float)
            b = np.array([np.nan if f.get(n) is None else f[n] for f in feats_unsub], float)
            v = _smd(a, b)
            if v is not None and abs(v) >= 0.3:
                smd[n] = v
    # grade distribution by declared selection basis and photo-quality tier
    by_basis, by_tier = {}, {}
    for r in graded_rows:
        for d, k in ((by_basis, r["selection_basis"]), (by_tier, r["quality_tier"])):
            e = d.setdefault(k, {"n": 0, "psa10": 0})
            e["n"] += 1
            e["psa10"] += r["psa_grade"] == 10
    fmt = lambda d: {k: _prop(v["psa10"], v["n"]) | {"n_graded": v["n"]} for k, v in d.items()}
    abst = [r for r in graded_rows if r["snapshot"].get("verdict") == "insufficient_evidence"]
    return {
        "funnel": funnel,
        "not_submitted_vs_graded": {"n_not_submitted_with_photos": len(feats_unsub), "features_differing_by_0.3_sd_or_more": smd,
                                    "reading": "Large differences mean the cards you chose to grade differ from those you did not, so accuracy "
                                               "measured on graded cards may not hold for the others." if smd else
                                    ("Too few photographed-but-unsubmitted cards to compare." if len(feats_unsub) < 5 else "No feature differs by 0.3 SD or more.")},
        "psa10_rate_by_declared_selection_basis": fmt(by_basis),
        "psa10_rate_by_photo_quality_tier": fmt(by_tier),
        "pipeline_abstentions_among_graded": {"n": len(abst), "psa10_rate": _prop(sum(r["psa_grade"] == 10 for r in abst), len(abst))},
        "what_cannot_be_corrected": [
            "Cards never photographed or never submitted leave no outcome: PSA 10 rates here describe the kinds of cards YOU choose to submit.",
            "If you submit mostly cards you already think are strong, precision will look better than it would on a random card, and recall on weak cards is untested.",
            "Photo-quality selection: if only well-photographed cards get graded, accuracy is unmeasured for poorly photographed ones (compare the tiers above).",
            "No inverse-probability weighting is applied because the probability of submission is unknown; stratified reports are provided instead.",
            "Grader-to-grader variance and PSA standard changes over time are part of the noise in any label."],
    }


# ------------------------------------------------------------------------------------- prediction error tracking
def prediction_errors(ds, since_ts: float | None = None) -> dict:
    """Join what the system said at analysis time (append-only prediction_log) with what PSA returned."""
    with ds.store._conn() as c:
        data = c.execute("""SELECT s.card_uid, s.grade, s.verification_status, k.card_type, p.* FROM submissions s
                            JOIN cards k ON k.card_uid=s.card_uid
                            JOIN prediction_log p ON p.analysis_id=s.primary_analysis_id
                            WHERE s.grade IS NOT NULL AND p.id=(SELECT MIN(id) FROM prediction_log WHERE analysis_id=s.primary_analysis_id)""").fetchall()
    recs = [dict(r) for r in data if since_ts is None or r["ts"] >= since_ts]
    out: dict = {"n": len(recs)}
    if not recs:
        return {**out, "message": "No graded cards have a prediction made before the result was recorded."}
    out["ceiling_violations"] = _prop(sum(r["grade"] > r["conservative"] for r in recs if r["conservative"] is not None),
                                      sum(1 for r in recs if r["conservative"] is not None))
    out["ceiling_violations"]["meaning"] = "PSA grade above the conservative ceiling: the rules were too harsh on these cards."
    out["ceiling_gap"] = {"mean_ceiling_minus_grade": round(float(np.mean([r["conservative"] - r["grade"] for r in recs if r["conservative"] is not None])), 3)} \
        if any(r["conservative"] is not None for r in recs) else None
    wp = [r for r in recs if r["p_psa10"] is not None]
    if wp:
        y = np.array([1.0 if r["grade"] == 10 else 0.0 for r in wp]); p = np.array([r["p_psa10"] for r in wp])
        out["probability_errors"] = {"n": len(wp), "brier": round(float(np.mean((p - y) ** 2)), 4),
                                     "mean_predicted": round(float(p.mean()), 4), "observed_psa10_rate": round(float(y.mean()), 4),
                                     "by_model_version": {}}
        for v in sorted({r["model_version"] or "unversioned" for r in wp}):
            sub = [r for r in wp if (r["model_version"] or "unversioned") == v]
            ys = np.array([1.0 if r["grade"] == 10 else 0.0 for r in sub]); ps = np.array([r["p_psa10"] for r in sub])
            fp = sum(1 for r in sub if r["threshold"] is not None and r["p_psa10"] >= r["threshold"] and r["grade"] != 10)
            fn = sum(1 for r in sub if r["threshold"] is not None and r["p_psa10"] < r["threshold"] and r["grade"] == 10)
            out["probability_errors"]["by_model_version"][v] = {"n": len(sub), "brier": round(float(np.mean((ps - ys) ** 2)), 4),
                                                                "false_positives": fp, "false_negatives": fn,
                                                                "mean_predicted": round(float(ps.mean()), 4), "observed_rate": round(float(ys.mean()), 4)}
    out["largest_model_errors"] = sorted(
        [{"card_uid": r["card_uid"], "grade": r["grade"], "p_psa10": r["p_psa10"], "error": round(abs(r["p_psa10"] - (1.0 if r["grade"] == 10 else 0.0)), 3)}
         for r in wp], key=lambda x: -x["error"])[:10]
    out["rules_misses"] = {
        "psa10_that_rules_called_ruled_out_or_unlikely": [r["card_uid"] for r in recs if r["grade"] == 10 and r["rules_verdict"] in ("ruled_out", "unlikely")],
        "credible_candidates_graded_8_or_lower": [r["card_uid"] for r in recs if r["rules_verdict"] == "credible_candidate" and r["grade"] <= 8]}
    return out
