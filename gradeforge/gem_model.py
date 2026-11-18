"""Trained predictive modeling and the feedback loop for the Gem Mint engine.

Nothing here invents a number. The pieces are:

  snapshot()            freeze the features + rules verdict of an analysis (done when a PSA outcome is recorded).
  rules_performance()   how the RULES verdicts compare with real PSA outcomes (works from the first outcome, with
                        Wilson intervals and explicit small-sample warnings).
  train_model()         PSA 10 vs 9-or-lower logistic model. Refuses to train below a data gate. Evaluated only by
                        repeated stratified cross-validation; `validated` is True only if pre-declared criteria pass.
  predict()             a probability plus the OBSERVED outcome rate of cross-validated predictions in the same range.

Eligibility of a row for training/validation (enforced in storage.training_rows):
  * a real PSA grade was recorded,
  * the photos were taken BEFORE submission (explicit flag),
  * the row is not excluded (re-holder, cracked slab, regrade, wrong card ...),
  * the features were frozen at the time the outcome was recorded, by the same feature version.

Known limits, stated rather than hidden: PSA grading has grader-to-grader variance (label noise); a reviewer who
confirmed findings after learning the grade would leak the label (the freeze point and an audit flag limit, but
cannot prove, this); a model trained on one photographer's setup may not transfer to another's.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from .validation import _sigmoid, fit_logistic

FEATURE_VERSION = 1
MIN_TRAIN_N = 300
MIN_PER_CLASS = 40
# Pre-declared validity criteria (fixed in code so they cannot be tuned after seeing results).
MIN_BRIER_IMPROVEMENT = 0.01
MIN_AUC_LOWER_BOUND = 0.60
SLOPE_RANGE = (0.7, 1.3)

FEATURE_NAMES = [
    "front_pct", "back_pct", "front_unc", "back_unc", "back_missing",
    "n_conf_minor", "n_conf_moderate", "n_cand_minor", "n_cand_moderate", "n_unresolved",
    "n_corner_findings", "n_edge_findings", "n_surface_findings",
    "max_corner_whitening", "max_edge_whitening_run_mm", "max_corner_radius_dev_mm", "max_edge_roughness_mm",
    "log_max_relief_strength", "n_relief_lines", "n_dipoles",
    "coverage_fraction", "lighting_front", "lighting_back", "finish_chrome", "resolution_floor_mm",
]
VERDICT_RANK = {"credible_candidate": 3, "uncertain": 2, "unlikely": 1, "ruled_out": 0}
ZONES = ("inside_nominal", "straddles_nominal", "discretion_band", "straddles_outer", "beyond_outer", "beyond_nominal")


# ---------------------------------------------------------------------------------------------------------
# Features and snapshots
# ---------------------------------------------------------------------------------------------------------

def _opt_max(vals):
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def extract_features(gem: dict, result: dict) -> dict:
    """Numeric features (None = unavailable) from a gem analysis and the pipeline result it was computed from."""
    sides = [result.get(k) for k in ("front", "back")]
    f, b = sides
    fd = next((c["detail"] for c in gem["characteristics"] if c["id"] == "front_centering"), None) or {}
    bd = next((c["detail"] for c in gem["characteristics"] if c["id"] == "back_centering"), None) or {}
    out = {n: None for n in FEATURE_NAMES}
    out["front_pct"], out["front_unc"] = fd.get("worst_pct"), fd.get("uncertainty_pct")
    out["back_pct"], out["back_unc"] = bd.get("worst_pct"), bd.get("uncertainty_pct")
    out["back_missing"] = 0.0 if bd else 1.0
    counts = dict(n_conf_minor=0, n_conf_moderate=0, n_cand_minor=0, n_cand_moderate=0, n_unresolved=0)
    ncor = ned = nsur = 0
    relief_strength, nlines, ndip = [], 0, 0
    cw, ew, cr, er = [], [], [], []
    floors = []
    for s in sides:
        if not (s and s.get("ok")):
            continue
        floors.append(s["resolution"]["min_resolvable_feature_mm"])
        cw += [c.get("whitening_fraction") for c in s["wear"]["corners"]]
        cr += [c.get("radius_deviation_mm") for c in s["wear"]["corners"]]
        ew += [e.get("longest_whitening_run_mm") for e in s["wear"]["edges"]]
        er += [e.get("roughness_rms_mm") for e in s["wear"]["edges"]]
        for fi in s["findings"]:
            if fi.get("status") == "rejected" or fi["category"] in ("glare_region",) or fi.get("severity") == "info":
                continue
            conf = fi.get("status") == "confirmed"
            cls = fi["classification"]
            physical = cls.startswith("physical") or cls in ("geometry_measurement", "manual_observation")
            sev = fi["severity"]
            if conf:
                counts["n_conf_moderate" if sev == "moderate" else "n_conf_minor"] += 1
            elif physical:
                counts["n_cand_moderate" if sev == "moderate" else "n_cand_minor"] += 1
            elif cls.startswith("indeterminate"):
                counts["n_unresolved"] += 1
            cat = fi["category"]
            ncor += cat.startswith("corner_")
            ned += cat.startswith("edge_")
            nsur += cat.startswith("surface_")
            if cat in ("surface_relief_line", "surface_relief_dipole", "surface_gloss_variation"):
                relief_strength.append(fi["measurements"].get("strength") or 0.0)
            nlines += cat == "surface_relief_line"
            ndip += cat == "surface_relief_dipole"
    out.update(counts)
    out.update(n_corner_findings=float(ncor), n_edge_findings=float(ned), n_surface_findings=float(nsur),
               max_corner_whitening=_opt_max(cw), max_edge_whitening_run_mm=_opt_max(ew),
               max_corner_radius_dev_mm=_opt_max([abs(v) for v in cr if v is not None]), max_edge_roughness_mm=_opt_max(er),
               log_max_relief_strength=math.log1p(max(relief_strength)) if relief_strength else 0.0,
               n_relief_lines=float(nlines), n_dipoles=float(ndip))
    phys = [c for c in gem["characteristics"] if not c["human_required"] and c["group"] != "centering"]
    out["coverage_fraction"] = (sum(1 for c in phys if c["status"] in ("clear", "minor_concern", "blocks_10", "severe")) / len(phys)) if phys else 0.0
    out["lighting_front"] = float(f["registration"]["lighting_stack"]["images"]) if f and f.get("ok") else None
    out["lighting_back"] = float(b["registration"]["lighting_stack"]["images"]) if b and b.get("ok") else None
    out["finish_chrome"] = 1.0 if gem["card_traits"]["finish"] in ("chrome", "refractor", "foil_holo") else 0.0
    out["resolution_floor_mm"] = max(floors) if floors else None
    return out


def snapshot(gem: dict, result: dict) -> dict:
    group_worst = {}
    for c in gem["characteristics"]:
        if c["human_required"]:
            continue
        rank = {"none": 0, "open": 1, "possible": 2, "likely_block": 3, "block": 4}[c["impact"]]
        group_worst[c["group"]] = max(group_worst.get(c["group"], 0), rank)
    zones = {c["side"]: (c["detail"] or {}).get("zone") for c in gem["characteristics"] if c["group"] == "centering"}
    return {"feature_version": FEATURE_VERSION, "gem_engine": gem["engine_version"], "features": extract_features(gem, result),
            "verdict": gem["psa10_eligibility"]["status"], "grade_ceiling": gem["grade_ceiling"],
            "group_worst_impact": group_worst, "zones": zones, "finish": gem["card_traits"]["finish"]}


# ---------------------------------------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _bootstrap_auc_lower(y, p, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = auc(y[idx], p[idx])
        if a is not None:
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def calibration_bins(y, p, bins=5):
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        k, n = int(y[chunk].sum()), len(chunk)
        out.append({"p_low": round(float(p[chunk].min()), 4), "p_high": round(float(p[chunk].max()), 4),
                    "mean_predicted": round(float(p[chunk].mean()), 4), "n": n, "n_psa10": k,
                    "observed_rate": round(k / n, 4), "ci95": wilson(k, n)})
    return out


def calibration_slope(y, p):
    eps = 1e-4
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))[:, None]
    w = fit_logistic(z, y.astype(float), l2=1e-6)
    return float(w[1])


# ---------------------------------------------------------------------------------------------------------
# Matrix building, training, evaluation
# ---------------------------------------------------------------------------------------------------------

def build_matrix(rows: list[dict], impute: dict | None = None):
    """rows: [{'features': {...}, 'psa_grade': int}] -> X, y, impute-values."""
    raw = np.array([[np.nan if r["features"].get(n) is None else float(r["features"][n]) for n in FEATURE_NAMES] for r in rows], float)
    if impute is None:
        impute = {n: (float(np.nanmedian(raw[:, i])) if np.isfinite(raw[:, i]).any() else 0.0) for i, n in enumerate(FEATURE_NAMES)}
    x = np.where(np.isnan(raw), np.array([impute[n] for n in FEATURE_NAMES])[None, :], raw)
    y = np.array([1 if r.get("psa_grade") == 10 else 0 for r in rows], float) if rows and "psa_grade" in rows[0] else None
    return x, y, impute


def _fit_std(x, y, l2):
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w = fit_logistic((x - mu) / sd, y, l2=l2)
    return mu, sd, w


def _predict_raw(x, mu, sd, w):
    return _sigmoid(np.hstack([np.ones((len(x), 1)), (x - mu) / sd]) @ w)


def cross_validate(x, y, l2=3.0, k=5, repeats=3, seed=0):
    """Repeated stratified k-fold; returns mean out-of-fold prediction per row."""
    n = len(y)
    acc, cnt = np.zeros(n), np.zeros(n)
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        folds = [[] for _ in range(k)]
        for cls in (0, 1):
            idx = rng.permutation(np.where(y == cls)[0])
            for i, j in enumerate(idx):
                folds[i % k].append(j)
        for i in range(k):
            te = np.array(folds[i], int)
            tr = np.array([j for g in range(k) if g != i for j in folds[g]], int)
            mu, sd, w = _fit_std(x[tr], y[tr], l2)
            acc[te] += _predict_raw(x[te], mu, sd, w)
            cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def evaluate_predictions(y, p, threshold=0.5) -> dict:
    base = float(np.mean((y.mean() - y) ** 2))
    brier = float(np.mean((p - y) ** 2))
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    lo, hi = _bootstrap_auc_lower(y, p)
    return {"n": int(len(y)), "n_psa10": int(y.sum()), "brier": round(brier, 4), "baseline_brier": round(base, 4),
            "brier_improvement": round(base - brier, 4), "auc": None if auc(y, p) is None else round(auc(y, p), 4),
            "auc_ci95": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
            "accuracy": round(float(((p >= threshold) == (y == 1)).mean()), 4),
            "majority_baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
            "psa10_precision": None if tp + fp == 0 else round(tp / (tp + fp), 4),
            "psa10_precision_ci95": wilson(tp, tp + fp), "psa10_recall": None if tp + fn == 0 else round(tp / (tp + fn), 4),
            "psa10_recall_ci95": wilson(tp, tp + fn), "calibration_slope": round(calibration_slope(y, p), 3),
            "calibration_bins": calibration_bins(y, p)}


def can_train(rows: list[dict]) -> tuple[bool, str]:
    n = len(rows)
    n10 = sum(1 for r in rows if r["psa_grade"] == 10)
    if n < MIN_TRAIN_N or n10 < MIN_PER_CLASS or n - n10 < MIN_PER_CLASS:
        return False, (f"Training needs at least {MIN_TRAIN_N} eligible cards with at least {MIN_PER_CLASS} PSA 10s and "
                       f"{MIN_PER_CLASS} non-10s; have {n} ({n10} PSA 10s).")
    return True, "ok"


L2_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def train_model(rows: list[dict], l2: float | None = None, seed: int = 0) -> dict:
    ok, why = can_train(rows)
    if not ok:
        return {"trained": False, "validated": False, "reason": why, "n_eligible": len(rows)}
    x, y, impute = build_matrix(rows)
    if l2 is None:
        # regularization strength chosen on one set of CV splits, performance REPORTED on different splits
        best = min(L2_GRID, key=lambda v: float(np.mean((cross_validate(x, y, l2=v, repeats=2, seed=seed + 1000) - y) ** 2)))
        l2 = best
    oof = cross_validate(x, y, l2=l2, seed=seed)
    # decision threshold = the one maximizing balanced accuracy on out-of-fold predictions (not on the final fit)
    ths = np.unique(np.round(oof, 3))
    best_t, best = 0.5, -1.0
    for t in ths:
        pr = oof >= t
        tpr = ((pr == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
        tnr = ((pr == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
        if (tpr + tnr) / 2 > best:
            best, best_t = (tpr + tnr) / 2, float(t)
    metrics = evaluate_predictions(y, oof, best_t)
    lo = metrics["auc_ci95"][0]
    slope = metrics["calibration_slope"]
    checks = {
        "brier_beats_base_rate_by_margin": metrics["brier_improvement"] >= MIN_BRIER_IMPROVEMENT,
        "auc_lower_bound_ok": lo is not None and lo >= MIN_AUC_LOWER_BOUND,
        "calibration_slope_ok": SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1],
    }
    mu, sd, w = _fit_std(x, y, l2)
    data_hash = hashlib.sha256(json.dumps([[r["analysis_id"], r["psa_grade"]] for r in rows], sort_keys=True).encode()).hexdigest()
    return {"trained": True, "validated": all(checks.values()), "trained_at": time.time(), "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES, "impute": impute, "mu": mu.tolist(), "sd": sd.tolist(), "weights": w.tolist(), "l2": l2,
            "threshold": best_t, "n_train": len(rows), "n_psa10": int(y.sum()), "cv": {"folds": 5, "repeats": 3, "metrics": metrics},
            "criteria": {"min_brier_improvement": MIN_BRIER_IMPROVEMENT, "min_auc_lower_bound": MIN_AUC_LOWER_BOUND,
                         "calibration_slope_range": list(SLOPE_RANGE)},
            "criteria_results": checks, "data_hash": data_hash,
            "caveat": "Cross-validated on this store's own submissions. Grader variance, one photographer's setup and any "
                      "label leakage in review timing are not removed by cross-validation."}


def predict(model: dict, features: dict) -> dict | None:
    if not model or not model.get("trained") or model.get("feature_version") != FEATURE_VERSION:
        return None
    x, _, _ = build_matrix([{"features": features}], model["impute"])
    p = float(_predict_raw(x, np.array(model["mu"]), np.array(model["sd"]), np.array(model["weights"]))[0])
    cal = model.get("calibration")
    if cal:   # logistic recalibration fitted on the validation partition
        e = min(max(p, 1e-4), 1 - 1e-4)
        p = 1.0 / (1.0 + math.exp(-(cal["a"] + cal["b"] * math.log(e / (1 - e)))))
    bins = model.get("reference_bins") or model["cv"]["metrics"]["calibration_bins"]
    label = model.get("reference_label", "cross-validated predictions")
    near = next((b for b in bins if b["p_low"] <= p <= b["p_high"]), None) or min(bins, key=lambda b: abs(b["mean_predicted"] - p))
    return {"p_psa10": round(p, 4),
            "observed_in_similar_cv_predictions": {"p_range": [near["p_low"], near["p_high"]], "n": near["n"], "n_psa10": near["n_psa10"],
                                                   "observed_rate": near["observed_rate"], "ci95": near["ci95"]},
            "interpretation": (f"Among {near['n']} {label} in this range, {near['n_psa10']} received PSA 10 "
                               f"(95% CI {near['ci95'][0]:.2f}-{near['ci95'][1]:.2f}).")}


def model_public(model: dict | None, features: dict | None = None) -> dict | None:
    """What the gem engine may see: only a validated model, with its prediction for this card."""
    if not model or not model.get("trained"):
        return None
    out = {"trained": True, "validated": bool(model["validated"]), "version": model.get("version"), "threshold": model["threshold"], "n_train": model["n_train"],
           "criteria_results": model["criteria_results"], "trained_at": model["trained_at"]}
    if model["validated"] and features is not None:
        out["prediction"] = predict(model, features)
    return out


# ---------------------------------------------------------------------------------------------------------
# Feedback: how the RULES verdicts compare with real outcomes
# ---------------------------------------------------------------------------------------------------------

def rules_performance(rows: list[dict]) -> dict:
    """rows: eligible rows with 'snapshot' (frozen) and 'psa_grade'."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "message": "No eligible PSA outcomes recorded yet. Record the real grade (with the before-submission photo flag) "
                                   "on analyses to evaluate the rules."}
    by = {}
    for r in rows:
        v = r["snapshot"]["verdict"]
        d = by.setdefault(v, {"n": 0, "grades": {}, "n_psa10": 0})
        d["n"] += 1
        d["n_psa10"] += r["psa_grade"] == 10
        d["grades"][str(r["psa_grade"])] = d["grades"].get(str(r["psa_grade"]), 0) + 1
    for v, d in by.items():
        d["psa10_rate"] = round(d["n_psa10"] / d["n"], 3)
        d["psa10_rate_ci95"] = wilson(d["n_psa10"], d["n"])
    scored = [r for r in rows if r["snapshot"]["verdict"] in VERDICT_RANK]
    y = np.array([1 if r["psa_grade"] == 10 else 0 for r in scored], float)
    out = {"n": n, "n_scored": len(scored), "by_verdict": by}
    if len(scored):
        s = np.array([VERDICT_RANK[r["snapshot"]["verdict"]] for r in scored], float)
        out["rules_auc_for_psa10"] = None if auc(y, s) is None else round(auc(y, s), 3)
        for name, keep in (("credible_only", {"credible_candidate"}), ("not_ruled_out", {"credible_candidate", "uncertain"})):
            sel = np.array([r["snapshot"]["verdict"] in keep for r in scored])
            tp = int((sel & (y == 1)).sum()); fp = int((sel & (y == 0)).sum()); fn = int((~sel & (y == 1)).sum())
            out[name] = {"flagged": int(sel.sum()), "precision": None if tp + fp == 0 else round(tp / (tp + fp), 3),
                         "precision_ci95": wilson(tp, tp + fp), "recall": None if tp + fn == 0 else round(tp / (tp + fn), 3),
                         "recall_ci95": wilson(tp, tp + fn), "false_exclusions": fn}
        ruled = [r for r in scored if r["snapshot"]["verdict"] in ("ruled_out", "unlikely")]
        out["psa10_among_ruled_out_or_unlikely"] = {"n": len(ruled), "n_psa10": sum(r["psa_grade"] == 10 for r in ruled)}
    # characteristic-group audit: PSA 10 rate when a group had a likely/definite blocker
    audit = {}
    for r in scored:
        for g, rank in r["snapshot"]["group_worst_impact"].items():
            key = f"{g}:{'blocker' if rank >= 3 else 'concern' if rank == 2 else 'open' if rank == 1 else 'clear'}"
            d = audit.setdefault(key, {"n": 0, "n_psa10": 0})
            d["n"] += 1
            d["n_psa10"] += r["psa_grade"] == 10
    for d in audit.values():
        d["psa10_rate"] = round(d["n_psa10"] / d["n"], 3)
        d["ci95"] = wilson(d["n_psa10"], d["n"])
    out["group_audit"] = audit
    out["warning"] = (f"Only {n} outcomes: intervals are wide and no rule should be changed on this evidence." if n < 100 else
                      "Treat rule changes as hypotheses; test them on new outcomes, not the ones that suggested them.")
    out["review_hint"] = ("A group whose 'blocker' state still shows a high PSA 10 rate suggests that rule is too strict; a 'clear' state "
                          "with a low rate suggests missing detection. Rules are never changed automatically.")
    return out


def reliability_report(rows: list[dict], model: dict | None, n_outcomes_total: int | None = None, excluded: dict | None = None) -> dict:
    ok, why = can_train(rows)
    perf = rules_performance(rows)
    return {"n_eligible": len(rows), "n_outcomes_total": n_outcomes_total, "excluded_reasons": excluded or {},
            "training_gate": {"met": ok, "message": why, "required_n": MIN_TRAIN_N, "required_per_class": MIN_PER_CLASS},
            "rules_performance": perf, "rules_validated": False,
            "rules_validation_note": ("Rules are reported with measured performance only; they are not called validated until enough "
                                      "outcomes exist to bound their error."),
            "model": None if not model or not model.get("trained") else {
                "validated": model["validated"], "n_train": model["n_train"], "trained_at": model["trained_at"],
                "cv_metrics": model["cv"]["metrics"], "criteria_results": model["criteria_results"], "caveat": model["caveat"]},
            "model_validated": bool(model and model.get("validated"))}
