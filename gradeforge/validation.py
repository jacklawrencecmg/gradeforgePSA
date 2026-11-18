"""Validation against REAL PSA outcomes.

This module never invents numbers: with no labeled data it reports that and nothing else.

Labeled dataset CSV columns:
    card_id, front_image, back_image, psa_grade
`psa_grade` must be the grade PSA actually assigned, and the images must have been taken
BEFORE submission. Image paths are resolved relative to the CSV file.

Usage:
    python -m gradeforge.validation path/to/labeled.csv [--consistency]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MIN_CALIBRATION_N = 300
MIN_CALIBRATION_PER_CLASS = 40


def record_from_result(actual: int, result: dict) -> dict:
    a = result["assessment"]
    return {
        "actual": int(actual),
        "status": a["prediction_status"],
        "conservative": a.get("conservative_ceiling"),
        "optimistic": a.get("optimistic_ceiling"),
        "p10_status": a["psa10_eligibility"]["status"],
    }


def compute_metrics(records: list[dict]) -> dict:
    """records: dicts from record_from_result. Returns metrics, or an explicit 'no data' message."""
    n_total = len(records)
    if n_total == 0:
        return {"n_total": 0, "message": "No labeled data: no validation metrics computed."}
    scored = [r for r in records if r["status"] == "ceilings_only"]
    out = {"n_total": n_total, "n_scored": len(scored),
           "abstained": n_total - len(scored),
           "abstention_rate": round((n_total - len(scored)) / n_total, 4)}
    if not scored:
        out["message"] = "Every card was abstained on (insufficient evidence); no accuracy metrics possible."
        return out
    act = np.array([r["actual"] for r in scored])
    cons = np.array([r["conservative"] for r in scored])
    opt = np.array([r["optimistic"] for r in scored])
    # Safety property: PSA's actual grade should never exceed the optimistic ceiling.
    out["ceiling_violations"] = int((act > opt).sum())
    out["ceiling_violation_rate"] = round(float((act > opt).mean()), 4)
    out["mean_gap_optimistic_minus_actual"] = round(float((opt - act).mean()), 3)
    # Baseline point predictor = conservative ceiling (the engine itself gives no point estimate).
    out["baseline_predictor"] = "conservative_ceiling"
    out["exact_accuracy"] = round(float((cons == act).mean()), 4)
    out["within_one_accuracy"] = round(float((np.abs(cons - act) <= 1).mean()), 4)
    # PSA 10 screening: positive = "not ruled out".
    is10 = act == 10
    kept = np.array([r["p10_status"] == "not_ruled_out" for r in scored])
    tp = int((kept & is10).sum())
    out["psa10_actual_count"] = int(is10.sum())
    out["psa10_screen_recall"] = round(tp / is10.sum(), 4) if is10.any() else None
    out["psa10_screen_precision"] = round(tp / kept.sum(), 4) if kept.any() else None
    out["psa10_false_exclusions"] = int((~kept & is10).sum())
    # Predicting exactly 10 (needs a calibrated point predictor; the baseline never predicts 10).
    pred10 = cons == 10
    out["psa10_exact_precision"] = round(float((pred10 & is10).sum() / pred10.sum()), 4) if pred10.any() else None
    out["psa10_exact_recall"] = round(float((pred10 & is10).sum() / is10.sum()), 4) if is10.any() else None
    if len(scored) < 100:
        out["warning"] = f"Only {len(scored)} scored cards; these metrics are too noisy to draw conclusions from."
    return out


def _perturb(data: bytes, seed: int) -> bytes:
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    ang = float(rng.uniform(-0.7, 0.7))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(82, 94))])
    return buf.tobytes()


def consistency(front: bytes, back: bytes, trials: int = 3) -> dict:
    """Re-analyze slightly perturbed copies of the same photos (small rotation, recompression)."""
    from .analyze import analyze_card
    base, _ = analyze_card(front, back)
    b = base["assessment"]
    outs = [analyze_card(_perturb(front, i), _perturb(back, 100 + i))[0]["assessment"] for i in range(trials)]
    same = sum(1 for o in outs if (o.get("conservative_ceiling"), o.get("optimistic_ceiling")) ==
               (b.get("conservative_ceiling"), b.get("optimistic_ceiling")))
    return {"trials": trials, "identical_ceilings": same, "ceilings_consistent": same == trials}


def evaluate_dataset(csv_path: str, with_consistency: bool = False) -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"n_total": 0, "message": f"No dataset at {csv_path}: no validation metrics computed."}
    records, cons_rows = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                grade = int(row["psa_grade"])
                fb = (path.parent / row["front_image"]).read_bytes()
                bb = (path.parent / row["back_image"]).read_bytes()
            except (KeyError, ValueError, OSError) as exc:
                records.append({"actual": 0, "status": "error", "error": str(exc)})
                continue
            if not 1 <= grade <= 10:
                continue
            from .analyze import analyze_card
            result, _ = analyze_card(fb, bb)
            records.append(record_from_result(grade, result))
            if with_consistency:
                cons_rows.append(consistency(fb, bb))
    records = [r for r in records if r["status"] != "error"]
    metrics = compute_metrics(records)
    if cons_rows:
        metrics["consistency_cards"] = len(cons_rows)
        metrics["consistency_rate"] = round(sum(c["ceilings_consistent"] for c in cons_rows) / len(cons_rows), 4)
    return metrics


# ---------- Calibration gate ----------

def can_calibrate(labels: list[int]) -> tuple[bool, str]:
    n = len(labels)
    n10 = sum(1 for g in labels if g == 10)
    if n < MIN_CALIBRATION_N or n10 < MIN_CALIBRATION_PER_CLASS or (n - n10) < MIN_CALIBRATION_PER_CLASS:
        return False, (f"Calibration needs at least {MIN_CALIBRATION_N} labeled cards with at least "
                       f"{MIN_CALIBRATION_PER_CLASS} PSA 10s and {MIN_CALIBRATION_PER_CLASS} non-10s; "
                       f"have {n} ({n10} PSA 10s).")
    return True, "ok"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    xb = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(xb.shape[1])
    reg = np.eye(xb.shape[1]) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = _sigmoid(xb @ w)
        g = xb.T @ (p - y) + reg @ w
        h = xb.T @ (xb * (p * (1 - p))[:, None]) + reg + 1e-6 * np.eye(xb.shape[1])
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def cross_validated_brier(x: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    preds = np.zeros(len(y))
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        w = fit_logistic((x[tr] - mu) / sd, y[tr])
        preds[te] = _sigmoid(np.hstack([np.ones((len(te), 1)), (x[te] - mu) / sd]) @ w)
    brier = float(np.mean((preds - y) ** 2))
    baseline = float(np.mean((y.mean() - y) ** 2))
    return {"cv_brier": round(brier, 4), "baseline_brier": round(baseline, 4),
            "beats_baseline": brier < baseline, "folds": k}


def main():
    ap = argparse.ArgumentParser(description="Evaluate GradeForge against real PSA-graded cards.")
    ap.add_argument("dataset_csv")
    ap.add_argument("--consistency", action="store_true", help="also test prediction stability under small image perturbations")
    args = ap.parse_args()
    print(json.dumps(evaluate_dataset(args.dataset_csv, args.consistency), indent=2))


if __name__ == "__main__":
    main()
