"""Model versions: retrain on the TRAIN partition, calibrate on VALIDATION, evaluate once on the untouched TEST partition.

Lifecycle:  candidate --(one-shot test)--> passed_test | failed_test --(promote)--> champion --> retired
* Training data: only cards in the train partition of a frozen dataset snapshot.
* Validation partition: chooses the L2 strength, fits the probability calibration and the decision threshold.
  Numbers measured there are optimistic and are labeled so.
* Test partition: touched once per model version, and only if it holds enough cards of each class. Every touch is logged.
* A model trained on synthetic cards is marked synthetic and can never be promoted by a normal dataset.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from . import evaluation as ev
from . import gem_model as gm
from .dataset import Dataset

VAL_MIN_N = 100
VAL_MIN_PER_CLASS = 20
TEST_MIN_PER_CLASS = 20
MAX_COMFORTABLE_TEST_LOOKS = 3


class Registry:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.store = ds.store
        self.vdir = self.store.dir / "models" / "versions"
        self.vdir.mkdir(parents=True, exist_ok=True)
        with self.store._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS model_versions(
              version TEXT PRIMARY KEY, created REAL, snapshot_id TEXT, status TEXT, synthetic INTEGER, feature_version INTEGER,
              params_json TEXT, val_json TEXT, test_json TEXT, parent TEXT, notes TEXT, promoted_ts REAL, retired_ts REAL);
            CREATE TABLE IF NOT EXISTS test_looks(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, version TEXT, snapshot_id TEXT, purpose TEXT, n INTEGER, result_json TEXT);
            """)

    # ---------------------------------------------------------------- files
    def _path(self, version: str):
        return self.vdir / f"{version}.json"

    def _load(self, version: str) -> dict | None:
        p = self._path(version)
        return json.loads(p.read_text()) if p.exists() else None

    def _save(self, version: str, model: dict) -> None:
        self._path(version).write_text(json.dumps(model))

    def _row(self, version: str):
        with self.store._conn() as c:
            return c.execute("SELECT * FROM model_versions WHERE version=?", (version,)).fetchone()

    def _set(self, version: str, **kw) -> None:
        cols = ", ".join(f"{k}=?" for k in kw)
        with self.store._conn() as c:
            c.execute(f"UPDATE model_versions SET {cols} WHERE version=?", [*kw.values(), version])

    def champion(self) -> str | None:
        with self.store._conn() as c:
            r = c.execute("SELECT version FROM model_versions WHERE status='champion'").fetchone()
        return r["version"] if r else None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _matrix(rows, model):
        x, y, _ = gm.build_matrix(rows, model["impute"])
        raw = gm._predict_raw(x, np.array(model["mu"]), np.array(model["sd"]), np.array(model["weights"]))
        return raw, y

    def predict_rows(self, model: dict, rows: list[dict]):
        raw, y = self._matrix(rows, model)
        return ev.apply_calibration(model.get("calibration"), raw), raw, y

    @staticmethod
    def _balance(rows):
        n10 = sum(1 for r in rows if r["psa_grade"] == 10)
        return len(rows), n10, len(rows) - n10

    def readiness(self) -> dict:
        s = self.ds.summary()["partitions"]
        need = {"train": f"at least {gm.MIN_TRAIN_N} cards with at least {gm.MIN_PER_CLASS} PSA 10s and {gm.MIN_PER_CLASS} non-10s",
                "val": f"at least {VAL_MIN_N} cards with at least {VAL_MIN_PER_CLASS} of each class",
                "test": f"at least {TEST_MIN_PER_CLASS} PSA 10s and {TEST_MIN_PER_CLASS} non-10s"}
        ok = {"train": s["train"]["eligible"] >= gm.MIN_TRAIN_N and min(s["train"]["psa10"], s["train"]["grade_le9"]) >= gm.MIN_PER_CLASS,
              "val": s["val"]["eligible"] >= VAL_MIN_N and min(s["val"]["psa10"], s["val"]["grade_le9"]) >= VAL_MIN_PER_CLASS,
              "test": min(s["test"]["psa10"], s["test"]["grade_le9"]) >= TEST_MIN_PER_CLASS}
        return {"ready_to_train": ok["train"] and ok["val"], "ready_to_test": ok["test"], "partitions": s, "requirements": need,
                "satisfied": ok}

    # ---------------------------------------------------------------- retrain
    def retrain(self, note: str = "", target_precision: float | None = None, seed: int = 0) -> dict:
        ready = self.readiness()
        if not ready["ready_to_train"]:
            miss = [f"{k}: needs {ready['requirements'][k]}" for k in ("train", "val") if not ready["satisfied"][k]]
            return {"trained": False, "reason": "Not enough verified, eligible cards in the partitions. " + "; ".join(miss),
                    "readiness": ready}
        snap = self.ds.create_snapshot(note or "retrain")
        tr, va = self.ds.snapshot_rows(snap["id"], "train"), self.ds.snapshot_rows(snap["id"], "val")
        ok, why = gm.can_train(tr)
        n_va, n10_va, n0_va = self._balance(va)
        if not ok or n_va < VAL_MIN_N or min(n10_va, n0_va) < VAL_MIN_PER_CLASS:
            return {"trained": False, "reason": why if not ok else "Validation partition too small in the frozen snapshot.", "snapshot": snap["id"]}
        x, y, impute = gm.build_matrix(tr)                        # imputation medians from TRAIN rows only
        xv, yv, _ = gm.build_matrix(va, impute)
        best, scores = None, {}
        for l2 in gm.L2_GRID:                                      # strength chosen on validation, not train or test
            mu, sd, w = gm._fit_std(x, y, l2)
            scores[l2] = float(np.mean((gm._predict_raw(xv, mu, sd, w) - yv) ** 2))
        l2 = min(scores, key=scores.get)
        mu, sd, w = gm._fit_std(x, y, l2)
        pv_raw = gm._predict_raw(xv, mu, sd, w)
        cal = ev.fit_platt(pv_raw, yv)
        pv = ev.apply_calibration(cal, pv_raw)
        op = ev.operating_point(yv, pv, target_precision)
        val_perf = ev.performance(va, pv, op["threshold"], baseline_rows=tr, p_uncalibrated=pv_raw)
        with self.store._conn() as c:
            n = c.execute("SELECT COUNT(*) n FROM model_versions WHERE version LIKE ?", (f"gm-{time.strftime('%Y%m%d')}-%",)).fetchone()["n"]
        version = f"gm-{time.strftime('%Y%m%d')}-{n + 1:02d}"
        synthetic = any(r["synthetic"] for r in tr + va)
        model = {"version": version, "trained": True, "validated": False, "trained_at": time.time(), "feature_version": gm.FEATURE_VERSION,
                 "feature_names": gm.FEATURE_NAMES, "impute": impute, "mu": mu.tolist(), "sd": sd.tolist(), "weights": w.tolist(), "l2": l2,
                 "l2_validation_brier": {str(k): round(v, 5) for k, v in scores.items()}, "calibration": cal, "threshold": op["threshold"],
                 "threshold_rule": op["rule"], "n_train": len(tr), "n_psa10": int(y.sum()), "snapshot_id": snap["id"],
                 "train_uids": sorted(r["card_uid"] for r in tr), "synthetic": synthetic, "criteria_results": {},
                 "cv": {"metrics": {"calibration_bins": val_perf["calibration"]["bins"]}},
                 "reference_bins": None,
                 "caveat": "Statistical estimate from this store's own submissions. Not validated until it passes the one-shot test partition; "
                           "never a guarantee of a PSA grade." + (" TRAINED ON SYNTHETIC TEST DATA: plumbing test only." if synthetic else "")}
        self._save(version, model)
        parent = self.champion()
        with self.store._conn() as c:
            c.execute("INSERT INTO model_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (version, time.time(), snap["id"], "candidate", int(synthetic), gm.FEATURE_VERSION,
                       json.dumps({"l2": l2, "threshold": op, "calibration": cal, "n_train": len(tr), "n_val": len(va)}),
                       json.dumps(val_perf), None, parent, note[:200], None, None))
        return {"trained": True, "version": version, "status": "candidate", "synthetic": synthetic, "snapshot": snap["id"], "l2": l2,
                "threshold": op, "calibration": cal, "validation_performance": val_perf,
                "validation_note": "Measured on the validation partition, which chose the regularization, calibration and threshold: optimistic. "
                                   "The model is NOT validated until it passes the one-shot test evaluation.",
                "next_step": f"POST /api/dataset/models/{version}/test with confirm=true"}

    # ---------------------------------------------------------------- one-shot test
    def evaluate_on_test(self, version: str, confirm: bool = False) -> dict:
        row = self._row(version)
        if not row:
            raise KeyError("unknown model version")
        if not confirm:
            raise ValueError("The test partition can be evaluated once per model version. Pass confirm=true to use it.")
        with self.store._conn() as c:
            prior = c.execute("SELECT result_json FROM test_looks WHERE version=? AND purpose='primary'", (version,)).fetchone()
        if prior:
            return {**json.loads(prior["result_json"]), "already_evaluated": True,
                    "note": "This version already used its single test evaluation; the stored result is returned unchanged."}
        model = self._load(version)
        te = self.ds.snapshot_rows(row["snapshot_id"], "test")
        n, n10, n0 = self._balance(te)
        if min(n10, n0) < TEST_MIN_PER_CLASS:
            return {"evaluated": False, "version": version,
                    "reason": f"The test partition of this dataset snapshot has {n10} PSA 10s and {n0} others; each class needs at least "
                              f"{TEST_MIN_PER_CLASS}. The test set was NOT used, so it stays untouched. Collect more cards and retrain."}
        p, raw, _ = self.predict_rows(model, te)
        tr = self.ds.snapshot_rows(row["snapshot_id"], "train")
        perf = ev.performance(te, p, model["threshold"], baseline_rows=tr, p_uncalibrated=raw)
        checks = ev.validity_checks(perf, TEST_MIN_PER_CLASS)
        status = "passed_test" if checks["passed"] else "failed_test"
        result = {"evaluated": True, "version": version, "status": status, "snapshot": row["snapshot_id"], "performance": perf,
                  "validity": checks, "synthetic": bool(row["synthetic"]),
                  "meaning": ("Passed the pre-declared criteria on cards the model never saw. That is evidence of predictive value on cards "
                              "like these, not a guarantee for any single card.") if checks["passed"] else
                             "Did NOT meet the pre-declared criteria. The model must not be used for predictions."}
        with self.store._conn() as c:
            c.execute("INSERT INTO test_looks(ts, version, snapshot_id, purpose, n, result_json) VALUES (?,?,?,?,?,?)",
                      (time.time(), version, row["snapshot_id"], "primary", len(te), json.dumps(result)))
        model["validated"] = bool(checks["passed"])
        model["criteria_results"] = checks.get("checks", {})
        if checks["passed"]:
            model["reference_bins"] = perf["calibration"]["bins"]
            model["reference_label"] = "held-out test predictions"
        self._save(version, model)
        self._set(version, status=status, test_json=json.dumps(result))
        with self.store._conn() as c:
            looks = c.execute("SELECT COUNT(DISTINCT version) n FROM test_looks WHERE snapshot_id=?", (row["snapshot_id"],)).fetchone()["n"]
        if looks > MAX_COMFORTABLE_TEST_LOOKS:
            result["warning"] = (f"{looks} different models have now been scored on this same test set. Choosing among many models by test "
                                 "results makes the test numbers optimistic; add newly graded cards and re-check on those.")
        return result

    # ---------------------------------------------------------------- promote / rollback
    def promote(self, version: str) -> dict:
        row = self._row(version)
        if not row:
            raise KeyError("unknown model version")
        if row["status"] != "passed_test":
            raise ValueError(f"Only a model that passed the test partition can be promoted (status is '{row['status']}').")
        if row["synthetic"] and not self.ds.allow_synthetic:
            raise ValueError("Models trained on synthetic data cannot be promoted.")
        model = self._load(version)
        cur = self.champion()
        comparison = None
        if cur:
            cm = self._load(cur)
            te = self.ds.snapshot_rows(row["snapshot_id"], "test")
            pc, _, y = self.predict_rows(cm, te)
            pn, _, _ = self.predict_rows(model, te)
            bc, bn = float(np.mean((pc - y) ** 2)), float(np.mean((pn - y) ** 2))
            comparison = {"on_cards": len(te), "champion": cur, "champion_brier": round(bc, 4), "candidate_brier": round(bn, 4),
                          "note": "Both scored on the candidate's test partition (never used for training either model)."}
            with self.store._conn() as c:
                c.execute("INSERT INTO test_looks(ts, version, snapshot_id, purpose, n, result_json) VALUES (?,?,?,?,?,?)",
                          (time.time(), cur, row["snapshot_id"], "champion_comparison", len(te), json.dumps(comparison)))
            if bn > bc:
                raise ValueError(f"The candidate (Brier {bn:.4f}) is worse than the current champion {cur} (Brier {bc:.4f}) on the same test cards.")
            self._set(cur, status="retired", retired_ts=time.time())
        self._set(version, status="champion", promoted_ts=time.time())
        (self.store.dir / "models" / "champion.json").write_text(json.dumps(model))
        return {"promoted": version, "retired": cur, "comparison": comparison}

    def rollback(self) -> dict:
        cur = self.champion()
        if not cur:
            raise ValueError("There is no champion to roll back.")
        self._set(cur, status="rolled_back", retired_ts=time.time())
        with self.store._conn() as c:
            prev = c.execute("SELECT version FROM model_versions WHERE status='retired' ORDER BY retired_ts DESC LIMIT 1").fetchone()
        champ = self.store.dir / "models" / "champion.json"
        if prev:
            self._set(prev["version"], status="champion", retired_ts=None)
            champ.write_text(json.dumps(self._load(prev["version"])))
        elif champ.exists():
            champ.unlink()
        return {"rolled_back": cur, "restored": prev["version"] if prev else None,
                "now_active": prev["version"] if prev else "no registry model (rules only)"}

    def versions(self) -> list[dict]:
        with self.store._conn() as c:
            rows = c.execute("SELECT * FROM model_versions ORDER BY created DESC").fetchall()
        out = []
        for r in rows:
            t = json.loads(r["test_json"]) if r["test_json"] else None
            out.append({"version": r["version"], "status": r["status"], "created": r["created"], "snapshot_id": r["snapshot_id"],
                        "synthetic": bool(r["synthetic"]), "parent": r["parent"], "notes": r["notes"], "promoted": r["promoted_ts"],
                        "test": None if not t else {"n": t["performance"]["n"], "passed": t["validity"]["passed"],
                                                    "auc": t["performance"]["model"]["auc"], "brier_improvement": t["performance"]["model"]["brier_improvement"]}})
        return out

    # ---------------------------------------------------------------- reports
    def report(self, version: str, group_by: str = "card_type") -> dict:
        row = self._row(version)
        if not row:
            raise KeyError("unknown model version")
        model = self._load(version)
        tested = row["status"] in ("passed_test", "failed_test", "champion", "retired", "rolled_back") and row["test_json"]
        partition = "test" if tested else "val"
        rows = self.ds.snapshot_rows(row["snapshot_id"], partition)
        p, raw, _ = self.predict_rows(model, rows) if rows else (None, None, None)
        tr = self.ds.snapshot_rows(row["snapshot_id"], "train")
        perf = ev.performance(rows, p, model["threshold"], baseline_rows=tr, p_uncalibrated=raw)
        return {"version": version, "status": row["status"], "partition": partition, "synthetic": bool(row["synthetic"]),
                "caveat": None if partition == "test" else "Not tested yet: showing the validation partition, which chose this model's "
                                                          "calibration and threshold, so these numbers are optimistic.",
                "overall": perf, "by_group": ev.group_report(rows, p, model["threshold"], group_by),
                "synthetic_warning": "Built from synthetic test cards: plumbing demonstration only." if row["synthetic"] else None}

    def monitor(self, version: str | None = None) -> dict:
        """Prospective check: cards graded AFTER the model was trained, scored with the model as it was."""
        version = version or self.champion()
        if not version:
            return {"message": "No champion model to monitor."}
        model = self._load(version)
        row = self._row(version)
        in_snapshot = {m["card_uid"] for m in self.ds.get_snapshot(row["snapshot_id"])["manifest"]}
        new = [r for r in self.ds.rows() if r["card_uid"] not in in_snapshot and r["frozen_ts"] > model["trained_at"]]
        if not new:
            return {"version": version, "n": 0, "message": "No cards have been graded since this model was trained."}
        p, raw, _ = self.predict_rows(model, new)
        perf = ev.performance(new, p, model["threshold"], baseline_rows=self.ds.snapshot_rows(row["snapshot_id"], "train"))
        t = json.loads(row["test_json"]) if row["test_json"] else None
        return {"version": version, "n": len(new), "performance": perf,
                "compared_with_test": None if not t else {"test_auc": t["performance"]["model"]["auc"], "test_brier": t["performance"]["model"]["brier"]},
                "note": "These cards were graded after training, so they are a prospective check. Small n means wide intervals: "
                        "do not react to one or two misses."}

    # ---------------------------------------------------------------- provenance (leakage) checks
    def provenance_checks(self) -> list[dict]:
        checks = []
        with self.store._conn() as c:
            vs = [dict(r) for r in c.execute("SELECT * FROM model_versions").fetchall()]
            looks = [dict(r) for r in c.execute("SELECT version, purpose FROM test_looks").fetchall()]
        part = {r["card_uid"]: r["partition"] for r in self.ds.list_cards()}
        bad_part, bad_impute, bad_looks = [], [], []
        for v in vs:
            m = self._load(v["version"])
            if not m:
                continue
            if any(part.get(u) != "train" for u in m["train_uids"]):
                bad_part.append(v["version"])
            rows = self.ds.snapshot_rows(v["snapshot_id"], "train")
            if rows:
                _, _, imp = gm.build_matrix(rows)
                if any(abs(imp[k] - m["impute"][k]) > 1e-9 for k in imp):
                    bad_impute.append(v["version"])
            if sum(1 for lk in looks if lk["version"] == v["version"] and lk["purpose"] == "primary") > 1:
                bad_looks.append(v["version"])
        mk = lambda name, bad, ok: checks.append({"check": name, "passed": not bad, "detail": ok if not bad else "violation", "offenders": bad})
        mk("Every model was trained only on train-partition cards", bad_part, "no model saw validation or test cards")
        mk("Imputation statistics reproduce from the train partition alone", bad_impute, "recomputed medians match")
        mk("No model version used the test partition more than once", bad_looks, "one primary test evaluation per version")
        return checks
