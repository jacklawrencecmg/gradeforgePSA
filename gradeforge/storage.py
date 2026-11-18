"""SQLite history and PSA-outcome tracking."""
from __future__ import annotations

import os

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .analyze import reassess
from .cv.detect import decode_image
from .cv.evidence import write_evidence
from .findings import CATEGORIES
from . import gem_model
from .validation import compute_metrics

DATA_DIR = Path(os.environ.get("GRADEFORGE_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")


class Store:
    def __init__(self, data_dir: Path | None = None):
        self.dir = Path(data_dir or DATA_DIR)
        for d in ("images", "evidence", "originals", "models"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.db = self.dir / "gradeforge.sqlite3"
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS analyses(
                id TEXT PRIMARY KEY, created REAL, label TEXT, result_json TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS outcomes(
                analysis_id TEXT PRIMARY KEY, psa_grade INTEGER, note TEXT, recorded REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS outcome_meta(
                analysis_id TEXT PRIMARY KEY, qualifier TEXT, pre_submission_photos INTEGER, submitted_on TEXT,
                service TEXT, cert TEXT, excluded INTEGER, exclude_reason TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS gem_snapshots(
                analysis_id TEXT PRIMARY KEY, feature_version INTEGER, snapshot_json TEXT, frozen INTEGER, ts REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS prediction_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, ts REAL, model_version TEXT, p_psa10 REAL,
                threshold REAL, rules_verdict TEXT, conservative INTEGER, optimistic INTEGER, most_likely INTEGER)""")
            c.execute("""CREATE TABLE IF NOT EXISTS review_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, side TEXT, finding_id TEXT, action TEXT,
                reviewer TEXT, note TEXT, payload TEXT, ts REAL)""")

    def _conn(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, label: str, result: dict, arts: dict) -> str:
        aid = uuid.uuid4().hex[:12]
        for side, art in arts.items():
            if not art:
                continue
            cv2.imwrite(str(self.dir / "images" / f"{aid}_{side}.png"), art["rect"])
            ev = self.dir / "evidence" / aid
            ev.mkdir(parents=True, exist_ok=True)
            for name, data in art["evidence"].items():
                (ev / name).write_bytes(data)
            od = self.dir / "originals" / aid
            od.mkdir(parents=True, exist_ok=True)
            for role, data in art["originals"].items():
                (od / f"{side}_{role}.bin").write_bytes(data)
        with self._conn() as c:
            c.execute("INSERT INTO analyses VALUES (?,?,?,?)", (aid, time.time(), label, json.dumps(result)))
            if result.get("gem"):
                c.execute("INSERT OR REPLACE INTO gem_snapshots VALUES (?,?,?,?,?)",
                          (aid, gem_model.FEATURE_VERSION, json.dumps(gem_model.snapshot(result["gem"], result)), 0, time.time()))
        return aid

    def log_prediction(self, aid: str, result: dict, model_version: str | None = None) -> None:
        """Record what the system said at analysis time. Append-only: later reviews never rewrite it."""
        g = result.get("gem") or {}
        pred = ((g.get("reliability") or {}).get("model_prediction")) or {}
        mp = (g.get("most_likely") or {}).get("grade")
        thr = ((g.get("reliability") or {}).get("model") or {}).get("threshold")
        model_version = model_version or ((g.get("reliability") or {}).get("model") or {}).get("version")
        with self._conn() as c:
            c.execute("""INSERT INTO prediction_log(analysis_id, ts, model_version, p_psa10, threshold, rules_verdict, conservative, optimistic, most_likely)
                         VALUES (?,?,?,?,?,?,?,?,?)""",
                      (aid, time.time(), model_version, pred.get("p_psa10"), thr, (g.get("psa10_eligibility") or {}).get("status"),
                       (g.get("grade_ceiling") or {}).get("conservative"), (g.get("grade_ceiling") or {}).get("optimistic"), mp))

    def _post_outcome(self, aid: str, rec: dict) -> bool:
        if rec.get("outcome") is not None:
            return True
        with self._conn() as c:
            return c.execute("SELECT 1 FROM submissions WHERE primary_analysis_id=? AND grade IS NOT NULL", (aid,)).fetchone() is not None \
                if self._has_table(c, "submissions") else False

    @staticmethod
    def _has_table(c, name: str) -> bool:
        return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    def _refresh(self, aid: str, result: dict):
        """Recompute assessment + Gem analysis. The frozen snapshot (if an outcome exists) is deliberately NOT touched."""
        reassess(result, model=self.load_model(), reliability=self.reliability())
        self._write_result(aid, result)
        with self._conn() as c:
            c.execute("UPDATE gem_snapshots SET snapshot_json=?, ts=? WHERE analysis_id=? AND frozen=0",
                      (json.dumps(gem_model.snapshot(result["gem"], result)), time.time(), aid))

    def _write_result(self, aid: str, result: dict):
        with self._conn() as c:
            c.execute("UPDATE analyses SET result_json=? WHERE id=?", (json.dumps(result), aid))

    def _log(self, aid, side, fid, action, reviewer, note, payload):
        with self._conn() as c:
            c.execute("INSERT INTO review_log(analysis_id, side, finding_id, action, reviewer, note, payload, ts) "
                      "VALUES (?,?,?,?,?,?,?,?)", (aid, side, fid, action, reviewer, note, json.dumps(payload), time.time()))

    def audit_log(self, aid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM review_log WHERE analysis_id=? ORDER BY id", (aid,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]

    def evidence_path(self, aid: str, name: str) -> Path | None:
        if not aid.isalnum() or not re.fullmatch(r"(front|back)_[A-Za-z0-9_]+\.png", name or ""):
            return None
        p = self.dir / "evidence" / aid / name
        return p if p.is_file() else None

    def original_path(self, aid: str, side: str, role: str) -> Path | None:
        if not aid.isalnum() or side not in ("front", "back") or not re.fullmatch(r"(primary|light\d+|macro\d+)", role):
            return None
        p = self.dir / "originals" / aid / f"{side}_{role}.bin"
        return p if p.is_file() else None

    # ---------------- review workflow ----------------
    def _find(self, rec: dict, side: str, fid: str):
        s = rec["result"].get(side)
        if not s or not s.get("ok"):
            raise ValueError(f"no analyzed {side} side")
        for f in s["findings"]:
            if f["id"] == fid:
                return f
        raise ValueError(f"unknown finding {fid}")

    def review_finding(self, aid: str, side: str, fid: str, action: str, reviewer: str = "", note: str = "",
                       evidence_keys: list[str] | None = None, severity: str | None = None) -> dict:
        """confirm needs at least one evidence file key from the finding (traceability); reject needs a reason."""
        rec = self.get(aid)
        if rec is None:
            raise KeyError(aid)
        f = self._find(rec, side, fid)
        reviewer = (reviewer or "").strip()[:80] or "unnamed reviewer"
        if action == "confirm":
            keys = evidence_keys or []
            have = f.get("evidence", {}).get("files", {})
            if not keys or any(k not in have for k in keys):
                raise ValueError("Confirming requires citing at least one of this finding's evidence crops.")
            if severity:
                if severity not in ("minor", "moderate", "info"):
                    raise ValueError("severity must be minor, moderate or info")
                f["severity"] = severity
            f["status"] = "confirmed"
            f["review"] = {"action": "confirm", "reviewer": reviewer, "note": note[:1000], "evidence_keys": keys, "time": time.time()}
        elif action == "reject":
            if not note.strip():
                raise ValueError("Rejecting requires a short reason.")
            f["status"] = "rejected"
            f["review"] = {"action": "reject", "reviewer": reviewer, "note": note[:1000], "evidence_keys": evidence_keys or [], "time": time.time()}
        elif action == "reset":
            f["status"] = "candidate"
            f["review"] = None
        else:
            raise ValueError("action must be confirm, reject or reset")
        self._log(aid, side, fid, action, reviewer, note, {"evidence_keys": evidence_keys, "severity": severity,
                                                           "post_outcome": self._post_outcome(aid, rec)})
        self._refresh(aid, rec["result"])
        return rec["result"]

    def add_manual_finding(self, aid: str, side: str, box: list[int], target: str, severity: str, note: str,
                           reviewer: str = "") -> dict:
        """A person marks a region the detectors missed. Evidence crops are cut from the stored original photo."""
        rec = self.get(aid)
        if rec is None:
            raise KeyError(aid)
        s = rec["result"].get(side)
        if not s or not s.get("ok"):
            raise ValueError(f"no analyzed {side} side")
        if severity not in ("minor", "moderate", "info"):
            raise ValueError("severity must be minor, moderate or info")
        if not note.strip():
            raise ValueError("Describe what you see.")
        w, h = s["detection"]["canvas"]
        x, y, bw, bh = [int(round(v)) for v in box]
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        if bw < 4 or bh < 4:
            raise ValueError("The box is too small or outside the card.")
        valid_t = {"corner_tl", "corner_tr", "corner_br", "corner_bl", "edge_top", "edge_right", "edge_bottom",
                   "edge_left", "artwork", "ring_border", "other"}
        if target not in valid_t:
            raise ValueError("unknown target")
        rect = cv2.imread(str(self.dir / "images" / f"{aid}_{side}.png"))
        op = self.original_path(aid, side, "primary")
        orig = decode_image(op.read_bytes()) if op else None
        if rect is None or orig is None:
            raise ValueError("Stored photos are not available for this analysis.")
        reviewer = (reviewer or "").strip()[:80] or "unnamed reviewer"
        n = 1 + sum(1 for f in s["findings"] if f["category"] == "manual")
        from .findings import new_finding
        f = new_finding(side, "manual", target, [x, y, bw, bh], {"strength": 0.0}, severity, "manual_observation",
                        ["Entered by a person; the evidence crops show the marked region of the stored photo."],
                        "manual.person", note.strip()[:300])
        f["id"] = f"{side[0]}m{n:02d}"
        M = np.array(s["detection"]["transform_original_to_canvas"], np.float64)
        files = write_evidence([f], side, rect, orig, M, s["detection"]["source_scale"], [], None, [],
                               [{"role": "primary", "sha256": s["detection"]["source_sha256"]}])
        ev = self.dir / "evidence" / aid
        for name, data in files.items():
            (ev / name).write_bytes(data)
        f["status"] = "confirmed"
        f["review"] = {"action": "manual_entry", "reviewer": reviewer, "note": note[:1000],
                       "evidence_keys": ["primary_original", "primary_canvas"], "time": time.time()}
        s["findings"].append(f)
        self._log(aid, side, f["id"], "manual_entry", reviewer, note, {"box": [x, y, bw, bh], "target": target,
                                                                      "post_outcome": self._post_outcome(aid, rec)})
        self._refresh(aid, rec["result"])
        return rec["result"]

    def get(self, aid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
            if not row:
                return None
            out = c.execute("""SELECT o.psa_grade, o.note, m.qualifier, m.pre_submission_photos, m.submitted_on, m.service, m.excluded
                               FROM outcomes o LEFT JOIN outcome_meta m ON m.analysis_id=o.analysis_id WHERE o.analysis_id=?""", (aid,)).fetchone()
        return {"id": row["id"], "created": row["created"], "label": row["label"],
                "result": json.loads(row["result_json"]),
                "outcome": dict(out) if out else None}

    def image_path(self, aid: str, side: str) -> Path | None:
        if side not in ("front", "back") or not aid.isalnum():
            return None
        p = self.dir / "images" / f"{aid}_{side}.png"
        return p if p.exists() else None

    def history(self, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""SELECT a.id, a.created, a.label, a.result_json, o.psa_grade
                                FROM analyses a LEFT JOIN outcomes o ON o.analysis_id=a.id
                                ORDER BY a.created DESC LIMIT ?""", (limit,)).fetchall()
        out = []
        for r in rows:
            full = json.loads(r["result_json"])
            a = full["assessment"]
            out.append({"id": r["id"], "created": r["created"], "label": r["label"],
                        "gem_verdict": (full.get("gem") or {}).get("psa10_eligibility", {}).get("status"),
                        "status": a["prediction_status"], "conservative": a.get("conservative_ceiling"),
                        "optimistic": a.get("optimistic_ceiling"), "psa10": a["psa10_eligibility"]["status"],
                        "actual_psa_grade": r["psa_grade"]})
        return out

    def set_outcome(self, aid: str, grade: int, note: str = "", *, qualifier: str | None = None,
                    pre_submission_photos: bool = False, submitted_on: str | None = None, service: str | None = None,
                    cert: str | None = None, excluded: bool = False, exclude_reason: str = "") -> bool:
        """Record the real PSA result and FREEZE the features the engine saw at this moment."""
        rec = self.get(aid)
        if rec is None:
            return False
        res = rec["result"]
        snap = gem_model.snapshot(res["gem"], res) if res.get("gem") else None
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)", (aid, int(grade), note, time.time()))
            c.execute("INSERT OR REPLACE INTO outcome_meta VALUES (?,?,?,?,?,?,?,?)",
                      (aid, (qualifier or "")[:8], int(bool(pre_submission_photos)), (submitted_on or "")[:20], (service or "")[:40],
                       (cert or "")[:20], int(bool(excluded)), exclude_reason[:200]))
            if snap:
                c.execute("INSERT OR REPLACE INTO gem_snapshots VALUES (?,?,?,?,?)",
                          (aid, gem_model.FEATURE_VERSION, json.dumps(snap), 1, time.time()))
        return True

    def training_rows(self) -> tuple[list[dict], int, dict]:
        """Eligible rows (frozen features, pre-submission photos, not excluded, current feature version)."""
        rows, total, why = [], 0, {}
        with self._conn() as c:
            data = c.execute("""SELECT o.analysis_id, o.psa_grade, o.recorded, m.pre_submission_photos AS pre, m.excluded, m.exclude_reason,
                                       m.qualifier, s.snapshot_json, s.frozen, s.feature_version
                                FROM outcomes o LEFT JOIN outcome_meta m ON m.analysis_id=o.analysis_id
                                LEFT JOIN gem_snapshots s ON s.analysis_id=o.analysis_id""").fetchall()
        for r in data:
            total += 1
            reason = None
            if r["snapshot_json"] is None or not r["frozen"]:
                reason = "no frozen feature snapshot"
            elif r["feature_version"] != gem_model.FEATURE_VERSION:
                reason = "older feature version"
            elif not r["pre"]:
                reason = "photos not confirmed as taken before submission"
            elif r["excluded"]:
                reason = "excluded by user: " + (r["exclude_reason"] or "no reason given")
            if reason:
                why[reason] = why.get(reason, 0) + 1
                continue
            snap = json.loads(r["snapshot_json"])
            rows.append({"analysis_id": r["analysis_id"], "psa_grade": r["psa_grade"], "qualifier": r["qualifier"] or "",
                         "snapshot": snap, "features": snap["features"]})
        return rows, total, why

    # ---------------- model + reliability ----------------
    def _model_file(self) -> Path:
        return self.dir / "models" / "gem_model.json"

    def load_model(self) -> dict | None:
        """The registry's champion (validated on a held-out test partition) wins; else the quick-feedback model."""
        champ = self.dir / "models" / "champion.json"
        try:
            if champ.exists():
                return json.loads(champ.read_text())
        except (OSError, ValueError):
            pass
        p = self._model_file()
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except (OSError, ValueError):
            return None

    def train_model(self) -> dict:
        rows, total, why = self.training_rows()
        model = gem_model.train_model(rows)
        if model.get("trained"):
            stamp = time.strftime("%Y%m%d%H%M%S")
            (self.dir / "models" / f"gem_model_{stamp}.json").write_text(json.dumps(model))
            self._model_file().write_text(json.dumps(model))
        return {**model, "n_outcomes_total": total, "excluded_reasons": why}

    def reliability(self) -> dict:
        rows, total, why = self.training_rows()
        return gem_model.reliability_report(rows, self.load_model(), total, why)

    def training_csv(self) -> str:
        rows, _, _ = self.training_rows()
        names = gem_model.FEATURE_NAMES
        lines = [",".join(["analysis_id", "psa_grade", "qualifier", "verdict"] + names)]
        for r in rows:
            vals = ["" if r["features"].get(n) is None else repr(float(r["features"][n])) for n in names]
            lines.append(",".join([r["analysis_id"], str(r["psa_grade"]), r["qualifier"], r["snapshot"]["verdict"]] + vals))
        return "\n".join(lines) + "\n"

    def comparison(self) -> dict:
        records = []
        with self._conn() as c:
            rows = c.execute("""SELECT a.result_json, o.psa_grade FROM analyses a
                                JOIN outcomes o ON o.analysis_id=a.id""").fetchall()
        for r in rows:
            a = json.loads(r["result_json"])["assessment"]
            records.append({"actual": r["psa_grade"], "status": a["prediction_status"],
                            "conservative": a.get("conservative_ceiling"), "optimistic": a.get("optimistic_ceiling"),
                            "p10_status": a["psa10_eligibility"]["status"]})
        return compute_metrics(records)
