"""Curated dataset of PHYSICAL cards for measured validation.

One row per physical card (not per upload). Each card links to pre-grading analyses, one PSA result with certification
data and a verification step, human defect annotations, and a partition (train / val / test) that is assigned once,
deterministically, from a stored random salt and never changes.

Nothing here invents data. Synthetic cards exist only so the plumbing can be tested; they are flagged
(origin = synthetic_test), refused unless the Dataset was created with allow_synthetic=True, and any model trained on
them is marked synthetic and can never become the production model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
import uuid
from datetime import date

from . import gem_model
from .storage import Store

# ---------------------------------------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------------------------------------
CARD_TYPES = {
    "sports_modern_base": "Modern sports, standard cardboard/base",
    "sports_modern_chrome": "Modern sports, chrome / refractor / prizm-style",
    "sports_vintage": "Sports, pre-1980",
    "sports_junk_wax": "Sports, 1980-1999",
    "tcg_standard": "Trading card game, non-holo",
    "tcg_holo": "Trading card game, holo / foil / full-art",
    "non_sport_other": "Other non-sports / entertainment",
    "other": "Other",
}
FINISHES = ("unknown", "standard", "chrome", "refractor", "foil_holo", "matte")
SELECTION_BASES = {
    "all_owned_cards": "Every eligible card I owned was submitted, regardless of how it looked",
    "prescreened_by_owner": "I only submitted cards I already believed were high-grade candidates",
    "mixed_or_unknown": "Mixed or unknown reason for submission",
}
ORIGINS = ("real_physical_card", "synthetic_test")
IMAGE_SOURCES = ("own_photographs", "licensed_with_permission")
PHASES = {
    "pre_grading": "Photographed before submission to PSA. The ONLY phase that can be used for training or evaluation.",
    "post_grading_slab": "Photographed inside the PSA slab/holder. Stored for reference, never used for training.",
    "other": "Anything else (reference photos). Never used for training.",
}
QUALIFIERS = ("", "OC", "MK", "ST", "PD", "OF", "MC")
ANNOTATION_KINDS = {
    "corner_whitening": ("corner_whitening",),
    "corner_damage": ("corner_chip", "corner_missing_stock", "corner_fray", "corner_rounding", "corner_softness", "corner_relief"),
    "edge_whitening": ("edge_whitening",),
    "edge_damage": ("edge_notch", "edge_rough_cut", "edge_layer_anomaly"),
    "hairline_or_scratch": ("surface_relief_line", "surface_linear_mark"),
    "print_line": ("surface_linear_mark", "surface_relief_line"),
    "dent_or_indentation": ("surface_relief_dipole", "corner_relief"),
    "stain_or_spot": ("surface_stain", "surface_spot"),
    "gloss_irregularity": ("surface_gloss_variation", "surface_periodic_banding"),
    "print_defect": ("surface_spot", "surface_linear_mark"),
    "other": (),
}
VERIFICATION = ("unverified", "verified", "conflict")

SPLIT = (("train", 0.70), ("val", 0.15), ("test", 0.15))
# Policy, declared in advance. Changing these is a visible edit to this file, not a silent runtime choice.
POLICY = {"require_verified_grade": True, "require_pre_grading_images": True, "require_capture_before_submission": True,
          "feature_version": gem_model.FEATURE_VERSION, "min_group_n_for_metrics": 30}


def _today() -> str:
    return date.today().isoformat()


def _iso(d: str | None, name: str) -> str | None:
    if d in (None, ""):
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(d)):
        raise ValueError(f"{name} must be YYYY-MM-DD")
    date.fromisoformat(d)
    return d


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def cert_verification_url(cert: str) -> str:
    return f"https://www.psacard.com/cert/{cert}"


def photo_quality(result: dict) -> dict:
    """Photo-quality tier from the pipeline result: used only to REPORT selection effects, never as a grade input."""
    reasons = []
    for side in ("front", "back"):
        s = result.get(side)
        if not (s and s.get("ok")):
            reasons.append(f"{side} not measurable")
            continue
        suff = s["quality"]["sufficient_for"]
        if not all(suff.values()):
            reasons.append(f"{side} resolution/exposure insufficient for: " + ", ".join(k for k, v in suff.items() if not v))
        if s["registration"]["lighting_stack"]["images"] < 2:     # the stack counts the primary photo
            reasons.append(f"{side} has no registered extra lighting photo")
    return {"tier": "limited" if reasons else "good", "reasons": reasons}


class Dataset:
    def __init__(self, store: Store, allow_synthetic: bool = False):
        self.store = store
        self.allow_synthetic = allow_synthetic
        with store._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS dataset_meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS cards(
              card_uid TEXT PRIMARY KEY, created REAL, label TEXT, origin TEXT, card_type TEXT, finish TEXT,
              set_name TEXT, year INTEGER, subject TEXT, card_number TEXT, variant TEXT, group_key TEXT, partition TEXT,
              selection_basis TEXT, rights_attested INTEGER, image_source TEXT, owner_note TEXT,
              excluded INTEGER DEFAULT 0, exclude_reason TEXT DEFAULT '', not_submitted_reason TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS card_images(
              id INTEGER PRIMARY KEY AUTOINCREMENT, card_uid TEXT, analysis_id TEXT UNIQUE, phase TEXT, captured_on TEXT,
              note TEXT, content_sha TEXT, linked REAL);
            CREATE TABLE IF NOT EXISTS submissions(
              card_uid TEXT PRIMARY KEY, submitted_on TEXT, service TEXT, submission_no TEXT, cert TEXT,
              grade INTEGER, qualifier TEXT, grade_date TEXT, result_recorded REAL, primary_analysis_id TEXT,
              verification_status TEXT DEFAULT 'unverified', verified_by TEXT, verified_on TEXT, verification_ref TEXT,
              cert_page_grade INTEGER, cert_page_checked_url TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS submissions_cert ON submissions(cert) WHERE cert IS NOT NULL AND cert<>'';
            CREATE TABLE IF NOT EXISTS frozen(
              card_uid TEXT PRIMARY KEY, analysis_id TEXT, frozen_ts REAL, snapshot_sha TEXT, quality_tier TEXT, quality_json TEXT,
              initial_prediction_json TEXT);
            CREATE TABLE IF NOT EXISTS annotations(
              id INTEGER PRIMARY KEY AUTOINCREMENT, card_uid TEXT, analysis_id TEXT, side TEXT, kind TEXT, box TEXT,
              severity TEXT, annotator TEXT, note TEXT, created REAL, blind INTEGER, withdrawn INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS annotation_status(
              analysis_id TEXT, side TEXT, annotator TEXT, complete INTEGER, ts REAL, PRIMARY KEY(analysis_id, side, annotator));
            CREATE TABLE IF NOT EXISTS dataset_snapshots(
              id TEXT PRIMARY KEY, created REAL, note TEXT, sha TEXT, manifest_json TEXT, policy_json TEXT);
            CREATE TABLE IF NOT EXISTS dataset_audit(
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, card_uid TEXT, action TEXT, actor TEXT, payload TEXT);
            """)
            if not c.execute("SELECT 1 FROM dataset_meta WHERE key='salt'").fetchone():
                c.execute("INSERT INTO dataset_meta VALUES ('salt', ?)", (secrets.token_hex(16),))

    # ------------------------------------------------------------------ helpers
    def _salt(self) -> str:
        with self.store._conn() as c:
            return c.execute("SELECT value FROM dataset_meta WHERE key='salt'").fetchone()["value"]

    def partition_for(self, group: str) -> str:
        u = int(hashlib.sha256(f"{self._salt()}:{group}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        acc = 0.0
        for name, frac in SPLIT:
            acc += frac
            if u < acc:
                return name
        return SPLIT[-1][0]

    def _audit(self, c, uid, action, actor, payload):
        c.execute("INSERT INTO dataset_audit(ts, card_uid, action, actor, payload) VALUES (?,?,?,?,?)",
                  (time.time(), uid, action, actor or "", json.dumps(payload)))

    def audit(self, uid: str | None = None) -> list[dict]:
        with self.store._conn() as c:
            q, a = ("SELECT * FROM dataset_audit WHERE card_uid=? ORDER BY id", (uid,)) if uid else ("SELECT * FROM dataset_audit ORDER BY id", ())
            return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in c.execute(q, a).fetchall()]

    # ------------------------------------------------------------------ cards
    def create_card(self, *, label: str = "", card_type: str, finish: str = "unknown", set_name: str = "", year: int | None = None,
                    subject: str = "", card_number: str = "", variant: str = "", group_key: str = "",
                    selection_basis: str = "mixed_or_unknown", rights_attested: bool = False,
                    image_source: str = "own_photographs", owner_note: str = "", origin: str = "real_physical_card",
                    actor: str = "") -> dict:
        if card_type not in CARD_TYPES:
            raise ValueError(f"card_type must be one of {sorted(CARD_TYPES)}")
        if finish not in FINISHES:
            raise ValueError(f"finish must be one of {FINISHES}")
        if selection_basis not in SELECTION_BASES:
            raise ValueError(f"selection_basis must be one of {sorted(SELECTION_BASES)}")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        if origin == "synthetic_test" and not self.allow_synthetic:
            raise ValueError("Synthetic cards are refused: this dataset accepts only real physical cards.")
        if image_source not in IMAGE_SOURCES:
            raise ValueError(f"image_source must be one of {IMAGE_SOURCES}")
        if not rights_attested:
            raise ValueError("rights_attested must be true: confirm you took these photographs (or have permission to use "
                             "them) and that using them to train and evaluate GradeForge is allowed.")
        issue = "|".join(_norm(x) for x in (set_name, str(year or ""), card_number, variant))
        group = _norm(group_key) or (f"issue:{issue}" if set_name and year and card_number else "")
        uid = "c" + uuid.uuid4().hex[:10]
        group = group or uid
        part = self.partition_for(group)
        with self.store._conn() as c:
            c.execute("""INSERT INTO cards(card_uid, created, label, origin, card_type, finish, set_name, year, subject, card_number,
                         variant, group_key, partition, selection_basis, rights_attested, image_source, owner_note)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (uid, time.time(), label[:120], origin, card_type, finish, set_name[:80], year, subject[:80], card_number[:20],
                       variant[:60], group, part, selection_basis, 1, image_source, owner_note[:500]))
            self._audit(c, uid, "card_created", actor, {"partition": part, "group": group})
        return self.get_card(uid)

    def _card_row(self, uid: str):
        with self.store._conn() as c:
            return c.execute("SELECT * FROM cards WHERE card_uid=?", (uid,)).fetchone()

    def get_card(self, uid: str) -> dict | None:
        r = self._card_row(uid)
        if not r:
            return None
        with self.store._conn() as c:
            imgs = [dict(x) for x in c.execute("SELECT * FROM card_images WHERE card_uid=? ORDER BY id", (uid,)).fetchall()]
            sub = c.execute("SELECT * FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            frz = c.execute("SELECT * FROM frozen WHERE card_uid=?", (uid,)).fetchone()
            nann = c.execute("SELECT COUNT(*) n FROM annotations WHERE card_uid=? AND withdrawn=0", (uid,)).fetchone()["n"]
        ok, why = self.eligibility(uid)
        card = {**dict(r), "images": imgs, "submission": dict(sub) if sub else None, "frozen": dict(frz) if frz else None,
                "n_annotations": nann, "eligible_for_training": ok, "ineligible_reasons": why,
                "status": ("graded" if sub and sub["grade"] else "submitted" if sub else "photographed" if imgs else "registered")}
        card.pop("rights_attested", None)
        return card

    def list_cards(self, partition: str | None = None, status: str | None = None) -> list[dict]:
        with self.store._conn() as c:
            uids = [r["card_uid"] for r in c.execute("SELECT card_uid FROM cards ORDER BY created DESC").fetchall()]
        out = []
        for u in uids:
            card = self.get_card(u)
            if partition and card["partition"] != partition or status and card["status"] != status:
                continue
            out.append({k: card[k] for k in ("card_uid", "label", "origin", "card_type", "finish", "partition", "status",
                                              "eligible_for_training", "ineligible_reasons", "selection_basis", "n_annotations")}
                       | {"n_images": len(card["images"]), "grade": (card["submission"] or {}).get("grade"),
                          "verification": (card["submission"] or {}).get("verification_status")})
        return out

    def exclude_card(self, uid: str, reason: str, actor: str = "") -> None:
        if not reason.strip():
            raise ValueError("A reason is required to exclude a card.")
        with self.store._conn() as c:
            c.execute("UPDATE cards SET excluded=1, exclude_reason=? WHERE card_uid=?", (reason[:200], uid))
            self._audit(c, uid, "excluded", actor, {"reason": reason})

    # ------------------------------------------------------------------ images
    def _content_sha(self, aid: str) -> str | None:
        parts = []
        for side in ("front", "back"):
            p = self.store.dir / "originals" / aid / f"{side}_primary.bin"
            if p.is_file():
                parts.append(f"{side}:{hashlib.sha256(p.read_bytes()).hexdigest()}")
        return "|".join(parts) or None

    def link_image(self, uid: str, analysis_id: str, phase: str, captured_on: str | None = None, note: str = "", actor: str = "") -> dict:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {sorted(PHASES)}. Choose it explicitly: slab photos are not pre-grading photos.")
        card = self._card_row(uid)
        if not card:
            raise KeyError("unknown card")
        rec = self.store.get(analysis_id)
        if rec is None:
            raise KeyError("unknown analysis")
        captured_on = _iso(captured_on, "captured_on")
        with self.store._conn() as c:
            sub = c.execute("SELECT grade FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            taken = c.execute("SELECT card_uid FROM card_images WHERE analysis_id=?", (analysis_id,)).fetchone()
        if taken:
            raise ValueError(f"This analysis is already linked to card {taken['card_uid']}.")
        if phase == "pre_grading":
            if sub and sub["grade"]:
                raise ValueError("The PSA grade is already recorded for this card, so new photographs cannot count as pre-grading.")
            if not captured_on:
                raise ValueError("captured_on (the date you took the photographs) is required for pre-grading images.")
        sha = self._content_sha(analysis_id)
        if sha:
            with self.store._conn() as c:
                for r in c.execute("SELECT card_uid, content_sha FROM card_images WHERE content_sha IS NOT NULL AND card_uid<>?", (uid,)):
                    if set(r["content_sha"].split("|")) & set(sha.split("|")):
                        raise ValueError(f"These exact photo files are already linked to card {r['card_uid']}: "
                                         "the same photo cannot belong to two cards.")
        with self.store._conn() as c:
            c.execute("INSERT INTO card_images(card_uid, analysis_id, phase, captured_on, note, content_sha, linked) VALUES (?,?,?,?,?,?,?)",
                      (uid, analysis_id, phase, captured_on, note[:200], sha, time.time()))
            self._audit(c, uid, "image_linked", actor, {"analysis_id": analysis_id, "phase": phase, "captured_on": captured_on})
        return self.get_card(uid)

    # ------------------------------------------------------------------ submission, result, verification
    def record_submission(self, uid: str, submitted_on: str, service: str = "", submission_no: str = "", actor: str = "") -> dict:
        if not self._card_row(uid):
            raise KeyError("unknown card")
        submitted_on = _iso(submitted_on, "submitted_on")
        if not submitted_on:
            raise ValueError("submitted_on is required")
        with self.store._conn() as c:
            row = c.execute("SELECT grade FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            if row and row["grade"]:
                raise ValueError("A result is already recorded; the submission date can no longer change.")
            c.execute("""INSERT INTO submissions(card_uid, submitted_on, service, submission_no) VALUES (?,?,?,?)
                         ON CONFLICT(card_uid) DO UPDATE SET submitted_on=excluded.submitted_on, service=excluded.service,
                         submission_no=excluded.submission_no""", (uid, submitted_on, service[:40], submission_no[:30]))
            self._audit(c, uid, "submitted", actor, {"submitted_on": submitted_on, "service": service})
        return self.get_card(uid)

    def record_result(self, uid: str, grade: int, *, cert: str = "", qualifier: str = "", grade_date: str | None = None,
                      primary_analysis_id: str | None = None, submitted_on: str | None = None, service: str = "", actor: str = "") -> dict:
        """Record the PSA result ONCE and freeze the features the engine saw for the primary pre-grading analysis."""
        card = self._card_row(uid)
        if not card:
            raise KeyError("unknown card")
        if not isinstance(grade, int) or not 1 <= grade <= 10:
            raise ValueError("grade must be an integer 1-10")
        qualifier = (qualifier or "").upper()
        if qualifier not in QUALIFIERS:
            raise ValueError(f"qualifier must be one of {QUALIFIERS}")
        cert = re.sub(r"\s+", "", cert or "")
        if cert and not re.fullmatch(r"\d{6,12}", cert):
            raise ValueError("cert must be the numeric PSA certification number (6-12 digits)")
        grade_date = _iso(grade_date, "grade_date")
        with self.store._conn() as c:
            sub = c.execute("SELECT * FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            if sub and sub["grade"]:
                raise ValueError("A result is already recorded for this card. Use correct_result() so the change is audited.")
            if cert:
                dup = c.execute("SELECT card_uid FROM submissions WHERE cert=? AND card_uid<>?", (cert, uid)).fetchone()
                if dup:
                    raise ValueError(f"Cert {cert} is already recorded for card {dup['card_uid']}.")
            pre = c.execute("SELECT analysis_id FROM card_images WHERE card_uid=? AND phase='pre_grading' ORDER BY linked", (uid,)).fetchall()
        if not pre:
            raise ValueError("Link at least one pre-grading photograph analysis before recording the result.")
        pre_ids = [r["analysis_id"] for r in pre]
        primary = primary_analysis_id or pre_ids[-1]
        if primary not in pre_ids:
            raise ValueError("primary_analysis_id must be one of this card's pre-grading analyses.")
        rec = self.store.get(primary)
        if not rec or not rec["result"].get("gem"):
            raise ValueError("The primary analysis has no Gem Mint evaluation to freeze.")
        submitted_on = _iso(submitted_on, "submitted_on") or (sub["submitted_on"] if sub else None)
        snap = gem_model.snapshot(rec["result"]["gem"], rec["result"])
        snap_json = json.dumps(snap, sort_keys=True)
        quality = photo_quality(rec["result"])
        now = time.time()
        with self.store._conn() as c:
            c.execute("""INSERT INTO submissions(card_uid, submitted_on, service, cert, grade, qualifier, grade_date, result_recorded,
                         primary_analysis_id) VALUES (?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(card_uid) DO UPDATE SET cert=excluded.cert, grade=excluded.grade, qualifier=excluded.qualifier,
                         grade_date=excluded.grade_date, result_recorded=excluded.result_recorded, primary_analysis_id=excluded.primary_analysis_id,
                         submitted_on=COALESCE(excluded.submitted_on, submissions.submitted_on),
                         service=CASE WHEN excluded.service<>'' THEN excluded.service ELSE submissions.service END""",
                      (uid, submitted_on, service[:40], cert, grade, qualifier, grade_date, now, primary))
            c.execute("INSERT OR REPLACE INTO gem_snapshots VALUES (?,?,?,?,?)", (primary, gem_model.FEATURE_VERSION, snap_json, 1, now))
            c.execute("INSERT OR REPLACE INTO frozen VALUES (?,?,?,?,?,?,?)",
                      (uid, primary, now, hashlib.sha256(snap_json.encode()).hexdigest(), quality["tier"], json.dumps(quality),
                       self._initial_prediction_json(primary)))
            self._audit(c, uid, "result_recorded", actor, {"grade": grade, "qualifier": qualifier, "cert": cert, "primary": primary})
        return self.get_card(uid)

    def _initial_prediction_json(self, aid: str) -> str | None:
        with self.store._conn() as c:
            try:
                r = c.execute("SELECT * FROM prediction_log WHERE analysis_id=? ORDER BY id LIMIT 1", (aid,)).fetchone()
            except Exception:
                return None
        return json.dumps(dict(r)) if r else None

    def correct_result(self, uid: str, *, grade: int | None = None, cert: str | None = None, reason: str, actor: str = "") -> dict:
        if not reason.strip():
            raise ValueError("A reason is required to correct a recorded result.")
        with self.store._conn() as c:
            sub = c.execute("SELECT * FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            if not sub or not sub["grade"]:
                raise ValueError("No result recorded yet.")
            old = {"grade": sub["grade"], "cert": sub["cert"], "verification_status": sub["verification_status"]}
            if grade is not None:
                if not 1 <= int(grade) <= 10:
                    raise ValueError("grade must be 1-10")
                c.execute("UPDATE submissions SET grade=? WHERE card_uid=?", (int(grade), uid))
            if cert is not None:
                c.execute("UPDATE submissions SET cert=? WHERE card_uid=?", (re.sub(r"\s+", "", cert), uid))
            c.execute("UPDATE submissions SET verification_status='unverified', verified_by=NULL, verified_on=NULL WHERE card_uid=?", (uid,))
            self._audit(c, uid, "result_corrected", actor, {"reason": reason, "before": old, "grade": grade, "cert": cert})
        return self.get_card(uid)

    def verify_result(self, uid: str, *, verified_by: str, cert_number_seen: str, cert_page_grade: int, reference: str = "") -> dict:
        """A person looked the certification up on PSA's cert page and reports what it shows. GradeForge does not
        contact PSA itself and never marks a result verified on its own."""
        if not verified_by.strip():
            raise ValueError("verified_by is required")
        with self.store._conn() as c:
            sub = c.execute("SELECT * FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            if not sub or not sub["grade"]:
                raise ValueError("Record the result first.")
            if not sub["cert"]:
                raise ValueError("No certification number recorded: record the cert before verifying.")
            seen = re.sub(r"\s+", "", cert_number_seen or "")
            ok = seen == sub["cert"] and int(cert_page_grade) == int(sub["grade"])
            status = "verified" if ok else "conflict"
            c.execute("""UPDATE submissions SET verification_status=?, verified_by=?, verified_on=?, verification_ref=?, cert_page_grade=?,
                         cert_page_checked_url=? WHERE card_uid=?""",
                      (status, verified_by[:60], _today(), reference[:200], int(cert_page_grade), cert_verification_url(seen), uid))
            self._audit(c, uid, "verification", verified_by, {"status": status, "cert_seen": seen, "page_grade": cert_page_grade})
        return self.get_card(uid)

    # ------------------------------------------------------------------ eligibility
    def eligibility(self, uid: str, policy: dict | None = None) -> tuple[bool, list[str]]:
        pol = policy or POLICY
        card = self._card_row(uid)
        why: list[str] = []
        if card["excluded"]:
            why.append("excluded: " + (card["exclude_reason"] or "no reason"))
        if card["origin"] == "synthetic_test" and not self.allow_synthetic:
            why.append("synthetic test card")
        with self.store._conn() as c:
            sub = c.execute("SELECT * FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            frz = c.execute("SELECT * FROM frozen WHERE card_uid=?", (uid,)).fetchone()
            snap = c.execute("SELECT feature_version, frozen FROM gem_snapshots WHERE analysis_id=?",
                             (frz["analysis_id"],)).fetchone() if frz else None
            img = c.execute("SELECT * FROM card_images WHERE analysis_id=?", (frz["analysis_id"],)).fetchone() if frz else None
        if not sub or not sub["grade"]:
            why.append("no PSA result recorded")
            return False, why
        if not frz or not snap or not snap["frozen"]:
            why.append("no frozen feature snapshot")
        elif snap["feature_version"] != pol["feature_version"]:
            why.append("older feature version")
        if img is None or img["phase"] != "pre_grading":
            why.append("primary image is not a pre-grading photograph")
        elif pol["require_capture_before_submission"]:
            if not sub["submitted_on"]:
                why.append("submission date missing, so photo timing cannot be checked")
            elif not img["captured_on"] or img["captured_on"] > sub["submitted_on"]:
                why.append("photographs not captured on or before the submission date")
        if pol["require_verified_grade"] and sub["verification_status"] != "verified":
            why.append("grade not verified against the PSA cert page" if sub["verification_status"] == "unverified"
                       else "grade conflicts with the PSA cert page")
        return (not why), why

    def rows(self, partition: str | None = None, include_ineligible: bool = False) -> list[dict]:
        """One row per eligible physical card, built ONLY from the frozen snapshot."""
        out = []
        with self.store._conn() as c:
            uids = [r["card_uid"] for r in c.execute("SELECT card_uid FROM cards ORDER BY created").fetchall()]
        for uid in uids:
            ok, why = self.eligibility(uid)
            if not ok and not include_ineligible:
                continue
            card = self._card_row(uid)
            if partition and card["partition"] != partition:
                continue
            with self.store._conn() as c:
                sub = c.execute("SELECT * FROM submissions WHERE card_uid=?", (uid,)).fetchone()
                frz = c.execute("SELECT * FROM frozen WHERE card_uid=?", (uid,)).fetchone()
                snap = c.execute("SELECT snapshot_json FROM gem_snapshots WHERE analysis_id=?", (frz["analysis_id"],)).fetchone() if frz else None
            if not (sub and sub["grade"] and snap):
                continue
            s = json.loads(snap["snapshot_json"])
            out.append({"card_uid": uid, "analysis_id": frz["analysis_id"], "partition": card["partition"], "psa_grade": sub["grade"],
                        "qualifier": sub["qualifier"] or "", "card_type": card["card_type"], "finish": card["finish"],
                        "year": card["year"], "set_name": card["set_name"], "selection_basis": card["selection_basis"],
                        "quality_tier": frz["quality_tier"], "frozen_ts": frz["frozen_ts"], "snapshot": s, "features": s["features"],
                        "synthetic": card["origin"] == "synthetic_test", "eligible": ok, "ineligible_reasons": why,
                        "initial_prediction": json.loads(frz["initial_prediction_json"]) if frz["initial_prediction_json"] else None})
        return out

    # ------------------------------------------------------------------ snapshots (frozen dataset versions)
    def create_snapshot(self, note: str = "") -> dict:
        rows = self.rows()
        manifest = [{"card_uid": r["card_uid"], "partition": r["partition"], "grade": r["psa_grade"], "analysis_id": r["analysis_id"],
                     "snapshot_sha": hashlib.sha256(json.dumps(r["snapshot"], sort_keys=True).encode()).hexdigest()} for r in rows]
        manifest.sort(key=lambda m: m["card_uid"])
        sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        sid = "ds-" + time.strftime("%Y%m%d%H%M%S") + "-" + sha[:6]
        with self.store._conn() as c:
            existing = c.execute("SELECT id FROM dataset_snapshots WHERE sha=?", (sha,)).fetchone()
            if existing:
                return self.get_snapshot(existing["id"])
            c.execute("INSERT INTO dataset_snapshots VALUES (?,?,?,?,?,?)", (sid, time.time(), note[:200], sha, json.dumps(manifest), json.dumps(POLICY)))
        return self.get_snapshot(sid)

    def get_snapshot(self, sid: str) -> dict | None:
        with self.store._conn() as c:
            r = c.execute("SELECT * FROM dataset_snapshots WHERE id=?", (sid,)).fetchone()
        if not r:
            return None
        m = json.loads(r["manifest_json"])
        counts = {p: sum(1 for x in m if x["partition"] == p) for p, _ in SPLIT}
        return {"id": r["id"], "created": r["created"], "note": r["note"], "sha": r["sha"], "n": len(m), "counts": counts,
                "manifest": m, "policy": json.loads(r["policy_json"])}

    def list_snapshots(self) -> list[dict]:
        with self.store._conn() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM dataset_snapshots ORDER BY created DESC").fetchall()]
        return [{k: v for k, v in self.get_snapshot(i).items() if k != "manifest"} for i in ids]

    def snapshot_rows(self, sid: str, partition: str | None = None) -> list[dict]:
        """Rows exactly as they were in a dataset snapshot (frozen features; later card changes do not leak in)."""
        snap = self.get_snapshot(sid)
        want = {m["card_uid"]: m for m in snap["manifest"] if not partition or m["partition"] == partition}
        by = {r["card_uid"]: r for r in self.rows(include_ineligible=True)}
        out = []
        for u, m in want.items():
            if u in by:
                out.append({**by[u], "psa_grade": m["grade"]})   # the grade as it was when the dataset version was cut
        return out

    # ------------------------------------------------------------------ annotations
    def add_annotation(self, uid: str, analysis_id: str, side: str, kind: str, box: list[float], severity: str = "minor",
                       annotator: str = "", note: str = "") -> dict:
        if kind not in ANNOTATION_KINDS:
            raise ValueError(f"kind must be one of {sorted(ANNOTATION_KINDS)}")
        if side not in ("front", "back") or severity not in ("minor", "moderate", "severe"):
            raise ValueError("side must be front/back and severity minor/moderate/severe")
        if not annotator.strip():
            raise ValueError("annotator is required (who made this judgement)")
        if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
            raise ValueError("box must be [x, y, width, height] in rectified-card pixels")
        with self.store._conn() as c:
            link = c.execute("SELECT phase FROM card_images WHERE card_uid=? AND analysis_id=?", (uid, analysis_id)).fetchone()
            sub = c.execute("SELECT grade FROM submissions WHERE card_uid=?", (uid,)).fetchone()
            if not link:
                raise ValueError("That analysis is not linked to this card.")
            if link["phase"] != "pre_grading":
                raise ValueError("Annotate pre-grading photographs only.")
            blind = 0 if (sub and sub["grade"]) else 1
            cur = c.execute("""INSERT INTO annotations(card_uid, analysis_id, side, kind, box, severity, annotator, note, created, blind)
                               VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            (uid, analysis_id, side, kind, json.dumps([float(v) for v in box]), severity, annotator[:60], note[:300], time.time(), blind))
            self._audit(c, uid, "annotation_added", annotator, {"id": cur.lastrowid, "kind": kind, "blind": bool(blind)})
            return {"id": cur.lastrowid, "blind": bool(blind),
                    "warning": None if blind else "The PSA grade was already recorded when this was annotated: marked NOT blind and excluded from detector evaluation."}

    def withdraw_annotation(self, ann_id: int, actor: str = "") -> None:
        with self.store._conn() as c:
            c.execute("UPDATE annotations SET withdrawn=1 WHERE id=?", (ann_id,))
            self._audit(c, None, "annotation_withdrawn", actor, {"id": ann_id})

    def list_annotations(self, uid: str | None = None, analysis_id: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM annotations WHERE withdrawn=0", []
        if uid:
            q += " AND card_uid=?"; a.append(uid)
        if analysis_id:
            q += " AND analysis_id=?"; a.append(analysis_id)
        with self.store._conn() as c:
            return [{**dict(r), "box": json.loads(r["box"])} for r in c.execute(q + " ORDER BY id", a).fetchall()]

    def set_annotation_complete(self, analysis_id: str, side: str, annotator: str, complete: bool = True) -> None:
        """'complete' = this annotator inspected the whole side and marked every defect they could see. Needed before
        detector PRECISION can be measured (otherwise unannotated detections are not necessarily false)."""
        with self.store._conn() as c:
            c.execute("INSERT OR REPLACE INTO annotation_status VALUES (?,?,?,?,?)", (analysis_id, side, annotator, int(complete), time.time()))

    @staticmethod
    def _boxes_match(a, b, pad: float = 10.0, min_overlap: float = 0.3) -> bool:
        ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[0] + a[2] + pad, a[1] + a[3] + pad
        bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[0] + b[2] + pad, b[1] + b[3] + pad
        iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
        if iw <= 0 or ih <= 0:
            return False
        return iw * ih / min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)) >= min_overlap

    def evaluate_detectors(self) -> dict:
        """Detector recall (and precision on completely annotated sides) against BLIND human annotations."""
        anns = [a for a in self.list_annotations() if a["blind"]]
        per: dict[str, dict] = {}

        def bucket(k):
            return per.setdefault(k, {"annotated": 0, "found": 0, "detections": 0, "detections_matched": 0})
        with self.store._conn() as c:
            complete = {(r["analysis_id"], r["side"]) for r in c.execute("SELECT * FROM annotation_status WHERE complete=1")}
        pairs = sorted({(a["analysis_id"], a["side"]) for a in anns})
        for aid, side in pairs:
            rec = self.store.get(aid)
            if not rec or not rec["result"].get(side, {}).get("ok"):
                continue
            dets = [f for f in rec["result"][side]["findings"] if f["classification"] != "manual_observation"
                    and f["category"] not in ("glare_region", "alteration_indicator") and f["severity"] != "info"]
            side_anns = [a for a in anns if a["analysis_id"] == aid and a["side"] == side]
            for a in side_anns:
                b = bucket(a["kind"]); b["annotated"] += 1
                cats = ANNOTATION_KINDS[a["kind"]]
                if any(f["category"] in cats and self._boxes_match(a["box"], f["box"]) for f in dets):
                    b["found"] += 1
            if (aid, side) in complete:
                for f in dets:
                    kinds = [k for k, cats in ANNOTATION_KINDS.items() if f["category"] in cats]
                    b = bucket(kinds[0] if kinds else "other")
                    b["detections"] += 1
                    if any(a["kind"] in kinds and self._boxes_match(a["box"], f["box"]) for a in side_anns):
                        b["detections_matched"] += 1
        out = {}
        for k, b in per.items():
            out[k] = {**b, "recall": None if not b["annotated"] else round(b["found"] / b["annotated"], 3),
                      "recall_ci95": gem_model.wilson(b["found"], b["annotated"]),
                      "precision_on_fully_annotated_sides": None if not b["detections"] else round(b["detections_matched"] / b["detections"], 3),
                      "precision_ci95": gem_model.wilson(b["detections_matched"], b["detections"]),
                      "small_sample": b["annotated"] < 20}
        return {"n_blind_annotations": len(anns), "n_not_blind_excluded": len(self.list_annotations()) - len(anns),
                "n_fully_annotated_sides": len(complete), "by_kind": out,
                "matching_rule": "A detection matches an annotation of a compatible kind when their boxes, each padded by 10 px, overlap "
                                 "by at least 30% of the smaller box. Detectors are hand-written, not trained, so these numbers are a "
                                 "direct measurement and not in-sample fit.",
                "note": "Precision counts a detection as wrong only on sides an annotator marked completely annotated."}

    def annotator_agreement(self) -> dict:
        anns = [a for a in self.list_annotations() if a["blind"]]
        groups: dict = {}
        for a in anns:
            groups.setdefault((a["analysis_id"], a["side"]), {}).setdefault(a["annotator"], []).append(a)
        n = m = 0
        for by in groups.values():
            if len(by) < 2:
                continue
            names = sorted(by)
            for x in by[names[0]]:
                n += 1
                m += any(y["kind"] == x["kind"] and self._boxes_match(x["box"], y["box"]) for y in by[names[1]])
        return {"pairs_compared": n, "matched": m, "agreement": None if not n else round(m / n, 3), "ci95": gem_model.wilson(m, n),
                "note": "Share of one annotator's marks that a second annotator also marked (same kind, overlapping box)."}

    # ------------------------------------------------------------------ leakage audit
    def leakage_audit(self, registry=None) -> dict:
        checks = []

        def add(name, bad, detail_ok, detail_bad=""):
            checks.append({"check": name, "passed": not bad, "detail": detail_ok if not bad else detail_bad, "offenders": bad[:20]})
        with self.store._conn() as c:
            cards = [dict(r) for r in c.execute("SELECT * FROM cards").fetchall()]
            imgs = [dict(r) for r in c.execute("SELECT * FROM card_images").fetchall()]
            subs = {r["card_uid"]: dict(r) for r in c.execute("SELECT * FROM submissions").fetchall()}
            frz = {r["card_uid"]: dict(r) for r in c.execute("SELECT * FROM frozen").fetchall()}
        add("Partition matches the stored salted hash of the card's group (assignment was never edited)",
            [x["card_uid"] for x in cards if self.partition_for(x["group_key"]) != x["partition"]],
            "every card's partition equals the deterministic assignment", "partition was changed by hand")
        groups: dict = {}
        for x in cards:
            groups.setdefault(x["group_key"], set()).add(x["partition"])
        add("Cards sharing a group (same issue / copies) never span partitions", [g for g, p in groups.items() if len(p) > 1],
            "no group is split across train/val/test", "a group spans several partitions")
        add("Every analysis belongs to at most one card",
            [k for k in {i["analysis_id"] for i in imgs} if sum(1 for i in imgs if i["analysis_id"] == k) > 1], "unique", "shared analysis")
        shas: dict = {}
        for i in imgs:
            for s in (i["content_sha"] or "").split("|"):
                if s:
                    shas.setdefault(s, set()).add(i["card_uid"])
        add("No identical photo file appears under two cards", [s[:16] for s, u in shas.items() if len(u) > 1], "no duplicate files", "duplicate photo files")
        certs: dict = {}
        for u, s in subs.items():
            if s["cert"]:
                certs.setdefault(s["cert"], []).append(u)
        add("No certification number is attached to two cards", [k for k, v in certs.items() if len(v) > 1], "unique certs", "duplicate certs")
        late = []
        for u, s in subs.items():
            f = frz.get(u)
            if not (s["grade"] and f):
                continue
            img = next((i for i in imgs if i["analysis_id"] == f["analysis_id"]), None)
            if not img or img["phase"] != "pre_grading" or (s["submitted_on"] and (not img["captured_on"] or img["captured_on"] > s["submitted_on"])):
                late.append(u)
        add("Every frozen primary image is a pre-grading photo captured on or before submission", late, "all primary images precede submission",
            "primary image is not pre-grading or post-dates submission")
        post_link = [i["card_uid"] for i in imgs if i["phase"] == "pre_grading" and subs.get(i["card_uid"], {}).get("result_recorded")
                     and i["linked"] > subs[i["card_uid"]]["result_recorded"]]
        add("No pre-grading photo was linked after the grade was recorded", post_link, "none", "photo linked as pre-grading after the result")
        fv = [u for u, f in frz.items() if self._snapshot_changed(f)]
        add("Frozen feature snapshots are unchanged since the grade was recorded", fv, "hashes match", "snapshot differs from frozen hash")
        banned = [n for n in gem_model.FEATURE_NAMES if re.search(r"grade|psa|cert|outcome|label|partition", n)]
        add("No feature name refers to the outcome, cert, label or partition", banned, "feature names are measurements only")
        post_reviews = 0
        for f in frz.values():
            post_reviews += sum(1 for e in self.store.audit_log(f["analysis_id"]) if e["payload"].get("post_outcome"))
        checks.append({"check": "Reviews made after the grade was known (informational: they cannot change frozen features)",
                       "passed": True, "detail": f"{post_reviews} review actions were flagged post-outcome", "offenders": []})
        if registry is not None:
            checks += registry.provenance_checks()
        return {"passed": all(c["passed"] for c in checks), "checks": checks,
                "not_checkable": ["Whether a photo shows PSA's slab or label is not detectable automatically: it depends on the phase you "
                                  "declare when linking. Declaring a slab photo as pre-grading would defeat the safeguard.",
                                  "Whether the physical card changed between photographing and submitting (handling, sleeving) is not recorded."]}

    def _snapshot_changed(self, f: dict) -> bool:
        with self.store._conn() as c:
            r = c.execute("SELECT snapshot_json FROM gem_snapshots WHERE analysis_id=?", (f["analysis_id"],)).fetchone()
        return r is None or hashlib.sha256(r["snapshot_json"].encode()).hexdigest() != f["snapshot_sha"]

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        cards = self.list_cards()
        rows = self.rows()
        by_part = {p: {"cards": 0, "eligible": 0, "psa10": 0, "grade_le9": 0} for p, _ in SPLIT}
        for c in cards:
            by_part[c["partition"]]["cards"] += 1
        for r in rows:
            b = by_part[r["partition"]]
            b["eligible"] += 1
            b["psa10"] += r["psa_grade"] == 10
            b["grade_le9"] += r["psa_grade"] != 10
        reasons: dict = {}
        for c in cards:
            for w in c["ineligible_reasons"]:
                reasons[w] = reasons.get(w, 0) + 1
        funnel = {"registered": len(cards), "photographed_pre_grading": sum(1 for c in cards if c["n_images"]),
                  "submitted_or_graded": sum(1 for c in cards if c["status"] in ("submitted", "graded")),
                  "graded": sum(1 for c in cards if c["status"] == "graded"),
                  "verified": sum(1 for c in cards if c["verification"] == "verified"), "eligible": len(rows)}
        return {"funnel": funnel, "partitions": by_part, "ineligible_reasons": reasons, "split": dict(SPLIT),
                "grade_counts": {str(g): sum(1 for r in rows if r["psa_grade"] == g) for g in range(1, 11) if any(r["psa_grade"] == g for r in rows)},
                "policy": POLICY, "synthetic_in_dataset": sum(1 for c in cards if c["origin"] == "synthetic_test")}
