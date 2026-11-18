"""Import REAL, already-graded cards from a CSV plus photo files.

    python -m gradeforge.dataset_import cards.csv --photos ./photos --attest-rights [--data ./data]

Columns (header row required; only the first group is mandatory):
  card_ref, front_image, back_image, card_type, captured_on, submitted_on, psa_grade, cert
  optional: finish, set_name, year, card_number, variant, subject, selection_basis, qualifier, grade_date, service,
            front_light_1..front_light_4, back_light_1..back_light_4,
            verified_by, cert_number_seen, cert_page_grade   (only if a person really looked the cert up on psacard.com)

Rules enforced:
* --attest-rights is required: you took the photos (or have permission) and they were taken BEFORE submission.
* captured_on (the date you photographed the card) is required and must not be after submitted_on.
* Photos are analysed exactly as in the app; PSA outcomes never feed the analysis.
* Rows whose image files are missing, unreadable, or already imported are reported and skipped; nothing is invented.
* Cards without verified_by / cert_page_grade stay UNVERIFIED and will not be used for training until verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .analyze import analyze_card
from .dataset import Dataset
from .storage import Store

REQUIRED = ("card_ref", "front_image", "card_type", "captured_on", "submitted_on", "psa_grade")


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def import_csv(csv_path: Path, photos: Path, store: Store, rights_attested: bool) -> dict:
    if not rights_attested:
        raise SystemExit("Refusing to import: pass --attest-rights to confirm the photographs are yours (or permitted) and pre-date submission.")
    ds = Dataset(store, allow_synthetic=False)
    existing = {c["label"] for c in ds.list_cards()}
    report = {"imported": [], "skipped": [], "unverified": 0}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if rows and any(k not in rows[0] for k in REQUIRED):
        raise SystemExit("Missing required columns: " + ", ".join(k for k in REQUIRED if k not in rows[0]))
    for n, r in enumerate(rows, 2):
        ref = (r["card_ref"] or "").strip()
        skip = lambda why: report["skipped"].append({"line": n, "card_ref": ref, "reason": why})
        if not ref:
            skip("empty card_ref"); continue
        if ref in existing:
            skip("already imported"); continue
        front = _read(photos / r["front_image"])
        back = _read(photos / r["back_image"]) if r.get("back_image") else None
        if front is None:
            skip(f"front image not found: {r['front_image']}"); continue
        try:
            grade = int(r["psa_grade"])
            lights = lambda side: [b for b in (_read(photos / r[f"{side}_light_{i}"]) for i in range(1, 5) if r.get(f"{side}_light_{i}")) if b] or None
            result, arts = analyze_card(front, back, lights("front"), lights("back"), None, None, None,
                                        {"finish": r.get("finish") or "unknown"}, None, None)
            aid = store.save(ref, result, arts)       # no prediction_log entry: this analysis happens after grading, so it is not prospective
            card = ds.create_card(label=ref, card_type=r["card_type"], finish=r.get("finish") or "unknown", set_name=r.get("set_name", ""),
                                  year=int(r["year"]) if r.get("year") else None, subject=r.get("subject", ""), card_number=r.get("card_number", ""),
                                  variant=r.get("variant", ""), selection_basis=r.get("selection_basis") or "mixed_or_unknown",
                                  rights_attested=True, actor="import")
            uid = card["card_uid"]
            ds.link_image(uid, aid, "pre_grading", r["captured_on"], actor="import")
            ds.record_submission(uid, r["submitted_on"], r.get("service", ""), actor="import")
            ds.record_result(uid, grade, cert=r.get("cert", ""), qualifier=r.get("qualifier", ""), grade_date=r.get("grade_date") or None, actor="import")
            if r.get("verified_by") and r.get("cert_page_grade"):
                ds.verify_result(uid, verified_by=r["verified_by"], cert_number_seen=r.get("cert_number_seen") or r.get("cert", ""),
                                 cert_page_grade=int(r["cert_page_grade"]))
            else:
                report["unverified"] += 1
            report["imported"].append({"line": n, "card_ref": ref, "card_uid": uid, "partition": card["partition"]})
        except (ValueError, KeyError) as exc:
            skip(str(exc))
    report["summary"] = ds.summary()["funnel"]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--photos", type=Path, default=Path("."))
    ap.add_argument("--data", type=Path, default=None, help="GradeForge data directory (default ./data)")
    ap.add_argument("--attest-rights", action="store_true")
    a = ap.parse_args(argv)
    rep = import_csv(a.csv, a.photos, Store(a.data), a.attest_rights)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
