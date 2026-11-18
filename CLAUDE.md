# GradeForge: project guide for Claude Code

GradeForge is a PSA card-grading **pre-assessment** web app (FastAPI + OpenCV + SQLite, one static HTML page). Owner: Jack Lawrence.
Run: `pip install -r requirements.txt && python -m uvicorn gradeforge.app:app --port 8000`. Tests: `python -m pytest -q` (149 tests, ~7 minutes; run in the background, and use `-x -q tests/test_paywall.py` for quick loops).

## Rules that must not be broken
- Never invent flaws, certainty, probabilities, datasets, metrics or validation results. The model is **unvalidated** until real verified PSA outcomes exist; no percentages are shown and no grade is guaranteed.
- Top priority is measurable accuracy separating PSA 9 from PSA 10. The model may never raise a grade above the evidence ceiling (`gem.py` dominance rule).
- Do not claim code was run or tested unless it was. Stripe code has only ever been tested against a fake HTTP layer and locally signed webhooks, never real Stripe.
- Do not use post-grading slab photos as substitutes for pre-grading photos (dataset rules in `dataset.py`).

## Map
- `cv/` detection, rectification, registration of angled-light and close-up photos; `analyze.py` orchestrates; `engine.py` finding states + `FINDING_CAPS` (uncalibrated assumptions); `gem.py` Gem Mint (PSA 10) eligibility, ceilings, inspection plan; `plain.py` plain-English summary (headline never shows the worst-case floor as the finding); `shots.py` photo categories, close-up slot checks, photo inventory.
- Data validation pipeline: `dataset.py`, `registry.py`, `evaluation.py`, `dataset_api.py`, `dataset_import.py` (partitions by salted group hash, leakage audit, frozen snapshots, one-shot test set, calibration, model versions).
- Business layer (only active with `GRADEFORGE_PAYWALL=1`): `config.py` (env settings; `gradeforge.env` loader), `accounts.py` (users, sessions, per-IP free grade, credits, ownership, admin stats), `billing.py` (Stripe Checkout subscription $6.99/mo and pay-per-analysis credits, webhook verification), wiring in `app.py` (`guard` middleware, `/api/me`, `/api/auth/*`, `/api/billing/*`, `/api/admin/*`).
- UI: `static/index.html` (+ `static/dataset.js`). Read `PAYWALL_SETUP.md` for deployment and its limits.

## Known gaps / next work (in priority order)
1. **Accuracy**: the detectors are uncalibrated. A card PSA graded 9 was reported with a lowest-case 8 because unconfirmed flags count as damage in the conservative ceiling. The headline now shows the range, but the real fix needs Jack's real photos plus PSA results: tune thresholds on a training partition only, then evaluate once on the untouched test partition.
2. Stripe has not been run against test-mode Stripe. Do that first (test card 4242 4242 4242 4242).
3. Not built: password reset, email verification, terms/privacy/refund pages, sales tax, a real GeoIP source (country comes from host headers only).
4. The per-grade price (`GRADEFORGE_PER_GRADE_CENTS`, default 199) is a placeholder.
5. Windows package (embeddable Python 3.12 + win_amd64 wheels) is untested on a real Windows machine.

## Conventions
- Edit with exact-match replacements and check each applied once; a blind string replace corrupted `gem.py` once.
- Secrets live only in `gradeforge.env` / environment variables; never commit them. `data/` holds user data and is never packaged.
- Keep `README.md` test counts and `PAYWALL_SETUP.md` in sync with changes.
