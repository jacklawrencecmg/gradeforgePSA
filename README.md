# GradeForge v0.4 — data-validated PSA prediction system (Gem Mint engine on the v0.2 inspection pipeline)

Evidence-based PSA grade **ceilings** from card photographs. A pre-grading aid, not a PSA opinion.

v0.2 replaces the single-photo wear checks with a multi-stage inspection pipeline: card alignment, registration of
several photos of the same card, comparison of evidence under different lighting, defect candidates with annotated
original-photo crops, artifact-vs-physical classification, manual confirmation, and inspection-completeness rules.

## v0.4: data collection and validation infrastructure

**Status: the model is NOT validated.** No real PSA-graded cards have been supplied to this project, so there are no
performance figures anywhere in this repository and none are shown by the app until you record real, verified outcomes.
v0.4 delivers the machinery that makes a valid measurement possible, and refuses to produce a number without it.

### What was built

| # | Requirement | Where |
|---|---|---|
| 1 | One record per physical card (type, finish, set/year/number, why it was submitted, rights attestation) | `dataset.py`, table `cards` |
| 2 | Links from a card to its pre-grading analyses, with explicit phase and capture date | table `card_images` |
| 3 | PSA cert number, grade, qualifier, dates, service, and a human verification step against psacard.com/cert | table `submissions` |
| 4 | Defect annotation tool (box on the rectified photo, kind, severity, annotator) with blind-timing flag and detector evaluation | `dataset.py`, `static/dataset.js` |
| 5 | Train / validation / test partitions: 70/15/15, assigned once by salted hash of the card's group, immutable | `Dataset.partition_for` |
| 6 | Leakage prevention (see below) and an automatic audit | `Dataset.leakage_audit` |
| 7 | Probability calibration (Platt scaling fitted on the validation partition only) | `evaluation.fit_platt` |
| 8 | Prediction error tracking from an append-only log written at analysis time | table `prediction_log`, `evaluation.prediction_errors` |
| 9 | Reports grouped by card type, finish, photo quality, selection basis, decade | `evaluation.group_report` |
| 10 | Model registry: dataset snapshots, versions, one-shot test, promote, roll back, prospective monitoring | `registry.py` |

Metrics tracked everywhere: exact-grade accuracy, within-one-grade accuracy, PSA 10 precision, recall and false-positive
rate (each with a Wilson 95% interval), AUC with bootstrap interval, Brier score against the base rate, calibration
slope/ECE/bins, and a separate **PSA 9 vs PSA 10 only** block (negatives are real 9s). Exact and within-one accuracy are
always shown beside simple baselines (majority grade, always-9, rules ceiling) so you can see whether the system adds anything.

### The rules that keep the measurement honest

* **Only physical cards, only verified grades.** A card counts only if it is graded, the grade was checked by a person against PSA's
  cert page (a mismatch is recorded as a conflict and excluded), the primary photos are marked `pre_grading`, and they were
  captured on or before the submission date. GradeForge never contacts PSA and never marks a grade verified by itself.
* **Slab photos are not substitutes.** Every image link needs an explicit phase. `post_grading_slab` and `other` are stored for
  reference and can never be the primary image or be annotated. Linking a new "pre-grading" photo after the grade is recorded is refused.
  GradeForge cannot detect a slab in a photo; the safeguard is the phase you declare, so declare it honestly.
* **Partition at card level, before any fitting.** Copies of the same issue (set + year + number) share a group and always land in
  the same partition. Partitions are never edited; the audit recomputes them.
* **Frozen features.** Recording the result freezes the Gem Mint feature snapshot of the primary analysis. Later reviews can
  change the displayed analysis but not the training data (they are flagged `post_outcome` in the audit log).
* **Fit on train only.** Imputation medians, scaling and weights come from train rows. Validation chooses the regularization,
  the calibration and the decision threshold (so validation numbers are labeled optimistic). The test partition is used once per
  model version, only if it has at least 20 PSA 10s and 20 others (otherwise it stays untouched), and every look is logged.
* **Duplicates.** The same photo file cannot belong to two cards, the same analysis cannot be linked twice, a cert number cannot repeat.
* **Pre-declared validity criteria** (evaluated on the test partition): Brier score better than the base rate by at least 0.01;
  AUC 95% lower bound at least 0.60; calibration slope not demonstrably outside 0.7-1.3 (the 95% interval must overlap that
  band; a point-estimate band would reject sound models by chance at these sample sizes). Failing models are never used.
* **No model without the gates**: at least 300 train cards (40 per class), 100 validation cards (20 per class), 20 per class in test.
* **A model can never raise a grade above the rules' ceiling.** A severe defect still caps the result, even for a positive model.
* **Synthetic data is fenced off.** Test fixtures use `origin = synthetic_test`; the production dataset rejects them, and a model
  trained on them is marked synthetic and cannot be promoted.

### Selection bias and photo quality

You can only learn from cards that were photographed, submitted and graded. GradeForge reports, but cannot remove, the resulting bias:
the collection funnel; the declared reason each card was submitted (`all_owned_cards` vs `prescreened_by_owner`) with PSA 10 rate per
reason; photographed-but-never-submitted cards compared with graded cards on measured features; PSA 10 rate and accuracy by photo-quality
tier; and how many graded cards the pipeline abstained on. Accuracy applies to cards like the ones you submit, photographed like yours.
No inverse-probability weighting is applied because the probability of submission is unknown. **To get unbiased evidence, submit a
random or all-inclusive set of cards and record every one.**

### How to add real verified outcomes

1. Photograph each card **before** sending it to PSA (diffuse light, plus 1-3 extra lighting photos per side; see the photography guide above).
2. Analyze it in the app, then open **Dataset & validation > Cards**, register the card (declare why you are submitting it), link the
   analysis as `pre_grading` with the date you photographed it, and record the submission date.
3. Optionally annotate defects (**Annotate**) before you know the grade; mark a side completely annotated only if you checked all of it.
4. When PSA returns it, record the grade and cert number, open `https://www.psacard.com/cert/<number>`, and enter what the page shows
   in **Record verification**. Only then does the card become eligible.
5. Already-graded cards with pre-submission photos can be loaded in bulk:
   `python -m gradeforge.dataset_import cards.csv --photos ./photos --attest-rights` (column list in the module docstring). Rows
   without `verified_by`/`cert_page_grade` stay unverified until you verify them in the app.
6. Watch **Overview** for the readiness gates. When they are met: **Models > Train a candidate**, then **Run test (once)**, then
   **Promote** only if it passed. **Check newer cards** scores later submissions with the model exactly as it was trained.
7. Retrain as data grows. A challenger is promoted only if it passes the test and is not worse than the champion on the same
   untouched test cards; **Roll back** restores the previous champion.

Inside the app, the champion (if any) is used by the Gem Mint engine, which still shows rules-based reasons, ceilings and inspection
plans; with a validated model it adds a probability together with the observed PSA 10 rate among held-out test predictions in the
same range. The v0.3 quick-feedback loop (Accuracy & feedback tab, `/api/gem/*`) remains for exploration; it has no partitions, so
the registry champion takes precedence when one exists.

### Additional API

```
GET  /api/dataset/vocab | summary | snapshots | manifest.csv | leakage-audit | selection-bias | errors | monitor
POST /api/dataset/cards                       GET /api/dataset/cards[/{uid}]       GET /api/dataset/cards/{uid}/audit
POST /api/dataset/cards/{uid}/images | submission | result | correct | verify | exclude
POST /api/dataset/annotations                 GET /api/dataset/annotations         DELETE /api/dataset/annotations/{id}
POST /api/dataset/annotations/complete        GET /api/dataset/annotations/evaluation
POST /api/dataset/models/retrain              GET /api/dataset/models
POST /api/dataset/models/{version}/test       (body {"confirm": true}; once per version)
POST /api/dataset/models/{version}/promote    POST /api/dataset/models/rollback
GET  /api/dataset/models/{version}/report?group_by=card_type|finish|quality_tier|selection_basis|decade
GET  /api/dataset/rules-report?group_by=...
```

### Database changes (created automatically on first start; existing data is untouched)

`dataset_meta` (random partition salt), `cards`, `card_images`, `submissions` (unique cert), `frozen`, `annotations`,
`annotation_status`, `dataset_snapshots`, `dataset_audit`, `model_versions`, `test_looks`, `prediction_log`. Model files live in
`data/models/versions/`; the active model is `data/models/champion.json`.

### What was and was not tested

The new test files (`test_dataset.py`, `test_evaluation.py`, `test_registry.py`, `test_dataset_api.py`) use **synthetic** cards, flagged
`synthetic_test`. They prove: the partition rules, leakage checks (including that tampering is detected), eligibility rules, one-shot test
semantics, promotion/rollback, calibration arithmetic, metric formulas, and that random labels cannot pass validation. They prove nothing
about accuracy on real cards. **Until you record real verified outcomes, treat every grade-likelihood output as a rules-based, uncalibrated
assumption.**

## v0.3: Gem Mint analysis engine

`gradeforge/gem.py` answers one question: is this a credible PSA 10 candidate, or more likely a 9? It runs after the
inspection pipeline and returns a `gem` block (also in the UI, PDF and `/api/analyze`).

**Output**: PSA 10 eligibility (`ruled_out`, `unlikely`, `uncertain`, `credible_candidate`, `insufficient_evidence`);
most likely grade (only when the evidence is sufficient, otherwise `null` with the reason); specific observable
reasons for a 9; evidence keeping a 10 plausible; an exact inspection plan (which photo, which physical check); the
defects photo limits could hide; every characteristic with its status; and a reliability statement.

**Characteristics evaluated**: front centering (55/45 vs 50/50 zones, with measurement uncertainty), reverse centering,
four corners, four edges, hairline/linear marks, dents, surface finish and gloss, stains, manufacturing/print defects,
chrome/refractor print lines, eye appeal. Print lines, finish irregularities, stains, print defects and eye appeal
always need a person: photos never clear them.

**Dominance rule**: the grade ceiling is the minimum over all characteristics. No score exists that centering can raise;
`dominance.offsets_applied` is always `false` and is tested.

### Official PSA wording vs GradeForge assumptions

Every statement carries a source tag: `official` (PSA's own wording, read from psacard.com/gradingstandards on
2026-11-03), `secondary` (third-party reports of PSA's standards), or `model` (a GradeForge assumption, uncalibrated).

| Item | Source |
|---|---|
| PSA 10 front centering approximately 55/45 to 60/40, back 75/25 | official |
| PSA 9 front 60/40 (65/35 with good eye appeal); 5% front leeway for grades 7+ | official |
| Four sharp corners, full original gloss, free of staining, eye appeal is a judgment | official |
| "One minor flaw" for a PSA 9 | secondary |
| Two or more flaw sites cap at 8; visibility thresholds; every defect-to-cap mapping | model |
| 55/45 treated as nominal, 55-60 as grader-discretion band | model |

The database (`standards_db/psa_standards_v1.1.0.json`) is marked *partially* verified: only the listed items were read
from the official page. Re-check them periodically; PSA can change its standards.

### Trained model and the feedback loop

Rules come first. A trained model is only used once real outcomes justify it.

1. Photograph the card **before** submitting it. Review defect candidates before you know the grade.
2. After PSA returns it, record the result (UI: "Record the actual PSA result", or `POST /api/history/{id}/outcome`) with
   qualifier, submission date, service, cert, and whether the photos preceded submission. Only outcomes with
   `pre_submission_photos: true` and not excluded count. The features and verdict are frozen at that moment, so later
   re-reviews cannot leak the answer into evaluation.
3. `GET /api/gem/reliability` (UI tab "Accuracy & feedback") shows rules verdicts vs real grades with Wilson intervals,
   precision/recall for the PSA 10 shortlist, PSA 10s wrongly excluded, and small-sample warnings. Rules are never
   changed automatically; you change them by hand after reading this.
4. `POST /api/gem/model/train` fits an L2 logistic model (PSA 10 vs 9 or lower) on frozen features. It refuses below 300
   eligible outcomes with at least 40 per class. It is marked validated only if repeated cross-validation meets
   criteria declared in advance: Brier better than the base rate by at least 0.01, AUC lower 95% bound at least 0.60,
   calibration slope 0.7-1.3. An unvalidated model never produces a prediction. A validated one reports the
   cross-validated observed PSA 10 rate in the same score range, not a bare number.

No probability is shown without that validation. No output is a guaranteed PSA grade.

### Integrating into an existing GradeForge

1. Copy `gem.py`, `gem_model.py`, `standards_db/psa_standards_v1.1.0.json` and the changed `standards.py`, `analyze.py`,
   `storage.py`, `app.py`, `report.py`, `validation.py`, `static/index.html`.
2. `analyze_card(...)` now also takes `traits` (finish), `model` and `reliability` and returns `result["gem"]`.
   `reassess(result)` recomputes it after review actions. Library use: `from gradeforge.gem import analyze_gem`.
3. New endpoints: `GET /api/gem/reliability`, `GET /api/gem/model`, `POST /api/gem/model/train`,
   `GET /api/gem/training.csv`; `/api/analyze` accepts `finish` (standard, chrome, refractor, foil_holo, matte, unknown);
   the outcome body gained qualifier, pre_submission_photos, submitted_on, service, cert, excluded, exclude_reason.
4. Existing SQLite files gain the `outcome_meta` and `gem_snapshots` tables on first start.

### What has and has not been tested

The tests (including `test_gem.py`, `test_gem_model.py`, `test_gem_e2e.py`) use synthetic cards and hand-built findings.
They verify the rules logic, the dominance guarantee, the training gate, leakage controls and the API plumbing.
**They say nothing about accuracy on real cards.** No real PSA-graded data was available; the model is disabled until you
supply it, and the rule thresholds remain uncalibrated assumptions.


## Install and run

Requires Python 3.10+ (developed on 3.13 with OpenCV 5.0 and numpy 2.5; OpenCV 4.9+ should also work).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn gradeforge.app:app --port 8000      # then open http://localhost:8000
python -m pytest tests -q                              # 149 tests, about 10 minutes (the registry tests build 900 synthetic cards) (many analyze full-size images)
```

No GPU, no model downloads, no network access are needed. Phones on the same network can use `--host 0.0.0.0`.
Data is stored under `./data` (SQLite, rectified images, evidence crops, original uploads).

### How to photograph a card for the best results

1. **Primary photo**: card at least 1200 px wide, square-on, plain contrasting background, diffuse light.
2. **Extra lighting photos of the same side** (1–4): same card, same side, light from clearly *different directions*
   (for example left, right, and a low raking angle). They do not need to be perfectly aligned; they are registered.
   This is the only way the tool can look at scratches, dents, creases and gloss in the artwork.
3. **Macro photos** (optional): close-ups of suspect corners/edges/areas. They are registered to the card and shown
   next to the finding they cover. They are **not scored automatically**.
4. Dark or saturated borders allow whitening checks. On light borders whitening is reported as not measurable.

## The pipeline

| Stage | What it does | Where |
|---|---|---|
| Preprocessing and alignment | Background-difference card mask, line-fitted quad, perspective rectification (63.5 x 88.9 mm), canvas up to 2000 px wide; the original-to-canvas transform is kept | `cv/detect.py` |
| Centering | Border-to-artwork transitions per side, worst-axis percent, uncertainty from scan-line spread | `cv/centering.py` |
| Registration | ECC (affine, on gradient images) of extra lighting photos with a phase-correlation residual check; ORB + RANSAC for macros. Poor registrations are rejected, not used | `cv/register.py` |
| Lighting comparison | Flat-fielded luminance stack, per-photo validity masks (glare excluded with a margin), lighting-difference map, noise estimate, a diversity gate (photos too alike are not used for relief) | `cv/lighting.py` |
| Corners | Robust circle fit (radius, outline RMS, notch depth, edge-spread softness, missing area vs the other corners, whitening fraction/depth/fragments) | `cv/corners.py` |
| Edges | Whitening runs / exposed-cardstock-like bands, rough-cut and notch measurements relative to the card's other edges, edge-layer intensity anomalies | `cv/edges.py` |
| Border surface | Luminance/chroma residuals, multi-scale ridge (line) detection, spots and stains, per-photo presence | `cv/surface.py`, `cv/ridge.py` |
| Artwork relief | Computed only from differences between registered lighting photos: relief lines, dipoles (dent-like lobe pairs), gloss patches, periodic banding. Edges that look the same in every photo are masked | `cv/surface.py` |
| Artifact vs physical | Rules over per-photo evidence (below) | `cv/artifacts.py` |
| Evidence | Every finding gets crops of the **original photo pixels** (native resolution), the rectified crop, an annotated crop, each lighting photo's crop, the lighting-difference map, and any registered macro. Each file has a SHA-256, and the source photos' hashes and the transform are recorded | `cv/evidence.py` |
| Completeness | Per region: inspectable yes / partial / no, with reasons and the specific photo to add | `cv/completeness.py` |

### Artifact vs physical rules

* Fewer than two usable photos at a location: `indeterminate_single_image`. Nothing is called physical from one photo.
* Same sign and similar strength in every valid photo: `physical_probable_albedo` (ink-like, stain-like).
* Present in all photos but changing with light, or found from the lighting difference: `physical_probable_relief`.
* Present in one photo only, near clipped highlights or as a broad soft patch: `artifact_glare`; thin and not near glare: `indeterminate_angle_dependent` (could be a raking-light scratch, stays unresolved).
* Differential candidates lying on strong artwork edges where alignment is imperfect: `artifact_registration`.
* Anything at or below ~2 source pixels wide: `indeterminate_resolution`. **The tool never claims to see scratches or dents it cannot resolve**; the report states the smallest resolvable width for each photo.

### Manual confirmation and correction

Every finding starts as a *candidate*. In the defect map you can **confirm** (must cite at least one of that finding's
evidence crops; severity can be corrected), **reject** (must give a reason), **reset**, or **mark a region the detectors
missed** (the crops are cut from the stored original photo). Only confirmed findings count as confirmed defects. All
actions are written to an audit log (`/api/history/{id}/audit`) and the grade ceilings are recomputed after each one.

### Defect map (UI)

Interactive zoom/pan viewer with finding markers colored by status/classification, filters (physical-probable,
indeterminate, artifact, rejected, centering lines), a findings list, and a detail panel showing the measurements,
classification reasons, per-photo presence, resolvability, detector provenance, and the evidence crops. The same
findings and crops are included in the PDF report.

## Detector provenance — important

* **All detectors are hand-written classical image processing (OpenCV/numpy). No trained or pretrained defect model is
  used**, and nothing here is presented as one. Each finding records `trained: false, validated: false`.
* No generic "enhancement" is used to create findings. The lighting-difference map is a measurement of how pixels change
  between photos, not an enhanced image.
* All thresholds and all defect-to-grade caps (`engine.FINDING_CAPS`) are **uncalibrated GradeForge assumptions**,
  kept separate from PSA's published standards in `standards_db/`.

## Honesty rules

- **PSA standards vs our model.** `standards_db/psa_standards_v1.0.0.json` holds PSA's published criteria (centering
  limits per grade). The official page could not be read automatically, so those values come from secondary sources and
  are flagged `official_page_verified: false`. Check them at https://www.psacard.com/gradingstandards, edit the JSON,
  set the flag, and bump the version.
- **Ceilings, not guesses.** Conservative and optimistic *upper bounds*. `most_likely` stays `null` until a calibrated
  model exists. A 10 can never be confirmed from photos.
- **No probabilities.** Confidence is inspection coverage.
- **Abstention.** Missing back, low resolution, unmeasurable front centering → "insufficient evidence" plus specific requests.
- **Always unassessed:** print lines, staining, foil damage inside the artwork, and authenticity/alteration. They cannot be
  separated from artwork content in photos; a person must judge them (use macros and the manual-confirmation flow).

## Validation

**Nothing here has been validated on real PSA-graded cards.** The tests use synthetic cards and synthetic lighting scenes
with known ground truth. They show the code measures what it claims on constructed images — they say nothing about
accuracy on real cards, real glossy surfaces, sleeves, or real lighting.

To measure real accuracy, build a CSV with photos taken **before** submission and the grade PSA returned:

```
card_id,front_image,back_image,psa_grade
c001,c001_front.jpg,c001_back.jpg,9
```

```bash
python -m gradeforge.validation labeled.csv --consistency
```

Reports abstention rate, ceiling violations, exact and within-one accuracy of the conservative baseline, PSA 10
screening precision/recall, false exclusions, and stability under perturbation. `can_calibrate` refuses to fit a model
below 300 labeled cards (≥40 PSA 10s and ≥40 non-10s). The CSV runner uses single photos; to evaluate the
multi-photo path, extend it with lighting columns before drawing conclusions about the relief detectors.

## API additions in v0.2

```
POST /api/analyze                      form fields: front, back, front_light[], back_light[], front_macro[], back_macro[], label
GET  /api/history/{id}/evidence/{file} PNG crop for a finding
GET  /api/history/{id}/original/{side}/{role}   the stored original upload (role: primary, light1.., macro1..)
POST /api/history/{id}/{side}/findings/{fid}/review   {action: confirm|reject|reset, reviewer, note, evidence_keys[], severity}
POST /api/history/{id}/{side}/findings                {box, target, severity, note, reviewer}  manual observation
GET  /api/history/{id}/audit           review audit log
```

## Layout

```
gradeforge/standards.py, standards_db/   versioned PSA criteria
gradeforge/cv/                            detect, centering, register, lighting, corners, edges, ridge, surface,
                                          artifacts, completeness, evidence, quality, pipeline
gradeforge/findings.py                    finding schema, categories, classes, provenance
gradeforge/engine.py                      findings-driven ceilings, PSA 10 status, 9-vs-10 checklist
gradeforge/validation.py                  metrics, consistency, calibration gate
gradeforge/app.py, storage.py, report.py  API, SQLite + evidence store + audit log, PDF
gradeforge/static/index.html              responsive UI with defect map
tests/                                    synthetic fixtures; unit and end-to-end tests
```

## Known technical limitations

- Synthetic-only testing (see Validation). Real cards have glossy foil, holo patterns, sleeves/slabs, and lighting that
  the synthetic model does not reproduce; expect false positives and misses until thresholds are tuned on labeled photos.
- Relief detection needs real lighting variety. If the photos look too alike the tool says so and skips relief rather than
  reporting noise. Strong artwork edges and a 3 mm margin around glare are not judged for relief, so a scratch lying on
  such an edge can be missed.
- Registration is affine; strong perspective differences between photos, or a card that bent between shots, degrade it.
  Poor registrations are rejected and listed.
- Centering needs a distinct, roughly uniform border; full-art and borderless cards are reported unmeasurable.
- Whitening is judged against the border color; it is only meaningful on dark or saturated borders.
- Corner "softness"/rounding are compared against the other corners of the same card; die-cut variance between sheets
  is not modeled.
- Edge layer anomalies are intensity irregularities only; a flat photo cannot show peeling or delamination, so they are
  never classed as physical without a person's confirmation.
- Detection assumes a contrasting plain background and rotation under about 40 degrees.
- The UI was exercised in headless Chromium on synthetic scenes (render, select, evidence crops, confirm-needs-evidence
  refusal, confirm). It has not been tested on phones or other browsers.


## v0.4.3 – plainer results and photos by category
- Results open with a plain-English summary: best possible grade, confirmed problems, flags waiting for your check, and what to do next.
- The lowest-case grade (every unconfirmed flag treated as real damage) is no longer the headline; it is shown as a floor only if you have unreviewed flags.
- Photos upload under named headers (Whole card, Angled-light photos, Corner close-ups, Edge close-ups, Surface close-ups) per side; the results list every photo, whether it was used, and what is still missing.
- Detector thresholds are still uncalibrated against real PSA outcomes.


## v0.5.0 - accounts, Stripe subscription and free-grade limit
See PAYWALL_SETUP.md. Off by default (set GRADEFORGE_PAYWALL=1). 28 new tests in tests/test_paywall.py.
