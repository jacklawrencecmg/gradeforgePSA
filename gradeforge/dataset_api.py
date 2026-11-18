"""HTTP endpoints for the data-collection and validation pipeline (mounted by app.create_app)."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import evaluation as ev
from . import gem_model
from .dataset import (ANNOTATION_KINDS, CARD_TYPES, FINISHES, IMAGE_SOURCES, PHASES, POLICY, QUALIFIERS, SELECTION_BASES, SPLIT,
                      Dataset, cert_verification_url)
from .registry import MAX_COMFORTABLE_TEST_LOOKS, TEST_MIN_PER_CLASS, VAL_MIN_N, VAL_MIN_PER_CLASS, Registry
from .storage import Store


class NewCard(BaseModel):
    label: str = Field(default="", max_length=120)
    card_type: str
    finish: str = "unknown"
    set_name: str = Field(default="", max_length=80)
    year: int | None = Field(default=None, ge=1800, le=2100)
    subject: str = Field(default="", max_length=80)
    card_number: str = Field(default="", max_length=20)
    variant: str = Field(default="", max_length=60)
    group_key: str = Field(default="", max_length=80)
    selection_basis: str = "mixed_or_unknown"
    rights_attested: bool = False
    image_source: str = "own_photographs"
    owner_note: str = Field(default="", max_length=500)
    actor: str = Field(default="", max_length=60)


class LinkImage(BaseModel):
    analysis_id: str
    phase: str = ""
    captured_on: str | None = None
    note: str = Field(default="", max_length=200)
    actor: str = Field(default="", max_length=60)


class Submission(BaseModel):
    submitted_on: str
    service: str = Field(default="", max_length=40)
    submission_no: str = Field(default="", max_length=30)
    actor: str = Field(default="", max_length=60)


class Result(BaseModel):
    grade: int = Field(ge=1, le=10)
    cert: str = Field(default="", max_length=20)
    qualifier: str = Field(default="", max_length=4)
    grade_date: str | None = None
    primary_analysis_id: str | None = None
    submitted_on: str | None = None
    service: str = Field(default="", max_length=40)
    actor: str = Field(default="", max_length=60)


class Correction(BaseModel):
    grade: int | None = Field(default=None, ge=1, le=10)
    cert: str | None = Field(default=None, max_length=20)
    reason: str = Field(max_length=300)
    actor: str = Field(default="", max_length=60)


class Verify(BaseModel):
    verified_by: str = Field(max_length=60)
    cert_number_seen: str = Field(max_length=20)
    cert_page_grade: int = Field(ge=1, le=10)
    reference: str = Field(default="", max_length=200)


class Exclude(BaseModel):
    reason: str = Field(max_length=200)
    actor: str = Field(default="", max_length=60)


class Annotation(BaseModel):
    card_uid: str
    analysis_id: str
    side: str
    kind: str
    box: list[float] = Field(min_length=4, max_length=4)
    severity: str = "minor"
    annotator: str = Field(max_length=60)
    note: str = Field(default="", max_length=300)


class AnnotationDone(BaseModel):
    analysis_id: str
    side: str
    annotator: str = Field(max_length=60)
    complete: bool = True


class Retrain(BaseModel):
    note: str = Field(default="", max_length=200)
    target_precision: float | None = Field(default=None, gt=0, lt=1)


class TestRun(BaseModel):
    confirm: bool = False


def register(app: FastAPI, store: Store) -> tuple[Dataset, Registry]:
    ds = Dataset(store, allow_synthetic=False)          # production: real physical cards only
    reg = Registry(ds)

    def call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\""))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/dataset/vocab")
    def vocab():
        return {"card_types": CARD_TYPES, "finishes": FINISHES, "selection_bases": SELECTION_BASES, "image_phases": PHASES,
                "image_sources": IMAGE_SOURCES, "qualifiers": QUALIFIERS, "annotation_kinds": sorted(ANNOTATION_KINDS),
                "split": dict(SPLIT), "policy": POLICY,
                "gates": {"train_min_cards": gem_model.MIN_TRAIN_N, "train_min_per_class": gem_model.MIN_PER_CLASS,
                          "validation_min_cards": VAL_MIN_N, "validation_min_per_class": VAL_MIN_PER_CLASS,
                          "test_min_per_class": TEST_MIN_PER_CLASS, "max_comfortable_test_looks": MAX_COMFORTABLE_TEST_LOOKS,
                          "validity": {"min_brier_improvement": gem_model.MIN_BRIER_IMPROVEMENT, "min_auc_lower_bound": gem_model.MIN_AUC_LOWER_BOUND,
                                       "calibration_slope_range": list(gem_model.SLOPE_RANGE)}}}

    @app.post("/api/dataset/cards")
    def new_card(b: NewCard):
        return call(ds.create_card, **b.model_dump())

    @app.get("/api/dataset/cards")
    def cards(partition: str | None = None, status: str | None = None):
        return {"items": ds.list_cards(partition, status)}

    @app.get("/api/dataset/cards/{uid}")
    def card(uid: str):
        c = ds.get_card(uid)
        if c is None:
            raise HTTPException(404, "Unknown card")
        return {**c, "cert_page": cert_verification_url(c["submission"]["cert"]) if c["submission"] and c["submission"]["cert"] else None}

    @app.get("/api/dataset/cards/{uid}/audit")
    def card_audit(uid: str):
        return {"items": ds.audit(uid)}

    @app.post("/api/dataset/cards/{uid}/exclude")
    def exclude(uid: str, b: Exclude):
        call(ds.exclude_card, uid, b.reason, b.actor)
        return {"ok": True}

    @app.post("/api/dataset/cards/{uid}/images")
    def link(uid: str, b: LinkImage):
        return call(ds.link_image, uid, b.analysis_id, b.phase, b.captured_on, b.note, b.actor)

    @app.post("/api/dataset/cards/{uid}/submission")
    def submission(uid: str, b: Submission):
        return call(ds.record_submission, uid, b.submitted_on, b.service, b.submission_no, b.actor)

    @app.post("/api/dataset/cards/{uid}/result")
    def result(uid: str, b: Result):
        return call(ds.record_result, uid, b.grade, cert=b.cert, qualifier=b.qualifier, grade_date=b.grade_date,
                    primary_analysis_id=b.primary_analysis_id, submitted_on=b.submitted_on, service=b.service, actor=b.actor)

    @app.post("/api/dataset/cards/{uid}/correct")
    def correct(uid: str, b: Correction):
        return call(ds.correct_result, uid, grade=b.grade, cert=b.cert, reason=b.reason, actor=b.actor)

    @app.post("/api/dataset/cards/{uid}/verify")
    def verify(uid: str, b: Verify):
        return call(ds.verify_result, uid, verified_by=b.verified_by, cert_number_seen=b.cert_number_seen,
                    cert_page_grade=b.cert_page_grade, reference=b.reference)

    @app.get("/api/dataset/summary")
    def summary():
        return {**ds.summary(), "readiness": reg.readiness(), "champion": reg.champion()}

    @app.get("/api/dataset/snapshots")
    def snapshots():
        return {"items": ds.list_snapshots()}

    @app.post("/api/dataset/snapshots")
    def snapshot(b: Retrain):
        s = ds.create_snapshot(b.note)
        return {k: v for k, v in s.items() if k != "manifest"}

    @app.get("/api/dataset/manifest.csv")
    def manifest():
        names = gem_model.FEATURE_NAMES
        lines = [",".join(["card_uid", "partition", "psa_grade", "qualifier", "card_type", "finish", "quality_tier", "rules_verdict"] + names)]
        for r in ds.rows():
            vals = ["" if r["features"].get(n) is None else repr(float(r["features"][n])) for n in names]
            lines.append(",".join([r["card_uid"], r["partition"], str(r["psa_grade"]), r["qualifier"], r["card_type"], r["finish"],
                                   r["quality_tier"] or "", r["snapshot"]["verdict"]] + vals))
        return Response("\n".join(lines) + "\n", media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="gradeforge_dataset.csv"'})

    # ---- annotation
    @app.post("/api/dataset/annotations")
    def annotate(b: Annotation):
        return call(ds.add_annotation, b.card_uid, b.analysis_id, b.side, b.kind, b.box, b.severity, b.annotator, b.note)

    @app.get("/api/dataset/annotations")
    def annotations(card_uid: str | None = None, analysis_id: str | None = None):
        return {"items": ds.list_annotations(card_uid, analysis_id)}

    @app.delete("/api/dataset/annotations/{ann_id}")
    def withdraw(ann_id: int):
        ds.withdraw_annotation(ann_id)
        return {"ok": True}

    @app.post("/api/dataset/annotations/complete")
    def complete(b: AnnotationDone):
        ds.set_annotation_complete(b.analysis_id, b.side, b.annotator, b.complete)
        return {"ok": True}

    @app.get("/api/dataset/annotations/evaluation")
    def annotation_eval():
        return {**ds.evaluate_detectors(), "annotator_agreement": ds.annotator_agreement()}

    # ---- integrity, models, reports
    @app.get("/api/dataset/leakage-audit")
    def leakage():
        return ds.leakage_audit(registry=reg)

    @app.post("/api/dataset/models/retrain")
    def retrain(b: Retrain):
        return call(reg.retrain, b.note, b.target_precision)

    @app.get("/api/dataset/models")
    def models():
        return {"items": reg.versions(), "champion": reg.champion(), "readiness": reg.readiness()}

    @app.post("/api/dataset/models/rollback")
    def rollback():
        return call(reg.rollback)

    @app.post("/api/dataset/models/{version}/test")
    def test_model(version: str, b: TestRun):
        return call(reg.evaluate_on_test, version, b.confirm)

    @app.post("/api/dataset/models/{version}/promote")
    def promote(version: str):
        return call(reg.promote, version)

    @app.get("/api/dataset/models/{version}/report")
    def model_report(version: str, group_by: str = "card_type"):
        return call(reg.report, version, group_by)

    @app.get("/api/dataset/monitor")
    def monitor(version: str | None = None):
        return call(reg.monitor, version)

    @app.get("/api/dataset/errors")
    def errors():
        return ev.prediction_errors(ds)

    @app.get("/api/dataset/selection-bias")
    def selection():
        return ev.selection_diagnostics(ds)

    @app.get("/api/dataset/rules-report")
    def rules_report(group_by: str = "card_type"):
        rows = ds.rows()
        return {"n": len(rows), "by_group": call(ev.group_report, rows, None, None, group_by),
                "overall": ev.performance(rows, None, None, baseline_rows=ds.rows("train")),
                "note": "Rules-only performance across ALL eligible cards (the rules have no fitted parameters, so no partition is needed)."}

    return ds, reg
