"""API and import flows through the REAL image pipeline on synthetic photographs (plumbing only)."""
import csv

import pytest
from fastapi.testclient import TestClient

from gradeforge.app import create_app
from gradeforge.dataset_import import import_csv
from gradeforge.storage import Store
from . import synth


def jpg(**kw):
    card, mask = synth.make_card(**kw)
    return synth.encode(synth.photograph(card, mask))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(tmp_path_factory.mktemp("api")))


def test_end_to_end_collection_flow_through_the_api(client):
    assert client.get("/api/dataset/vocab").json()["gates"]["test_min_per_class"] == 20
    bad = client.post("/api/dataset/cards", json={"card_type": "tcg_holo"})
    assert bad.status_code == 422 and "rights_attested" in bad.json()["detail"]
    assert client.post("/api/dataset/cards", json={"card_type": "tcg_holo", "rights_attested": True, "origin": "synthetic_test"}).status_code in (200, 422)
    card = client.post("/api/dataset/cards", json={"card_type": "sports_modern_chrome", "finish": "chrome", "rights_attested": True,
                                                   "selection_basis": "all_owned_cards", "label": "api card"}).json()
    uid = card["card_uid"]
    assert card["origin"] == "real_physical_card"
    r = client.post("/api/analyze", files={"front": ("f.jpg", jpg(), "image/jpeg"), "back": ("b.jpg", jpg(seed=3), "image/jpeg")}, data={"label": "api"})
    aid = r.json()["id"]
    assert client.post(f"/api/dataset/cards/{uid}/images", json={"analysis_id": aid}).status_code == 422          # phase is mandatory
    ok = client.post(f"/api/dataset/cards/{uid}/images", json={"analysis_id": aid, "phase": "pre_grading", "captured_on": "2026-01-10"})
    assert ok.status_code == 200
    assert client.post(f"/api/dataset/cards/{uid}/submission", json={"submitted_on": "2026-02-01", "service": "Regular"}).status_code == 200
    res = client.post(f"/api/dataset/cards/{uid}/result", json={"grade": 9, "cert": "98765432"})
    assert res.status_code == 200 and res.json()["frozen"]["analysis_id"] == aid
    got = client.get(f"/api/dataset/cards/{uid}").json()
    assert got["eligible_for_training"] is False and any("not verified" in x for x in got["ineligible_reasons"])
    assert got["cert_page"].endswith("/98765432")
    v = client.post(f"/api/dataset/cards/{uid}/verify", json={"verified_by": "tester", "cert_number_seen": "98765432", "cert_page_grade": 9})
    assert v.json()["submission"]["verification_status"] == "verified" and v.json()["eligible_for_training"] is True
    summ = client.get("/api/dataset/summary").json()
    assert summ["funnel"]["eligible"] == 1 and summ["readiness"]["ready_to_train"] is False and summ["champion"] is None
    # annotation, blindness, evaluation
    a = client.post("/api/dataset/annotations", json={"card_uid": uid, "analysis_id": aid, "side": "front", "kind": "corner_whitening",
                                                      "box": [10, 10, 30, 30], "annotator": "jl"})
    assert a.status_code == 200 and a.json()["blind"] is False                     # result was already recorded
    assert client.get("/api/dataset/annotations/evaluation").json()["n_not_blind_excluded"] == 1
    # model lifecycle guards
    rt = client.post("/api/dataset/models/retrain", json={}).json()
    assert rt["trained"] is False and "Not enough" in rt["reason"]
    assert client.post("/api/dataset/models/nope/test", json={"confirm": True}).status_code == 404
    assert client.post("/api/dataset/models/rollback").status_code == 422
    assert client.get("/api/dataset/leakage-audit").json()["passed"] is True
    assert client.get("/api/dataset/selection-bias").json()["funnel"]["graded"] == 1
    assert client.get("/api/dataset/errors").json()["n"] == 1                        # the prediction made at analysis time was logged
    csvtext = client.get("/api/dataset/manifest.csv").text
    assert csvtext.splitlines()[0].startswith("card_uid,partition,psa_grade") and len(csvtext.splitlines()) == 2
    assert client.get("/api/dataset/rules-report?group_by=card_type").json()["n"] == 1
    assert client.get("/api/dataset/rules-report?group_by=bogus").status_code == 422
    assert client.get("/").status_code == 200 and "dataset.js" in client.get("/").text
    assert client.get("/static/dataset.js").status_code == 200


def test_csv_import_requires_attestation_skips_bad_rows_and_never_verifies_on_its_own(tmp_path):
    photos = tmp_path / "photos"; photos.mkdir()
    for n in ("a", "b"):
        (photos / f"{n}_f.jpg").write_bytes(jpg(seed=1 if n == "a" else 7))
        (photos / f"{n}_b.jpg").write_bytes(jpg(seed=2 if n == "a" else 8))
    base = {"card_type": "tcg_standard", "captured_on": "2026-01-02", "submitted_on": "2026-02-02", "cert": ""}
    rows = [dict(base, card_ref="a", front_image="a_f.jpg", back_image="a_b.jpg", psa_grade="10", cert="11111111", verified_by="jl", cert_page_grade="10"),
            dict(base, card_ref="b", front_image="b_f.jpg", back_image="b_b.jpg", psa_grade="9", cert="22222222"),
            dict(base, card_ref="c", front_image="missing.jpg", back_image="", psa_grade="9"),
            dict(base, card_ref="d", front_image="a_f.jpg", back_image="a_b.jpg", psa_grade="9", captured_on="not-a-date")]
    path = tmp_path / "cards.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit):
        import_csv(path, photos, Store(tmp_path / "d1"), rights_attested=False)
    rep = import_csv(path, photos, Store(tmp_path / "d1"), rights_attested=True)
    assert [x["card_ref"] for x in rep["imported"]] == ["a", "b"]
    assert {x["card_ref"] for x in rep["skipped"]} == {"c", "d"} and rep["unverified"] == 1
    assert rep["summary"]["eligible"] == 1 and rep["summary"]["verified"] == 1          # only the row with a human cert check
    again = import_csv(path, photos, Store(tmp_path / "d1"), rights_attested=True)
    assert not again["imported"] and any(s["reason"] == "already imported" for s in again["skipped"])
