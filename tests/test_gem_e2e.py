"""End-to-end: real pipeline on synthetic photos -> gem block -> outcome -> reliability -> PDF."""
import pytest
from fastapi.testclient import TestClient

from gradeforge.app import create_app
from . import synth


def jpg(**kw):
    card, mask = synth.make_card(**kw)
    return synth.encode(synth.photograph(card, mask))


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_analyze_returns_gem_block_and_feedback_flow(client):
    r = client.post("/api/analyze", files={"front": ("f.jpg", jpg(), "image/jpeg"), "back": ("b.jpg", jpg(seed=3), "image/jpeg")},
                    data={"label": "e2e", "finish": "chrome"})
    assert r.status_code == 200, r.text
    body = r.json()
    g = body["gem"]
    assert g["card_traits"]["finish"] == "chrome"
    assert g["psa10_eligibility"]["status"] != "ruled_out" or g["grade_ceiling"]["conservative"] < 10
    assert g["dominance"]["offsets_applied"] is False
    # a single photo per side can never yield a confirmed 10 or a made-up probability
    assert g["most_likely"]["grade"] != 10 and g["psa10_eligibility"]["status"] in (
        "ruled_out", "unlikely", "uncertain", "credible_candidate", "insufficient_evidence")
    assert "p_psa10" not in g["reliability"] and g["reliability"].get("model_prediction") is None
    assert g["additional_inspection"], "a single-photo card must come with an inspection plan"
    aid = body["id"]

    pdf = client.get(f"/api/history/{aid}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    o = client.post(f"/api/history/{aid}/outcome", json={"psa_grade": 9, "pre_submission_photos": True, "qualifier": "OC"})
    assert o.status_code == 200 and o.json()["counts_for_training"] is True
    rel = client.get("/api/gem/reliability").json()
    assert rel["n_eligible"] == 1
    assert rel["rules_performance"]["n"] == 1
    assert client.post("/api/gem/model/train").json()["trained"] is False   # far below the training gate
    assert client.get("/api/history").json()["items"][0]["gem_verdict"]
