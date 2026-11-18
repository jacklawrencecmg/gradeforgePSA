"""End-to-end tests of the multi-photo inspection pipeline on SYNTHETIC scenes with known ground truth.
They show the detectors behave as designed on constructed images; they say nothing about accuracy on real cards."""
import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gradeforge.app import create_app
from gradeforge.cv.pipeline import analyze_side
from tests import synth

SC = ((300, 600), (700, 640))


def photos(variants, base=None, mask=None, **kw):
    if base is None:
        base, mask = synth.make_card(lr=(80, 80), tb=(80, 80))
    out = []
    for i, v in enumerate(variants):
        out.append(synth.encode(synth.photograph(synth.lit(base, **v), mask, angle=3 + 2 * i, seed=i, **kw)))
    return out


@pytest.fixture(scope="module")
def scene():
    ps = photos([dict(ramp=(0.1, 0), scratch=SC + (40, 3)),
                 dict(ramp=(-0.1, 0.05), scratch=SC + (-25, 3), dent=(500, 1000, 25, 60, -1)),
                 dict(ramp=(0, 0.1), dent=(500, 1000, 25, 60, 1))])
    res, art = analyze_side(ps[0], "front", ps[1:])
    return ps, res, art


def test_lighting_photos_register_across_different_poses(scene):
    _, res, _ = scene
    reg = res["registration"]["lighting"]
    assert len(reg) == 2 and all(r["ok"] for r in reg)
    assert all(r["residual_shift_px"] < 1.0 and r["cc"] > 0.9 for r in reg)


def test_scratch_and_dent_found_only_through_lighting_differences(scene):
    _, res, _ = scene
    cats = {f["category"]: f for f in res["findings"]}
    assert cats["surface_relief_line"]["classification"] == "physical_probable_relief"
    assert cats["surface_relief_dipole"]["classification"] == "physical_probable_relief"
    k = res["detection"]["canvas"][0] / 1000.0                         # canvas px per synthetic-card px
    x, y, w, h = cats["surface_relief_line"]["box"]
    assert x < 350 * k and x + w > 700 * k and 600 * k < y + h / 2 < 700 * k   # lies along the drawn scratch
    x, y, w, h = cats["surface_relief_dipole"]["box"]
    assert x < 500 * k < x + w and y < 1000 * k < y + h               # around the dent
    assert not [f for f in res["findings"] if f["category"].startswith("corner") or f["category"].startswith("edge")]


def test_every_finding_has_traceable_original_crops(scene):
    _, res, art = scene
    assert res["findings"]
    for f in res["findings"]:
        files = f["evidence"]["files"]
        assert "primary_original" in files and "primary_canvas" in files
        assert any(k.startswith("light_") for k in files)
        for rec in files.values():
            data = art["evidence"][rec["file"]]
            assert hashlib.sha256(data).hexdigest() == rec["sha256"]
        assert f["evidence"]["sources"][0]["sha256"] == res["detection"]["source_sha256"]
        assert f["detector"]["trained"] is False and f["detector"]["validated"] is False


def test_completeness_requires_second_lighting_photo():
    ps = photos([dict()])
    res, _ = analyze_side(ps[0], "front")
    rois = {r["id"]: r for r in res["completeness"]["rois"]}
    assert rois["artwork_relief"]["inspectable"] == "no"
    assert any("another photo" in q or "one more photo" in q for q in rois["artwork_relief"]["requests"])
    assert rois["artwork_print_and_stain"]["inspectable"] == "no"
    assert res["completeness"]["level"] != "complete"


def test_nearly_identical_lighting_is_rejected_as_not_diverse():
    ps = photos([dict(), dict()])
    res, _ = analyze_side(ps[0], "front", ps[1:])
    assert res["registration"]["lighting_stack"]["diversity_ok"] is False
    assert res["surface"]["artwork_relief_assessed"] is False
    assert [r for r in res["completeness"]["rois"] if r["id"] == "artwork_relief"][0]["inspectable"] == "no"


def test_albedo_stain_in_border_is_physical_when_present_in_every_photo():
    base, mask = synth.make_card(lr=(80, 80), tb=(80, 80))
    base = synth.add_stain(base, 40, 700, 25, 25, color=(60, 120, 170), strength=0.9)
    ps = photos([dict(), dict(ramp=(0.05, 0))], base=base, mask=mask)
    res, _ = analyze_side(ps[0], "front", ps[1:])
    st = [f for f in res["findings"] if f["category"] in ("surface_stain", "surface_spot")]
    assert st and st[0]["classification"] == "physical_probable_albedo"
    single, _ = analyze_side(ps[0], "front")
    st1 = [f for f in single["findings"] if f["category"] in ("surface_stain", "surface_spot")]
    assert st1 and all(f["classification"] == "indeterminate_single_image" for f in st1)


def test_glare_is_an_artifact_not_damage():
    base, mask = synth.make_card(lr=(80, 80), tb=(80, 80))
    p0 = synth.encode(synth.photograph(base, mask, glare_blob=(800, 1000, 120, 90)))
    p1 = synth.encode(synth.photograph(synth.lit(base, ramp=(0.1, 0)), mask, angle=5, seed=2))
    res, _ = analyze_side(p0, "front", [p1])
    g = [f for f in res["findings"] if f["category"] == "glare_region"]
    assert g and all(f["classification"] == "artifact_glare" for f in g)
    assert not [f for f in res["findings"] if f["classification"].startswith("physical") and f["category"] == "surface_relief_line"]


def test_subresolution_features_are_not_called_physical():
    # A card photographed far too small: the same 1-px scratch is below what the photo can resolve.
    base, mask = synth.make_card(lr=(80, 80), tb=(80, 80))
    ps = photos([dict(scratch=SC + (40, 1)), dict()], base=base, mask=mask, photo_size=(900, 1200), scale=0.62)
    res, _ = analyze_side(ps[0], "front", ps[1:])
    assert res["resolution"]["min_resolvable_feature_mm"] > 0.05
    for f in res["findings"]:
        if f["resolvability"].get("at_resolution_limit"):
            assert not f["classification"].startswith("physical")


# ---------------- review workflow through the API ----------------

@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path))


def post_scene(client):
    ps = photos([dict(ramp=(0.1, 0), scratch=SC + (40, 3)), dict(ramp=(-0.1, 0.05), scratch=SC + (-25, 3)), dict()])
    back = synth.encode(synth.photograph(synth.make_card(lr=(80, 80), tb=(80, 80))[0], synth.make_card()[1], seed=9))
    files = [("front", ("f.jpg", ps[0], "image/jpeg")), ("back", ("b.jpg", back, "image/jpeg")),
             ("front_light", ("l1.jpg", ps[1], "image/jpeg")), ("front_light", ("l2.jpg", ps[2], "image/jpeg"))]
    r = client.post("/api/analyze", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_review_flow_requires_evidence_and_changes_the_assessment(client):
    body = post_scene(client)
    aid = body["id"]
    f = next(x for x in body["front"]["findings"] if x["category"] == "surface_relief_line")
    url = f"/api/history/{aid}/front/findings/{f['id']}/review"
    # confirming without citing evidence is refused
    assert client.post(url, json={"action": "confirm"}).status_code == 422
    assert client.post(url, json={"action": "confirm", "evidence_keys": ["not_a_crop"]}).status_code == 422
    # rejecting needs a reason
    assert client.post(url, json={"action": "reject"}).status_code == 422
    ev = client.get(f"/api/history/{aid}/evidence/{f['evidence']['files']['primary_original']['file']}")
    assert ev.status_code == 200 and ev.headers["content-type"] == "image/png"
    r = client.post(url, json={"action": "confirm", "reviewer": "JL", "severity": "moderate",
                               "evidence_keys": ["primary_original", "light_light1"], "note": "visible hairline"})
    assert r.status_code == 200, r.text
    out = r.json()
    fc = next(x for x in out["findings"] if x["id"] == f["id"])
    assert fc["status"] == "confirmed" and fc["review"]["reviewer"] == "JL"
    item = next(c for c in out["assessment"]["checklist"] if c["id"] == "front_artwork_relief")
    assert item["status"] == "fail_confirmed" and f["id"] in item["finding_ids"]
    assert out["assessment"]["conservative_ceiling"] <= 8
    # reject flips it back out of the ceiling
    r2 = client.post(url, json={"action": "reject", "note": "dust on the sleeve"}).json()
    item2 = next(c for c in r2["assessment"]["checklist"] if c["id"] == "front_artwork_relief")
    assert item2["status"] != "fail_confirmed"
    log = client.get(f"/api/history/{aid}/audit").json()["items"]
    assert [e["action"] for e in log] == ["confirm", "reject"]
    # persisted
    rec = client.get(f"/api/history/{aid}").json()
    assert next(x for x in rec["result"]["front"]["findings"] if x["id"] == f["id"])["status"] == "rejected"


def test_manual_finding_gets_evidence_cut_from_stored_original(client):
    body = post_scene(client)
    aid = body["id"]
    r = client.post(f"/api/history/{aid}/front/findings",
                    json={"box": [30, 30, 120, 120], "target": "corner_tl", "severity": "moderate",
                          "note": "soft ding visible in the original", "reviewer": "JL"})
    assert r.status_code == 200, r.text
    f = next(x for x in r.json()["findings"] if x["category"] == "manual")
    assert f["status"] == "confirmed" and f["classification"] == "manual_observation"
    assert {"primary_original", "primary_canvas"} <= set(f["evidence"]["files"])
    assert client.get(f"/api/history/{aid}/evidence/{f['evidence']['files']['primary_original']['file']}").status_code == 200
    bad = client.post(f"/api/history/{aid}/front/findings", json={"box": [0, 0, 2, 2], "target": "corner_tl", "note": "x"})
    assert bad.status_code == 422
    assert client.post(f"/api/history/{aid}/front/findings",
                       json={"box": [30, 30, 120, 120], "target": "nope", "note": "x"}).status_code == 422


def test_evidence_endpoint_rejects_path_tricks(client):
    body = post_scene(client)
    aid = body["id"]
    assert client.get(f"/api/history/{aid}/evidence/..%2F..%2Fgradeforge.sqlite3").status_code in (404, 422)
    assert client.get(f"/api/history/{aid}/evidence/front_f001.txt").status_code == 404
    assert client.get(f"/api/history/{aid}/original/front/primary").status_code == 200
    assert client.get(f"/api/history/{aid}/original/front/../../x").status_code in (404, 422)
