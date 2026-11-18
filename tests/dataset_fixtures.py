"""Synthetic cards for exercising the data pipeline. They are flagged origin=synthetic_test and prove plumbing only."""
import numpy as np

from gradeforge.analyze import evaluate_gem
from gradeforge.dataset import Dataset
from gradeforge.storage import Store
from tests.gem_fixtures import finding, result, side

TYPES = ["sports_modern_base", "sports_modern_chrome", "tcg_holo", "sports_vintage"]


def make_analysis(store, i, pct, flaws=0, finish="standard", lights=3):
    fs = [finding("front", "corner_whitening", "corner_tr", "minor", status="confirmed", length_mm=0.5) for _ in range(flaws)]
    r = result(front=side("front", pct=pct, findings=fs, lights=lights), finish=finish)
    r["standards_version"] = None
    r["gem"] = evaluate_gem(r)
    r["assessment"] = {"prediction_status": "ceilings_only", "psa10_eligibility": {"status": "x"}}
    aid = store.save(f"card{i}", r, {})
    store.log_prediction(aid, r)
    return aid, r


def add_card(ds, i, grade, pct=51.0, flaws=0, *, verified=True, captured="2026-01-05", submitted="2026-02-01", card_type=None,
             basis="all_owned_cards", finish="standard", year=None, lights=3, **extra):
    aid, r = make_analysis(ds.store, i, pct, flaws, finish, lights)
    extra.setdefault("group_key", f"fixture-{i}")      # deterministic partition under the fixed test salt
    c = ds.create_card(label=f"c{i}", card_type=card_type or TYPES[i % 4], finish=finish, rights_attested=True,
                       origin="synthetic_test", selection_basis=basis, year=year, **extra)
    uid = c["card_uid"]
    ds.link_image(uid, aid, "pre_grading", captured)
    ds.record_submission(uid, submitted, "Regular")
    ds.record_result(uid, grade, cert=f"{50000000 + i}")
    if verified:
        ds.verify_result(uid, verified_by="tester", cert_number_seen=f"{50000000 + i}", cert_page_grade=grade)
    return uid, aid


def populate(ds, n, learnable=True, seed=0, start=0):
    rng = np.random.default_rng(seed)
    uids = []
    for k in range(n):
        i = start + k
        pct = float(rng.uniform(50, 62)); flaws = int(rng.integers(0, 3))
        score = (56 - pct) * 0.8 - flaws * 1.5 + rng.normal(0, 0.8)
        win = score > 0 if learnable else bool(rng.integers(0, 2))
        grade = 10 if win else (8 if rng.random() < 0.25 else 9)
        uids.append(add_card(ds, i, grade, pct, flaws)[0])
    return uids


def new_ds(tmp_path, synthetic=True, salt="fixture-salt-1"):
    ds = Dataset(Store(tmp_path), allow_synthetic=synthetic)
    with ds.store._conn() as c:                  # fixed salt: tests must be deterministic (production salt is random per store)
        c.execute("UPDATE dataset_meta SET value=? WHERE key='salt'", (salt,))
    return ds
