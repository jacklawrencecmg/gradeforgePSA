"""Accounts, free-grade metering, Stripe plumbing (with a fake Stripe HTTP layer and locally signed webhooks) and data isolation.

Nothing here talks to real Stripe: it proves our side of the protocol, not Stripe's behaviour.
"""
import hashlib
import hmac
import json
import time
import urllib.parse

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gradeforge import billing
from gradeforge.app import create_app
from gradeforge.config import Settings
from . import synth

WHSEC = "whsec_test"


def jpg(seed=0):
    card, mask = synth.make_card(seed=seed)
    return synth.encode(synth.photograph(card, mask))


@pytest.fixture(scope="module")
def photo():
    return jpg()


class FakeStripe:
    def __init__(self):
        self.calls = []
        self.session = {}

    def http(self, method, url, headers, body):
        form = dict(urllib.parse.parse_qsl(body.decode())) if body else {}
        self.calls.append((method, url, form, headers))
        if url.endswith("/checkout/sessions") and method == "POST":
            return 200, {"id": "cs_1", "url": "https://checkout.stripe.test/c/cs_1"}
        if "/checkout/sessions/" in url:
            return 200, self.session
        if url.endswith("/billing_portal/sessions"):
            return 200, {"url": "https://billing.stripe.test/p/1"}
        return 404, {"error": {"message": "nope"}}


def make(tmp_path, **kw):
    fake = FakeStripe()
    cfg = Settings(paywall=True, stripe_secret="sk_test_x", stripe_webhook_secret=WHSEC, admin_emails={"admin@x.com"}, **kw)
    app = create_app(tmp_path, cfg, billing.Stripe("sk_test_x", "", 699, "usd", http=fake.http))
    return app, fake


def client(app):
    return TestClient(app, base_url="http://testserver")


def signup(c, email="a@x.com", pw="password123"):
    r = c.post("/api/auth/signup", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return r


def sign_event(payload: dict, ts=None, secret=WHSEC):
    raw = json.dumps(payload).encode()
    ts = int(ts or time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
    return raw, f"t={ts},v1={sig}"


def analyze(c, photo):
    return c.post("/api/analyze", files={"front": ("f.jpg", photo, "image/jpeg")})


def webhook_activate(c, app, email):
    """Drive subscription state through the real webhook endpoint."""
    uid = _uid(app, email)
    evt = {"id": f"evt_{email}{time.time()}", "type": "customer.subscription.created",
           "data": {"object": {"id": "sub_1", "customer": "cus_1", "status": "active", "current_period_end": time.time() + 30 * 86400,
                               "metadata": {"user_id": uid}}}}
    raw, sig = sign_event(evt)
    r = c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": sig})
    assert r.status_code == 200, r.text


def _uid(app, email):
    import sqlite3
    store_dir = app.state.dir
    row = sqlite3.connect(store_dir / "accounts.sqlite3").execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    return row[0]


def make_app(tmp_path, **kw):
    app, fake = make(tmp_path, **kw)
    app.state.dir = tmp_path
    return app, fake


# ------------------------------------------------------------------ free grade
def test_one_free_grade_then_blocked_by_ip_even_with_new_browser(tmp_path, photo):
    app, _ = make_app(tmp_path)
    c = client(app)
    assert c.get("/api/me").json()["free_available"] is True
    assert analyze(c, photo).status_code == 200
    r = analyze(c, photo)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "free_used"
    assert r.json()["detail"]["price"] == "$6.99/month"
    # a brand-new browser (no cookies) on the same IP is still blocked
    assert analyze(client(app), photo).status_code == 402
    assert c.get("/api/me").json()["free_available"] is False


def test_failed_detection_does_not_use_up_free_grade(tmp_path):
    app, _ = make_app(tmp_path)
    c = client(app)
    blank = cv2.imencode(".jpg", np.full((600, 800, 3), 128, np.uint8))[1].tobytes()
    assert analyze(c, blank).status_code == 200
    assert c.get("/api/me").json()["free_available"] is True


def test_ip_from_forwarded_header_ignored_unless_trusted(tmp_path, photo):
    app, _ = make_app(tmp_path)
    assert analyze(client(app), photo).status_code == 200
    r = analyze(client(app), photo)           # spoofed header must not give a fresh free grade
    assert r.status_code == 402
    c2 = client(app)
    r = c2.post("/api/analyze", files={"front": ("f.jpg", photo, "image/jpeg")}, headers={"X-Forwarded-For": "9.9.9.9"})
    assert r.status_code == 402


# ------------------------------------------------------------------ accounts
def test_signup_login_logout_and_validation(tmp_path):
    app, _ = make_app(tmp_path)
    c = client(app)
    assert c.post("/api/auth/signup", json={"email": "bad", "password": "password123"}).status_code == 400
    assert c.post("/api/auth/signup", json={"email": "a@x.com", "password": "short"}).status_code == 400
    signup(c)
    assert c.get("/api/me").json()["user"]["email"] == "a@x.com"
    assert c.post("/api/auth/signup", json={"email": "A@x.com", "password": "password123"}).status_code == 409
    c.post("/api/auth/logout")
    assert c.get("/api/me").json()["user"] is None
    assert c.post("/api/auth/login", json={"email": "a@x.com", "password": "wrong-password"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "a@x.com", "password": "password123"}).status_code == 200
    assert "samesite=lax" in c.post("/api/auth/login", json={"email": "a@x.com", "password": "password123"}).headers["set-cookie"].lower()


def test_login_is_throttled(tmp_path):
    app, _ = make_app(tmp_path)
    c = client(app)
    signup(c)
    c.post("/api/auth/logout")
    codes = [c.post("/api/auth/login", json={"email": "a@x.com", "password": "nope-nope"}).status_code for _ in range(8)]
    assert 429 in codes
    assert c.post("/api/auth/login", json={"email": "a@x.com", "password": "password123"}).status_code == 429  # even the right one is locked out for now


def test_password_is_not_stored_in_clear(tmp_path):
    import sqlite3
    app, _ = make_app(tmp_path)
    signup(client(app), pw="correct horse battery")
    v = sqlite3.connect(tmp_path / "accounts.sqlite3").execute("SELECT pw_hash FROM users").fetchone()[0]
    assert v.startswith("scrypt$") and "correct horse" not in v


def test_raw_ip_is_never_stored(tmp_path, photo):
    app, _ = make_app(tmp_path)
    analyze(client(app), photo)
    blob = b"".join(p.read_bytes() for p in tmp_path.glob("accounts.sqlite3"))
    assert b"testclient" not in blob and b"127.0.0.1" not in blob


# ------------------------------------------------------------------ gating for logged-in users
def test_logged_in_but_unsubscribed_is_blocked(tmp_path, photo):
    app, _ = make_app(tmp_path)
    c = client(app)
    signup(c)
    r = analyze(c, photo)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "subscription_required"


def test_subscriber_can_grade_and_batch_requires_subscription(tmp_path, photo):
    app, _ = make_app(tmp_path)
    c = client(app)
    signup(c)
    webhook_activate(c, app, "a@x.com")
    assert c.get("/api/me").json()["user"]["subscribed"] is True
    assert analyze(c, photo).status_code == 200
    assert analyze(c, photo).status_code == 200            # no per-IP block for subscribers


def test_anonymous_cannot_batch(tmp_path):
    import io, zipfile
    app, _ = make_app(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("c1_front.jpg", jpg())
        z.writestr("c2_front.jpg", jpg(seed=2))
    r = client(app).post("/api/batch", files={"archive": ("a.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 402


def test_monthly_cap(tmp_path, photo):
    app, _ = make_app(tmp_path, monthly_cap=1)
    c = client(app)
    signup(c)
    webhook_activate(c, app, "a@x.com")
    assert analyze(c, photo).status_code == 200
    r = analyze(c, photo)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "cap_reached"


def test_expired_period_is_not_subscribed(tmp_path):
    app, _ = make_app(tmp_path)
    c = client(app)
    signup(c)
    uid = _uid(app, "a@x.com")
    evt = {"id": "evt_old", "type": "customer.subscription.updated",
           "data": {"object": {"id": "sub_1", "customer": "cus_1", "status": "active", "current_period_end": time.time() - 10 * 86400,
                               "metadata": {"user_id": uid}}}}
    raw, sig = sign_event(evt)
    c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": sig})
    assert c.get("/api/me").json()["user"]["subscribed"] is False


# ------------------------------------------------------------------ stripe
def test_checkout_request_is_6_99_monthly_and_tied_to_user(tmp_path):
    app, fake = make_app(tmp_path)
    c = client(app)
    assert c.post("/api/billing/checkout").status_code == 401
    signup(c)
    r = c.post("/api/billing/checkout")
    assert r.status_code == 200 and r.json()["url"].startswith("https://checkout.stripe.test/")
    method, url, form, headers = fake.calls[-1]
    assert form["mode"] == "subscription"
    assert form["line_items[0][price_data][unit_amount]"] == "699"
    assert form["line_items[0][price_data][recurring][interval]"] == "month"
    assert form["line_items[0][price_data][currency]"] == "usd"
    assert form["client_reference_id"] == _uid(app, "a@x.com")
    assert form["customer_email"] == "a@x.com"
    assert headers["Authorization"] == "Bearer sk_test_x"
    assert "{CHECKOUT_SESSION_ID}" in form["success_url"]


def test_checkout_uses_configured_price_id(tmp_path):
    fake = FakeStripe()
    cfg = Settings(paywall=True, stripe_secret="sk")
    app = create_app(tmp_path, cfg, billing.Stripe("sk", "price_123", http=fake.http))
    c = client(app)
    signup(c)
    c.post("/api/billing/checkout")
    assert fake.calls[-1][2]["line_items[0][price]"] == "price_123"


def test_checkout_not_configured(tmp_path):
    cfg = Settings(paywall=True)
    app = create_app(tmp_path, cfg)
    c = client(app)
    signup(c)
    r = c.post("/api/billing/checkout")
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_confirm_activates_and_rejects_other_users_session(tmp_path):
    app, fake = make_app(tmp_path)
    c = client(app)
    signup(c)
    uid = _uid(app, "a@x.com")
    fake.session = {"id": "cs_1", "mode": "subscription", "client_reference_id": uid, "customer": "cus_9",
                    "subscription": {"id": "sub_9", "status": "active", "items": {"data": [{"current_period_end": time.time() + 86400 * 30}]}}}
    r = c.post("/api/billing/confirm", params={"session_id": "cs_1"})
    assert r.status_code == 200 and r.json()["subscribed"] is True
    c2 = client(app)
    signup(c2, "b@x.com")
    fake.session["client_reference_id"] = uid
    assert c2.post("/api/billing/confirm", params={"session_id": "cs_1"}).status_code == 403


def test_confirm_ignores_unpaid_session(tmp_path):
    app, fake = make_app(tmp_path)
    c = client(app)
    signup(c)
    fake.session = {"mode": "subscription", "client_reference_id": _uid(app, "a@x.com"), "customer": "c",
                    "subscription": {"id": "s", "status": "incomplete", "current_period_end": time.time() + 99999}}
    assert c.post("/api/billing/confirm", params={"session_id": "cs_1"}).json()["subscribed"] is False


def test_webhook_signature_checks(tmp_path):
    app, _ = make_app(tmp_path)
    c = client(app)
    evt = {"id": "evt_1", "type": "customer.subscription.created", "data": {"object": {}}}
    raw, sig = sign_event(evt)
    assert c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": sig}).status_code == 200
    assert c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": "t=1,v1=bad"}).status_code == 400
    assert c.post("/api/billing/webhook", content=raw).status_code == 400
    raw2, sig2 = sign_event(evt, secret="whsec_other")
    assert c.post("/api/billing/webhook", content=raw2, headers={"stripe-signature": sig2}).status_code == 400
    raw3, sig3 = sign_event({**evt, "id": "evt_3"}, ts=time.time() - 3600)
    assert c.post("/api/billing/webhook", content=raw3, headers={"stripe-signature": sig3}).status_code == 400   # replay of an old delivery
    assert c.post("/api/billing/webhook", content=raw + b" ", headers={"stripe-signature": sig}).status_code == 400   # tampered body


def test_webhook_duplicate_and_cancellation(tmp_path):
    app, _ = make_app(tmp_path)
    c = client(app)
    signup(c)
    webhook_activate(c, app, "a@x.com")
    uid = _uid(app, "a@x.com")
    evt = {"id": "evt_del", "type": "customer.subscription.deleted", "data": {"object": {"id": "sub_1", "customer": "cus_1", "status": "canceled",
                                                                                       "current_period_end": time.time() + 86400, "metadata": {"user_id": uid}}}}
    raw, sig = sign_event(evt)
    assert c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": sig}).json()["result"] == "status:canceled"
    assert c.get("/api/me").json()["user"]["subscribed"] is False
    assert c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": sig}).json()["result"] == "duplicate"


# ------------------------------------------------------------------ isolation
def test_users_cannot_see_each_others_analyses(tmp_path, photo):
    app, _ = make_app(tmp_path)
    a, b = client(app), client(app)
    signup(a)
    webhook_activate(a, app, "a@x.com")
    aid = analyze(a, photo).json()["id"]
    signup(b, "b@x.com")
    assert a.get(f"/api/history/{aid}").status_code == 200
    assert a.get(f"/api/history/{aid}/image/front").status_code == 200
    for path in (f"/api/history/{aid}", f"/api/history/{aid}/image/front", f"/api/history/{aid}/report.pdf", f"/api/history/{aid}/audit"):
        assert b.get(path).status_code == 404, path
    assert b.post(f"/api/history/{aid}/outcome", json={"psa_grade": 9}).status_code == 404
    assert [i["id"] for i in a.get("/api/history").json()["items"]] == [aid]
    assert b.get("/api/history").json()["items"] == []
    assert client(app).get(f"/api/history/{aid}").status_code == 404      # logged out


def test_anonymous_result_is_visible_to_that_browser_and_moves_into_new_account(tmp_path, photo):
    app, _ = make_app(tmp_path)
    c = client(app)
    aid = analyze(c, photo).json()["id"]
    assert c.get(f"/api/history/{aid}/image/front").status_code == 200
    assert client(app).get(f"/api/history/{aid}/image/front").status_code == 404
    signup(c)
    assert [i["id"] for i in c.get("/api/history").json()["items"]] == [aid]


def test_admin_only_endpoints(tmp_path, photo):
    app, _ = make_app(tmp_path)
    c = client(app)
    assert c.get("/api/dataset/summary").status_code == 401
    signup(c)
    for p in ("/api/dataset/summary", "/api/gem/reliability", "/api/comparison", "/api/gem/training.csv"):
        assert c.get(p).status_code == 403, p
    assert c.post("/api/gem/model/train").status_code == 403
    ad = client(app)
    assert ad.post("/api/auth/signup", json={"email": "admin@x.com", "password": "password123"}).status_code == 403   # reserved email, no code
    assert ad.post("/api/auth/signup", json={"email": "admin@x.com", "password": "password123", "code": "wrong"}).status_code == 403
    code = (tmp_path / "admin_setup_code.txt").read_text().strip()
    assert ad.post("/api/auth/signup", json={"email": "admin@x.com", "password": "password123", "code": code}).status_code == 200
    assert ad.get("/api/dataset/summary").status_code == 200
    assert ad.get("/api/gem/reliability").status_code == 200


def test_pre_paywall_analyses_are_admin_only(tmp_path, photo):
    from gradeforge.storage import Store
    st = Store(tmp_path)
    app, _ = make_app(tmp_path)
    c = client(app)
    signup(c)
    assert c.get("/api/history/abc123").status_code == 404


def test_cross_site_post_is_blocked(tmp_path):
    app, _ = make_app(tmp_path)
    r = client(app).post("/api/auth/login", json={"email": "a@x.com", "password": "x"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client(app).post("/api/auth/login", json={"email": "a@x.com", "password": "x"}, headers={"Origin": "http://testserver"})
    assert r.status_code in (401, 429)


def test_paywall_off_by_default_is_open(tmp_path, photo):
    app = create_app(tmp_path, Settings(paywall=False))
    c = client(app)
    assert c.get("/api/me").json() == {"paywall": False}
    assert analyze(c, photo).status_code == 200 and analyze(c, photo).status_code == 200
    assert c.get("/api/dataset/summary").status_code == 200


# ------------------------------------------------------------------ admin tracking
def make_admin(app, tmp_path):
    ad = client(app)
    code = (tmp_path / "admin_setup_code.txt").read_text().strip()
    assert ad.post("/api/auth/signup", json={"email": "admin@x.com", "password": "password123", "code": code}).status_code == 200
    return ad


def test_default_admin_email_is_configured(monkeypatch):
    monkeypatch.delenv("GRADEFORGE_ADMIN_EMAILS", raising=False)
    assert Settings.from_env().admin_emails == {"jacklawrence713@gmail.com"}


def test_admin_dashboard_tracks_visits_grades_sources_and_cards(tmp_path, photo):
    app, _ = make_app(tmp_path, trust_proxy=True)
    hdr = {"X-Forwarded-For": "8.8.4.4", "CF-IPCountry": "US", "Referer": "https://www.google.com/search?q=psa", "User-Agent": "Mozilla/5.0"}
    v = client(app)
    assert v.get("/", headers=hdr).status_code == 200
    assert v.get("/", headers=hdr).status_code == 200                      # same visitor, same half hour: one visit
    assert v.get("/", headers={**hdr, "User-Agent": "Googlebot"}).status_code == 200   # bots are not counted
    v2 = client(app)
    v2.get("/", headers={**hdr, "X-Forwarded-For": "1.1.1.1", "CF-IPCountry": "GB", "Referer": ""})
    r = v.post("/api/analyze", files={"front": ("f.jpg", photo, "image/jpeg")}, data={"label": "My Charizard"}, headers=hdr)
    assert r.status_code == 200
    sub = client(app)
    signup(sub, "s@x.com")
    webhook_activate(sub, app, "s@x.com")
    sub.post("/api/analyze", files={"front": ("f.jpg", photo, "image/jpeg")}, data={"label": "Sub card"}, headers={**hdr, "X-Forwarded-For": "2.2.2.2", "CF-IPCountry": "CA"})
    ad = make_admin(app, tmp_path)
    s = ad.get("/api/admin/stats?days=7").json()
    assert s["totals"]["unique_visitors"] == 2 and s["totals"]["cards_graded"] == 2
    today = s["daily"][-1]
    assert today["visitors"] == 2 and today["grades"] == 2 and today["free"] == 1 and today["subscriber"] == 1
    assert s["by_source"] == {"free": 1, "subscriber": 1}
    assert {c["country"] for c in s["grades_by_country"]} == {"US", "CA"}
    assert s["by_referrer"][0]["referrer"] == "www.google.com"
    cards = {r["card"]: r for r in s["recent"]}
    assert cards["My Charizard"]["who"] == "anonymous" and cards["My Charizard"]["source"] == "free"
    assert cards["Sub card"]["who"] == "s@x.com"
    assert cards["My Charizard"]["verdict"] and cards["My Charizard"]["best_possible"] in range(1, 11)
    assert cards["My Charizard"]["actual_psa"] is None
    assert ad.get("/api/admin/grades.csv").text.count("\n") == 3
    # not admin -> no access
    assert sub.get("/api/admin/stats").status_code == 403
    assert client(app).get("/api/admin/stats").status_code == 401


def test_country_header_ignored_without_trusted_proxy(tmp_path, photo):
    app, _ = make_app(tmp_path)                                         # trust_proxy False
    c = client(app)
    c.post("/api/analyze", files={"front": ("f.jpg", photo, "image/jpeg")}, headers={"CF-IPCountry": "US"})
    ad = make_admin(app, tmp_path)
    assert ad.get("/api/admin/stats").json()["grades_by_country"][0]["country"] == "unknown"


# ------------------------------------------------------------------ pay per analysis
def test_checkout_for_credits_and_grant_once(tmp_path, photo):
    app, fake = make_app(tmp_path)
    c = client(app)
    signup(c)
    uid = _uid(app, "a@x.com")
    assert c.post("/api/billing/credits?qty=3").status_code == 400          # only the offered packs
    r = c.post("/api/billing/credits?qty=5")
    assert r.status_code == 200
    form = fake.calls[-1][2]
    assert form["mode"] == "payment" and form["line_items[0][quantity]"] == "5" and form["line_items[0][price_data][unit_amount]"] == "199"
    assert form["metadata[kind]"] == "credits" and form["client_reference_id"] == uid
    fake.session = {"id": "cs_pay", "mode": "payment", "payment_status": "paid", "client_reference_id": uid, "metadata": {"kind": "credits", "qty": "5", "user_id": uid}}
    assert c.post("/api/billing/confirm", params={"session_id": "cs_pay"}).json()["credits"] == 5
    assert c.post("/api/billing/confirm", params={"session_id": "cs_pay"}).json()["credits"] == 5     # return page reloaded: no double grant
    evt = {"id": "evt_pay", "type": "checkout.session.completed", "data": {"object": {"id": "cs_pay", "mode": "payment", "payment_status": "paid",
           "client_reference_id": uid, "metadata": {"kind": "credits", "qty": "5", "user_id": uid}}}}
    raw, sig = sign_event(evt)
    assert c.post("/api/billing/webhook", content=raw, headers={"stripe-signature": sig}).json()["result"] == "duplicate"   # webhook after return page: still once
    assert c.get("/api/me").json()["user"]["credits"] == 5


def test_unpaid_or_foreign_credit_session_grants_nothing(tmp_path):
    app, fake = make_app(tmp_path)
    c = client(app)
    signup(c)
    uid = _uid(app, "a@x.com")
    fake.session = {"id": "cs_u", "mode": "payment", "payment_status": "unpaid", "client_reference_id": uid, "metadata": {"kind": "credits", "qty": "5"}}
    assert c.post("/api/billing/confirm", params={"session_id": "cs_u"}).json()["credits"] == 0
    fake.session["client_reference_id"] = "someone-else"
    assert c.post("/api/billing/confirm", params={"session_id": "cs_u"}).status_code == 403


def test_credit_is_spent_per_analysis_and_refunded_on_failed_detection(tmp_path, photo):
    app, fake = make_app(tmp_path)
    c = client(app)
    signup(c)
    uid = _uid(app, "a@x.com")
    fake.session = {"id": "cs_2", "mode": "payment", "payment_status": "paid", "client_reference_id": uid, "metadata": {"kind": "credits", "qty": "1"}}
    c.post("/api/billing/confirm", params={"session_id": "cs_2"})
    blank = cv2.imencode(".jpg", np.full((600, 800, 3), 128, np.uint8))[1].tobytes()
    assert analyze(c, blank).status_code == 200                                # no card found: credit comes back
    assert c.get("/api/me").json()["user"]["credits"] == 1
    assert analyze(c, photo).status_code == 200
    assert c.get("/api/me").json()["user"]["credits"] == 0
    r = analyze(c, photo)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "subscription_required"
    ad = make_admin(app, tmp_path)
    assert ad.get("/api/admin/stats").json()["by_source"].get("credit") == 1
