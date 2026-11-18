"""Stripe Checkout (subscription), customer portal and webhook verification. Uses plain HTTPS (no extra dependency).

NOT exercised against live Stripe in this build: tests use a fake HTTP layer and locally signed webhooks.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.stripe.com/v1"
TOLERANCE = 300


class BillingError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message, self.status = message, status


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    out = []
    for k, v in d.items():
        key = f"{prefix}[{k}]" if prefix else k
        if isinstance(v, dict):
            out += _flatten(v, key)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                out += _flatten(x, f"{key}[{i}]") if isinstance(x, dict) else [(f"{key}[{i}]", str(x))]
        elif v is not None:
            out.append((key, str(v).lower() if isinstance(v, bool) else str(v)))
    return out


def _urllib_http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:                       # network failure
        raise BillingError(f"Could not reach Stripe: {e}")


class Stripe:
    def __init__(self, secret: str, price_id: str = "", price_cents: int = 699, currency: str = "usd", http=None, per_grade_cents: int = 199):
        self.secret, self.price_id, self.cents, self.cur, self.per_grade_cents = secret, price_id, price_cents, currency, per_grade_cents
        self.http = http or _urllib_http

    def _call(self, method: str, path: str, data: dict | None = None) -> dict:
        if not self.secret:
            raise BillingError("Payments are not configured on this server yet.", 503)
        headers = {"Authorization": f"Bearer {self.secret}", "Content-Type": "application/x-www-form-urlencoded"}
        body = urllib.parse.urlencode(_flatten(data)).encode() if data else None
        status, js = self.http(method, API + path, headers, body)
        if status >= 400:
            msg = (js.get("error") or {}).get("message", "Stripe rejected the request.")
            raise BillingError(msg, 502)
        return js

    def checkout(self, user: dict, success_url: str, cancel_url: str) -> dict:
        line = {"price": self.price_id, "quantity": 1} if self.price_id else {
            "quantity": 1, "price_data": {"currency": self.cur, "unit_amount": self.cents, "recurring": {"interval": "month"},
                                          "product_data": {"name": "GradeForge subscription"}}}
        data = {"mode": "subscription", "line_items": [line], "success_url": success_url, "cancel_url": cancel_url,
                "client_reference_id": user["id"], "allow_promotion_codes": True,
                "metadata": {"user_id": user["id"]}, "subscription_data": {"metadata": {"user_id": user["id"]}}}
        if user.get("stripe_customer"):
            data["customer"] = user["stripe_customer"]
        else:
            data["customer_email"] = user["email"]
        return self._call("POST", "/checkout/sessions", data)

    def checkout_credits(self, user: dict, qty: int, success_url: str, cancel_url: str) -> dict:
        data = {"mode": "payment", "success_url": success_url, "cancel_url": cancel_url, "client_reference_id": user["id"], "customer_email": user["email"],
                "line_items": [{"quantity": qty, "price_data": {"currency": self.cur, "unit_amount": self.per_grade_cents,
                                                                "product_data": {"name": "GradeForge analysis credit"}}}],
                "metadata": {"user_id": user["id"], "kind": "credits", "qty": qty}, "payment_intent_data": {"metadata": {"user_id": user["id"], "kind": "credits"}}}
        return self._call("POST", "/checkout/sessions", data)

    def portal(self, customer: str, return_url: str) -> dict:
        return self._call("POST", "/billing_portal/sessions", {"customer": customer, "return_url": return_url})

    def get_checkout_session(self, session_id: str) -> dict:
        return self._call("GET", f"/checkout/sessions/{urllib.parse.quote(session_id, safe='')}?expand[]=subscription")


def period_end(sub: dict) -> float | None:
    """current_period_end moved onto subscription items in newer Stripe API versions; accept both."""
    pe = sub.get("current_period_end")
    if pe is None:
        items = ((sub.get("items") or {}).get("data")) or []
        ends = [i.get("current_period_end") for i in items if i.get("current_period_end")]
        pe = max(ends) if ends else None
    return float(pe) if pe else None


def verify_webhook(payload: bytes, header: str, secret: str, now: float | None = None) -> dict:
    """Check Stripe's signature (HMAC-SHA256 over '<timestamp>.<payload>') and timestamp tolerance; return the event."""
    if not secret:
        raise BillingError("Webhook secret is not configured.", 503)
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
        sigs = [p.split("=", 1)[1] for p in header.split(",") if p.startswith("v1=")]
    except Exception:
        raise BillingError("Malformed signature header.", 400)
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in sigs):
        raise BillingError("Signature mismatch.", 400)
    if abs((now or time.time()) - ts) > TOLERANCE:
        raise BillingError("Timestamp outside tolerance.", 400)
    return json.loads(payload)


def apply_event(acct, event: dict) -> str:
    """Update subscription state from a verified event. Returns what happened (for logs/tests)."""
    if acct.seen_event(event.get("id", "")):
        return "duplicate"
    obj = (event.get("data") or {}).get("object") or {}
    t = event.get("type", "")
    if t == "checkout.session.completed" and obj.get("mode") == "payment":
        uid = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("user_id")
        qty = int((obj.get("metadata") or {}).get("qty") or 0)
        if uid and acct.user(uid) and (obj.get("metadata") or {}).get("kind") == "credits" and obj.get("payment_status") == "paid" and 0 < qty <= 100:
            return "credits_added" if acct.add_credits(uid, qty, obj.get("id", "")) else "duplicate"
        return "ignored"
    if t == "checkout.session.completed":
        uid = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("user_id")
        if uid and obj.get("mode") == "subscription" and acct.user(uid):
            acct.set_subscription(uid, customer=obj.get("customer"), sub=obj.get("subscription") if isinstance(obj.get("subscription"), str) else None)
            return "linked"
        return "ignored"
    if t in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        uid = (obj.get("metadata") or {}).get("user_id")
        u = acct.user(uid) if uid else acct.user_by_customer(obj.get("customer", ""))
        if not u:
            return "unknown_user"
        status = "canceled" if t.endswith("deleted") else obj.get("status")
        acct.set_subscription(u["id"], customer=obj.get("customer"), sub=obj.get("id"), status=status, period_end=period_end(obj))
        return f"status:{status}"
    return "ignored"
