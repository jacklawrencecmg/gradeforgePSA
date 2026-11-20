"""FastAPI application: upload, analyze, history, outcomes, PDF reports, batch."""
from __future__ import annotations

import io
import re
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import Request, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .analyze import analyze_card
from .gem import FINISHES
from .report import build_pdf
from .standards import available_versions, load_standards
from .storage import Store
from . import accounts as acc_mod, billing, dataset_api, shots
from .config import Settings, load_env_file
from .valuation import get_valuation_service

MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_BATCH_CARDS = 100
MAX_ZIP_UNCOMPRESSED = 600 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
IMG_RE = re.compile(r"^(?P<card>.+?)_(?P<side>front|back)\.(jpg|jpeg|png|webp|bmp|tif|tiff)$", re.I)


class Review(BaseModel):
    action: str
    reviewer: str = Field(default="", max_length=80)
    note: str = Field(default="", max_length=1000)
    evidence_keys: list[str] = Field(default_factory=list, max_length=20)
    severity: str | None = None


class ManualFinding(BaseModel):
    box: list[float] = Field(min_length=4, max_length=4)
    target: str
    severity: str = "minor"
    note: str = Field(max_length=1000)
    reviewer: str = Field(default="", max_length=80)


class Outcome(BaseModel):
    psa_grade: int = Field(ge=1, le=10)
    note: str = Field(default="", max_length=500)
    qualifier: str | None = Field(default=None, max_length=8)           # OC, MK, ST, PD, OF, MC
    pre_submission_photos: bool = False                                    # must be explicitly true to count as training data
    submitted_on: str | None = Field(default=None, max_length=20)
    service: str | None = Field(default=None, max_length=40)
    cert: str | None = Field(default=None, max_length=20)
    excluded: bool = False
    exclude_reason: str = Field(default="", max_length=200)


async def _read(upload: UploadFile | None) -> bytes | None:
    if upload is None or not upload.filename:
        return None
    data = await upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{upload.filename} exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB")
    return data or None


ADMIN_ONLY = ("/api/dataset", "/api/gem", "/api/comparison", "/api/admin")
HISTORY_RE = re.compile(r"^/api/history/([A-Za-z0-9]+)(/|$)")


class Credentials(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=200)
    code: str = Field(default="", max_length=64)


def create_app(data_dir: str | Path | None = None, settings: Settings | None = None, stripe: billing.Stripe | None = None) -> FastAPI:
    app = FastAPI(title="GradeForge", version=__version__)
    store = Store(Path(data_dir) if data_dir else None)
    cfg = settings or Settings.from_env()
    acct = acc_mod.Accounts(store.dir, cfg) if cfg.paywall else None
    pay = stripe or billing.Stripe(cfg.stripe_secret, cfg.stripe_price_id, cfg.price_cents, cfg.currency, per_grade_cents=cfg.per_grade_cents)

    def set_cookie(request: Request, response: Response, name: str, value: str, days: int = 365):
        response.set_cookie(name, value, max_age=days * 86400, httponly=True, samesite="lax", secure=acct.secure(request), path="/")

    def deny(code: str, message: str, status: int = 402):
        raise HTTPException(status, detail={"code": code, "message": message, "price": cfg.price_label})

    if acct:
        @app.middleware("http")
        async def guard(request: Request, call_next):
            path = request.url.path
            if path in ("/", "/admin") and request.method == "GET":
                try:
                    acct.log_visit(request)
                except Exception:
                    pass
            if path.startswith("/api/"):
                if request.method not in ("GET", "HEAD", "OPTIONS") and path != "/api/billing/webhook":
                    origin = request.headers.get("origin")
                    if origin and origin.split("://", 1)[-1] != request.headers.get("host", ""):
                        return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
                user = acct.from_request(request)
                if path.startswith(ADMIN_ONLY) and not (user and user["is_admin"]):
                    return JSONResponse({"detail": "Administrator access only."}, status_code=403 if user else 401)
                m = HISTORY_RE.match(path)
                if m and not acct.can_access(m.group(1), request, user):
                    return JSONResponse({"detail": "Unknown analysis id"}, status_code=404)
            return await call_next(request)

        @app.get("/api/me")
        def me(request: Request):
            u = acct.from_request(request)
            return {"paywall": True, "price": cfg.price_label, "billing_ready": cfg.billing_configured,
                    "per_grade": cfg.per_grade_label, "packs": [{"qty": q, "total": f"${q * cfg.per_grade_cents / 100:.2f}"} for q in cfg.credit_packs],
                    "free_available": (not u) and acct.free_available(request),
                    "user": ({"email": u["email"], "subscribed": u["subscribed"], "status": u["sub_status"], "period_end": u["period_end"],
                             "is_admin": u["is_admin"], "credits": u["credits"], "can_manage": bool(u["stripe_customer"])} if u else None)}

        def _login_response(request, response, u):
            set_cookie(request, response, acc_mod.COOKIE_SESSION, acct.new_session(u["id"]), acc_mod.SESSION_DAYS)
            acct.claim_anon(request.cookies.get(acc_mod.COOKIE_ANON), u["id"])
            return {"ok": True, "email": u["email"], "subscribed": u["subscribed"]}

        @app.get("/api/admin/stats")
        def admin_stats(days: int = 30, tz: int = 0):
            return acct.stats(store, days, tz)

        @app.get("/api/admin/grades.csv")
        def admin_csv(tz: int = 0):
            import csv
            s = acct.stats(store, 365, tz, recent=100000)
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["time_utc", "card", "who", "source", "country", "referrer", "verdict", "best_possible", "floor", "most_likely", "actual_psa", "analysis_id"])
            for r in s["recent"]:
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["ts"])), r["card"], r["who"], r["source"], r["country"], r["referrer"], r["verdict"],
                            r["best_possible"], r["floor"], r["most_likely"], r["actual_psa"], r["analysis_id"]])
            return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="gradeforge_grades.csv"'})

        @app.post("/api/auth/signup")
        def signup(body: Credentials, request: Request, response: Response):
            try:
                return _login_response(request, response, acct.signup(body.email, body.password, request, body.code))
            except acc_mod.AuthError as e:
                raise HTTPException(e.status, e.message)

        @app.post("/api/auth/login")
        def login(body: Credentials, request: Request, response: Response):
            try:
                return _login_response(request, response, acct.login(body.email, body.password, request))
            except acc_mod.AuthError as e:
                raise HTTPException(e.status, e.message)

        @app.post("/api/auth/logout")
        def logout(request: Request, response: Response):
            acct.end_session(request.cookies.get(acc_mod.COOKIE_SESSION))
            response.delete_cookie(acc_mod.COOKIE_SESSION, path="/")
            return {"ok": True}
        @app.post("/api/auth/reset-password")
        def reset_password(body: Credentials):
            """Reset admin password with setup code"""
            try:
                result = acct.reset_admin_password(body.email, body.code)
                return result
            except acc_mod.AuthError as e:
                raise HTTPException(e.status, e.message)

        def _base(request: Request) -> str:
            return cfg.base_url or str(request.base_url).rstrip("/")

        @app.post("/api/billing/checkout")
        def checkout(request: Request):
            u = acct.from_request(request)
            if not u:
                raise HTTPException(401, "Log in or create an account first.")
            if u["subscribed"]:
                raise HTTPException(409, "This account is already subscribed.")
            try:
                s = pay.checkout(u, f"{_base(request)}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}", f"{_base(request)}/?checkout=cancel")
            except billing.BillingError as e:
                raise HTTPException(e.status, e.message)
            return {"url": s["url"]}

        @app.post("/api/billing/credits")
        def buy_credits(request: Request, qty: int = 1):
            u = acct.from_request(request)
            if not u:
                raise HTTPException(401, "Log in or create an account first.")
            if qty not in cfg.credit_packs:
                raise HTTPException(400, "Choose one of the offered packs.")
            try:
                s = pay.checkout_credits(u, qty, f"{_base(request)}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}", f"{_base(request)}/?checkout=cancel")
            except billing.BillingError as e:
                raise HTTPException(e.status, e.message)
            return {"url": s["url"]}

        @app.post("/api/billing/portal")
        def portal(request: Request):
            u = acct.from_request(request)
            if not u or not u["stripe_customer"]:
                raise HTTPException(404, "No billing account yet.")
            try:
                return {"url": pay.portal(u["stripe_customer"], _base(request))["url"]}
            except billing.BillingError as e:
                raise HTTPException(e.status, e.message)

        @app.post("/api/billing/confirm")
        def confirm(request: Request, session_id: str):
            """Called when the customer returns from Stripe: asks Stripe directly, so access starts even before the webhook arrives."""
            u = acct.from_request(request)
            if not u:
                raise HTTPException(401, "Log in first.")
            try:
                s = pay.get_checkout_session(session_id)
            except billing.BillingError as e:
                raise HTTPException(e.status, e.message)
            if s.get("client_reference_id") != u["id"] or s.get("mode") not in ("subscription", "payment"):
                raise HTTPException(403, "That checkout belongs to a different account.")
            if s.get("mode") == "payment":
                md = s.get("metadata") or {}
                qty = int(md.get("qty") or 0)
                if s.get("payment_status") == "paid" and md.get("kind") == "credits" and 0 < qty <= 100:
                    acct.add_credits(u["id"], qty, s.get("id", ""))
                nu = acct.user(u["id"])
                return {"subscribed": bool(nu["subscribed"]), "credits": nu["credits"]}
            sub = s.get("subscription")
            if isinstance(sub, dict) and sub.get("status") in acc_mod.ACTIVE:
                acct.set_subscription(u["id"], customer=s.get("customer"), sub=sub.get("id"), status=sub["status"], period_end=billing.period_end(sub))
            return {"subscribed": bool(acct.user(u["id"])["subscribed"])}

        @app.post("/api/billing/webhook")
        async def webhook(request: Request):
            payload = await request.body()
            try:
                event = billing.verify_webhook(payload, request.headers.get("stripe-signature", ""), cfg.stripe_webhook_secret)
            except billing.BillingError as e:
                raise HTTPException(e.status, e.message)
            return {"result": billing.apply_event(acct, event)}
    else:
        @app.get("/api/me")
        def me_open():
            return {"paywall": False}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "standards_versions": available_versions()}
    @app.get("/api/setup-code")
    def get_setup_code():
        """Debug endpoint: return current admin setup code"""
        if not acct:
            return {"error": "Paywall not enabled"}
        code_file = acct.admin_code_file
        if code_file.exists():
            return {"setup_code": code_file.read_text().strip()}
        return {"error": "No setup code found"}

    @app.get("/api/photo-guide")
    def photo_guide():
        return {"headers": shots.HEADERS, "howto": shots.HOWTO, "why": shots.WHY}

    @app.get("/api/standards")
    def standards():
        std = load_standards()
        return {"meta": std.meta(), "grades": std.data["grades"], "leeway_note": std.data.get("leeway_note")}

    @app.post("/api/analyze")
    async def analyze(request: Request, response: Response, front: UploadFile = File(...), back: UploadFile | None = File(None),
                      front_light: list[UploadFile] = File(default=[]), back_light: list[UploadFile] = File(default=[]),
                      front_macro: list[UploadFile] = File(default=[]), back_macro: list[UploadFile] = File(default=[]),
                      label: str = Form(""), finish: str = Form("unknown")):
        fb = await _read(front)
        if fb is None:
            raise HTTPException(400, "A front image is required.")
        bb = await _read(back)
        ticket = user = None
        if acct:
            try:
                ticket = acct.reserve(request)
            except acc_mod.Paywall as e:
                deny(e.code, e.message)
            user = acct.from_request(request)
        try:
            async def many(files, cap):
                got = []
                for u in files[:cap]:
                    d = await _read(u)
                    if d:
                        got.append(((u.filename or "photo")[:120], d))
                return got

            async def named(files, cap):
                out = []
                for i, u in enumerate(files[:cap]):
                    d = await _read(u)
                    if d:
                        out.append(((u.filename or f"macro{i + 1}")[:120], d))
                return out

            fl_n, bl_n = await many(front_light, 4), await many(back_light, 4)
            fl, bl = [d for _, d in fl_n], [d for _, d in bl_n]
            fm, bm = await named(front_macro, 6), await named(back_macro, 6)
            # photos uploaded under a named close-up category: shot__<front|back>__<slot>
            form = await request.form()
            per_side = {"front": fm, "back": bm}
            counts: dict = {}
            for key, val in form.multi_items():
                parts = key.split("__")
                if len(parts) != 3 or parts[0] != "shot" or parts[1] not in per_side or parts[2] not in shots.CLOSEUP_SLOTS or not hasattr(val, "read"):
                    continue
                side_k, slot = parts[1], parts[2]
                counts[(side_k, slot)] = counts.get((side_k, slot), 0) + 1
                if counts[(side_k, slot)] > (shots.SURFACE_MAX if slot == "surface" else 1):
                    continue
                d = await _read(val)
                if d:
                    per_side[side_k].append(((val.filename or slot)[:120], d, slot))
            uploads = {"front": {"primary": front.filename, "lights": [n for n, _ in fl_n]},
                       "back": {"primary": back.filename if back else None, "lights": [n for n, _ in bl_n]}}
            traits = {"finish": finish if finish in FINISHES else "unknown"}
            result, arts = await run_in_threadpool(analyze_card, fb, bb, fl or None, bl or None, per_side["front"] or None, per_side["back"] or None, None, traits,
                                                   store.load_model(), store.reliability())
            result["photos"] = shots.build_inventory(result, uploads)
            aid = store.save(label.strip()[:120], result, arts)
            store.log_prediction(aid, result)
        except BaseException:
            if acct:
                acct.refund(ticket)
            raise
        if acct:
            anon = request.cookies.get(acc_mod.COOKIE_ANON) or (None if user else uuid.uuid4().hex)
            acct.set_owner(aid, ticket, anon, user)
            measured = any(result.get(sd) and result[sd].get("ok") for sd in ("front", "back"))
            if not measured:
                acct.refund(ticket)          # no card could be found in the photos: this attempt does not use up a grade
            if measured:
                acct.log_grade(aid, {"free": "free", "credit": "credit"}.get(ticket["kind"]) or ("admin" if user and user["is_admin"] else "subscriber"), request, user)
            if not user:
                set_cookie(request, response, acc_mod.COOKIE_ANON, anon)
                if measured:
                    set_cookie(request, response, acc_mod.COOKIE_FREE, "1")
        return {"id": aid, "label": label.strip()[:120], **result}

    @app.post("/api/batch")
    async def batch(request: Request, archive: UploadFile = File(...)):
        data = await _read(archive)
        if not data:
            raise HTTPException(400, "Upload a .zip containing <card>_front.jpg and <card>_back.jpg files.")
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise HTTPException(400, "That file is not a valid zip archive.")
        infos = zf.infolist()
        if sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
            raise HTTPException(413, "Archive expands to more than 600 MB.")
        cards: dict[str, dict] = {}
        for info in infos:
            m = IMG_RE.match(Path(info.filename).name)
            if info.is_dir() or not m or info.file_size > MAX_FILE_BYTES:
                continue
            cards.setdefault(m["card"], {})[m["side"].lower()] = info
        if not cards:
            raise HTTPException(400, "No files named <card>_front.<ext> / <card>_back.<ext> found.")
        if len(cards) > MAX_BATCH_CARDS:
            raise HTTPException(413, f"At most {MAX_BATCH_CARDS} cards per batch.")
        ticket = user = None
        if acct:
            try:
                ticket = acct.reserve(request, len(cards))
            except acc_mod.Paywall as e:
                deny(e.code, e.message)
            user = acct.from_request(request)
        rows = []
        for card, sides in sorted(cards.items()):
            fb = zf.read(sides["front"]) if "front" in sides else None
            bb = zf.read(sides["back"]) if "back" in sides else None
            try:
                result, arts = await run_in_threadpool(analyze_card, fb, bb, None, None, None, None, None, None,
                                                       store.load_model(), store.reliability())
            except BaseException:
                if acct and not rows:
                    acct.refund(ticket)
                raise
            aid = store.save(card[:120], result, arts)
            store.log_prediction(aid, result)
            if acct:
                acct.set_owner(aid, ticket, None, user)
                acct.log_grade(aid, "batch", request, user)
            a = result["assessment"]
            rows.append({"id": aid, "card": card, "status": a["prediction_status"],
                         "conservative": a.get("conservative_ceiling"), "optimistic": a.get("optimistic_ceiling"),
                         "psa10": a["psa10_eligibility"]["status"], "reasons": a.get("reasons", [])})
        return {"count": len(rows), "results": rows}

    @app.get("/api/history")
    def history(request: Request):
        items = store.history(100000 if acct else 200)
        if acct:
            u = acct.from_request(request)
            if not (u and u["is_admin"]):
                mine = acct.owned_ids(request, u)
                items = [i for i in items if i["id"] in mine]
            items = items[:200]
        return {"items": items}

    @app.get("/api/history/{aid}")
    def history_item(aid: str):
        rec = store.get(aid)
        if rec is None:
            raise HTTPException(404, "Unknown analysis id")
        return rec

    @app.post("/api/history/{aid}/outcome")
    def outcome(aid: str, body: Outcome):
        if not store.set_outcome(aid, body.psa_grade, body.note, qualifier=body.qualifier,
                                 pre_submission_photos=body.pre_submission_photos, submitted_on=body.submitted_on,
                                 service=body.service, cert=body.cert, excluded=body.excluded, exclude_reason=body.exclude_reason):
            raise HTTPException(404, "Unknown analysis id")
        return {"ok": True, "counts_for_training": bool(body.pre_submission_photos and not body.excluded)}

    @app.get("/api/gem/reliability")
    def gem_reliability():
        return store.reliability()

    @app.get("/api/gem/model")
    def gem_model_info():
        m = store.load_model()
        if not m:
            return {"trained": False, "message": "No model has been trained.", "gate": store.reliability()["training_gate"]}
        return {k: v for k, v in m.items() if k not in ("weights", "mu", "sd", "impute")}

    @app.post("/api/gem/model/train")
    def gem_model_train():
        r = store.train_model()
        return {k: v for k, v in r.items() if k not in ("weights", "mu", "sd", "impute")}

    @app.get("/api/gem/training.csv")
    def gem_training_csv():
        return Response(store.training_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="gradeforge_training_rows.csv"'})

    @app.get("/api/comparison")
    def comparison():
        return store.comparison()

    @app.get("/api/history/{aid}/image/{side}")
    def image(aid: str, side: str):
        p = store.image_path(aid, side)
        if p is None:
            raise HTTPException(404, "No such image")
        return FileResponse(p, media_type="image/png")

    @app.get("/api/history/{aid}/evidence/{name}")
    def evidence(aid: str, name: str):
        p = store.evidence_path(aid, name)
        if p is None:
            raise HTTPException(404, "No such evidence file")
        return FileResponse(p, media_type="image/png")

    @app.get("/api/history/{aid}/original/{side}/{role}")
    def original(aid: str, side: str, role: str):
        p = store.original_path(aid, side, role)
        if p is None:
            raise HTTPException(404, "No such original")
        return FileResponse(p, media_type="application/octet-stream")

    @app.post("/api/history/{aid}/{side}/findings/{fid}/review")
    def review(aid: str, side: str, fid: str, body: Review):
        try:
            result = store.review_finding(aid, side, fid, body.action, body.reviewer, body.note, body.evidence_keys, body.severity)
        except KeyError:
            raise HTTPException(404, "Unknown analysis id")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"ok": True, "assessment": result["assessment"], "gem": result.get("gem"), "plain": result.get("plain"), "findings": result[side]["findings"]}

    @app.post("/api/history/{aid}/{side}/findings")
    def add_finding(aid: str, side: str, body: ManualFinding):
        try:
            result = store.add_manual_finding(aid, side, body.box, body.target, body.severity, body.note, body.reviewer)
        except KeyError:
            raise HTTPException(404, "Unknown analysis id")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"ok": True, "assessment": result["assessment"], "gem": result.get("gem"), "plain": result.get("plain"), "findings": result[side]["findings"]}

    @app.get("/api/history/{aid}/audit")
    def audit(aid: str):
        return {"items": store.audit_log(aid)}

    @app.get("/api/valuation/{aid}")
    def get_card_valuation(aid: str):
        """Get estimated card value based on recent market data (eBay, Card Ladder)."""
        rec = store.get(aid)
        if rec is None:
            raise HTTPException(404, "Unknown analysis id")

        valuation_service = get_valuation_service()
        valuation = valuation_service.get_valuation_for_result(rec)
        return valuation

    @app.post("/api/valuation")
    def lookup_card_value(card_name: str, set_name: str = "", card_number: str = "", grade: int | None = None, finish: str = "standard"):
        """
        Look up estimated card value.

        Query parameters:
        - card_name: e.g. "Charizard ex"
        - set_name: e.g. "Scarlet & Violet" (optional)
        - card_number: e.g. "223" (optional)
        - grade: PSA grade 1-10 (optional)
        - finish: standard, chrome, refractor, etc.
        """
        valuation_service = get_valuation_service()
        valuation = valuation_service.get_valuation(card_name, set_name, card_number, grade, finish)
        return valuation

    @app.get("/api/history/{aid}/report.pdf")
    def report(aid: str):
        rec = store.get(aid)
        if rec is None:
            raise HTTPException(404, "Unknown analysis id")
        paths = {s: store.image_path(aid, s) for s in ("front", "back")}
        pdf = build_pdf(rec, paths, store.dir / "evidence" / aid)
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="gradeforge_{aid}.pdf"'})

    dataset_api.register(app, store)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def landing():
        return FileResponse(STATIC / "landing.html")

    @app.get("/app")
    @app.get("/admin")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


load_env_file()
app = create_app()
