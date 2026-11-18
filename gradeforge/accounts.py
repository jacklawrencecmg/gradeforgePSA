"""Accounts, sessions, free-grade metering and subscription state (separate SQLite file so the grading data is untouched).

Privacy: raw IP addresses are never stored, only a salted hash. Passwords are scrypt-hashed. Session tokens are stored hashed.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

from .config import Settings

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
SESSION_DAYS = 30
GRACE_SECONDS = 3 * 86400            # safety margin past the paid period if a webhook is late
ACTIVE = ("active", "trialing")
COOKIE_SESSION, COOKIE_ANON, COOKIE_FREE = "gf_session", "gf_anon", "gf_free"


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


class Paywall(Exception):
    """Raised when an analysis is not allowed; carries a machine-readable code for the UI."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _hash_pw(pw: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${dk.hex()}"


def _check_pw(pw: str, stored: str) -> bool:
    try:
        _, salt, dk = stored.split("$")
        new = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(new.hex(), dk)
    except Exception:
        return False


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class Accounts:
    def __init__(self, data_dir: Path, settings: Settings):
        self.s = settings
        self.db = Path(data_dir) / "accounts.sqlite3"
        salt_file = Path(data_dir) / "ip_salt.key"
        if not salt_file.exists():
            salt_file.write_text(secrets.token_hex(32))
        self._salt = salt_file.read_text().strip()
        with self._c() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT UNIQUE, pw_hash TEXT, created REAL,
                stripe_customer TEXT, stripe_sub TEXT, sub_status TEXT, period_end REAL, credits INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id TEXT, expires REAL);
            CREATE TABLE IF NOT EXISTS free_uses(ip_hash TEXT PRIMARY KEY, n INTEGER, first_ts REAL);
            CREATE TABLE IF NOT EXISTS usage(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts REAL, n INTEGER);
            CREATE TABLE IF NOT EXISTS owners(analysis_id TEXT PRIMARY KEY, owner TEXT);
            CREATE TABLE IF NOT EXISTS stripe_events(event_id TEXT PRIMARY KEY, ts REAL);
            CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, key TEXT, ts REAL);
            CREATE INDEX IF NOT EXISTS ix_att ON attempts(kind, key, ts);
            CREATE TABLE IF NOT EXISTS visits(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, vid TEXT, country TEXT, ref TEXT);
            CREATE INDEX IF NOT EXISTS ix_vis ON visits(ts);
            CREATE TABLE IF NOT EXISTS grades(analysis_id TEXT PRIMARY KEY, ts REAL, source TEXT, who TEXT, vid TEXT, country TEXT, ref TEXT);
            CREATE INDEX IF NOT EXISTS ix_gr ON grades(ts);
            """)
        with self._c() as c:                                  # upgrade databases created before pay-per-analysis existed
            if "credits" not in [r[1] for r in c.execute("PRAGMA table_info(users)")]:
                c.execute("ALTER TABLE users ADD COLUMN credits INTEGER DEFAULT 0")
        # Accounts are not email-verified, so an admin email can only be registered with a setup code that lives on the server.
        self.admin_code_file = Path(data_dir) / "admin_setup_code.txt"
        if self.s.admin_emails and not self.admin_code_file.exists():
            self.admin_code_file.write_text(secrets.token_hex(6))
        if self.s.admin_emails:
            print(f"[GradeForge] Admin setup code (needed once to register {', '.join(sorted(self.s.admin_emails))}): "
                  f"{self.admin_code_file.read_text().strip()}  (also saved in {self.admin_code_file})", flush=True)

    def _c(self):
        c = sqlite3.connect(self.db, timeout=15)
        c.row_factory = sqlite3.Row
        return c

    # ---------------------------------------------------------------- request helpers
    def client_ip(self, request) -> str:
        ip = request.client.host if request.client else "0.0.0.0"
        if self.s.trust_proxy:
            xff = request.headers.get("x-forwarded-for", "")
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            if parts:
                ip = parts[-1]          # the entry added by OUR proxy; earlier entries can be forged by the client
        return ip

    def ip_hash(self, request) -> str:
        ip = self.client_ip(request)
        try:
            a = ipaddress.ip_address(ip)
            if a.version == 6:                                   # a household/ISP gets a whole /64: count it as one address
                ip = str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address)
        except ValueError:
            pass
        return _sha(self._salt + ip)

    def secure(self, request) -> bool:
        if self.s.secure_cookies is not None:
            return self.s.secure_cookies
        proto = request.headers.get("x-forwarded-proto") if self.s.trust_proxy else None
        return (proto or request.url.scheme) == "https"

    # ---------------------------------------------------------------- users
    def _throttle(self, kind: str, key: str, limit: int, window: int) -> None:
        with self._c() as c:
            n = c.execute("SELECT COUNT(*) FROM attempts WHERE kind=? AND key=? AND ts>?", (kind, key, time.time() - window)).fetchone()[0]
        if n >= limit:
            raise AuthError("Too many attempts. Please wait a few minutes and try again.", 429)

    def _note(self, kind: str, key: str) -> None:
        with self._c() as c:
            c.execute("INSERT INTO attempts(kind,key,ts) VALUES (?,?,?)", (kind, key, time.time()))

    def signup(self, email: str, password: str, request, code: str = "") -> dict:
        email = (email or "").strip().lower()
        if not EMAIL_RE.match(email):
            raise AuthError("Enter a valid email address.")
        if len(password or "") < 8 or len(password) > 200:
            raise AuthError("Use a password of at least 8 characters.")
        iph = self.ip_hash(request)
        self._throttle("signup", iph, 8, 86400)
        if email in self.s.admin_emails:
            real = self.admin_code_file.read_text().strip() if self.admin_code_file.exists() else ""
            if not (code and real and hmac.compare_digest(code.strip(), real)):
                self._note("signup", iph)
                raise AuthError("This email is reserved for the site administrator. Enter the admin setup code (printed in the server log and saved in admin_setup_code.txt in the data folder).", 403)
        uid = uuid.uuid4().hex
        try:
            with self._c() as c:
                c.execute("INSERT INTO users(id,email,pw_hash,created) VALUES (?,?,?,?)", (uid, email, _hash_pw(password), time.time()))
        except sqlite3.IntegrityError:
            raise AuthError("An account with that email already exists. Log in instead.", 409)
        self._note("signup", iph)
        return self.user(uid)

    def login(self, email: str, password: str, request) -> dict:
        email = (email or "").strip().lower()
        iph = self.ip_hash(request)
        self._throttle("login_ip", iph, 20, 900)
        self._throttle("login_email", email, 6, 900)
        with self._c() as c:
            row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row:
            ok = _check_pw(password or "", row["pw_hash"])
        else:
            _hash_pw(password or "")      # spend the same time so response timing does not reveal which emails exist
            ok = False
        if not ok:
            self._note("login_ip", iph)
            self._note("login_email", email)
            raise AuthError("Incorrect email or password.", 401)
        return self.user(row["id"])

    def new_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._c() as c:
            c.execute("INSERT INTO sessions VALUES (?,?,?)", (_sha(token), user_id, time.time() + SESSION_DAYS * 86400))
            c.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))
        return token

    def end_session(self, token: str | None) -> None:
        if token:
            with self._c() as c:
                c.execute("DELETE FROM sessions WHERE token_hash=?", (_sha(token),))

    def user(self, uid: str) -> dict | None:
        with self._c() as c:
            r = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d.pop("pw_hash", None)
        d["credits"] = d.get("credits") or 0
        d["subscribed"] = self.is_subscribed(d)
        d["is_admin"] = d["email"] in self.s.admin_emails
        return d

    def user_by_customer(self, customer: str) -> dict | None:
        with self._c() as c:
            r = c.execute("SELECT id FROM users WHERE stripe_customer=?", (customer,)).fetchone()
        return self.user(r["id"]) if r else None

    def from_request(self, request) -> dict | None:
        tok = request.cookies.get(COOKIE_SESSION)
        if not tok:
            return None
        with self._c() as c:
            r = c.execute("SELECT user_id FROM sessions WHERE token_hash=? AND expires>?", (_sha(tok), time.time())).fetchone()
        return self.user(r["user_id"]) if r else None

    @staticmethod
    def is_subscribed(u: dict) -> bool:
        pe = u.get("period_end")
        return u.get("sub_status") in ACTIVE and bool(pe) and time.time() < pe + GRACE_SECONDS

    def set_subscription(self, uid: str, *, customer=None, sub=None, status=None, period_end=None) -> None:
        sets, vals = [], []
        for col, v in (("stripe_customer", customer), ("stripe_sub", sub), ("sub_status", status), ("period_end", period_end)):
            if v is not None:
                sets.append(f"{col}=?")
                vals.append(v)
        if sets:
            with self._c() as c:
                c.execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", (*vals, uid))

    def add_credits(self, uid: str, n: int, session_id: str) -> bool:
        """Grant purchased credits exactly once per Stripe checkout session (the webhook and the return page can both arrive)."""
        if self.seen_event(f"cs:{session_id}"):
            return False
        with self._c() as c:
            c.execute("UPDATE users SET credits=COALESCE(credits,0)+? WHERE id=?", (n, uid))
        return True

    def seen_event(self, event_id: str) -> bool:
        """True if this Stripe event was already processed (webhooks are delivered at least once)."""
        with self._c() as c:
            cur = c.execute("INSERT OR IGNORE INTO stripe_events VALUES (?,?)", (event_id, time.time()))
            return cur.rowcount == 0

    # ---------------------------------------------------------------- metering
    def reserve(self, request, n: int = 1) -> dict:
        """Decide whether this request may run n analyses and reserve them. Raises Paywall. Call refund(ticket) if the work fails."""
        user = self.from_request(request)
        if user and (user["subscribed"] or user["is_admin"]):
            cap = self.s.monthly_cap
            if cap and not user["is_admin"]:
                with self._c() as c:
                    used = c.execute("SELECT COALESCE(SUM(n),0) FROM usage WHERE user_id=? AND ts>?", (user["id"], time.time() - 30 * 86400)).fetchone()[0]
                if used + n > cap:
                    raise Paywall("cap_reached", f"You have reached the fair-use limit of {cap} analyses in 30 days.")
            with self._c() as c:
                cur = c.execute("INSERT INTO usage(user_id,ts,n) VALUES (?,?,?)", (user["id"], time.time(), n))
            return {"kind": "user", "user_id": user["id"], "usage_id": cur.lastrowid}
        if user and user["credits"] >= n:
            with self._c() as c:
                cur = c.execute("UPDATE users SET credits=credits-? WHERE id=? AND credits>=?", (n, user["id"], n))
            if cur.rowcount:
                return {"kind": "credit", "user_id": user["id"], "n": n}
        if user:
            raise Paywall("subscription_required", f"Subscribe for {self.s.price_label}, or buy analysis credits ({self.s.per_grade_label} each), to keep grading cards.")
        # anonymous visitor: free grades are counted per IP address (and remembered in a cookie)
        if n > 1:
            raise Paywall("login_required", "Batch grading is for subscribers. Create an account and subscribe.")
        iph = self.ip_hash(request)
        if request.cookies.get(COOKIE_FREE) == "1" and self.s.free_grades <= 1:
            raise Paywall("free_used", "Your free grade has been used. Create an account and subscribe to keep grading cards.")
        with self._c() as c:
            c.execute("INSERT OR IGNORE INTO free_uses VALUES (?,?,?)", (iph, 0, time.time()))
            cur = c.execute("UPDATE free_uses SET n=n+1 WHERE ip_hash=? AND n<?", (iph, self.s.free_grades))
            if cur.rowcount == 0:
                raise Paywall("free_used", "The free grade for this network has been used. Create an account and subscribe to keep grading cards.")
        return {"kind": "free", "ip_hash": iph}

    def refund(self, ticket: dict | None) -> None:
        if not ticket:
            return
        with self._c() as c:
            if ticket["kind"] == "free":
                c.execute("UPDATE free_uses SET n=MAX(0,n-1) WHERE ip_hash=?", (ticket["ip_hash"],))
            elif ticket["kind"] == "user":
                c.execute("DELETE FROM usage WHERE id=?", (ticket["usage_id"],))
            elif ticket["kind"] == "credit":
                c.execute("UPDATE users SET credits=credits+? WHERE id=?", (ticket["n"], ticket["user_id"]))

    def free_available(self, request) -> bool:
        if request.cookies.get(COOKIE_FREE) == "1" and self.s.free_grades <= 1:
            return False
        with self._c() as c:
            r = c.execute("SELECT n FROM free_uses WHERE ip_hash=?", (self.ip_hash(request),)).fetchone()
        return (r["n"] if r else 0) < self.s.free_grades

    # ---------------------------------------------------------------- ownership
    def set_owner(self, aid: str, ticket: dict, anon_token: str | None, user: dict | None) -> None:
        owner = f"u:{user['id']}" if user else f"a:{anon_token}"
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO owners VALUES (?,?)", (aid, owner))

    def owner_keys(self, request, user: dict | None) -> list[str]:
        keys = []
        if user:
            keys.append(f"u:{user['id']}")
        if request.cookies.get(COOKIE_ANON):
            keys.append(f"a:{request.cookies[COOKIE_ANON]}")
        return keys

    def can_access(self, aid: str, request, user: dict | None) -> bool:
        if user and user["is_admin"]:
            return True
        with self._c() as c:
            r = c.execute("SELECT owner FROM owners WHERE analysis_id=?", (aid,)).fetchone()
        return bool(r) and r["owner"] in self.owner_keys(request, user)

    def owned_ids(self, request, user: dict | None) -> set[str]:
        keys = self.owner_keys(request, user)
        if not keys:
            return set()
        with self._c() as c:
            rows = c.execute(f"SELECT analysis_id FROM owners WHERE owner IN ({','.join('?' * len(keys))})", keys).fetchall()
        return {r[0] for r in rows}

    def claim_anon(self, anon_token: str | None, user_id: str) -> None:
        """When a visitor creates an account or logs in, their earlier anonymous result moves into the account."""
        if anon_token:
            with self._c() as c:
                c.execute("UPDATE owners SET owner=? WHERE owner=?", (f"u:{user_id}", f"a:{anon_token}"))

    # ---------------------------------------------------------------- tracking (admin dashboard)
    COUNTRY_HEADERS = ("cf-ipcountry", "x-vercel-ip-country", "cloudfront-viewer-country", "x-appengine-country", "x-country-code")

    def country(self, request) -> str:
        """Visitor country, only when a proxy we trust supplies it (no GeoIP database is bundled). Otherwise 'unknown'."""
        if self.s.trust_proxy:
            for h in self.COUNTRY_HEADERS:
                v = (request.headers.get(h) or "").strip().upper()
                if re.fullmatch(r"[A-Z]{2}", v) and v not in ("XX", "T1"):
                    return v
        return "unknown"

    @staticmethod
    def _ref(request) -> str:
        ref = request.headers.get("referer", "")
        host = re.sub(r"^[a-z]+://", "", ref).split("/")[0].split("?")[0].lower()[:100]
        own = (request.headers.get("host") or "").lower()
        return "" if not host or host == own else host

    def log_visit(self, request) -> None:
        if re.search(r"bot|crawl|spider|slurp|monitor|curl|python", request.headers.get("user-agent", ""), re.I):
            return
        vid, now = self.ip_hash(request), time.time()
        with self._c() as c:
            if c.execute("SELECT 1 FROM visits WHERE vid=? AND ts>?", (vid, now - 1800)).fetchone():
                return                                                    # same visitor within 30 minutes: one visit
            c.execute("INSERT INTO visits(ts,vid,country,ref) VALUES (?,?,?,?)", (now, vid, self.country(request), self._ref(request)))

    def log_grade(self, aid: str, source: str, request, user: dict | None) -> None:
        vid = self.ip_hash(request)
        with self._c() as c:
            r = c.execute("SELECT ref FROM visits WHERE vid=? AND ref!='' ORDER BY ts DESC LIMIT 1", (vid,)).fetchone()
            c.execute("INSERT OR REPLACE INTO grades VALUES (?,?,?,?,?,?,?)",
                      (aid, time.time(), source, user["email"] if user else "anonymous", vid, self.country(request), r["ref"] if r else ""))

    def stats(self, store, days: int = 30, tz_minutes: int = 0, recent: int = 200) -> dict:
        """tz_minutes follows JavaScript getTimezoneOffset(): minutes the viewer's clock is BEHIND UTC (New York in winter = 300)."""
        days = max(1, min(int(days), 365))
        off = int(tz_minutes) * 60
        now = time.time()
        since = now - days * 86400 - 86400
        day = lambda col: f"date({col} - {off}, 'unixepoch')"
        today = time.strftime("%Y-%m-%d", time.gmtime(now - off))
        with self._c() as c:
            vis = {r["d"]: r["n"] for r in c.execute(f"SELECT {day('ts')} d, COUNT(DISTINCT vid) n FROM visits WHERE ts>? GROUP BY d", (since,))}
            gr = {}
            for r in c.execute(f"SELECT {day('ts')} d, source, COUNT(*) n FROM grades WHERE ts>? GROUP BY d, source", (since,)):
                gr.setdefault(r["d"], {})[r["source"]] = r["n"]
            by_source = {r["source"]: r["n"] for r in c.execute("SELECT source, COUNT(*) n FROM grades GROUP BY source")}
            by_country = [dict(r) for r in c.execute("""SELECT g.country country, COUNT(*) grades, COUNT(DISTINCT g.vid) people FROM grades g
                                                         GROUP BY g.country ORDER BY grades DESC LIMIT 25""")]
            by_ref = [dict(r) for r in c.execute("""SELECT CASE WHEN ref='' THEN '(direct / unknown)' ELSE ref END referrer, COUNT(*) grades
                                                     FROM grades GROUP BY referrer ORDER BY grades DESC LIMIT 25""")]
            vis_country = [dict(r) for r in c.execute("SELECT country, COUNT(DISTINCT vid) visitors FROM visits GROUP BY country ORDER BY visitors DESC LIMIT 25")]
            tot_vis = c.execute("SELECT COUNT(DISTINCT vid) FROM visits").fetchone()[0]
            tot_gr = c.execute("SELECT COUNT(*) FROM grades").fetchone()[0]
            users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            subs = c.execute("SELECT COUNT(*) FROM users WHERE sub_status IN ('active','trialing') AND period_end>?", (now - GRACE_SECONDS,)).fetchone()[0]
            rows = c.execute("SELECT * FROM grades ORDER BY ts DESC LIMIT ?", (recent,)).fetchall()
        series = []
        for i in range(days - 1, -1, -1):
            d = time.strftime("%Y-%m-%d", time.gmtime(now - off - i * 86400))
            g = gr.get(d, {})
            series.append({"day": d, "visitors": vis.get(d, 0), "grades": sum(g.values()), **{k: g.get(k, 0) for k in ("free", "subscriber", "credit", "batch", "admin")}})
        recent_rows = []
        for r in rows:
            rec = store.get(r["analysis_id"])
            res = (rec or {}).get("result") or {}
            gem = res.get("gem") or {}
            ceil = gem.get("grade_ceiling") or {}
            recent_rows.append({"ts": r["ts"], "analysis_id": r["analysis_id"], "card": (rec or {}).get("label") or r["analysis_id"], "who": r["who"],
                                "source": r["source"], "country": r["country"], "referrer": r["ref"] or "",
                                "verdict": (gem.get("psa10_eligibility") or {}).get("status"), "best_possible": ceil.get("optimistic"),
                                "floor": ceil.get("conservative"), "most_likely": (gem.get("most_likely") or {}).get("grade"),
                                "actual_psa": ((rec or {}).get("outcome") or {}).get("psa_grade")})
        return {"days": days, "today": today, "totals": {"unique_visitors": tot_vis, "cards_graded": tot_gr, "accounts": users, "active_subscribers": subs},
                "daily": series, "by_source": by_source, "grades_by_country": by_country, "visitors_by_country": vis_country, "by_referrer": by_ref, "recent": recent_rows,
                "notes": ["Days follow your browser's time zone. A unique visitor is one IP address (hashed) seen in a day; a visit within 30 minutes of the last one is not counted again.",
                          "'Grade' means GradeForge's photo-based estimate (best possible grade, floor, verdict), not a PSA grade. 'actual_psa' appears only where the real PSA result was recorded.",
                          "Country needs a host that adds a country header (Cloudflare, Vercel, CloudFront, App Engine) and GRADEFORGE_TRUST_PROXY=1; otherwise it shows 'unknown'. Raw IP addresses are not stored."]}
