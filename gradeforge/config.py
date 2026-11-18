"""Deployment settings. Everything is read from environment variables (or a gradeforge.env file next to where the app starts).

The paywall is OFF unless GRADEFORGE_PAYWALL=1, so the local, single-user app keeps working exactly as before.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from gradeforge.env (never overrides variables already set in the real environment)."""
    p = Path(path or os.environ.get("GRADEFORGE_ENV_FILE", "gradeforge.env"))
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


DEFAULT_ADMIN = "jacklawrence713@gmail.com"


def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    paywall: bool = False
    free_grades: int = 1                 # free analyses per visitor (by IP address) before an account + subscription is needed
    price_cents: int = 699               # $6.99 / month; display only when STRIPE_PRICE_ID is used (the price object in Stripe is what is charged)
    currency: str = "usd"
    per_grade_cents: int = 199           # pay-per-analysis price per credit (PLACEHOLDER: set GRADEFORGE_PER_GRADE_CENTS to your real price)
    credit_packs: tuple = (1, 5, 10)
    monthly_cap: int = 0                 # 0 = no cap. Optional fair-use cap on analyses per subscriber per 30 days.
    trust_proxy: bool = False            # take the client IP from X-Forwarded-For (ONLY behind a proxy you control)
    secure_cookies: bool | None = None   # None = decide from the request scheme
    admin_emails: set = field(default_factory=set)
    stripe_secret: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_id: str = ""
    base_url: str = ""                   # public https URL of the site, used for Stripe return links

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        return cls(
            paywall=_flag("GRADEFORGE_PAYWALL"),
            free_grades=int(e.get("GRADEFORGE_FREE_GRADES", "1")),
            monthly_cap=int(e.get("GRADEFORGE_MONTHLY_CAP", "0")),
            per_grade_cents=int(e.get("GRADEFORGE_PER_GRADE_CENTS", "199")),
            trust_proxy=_flag("GRADEFORGE_TRUST_PROXY"),
            secure_cookies=None if "GRADEFORGE_SECURE_COOKIES" not in e else _flag("GRADEFORGE_SECURE_COOKIES"),
            admin_emails={x.strip().lower() for x in e.get("GRADEFORGE_ADMIN_EMAILS", DEFAULT_ADMIN).split(",") if x.strip()},
            stripe_secret=e.get("STRIPE_SECRET_KEY", ""), stripe_webhook_secret=e.get("STRIPE_WEBHOOK_SECRET", ""),
            stripe_price_id=e.get("STRIPE_PRICE_ID", ""), base_url=e.get("GRADEFORGE_BASE_URL", "").rstrip("/"),
        )

    @property
    def billing_configured(self) -> bool:
        return bool(self.stripe_secret)

    @property
    def per_grade_label(self) -> str:
        return f"${self.per_grade_cents / 100:.2f}"

    @property
    def price_label(self) -> str:
        return f"${self.price_cents / 100:.2f}/month"
