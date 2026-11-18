"""Versioned PSA criteria database.

Everything in the JSON files is *PSA's published standard as reported by sources*.
Everything in engine.py is *GradeForge's own model*. Keep the two separate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DB_DIR = Path(__file__).parent / "standards_db"


def _version_key(path: Path):
    m = re.search(r"v(\d+)\.(\d+)\.(\d+)", path.name)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def available_versions() -> list[str]:
    files = sorted(_DB_DIR.glob("psa_standards_v*.json"), key=_version_key)
    return [".".join(map(str, _version_key(f))) for f in files]


class Standards:
    def __init__(self, data: dict):
        self.data = data
        self.grades = {int(k): v for k, v in data["grades"].items()}

    @property
    def version(self) -> str:
        return self.data["database_version"]

    @property
    def official_page_verified(self) -> bool:
        return bool(self.data.get("official_page_verified"))

    @property
    def leeway_pct(self) -> float:
        return float(self.data.get("front_leeway_pct", 0))

    @property
    def leeway_min_grade(self) -> int:
        return int(self.data.get("leeway_applies_to_grades_at_or_above", 7))

    def front_limit(self, grade: int):
        return self.grades[grade].get("front_max_pct")

    def back_limit(self, grade: int):
        return self.grades[grade].get("back_max_pct")

    @property
    def official_items(self) -> list:
        return self.data.get("official_items", [])

    def front_discretion(self, grade: int):
        """Outer front limit reachable only through grader discretion/eye appeal (official wording), or None."""
        g = self.grades[grade]
        return g.get("front_discretion_max_pct") or g.get("front_eye_appeal_max_pct")

    def meta(self) -> dict:
        return {
            "database_version": self.version,
            "compiled_on": self.data.get("compiled_on"),
            "official_source": self.data.get("official_source"),
            "official_page_verified": self.official_page_verified,
            "official_page_partially_verified": bool(self.data.get("official_page_partially_verified")),
            "official_verified_on": self.data.get("official_verified_on"),
            "official_items": self.official_items,
            "verification_note": self.data.get("verification_note"),
            "known_conflicts": self.data.get("known_conflicts", []),
        }


def load_standards(version: str | None = None) -> Standards:
    files = sorted(_DB_DIR.glob("psa_standards_v*.json"), key=_version_key)
    if not files:
        raise FileNotFoundError("No PSA standards database found")
    if version is None:
        path = files[-1]
    else:
        matches = [f for f in files if ".".join(map(str, _version_key(f))) == version]
        if not matches:
            raise ValueError(f"Unknown standards version {version}; available: {available_versions()}")
        path = matches[0]
    return Standards(json.loads(path.read_text(encoding="utf-8")))
