"""Glue: run both sides through the CV pipeline, then the grading engine."""
from __future__ import annotations

from .cv.pipeline import analyze_side
from .plain import plain_summary
from .engine import assess_card
from .gem import analyze_gem
from .gem_model import extract_features, model_public
from .standards import load_standards


def evaluate_gem(result: dict, std=None, model: dict | None = None, reliability: dict | None = None) -> dict:
    """Gem Mint analysis for a pipeline result. `model` is a trained-model dict from the store (or None)."""
    std = std or load_standards(result.get("standards_version"))
    gem = analyze_gem(result, std, reliability=reliability)
    if model and model.get("validated"):
        gem = analyze_gem(result, std, model=model_public(model, extract_features(gem, result)), reliability=reliability)
    elif model:
        gem = analyze_gem(result, std, model=model_public(model), reliability=reliability)
    return gem


def analyze_card(front_bytes: bytes | None, back_bytes: bytes | None,
                 front_lighting: list[bytes] | None = None, back_lighting: list[bytes] | None = None,
                 front_macros: list | None = None, back_macros: list | None = None,
                 standards_version: str | None = None, traits: dict | None = None,
                 model: dict | None = None, reliability: dict | None = None):
    """Returns (result_dict, {"front": artifacts|None, "back": artifacts|None}).
    artifacts = {rect, evidence{filename: png bytes}, originals{role: bytes}}."""
    std = load_standards(standards_version)
    arts = {"front": None, "back": None}
    sides = {"front": None, "back": None}
    if front_bytes:
        sides["front"], arts["front"] = analyze_side(front_bytes, "front", front_lighting, front_macros)
    if back_bytes:
        sides["back"], arts["back"] = analyze_side(back_bytes, "back", back_lighting, back_macros)
    result = {"assessment": assess_card(sides["front"], sides["back"], std), "front": sides["front"], "back": sides["back"],
              "standards_version": standards_version, "card_traits": traits or {}}
    result["gem"] = evaluate_gem(result, std, model, reliability)
    result["plain"] = plain_summary(result)
    return result, arts


def reassess(result: dict, model: dict | None = None, reliability: dict | None = None) -> dict:
    """Recompute the assessment and Gem Mint analysis after a review changed finding states."""
    std = load_standards(result.get("standards_version"))
    result["assessment"] = assess_card(result.get("front"), result.get("back"), std)
    result["gem"] = evaluate_gem(result, std, model, reliability)
    result["plain"] = plain_summary(result)
    return result
