"""Pure normalization and fusion formulas for Stage 4 ranking."""

from __future__ import annotations

import math
from typing import Any

TEXT_CANDIDATE_WEIGHTS = {
    "sentence_bert": 0.20,
    "structured_kg_alignment": 0.45,
    "judge": 0.35,
}
IMAGE_CANDIDATE_WEIGHTS = {
    "clip": 0.20,
    "blip_itm": 0.30,
    "judge": 0.50,
}


def normalize_cosine(cosine: float) -> float:
    return _clamp01((_finite(cosine, "cosine") + 1.0) / 2.0)


def normalize_judge_1_to_5(score: float) -> float:
    return _clamp01((_finite(score, "score") - 1.0) / 4.0)


def normalize_visual_quality(raw_0_100: float) -> float:
    """Normalize an uncalibrated visual score from 0-100 to 0-1."""
    return _clamp01(_finite(raw_0_100, "raw_0_100") / 100.0)


def fuse_text_candidate_score(
    *,
    sentence_bert: float,
    structured_kg_alignment: float,
    judge: float,
) -> float:
    """Fuse semantic, structured-KG, and holistic text signals."""
    fused = (
        TEXT_CANDIDATE_WEIGHTS["sentence_bert"] * _unit(sentence_bert, "sentence_bert")
        + TEXT_CANDIDATE_WEIGHTS["structured_kg_alignment"]
        * _unit(structured_kg_alignment, "structured_kg_alignment")
        + TEXT_CANDIDATE_WEIGHTS["judge"] * _unit(judge, "judge")
    )
    return _clamp01(fused)


def fuse_image_candidate_score(
    *,
    clip: float,
    blip: float,
    judge: float,
) -> float:
    """Fuse image scores without the temporarily disabled visual graph signal."""
    fused = (
        IMAGE_CANDIDATE_WEIGHTS["clip"] * _unit(clip, "clip")
        + IMAGE_CANDIDATE_WEIGHTS["blip_itm"] * _unit(blip, "blip")
        + IMAGE_CANDIDATE_WEIGHTS["judge"] * _unit(judge, "judge")
    )
    return _clamp01(fused)


def _unit(value: Any, field_name: str) -> float:
    return _clamp01(_finite(value, field_name))


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
