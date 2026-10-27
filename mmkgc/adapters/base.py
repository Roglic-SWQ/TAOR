"""Deterministic generation adapter protocols and rule-based defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedImageArtifact:
    """In-memory generated image plus metadata for the run writer."""

    entity_key: str
    candidate_id: str
    relative_image_path: str
    relative_metadata_path: str
    image_bytes: bytes
    metadata: dict[str, Any]
    model_call: dict[str, Any]


class TextGenerator(Protocol):
    """Protocol for text generators used by Stage 3."""

    # Optional model adapters may additionally implement:
    # - generate_with_context(..., context=structured_candidate_context)
    # - consume_last_call(entity_key, candidate_id)
    # - last_candidate_metadata()

    def generate(
        self,
        record_label: str,
        entity_type: str,
        relation: str,
        angle: str,
        facts: list[str],
    ) -> str:
        """Generate a text candidate from grounded relation facts."""


class ImagePromptGenerator(Protocol):
    """Protocol for image prompt planners used by Stage 3."""

    # Model adapters can implement generate_with_context(..., context=...) to
    # receive facts, requirements and rendering choices as separate fields.

    def generate(
        self,
        record_label: str,
        entity_type: str,
        relation: str,
        angle: str,
        facts: list[str],
    ) -> dict[str, Any]:
        """Generate a prompt package from grounded relation facts."""


class ImageGenerator(Protocol):
    """Protocol for optional real image generators used after prompt planning."""

    def generate(
        self,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
    ) -> GeneratedImageArtifact:
        """Generate an image artifact from a planned prompt package."""


class RuleTextGenerator:
    """Small deterministic text generator for reproducible tests and baselines."""

    def generate(
        self,
        record_label: str,
        entity_type: str,
        relation: str,
        angle: str,
        facts: list[str],
    ) -> str:
        fact_text = "; ".join(facts[:3]) if facts else "available graph evidence is sparse"
        if relation == "global_profile":
            return f"{record_label} is described as a {entity_type}; {fact_text}."
        return (
            f"{record_label} is described as a {entity_type}. "
            f"From the {angle} angle, use {relation} evidence: {fact_text}."
        )


class RuleImagePromptGenerator:
    """Small deterministic image prompt planner for reproducible tests and baselines."""

    def generate(
        self,
        record_label: str,
        entity_type: str,
        relation: str,
        angle: str,
        facts: list[str],
    ) -> dict[str, Any]:
        relation_phrase = "profile context" if relation == "global_profile" else f"{relation} context"
        scene = _image_scene_for_prompt(entity_type, relation)
        positive = ", ".join(
            part
            for part in (
                _compact_prompt_text(record_label, max_words=8),
                entity_type,
                scene,
                relation_phrase,
                _compact_angle(angle),
                *_fact_prompt_details(facts),
                *_evidence_prompt_hints(relation, facts),
                "grounded visual concept",
            )
            if part
        )
        return {
            "prompt_or_instruction": (
                f"Plan a short image prompt using only supported {relation_phrase}."
            ),
            "positive_prompt": positive,
        }


def _fact_prompt_details(facts: list[str]) -> list[str]:
    details = []
    for fact in facts:
        text = _compact_prompt_text(fact, max_words=14)
        if not text:
            continue
        detail = f"supported fact: {text}"
        if detail not in details:
            details.append(detail)
    return details[:2]


def _compact_prompt_text(value: str, *, max_words: int) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _evidence_prompt_hints(relation: str, facts: list[str]) -> list[str]:
    hints = []

    def add(label: str) -> None:
        normalized = _safe_hint_label(label)
        if not normalized:
            return
        hint = f"{normalized} evidence"
        if hint not in hints:
            hints.append(hint)

    if relation == "global_profile":
        add("profile")
    else:
        add(relation)
    for fact in facts:
        if ":" not in str(fact):
            continue
        prefix = str(fact).split(":", 1)[0]
        add(prefix)
    return hints[:2]


def _safe_hint_label(value: str) -> str:
    normalized = str(value or "").replace("_", " ").replace("-", " ").strip().lower()
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        return ""
    blocked_prefixes = ("reduce ", "regenerate ")
    if normalized.startswith(blocked_prefixes):
        return ""
    words = normalized.split()
    if len(words) > 5:
        return ""
    return " ".join(words)


def _compact_angle(angle: str) -> str:
    normalized = str(angle or "").replace("safe retry", "").strip()
    if "identity" in normalized:
        return "identity-safe portrait"
    if "evidence" in normalized:
        return "evidence-grounded scene"
    if "relation" in normalized:
        return "relation-focused scene"
    if "context" in normalized or "attribute" in normalized:
        return "context-rich scene"
    return "clear visual scene"


def _image_scene_for_prompt(entity_type: str, relation: str) -> str:
    entity_scene = _image_scene_for_entity_type(entity_type)
    if entity_scene != "contextual visual scene":
        return entity_scene
    normalized_relation = str(relation or "").lower()
    if "performer" in normalized_relation or "singer" in normalized_relation:
        return "stage performance scene"
    if "cast member" in normalized_relation or "actor" in normalized_relation:
        return "cinematic screen performance scene"
    if "director" in normalized_relation:
        return "film direction scene"
    if "occupation" in normalized_relation:
        return "professional identity scene"
    return entity_scene


def _image_scene_for_entity_type(entity_type: str) -> str:
    normalized = str(entity_type or "entity").lower()
    if "actor" in normalized:
        return "cinematic screen performance scene"
    if "music" in normalized or "rapper" in normalized or "singer" in normalized:
        return "studio performance scene"
    if "film" in normalized or "movie" in normalized:
        return "cinematic film scene"
    if "album" in normalized or "song" in normalized:
        return "music release visual concept"
    return "contextual visual scene"
