from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmkgc.evaluation.loader import RunEntityArtifacts


class ImageQualityTargetError(RuntimeError):
    """Raised when image quality target extraction cannot run safely."""


@dataclass(frozen=True)
class ImageQualityTarget:
    entity_key: str
    label: str
    image_candidate_id: str
    relative_image_path: str
    image_path: Path
    prompt_text: str
    candidate_text: str
    evidence_summary_text: str


def collect_entity_image_quality_targets(
    artifacts: RunEntityArtifacts,
    *,
    error_cls: type[Exception] = ImageQualityTargetError,
) -> list[ImageQualityTarget]:
    image_candidates = _image_quality_candidates(artifacts)
    entity_key = target_entity_key(artifacts)
    label = target_entity_label(artifacts)
    evidence_summary_text = build_entity_evidence_summary(artifacts)
    targets: list[ImageQualityTarget] = []

    for image_candidate in image_candidates:
        if not isinstance(image_candidate, dict):
            continue
        image_candidate_id = _candidate_id(image_candidate)
        image_path = _resolve_entity_image_path(
            artifacts,
            image_candidate,
            error_cls=error_cls,
        )
        targets.append(
            ImageQualityTarget(
                entity_key=entity_key,
                label=label,
                image_candidate_id=image_candidate_id,
                relative_image_path=_relative_path(image_path, artifacts.entity_dir),
                image_path=image_path,
                prompt_text=_clean_text(_prompt_text(image_candidate)),
                candidate_text=_clean_text(
                    _candidate_text_for_image(artifacts, image_candidate_id)
                ),
                evidence_summary_text=_clean_text(evidence_summary_text),
            )
        )

    return targets


def _image_quality_candidates(artifacts: RunEntityArtifacts) -> list[Any]:
    image_ranking = _as_dict(artifacts.ranking.get("image_ranking"))
    candidates: list[Any] = []
    seen_ids: set[str] = set()
    for field_name in ("ranked_candidates", "rejected_candidates"):
        for candidate in _as_list(image_ranking.get(field_name)):
            if not isinstance(candidate, dict):
                candidates.append(candidate)
                continue
            candidate_id = _candidate_id(candidate)
            if candidate_id and candidate_id in seen_ids:
                continue
            if candidate_id:
                seen_ids.add(candidate_id)
            candidates.append(candidate)
    candidates.extend(_unranked_generated_image_candidates(artifacts, seen_ids))
    return candidates


def _unranked_generated_image_candidates(
    artifacts: RunEntityArtifacts,
    seen_ids: set[str],
) -> list[dict[str, Any]]:
    generated_dir = artifacts.entity_dir / "generated_images"
    if not generated_dir.is_dir():
        return []
    candidates = []
    for image_path in sorted(generated_dir.glob("*.png")):
        candidate_id = image_path.stem
        if candidate_id in seen_ids:
            continue
        metadata = _read_generated_image_metadata(image_path.with_suffix(".json"))
        prompt = _first_text(metadata.get("positive_prompt"), metadata.get("prompt"))
        candidate: dict[str, Any] = {
            "candidate_id": candidate_id,
            "target_modality": "image",
            "generated_content": {
                "type": "image",
                "image_path": str(image_path.relative_to(artifacts.entity_dir)),
                "image_artifact_id": candidate_id,
            },
        }
        if prompt:
            candidate["positive_prompt"] = prompt
            candidate["generated_content"]["prompt"] = prompt
            candidate["generated_content"]["value"] = prompt
        candidates.append(candidate)
        seen_ids.add(candidate_id)
    return candidates


def _read_generated_image_metadata(path: Path) -> dict[str, Any]:
    try:
        return _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def collect_image_quality_targets(
    artifacts: RunEntityArtifacts,
    *,
    error_cls: type[Exception] = ImageQualityTargetError,
) -> list[ImageQualityTarget]:
    return collect_entity_image_quality_targets(artifacts, error_cls=error_cls)


def target_entity_key(artifacts: RunEntityArtifacts) -> str:
    final_entity = _as_dict(artifacts.final.get("entity"))
    profile = _as_dict(artifacts.knowledge.get("entity_profile"))
    return _first_text(
        final_entity.get("entity_key"),
        profile.get("entity_key"),
        artifacts.entity_key,
    )


def target_entity_label(artifacts: RunEntityArtifacts) -> str:
    final_entity = _as_dict(artifacts.final.get("entity"))
    profile = _as_dict(artifacts.knowledge.get("entity_profile"))
    return _first_text(
        final_entity.get("label"),
        profile.get("entity_label"),
        profile.get("label"),
    )


def build_entity_evidence_summary(artifacts: RunEntityArtifacts) -> str:
    profile = _as_dict(artifacts.knowledge.get("entity_profile"))
    final_entity = _as_dict(artifacts.final.get("entity"))

    label = _first_text(
        profile.get("entity_label"),
        profile.get("label"),
        final_entity.get("label"),
    )
    entity_type = _first_text(profile.get("entity_type"), profile.get("type"))
    entity_key = _first_text(
        final_entity.get("entity_key"),
        profile.get("entity_key"),
        artifacts.entity_key,
    )

    parts: list[str] = []
    if label:
        parts.append(f"Entity: {label}")
    if entity_key and (label or entity_type):
        parts.append(f"Entity key: {entity_key}")
    if entity_type:
        parts.append(f"Type: {entity_type}")

    relation_snippets = _relation_evidence_snippets(artifacts.knowledge)
    if relation_snippets:
        parts.append(f"Evidence: {'; '.join(relation_snippets)}")

    visual_terms = _image_constraint_terms(artifacts.reasoning)
    if visual_terms:
        parts.append(f"Visual requirements: {', '.join(_dedupe(visual_terms))}")

    return ". ".join(parts)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_text_list(item))
        return result
    if isinstance(value, dict):
        return []
    text = _clean_text(value)
    return [text] if text else []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _relation_evidence_snippets(
    knowledge: dict[str, Any],
    *,
    limit: int = 5,
) -> list[str]:
    snippets: list[str] = []
    for fact in _as_list(knowledge.get("relation_facts")):
        if not isinstance(fact, dict):
            continue

        relation = _first_text(
            fact.get("relation_label"),
            fact.get("relation"),
            fact.get("predicate_label"),
            fact.get("property_label"),
        )
        values = _relation_fact_values(fact)
        if relation and values:
            snippets.append(f"{relation}: {', '.join(values)}")
        elif relation:
            snippets.append(relation)
        elif values:
            snippets.append(", ".join(values))

        if len(snippets) >= limit:
            break
    return snippets


def _relation_fact_values(fact: dict[str, Any]) -> list[str]:
    value_keys = (
        "neighbor_label",
        "object_label",
        "value",
        "evidence_label",
        "evidence_labels",
        "neighbor",
        "object",
    )
    values: list[str] = []
    for key in value_keys:
        values.extend(_text_list(fact.get(key)))

    evidence = fact.get("evidence")
    if isinstance(evidence, dict):
        values.extend(_text_list(evidence.get("label")))
        values.extend(_text_list(evidence.get("labels")))
    elif isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                values.extend(_text_list(item.get("label")))
                values.extend(_text_list(item.get("neighbor_label")))
                values.extend(_text_list(item.get("object_label")))
            else:
                values.extend(_text_list(item))

    return _dedupe(values)


def _image_constraint_terms(reasoning: dict[str, Any]) -> list[str]:
    constraints = _as_dict(reasoning.get("modality_constraints"))
    image_constraints = _as_dict(constraints.get("image"))
    terms: list[str] = []

    for key in ("must_include", "should_include", "context_only"):
        terms.extend(_constraint_terms(image_constraints.get(key)))

    for key in ("must_include", "required_visual_terms"):
        terms.extend(_constraint_terms(constraints.get(key)))

    return _dedupe(terms)


def _constraint_terms(value: Any, *, limit: int = 12) -> list[str]:
    terms: list[str] = []
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, dict):
            terms.extend(_constraint_dict_terms(item))
        else:
            terms.extend(_text_list(item))
        if len(terms) >= limit:
            break
    return _dedupe(terms)[:limit]


def _constraint_dict_terms(constraint: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    relation = _first_text(
        constraint.get("relation"),
        constraint.get("relation_label"),
        constraint.get("predicate_label"),
        constraint.get("property_label"),
    )
    labels = _constraint_label_values(constraint)
    if relation and labels:
        terms.append(f"{relation}: {', '.join(labels)}")
    elif relation:
        terms.append(relation)
    else:
        terms.extend(_text_list(constraint.get("constraint")))
        terms.extend(_text_list(constraint.get("text")))
        terms.extend(labels)

    terms.extend(_constraint_evidence_triple_terms(constraint.get("evidence_triples")))
    return _dedupe(terms)


def _constraint_label_values(constraint: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "label",
        "value",
        "object_label",
        "neighbor_label",
        "tail_label",
        "head_label",
        "required_term",
    ):
        values.extend(_text_list(constraint.get(key)))
    return _dedupe(values)


def _constraint_evidence_triple_terms(value: Any, *, limit: int = 3) -> list[str]:
    terms: list[str] = []
    for triple in _as_list(value)[:limit]:
        if not isinstance(triple, dict):
            terms.extend(_text_list(triple))
            continue
        relation = _first_text(
            triple.get("relation"),
            triple.get("relation_label"),
            triple.get("predicate_label"),
            triple.get("property_label"),
        )
        labels = _constraint_label_values(triple)
        if relation and labels:
            terms.append(f"{relation}: {', '.join(labels)}")
        elif relation:
            terms.append(relation)
        elif labels:
            terms.extend(labels)
    return _dedupe(terms)


def _candidate_id(candidate: dict[str, Any]) -> str:
    return _first_text(candidate.get("candidate_id"), candidate.get("image_candidate_id"))


def _resolve_entity_image_path(
    artifacts: RunEntityArtifacts,
    image_candidate: dict[str, Any],
    *,
    error_cls: type[Exception] = ImageQualityTargetError,
) -> Path:
    generated_content = _as_dict(image_candidate.get("generated_content"))
    raw_path = generated_content.get("image_path")
    if not raw_path:
        raise error_cls(
            f"image candidate {_candidate_id(image_candidate)} has no image_path"
        )

    entity_dir = artifacts.entity_dir.resolve()
    candidate_path = Path(str(raw_path))
    if not candidate_path.is_absolute():
        candidate_path = entity_dir / candidate_path
    resolved = candidate_path.resolve()

    try:
        resolved.relative_to(entity_dir)
    except ValueError as exc:
        raise error_cls(
            f"image path must stay under entity directory {entity_dir}: {resolved}"
        ) from exc

    if not resolved.is_file():
        raise error_cls(f"image file does not exist: {resolved}")

    return resolved


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _prompt_text(image_candidate: dict[str, Any]) -> str:
    generated_content = _as_dict(image_candidate.get("generated_content"))
    return _first_text(
        image_candidate.get("positive_prompt"),
        generated_content.get("prompt"),
        generated_content.get("value"),
    )


def _candidate_text_for_image(
    artifacts: RunEntityArtifacts,
    image_candidate_id: str,
) -> str:
    text_candidates = _ranked_text_candidates_by_id(artifacts)
    pair_ranking = _as_dict(artifacts.ranking.get("pair_ranking"))
    for pair in _as_list(pair_ranking.get("ranked_pairs")):
        if not isinstance(pair, dict):
            continue
        image_candidate = _as_dict(pair.get("image_candidate"))
        pair_image_id = _first_text(
            image_candidate.get("candidate_id"),
            image_candidate.get("image_candidate_id"),
            pair.get("image_candidate_id"),
        )
        if pair_image_id != image_candidate_id:
            continue
        text = _text_from_candidate(_as_dict(pair.get("text_candidate")))
        if text:
            return text
        text_candidate_id = _first_text(
            pair.get("text_candidate_id"),
            _as_dict(pair.get("text_candidate")).get("candidate_id"),
        )
        text = _text_from_candidate(text_candidates.get(text_candidate_id, {}))
        if text:
            return text

    for text_candidate in text_candidates.values():
        text = _text_from_candidate(text_candidate)
        if text:
            return text
    return ""


def _ranked_text_candidates_by_id(
    artifacts: RunEntityArtifacts,
) -> dict[str, dict[str, Any]]:
    text_ranking = _as_dict(artifacts.ranking.get("text_ranking"))
    candidates: dict[str, dict[str, Any]] = {}
    for index, text_candidate in enumerate(
        _as_list(text_ranking.get("ranked_candidates")),
    ):
        if not isinstance(text_candidate, dict):
            continue
        candidate_id = _first_text(text_candidate.get("candidate_id"), str(index))
        candidates[candidate_id] = text_candidate
    return candidates


def _text_from_candidate(candidate: dict[str, Any]) -> str:
    generated_content = _as_dict(candidate.get("generated_content"))
    return _first_text(
        generated_content.get("value"),
        candidate.get("text"),
        candidate.get("value"),
    )
