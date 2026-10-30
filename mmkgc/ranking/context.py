"""Leak-resistant ranking context and positive reference builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from mmkgc.ranking.contracts import freeze_value
from mmkgc.ranking.image_input import ImageInput
from mmkgc.observation import validate_verification_targets


CONTEXT_SCHEMA_VERSION = "ranking_context_v1"
_MISSING_MODES = frozenset({"text", "image"})

_PROFILE_FIELDS = (
    "core_identity",
    "entity_type",
    "type",
    "secondary_types",
    "confidence",
    "type_confidence",
    "semantic_domain",
    "domain",
    "visual_domain",
    "visual_style",
    "style",
    "safe_visual_style",
    "safe_style",
    "key_attributes",
    "type_evidence",
    "conflicting_evidence",
    "uncertain_aspects",
    "uncertainty",
)
_KEY_ATTRIBUTE_FIELDS = frozenset({"attribute", "attribute_type", "source", "evidence"})
_RELATION_FACT_FIELDS = (
    "fact_id", "source_fact_refs", "evidence_refs", "evidence_match_score",
    "selection_source", "selected_role", "source_plan_id",
    "relation",
    "source_relations",
    "subject",
    "predicate",
    "object",
    "fact",
    "visual_clue",
    "clue",
    "fact_type",
    "evidence_triples",
    "support_level",
    "support",
    "confidence",
    "confidence_level",
    "evidence_strength",
    "coarse_relevance_score",
    "credibility_score",
    "credibility_label",
    "credibility_reason",
    "allowed_use",
    "direction",
    "neighbor_entities",
    "generation_use",
    "risk",
    "relation_role",
    "source_relation_role",
    "role",
    "selection_role",
    "fact_role",
    "priority",
    "constraint_type",
    "status",
    "safety",
    "supported",
    "is_supported",
    "excluded",
    "context_only",
    "uncertain",
)
_SUPPORTED_FACT_FIELDS = (
    "fact_id", "source_fact_refs", "evidence_refs", "evidence_match_score",
    "selection_source", "selected_role", "source_plan_id",
    "fact",
    "visual_clue",
    "clue",
    "source_relations",
    "evidence",
    "evidence_triples",
    "support_level",
    "support",
    "confidence",
    "confidence_level",
    "evidence_strength",
    "credibility_score",
    "credibility_label",
    "credibility_reason",
    "allowed_use",
    "relation_role",
    "source_relation_role",
    "role",
    "selection_role",
    "fact_role",
    "priority",
    "fact_type",
    "constraint_type",
    "status",
    "safety",
    "supported",
    "is_supported",
    "excluded",
    "context_only",
    "uncertain",
)
_SAFE_INFERENCE_FIELDS = (
    "inference",
    "inference_type",
    "based_on_facts",
    "source_relations",
    "evidence_triples",
    "usable_for",
    "confidence",
    "modality_usage",
    "risk",
    "reason",
)
_FACT_SEQUENCE_FIELDS = {
    "source_fact_refs", "evidence_refs",
    "source_relations",
    "evidence",
    "evidence_triples",
    "neighbor_entities",
    "based_on_facts",
    "usable_for",
}
_FACT_BOOL_FIELDS = {
    "supported",
    "is_supported",
    "excluded",
    "context_only",
    "uncertain",
}
_FACT_ROLE_FIELDS = (
    "relation_role",
    "source_relation_role",
    "role",
    "selection_role",
    "fact_role",
    "priority",
)
_CONSTRAINT_FIELDS = (
    "selection_source", "source_fact_refs", "evidence_refs",
    "constraint",
    "constraint_type",
    "source",
    "source_relations",
    "evidence_triples",
    "confidence",
    "reason",
    "severity",
    "gate_action",
    "violation_code",
    "core",
    "evidence_policy",
    "safe",
    "safety",
    "status",
    "role",
    "excluded",
    "context_only",
    "uncertain",
)
_CONSTRAINT_BUCKETS = ("must_include", "should_include", "context_only", "avoid")
_CANDIDATE_FIELDS = (
    "candidate_id",
    "source_candidate_id",
    "verification_targets",
    "target_modality",
    "primary_relation",
    "supporting_relations",
    "relation_role",
    "generation_angle",
    "used_evidence_triples",
    "expected_relation_coverage",
    "expected_relation_coverage_detail",
    "generated_content",
    "applied_forbidden_content",
    "fallback",
    "status",
    "metadata",
)
_CANDIDATE_STRING_FIELDS = {
    "candidate_id",
    "source_candidate_id",
    "target_modality",
    "primary_relation",
    "relation_role",
    "generation_angle",
    "status",
}
_CANDIDATE_SEQUENCE_FIELDS = {
    "supporting_relations",
    "used_evidence_triples",
    "expected_relation_coverage",
    "applied_forbidden_content",
}
_METADATA_FIELDS = (
    "relation_role",
    "fallback",
    "entity_type",
    "status",
    "generator",
    "model_name",
    "model_version",
    "fallback_used",
    "error_type",
    "provider",
    "width",
    "height",
    "steps",
    "seed",
    "stage3_regeneration",
    "regeneration_attempt",
)
_IMAGE_CONTENT_FIELDS = ("type", "status", "image_artifact_id", "artifact_id")
_NON_POSITIVE_LABELS = {
    "avoid",
    "context",
    "context-only",
    "context_only",
    "excluded",
    "fallback",
    "future",
    "low",
    "rejected",
    "tertiary",
    "uncertain",
    "unknown",
    "unsafe",
    "unsupported",
    "weak",
}
_POSITIVE_SUPPORT_LABELS = frozenset(
    {"moderate", "medium", "strong", "high", "supported", "verified"}
)
_POSITIVE_STATUS_LABELS = frozenset(
    {"supported", "verified", "confirmed", "approved", "accepted", "valid"}
)
_POSITIVE_SAFETY_LABELS = frozenset({"safe", "verified", "approved", "allowed"})
_POSITIVE_CONFIDENCE_LABELS = frozenset({"moderate", "medium", "strong", "high"})
_FACT_ROLES = frozenset({"primary", "secondary"})
_POSITIVE_CONSTRAINT_TYPES = frozenset({"identity", "grounded_fact"})
_NEGATIVE_CONSTRAINT_TYPES = frozenset({"context", "avoid_unsupported_detail"})
_CONSTRAINT_SEVERITIES = frozenset({"hard", "soft"})
_CONSTRAINT_GATE_ACTIONS = frozenset(
    {
        "hard_required",
        "soft_required",
        "prefer",
        "context_only",
        "hard_forbidden",
        "avoid_or_context_only",
    }
)
_EXACT_PATH_TYPES = frozenset({type(Path("."))})


def _exact_string(value: str) -> str:
    return value if type(value) is str else str.__str__(value)


def _exact_number(value: int | float) -> int | float:
    if type(value) in {int, float}:
        return value
    if isinstance(value, int):
        return int.__int__(value)
    return float.__float__(value)


def _json_scalar(value: Any) -> bool:
    if value is None or type(value) is bool:
        return True
    if isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and math.isfinite(float(_exact_number(value)))


def _validated_context_key(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = _exact_string(value)
    if not str.strip(normalized):
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _validated_missing_mode(value: Any) -> str:
    missing_mode = _validated_context_key("missing_mode", value)
    if missing_mode not in _MISSING_MODES:
        raise ValueError("missing_mode must be one of: text, image")
    return missing_mode


def _clean_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = str.strip(_exact_string(value))
    return normalized or None


def _scalar_sequence(value: Any) -> tuple[str | int | float | bool | None, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if _json_scalar(item))


def _profile_mapping_sequence(value: Any, allowed_fields: frozenset[str] | None = None) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        row: dict[str, Any] = {}
        keys = allowed_fields or frozenset(str(key) for key in item)
        for key in keys:
            if key not in item:
                continue
            field_value = item.get(key)
            if _json_scalar(field_value):
                row[key] = field_value
            elif isinstance(field_value, (list, tuple)):
                sequence = _scalar_sequence(field_value)
                if sequence:
                    row[key] = sequence
        if row:
            rows.append(row)
    return tuple(rows)


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(text for item in value if (text := _clean_string(item)) is not None)


def _strict_provenance_sequence(
    value: Any, *, modalities: bool = False, allow_empty: bool = False
) -> tuple[str, ...] | None:
    """Validate a provenance sequence atomically; never filter individual leaves."""
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        return None
    if any(
        not isinstance(item, str) or not str.strip(_exact_string(item))
        for item in value
    ):
        return None
    result = tuple(str.strip(_exact_string(item)) for item in value)
    if modalities and any(item not in {"text", "image"} for item in result):
        return None
    return result


def _sanitize_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in _PROFILE_FIELDS:
        item = value.get(key)
        if key == "key_attributes":
            mappings = _profile_mapping_sequence(item, _KEY_ATTRIBUTE_FIELDS)
            if mappings:
                result[key] = mappings
            else:
                sequence = _scalar_sequence(item)
                if sequence:
                    result[key] = sequence
        elif key in {"secondary_types", "semantic_domain", "type_evidence", "uncertainty"}:
            sequence = _scalar_sequence(item)
            if sequence:
                result[key] = sequence
            elif (text := _clean_string(item)) is not None:
                result[key] = text
        elif key in {"conflicting_evidence", "uncertain_aspects"}:
            mappings = _profile_mapping_sequence(item)
            if mappings:
                result[key] = mappings
        elif key == "confidence":
            if item is not None and _json_scalar(item):
                result[key] = item
            elif item is not None:
                raise TypeError(
                    f"entity_profile.confidence must be a finite JSON scalar; got {type(item).__name__}"
                )
        elif (text := _clean_string(item)) is not None:
            result[key] = text
    return result


def _sanitize_flat_fact(
    value: Any,
    field_order: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in field_order:
        if key not in value:
            continue
        item = value.get(key)
        if key == "source_relations":
            sequence = _strict_provenance_sequence(
                item,
            )
            if sequence is not None:
                result[key] = sequence
            elif item is not None:
                return {}
        elif key in _FACT_SEQUENCE_FIELDS:
            sequence = _strict_provenance_sequence(item)
            if sequence is not None:
                result[key] = sequence
            elif item is not None:
                return {}
        elif key in _FACT_BOOL_FIELDS:
            if isinstance(item, bool):
                result[key] = item
            elif item is not None:
                return {}
        elif key == "status":
            if (
                isinstance(item, str)
                and _normalized_label(item) in _POSITIVE_STATUS_LABELS
            ):
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key == "safety":
            if (
                isinstance(item, str)
                and _normalized_label(item) in _POSITIVE_SAFETY_LABELS
            ):
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key in _FACT_ROLE_FIELDS:
            if isinstance(item, str) and _normalized_label(item) in _FACT_ROLES:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key in {
            "support_level",
            "support",
            "confidence",
            "confidence_level",
            "evidence_strength",
        }:
            if _positive_labeled_value(
                item,
                _POSITIVE_SUPPORT_LABELS
                if key in {"support_level", "support", "evidence_strength"}
                else _POSITIVE_CONFIDENCE_LABELS,
                numeric_minimum=0.5,
            ):
                result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
            elif item is not None:
                return {}
        elif _json_scalar(item) and item is not None:
            result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
        elif item is not None and key in {
            "support_level",
            "support",
            "confidence",
            "confidence_level",
            "evidence_strength",
            "status",
            "safety",
            "constraint_type",
            *_FACT_ROLE_FIELDS,
        }:
            return {}
    return result


def _sanitize_relation_facts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sanitized
        for item in value
        if (sanitized := _sanitize_flat_fact(item, _RELATION_FACT_FIELDS))
    )


def _sanitize_supported_facts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sanitized
        for item in value
        if (sanitized := _sanitize_flat_fact(item, _SUPPORTED_FACT_FIELDS))
    )


def _sanitize_safe_inference(value: Any) -> str | dict[str, Any] | None:
    if isinstance(value, str):
        return _clean_string(value)
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for key in _SAFE_INFERENCE_FIELDS:
        item = value.get(key)
        if key in _FACT_SEQUENCE_FIELDS:
            sequence = _strict_provenance_sequence(item)
            if sequence is not None:
                result[key] = sequence
            elif item is not None:
                return None
        elif key == "modality_usage":
            if isinstance(item, Mapping):
                usage = {
                    modality: bucket
                    for modality, bucket in item.items()
                    if modality in {"text", "image"}
                    and bucket in _CONSTRAINT_BUCKETS
                }
                if usage:
                    result[key] = usage
            elif item is not None:
                return None
        elif key == "confidence":
            if _positive_labeled_value(
                item, _POSITIVE_CONFIDENCE_LABELS, numeric_minimum=0.5
            ):
                result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
            elif item is not None:
                return None
        elif _json_scalar(item) and item is not None:
            result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
        elif item is not None:
            return None
    return result or None


def _sanitize_safe_inferences(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sanitized
        for item in value
        if (sanitized := _sanitize_safe_inference(item)) is not None
    )


def _sanitize_constraint(value: Any, bucket: str) -> dict[str, Any]:
    if isinstance(value, str):
        normalized = _clean_string(value)
        return {"constraint": normalized} if normalized is not None else {}
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in _CONSTRAINT_FIELDS:
        item = value.get(key)
        if key in {"source_relations", "evidence_triples"}:
            sequence = _strict_provenance_sequence(item)
            if sequence is not None:
                result[key] = sequence
            elif item is not None:
                return {}
        elif key in {"safe", "excluded", "context_only", "uncertain", "core"}:
            if isinstance(item, bool):
                result[key] = item
            elif item is not None:
                return {}
        elif key == "constraint_type":
            allowed_types = (
                _POSITIVE_CONSTRAINT_TYPES
                if bucket in {"must_include", "should_include"}
                else _NEGATIVE_CONSTRAINT_TYPES
            )
            if isinstance(item, str) and _normalized_label(item) in allowed_types:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key == "confidence":
            if _positive_labeled_value(
                item, _POSITIVE_CONFIDENCE_LABELS, numeric_minimum=0.5
            ):
                result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
            elif item is not None:
                return {}
        elif key == "severity":
            if isinstance(item, str) and _normalized_label(item) in _CONSTRAINT_SEVERITIES:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key == "gate_action":
            if isinstance(item, str) and _normalized_label(item) in _CONSTRAINT_GATE_ACTIONS:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key == "status":
            if isinstance(item, str) and _normalized_label(item) in _POSITIVE_STATUS_LABELS:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key == "safety":
            if isinstance(item, str) and _normalized_label(item) in _POSITIVE_SAFETY_LABELS:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif key == "role":
            if isinstance(item, str) and _normalized_label(item) in _FACT_ROLES:
                result[key] = str.strip(_exact_string(item))
            elif item is not None:
                return {}
        elif _json_scalar(item) and item is not None:
            result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
        elif item is not None and key in {
            "confidence",
            "safety",
            "status",
            "constraint_type",
            "role",
        }:
            return {}
    return result


def _sanitize_modality_constraints(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for modality in ("text", "image"):
        modality_value = value.get(modality)
        if not isinstance(modality_value, Mapping):
            continue
        buckets: dict[str, Any] = {}
        for bucket in _CONSTRAINT_BUCKETS:
            items = modality_value.get(bucket)
            if not isinstance(items, (list, tuple)):
                continue
            buckets[bucket] = tuple(
                sanitized
                for item in items
                if (sanitized := _sanitize_constraint(item, bucket))
            )
        result[modality] = buckets
    return result


def _sanitize_forbidden_content(value: Any) -> tuple[str, ...]:
    return _string_sequence(value)


def _sanitize_expected_detail(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    if "primary" in value:
        primary = _clean_string(value.get("primary"))
        if primary is None:
            return None
        result["primary"] = primary
    if "supporting" in value:
        supporting = _strict_provenance_sequence(value.get("supporting"), allow_empty=True)
        if supporting is None:
            return None
        result["supporting"] = supporting
    return result


def _sanitize_generated_content(value: Any, modality: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    content_type = _clean_string(value.get("type"))
    result: dict[str, Any] = {}
    if modality == "text":
        if content_type not in {None, "text"}:
            return {}
        if content_type == "text":
            result["type"] = "text"
        if (generated_text := _clean_string(value.get("value"))) is not None:
            result["value"] = generated_text
        return result
    for key in _IMAGE_CONTENT_FIELDS:
        item = value.get(key)
        if key == "type":
            if content_type not in {"image", "image_prompt"}:
                continue
            result[key] = content_type
            continue
        if _json_scalar(item) and item is not None:
            result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
    return result


def _sanitize_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in _METADATA_FIELDS:
        item = value.get(key)
        if _json_scalar(item) and item is not None:
            result[key] = str.strip(_exact_string(item)) if isinstance(item, str) else item
    return result


def _sanitize_candidate(value: Any, modality: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in _CANDIDATE_FIELDS:
        item = value.get(key)
        if key == "target_modality":
            result[key] = modality
        elif key == "verification_targets" and item is not None:
            result[key] = validate_verification_targets(item)
        elif key in _CANDIDATE_STRING_FIELDS:
            if (text := _clean_string(item)) is not None:
                result[key] = text
        elif key in _CANDIDATE_SEQUENCE_FIELDS:
            sequence = _strict_provenance_sequence(item, allow_empty=key == "supporting_relations")
            if sequence is not None:
                result[key] = sequence
            elif item is not None:
                return {}
        elif key == "expected_relation_coverage_detail":
            detail = _sanitize_expected_detail(item)
            if detail is None:
                if item is not None:
                    return {}
            elif detail:
                result[key] = detail
        elif key == "generated_content":
            content = _sanitize_generated_content(item, modality)
            if content:
                result[key] = content
        elif key == "metadata":
            metadata = _sanitize_metadata(item)
            if metadata:
                result[key] = metadata
        elif key == "fallback" and isinstance(item, bool):
            result[key] = item
    return result


def _sanitize_candidates(value: Any, modality: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        sanitized
        for item in value
        if (sanitized := _sanitize_candidate(item, modality))
    )


def _candidate_relation_roles(*candidate_groups: Any) -> dict[str, str]:
    roles: dict[str, str] = {}
    for group in candidate_groups:
        if not isinstance(group, (list, tuple)):
            continue
        for candidate in group:
            if not isinstance(candidate, Mapping):
                continue
            relation = _clean_string(candidate.get("primary_relation"))
            metadata = candidate.get("metadata")
            metadata_role = metadata.get("relation_role") if isinstance(metadata, Mapping) else None
            role = _clean_string(metadata_role)
            if relation is None or role not in {"primary", "secondary"}:
                continue
            current = roles.get(relation)
            if current is None or (current == "secondary" and role == "primary"):
                roles[relation] = role
    return roles


def _enrich_fact_provenance(
    relation_facts: tuple[dict[str, Any], ...],
    supported_facts: tuple[dict[str, Any], ...],
    relation_roles: Mapping[str, str],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    relation_index: dict[str, dict[str, Any]] = {}
    planned_facts = {
        item["fact_id"]: item for item in relation_facts
        if item.get("selection_source") == "stage2_fact_plan"
        and item.get("selected_role") in {"primary", "secondary"} and item.get("fact_id")
    }
    invalid_relations: set[str] = set()
    enriched_relations = []
    for fact in relation_facts:
        item = dict(fact)
        item_invalid = bool(item.pop("_invalid_provenance", False)) or bool(
            item.pop("_invalid_semantics", False)
        )
        for field_name in _FACT_ROLE_FIELDS:
            item.pop(field_name, None)
        if item.get("fact_id") in planned_facts:
            item["relation_role"] = item["selected_role"]
        relation = _clean_string(item.get("relation"))
        if relation is not None:
            relation_index[relation] = item
            if item_invalid:
                invalid_relations.add(relation)
            if "relation_role" not in item and relation in relation_roles:
                item["relation_role"] = relation_roles[relation]
        enriched_relations.append(item)

    enriched_supported = []
    for fact in supported_facts:
        item = dict(fact)
        item_invalid = bool(item.pop("_invalid_provenance", False)) or bool(
            item.pop("_invalid_semantics", False)
        )
        for field_name in _FACT_ROLE_FIELDS:
            item.pop(field_name, None)
        if item.get("selection_source") == "stage2_fact_plan":
            refs = item.get("source_fact_refs", ())
            matched_plans = [planned_facts[ref] for ref in refs if ref in planned_facts]
            if refs and len(matched_plans) == len(refs) and not item_invalid:
                role = "primary" if any(plan["selected_role"] == "primary" for plan in matched_plans) else "secondary"
                item["relation_role"] = item["source_relation_role"] = role
            enriched_supported.append(item)
            continue
        declared_sources = item.get("source_relations")
        source_relations = _string_sequence(declared_sources)
        valid_source_declaration = isinstance(declared_sources, (list, tuple)) and len(
            source_relations
        ) == len(declared_sources)
        matched = [relation_index[relation] for relation in source_relations if relation in relation_index]
        complete_sources = (
            bool(source_relations)
            and valid_source_declaration
            and len(matched) == len(source_relations)
            and not item_invalid
            and not any(relation in invalid_relations for relation in source_relations)
        )
        matched_roles = [relation_roles.get(relation) for relation in source_relations]
        complete_role_provenance = complete_sources and all(
            role in {"primary", "secondary"} for role in matched_roles
        )
        if complete_role_provenance and "primary" in matched_roles:
            item["relation_role"] = "primary"
            item["source_relation_role"] = "primary"
        elif complete_role_provenance and all(role == "secondary" for role in matched_roles):
            item["relation_role"] = "secondary"
            item["source_relation_role"] = "secondary"
        enriched_supported.append(item)
    return tuple(enriched_relations), tuple(enriched_supported)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key if type(key) is str else _exact_string(key): _thaw_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class _TransportFrozenDict(dict):
    """Immutable mapping for opaque image transport values.

    Transport values are deliberately outside the JSON context contract: image
    paths, bytes, and adapter-owned objects may be retained here, but this map
    is never emitted by ``to_prompt_payload``.
    """

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("opaque transport mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze_transport_value(value: Any, path: str = "$") -> Any:
    """Detach trusted transport containers without admitting mutable aliases."""
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"opaque transport key at {path} must be a string; got {type(key).__name__}"
                )
            if type(key) is not str:
                raise TypeError(f"opaque transport key at {path} must be an exact string")
            items[key] = _freeze_transport_value(item, f"{path}[{key!r}]")
        frozen = _TransportFrozenDict()
        dict.__init__(frozen, items)
        return frozen
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_transport_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"opaque transport set is not deterministic at {path}")
    if value is None or type(value) is bool:
        return value
    if type(value) is str or type(value) is bytes or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"opaque transport numeric value at {path} must be finite")
        return value
    if type(value) is ImageInput:
        return value
    if isinstance(value, Path):
        if type(value) in _EXACT_PATH_TYPES:
            return value
        raise TypeError(f"opaque transport value at {path} must not be a Path subclass")
    if isinstance(value, (str, bytes, int, float)):
        raise TypeError(f"opaque transport value at {path} must not be a scalar subclass")
    raise TypeError(
        f"opaque transport value at {path} must be a trusted immutable value"
    )


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping; got {type(value).__name__}")
    return value


def _require_sequence(name: str, value: Any) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple; got {type(value).__name__}")
    return value


def _optional_mapping(parent_name: str, parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in parent:
        return {}
    return _require_mapping(f"{parent_name}.{key}", parent[key])


def _optional_sequence(parent_name: str, parent: Mapping[str, Any], key: str) -> list[Any] | tuple[Any, ...]:
    if key not in parent:
        return ()
    return _require_sequence(f"{parent_name}.{key}", parent[key])


@dataclass(frozen=True)
class RankingContext:
    entity_key: str
    entity_label: str
    missing_mode: str
    existing_modalities: dict[str, str | None]
    entity_profile: dict[str, Any]
    relation_facts: tuple[dict[str, Any], ...]
    supported_facts: tuple[dict[str, Any], ...]
    safe_inferences: tuple[Any, ...]
    modality_constraints: dict[str, Any]
    forbidden_content: tuple[Any, ...]
    text_candidates: tuple[dict[str, Any], ...]
    image_candidates: tuple[dict[str, Any], ...]
    generated_image_inputs: dict[str, Any]
    existing_image_input: Any | None = None
    ground_truth_used: bool = field(default=False, init=False)
    context_schema_version: str = field(default=CONTEXT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        entity_key = _validated_context_key("entity_key", self.entity_key)
        entity_label = _validated_context_key("entity_label", self.entity_label)
        missing_mode = _validated_missing_mode(self.missing_mode)
        object.__setattr__(self, "entity_key", entity_key)
        object.__setattr__(self, "entity_label", entity_label)
        object.__setattr__(self, "missing_mode", missing_mode)
        _require_mapping("existing_modalities", self.existing_modalities)
        _require_mapping("entity_profile", self.entity_profile)
        _require_mapping("modality_constraints", self.modality_constraints)
        _require_mapping("generated_image_inputs", self.generated_image_inputs)
        for field_name, field_value in (
            ("relation_facts", self.relation_facts),
            ("supported_facts", self.supported_facts),
            ("safe_inferences", self.safe_inferences),
            ("forbidden_content", self.forbidden_content),
            ("text_candidates", self.text_candidates),
            ("image_candidates", self.image_candidates),
        ):
            _require_sequence(field_name, field_value)
        allowed_modalities = {self.missing_mode}
        text_candidates = (
            _sanitize_candidates(self.text_candidates, "text")
            if "text" in allowed_modalities
            else ()
        )
        image_candidates = (
            _sanitize_candidates(self.image_candidates, "image")
            if "image" in allowed_modalities
            else ()
        )
        relation_facts, supported_facts = _enrich_fact_provenance(
            _sanitize_relation_facts(self.relation_facts),
            _sanitize_supported_facts(self.supported_facts),
            _candidate_relation_roles(text_candidates, image_candidates),
        )
        existing_modalities = self.existing_modalities
        safe_existing: dict[str, str | None] = {}
        for modality in ("text", "image"):
            value = existing_modalities.get(modality)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"existing_modalities.{modality} must be a string or None; got {type(value).__name__}"
                )
            safe_existing[modality] = value if modality in allowed_modalities else None
        object.__setattr__(self, "existing_modalities", freeze_value(safe_existing))
        object.__setattr__(self, "entity_profile", freeze_value(_sanitize_profile(self.entity_profile)))
        object.__setattr__(
            self,
            "relation_facts",
            freeze_value(relation_facts),
        )
        object.__setattr__(
            self,
            "supported_facts",
            freeze_value(supported_facts),
        )
        object.__setattr__(
            self,
            "safe_inferences",
            freeze_value(_sanitize_safe_inferences(self.safe_inferences)),
        )
        object.__setattr__(
            self,
            "modality_constraints",
            freeze_value(
                {
                    modality: buckets
                    for modality, buckets in _sanitize_modality_constraints(
                        self.modality_constraints
                    ).items()
                    if modality in allowed_modalities
                }
            ),
        )
        object.__setattr__(
            self,
            "forbidden_content",
            freeze_value(_sanitize_forbidden_content(self.forbidden_content)),
        )
        object.__setattr__(
            self,
            "text_candidates",
            freeze_value(text_candidates),
        )
        object.__setattr__(
            self,
            "image_candidates",
            freeze_value(image_candidates),
        )
        object.__setattr__(
            self,
            "generated_image_inputs",
            _freeze_transport_value(self.generated_image_inputs)
            if "image" in allowed_modalities
            else _freeze_transport_value({}),
        )
        object.__setattr__(
            self,
            "existing_image_input",
            _freeze_transport_value(self.existing_image_input)
            if "image" in allowed_modalities or self.missing_mode == "text"
            else None,
        )

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return a detached, JSON-safe context for Judge prompt construction."""
        existing_text = self.existing_modalities.get("text")
        metadata = {
            "ground_truth_used": self.ground_truth_used,
            "context_schema_version": self.context_schema_version,
        }
        return {
            "entity_key": self.entity_key,
            "entity_label": self.entity_label,
            "missing_mode": self.missing_mode,
            "existing_modalities": {"text": existing_text} if existing_text is not None else {},
            "has_existing_image": bool(
                self.existing_image_input is not None
                or self.existing_modalities.get("image") is not None
            ),
            "entity_profile": _thaw_json(self.entity_profile),
            "relation_facts": _thaw_json(self.relation_facts),
            "supported_facts": _thaw_json(self.supported_facts),
            "safe_inferences": _thaw_json(self.safe_inferences),
            "modality_constraints": _thaw_json(self.modality_constraints),
            "forbidden_content": _thaw_json(self.forbidden_content),
            "ground_truth_used": self.ground_truth_used,
            "context_schema_version": self.context_schema_version,
            "metadata": metadata,
        }


def build_ranking_context(
    entity_key: str,
    entity_label: str,
    missing_mode: str,
    existing_modalities: Mapping[str, Any],
    knowledge: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    generation: Mapping[str, Any],
    generated_image_inputs: Mapping[str, Any],
    existing_image_input: Any | None = None,
) -> RankingContext:
    """Build a context by explicitly extracting allowed Stage 1/2/4 fields."""
    entity_key = _validated_context_key("entity_key", entity_key)
    entity_label = _validated_context_key("entity_label", entity_label)
    missing_mode = _validated_missing_mode(missing_mode)
    existing_modalities = _require_mapping("existing_modalities", existing_modalities)
    knowledge = _require_mapping("knowledge", knowledge)
    reasoning = _require_mapping("reasoning", reasoning)
    generation = _require_mapping("generation", generation)
    generated_image_inputs = _require_mapping(
        "generated_image_inputs", generated_image_inputs
    )

    entity_profile = _optional_mapping("knowledge", knowledge, "entity_profile")
    raw_relation_facts = _optional_sequence("knowledge", knowledge, "relation_facts")
    raw_supported_facts = _optional_sequence("reasoning", reasoning, "supported_facts")
    raw_safe_inferences = _optional_sequence("reasoning", reasoning, "safe_inferences")
    raw_forbidden_content = _optional_sequence("reasoning", reasoning, "forbidden_content")
    modality_constraints = _optional_mapping(
        "reasoning", reasoning, "modality_constraints"
    )
    stage_four = _optional_mapping(
        "generation", generation, "relation_alternation_generation"
    )
    raw_text_candidates = _optional_sequence(
        "generation.relation_alternation_generation", stage_four, "text_candidates"
    )
    raw_image_candidates = _optional_sequence(
        "generation.relation_alternation_generation", stage_four, "image_candidates"
    )
    return RankingContext(
        entity_key=entity_key,
        entity_label=entity_label,
        missing_mode=missing_mode,
        existing_modalities=existing_modalities,
        entity_profile=entity_profile,
        relation_facts=raw_relation_facts,
        supported_facts=raw_supported_facts,
        safe_inferences=raw_safe_inferences,
        modality_constraints=modality_constraints,
        forbidden_content=raw_forbidden_content,
        text_candidates=raw_text_candidates,
        image_candidates=raw_image_candidates,
        generated_image_inputs=generated_image_inputs,
        existing_image_input=existing_image_input,
    )


def candidate_generated_text(candidate: Mapping[str, Any]) -> str:
    """Return only a candidate's generated text value, or a safe empty result."""
    content = candidate.get("generated_content")
    if not isinstance(content, Mapping):
        return ""
    return _clean_string(content.get("value")) or ""


def _normalized_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = _exact_string(value)
    return str.replace(str.casefold(str.strip(normalized)), " ", "_")


def _is_non_positive_label(value: Any) -> bool:
    normalized = _normalized_label(value)
    return normalized in _NON_POSITIVE_LABELS or normalized.startswith(
        (
            "avoid_",
            "context_",
            "excluded_",
            "fallback_",
            "future_",
            "uncertain_",
            "unknown_",
        )
    ) or normalized.endswith("_unknown")


def _fact_role(item: Mapping[str, Any]) -> str:
    for key in _FACT_ROLE_FIELDS:
        if item.get(key) is not None:
            return _normalized_label(item[key])
    return ""


def _positive_labeled_value(
    value: Any,
    labels: frozenset[str],
    *,
    numeric_minimum: float | None = None,
) -> bool:
    if isinstance(value, str):
        return _normalized_label(value) in labels
    if (
        numeric_minimum is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        numeric = float(_exact_number(value))
        return math.isfinite(numeric) and numeric_minimum <= numeric <= 1.0
    return False


def _fact_is_supported(item: Mapping[str, Any]) -> bool:
    if item.get("_invalid_semantics") is True:
        return False
    for key in ("supported", "is_supported"):
        if key in item and item[key] is not True:
            return False
    for key in ("excluded", "context_only", "uncertain"):
        if key in item and item[key] is not False:
            return False
    if item.get("selection_source") == "stage2_fact_plan":
        return _fact_role(item) in {"primary", "secondary"}

    support_values = [
        item[key]
        for key in ("support_level", "support")
        if key in item and item[key] is not None
    ]
    if not support_values or not all(
        _positive_labeled_value(
            value,
            _POSITIVE_SUPPORT_LABELS,
            numeric_minimum=0.5,
        )
        for value in support_values
    ):
        return False

    for key, labels, numeric_minimum in (
        ("status", _POSITIVE_STATUS_LABELS, None),
        ("safety", _POSITIVE_SAFETY_LABELS, None),
        ("confidence", _POSITIVE_CONFIDENCE_LABELS, 0.5),
        ("confidence_level", _POSITIVE_CONFIDENCE_LABELS, 0.5),
        ("evidence_strength", _POSITIVE_SUPPORT_LABELS, 0.5),
    ):
        if key in item and item[key] is not None and not _positive_labeled_value(
            item[key],
            labels,
            numeric_minimum=numeric_minimum,
        ):
            return False
    return True


def _secondary_is_strong(item: Mapping[str, Any]) -> bool:
    if item.get("selection_source") == "stage2_fact_plan":
        return _fact_role(item) == "secondary"
    for key in (
        "support_level",
        "support",
        "confidence",
        "confidence_level",
        "evidence_strength",
    ):
        value = item.get(key)
        if isinstance(value, str) and _normalized_label(value) in {"high", "strong"}:
            return True
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0.75:
            return True
    return False


def _fact_text(item: Mapping[str, Any]) -> str:
    for key in ("fact", "visual_clue", "clue"):
        if (value := _clean_string(item.get(key))) is not None:
            return value
    return ""


def _forbidden_strings(context: RankingContext) -> tuple[str, ...]:
    return tuple(
        str.casefold(_exact_string(item))
        for item in context.forbidden_content
        if isinstance(item, str)
    )


def _is_safe_positive_text(
    text: str,
    context: RankingContext,
    item: Mapping[str, Any] | None = None,
) -> bool:
    stripped = str.strip(_exact_string(text))
    if not stripped:
        return False
    entity_type_prefix = "entity type:"
    if str.startswith(str.casefold(stripped), entity_type_prefix) and _is_non_positive_label(
        stripped[len(entity_type_prefix) :]
    ):
        return False
    if item is not None:
        if item.get("safe") is False or any(
            item.get(key) is True for key in ("excluded", "context_only", "uncertain")
        ):
            return False
        for key in ("safety", "status", "confidence", "constraint_type", "role"):
            if item.get(key) is not None and _is_non_positive_label(item[key]):
                return False
    normalized = str.casefold(stripped)
    return not any(forbidden in normalized for forbidden in _forbidden_strings(context))


def _constraint_is_positive(item: Mapping[str, Any]) -> bool:
    if item.get("_invalid_semantics") is True:
        return False
    if "safe" in item and item["safe"] is not True:
        return False
    for key in ("excluded", "context_only", "uncertain"):
        if key in item and item[key] is not False:
            return False
    if item.get("selection_source") == "stage2_fact_plan":
        return item.get("constraint_type") != "context"
    confidence = item.get("confidence")
    if not _positive_labeled_value(
        confidence,
        _POSITIVE_CONFIDENCE_LABELS,
        numeric_minimum=0.5,
    ):
        return False
    for key, labels in (
        ("status", _POSITIVE_STATUS_LABELS),
        ("safety", _POSITIVE_SAFETY_LABELS),
    ):
        if key in item and item[key] is not None and not _positive_labeled_value(
            item[key], labels
        ):
            return False
    return True


def _constraint_texts(context: RankingContext, modality: str, bucket: str) -> list[str]:
    modality_data = context.modality_constraints.get(modality, {})
    if not isinstance(modality_data, Mapping):
        return []
    items = modality_data.get(bucket, ())
    if not isinstance(items, (list, tuple)):
        return []
    result = []
    for item in items:
        metadata = item if isinstance(item, Mapping) else None
        value = item.get("constraint") if metadata is not None else item
        if (
            metadata is not None
            and _constraint_is_positive(metadata)
            and isinstance(value, str)
            and _is_safe_positive_text(value, context, metadata)
        ):
            result.append(str.strip(_exact_string(value)))
    return result


def _append_stable(parts: list[str], values: list[str] | tuple[str, ...]) -> None:
    seen = {str.casefold(_exact_string(item)) for item in parts}
    for value in values:
        text = str.strip(_exact_string(value))
        normalized = str.casefold(text)
        if text and normalized not in seen:
            parts.append(text)
            seen.add(normalized)


def _supported_reference_facts(context: RankingContext, modality: str) -> list[str]:
    # Stage 2 is the only authority for positive supported facts.  In
    # particular, an empty Stage-2 result must not silently promote Stage-1
    # relation facts into ranking references.
    source = context.supported_facts
    result = []
    for item in source:
        role = _fact_role(item)
        if role not in {"primary", "secondary"}:
            continue
        if not _fact_is_supported(item):
            continue
        if role == "secondary" and not _secondary_is_strong(item):
            continue
        text = _fact_text(item)
        if text and _is_safe_positive_text(text, context, item):
            result.append(text)
    return result


def build_kg_reference(context: RankingContext, modality: str) -> str:
    """Build a stable, positive-only KG reference for one target modality."""
    if not isinstance(modality, str):
        raise ValueError("modality must be one of: text, image")
    modality = _exact_string(modality)
    if modality not in {"text", "image"}:
        raise ValueError("modality must be one of: text, image")
    if modality != context.missing_mode:
        raise ValueError("modality is excluded by context.missing_mode")
    parts: list[str] = []
    profile_confidence = context.entity_profile.get("confidence")
    profile_is_positive = "confidence" not in context.entity_profile or (
        profile_confidence is not None
        and _positive_labeled_value(
            profile_confidence,
            _POSITIVE_CONFIDENCE_LABELS,
            numeric_minimum=0.5,
        )
    )
    profile_identity = context.entity_profile.get("core_identity")
    identity = (
        profile_identity
        if profile_is_positive
        and isinstance(profile_identity, str)
        and not _is_non_positive_label(profile_identity)
        else context.entity_label
    )
    entity_type = context.entity_profile.get("entity_type") or context.entity_profile.get("type")
    if isinstance(identity, str) and _is_safe_positive_text(identity, context):
        _append_stable(parts, [identity])
    if (
        profile_is_positive
        and isinstance(entity_type, str)
        and str.strip(_exact_string(entity_type))
        and not _is_non_positive_label(entity_type)
    ):
        type_text = f"entity type: {str.strip(_exact_string(entity_type))}"
        if _is_safe_positive_text(type_text, context):
            _append_stable(parts, [type_text])
    _append_stable(parts, _supported_reference_facts(context, modality))
    _append_stable(parts, _constraint_texts(context, modality, "must_include"))
    _append_stable(parts, _constraint_texts(context, modality, "should_include"))
    return " ".join(parts)


def build_structured_kg_triples(
    context: RankingContext,
    modality: str,
) -> list[dict[str, Any]]:
    """Return Stage-2-supported original KG triples for one modality."""
    if modality not in {"text", "image"}:
        raise ValueError("modality must be one of: text, image")
    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for supported in context.supported_facts:
        role = _fact_role(supported)
        if role not in {"primary", "secondary"}:
            continue
        if not _fact_is_supported(supported):
            continue
        if role == "secondary" and not _secondary_is_strong(supported):
            continue
        source_relations = {
            str.casefold(str.strip(_exact_string(value)))
            for value in supported.get("source_relations", ())
            if isinstance(value, str) and str.strip(_exact_string(value))
        }
        matched_relation = False
        for fact in context.relation_facts:
            relation = fact.get("relation") or fact.get("predicate")
            if not isinstance(relation, str):
                continue
            normalized_relation = str.casefold(str.strip(_exact_string(relation)))
            if source_relations and normalized_relation not in source_relations:
                continue
            if not _fact_is_supported(fact):
                continue
            subject = _clean_string(fact.get("subject")) or context.entity_label
            predicate = _clean_string(fact.get("predicate")) or str.strip(
                _exact_string(relation)
            )
            object_value = _clean_string(fact.get("object")) or _fact_text(fact)
            if not object_value:
                continue
            _append_triple(
                triples,
                seen,
                subject,
                predicate,
                object_value,
                source_relation=str.strip(_exact_string(relation)),
            )
            matched_relation = True
        if not matched_relation:
            fact_text = _fact_text(supported)
            for relation in source_relations or {"supported_fact"}:
                if fact_text:
                    _append_triple(
                        triples,
                        seen,
                        context.entity_label,
                        relation,
                        fact_text,
                        source_relation=relation,
                    )
    if not triples:
        for constraint in _constraint_texts(context, modality, "must_include"):
            _append_triple(
                triples,
                seen,
                context.entity_label,
                "has_required_content",
                constraint,
                source_relation="must_include",
            )
    return triples


def build_visual_target_graph(context: RankingContext) -> dict[str, Any]:
    """Build a visual projection of the entity's supported KG subgraph.

    The complete KG is not a valid image target: facts such as
    ``country=Philippines`` cannot be verified from ordinary pixels. This
    builder keeps only facts with an explicit visual clue or a conservative
    visual relation, and records excluded facts for audit.
    """
    nodes: list[dict[str, Any]] = []
    attributes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    included_facts: list[dict[str, Any]] = []
    excluded_facts: list[dict[str, Any]] = []
    profile = context.entity_profile
    entity_type = (
        profile.get("entity_type_hint")
        or profile.get("semantic_domain")
        or profile.get("semantic_domains")
        or profile.get("entity_type")
        or profile.get("type")
        or profile.get("primary_type")
    )
    if isinstance(entity_type, (list, tuple)):
        entity_type = entity_type[0] if entity_type else ""
    if isinstance(entity_type, str) and not _is_non_positive_label(entity_type):
        visual_type = _visual_constraint_text(entity_type)
        if visual_type:
            nodes.append(
                {
                    "id": "entity",
                    "label": visual_type,
                    "role": "scene",
                    "weight": 1.0,
                    "source": "entity_profile",
                }
            )

    for field_name, weight in (("safe_visual_style", 0.7),):
        value = profile.get(field_name)
        if isinstance(value, str):
            for part in _split_visual_values(value):
                if part and not _is_non_positive_label(part):
                    attributes.append(
                        {
                            "text": part,
                            "weight": weight,
                            "source": f"entity_profile.{field_name}",
                        }
                    )

    for bucket in ("should_include",):
        for value in _constraint_texts(context, "image", bucket):
            visual_text = _visual_constraint_text(value)
            if visual_text:
                attributes.append(
                    {
                        "text": visual_text,
                        "weight": 0.7,
                        "source": f"image_constraints.{bucket}",
                    }
                )

    for fact in context.relation_facts:
        if not _fact_is_supported(fact):
            continue
        relation = _clean_string(fact.get("relation") or fact.get("predicate")) or ""
        clue = (
            _clean_string(fact.get("visual_clue"))
            or _clean_string(fact.get("clue"))
        )
        if clue:
            visual_text = _visual_constraint_text(clue)
            if visual_text:
                attributes.append(
                    {
                        "text": visual_text,
                        "weight": _fact_visual_weight(fact),
                        "source": "relation_fact.visual_clue",
                        "source_relation": relation,
                    }
                )
                included_facts.append(_visual_fact_record(fact, "visual_clue"))
                continue

        if _is_visual_relation(relation):
            object_values = _fact_object_values(fact)
            for object_value in object_values:
                object_id = _stable_visual_id(object_value)
                nodes.append(
                    {
                        "id": object_id,
                        "label": object_value,
                        "role": "object",
                        "weight": _fact_visual_weight(fact),
                        "source": "relation_fact",
                        "source_relation": relation,
                    }
                )
                edges.append(
                    {
                        "subject": "entity",
                        "predicate": relation,
                        "object": object_id,
                        "confidence": _fact_confidence(fact),
                        "weight": _fact_visual_weight(fact),
                        "source_relation": relation,
                    }
                )
            if object_values:
                included_facts.append(_visual_fact_record(fact, "visual_relation"))
                continue

        excluded_facts.append(
            _visual_fact_record(fact, "not_directly_visualizable")
        )

    return {
        "nodes": _unique_graph_items(nodes, ("label",)),
        "attributes": _unique_graph_items(attributes, ("text",)),
        "edges": _unique_graph_items(edges, ("subject", "predicate", "object")),
        "ocr": [],
        "source_subgraph": {
            "included_facts": included_facts,
            "excluded_facts": excluded_facts,
        },
    }


_VISUAL_RELATION_HINTS = frozenset(
    {
        "architectural style",
        "architecture",
        "building material",
        "color",
        "contains",
        "coastline",
        "environment",
        "has landmark",
        "has part",
        "has terrain",
        "landform",
        "landscape",
        "notable feature",
        "terrain",
        "vegetation",
        "water feature",
    }
)


def _is_visual_relation(value: str) -> bool:
    normalized = " ".join(str(value or "").casefold().replace("_", " ").split())
    return normalized in _VISUAL_RELATION_HINTS


def _fact_object_values(fact: Mapping[str, Any]) -> list[str]:
    values = fact.get("neighbor_entities")
    if isinstance(values, (list, tuple)):
        result = [
            text
            for item in values
            if (text := _clean_string(item)) is not None
            and not _is_non_positive_label(text)
        ]
        if result:
            return result
    for key in ("object", "tail"):
        text = _clean_string(fact.get(key))
        if text and not _is_non_positive_label(text):
            return [text]
    return []


def _fact_confidence(fact: Mapping[str, Any]) -> float:
    value = fact.get("credibility_score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    label = str(fact.get("credibility_label") or "").casefold()
    return {"high": 0.9, "medium": 0.7, "low": 0.45}.get(label, 0.6)


def _fact_visual_weight(fact: Mapping[str, Any]) -> float:
    return round(_fact_confidence(fact), 6)


def _visual_fact_record(fact: Mapping[str, Any], projection: str) -> dict[str, Any]:
    return {
        "fact_id": fact.get("fact_id"),
        "subject": fact.get("subject"),
        "relation": fact.get("relation") or fact.get("predicate"),
        "object": fact.get("object") or fact.get("neighbor_entities"),
        "projection": projection,
        "confidence": _fact_confidence(fact),
    }


def _stable_visual_id(value: str) -> str:
    normalized = "_".join(str(value).casefold().split())
    return f"kg_{normalized or 'object'}"


def _split_visual_values(value: str) -> list[str]:
    return [
        item.strip(" ,;:")
        for item in str(value).replace(";", ",").split(",")
        if item.strip(" ,;:")
    ]

def _visual_constraint_text(value: Any) -> str:
    """Remove pipeline labels and retain the observable part of a constraint."""
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    prefixes = (
        "entity type:",
        "text visual cue:",
        "visual cue:",
        "text context cue:",
        "context cue:",
    )
    lowered = text.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text.strip(" ,;:")


def _append_triple(
    output: list[dict[str, Any]],
    seen: set[tuple[str, str, str, str]],
    subject: str,
    predicate: str,
    object_value: str,
    *,
    source_relation: str,
) -> None:
    key = (
        str.casefold(subject),
        str.casefold(predicate),
        str.casefold(object_value),
        "positive",
    )
    if key in seen:
        return
    seen.add(key)
    output.append(
        {
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "polarity": "positive",
            "confidence": 1.0,
            "source_relation": source_relation,
        }
    )


def _unique_graph_items(
    values: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        key = tuple(str.casefold(str(value.get(field, "")).strip()) for field in fields)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _primary_relation_visual_clue(
    context: RankingContext,
    candidate: Mapping[str, Any],
) -> str:
    primary_relation = _clean_string(candidate.get("primary_relation"))
    if primary_relation is None:
        return ""
    normalized_relation = str.casefold(primary_relation)
    stage_two_match = any(
        _fact_role(item) == "primary"
        and _fact_is_supported(item)
        and any(
            isinstance(source, str)
            and str.casefold(str.strip(_exact_string(source))) == normalized_relation
            for source in item.get("source_relations", ())
        )
        for item in context.supported_facts
    )
    if not stage_two_match:
        return ""
    for item in context.relation_facts:
        relation = item.get("relation") or item.get("predicate")
        if not isinstance(relation, str) or str.casefold(
            str.strip(_exact_string(relation))
        ) != normalized_relation:
            continue
        if _fact_role(item) != "primary" or not _fact_is_supported(item):
            continue
        clue = _fact_text(item)
        if clue and _is_safe_positive_text(clue, context, item):
            return clue
    return ""


def _safe_visual_profile_values(context: RankingContext) -> list[str]:
    confidence = context.entity_profile.get("confidence")
    if "confidence" in context.entity_profile and not _positive_labeled_value(
        confidence,
        _POSITIVE_CONFIDENCE_LABELS,
        numeric_minimum=0.5,
    ):
        return []
    values = []
    for key in (
        "semantic_domain",
        "domain",
        "visual_domain",
        "visual_style",
        "style",
        "safe_visual_style",
        "safe_style",
    ):
        value = context.entity_profile.get(key)
        candidates = value if isinstance(value, tuple) else (value,)
        for candidate in candidates:
            if (
                isinstance(candidate, str)
                and not _is_non_positive_label(candidate)
                and _is_safe_positive_text(candidate, context)
            ):
                values.append(candidate)
    return values


def _compact_clip_constraint_text(parts: list[str]) -> str:
    """Keep CLIP's text input comfortably below its 77-token context limit."""
    compact_parts = []
    for part in parts:
        words = str.split(part)
        if not words:
            continue
        compact_parts.append(" ".join(words[:7]))
    compact = " ".join(compact_parts)
    if len(compact) <= 240:
        return compact
    return compact[:240].rsplit(" ", 1)[0]


def build_visual_constraint_text(
    context: RankingContext,
    candidate: Mapping[str, Any],
    *,
    clip_safe: bool = False,
) -> str:
    """Build positive visual constraints without reading generation instructions."""
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be an image candidate mapping")
    target_modality = candidate.get("target_modality")
    if not isinstance(target_modality, str) or _exact_string(target_modality) != "image":
        raise ValueError("candidate must target image modality")
    if context.missing_mode == "text":
        raise ValueError("image visual constraints are excluded by context.missing_mode")
    parts: list[str] = []
    entity_type = context.entity_profile.get("entity_type") or context.entity_profile.get("type")
    profile_confidence = context.entity_profile.get("confidence")
    profile_is_positive = "confidence" not in context.entity_profile or (
        profile_confidence is not None
        and _positive_labeled_value(
            profile_confidence,
            _POSITIVE_CONFIDENCE_LABELS,
            numeric_minimum=0.5,
        )
    )
    if (
        profile_is_positive
        and isinstance(entity_type, str)
        and str.strip(_exact_string(entity_type))
        and not _is_non_positive_label(entity_type)
    ):
        type_text = f"entity type: {str.strip(_exact_string(entity_type))}"
        if _is_safe_positive_text(type_text, context):
            _append_stable(parts, [type_text])
    _append_stable(parts, _constraint_texts(context, "image", "must_include"))
    clue = _primary_relation_visual_clue(context, candidate)
    if clue:
        _append_stable(parts, [clue])
    _append_stable(parts, _constraint_texts(context, "image", "should_include"))
    _append_stable(parts, _safe_visual_profile_values(context))
    return _compact_clip_constraint_text(parts) if clip_safe else " ".join(parts)
