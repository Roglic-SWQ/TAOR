"""Hard-gate helpers for Stage 4 model fusion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from pathlib import Path
from typing import Any

from mmkgc.ranking.contracts import GateResult
from mmkgc.ranking.image_input import ImageInput
from mmkgc.ranking.profiles import validate_stage4_mode


def deterministic_candidate_gate(
    candidate: dict[str, Any],
    modality: str,
    image_input: ImageInput | None = None,
) -> GateResult:
    validated_modality = validate_stage4_mode(modality)
    hard: list[str] = []
    soft: list[str] = []

    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        hard.append("candidate_id_missing")

    target_modality = candidate.get("target_modality")
    if isinstance(target_modality, str) and target_modality != validated_modality:
        hard.append("target_modality_mismatch")

    if candidate.get("status") in {"failed", "error"}:
        hard.append("generator_failed")

    generated = candidate.get("generated_content")
    if not isinstance(generated, Mapping):
        hard.append("schema_invalid")
    else:
        if generated.get("status") in {"failed", "error"}:
            hard.append("generator_failed")
        content_type = generated.get("type")
        if content_type != validated_modality:
            hard.append("target_modality_mismatch")
        if validated_modality == "text":
            value = generated.get("value")
            if not isinstance(value, str) or not value.strip():
                hard.append("empty_generated_content")
        elif validated_modality == "image":
            if not _has_image_source(generated, image_input):
                hard.append("missing_image_artifact")
        else:
            hard.append("schema_invalid")

    if not isinstance(candidate.get("used_evidence_triples", ()), (list, tuple)):
        soft.append("used_evidence_triples_invalid")

    hard_tuple = _unique(hard)
    decision = "reject" if hard_tuple else "pass"
    return GateResult(
        decision,
        "deterministic",
        hard_violations=hard_tuple,
        soft_violations=_unique(soft),
    )


def gate_from_judge(judge: Any, allowed_codes: tuple[str, ...]) -> GateResult:
    allowed = set(allowed_codes)
    hard_codes = _string_sequence(
        _get_alias(
            judge,
            (
                "hard_violation_codes",
                "hard_rejection_codes",
                "hard_violations",
            ),
            (),
        )
    )
    unknown = [code for code in hard_codes if code not in allowed]
    if unknown:
        raise ValueError(f"unknown hard violation code: {unknown[0]}")

    soft_codes = list(
        _string_sequence(
            _get_alias(
                judge,
                (
                    "soft_violation_codes",
                    "soft_violations",
                ),
                (),
            )
        )
    )
    confidence = _finite_float(
        _get_alias(judge, ("confidence", "judge_confidence"), 0.0),
        "confidence",
    )
    evidence = _string_sequence(_get(judge, "evidence", ()))
    reject_requested = bool(
        _get_alias(judge, ("reject", "hard_reject"), False)
        or _get(judge, "decision", "") == "reject"
    )

    hard_reject = (
        reject_requested
        and bool(hard_codes)
        and confidence >= 0.80
        and bool(evidence)
    )
    if hard_reject:
        decision = "reject"
        final_hard = hard_codes
    else:
        decision = "pass"
        final_hard = ()
        soft_codes.extend(hard_codes)

    source = _get(judge, "source", "gpt4omini_judge")
    if not isinstance(source, str) or not source.strip():
        source = "gpt4omini_judge"
    reason = _get(judge, "reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return GateResult(
        decision,
        source,
        hard_violations=_unique(final_hard),
        soft_violations=_unique(soft_codes),
        judge_confidence=confidence,
        evidence=evidence,
        reason=reason,
    )


def _has_image_source(
    generated: Mapping[str, Any],
    image_input: ImageInput | None,
) -> bool:
    if image_input is not None:
        return True
    for key in ("image_artifact_id", "artifact_id"):
        value = generated.get(key)
        if isinstance(value, str) and value.strip():
            return True
    image_path = generated.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return Path(image_path).is_file()
    return False


def _get(judge: Any, field_name: str, default: Any) -> Any:
    if isinstance(judge, Mapping):
        return judge.get(field_name, default)
    return getattr(judge, field_name, default)


def _get_alias(judge: Any, field_names: tuple[str, ...], default: Any) -> Any:
    for field_name in field_names:
        value = _get(judge, field_name, None)
        if value is not None:
            return value
    return default


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("judge code/evidence fields must be sequences")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("judge code/evidence fields must contain strings")
        result.append(item)
    return _unique(result)


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
