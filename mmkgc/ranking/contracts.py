"""Shared contracts for model-fusion ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Literal


RankingStatus = Literal[
    "completed",
    "degraded",
    "no_valid_candidate",
    "failed",
]


class RankingError(RuntimeError):
    """Base error raised by model-fusion ranking."""


class RequiredScorerError(RankingError):
    """Raised when a required scorer cannot produce a result."""


class FrozenDict(dict):
    """A JSON/asdict-compatible mapping that rejects mutation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_initialized", False):
            raise TypeError("FrozenDict is immutable")
        source = dict(*args, **kwargs)
        rebuilt = _freeze_json_mapping(source, "$")
        dict.__init__(self, rebuilt)
        self._initialized = True

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def freeze_value(value: Any) -> Any:
    """Recursively validate, detach, and freeze a strict JSON value.

    This function is intentionally limited to JSON scalar/container values.
    Opaque transport values (for example image paths or bytes) are copied by
    ``ranking.context`` using its separate transport boundary and must never be
    passed through this validator.
    """
    return _freeze_json_value(value, "$")


def _path_for_key(path: str, key: str) -> str:
    return f"{path}[{key!r}]"


def _new_frozen_mapping(items: list[tuple[str, Any]]) -> FrozenDict:
    result = dict.__new__(FrozenDict)
    dict.__init__(result, items)
    result._initialized = True
    return result


def _freeze_json_mapping(value: Mapping[Any, Any], path: str) -> FrozenDict:
    items: list[tuple[str, Any]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"JSON object key at {path} must be a string; got {type(key).__name__}"
            )
        normalized_key = key if type(key) is str else str.__str__(key)
        items.append((normalized_key, _freeze_json_value(item, _path_for_key(path, normalized_key))))
    return _new_frozen_mapping(items)


def _freeze_json_value(value: Any, path: str) -> Any:
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return value if type(value) is str else str.__str__(value)
    if isinstance(value, int):
        return value if type(value) is int else int.__int__(value)
    if isinstance(value, float):
        normalized = value if type(value) is float else float.__float__(value)
        if not math.isfinite(normalized):
            raise ValueError(f"non-finite numeric value at {path}")
        return normalized
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, path)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"set is not a JSON container at {path}")
    raise TypeError(
        f"unsupported JSON value at {path}: {type(value).__name__}"
    )


def _finite_number(
    field_name: str,
    value: Any,
    *,
    allow_none: bool = False,
) -> float | None:
    """Validate and normalize a finite contract number without booleans."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite numeric value")
    normalized = (
        float(value)
        if type(value) is float
        else float(int.__int__(value))
        if isinstance(value, int)
        else float.__float__(value)
    )
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite numeric value")
    return normalized


def _string(field_name: str, value: Any, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value if type(value) is str else str.__str__(value)
    if nonempty and not normalized.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _string_tuple(field_name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must contain only strings")
    return tuple(item if type(item) is str else str.__str__(item) for item in value)


@dataclass(frozen=True)
class ScoreResult:
    raw_score: float | None
    normalized_score: float | None
    status: Literal["scored", "failed", "skipped"]
    model_name: str
    model_version: str = ""
    latency_ms: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_score = _finite_number("raw_score", self.raw_score, allow_none=True)
        normalized_score = _finite_number(
            "normalized_score", self.normalized_score, allow_none=True
        )
        status = _string("status", self.status)
        model_name = _string("model_name", self.model_name)
        model_version = _string("model_version", self.model_version)
        if normalized_score is not None and not 0.0 <= normalized_score <= 1.0:
            raise ValueError("normalized_score must be between 0 and 1")
        if status not in {
            "scored",
            "failed",
            "skipped",
        }:
            raise ValueError("status must be one of: scored, failed, skipped")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, int):
            raise TypeError("latency_ms must be a non-negative integer")
        latency_ms = (
            self.latency_ms
            if type(self.latency_ms) is int
            else int.__int__(self.latency_ms)
        )
        if latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative integer")
        if not isinstance(self.details, Mapping):
            raise TypeError(
                f"ScoreResult.details must be a JSON object at $; got {type(self.details).__name__}"
            )
        object.__setattr__(self, "raw_score", raw_score)
        object.__setattr__(self, "normalized_score", normalized_score)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "details", freeze_value(self.details))


@dataclass(frozen=True)
class GateResult:
    decision: Literal["pass", "reject"]
    source: str
    hard_violations: tuple[str, ...] = ()
    soft_violations: tuple[str, ...] = ()
    judge_confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        decision = _string("decision", self.decision)
        source = _string("source", self.source)
        judge_confidence = _finite_number("judge_confidence", self.judge_confidence)
        reason = _string("reason", self.reason)
        if decision not in {"pass", "reject"}:
            raise ValueError("decision must be one of: pass, reject")
        if judge_confidence is None or not 0.0 <= judge_confidence <= 1.0:
            raise ValueError("judge_confidence must be between 0 and 1")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "judge_confidence", judge_confidence)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self, "hard_violations", _string_tuple("hard_violations", self.hard_violations)
        )
        object.__setattr__(
            self, "soft_violations", _string_tuple("soft_violations", self.soft_violations)
        )
        object.__setattr__(self, "evidence", _string_tuple("evidence", self.evidence))
