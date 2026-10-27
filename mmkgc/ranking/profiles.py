"""Strict Stage 4 model-fusion profile loading and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Literal


Stage4Mode = Literal["text", "image"]

_TOP_LEVEL_KEYS = frozenset(
    {
        "profile_name",
        "sentence_bert",
        "bertscore",
        "clip",
        "blip_itm",
        "gpt4omini_judge",
    }
)
_LOCAL_SCORER_KEYS = frozenset({"required", "enabled", "model_dir", "device"})
_GPT4OMINI_JUDGE_KEYS = frozenset(
    {
        "required",
        "model_config",
        "timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
        "max_concurrency",
    }
)
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token)", re.IGNORECASE)


@dataclass(frozen=True)
class LocalScorerConfig:
    required: bool
    enabled: bool
    model_dir: Path | None
    device: str | None


@dataclass(frozen=True)
class GPT4OMiniJudgeConfig:
    required: bool
    model_config: Path
    timeout_seconds: float
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("gpt4omini_judge.max_concurrency must be a positive integer")


@dataclass(frozen=True)
class Stage4Profile:
    profile_name: str
    sentence_bert: LocalScorerConfig
    bertscore: LocalScorerConfig
    clip: LocalScorerConfig
    blip_itm: LocalScorerConfig
    gpt4omini_judge: GPT4OMiniJudgeConfig


def load_stage4_profile(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> Stage4Profile:
    """Load a Stage 4 YAML profile with exact-key and path-containment checks."""
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is present in tests.
        raise RuntimeError("PyYAML is required to load Stage 4 profiles") from exc

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Stage 4 profile YAML must be a mapping")

    _reject_secret_keys(raw)
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "")

    root = Path(project_root).resolve()
    profile_name = _required_string(raw, "profile_name")

    sentence_bert = _load_local_scorer(raw, "sentence_bert", root)
    bertscore = _load_optional_local_scorer(raw, "bertscore", root)
    clip = _load_local_scorer(raw, "clip", root)
    blip_itm = _load_local_scorer(raw, "blip_itm", root)
    gpt4omini_judge = _load_gpt4omini_judge(raw, root)

    return Stage4Profile(
        profile_name=profile_name,
        sentence_bert=sentence_bert,
        bertscore=bertscore,
        clip=clip,
        blip_itm=blip_itm,
        gpt4omini_judge=gpt4omini_judge,
    )


def validate_stage4_mode(mode: str) -> Stage4Mode:
    if mode not in {"text", "image"}:
        raise ValueError("mode must be one of: text, image")
    return mode  # type: ignore[return-value]


def required_scorers_for_mode(
    profile: Stage4Profile,
    mode: str,
    *,
    has_existing_image: bool = False,
    has_existing_text: bool = False,
) -> set[str]:
    """Return scorer names that must be available for the selected ranking mode."""
    validated = validate_stage4_mode(mode)
    required: set[str] = set()

    if validated == "text" and _is_required(profile.sentence_bert):
        required.add("sentence_bert")
    if validated == "text" and _is_required(profile.bertscore):
        required.add("bertscore")

    needs_visual = validated == "image" or (
        validated == "text" and has_existing_image
    )
    if validated == "image" and has_existing_text:
        needs_visual = True

    if needs_visual:
        if _is_required(profile.clip):
            required.add("clip")
        if _is_required(profile.blip_itm):
            required.add("blip_itm")

    if profile.gpt4omini_judge.required:
        required.add("gpt4omini_judge")

    return required


def preflight_stage4_profile(
    profile: Stage4Profile,
    *,
    project_root: str | Path,
) -> list[str]:
    """Return missing required local resources without initializing model stacks."""
    root = Path(project_root).resolve()
    failures: list[str] = []
    for field_name in ("sentence_bert", "bertscore", "clip", "blip_itm"):
        scorer = getattr(profile, field_name)
        if _is_required(scorer):
            if scorer.model_dir is None:
                failures.append(f"{field_name}.model_dir is required")
                continue
            path = _resolve_contained(scorer.model_dir, root, f"{field_name}.model_dir")
            if not path.is_dir():
                failures.append(f"{field_name}.model_dir does not exist: {path}")

    if profile.gpt4omini_judge.required:
        path = _resolve_contained(
            profile.gpt4omini_judge.model_config,
            root,
            "gpt4omini_judge.model_config",
        )
        if not path.is_file():
            failures.append(f"gpt4omini_judge.model_config does not exist: {path}")

    return failures


def _load_local_scorer(
    raw: dict[str, Any],
    field_name: str,
    root: Path,
) -> LocalScorerConfig:
    section = _required_mapping(raw, field_name)
    _reject_unknown_keys(section, _LOCAL_SCORER_KEYS, field_name)
    required = _required_bool(section, "required", prefix=field_name)
    enabled = _required_bool(section, "enabled", prefix=field_name)
    model_dir = _optional_contained_path(
        section.get("model_dir"),
        root,
        f"{field_name}.model_dir",
    )
    device = _optional_device(section.get("device"), f"{field_name}.device")
    return LocalScorerConfig(
        required=required,
        enabled=enabled,
        model_dir=model_dir,
        device=device,
    )


def _load_optional_local_scorer(
    raw: dict[str, Any],
    field_name: str,
    root: Path,
) -> LocalScorerConfig:
    section = raw.get(field_name)
    if section is None:
        return LocalScorerConfig(
            required=False,
            enabled=False,
            model_dir=None,
            device=None,
        )
    if not isinstance(section, dict):
        raise ValueError(f"{field_name} must be a mapping")
    _reject_unknown_keys(section, _LOCAL_SCORER_KEYS, field_name)
    required = bool(section.get("required", False))
    if not isinstance(section.get("required", False), bool):
        raise ValueError(f"{field_name}.required must be a boolean")
    enabled = bool(section.get("enabled", False))
    if not isinstance(section.get("enabled", False), bool):
        raise ValueError(f"{field_name}.enabled must be a boolean")
    model_dir = _optional_contained_path(
        section.get("model_dir"),
        root,
        f"{field_name}.model_dir",
    )
    device = _optional_device(section.get("device"), f"{field_name}.device")
    return LocalScorerConfig(
        required=required,
        enabled=enabled,
        model_dir=model_dir,
        device=device,
    )


def _load_gpt4omini_judge(raw: dict[str, Any], root: Path) -> GPT4OMiniJudgeConfig:
    section = _required_mapping(raw, "gpt4omini_judge")
    _reject_unknown_keys(section, _GPT4OMINI_JUDGE_KEYS, "gpt4omini_judge")
    timeout = _required_float(section, "timeout_seconds", prefix="gpt4omini_judge")
    if timeout <= 0:
        raise ValueError("gpt4omini_judge.timeout_seconds must be positive")
    max_retries = _optional_non_negative_int(
        section.get("max_retries", 0),
        "gpt4omini_judge.max_retries",
    )
    retry_backoff_seconds = _optional_non_negative_float(
        section.get("retry_backoff_seconds", 0.0),
        "gpt4omini_judge.retry_backoff_seconds",
    )
    max_concurrency = _optional_non_negative_int(
        section.get("max_concurrency", 1), "gpt4omini_judge.max_concurrency",
    )
    if max_concurrency < 1:
        raise ValueError("gpt4omini_judge.max_concurrency must be a positive integer")
    return GPT4OMiniJudgeConfig(
        required=_required_bool(section, "required", prefix="gpt4omini_judge"),
        model_config=_required_contained_path(
            section,
            "model_config",
            root,
            "gpt4omini_judge",
        ),
        timeout_seconds=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        max_concurrency=max_concurrency,
    )


def _is_required(config: LocalScorerConfig) -> bool:
    return config.required and config.enabled


def _reject_secret_keys(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"YAML key at {path or '$'} must be a string")
            full_path = f"{path}.{key}" if path else key
            if _SECRET_KEY_RE.search(key):
                raise ValueError(f"secret-like YAML key is not allowed: {full_path}")
            _reject_secret_keys(item, full_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_keys(item, f"{path}[{index}]")


def _reject_unknown_keys(
    section: dict[str, Any],
    allowed: frozenset[str],
    prefix: str,
) -> None:
    for key in section:
        if key not in allowed:
            if prefix:
                raise ValueError(f"unknown nested key: {prefix}.{key}")
            raise ValueError(f"unknown top-level key: {key}")


def _required_mapping(
    raw: dict[str, Any],
    key: str,
    *,
    prefix: str | None = None,
) -> dict[str, Any]:
    path = f"{prefix}.{key}" if prefix else key
    if key not in raw:
        raise ValueError(f"missing required key: {path}")
    value = raw[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required_string(
    raw: dict[str, Any],
    key: str,
    *,
    prefix: str | None = None,
) -> str:
    path = f"{prefix}.{key}" if prefix else key
    if key not in raw:
        raise ValueError(f"missing required key: {path}")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _required_bool(raw: dict[str, Any], key: str, *, prefix: str) -> bool:
    path = f"{prefix}.{key}"
    if key not in raw:
        raise ValueError(f"missing required key: {path}")
    value = raw[key]
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _required_float(raw: dict[str, Any], key: str, *, prefix: str) -> float:
    path = f"{prefix}.{key}"
    if key not in raw:
        raise ValueError(f"missing required key: {path}")
    return _finite_float(raw[key], path)


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{path} must be a finite number")
    return normalized


def _optional_non_negative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _optional_non_negative_float(value: Any, path: str) -> float:
    normalized = _finite_float(value, path)
    if normalized < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return normalized


def _required_contained_path(
    raw: dict[str, Any],
    key: str,
    root: Path,
    prefix: str,
) -> Path:
    path = f"{prefix}.{key}"
    if key not in raw:
        raise ValueError(f"missing required key: {path}")
    value = raw[key]
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{path} must be a non-empty path")
    return _resolve_contained(Path(value), root, path)


def _optional_contained_path(value: Any, root: Path, path: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{path} must be a non-empty path")
    return _resolve_contained(Path(value), root, path)


def _resolve_contained(path: Path, root: Path, field_name: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must stay inside project root") from exc
    return resolved


def _optional_device(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a device string")
    if value == "cpu" or value == "cuda" or re.fullmatch(r"cuda:\d+", value):
        return value
    raise ValueError(f"{path} must be cpu, cuda, or cuda:<index>")
