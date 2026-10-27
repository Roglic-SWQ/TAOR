"""Qwen-VL image judge provider for image quality evaluation."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

from mmkgc.evaluation.image_quality_targets import (
    ImageQualityTarget,
    collect_entity_image_quality_targets,
    target_entity_key,
    target_entity_label,
)
from mmkgc.evaluation.loader import RunEntityArtifacts

try:
    import yaml
except Exception:  # pragma: no cover - explicit fallback if PyYAML is unavailable.
    yaml = None


QWEN_VL_JUDGE_PROVIDER = "qwen-vl-judge"
QWEN_VL_JUDGE_SCORE_FIELDS = (
    "entity_match_score",
    "prompt_alignment_score",
    "kg_evidence_score",
    "artifact_score",
    "overall_judge_score",
)
REDACTED_VALUE = "[REDACTED]"
SENSITIVE_KEY_PARTS = ("api_key", "authorization", "token", "secret")


class QwenVLJudgeError(RuntimeError):
    """Raised when Qwen-VL judge configuration, HTTP, or parsing fails."""


@dataclass(frozen=True)
class QwenVLJudgeConfig:
    model_name: str
    base_url: str
    api_key: str = field(repr=False)
    max_tokens: int = 512
    temperature: float = 0.0
    timeout_seconds: int = 120


class QwenVLJudgeProtocol(Protocol):
    def judge_image(self, image_path: Path, prompt: str) -> dict[str, Any]:
        """Return normalized judge scores for one image."""


@dataclass
class FixedQwenVLJudge:
    """Test judge that returns configured results and records calls."""

    results: list[Any]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def judge_image(self, image_path: Path, prompt: str) -> dict[str, Any]:
        call_index = len(self.calls)
        self.calls.append({"image_path": image_path, "prompt": prompt})
        if call_index >= len(self.results):
            raise QwenVLJudgeError(
                f"fake Qwen-VL judge has no result for call {call_index + 1}"
            )
        raw_result = self.results[call_index]
        normalized = _normalize_judge_payload(raw_result)
        if isinstance(raw_result, dict):
            normalized["latency_ms"] = _coerce_latency_ms(
                raw_result.get("latency_ms", 0)
            )
            normalized["raw_response"] = _coerce_raw_response(
                raw_result.get("raw_response", {})
            )
        else:
            normalized["latency_ms"] = 0
            normalized["raw_response"] = {}
        return normalized


class QwenVLJudgeClient:
    """HTTP client for an OpenAI-compatible Qwen-VL judge endpoint."""

    def __init__(self, config: QwenVLJudgeConfig) -> None:
        self.config = config

    def judge_image(self, image_path: Path, prompt: str) -> dict[str, Any]:
        start_time = time.perf_counter()
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image_path)},
                        },
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(
            _chat_completions_url(self.config.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
                latency_ms = int((time.perf_counter() - start_time) * 1000)
        except error.HTTPError as exc:
            body = _redact_sensitive(
                _read_http_error_body(exc),
                secrets=(self.config.api_key,),
            )
            reason = _redact_sensitive(
                str(exc.reason),
                secrets=(self.config.api_key,),
            )
            detail = f"{exc.code} {reason}"
            if body:
                detail = f"{detail}: {body}"
            raise QwenVLJudgeError(f"Qwen-VL judge HTTP error: {detail}") from exc
        except error.URLError as exc:
            safe_error = _redact_sensitive(
                str(exc),
                secrets=(self.config.api_key,),
            )
            raise QwenVLJudgeError(f"Qwen-VL judge URL error: {safe_error}") from exc
        except Exception as exc:
            safe_error = _redact_sensitive(
                str(exc),
                secrets=(self.config.api_key,),
            )
            raise QwenVLJudgeError(
                f"Qwen-VL judge request failed: {safe_error}"
            ) from exc

        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise QwenVLJudgeError(
                f"Qwen-VL judge response was invalid JSON: {exc.msg}"
            ) from exc

        content = _extract_openai_message_content(response_json)
        parsed = parse_qwen_vl_judge_response(content)
        parsed["latency_ms"] = latency_ms
        parsed["raw_response"] = _redact_sensitive(
            response_json,
            secrets=(self.config.api_key,),
        )
        return parsed


def load_qwen_vl_judge_config(
    path: str | Path,
    timeout_seconds: int = 120,
) -> QwenVLJudgeConfig:
    """Load Qwen-VL judge settings from a Qwen-compatible YAML config."""

    if yaml is None:
        raise QwenVLJudgeError(
            "PyYAML is required to load Qwen-VL judge config. Please install PyYAML."
        )

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        raise QwenVLJudgeError(f"failed to read Qwen-VL judge config: {exc}") from exc

    return _coerce_qwen_vl_judge_config(data or {}, config_path, timeout_seconds)


def image_to_data_url(image_path: str | Path) -> str:
    """Encode a local PNG/JPEG image as a base64 data URL."""

    path = Path(image_path)
    if not path.is_file():
        raise QwenVLJudgeError(f"image file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    else:
        raise QwenVLJudgeError(
            f"unsupported image extension {path.suffix!r}; supported: png, jpg, jpeg"
        )

    return (
        f"data:{mime_type};base64,"
        f"{base64.b64encode(path.read_bytes()).decode('ascii')}"
    )


def build_qwen_vl_judge_prompt(target: ImageQualityTarget) -> str:
    """Build the strict JSON-only Chinese judge prompt for one target."""

    return (
        "你是 MKG-W 图像质量评审器。请根据输入图像、实体信息、图像生成 prompt、"
        "配对的 generated text 和 KG evidence 判断该图片是否适合作为实体的图像补全结果。\n"
        "请只输出一个 JSON object，不要使用 Markdown 代码块，不要输出解释性前后缀。\n"
        "\n"
        "上下文:\n"
        f"entity_key: {target.entity_key}\n"
        f"entity_label: {target.label}\n"
        f"image_candidate_id: {target.image_candidate_id}\n"
        f"prompt_text: {target.prompt_text}\n"
        f"candidate_text: {target.candidate_text}\n"
        f"evidence_summary_text: {target.evidence_summary_text}\n"
        "\n"
        "字段说明:\n"
        "- entity_match_score: 0-100，图像是否匹配实体身份与标签。\n"
        "- prompt_alignment_score: 0-100，图像是否遵循 prompt_text 的视觉要求。\n"
        "- kg_evidence_score: 0-100，图像是否符合 KG evidence 与 generated text 中的关键事实。\n"
        "- artifact_score: 0-100，图像干净程度，100 表示没有明显伪影、畸变、水印或文字错误。\n"
        "- overall_judge_score: 0-100，综合质量判断。\n"
        "- rationale: 简短中文理由，必须是非空字符串。\n"
        "\n"
        "输出 JSON object 必须包含且只需要包含这些字段: "
        "entity_match_score, prompt_alignment_score, kg_evidence_score, "
        "artifact_score, overall_judge_score, rationale。所有分数必须是 0 到 100 的有限数字。"
    )


def parse_qwen_vl_judge_response(response_text: str) -> dict[str, Any]:
    """Parse and validate the Qwen-VL judge JSON response."""

    stripped = _strip_json_code_fence(str(response_text))
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise QwenVLJudgeError(f"invalid JSON from Qwen-VL judge: {exc.msg}") from exc
    return _normalize_judge_payload(payload)


def evaluate_entity_qwen_vl_judge(
    artifacts: RunEntityArtifacts,
    *,
    judge: QwenVLJudgeProtocol,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    """Judge all generated image candidates for one entity."""

    targets = collect_entity_image_quality_targets(
        artifacts,
        error_cls=QwenVLJudgeError,
    )
    normalized_base_url = base_url.rstrip("/")
    image_scores: list[dict[str, Any]] = []

    for target in targets:
        prompt = build_qwen_vl_judge_prompt(target)
        try:
            raw_judge_result = judge.judge_image(target.image_path, prompt)
            judge_result = _normalize_judge_payload(raw_judge_result)
        except QwenVLJudgeError as exc:
            image_scores.append(
                _failed_image_score(
                    target,
                    model_name=model_name,
                    base_url=normalized_base_url,
                    error=exc,
                )
            )
            continue
        image_score = {
            "entity_key": target.entity_key,
            "image_candidate_id": target.image_candidate_id,
            "image_path": target.relative_image_path,
            "provider": QWEN_VL_JUDGE_PROVIDER,
            "model_name": model_name,
            "base_url": normalized_base_url,
            "status": "judged",
            "fallback_used": False,
            "parse_failed": False,
            "latency_ms": _result_latency_ms(raw_judge_result),
            "raw_response": _result_raw_response(raw_judge_result),
            "rationale": judge_result["rationale"],
            "text_targets": {
                "prompt_text": target.prompt_text,
                "candidate_text": target.candidate_text,
                "evidence_summary_text": target.evidence_summary_text,
            },
        }
        for field_name in QWEN_VL_JUDGE_SCORE_FIELDS:
            image_score[field_name] = judge_result[field_name]
        image_scores.append(image_score)

    averages = {
        field_name: _average_score([score[field_name] for score in image_scores])
        for field_name in QWEN_VL_JUDGE_SCORE_FIELDS
    }

    return {
        "entity": {
            "entity_key": target_entity_key(artifacts),
            "label": target_entity_label(artifacts),
            "missing_mode": artifacts.final.get("missing_mode"),
        },
        "image_count": len(image_scores),
        "model_name": model_name,
        "base_url": normalized_base_url,
        "averages": averages,
        "best_image_candidate_id": _extreme_image_candidate_id(
            image_scores,
            reverse=True,
        ),
        "worst_image_candidate_id": _extreme_image_candidate_id(
            image_scores,
            reverse=False,
        ),
        "image_scores": image_scores,
    }


def _failed_image_score(
    target: ImageQualityTarget,
    *,
    model_name: str,
    base_url: str,
    error: QwenVLJudgeError,
) -> dict[str, Any]:
    image_score: dict[str, Any] = {
        "entity_key": target.entity_key,
        "image_candidate_id": target.image_candidate_id,
        "image_path": target.relative_image_path,
        "provider": QWEN_VL_JUDGE_PROVIDER,
        "model_name": model_name,
        "base_url": base_url,
        "status": "failed",
        "fallback_used": True,
        "parse_failed": False,
        "latency_ms": 0,
        "raw_response": {},
        "rationale": "",
        "error": str(error),
        "error_type": type(error).__name__,
        "text_targets": {
            "prompt_text": target.prompt_text,
            "candidate_text": target.candidate_text,
            "evidence_summary_text": target.evidence_summary_text,
        },
    }
    for field_name in QWEN_VL_JUDGE_SCORE_FIELDS:
        image_score[field_name] = 0.0
    return image_score


def _coerce_qwen_vl_judge_config(
    data: Any,
    source_path: Path,
    timeout_seconds: int,
) -> QwenVLJudgeConfig:
    if not isinstance(data, dict):
        raise QwenVLJudgeError(f"{source_path}: config content must be a mapping")

    lmm = data.get("lmm", {})
    inference = data.get("inference", {})
    if not isinstance(lmm, dict):
        raise QwenVLJudgeError(f"{source_path}: config field 'lmm' must be a mapping")
    if not isinstance(inference, dict):
        raise QwenVLJudgeError(
            f"{source_path}: config field 'inference' must be a mapping"
        )

    model_name = _required_config_text_or_env(lmm, "model_name", source_path, "lmm")
    base_url = _required_config_text_or_env(lmm, "base_url", source_path, "lmm")
    api_key = _config_api_key(lmm, source_path)

    try:
        max_tokens = int(inference.get("max_token", 512))
        temperature = float(inference.get("temperature", 0.0))
        timeout = int(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise QwenVLJudgeError(
            f"{source_path}: inference.max_token, inference.temperature, and "
            "timeout_seconds must be numeric"
        ) from exc

    return QwenVLJudgeConfig(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout,
    )


def _config_api_key(lmm: dict[str, Any], source_path: Path) -> str:
    api_key = str(lmm.get("api_key", "") or "").strip()
    if api_key:
        return api_key

    api_key_env = str(lmm.get("api_key_env", "") or "").strip()
    if not api_key_env:
        raise QwenVLJudgeError(
            f"{source_path}: missing 'lmm.api_key' or 'lmm.api_key_env'"
        )
    api_key_from_env = os.environ.get(api_key_env, "").strip()
    if not api_key_from_env:
        raise QwenVLJudgeError(
            f"{source_path}: environment variable {api_key_env} is required"
        )
    return api_key_from_env


def _required_config_text_or_env(
    data: dict[str, Any],
    key: str,
    source_path: Path,
    section: str,
) -> str:
    value = data.get(key)
    text = str(value).strip() if value is not None else ""
    if text:
        return text

    env_key = f"{key}_env"
    env_name = str(data.get(env_key, "") or "").strip()
    if env_name:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
        raise QwenVLJudgeError(
            f"{source_path}: environment variable {env_name} is required"
        )

    raise QwenVLJudgeError(
        f"{source_path}: missing '{section}.{key}' or '{section}.{env_key}'"
    )


def _required_config_text(
    data: dict[str, Any],
    key: str,
    source_path: Path,
    section: str,
) -> str:
    value = data.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise QwenVLJudgeError(f"{source_path}: missing '{section}.{key}'")
    return text


def _strip_json_code_fence(response_text: str) -> str:
    text = response_text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _normalize_judge_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise QwenVLJudgeError("Qwen-VL judge response must be a JSON object")

    normalized: dict[str, Any] = {}
    for field_name in QWEN_VL_JUDGE_SCORE_FIELDS:
        if field_name not in payload:
            raise QwenVLJudgeError(f"missing required field: {field_name}")
        normalized[field_name] = _coerce_score(field_name, payload[field_name])

    if "rationale" not in payload:
        raise QwenVLJudgeError("missing required field: rationale")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise QwenVLJudgeError("rationale must be a non-empty string")
    normalized["rationale"] = rationale.strip()
    return normalized


def _coerce_score(field_name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise QwenVLJudgeError(f"{field_name} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise QwenVLJudgeError(f"{field_name} must be numeric") from exc
    if not math.isfinite(score):
        raise QwenVLJudgeError(f"{field_name} must be finite")
    if score < 0.0 or score > 100.0:
        raise QwenVLJudgeError(f"{field_name} must be between 0 and 100")
    return score


def _result_latency_ms(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    return _coerce_latency_ms(result.get("latency_ms", 0))


def _coerce_latency_ms(value: Any) -> int:
    if isinstance(value, bool):
        raise QwenVLJudgeError("latency_ms must be a non-negative integer")
    try:
        latency_ms = int(value)
    except (TypeError, ValueError) as exc:
        raise QwenVLJudgeError("latency_ms must be a non-negative integer") from exc
    if latency_ms < 0:
        raise QwenVLJudgeError("latency_ms must be a non-negative integer")
    return latency_ms


def _result_raw_response(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    return _coerce_raw_response(result.get("raw_response", {}))


def _coerce_raw_response(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise QwenVLJudgeError("raw_response must be a JSON object")
    redacted = _redact_sensitive(value)
    if not isinstance(redacted, dict):
        raise QwenVLJudgeError("raw_response must be a JSON object")
    return redacted


def _redact_sensitive(
    value: Any,
    *,
    secrets: tuple[str, ...] | list[str] = (),
) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                redacted[key_text] = REDACTED_VALUE
            else:
                redacted[key_text] = _redact_sensitive(item, secrets=secrets)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return _redact_sensitive_string(value, secrets=secrets)
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _redact_sensitive_string(
    value: str,
    *,
    secrets: tuple[str, ...] | list[str],
) -> str:
    redacted = value
    for secret in secrets:
        if not secret:
            continue
        secret_text = str(secret)
        if not secret_text:
            continue
        bearer_pattern = re.compile(
            rf"Bearer\s+{re.escape(secret_text)}",
            flags=re.IGNORECASE,
        )
        redacted = bearer_pattern.sub(f"Bearer {REDACTED_VALUE}", redacted)
        redacted = redacted.replace(secret_text, REDACTED_VALUE)
    return redacted


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _extract_openai_message_content(response_json: Any) -> str:
    if not isinstance(response_json, dict):
        raise QwenVLJudgeError("Qwen-VL judge HTTP response must be a JSON object")

    try:
        choice = response_json["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QwenVLJudgeError(
            "Qwen-VL judge HTTP response missing choices[0].message.content"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise QwenVLJudgeError(
            "Qwen-VL judge HTTP response content must be a non-empty string"
        )
    return content


def _read_http_error_body(exc: error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8").strip()
    except Exception:
        return ""


def _average_score(values: list[Any]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _extreme_image_candidate_id(
    image_scores: list[dict[str, Any]],
    *,
    reverse: bool,
) -> str | None:
    if not image_scores:
        return None
    selected = sorted(
        image_scores,
        key=lambda score: score["overall_judge_score"],
        reverse=reverse,
    )[0]
    return selected["image_candidate_id"]
