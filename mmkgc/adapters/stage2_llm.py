"""OpenAI-compatible semantic assessor for bounded Stage 2 score adjustments."""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib import error, request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mmkgc.adapters.qwen import QwenConfig


_PROVIDER = "qwen-local-stage2-semantic-assessor"
_SENSITIVE_KEY_PATTERN = (
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|token|password|secret)"
)
_FENCED_JSON_PATTERN = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<payload>.*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


class QwenLocalStage2LLMAssessor:
    """Request small, structured semantic corrections without replacing local scoring."""

    def __init__(self, config: QwenConfig, timeout_seconds: float = 60.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self._call_index = 0

    def assess(
        self,
        *,
        relation: str,
        fact: str,
        target_modalities: list[str],
        generation_plan: dict[str, Any],
        score_components: dict[str, float],
    ) -> dict[str, Any]:
        """Return an assessment plus an audit-only model-call record."""
        self._call_index += 1
        model_call_ref = f"stage2_llm:{self._call_index:04d}"
        prompt = self._build_prompt(
            relation=relation,
            fact=fact,
            target_modalities=target_modalities,
            generation_plan=generation_plan,
            score_components=score_components,
        )
        started = time.perf_counter()
        response_json: Any = None
        response_text = ""
        try:
            response_json = self._complete(prompt)
            if not isinstance(response_json, dict):
                raise ValueError("Stage 2 assessor response must be an object")
            response_text = self._response_text(response_json)
            assessment = self._parse_assessment(response_text)
            assessment = self._redact_value(assessment)
            status = "generated"
            error_text = None
        except Exception as exc:
            raise RuntimeError(
                "Stage 2 LLM semantic assessment failed: " + self._short_error(exc)
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        call = {
            "stage": "stage2_llm_assessment",
            "provider": _PROVIDER,
            "model_name": self.config.model_name,
            "base_url": self._redact_url(self.config.base_url),
            "temperature": 0.0,
            "max_tokens": max(self.config.max_token, 512),
            "prompt": self._redact_text(prompt),
            "response_text": self._redact_text(response_text),
            "latency_ms": latency_ms,
            "status": status,
            "error": self._redact_text(error_text) if error_text else None,
            "model_call_ref": model_call_ref,
        }
        if response_json is not None:
            call["response"] = self._redact_value(response_json)
        return {**assessment, "model_call_ref": model_call_ref, "model_call": call}

    def _complete(self, prompt: str) -> Any:
        request_body = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a constrained MMKGC Stage 2 semantic assessor. "
                        "Return exact JSON only. You may recommend only small score adjustments; "
                        "you must not add, remove, or alter evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max(self.config.max_token, 512),
            "temperature": 0.0,
        }
        req = request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload

    @staticmethod
    def _response_text(response: dict[str, Any]) -> str:
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return str(content) if content is not None else ""

    @staticmethod
    def _parse_assessment(response_text: str) -> dict[str, Any]:
        normalized_text = response_text.strip()
        fenced_match = _FENCED_JSON_PATTERN.fullmatch(normalized_text)
        if fenced_match is not None:
            normalized_text = fenced_match.group("payload")
        payload = json.loads(normalized_text)
        if not isinstance(payload, dict):
            raise ValueError("Stage 2 assessment must be a JSON object")
        return payload

    def _short_error(self, exc: Exception) -> str:
        if isinstance(exc, error.HTTPError):
            message = f"HTTPError: {exc.code} {exc.reason}"
        elif isinstance(exc, error.URLError) and getattr(exc, "reason", None):
            message = f"URLError: {exc.reason}"
        else:
            message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return self._redact_text(message)

    def _redact_text(self, value: str | None) -> str:
        text = str(value or "")
        if self.config.api_key:
            text = text.replace(self.config.api_key, "[REDACTED]")
        text = _redact_url_userinfo(text)
        return _redact_sensitive_assignments(text)

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): "[REDACTED]"
                if _is_sensitive_key(str(key))
                else self._redact_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _redact_url(self, value: str) -> str:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname or ""
            if not hostname:
                return self._redact_text(value)
            host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
            netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
            query = urlencode(
                [
                    (key, "[REDACTED]" if _is_sensitive_key(key) else item)
                    for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                ]
            )
            return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
        except (TypeError, ValueError):
            return self._redact_text(value)

    @staticmethod
    def _build_prompt(
        *,
        relation: str,
        fact: str,
        target_modalities: list[str],
        generation_plan: dict[str, Any],
        score_components: dict[str, float],
    ) -> str:
        payload = {
            "relation": relation,
            "fact": fact,
            "target_modalities": target_modalities,
            "generation_plan": generation_plan,
            "current_score_components": score_components,
        }
        return (
            "Assess whether this already-evidenced relation fact needs a small semantic score correction. "
            "Do not change evidence or infer new facts. Return exactly one JSON object with: "
            "status, recommended_role (optional), component_adjustments (object), "
            "risk_penalty_adjustment, confidence, and reason. "
            "Allowed status values: applied means apply the deltas; ok means no adjustment is needed; "
            "failed means the assessment is unavailable. "
            "component_adjustments may contain only modality_fit, constraint_alignment, generation_utility. "
            "Do not include evidence_strength or risk_penalty; express risk only with "
            "risk_penalty_adjustment. "
            "Every adjustment must be between -0.15 and 0.15.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )


def _is_sensitive_key(value: str) -> bool:
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized in {
        "api_key",
        "apikey",
        "authorization",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "secret",
    }


def _redact_url_userinfo(value: str) -> str:
    return re.sub(
        r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@",
        r"\1[REDACTED]@",
        value,
    )


def _redact_sensitive_assignments(value: str) -> str:
    quoted_pattern = re.compile(
        rf"(?i)(?P<prefix>(?:\\?[\"']){_SENSITIVE_KEY_PATTERN}"
        r"(?:\\?[\"'])\s*:\s*(?P<quote>\\?[\"']))(?P<secret>.*?)(?P=quote)"
    )

    def replace_quoted(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{match.group('prefix')}[REDACTED]{quote}"

    text = quoted_pattern.sub(replace_quoted, value)
    return re.sub(
        r"(?i)\b(?P<key>" + _SENSITIVE_KEY_PATTERN + r")\s*[:=]\s*(?!\\?[\"'])[^,;}&\r\n]+",
        r"\g<key>=[REDACTED]",
        text,
    )
