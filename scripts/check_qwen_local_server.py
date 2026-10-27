#!/usr/bin/env python3
"""Health check for the local Qwen OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import signal
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "mkgw-knowledge-extraction.yaml"
DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: Path) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {}
    section: str | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1]
            config[section] = {}
            continue

        if line.startswith("  ") and section:
            key, separator, value = stripped.partition(":")
            if not separator:
                raise ValueError(f"Invalid YAML line {line_number}: {raw_line}")
            config[section][key.strip()] = _parse_scalar(value)
            continue

        raise ValueError(f"Unsupported YAML line {line_number}: {raw_line}")

    return config


def latency_ms(started_at: float) -> int:
    return int(round((time.monotonic() - started_at) * 1000))


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def health_check_timeout_seconds() -> float:
    raw_timeout = os.environ.get(
        "QWEN_HEALTH_TIMEOUT_SECONDS", str(DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS)
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            f"QWEN_HEALTH_TIMEOUT_SECONDS must be numeric, got {raw_timeout!r}"
        ) from exc

    if timeout_seconds <= 0:
        raise ValueError("QWEN_HEALTH_TIMEOUT_SECONDS must be greater than 0")

    return timeout_seconds


def install_deadline(timeout_seconds: float) -> signal.Handlers | int | None:
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return None

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"health check exceeded {timeout_seconds:.1f} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    return previous_handler


def clear_deadline(previous_handler: signal.Handlers | int | None) -> None:
    if previous_handler is None:
        return

    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_handler)


def preview_response(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message")
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content", ""))
    elif "text" in first:
        content = str(first.get("text", ""))

    return " ".join(content.split())[:200]


def main() -> int:
    started_at = time.monotonic()
    previous_deadline_handler: signal.Handlers | int | None = None

    try:
        timeout_seconds = health_check_timeout_seconds()
        socket.setdefaulttimeout(timeout_seconds)
        previous_deadline_handler = install_deadline(timeout_seconds)

        config = load_config(CONFIG_PATH)
        lmm_config = config.get("lmm", {})
        inference_config = config.get("inference", {})
        model_name = str(lmm_config["model_name"])
        base_url = str(lmm_config["base_url"]).rstrip("/")
        api_key = str(lmm_config["api_key"])

        request_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 8,
            "temperature": float(inference_config.get("temperature", 0.0)),
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")

        response_payload = json.loads(response_body)
        print_json(
            {
                "ok": True,
                "model_name": model_name,
                "base_url": base_url,
                "latency_ms": latency_ms(started_at),
                "response_preview": preview_response(response_payload),
            }
        )
        return 0

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        error = f"HTTP {exc.code}: {body or exc.reason}"
        error_type = "http_error"
    except urllib.error.URLError as exc:
        error = str(exc.reason)
        error_type = "connection_error"
    except (socket.timeout, TimeoutError) as exc:
        error = str(exc)
        error_type = "timeout"
    except Exception as exc:  # noqa: BLE001 - health checks should report all failures as JSON.
        error = str(exc)
        error_type = exc.__class__.__name__
    finally:
        clear_deadline(previous_deadline_handler)

    print_json(
        {
            "ok": False,
            "error_type": error_type,
            "error": error,
            "latency_ms": latency_ms(started_at),
        }
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
