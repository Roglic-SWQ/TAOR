from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from mmkgc.ranking import factory


def test_openai_compatible_completion_client_uses_judge_token_floor(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(factory.request, "urlopen", fake_urlopen)
    client = factory._OpenAICompatibleCompletionClient(
        SimpleNamespace(
            base_url="http://localhost:12345/v1",
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            api_key="",
            temperature=0.1,
            max_token=512,
        ),
        timeout_seconds=120,
    )

    assert client.complete("prompt") == "{}"
    assert captured["timeout"] == 120
    assert captured["body"]["max_tokens"] == 1024
    assert "response_format" not in captured["body"]


def test_openai_compatible_completion_client_can_target_gpt_4o_mini_api(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = {name.lower(): value for name, value in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(factory.request, "urlopen", fake_urlopen)
    client = factory._OpenAICompatibleCompletionClient(
        SimpleNamespace(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            api_key="env-openai-key",
            temperature=0.0,
            max_token=512,
        ),
        timeout_seconds=90,
    )

    assert client.complete("prompt") == "{}"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer env-openai-key"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["max_tokens"] == 1024
    assert captured["timeout"] == 90


def test_openai_compatible_completion_client_sends_multimodal_image_url(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(factory.request, "urlopen", fake_urlopen)
    client = factory._OpenAICompatibleCompletionClient(
        SimpleNamespace(
            base_url="http://localhost:12345/v1",
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            api_key="",
            temperature=0.1,
            max_token=512,
        ),
        timeout_seconds=120,
    )

    assert client.complete_multimodal("prompt", "data:image/png;base64,AAA=") == "{}"

    messages = captured["body"]["messages"]
    assert messages[1]["content"] == [
        {"type": "text", "text": "prompt"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
    ]
    assert captured["body"]["max_tokens"] == 1024


def test_openai_compatible_completion_client_enforces_hard_timeout(
    monkeypatch,
) -> None:
    def slow_urlopen(_req, timeout):
        time.sleep(1.0)
        raise AssertionError("hard timeout did not interrupt urlopen")

    monkeypatch.setattr(factory.request, "urlopen", slow_urlopen)
    client = factory._OpenAICompatibleCompletionClient(
        SimpleNamespace(
            base_url="http://localhost:12345/v1",
            model_name="gpt-4o-mini",
            api_key="",
            temperature=0.1,
            max_token=512,
        ),
        timeout_seconds=0.05,
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="ranking judge request timed out"):
        client.complete("prompt")
    assert time.monotonic() - started < 0.5
