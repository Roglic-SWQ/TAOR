from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
import httpx

from mmkgc.ranking import factory


@pytest.fixture
def captured_request(monkeypatch):
    captured = {}

    async def handle(req):
        captured.update(url=str(req.url), headers=dict(req.headers), body=json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    real_client = httpx.AsyncClient

    def client(**kwargs):
        captured["timeout"] = kwargs["timeout"]
        return real_client(**kwargs, transport=httpx.MockTransport(handle), trust_env=False)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return captured


def test_openai_compatible_completion_client_uses_judge_token_floor(captured_request) -> None:
    captured = captured_request
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
    captured_request,
) -> None:
    captured = captured_request
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
    captured_request,
) -> None:
    captured = captured_request
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
    cancelled = []

    async def slow_request(req):
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(True)
        raise AssertionError("request deadline did not cancel the transport")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(slow_request), trust_env=False))
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
    assert cancelled
