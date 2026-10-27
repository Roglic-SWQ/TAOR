import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from mmkgc.adapters.qwen import (
    QwenConfig,
    QwenLocalImagePromptGenerator,
    QwenLocalTextGenerator,
    load_qwen_config,
)


class _FakeHTTPResponse:
    """Minimal response object to emulate urllib open/read context manager."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload if isinstance(self._payload, (bytes, bytearray)) else json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _config_path(tmp_path):
    content = """
lmm:
  model_name: ./Qwen/Qwen2-VL-7B-Instruct
  base_url: http://localhost:12345/v1
  api_key: test-api-key
inference:
  max_token: 192
  temperature: 0.2
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_qwen_config_reads_fields(tmp_path):
    config_path = _config_path(tmp_path)
    config = load_qwen_config(config_path)

    assert config.model_name == "./Qwen/Qwen2-VL-7B-Instruct"
    assert config.base_url == "http://localhost:12345/v1"
    assert config.api_key == "test-api-key"
    assert config.max_token == 192
    assert config.temperature == 0.2
    assert "test-api-key" not in repr(config)


def test_load_qwen_config_reads_api_key_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    config_path = tmp_path / "openai.yaml"
    config_path.write_text(
        """
lmm:
  model_name: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
inference:
  max_token: 1024
  temperature: 0.0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_qwen_config(config_path)

    assert config.model_name == "gpt-4o-mini"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "env-openai-key"
    assert config.max_token == 1024
    assert config.temperature == 0.0
    assert "env-openai-key" not in repr(config)


def test_load_qwen_config_reads_model_base_url_and_api_key_from_environment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-openai-key")
    config_path = tmp_path / "openai-proxy.yaml"
    config_path.write_text(
        """
lmm:
  model_name_env: DEFAULT_MODEL
  base_url_env: OPENAI_BASE_URL
  api_key_env: OPENAI_API_KEY
inference:
  max_token: 1024
  temperature: 0.0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_qwen_config(config_path)

    assert config.model_name == "gpt-4o-mini"
    assert config.base_url == "http://35.220.164.252:3888/v1"
    assert config.api_key == "proxy-openai-key"
    assert "proxy-openai-key" not in repr(config)


def test_generate_success_returns_response_text_and_records_generated(monkeypatch):
    config = QwenConfig(
        model_name="./Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1/",
        api_key="test-api-key",
        max_token=128,
        temperature=0.1,
    )
    generator = QwenLocalTextGenerator(config)
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": "Concise two sentence profile.",
                }
            }
        ]
    }

    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        assert req.full_url == "http://localhost:12345/v1/chat/completions"
        assert req.get_method() == "POST"
        assert timeout == 5.0
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "./Qwen/Qwen2-VL-7B-Instruct"
        assert payload["max_tokens"] == 128
        assert payload["temperature"] == 0.1
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "max_tokens" in payload
        headers = {name.lower(): value for name, value in req.header_items()}
        assert headers["authorization"].startswith("Bearer ")
        assert headers["authorization"].endswith("test-api-key")
        assert headers["content-type"] == "application/json"
        assert isinstance(req, Request)
        return _FakeHTTPResponse(response_payload)

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)
    generator = QwenLocalTextGenerator(config, timeout_seconds=5.0)
    generated = generator.generate(
        "Example Film", "film", "director", "identity", ["has director as Alice"]
    )
    metadata = generator.last_candidate_metadata()

    assert generated == "Concise two sentence profile."
    assert metadata["generator"] == "qwen2_vl_7b"
    assert metadata["model_name"] == "./Qwen/Qwen2-VL-7B-Instruct"
    assert metadata["model_status"] == "generated"
    assert metadata["fallback_used"] is False
    call = generator.consume_last_call("Q1", "text_001")

    assert call["provider"] == "qwen-local-openai-compatible"
    assert call["model_name"] == "./Qwen/Qwen2-VL-7B-Instruct"
    assert call["base_url"] == "http://localhost:12345/v1"
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 128
    assert call["status"] == "generated"
    assert "error" in call
    assert call["entity_key"] == "Q1"
    assert call["candidate_id"] == "text_001"
    assert "prompt" in call
    assert call["response_text"] == "Concise two sentence profile."
    assert call["error"] is None
    assert call["fallback_used"] is False
    assert set(["entity_key", "candidate_id", "provider", "model_name", "base_url", "temperature", "max_tokens", "prompt", "response_text", "latency_ms", "status", "error", "fallback_used"]).issubset(call)
    assert "api_key" not in call
    assert captured["timeout"] == 5.0

    # Consumption should be one-time.
    assert generator.consume_last_call("Q1", "text_001") is None


def test_image_prompt_generator_returns_json_prompts_and_records_call(monkeypatch):
    config = QwenConfig(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1/",
        api_key="test-api-key",
        max_token=256,
        image_prompt_max_token=256,
        temperature=0.0,
    )
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "positive_prompt": (
                                "comedy-drama film scene, arctic coastline, "
                                "gray whales trapped in sea ice, rescue workers in winter gear"
                            ),
                            "negative_prompt": "logos, readable text, recognizable actors",
                            "used_text_cues": ["gray whales", "sea ice", "rescue workers"],
                            "omitted_risky_details": ["exact actors", "official poster"],
                        }
                    )
                }
            }
        ]
    }
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        payload = json.loads(req.data.decode("utf-8"))
        captured["payload"] = payload
        assert req.full_url == "https://api.openai.com/v1/chat/completions"
        assert payload["model"] == "gpt-4o-mini"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 256
        headers = {name.lower(): value for name, value in req.header_items()}
        assert headers["authorization"] == "Bearer test-api-key"
        return _FakeHTTPResponse(response_payload)

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)

    generator = QwenLocalImagePromptGenerator(
        config,
        timeout_seconds=7.0,
        context={
            "entity_text_excerpt": "Big Miracle is about gray whales trapped in sea ice near Alaska.",
            "text_visual_cues": ["arctic coastline", "winter rescue scene"],
            "text_context_cues": ["gray whales", "sea ice", "rescue workers"],
        },
    )
    prompt = generator.generate(
        "Big Miracle",
        "film",
        "cast member",
        "identity anchor",
        ["Big Miracle has cast member: Drew Barrymore"],
    )

    assert "gray whales trapped in sea ice" in prompt["positive_prompt"]
    call = generator.consume_last_call("Q609236", "image_gpt_001")

    assert call["provider"] == "openai-compatible-image-prompt-planner"
    assert call["stage"] == "relation_alternation_generation_gpt_prompt"
    assert call["modality"] == "image_prompt"
    assert call["model_name"] == "gpt-4o-mini"
    assert call["entity_key"] == "Q609236"
    assert call["candidate_id"] == "image_gpt_001"
    assert call["status"] == "generated"
    assert call["fallback_used"] is False
    assert "gray whales" in call["prompt"]
    assert "api_key" not in call
    assert captured["timeout"] == 7.0


def test_image_prompt_generator_preserves_person_entity_name_in_positive_prompt(monkeypatch):
    config = QwenConfig(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-api-key",
        max_token=256,
        temperature=0.0,
    )
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "positive_prompt": (
                                "Kanye West performing on stage, vibrant lights, "
                                "energetic crowd, hip-hop concert atmosphere"
                            ),
                            "negative_prompt": "recognizable face",
                            "used_text_cues": ["music performance"],
                            "omitted_risky_details": ["exact likeness"],
                        }
                    )
                }
            }
        ]
    }

    captured = {}

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        captured["user_prompt"] = payload["messages"][1]["content"]
        return _FakeHTTPResponse(response_payload)

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)
    generator = QwenLocalImagePromptGenerator(config)

    prompt = generator.generate(
        "Kanye West",
        "music artist",
        "performer",
        "identity anchor",
        ["Example Song has performer: Kanye West"],
    )

    assert "Kanye West" in prompt["positive_prompt"]
    assert prompt["positive_prompt"].startswith("Kanye West performing on stage")
    assert "blocked_positive_terms" not in captured["user_prompt"]
    assert "Do not include the exact entity_label" not in captured["user_prompt"]


def test_image_prompt_generator_preserves_nonperson_entity_label_in_positive_prompt(monkeypatch):
    config = QwenConfig(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-api-key",
        max_token=256,
        temperature=0.0,
    )
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "positive_prompt": (
                                "Big Miracle poster-style arctic rescue scene, "
                                "gray whales trapped in sea ice, winter coastline"
                            ),
                            "negative_prompt": "readable title text",
                            "used_text_cues": ["arctic rescue"],
                            "omitted_risky_details": ["official poster"],
                        }
                    )
                }
            }
        ]
    }

    def fake_urlopen(_req, timeout=None):
        return _FakeHTTPResponse(response_payload)

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)
    generator = QwenLocalImagePromptGenerator(config)

    prompt = generator.generate(
        "Big Miracle",
        "film",
        "cast member",
        "identity anchor",
        ["Big Miracle has cast member: Drew Barrymore"],
    )

    assert "Big Miracle" in prompt["positive_prompt"]
    assert prompt["positive_prompt"].startswith("Big Miracle poster-style arctic rescue scene")


def test_generate_httperror_prefers_http_error_metadata(monkeypatch):
    config = QwenConfig(
        model_name="./Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1",
        api_key="test-api-key",
    )
    generator = QwenLocalTextGenerator(config)

    def fake_urlopen(_req, timeout=None):
        raise HTTPError("http://localhost:12345/v1/chat/completions", 503, "service unavailable", None, None)

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)
    with pytest.raises(HTTPError):
        generator.generate(
            "Example Film",
            "film",
            "director",
            "identity",
            ["has director as Alice"],
        )
    metadata = generator.last_candidate_metadata()
    call = generator.consume_last_call("Q1", "text_001")

    assert call["status"] == "failed"
    assert call["error"].startswith("HTTPError: 503 service unavailable")
    assert metadata["model_status"] == "failed"
    assert metadata["model_error"].startswith("HTTPError: 503 service unavailable")
    assert metadata["fallback_used"] is False


def test_generate_failure_is_recorded_and_raised(monkeypatch):
    config = QwenConfig(
        model_name="./Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1",
        api_key="test-api-key",
    )
    generator = QwenLocalTextGenerator(config)
    facts = ["A sparse fact about the film."]

    captured: dict[str, object] = {}

    def fake_urlopen(_req, timeout=None):
        captured["timeout"] = timeout
        raise URLError("forced network failure")

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)
    with pytest.raises(URLError, match="forced network failure"):
        generator.generate("Example Film", "film", "director", "identity", facts)
    metadata = generator.last_candidate_metadata()
    call = generator.consume_last_call("Q1", "text_001")

    assert metadata["generator"] == "qwen2_vl_7b"
    assert metadata["model_name"] == "./Qwen/Qwen2-VL-7B-Instruct"
    assert metadata["model_status"] == "failed"
    assert metadata["fallback_used"] is False
    assert metadata["model_error"] == "URLError: forced network failure"
    assert call["status"] == "failed"
    assert call["max_tokens"] == 512
    assert call["temperature"] == 0.1
    assert call["base_url"] == "http://localhost:12345/v1"
    assert call["error"] == "URLError: forced network failure"
    assert call["fallback_used"] is False
    assert call["response_text"] == ""
    assert call["provider"] == "qwen-local-openai-compatible"
    assert call["model_name"] == "./Qwen/Qwen2-VL-7B-Instruct"
    assert captured["timeout"] == 60.0



def test_image_prompt_generator_records_initial_and_repaired_responses(monkeypatch):
    config = QwenConfig(
        model_name="gpt-4o-mini",
        base_url="http://localhost:12345/v1",
        api_key="test-api-key",
        image_prompt_max_token=256,
        temperature=0.0,
    )
    initial_content = '{"positive_prompt": "truncated'
    repaired_content = json.dumps(
        {
            "positive_prompt": "a grounded scenic landscape",
            "used_text_cues": ["landscape"],
            "omitted_risky_details": [],
        }
    )
    responses = iter(
        [
            {"choices": [{"message": {"content": initial_content}}]},
            {"choices": [{"message": {"content": repaired_content}}]},
        ]
    )

    monkeypatch.setattr(
        "mmkgc.adapters.qwen.request.urlopen",
        lambda _req, timeout=None: _FakeHTTPResponse(next(responses)),
    )
    generator = QwenLocalImagePromptGenerator(config)

    result = generator.generate(
        "Example Place",
        "place",
        "global_profile",
        "identity anchor",
        ["Example Place is a place"],
    )
    call = generator.consume_last_call("Q1", "image_001")

    assert result["positive_prompt"] == "a grounded scenic landscape"
    assert call["initial_response_text"] == initial_content
    assert call["repaired_response_text"] == repaired_content
    assert call["initial_response"]["choices"]
    assert call["repaired_response"]["choices"]


def test_image_prompt_generator_records_failed_repair_responses(monkeypatch):
    config = QwenConfig(
        model_name="gpt-4o-mini",
        base_url="http://localhost:12345/v1",
        api_key="test-api-key",
        image_prompt_max_token=256,
        temperature=0.0,
    )
    initial_content = '{"positive_prompt": "first'
    repaired_content = '{"positive_prompt": "second'
    responses = iter(
        [
            {"choices": [{"message": {"content": initial_content}}]},
            {"choices": [{"message": {"content": repaired_content}}]},
        ]
    )
    monkeypatch.setattr(
        "mmkgc.adapters.qwen.request.urlopen",
        lambda _req, timeout=None: _FakeHTTPResponse(next(responses)),
    )
    generator = QwenLocalImagePromptGenerator(config)

    with pytest.raises(json.JSONDecodeError) as raised:
        generator.generate(
            "Example Place",
            "place",
            "global_profile",
            "identity anchor",
            ["Example Place is a place"],
        )

    call = generator.consume_last_call("Q1", "image_001")
    assert call["status"] == "failed"
    assert call["initial_response_text"] == initial_content
    assert call["repaired_response_text"] == repaired_content
    assert call["initial_response"]["choices"]
    assert call["repaired_response"]["choices"]
    assert raised.value.model_call["initial_response_text"] == initial_content



def test_image_prompt_generator_uses_fresh_repair_prompt_for_length_truncation(monkeypatch):
    config = QwenConfig(
        model_name="gpt-4o-mini",
        base_url="http://localhost:12345/v1",
        api_key="test-api-key",
        image_prompt_max_token=384,
        temperature=0.0,
    )
    initial_content = '{"positive_prompt": "repeated truncated content'
    repaired_content = json.dumps(
        {
            "positive_prompt": "a concise medieval stone town landscape",
            "used_text_cues": ["medieval architecture"],
            "omitted_risky_details": [],
        }
    )
    requests = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {"content": initial_content},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 1024},
            },
            {
                "choices": [
                    {
                        "message": {"content": repaired_content},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
    )

    def fake_urlopen(req, timeout=None):
        requests.append(json.loads(req.data.decode("utf-8")))
        return _FakeHTTPResponse(next(responses))

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", fake_urlopen)
    generator = QwenLocalImagePromptGenerator(config)

    result = generator.generate(
        "Example Place",
        "place",
        "global_profile",
        "identity anchor",
        ["Example Place is a place"],
    )
    call = generator.consume_last_call("Q1", "image_001")

    repair_prompt = requests[1]["messages"][1]["content"]
    assert result["positive_prompt"] == "a concise medieval stone town landscape"
    assert initial_content not in repair_prompt
    assert "Discard it completely" in repair_prompt
    assert call["initial_finish_reason"] == "length"
    assert call["repaired_finish_reason"] == "stop"
    assert call["max_tokens"] == 384
