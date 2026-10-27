from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib import error, request

import pytest
from PIL import Image

from mmkgc.evaluation.image_quality_targets import collect_entity_image_quality_targets
from mmkgc.evaluation.loader import iter_entity_artifact_results
from mmkgc.evaluation.qwen_vl_judge import (
    QWEN_VL_JUDGE_PROVIDER,
    QWEN_VL_JUDGE_SCORE_FIELDS,
    FixedQwenVLJudge,
    QwenVLJudgeClient,
    QwenVLJudgeConfig,
    QwenVLJudgeError,
    build_qwen_vl_judge_prompt,
    evaluate_entity_qwen_vl_judge,
    image_to_data_url,
    load_qwen_vl_judge_config,
    parse_qwen_vl_judge_response,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def minimal_image_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    entity_dir = run_dir / "entities" / "Q1"
    image_dir = entity_dir / "generated_images"
    image_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_dir / "image_001.png")
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {
            "entity_profile": {
                "entity_key": "Q1",
                "entity_label": "Ada Lovelace",
                "entity_type": "person",
            },
            "relation_facts": [
                {"relation_label": "occupation", "neighbor_label": "mathematician"}
            ],
        },
    )
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {
            "modality_constraints": {
                "image": {"must_include": ["portrait", "mathematician"]}
            }
        },
    )
    _write_json(
        entity_dir / "4_kg_aware_candidate_consistency_ranking.json",
        {
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Ada Lovelace was a mathematician."
                        },
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image_001",
                        "positive_prompt": (
                            "portrait of Ada Lovelace as a mathematician"
                        ),
                        "generated_content": {
                            "image_path": "generated_images/image_001.png"
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "text_candidate_id": "text_001",
                        "image_candidate_id": "image_001",
                    }
                ]
            },
        },
    )
    _write_json(
        entity_dir / "final.json",
        {"entity": {"entity_key": "Q1", "label": "Ada Lovelace"}},
    )
    return run_dir


def _artifacts(run_dir: Path):
    result = next(iter_entity_artifact_results(run_dir))
    assert result.artifacts is not None
    return result.artifacts


def _valid_judge_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "entity_match_score": 80,
        "prompt_alignment_score": 70,
        "kg_evidence_score": 60,
        "artifact_score": 90,
        "overall_judge_score": 75,
        "rationale": "图像基本符合。",
    }
    payload.update(overrides)
    return payload


class FailingSecondQwenVLJudge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def judge_image(self, image_path: Path, prompt: str) -> dict[str, object]:
        self.calls.append({"image_path": image_path, "prompt": prompt})
        if len(self.calls) == 2:
            raise QwenVLJudgeError("request timed out")
        return _valid_judge_payload()


def test_constants_define_qwen_vl_judge_provider_and_score_fields() -> None:
    assert QWEN_VL_JUDGE_PROVIDER == "qwen-vl-judge"
    assert QWEN_VL_JUDGE_SCORE_FIELDS == (
        "entity_match_score",
        "prompt_alignment_score",
        "kg_evidence_score",
        "artifact_score",
        "overall_judge_score",
    )


def test_load_qwen_vl_judge_config_reads_yaml_and_hides_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(
        """
lmm:
  model_name: Qwen/Qwen2-VL-7B-Instruct
  base_url: http://localhost:12345/v1
  api_key: super-secret-token
inference:
  max_token: 256
  temperature: 0.25
""".lstrip(),
        encoding="utf-8",
    )

    config = load_qwen_vl_judge_config(config_path, timeout_seconds=30)

    assert config == QwenVLJudgeConfig(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1",
        api_key="super-secret-token",
        max_tokens=256,
        temperature=0.25,
        timeout_seconds=30,
    )
    assert "super-secret-token" not in repr(config)


def test_load_qwen_vl_judge_config_reads_api_key_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
  temperature: 0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_qwen_vl_judge_config(config_path, timeout_seconds=45)

    assert config == QwenVLJudgeConfig(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="env-openai-key",
        max_tokens=1024,
        temperature=0.0,
        timeout_seconds=45,
    )
    assert "env-openai-key" not in repr(config)


def test_load_qwen_vl_judge_config_reads_model_base_url_and_api_key_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
  temperature: 0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_qwen_vl_judge_config(config_path, timeout_seconds=45)

    assert config == QwenVLJudgeConfig(
        model_name="gpt-4o-mini",
        base_url="http://35.220.164.252:3888/v1",
        api_key="proxy-openai-key",
        max_tokens=1024,
        temperature=0.0,
        timeout_seconds=45,
    )
    assert "proxy-openai-key" not in repr(config)


def test_image_to_data_url_encodes_png_and_jpeg(tmp_path: Path) -> None:
    png_path = tmp_path / "image.png"
    jpg_path = tmp_path / "photo.jpg"
    Image.new("RGB", (2, 2), color=(0, 128, 255)).save(png_path)
    Image.new("RGB", (2, 2), color=(255, 128, 0)).save(jpg_path)

    png_data_url = image_to_data_url(png_path)
    jpg_data_url = image_to_data_url(jpg_path)

    assert png_data_url.startswith("data:image/png;base64,")
    assert jpg_data_url.startswith("data:image/jpeg;base64,")


def test_image_to_data_url_rejects_missing_or_unsupported_file(tmp_path: Path) -> None:
    with pytest.raises(QwenVLJudgeError, match="image file does not exist"):
        image_to_data_url(tmp_path / "missing.png")

    unsupported = tmp_path / "image.gif"
    unsupported.write_bytes(b"GIF89a")
    with pytest.raises(QwenVLJudgeError, match="unsupported image extension"):
        image_to_data_url(unsupported)


def test_build_qwen_vl_judge_prompt_contains_context_and_json_only_requirement(
    minimal_image_run: Path,
) -> None:
    target = collect_entity_image_quality_targets(_artifacts(minimal_image_run))[0]

    prompt = build_qwen_vl_judge_prompt(target)

    assert "只输出一个 JSON object" in prompt
    assert "不要使用 Markdown" in prompt
    assert target.entity_key in prompt
    assert target.label in prompt
    assert target.image_candidate_id in prompt
    assert target.prompt_text in prompt
    assert target.candidate_text in prompt
    assert target.evidence_summary_text in prompt
    for field in QWEN_VL_JUDGE_SCORE_FIELDS:
        assert field in prompt
    assert "rationale" in prompt


def test_parse_qwen_vl_judge_response_accepts_json_object() -> None:
    parsed = parse_qwen_vl_judge_response(json.dumps(_valid_judge_payload()))

    assert parsed["overall_judge_score"] == 75.0
    assert parsed["rationale"] == "图像基本符合。"
    for field in QWEN_VL_JUDGE_SCORE_FIELDS:
        assert isinstance(parsed[field], float)


def test_parse_qwen_vl_judge_response_accepts_json_code_fence() -> None:
    parsed = parse_qwen_vl_judge_response(
        "```json\n" + json.dumps(_valid_judge_payload()) + "\n```"
    )

    assert parsed["entity_match_score"] == 80.0


def test_parse_qwen_vl_judge_response_rejects_missing_rationale() -> None:
    payload = _valid_judge_payload()
    del payload["rationale"]

    with pytest.raises(QwenVLJudgeError, match="missing required field"):
        parse_qwen_vl_judge_response(json.dumps(payload))


@pytest.mark.parametrize("field_name", QWEN_VL_JUDGE_SCORE_FIELDS)
def test_parse_qwen_vl_judge_response_rejects_missing_score_field(
    field_name: str,
) -> None:
    payload = _valid_judge_payload()
    del payload[field_name]

    with pytest.raises(QwenVLJudgeError, match=f"missing required field: {field_name}"):
        parse_qwen_vl_judge_response(json.dumps(payload))


def test_parse_qwen_vl_judge_response_rejects_out_of_range_score() -> None:
    with pytest.raises(QwenVLJudgeError, match="must be between 0 and 100"):
        parse_qwen_vl_judge_response(
            json.dumps(_valid_judge_payload(entity_match_score=101))
        )


@pytest.mark.parametrize("score_value", [float("nan"), float("inf"), float("-inf")])
def test_parse_qwen_vl_judge_response_rejects_non_finite_score(
    score_value: float,
) -> None:
    with pytest.raises(QwenVLJudgeError, match="must be finite"):
        parse_qwen_vl_judge_response(
            json.dumps(_valid_judge_payload(entity_match_score=score_value))
        )


@pytest.mark.parametrize("rationale", ["", "   "])
def test_parse_qwen_vl_judge_response_rejects_empty_rationale(
    rationale: str,
) -> None:
    with pytest.raises(QwenVLJudgeError, match="rationale must be a non-empty string"):
        parse_qwen_vl_judge_response(
            json.dumps(_valid_judge_payload(rationale=rationale))
        )


def test_parse_qwen_vl_judge_response_rejects_invalid_json() -> None:
    with pytest.raises(QwenVLJudgeError, match="invalid JSON"):
        parse_qwen_vl_judge_response("not json")


def test_fixed_qwen_vl_judge_records_calls_and_returns_configured_results() -> None:
    judge = FixedQwenVLJudge(
        [
            _valid_judge_payload(
                overall_judge_score=88,
                latency_ms=123,
                raw_response={"source": "fixed"},
            )
        ]
    )
    image_path = Path("image.png")

    result = judge.judge_image(image_path, "prompt text")

    assert result["overall_judge_score"] == 88.0
    assert result["latency_ms"] == 123
    assert result["raw_response"] == {"source": "fixed"}
    assert judge.calls == [{"image_path": image_path, "prompt": "prompt text"}]


def test_evaluate_entity_qwen_vl_judge_scores_each_image(
    minimal_image_run: Path,
) -> None:
    artifacts = _artifacts(minimal_image_run)
    judge = FixedQwenVLJudge([_valid_judge_payload()])

    result = evaluate_entity_qwen_vl_judge(
        artifacts,
        judge=judge,
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1/",
    )

    assert result["entity"] == {
        "entity_key": "Q1",
        "label": "Ada Lovelace",
        "missing_mode": None,
    }
    assert result["image_count"] == 1
    assert result["model_name"] == "Qwen/Qwen2-VL-7B-Instruct"
    assert result["base_url"] == "http://localhost:12345/v1"
    assert result["averages"]["overall_judge_score"] == 75.0
    assert result["best_image_candidate_id"] == "image_001"
    assert result["worst_image_candidate_id"] == "image_001"

    image_score = result["image_scores"][0]
    assert image_score["entity_key"] == "Q1"
    assert image_score["image_candidate_id"] == "image_001"
    assert image_score["image_path"] == "generated_images/image_001.png"
    assert image_score["provider"] == QWEN_VL_JUDGE_PROVIDER
    assert image_score["status"] == "judged"
    assert image_score["fallback_used"] is False
    assert image_score["parse_failed"] is False
    assert image_score["latency_ms"] == 0
    assert image_score["raw_response"] == {}
    assert image_score["overall_judge_score"] == 75.0
    assert image_score["rationale"] == "图像基本符合。"
    assert image_score["text_targets"] == {
        "prompt_text": "portrait of Ada Lovelace as a mathematician",
        "candidate_text": "Ada Lovelace was a mathematician.",
        "evidence_summary_text": (
            "Entity: Ada Lovelace. Entity key: Q1. Type: person. "
            "Evidence: occupation: mathematician. "
            "Visual requirements: portrait, mathematician"
        ),
    }
    assert len(judge.calls) == 1
    assert judge.calls[0]["image_path"].name == "image_001.png"
    assert "image_001" in judge.calls[0]["prompt"]


def test_evaluate_entity_qwen_vl_judge_records_failed_image_and_continues(
    minimal_image_run: Path,
) -> None:
    entity_dir = minimal_image_run / "entities" / "Q1"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(
        entity_dir / "generated_images" / "image_002.png"
    )
    (entity_dir / "generated_images" / "image_002.json").write_text(
        json.dumps(
            {
                "candidate_id": "image_002",
                "positive_prompt": "second image candidate",
            }
        ),
        encoding="utf-8",
    )
    artifacts = _artifacts(minimal_image_run)
    judge = FailingSecondQwenVLJudge()

    result = evaluate_entity_qwen_vl_judge(
        artifacts,
        judge=judge,
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1/",
    )

    assert result["image_count"] == 2
    assert len(judge.calls) == 2
    assert [score["status"] for score in result["image_scores"]] == [
        "judged",
        "failed",
    ]
    failed_score = result["image_scores"][1]
    assert failed_score["image_candidate_id"] == "image_002"
    assert failed_score["error"] == "request timed out"
    assert failed_score["fallback_used"] is True
    assert failed_score["parse_failed"] is False
    assert failed_score["overall_judge_score"] == 0.0
    assert result["averages"]["overall_judge_score"] == 37.5


def test_qwen_vl_judge_client_posts_openai_compatible_multimodal_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(image_path)
    captured: dict[str, object] = {}
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(_valid_judge_payload(overall_judge_score=91))
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(req: request.Request, timeout: float):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = QwenVLJudgeClient(
        QwenVLJudgeConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:12345/v1/",
            api_key="token-abc123",
            max_tokens=321,
            temperature=0.0,
            timeout_seconds=12,
        )
    )

    result = client.judge_image(image_path, "judge this image")

    assert result["overall_judge_score"] == 91.0
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0
    assert result["raw_response"] == response_payload
    assert "token-abc123" not in json.dumps(result["raw_response"])
    assert captured["url"] == "http://localhost:12345/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["headers"]["Authorization"] == "Bearer token-abc123"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["timeout"] == 12

    payload = captured["payload"]
    assert payload["model"] == "Qwen/Qwen2-VL-7B-Instruct"
    assert payload["max_tokens"] == 321
    assert payload["temperature"] == 0.0
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "judge this image"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_qwen_vl_judge_client_redacts_sensitive_raw_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "token-abc123"
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(image_path)
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(_valid_judge_payload()),
                    "api_key": secret,
                    "Authorization": f"Bearer {secret}",
                    "nested": {
                        "refresh_token": secret,
                        "echo": f"proxy echoed Authorization: Bearer {secret}",
                    },
                    "events": [
                        {"secret_value": secret},
                        f"raw bearer string Bearer {secret}",
                    ],
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(req: request.Request, timeout: float):
        return FakeResponse()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = QwenVLJudgeClient(
        QwenVLJudgeConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:12345/v1",
            api_key=secret,
        )
    )

    result = client.judge_image(image_path, "judge this image")
    raw_response_text = json.dumps(result["raw_response"], ensure_ascii=False)

    assert secret not in raw_response_text
    assert f"Bearer {secret}" not in raw_response_text
    assert "[REDACTED]" in raw_response_text
    message = result["raw_response"]["choices"][0]["message"]
    assert message["api_key"] == "[REDACTED]"
    assert message["Authorization"] == "[REDACTED]"
    assert message["nested"]["refresh_token"] == "[REDACTED]"
    assert "Bearer [REDACTED]" in message["nested"]["echo"]


def test_qwen_vl_judge_client_raises_on_http_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(image_path)

    def fake_urlopen(req: request.Request, timeout: float):
        raise error.HTTPError(
            req.full_url,
            500,
            "server exploded",
            hdrs={},
            fp=BytesIO(b'{"error":"boom"}'),
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = QwenVLJudgeClient(
        QwenVLJudgeConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:12345/v1",
            api_key="token-abc123",
        )
    )

    with pytest.raises(QwenVLJudgeError, match="HTTP error"):
        client.judge_image(image_path, "judge this image")


def test_qwen_vl_judge_client_redacts_sensitive_http_error_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "token-abc123"
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(image_path)
    error_body = (
        '{"error":"proxy echoed api_key token-abc123",'
        '"Authorization":"Bearer token-abc123"}'
    ).encode("utf-8")

    def fake_urlopen(req: request.Request, timeout: float):
        raise error.HTTPError(
            req.full_url,
            502,
            "bad gateway",
            hdrs={},
            fp=BytesIO(error_body),
        )

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = QwenVLJudgeClient(
        QwenVLJudgeConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:12345/v1",
            api_key=secret,
        )
    )

    with pytest.raises(QwenVLJudgeError) as exc_info:
        client.judge_image(image_path, "judge this image")

    error_text = str(exc_info.value)
    assert "HTTP error" in error_text
    assert secret not in error_text
    assert f"Bearer {secret}" not in error_text
    assert "[REDACTED]" in error_text


def test_qwen_vl_judge_client_raises_on_url_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(image_path)

    def fake_urlopen(req: request.Request, timeout: float):
        raise error.URLError("connection refused")

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = QwenVLJudgeClient(
        QwenVLJudgeConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:12345/v1",
            api_key="token-abc123",
        )
    )

    with pytest.raises(QwenVLJudgeError, match="URL error"):
        client.judge_image(image_path, "judge this image")


@pytest.mark.parametrize(
    "response_payload",
    [
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": ""}}]},
    ],
)
def test_qwen_vl_judge_client_raises_on_malformed_message_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_payload: dict[str, object],
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), color=(50, 60, 70)).save(image_path)

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(req: request.Request, timeout: float):
        return FakeResponse()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)
    client = QwenVLJudgeClient(
        QwenVLJudgeConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:12345/v1",
            api_key="token-abc123",
        )
    )

    with pytest.raises(QwenVLJudgeError, match="message.content|content"):
        client.judge_image(image_path, "judge this image")
