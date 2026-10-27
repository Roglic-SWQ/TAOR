from __future__ import annotations

import json
import math

import pytest

from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.contracts import GateResult, ScoreResult
from mmkgc.ranking.image_input import ImageInput
from mmkgc.ranking.scorers import (
    GPT4OMiniJudgeParseError,
    GPT4OMiniRankingJudge,
    build_image_judge_prompt,
    build_pair_judge_prompt,
    build_text_graph_extraction_prompt,
    build_text_judge_prompt,
    build_visual_graph_extraction_prompt,
    parse_image_judge_response,
    parse_pair_judge_response,
    parse_text_graph_response,
    parse_text_judge_response,
    parse_visual_graph_response,
)
from mmkgc.ranking.scorers.gpt4omini_judge import MAX_REPAIR_RESPONSE_CHARS


GROUND_TRUTH_SENTINEL = "ZZZ_SECRET_GROUND_TRUTH"
INSTRUCTION_SENTINEL = "ZZZ_SECRET_INSTRUCTION"
CONSENSUS_SENTINEL = "ZZZ_SECRET_CONSENSUS"
IMAGE_PATH_SENTINEL = "/private/ZZZ_SECRET_TRANSPORT/image.png"


class FakeJudgeClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("unexpected judge call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeJudgeMultimodalClient(FakeJudgeClient):
    def __init__(self, responses: list[str | Exception]) -> None:
        super().__init__(responses)
        self.multimodal_calls: list[dict[str, object]] = []

    def complete_multimodal(self, prompt: str, image_data_url: str) -> str:
        self.multimodal_calls.append(
            {"prompt": prompt, "image_data_url": image_data_url}
        )
        if not self.responses:
            raise AssertionError("unexpected judge multimodal call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_gpt4omini_prompt_builders_use_prompt_payload_and_generated_value_only() -> None:
    context = _build_context()
    text_candidate = dict(context.text_candidates[0])
    image_candidate = dict(context.image_candidates[0])
    prompt_payload_text = json.dumps(context.to_prompt_payload(), sort_keys=True)

    text_prompt = build_text_judge_prompt(context, text_candidate)
    image_prompt = build_image_judge_prompt(context, image_candidate)
    pair_prompt = build_pair_judge_prompt(context, text_candidate, image_candidate)

    assert "Example Film is a generated caption." in text_prompt
    assert "image_artifact_001" not in text_prompt
    assert "Example Film is a generated caption." not in image_prompt
    assert "image_artifact_001" in image_prompt
    assert "Example Film is a generated caption." in pair_prompt
    assert "image_artifact_001" in pair_prompt

    for prompt in (text_prompt, image_prompt, pair_prompt):
        assert "Return exactly one JSON object" in prompt
        assert "Evidence strings must cite candidate or context facts" in prompt
        assert "Keep evidence strings and reason concise" in prompt
        assert prompt_payload_text in prompt
        assert GROUND_TRUTH_SENTINEL not in prompt
        assert INSTRUCTION_SENTINEL not in prompt
        assert CONSENSUS_SENTINEL not in prompt
        assert "prompt_or_instruction" not in prompt
        assert "reviewer_consensus" not in prompt
        assert "offline_evaluation" not in prompt
        assert '"ground_truth":' not in prompt


def test_graph_extraction_prompts_are_kg_blind_and_parsers_are_strict() -> None:
    context = _build_context()
    text_candidate = dict(context.text_candidates[0])
    text_prompt = build_text_graph_extraction_prompt(text_candidate)
    visual_prompt = build_visual_graph_extraction_prompt()

    assert "Example Film is a generated caption." in text_prompt
    assert "relation_facts" not in text_prompt
    assert "supported_facts" not in text_prompt
    assert "attached image pixels" in visual_prompt
    assert "KG" not in visual_prompt

    text_graph = parse_text_graph_response(
        json.dumps(
            {
                "triples": [
                    {
                        "subject": "Example Film",
                        "predicate": "director",
                        "object": "Jane Example",
                        "polarity": "positive",
                        "confidence": 0.9,
                        "evidence": "Jane Example directed Example Film.",
                    }
                ]
            }
        )
    )
    visual_graph = parse_visual_graph_response(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "person_1",
                        "label": "person",
                        "attributes": ["black robe"],
                        "bbox": [0, 0, 10, 10],
                        "confidence": 0.8,
                        "evidence": "central person",
                    }
                ],
                "edges": [],
                "ocr": [],
            }
        )
    )

    assert text_graph["triples"][0]["predicate"] == "director"
    assert visual_graph["nodes"][0]["attributes"] == ["black robe"]


def test_visual_graph_extraction_repairs_non_string_attributes() -> None:
    invalid_response = json.dumps(
        {
            "nodes": [
                {
                    "id": "poster",
                    "label": "poster",
                    "attributes": {"color": "dark"},
                    "bbox": [],
                    "confidence": 0.8,
                    "evidence": "visible poster",
                }
            ],
            "edges": [],
            "ocr": [],
        }
    )
    repaired_response = json.dumps(
        {
            "nodes": [
                {
                    "id": "poster",
                    "label": "poster",
                    "attributes": ["dark"],
                    "bbox": [],
                    "confidence": 0.8,
                    "evidence": "visible poster",
                }
            ],
            "edges": [],
            "ocr": [],
        }
    )
    client = FakeJudgeMultimodalClient([invalid_response, repaired_response])
    judge = GPT4OMiniRankingJudge(client=client)
    image_input = ImageInput.from_bytes(
        entity_key="Q1",
        candidate_id="image_001",
        image_bytes=_png_bytes(),
    )

    graph = judge.extract_visual_graph(
        {"candidate_id": "image_001"},
        image_input=image_input,
    )

    assert graph["nodes"][0]["attributes"] == ["dark"]
    assert len(client.multimodal_calls) == 1
    assert len(client.prompts) == 1
    assert "attributes field must be an array of strings" in client.prompts[0]


def test_gpt4omini_prompt_builder_does_not_leak_generated_content_image_path() -> None:
    context = _build_context()
    image_candidate = {
        "candidate_id": "image_raw",
        "target_modality": "image",
        "generated_content": {
            "type": "image",
            "value": "visible image description",
            "image_path": IMAGE_PATH_SENTINEL,
            "image_artifact_id": "image_artifact_raw",
        },
    }

    image_prompt = build_image_judge_prompt(context, image_candidate)
    pair_prompt = build_pair_judge_prompt(
        context,
        dict(context.text_candidates[0]),
        image_candidate,
    )

    assert IMAGE_PATH_SENTINEL not in image_prompt
    assert IMAGE_PATH_SENTINEL not in pair_prompt
    assert "image_path" not in image_prompt
    assert "image_path" not in pair_prompt
    assert "visible image description" in image_prompt
    assert "image_artifact_raw" in image_prompt


def test_gpt4omini_image_and_pair_prompts_explain_textual_image_payloads() -> None:
    context = _build_context()
    image_candidate = dict(context.image_candidates[0])
    text_candidate = dict(context.text_candidates[0])

    image_prompt = build_image_judge_prompt(context, image_candidate)
    pair_prompt = build_pair_judge_prompt(context, text_candidate, image_candidate)

    for prompt in (image_prompt, pair_prompt):
        assert "image candidate is represented by generation prompt metadata" in prompt
        assert "Do not reject solely because the candidate type is image" in prompt
        assert "Forbidden or avoid-list items are checks to apply" in prompt
        assert "not evidence that the candidate contains those items" in prompt
        assert "must_include and should_include are positive grounding targets" in prompt
        assert "entity, identity, film, and grounded details are allowed descriptors" in prompt
        assert "Use low score and soft_violations for ordinary grounding weakness" in prompt


def test_gpt4omini_image_and_pair_prompts_require_visual_evidence_sources() -> None:
    context = _build_context()
    image_candidate = dict(context.image_candidates[0])
    text_candidate = dict(context.text_candidates[0])
    image_input = ImageInput.from_bytes(
        entity_key=context.entity_key,
        candidate_id="image_001",
        image_bytes=_png_bytes(),
    )

    prompt_without_pixels = build_image_judge_prompt(context, image_candidate)
    image_prompt = build_image_judge_prompt(
        context,
        image_candidate,
        image_input=image_input,
    )
    pair_prompt = build_pair_judge_prompt(
        context,
        text_candidate,
        image_candidate,
        image_input=image_input,
    )

    assert "Without attached pixels, score must be <= 50" in prompt_without_pixels
    for prompt in (image_prompt, pair_prompt):
        assert "Evidence strings must start with pixel:, metadata:, or context:" in prompt
        assert "KG/context facts are requirements, not pixel evidence" in prompt
        assert "Scores above 60 require at least one pixel: evidence item" in prompt
        assert "Generic entity-type matches alone must score <= 50" in prompt
        assert "metadata is provenance, not proof that pixels contain a fact" in prompt


def test_gpt4omini_ranking_judge_calls_once_per_candidate_or_pair() -> None:
    client = FakeJudgeClient(
        [
            _response(score=82, hard_reject=False, hard_rejection_codes=[]),
            _response(score=61, hard_reject=True, hard_rejection_codes=["forbidden_content"]),
            _response(score=73, hard_reject=False, hard_rejection_codes=[]),
        ]
    )
    judge = GPT4OMiniRankingJudge(client=client, model_name="gpt4omini-test")
    context = _build_context()

    text_result = judge.judge_text(context, dict(context.text_candidates[0]))
    image_result = judge.judge_image(context, dict(context.image_candidates[0]))
    pair_result = judge.judge_pair(
        context,
        dict(context.text_candidates[0]),
        dict(context.image_candidates[0]),
    )

    assert len(client.prompts) == 3
    assert text_result.score == ScoreResult(82.0, 0.82, "scored", "gpt4omini-test")
    assert text_result.gate == GateResult(
        "pass",
        "gpt4omini-test",
        judge_confidence=0.7,
        evidence=("clear match",),
        reason="ok",
    )
    assert image_result.gate == GateResult(
        "reject",
        "gpt4omini-test",
        hard_violations=("forbidden_content",),
        judge_confidence=0.7,
        evidence=("clear match",),
        reason="ok",
    )
    assert pair_result.score.normalized_score == 0.73


def test_gpt4omini_ranking_judge_retries_retryable_service_errors() -> None:
    client = FakeJudgeClient(
        [
            TimeoutError("first attempt timed out"),
            _response(score=84, hard_reject=False, hard_rejection_codes=[]),
        ]
    )
    judge = GPT4OMiniRankingJudge(
        client=client,
        model_name="gpt4omini-test",
        max_retries=1,
        retry_backoff_seconds=0.0,
    )

    result = judge.judge_text(_build_context(), dict(_build_context().text_candidates[0]))

    assert result.score.raw_score == 84.0
    assert len(client.prompts) == 2


def test_gpt4omini_ranking_judge_exposes_retry_diagnostic_after_exhaustion() -> None:
    client = FakeJudgeClient([TimeoutError("still timed out")])
    judge = GPT4OMiniRankingJudge(
        client=client,
        model_name="gpt4omini-test",
        max_retries=0,
    )

    with pytest.raises(TimeoutError) as exc_info:
        judge.judge_text(_build_context(), dict(_build_context().text_candidates[0]))

    diagnostic = exc_info.value.gpt4omini_judge_diagnostic
    assert diagnostic["kind"] == "text"
    assert diagnostic["candidate_id"] == "text_001"
    assert diagnostic["attempt"] == 1
    assert diagnostic["max_attempts"] == 1
    assert diagnostic["retryable"] is True
    assert diagnostic["will_retry"] is False


def test_gpt4omini_image_and_pair_judge_send_real_png_data_url() -> None:
    client = FakeJudgeMultimodalClient(
        [
            _response(score=61),
            _response(score=73),
        ]
    )
    judge = GPT4OMiniRankingJudge(client=client, model_name="gpt4omini-test")
    context = _build_context()
    image_input = ImageInput.from_bytes(
        entity_key=context.entity_key,
        candidate_id="image_001",
        image_bytes=_png_bytes(),
    )

    image_result = judge.judge_image(
        context,
        dict(context.image_candidates[0]),
        image_input=image_input,
    )
    pair_result = judge.judge_pair(
        context,
        dict(context.text_candidates[0]),
        dict(context.image_candidates[0]),
        image_input=image_input,
    )

    assert image_result.score.raw_score == 61.0
    assert pair_result.score.raw_score == 73.0
    assert client.prompts == []
    assert len(client.multimodal_calls) == 2
    assert all(
        str(call["image_data_url"]).startswith("data:image/png;base64,")
        for call in client.multimodal_calls
    )
    assert "Use the attached PNG pixels" in str(client.multimodal_calls[0]["prompt"])
    assert "Use the attached PNG pixels" in str(client.multimodal_calls[1]["prompt"])
    assert IMAGE_PATH_SENTINEL not in repr(client.multimodal_calls)


def test_gpt4omini_ranking_judge_retries_once_for_contract_repair() -> None:
    client = FakeJudgeClient(
        [
            _response(
                score=0,
                confidence=0,
                hard_reject=True,
                hard_rejection_codes=["unsupported_plot_details"],
            ),
            _response(score=72, hard_reject=False, hard_rejection_codes=[]),
        ]
    )
    judge = GPT4OMiniRankingJudge(client=client, model_name="gpt4omini-test")

    result = judge.judge_text(_build_context(), dict(_build_context().text_candidates[0]))

    assert result.score.raw_score == 72.0
    assert result.score.normalized_score == 0.72
    assert result.score.status == "scored"
    assert result.score.model_name == "gpt4omini-test"
    assert result.score.details == {
        "contract_repair_retries": 1,
        "contract_repair_error_type": "GPT4OMiniJudgeParseError",
        "contract_repair_error": "unknown hard rejection code: unsupported_plot_details",
        "contract_repair_previous_response_truncated": False,
    }
    assert len(client.prompts) == 2
    assert "Repair the previous judgment JSON" in client.prompts[1]
    assert "unknown hard rejection code" in client.prompts[1]
    assert "unsupported_plot_details" in client.prompts[1]
    assert "Use only these hard_rejection_codes values" in client.prompts[1]
    assert "Use at most one short evidence string" in client.prompts[1]
    assert "Do not preserve long malformed text" in client.prompts[1]
    assert "Return exactly this object shape" in client.prompts[1]


def test_gpt4omini_ranking_judge_truncates_previous_response_for_contract_repair() -> None:
    long_response = "x" * (MAX_REPAIR_RESPONSE_CHARS + 100)
    client = FakeJudgeClient([long_response, _response(score=72)])
    judge = GPT4OMiniRankingJudge(client=client)

    result = judge.judge_text(_build_context(), dict(_build_context().text_candidates[0]))

    assert result.score.details["contract_repair_previous_response_truncated"] is True
    assert "...[truncated previous response for contract repair]..." in client.prompts[1]
    assert len(client.prompts[1]) < MAX_REPAIR_RESPONSE_CHARS + 2000


def test_gpt4omini_ranking_judge_locally_salvages_truncated_judgment_json() -> None:
    truncated_response = (
        '{\n'
        '  "score": 60,\n'
        '  "confidence": 0.2,\n'
        '  "hard_reject": false,\n'
        '  "hard_rejection_codes": [],\n'
        '  "soft_violations": [],\n'
        '  "evidence": ["The candidate is grounded. The candidate is grounded.'
    )
    client = FakeJudgeClient([truncated_response])
    judge = GPT4OMiniRankingJudge(client=client, model_name="gpt4omini-test")

    result = judge.judge_pair(
        _build_context(),
        dict(_build_context().text_candidates[0]),
        dict(_build_context().image_candidates[0]),
    )

    assert result.score.raw_score == 60.0
    assert result.score.normalized_score == 0.6
    assert result.gate.decision == "pass"
    assert result.gate.judge_confidence == 0.2
    assert result.gate.evidence == ()
    assert result.gate.reason == "Partial malformed judge response salvaged."
    assert result.score.details == {
        "contract_local_salvage": True,
        "contract_local_salvage_error_type": "GPT4OMiniJudgeParseError",
        "contract_local_salvage_error": "malformed JSON judge response",
    }
    assert len(client.prompts) == 1


def test_gpt4omini_parser_rejects_malformed_json() -> None:
    with pytest.raises(GPT4OMiniJudgeParseError, match="malformed JSON"):
        parse_text_judge_response("{not json")


def test_gpt4omini_ranking_judge_accepts_fenced_json_object_without_repair() -> None:
    client = FakeJudgeClient(
        [
            "Here is the judgment:\n```json\n" + _response(score=76) + "\n```",
        ]
    )

    result = GPT4OMiniRankingJudge(client=client).judge_text(
        _build_context(),
        dict(_build_context().text_candidates[0]),
    )

    assert result.score.raw_score == 76.0
    assert result.score.normalized_score == 0.76
    assert len(client.prompts) == 1


def test_gpt4omini_parser_accepts_fenced_json_object() -> None:
    response = "Here is the judgment:\n```json\n" + _response(score=76) + "\n```"

    result = parse_text_judge_response(response)

    assert result.score.raw_score == 76.0


def test_gpt4omini_parser_accepts_prose_wrapped_json_object() -> None:
    response = "Judgment follows:\n" + _response(score=69) + "\nDone."

    result = parse_text_judge_response(response)

    assert result.score.raw_score == 69.0


def test_gpt4omini_parser_chooses_judgment_after_repeated_input_json() -> None:
    repeated_input = json.dumps(
        {
            "task": "stage4_pair_ranking_judge",
            "response_schema": {"score": "number from 0 to 100"},
        }
    )
    response = (
        "Input JSON:\n"
        + repeated_input
        + "\nJudgment JSON:\n"
        + _response(score=88)
    )

    result = parse_pair_judge_response(response)

    assert result.score.raw_score == 88.0


def test_gpt4omini_parser_rejects_extra_fields() -> None:
    payload = json.loads(_response())
    payload["extra"] = "not allowed"

    with pytest.raises(GPT4OMiniJudgeParseError, match="unexpected field"):
        parse_text_judge_response(json.dumps(payload))


def test_gpt4omini_parser_rejects_missing_required_fields() -> None:
    payload = json.loads(_response())
    del payload["confidence"]

    with pytest.raises(GPT4OMiniJudgeParseError, match="missing required field"):
        parse_image_judge_response(json.dumps(payload))


def test_gpt4omini_parser_rejects_unknown_hard_rejection_codes() -> None:
    payload = json.loads(_response(hard_reject=True, hard_rejection_codes=["mystery"]))

    with pytest.raises(GPT4OMiniJudgeParseError, match="unknown hard rejection code"):
        parse_pair_judge_response(json.dumps(payload))


def test_gpt4omini_parser_rejects_hard_reject_without_hard_codes() -> None:
    payload = json.loads(_response(hard_reject=True, hard_rejection_codes=[]))

    with pytest.raises(GPT4OMiniJudgeParseError, match="hard_reject requires"):
        parse_pair_judge_response(json.dumps(payload))


def test_gpt4omini_parser_rejects_hard_codes_without_hard_reject() -> None:
    payload = json.loads(
        _response(hard_reject=False, hard_rejection_codes=["forbidden_content"])
    )

    with pytest.raises(GPT4OMiniJudgeParseError, match="hard_rejection_codes require"):
        parse_pair_judge_response(json.dumps(payload))


def test_gpt4omini_repair_failure_exposes_observability_details() -> None:
    client = FakeJudgeClient(["not json", "{broken"])
    judge = GPT4OMiniRankingJudge(client=client)

    with pytest.raises(GPT4OMiniJudgeParseError) as exc_info:
        judge.judge_text(_build_context(), dict(_build_context().text_candidates[0]))

    assert exc_info.value.details == {
        "contract_repair_retries": 1,
        "contract_repair_error_type": "GPT4OMiniJudgeParseError",
        "contract_repair_error": "malformed JSON judge response",
        "contract_repair_previous_response_truncated": False,
        "contract_repair_failed": True,
        "contract_repair_failure_error_type": "GPT4OMiniJudgeParseError",
        "contract_repair_failure_error": "malformed JSON judge response",
    }


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
def test_gpt4omini_parser_rejects_non_finite_scores(bad_score: float) -> None:
    payload = json.loads(_response())
    payload["score"] = bad_score

    with pytest.raises(GPT4OMiniJudgeParseError, match="finite"):
        parse_text_judge_response(json.dumps(payload))


@pytest.mark.parametrize("bad_score", [-0.01, 100.01])
def test_gpt4omini_parser_rejects_scores_outside_zero_to_one_hundred(
    bad_score: float,
) -> None:
    payload = json.loads(_response())
    payload["score"] = bad_score

    with pytest.raises(GPT4OMiniJudgeParseError, match="score must be between 0 and 100"):
        parse_text_judge_response(json.dumps(payload))


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_gpt4omini_parser_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    payload = json.loads(_response())
    payload["confidence"] = confidence

    with pytest.raises(GPT4OMiniJudgeParseError, match="confidence"):
        parse_text_judge_response(json.dumps(payload))


@pytest.mark.parametrize("field", ["evidence", "soft_violations"])
@pytest.mark.parametrize("bad_value", ["not an array", [7]], ids=["non-array", "non-string"])
def test_gpt4omini_parser_rejects_evidence_and_soft_violation_shape(
    field: str,
    bad_value: object,
) -> None:
    payload = json.loads(_response())
    payload[field] = bad_value

    with pytest.raises(GPT4OMiniJudgeParseError, match=field):
        parse_text_judge_response(json.dumps(payload))


def _response(
    *,
    score: float = 80,
    confidence: float = 0.7,
    hard_reject: bool = False,
    hard_rejection_codes: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "score": score,
            "confidence": confidence,
            "hard_reject": hard_reject,
            "hard_rejection_codes": hard_rejection_codes or [],
            "soft_violations": [],
            "evidence": ["clear match"],
            "reason": "ok",
        }
    )


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
        b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _build_context():
    text_candidate = {
        "candidate_id": "text_001",
        "target_modality": "text",
        "prompt_or_instruction": INSTRUCTION_SENTINEL,
        "ground_truth": GROUND_TRUTH_SENTINEL,
        "reviewer_consensus": CONSENSUS_SENTINEL,
        "offline_evaluation": {"notes": GROUND_TRUTH_SENTINEL},
        "generated_content": {
            "type": "text",
            "value": "Example Film is a generated caption.",
        },
    }
    image_candidate = {
        "candidate_id": "image_001",
        "target_modality": "image",
        "prompt_or_instruction": INSTRUCTION_SENTINEL,
        "ground_truth": GROUND_TRUTH_SENTINEL,
        "reviewer_consensus": CONSENSUS_SENTINEL,
        "offline_evaluation": {"notes": GROUND_TRUTH_SENTINEL},
        "generated_content": {
            "type": "image",
            "image_artifact_id": "image_artifact_001",
        },
    }
    return build_ranking_context(
        entity_key="Q42",
        entity_label="Example Film",
        missing_mode="both",
        existing_modalities={"text": None, "image": None},
        knowledge={
            "entity_profile": {
                "entity_key": "Q42",
                "entity_label": "Example Film",
                "entity_type": "film",
                "confidence": "high",
            },
            "relation_facts": [],
        },
        reasoning={
            "supported_facts": [],
            "safe_inferences": [],
            "forbidden_content": [],
            "modality_constraints": {
                "text": {"must_include": [{"constraint": "film", "confidence": "high"}]},
                "image": {"must_include": [{"constraint": "poster", "confidence": "high"}]},
            },
        },
        generation={
            "relation_alternation_generation": {
                "text_candidates": [text_candidate],
                "image_candidates": [image_candidate],
                "model_calls": [{"response": GROUND_TRUTH_SENTINEL}],
            }
        },
        generated_image_inputs={},
    )
