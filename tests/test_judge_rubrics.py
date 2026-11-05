"""Evidence boundaries and compatibility of the production rubric judge."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from jsonschema import validate
import pytest

from mmkgc.io.writer import _compact_stage4_candidate
from mmkgc.observation import bind_observation, repair_operations
from mmkgc.ranking import factory, judge_rubrics
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.contracts import ScoreResult
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.image_input import ImageInput
from mmkgc.ranking.profiles import GPT4OMiniJudgeConfig, LocalScorerConfig, Stage4Profile
from mmkgc.ranking.scorers.gpt4omini_judge import (
    GPT4OMiniJudgeParseError, GPT4OMiniRankingJudge, _rubric_response_schema,
    build_image_judge_prompt, parse_image_judge_response, parse_text_judge_response,
)
from test_fact_observation import setup, raw_observation, png


def fixture(mode="text", *, values=(4, 4, 4, 4), status="supported"):
    record, _, _, _, _, targets = setup(mode)
    prefix = "text:" if mode == "text" else "pixel:"
    payload = {
        "scores": {name: {"score": value, "reason": "Based on supplied evidence.",
                    "evidence": [prefix + " a red tower with a roof clock"]}
                   for name, value in zip(judge_rubrics.DIMENSIONS[mode], values)},
        "observation": raw_observation(targets, mode=mode, status=status),
        "unverified_additions": [], "repair_suggestions": [],
    }
    context = SimpleNamespace(entity_key=record.entity_key, entity_label=record.label,
                              to_prompt_payload=lambda: {"legal_context": "red tower; roof clock"})
    candidate = {"candidate_id": mode + "_001", "verification_targets": targets,
                 "generated_content": {"type": mode, "value": "A red tower with a roof clock."}}
    return context, candidate, payload


class Client:
    def __init__(self, *payloads):
        self.responses = iter(payloads)
        self.calls = []

    def complete_structured(self, prompt, schema):
        self.calls.append((prompt, schema, None))
        return json.dumps(next(self.responses))

    def complete_multimodal_structured(self, prompt, image, schema):
        self.calls.append((prompt, schema, image))
        return json.dumps(next(self.responses))


def test_rubric_score_and_trace_survive_actual_adapter_and_writer():
    context, candidate, payload = fixture(values=(3, 4, 4, 4))
    client = Client(payload)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(context, candidate)
    validate(payload, client.calls[0][1])
    assert result.score.normalized_score == .75
    assert any(check["status"] == "supported" for check in result.observation["fact_checks"])
    details = {key: result.score.details[key] for key in (
        "rubric_version", "rubric_scores", "rubric_aggregation", "unverified_additions", "repair_suggestions")}
    compact = _compact_stage4_candidate({**candidate, "judge_rubrics": details}, rank=1, fallback_modality="text")
    assert compact["judge_rubrics"] == details


@pytest.mark.parametrize("bad", [True, 1.5, -1, 5, "4"])
def test_invalid_score_is_rejected(bad):
    _, _, payload = fixture()
    payload["scores"]["entity_semantics"]["score"] = bad
    with pytest.raises(GPT4OMiniJudgeParseError):
        parse_text_judge_response(json.dumps(payload))


@pytest.mark.parametrize("change", ["missing_dimension", "extra_overall", "source_only_evidence"])
def test_incomplete_or_unsupported_numeric_judgment_is_rejected(change):
    _, _, payload = fixture()
    if change == "missing_dimension":
        del payload["scores"]["semantic_coverage"]
    elif change == "extra_overall":
        payload["score"] = 99
    else:
        payload["scores"]["entity_semantics"]["evidence"] = ["context:rf:0001"]
    with pytest.raises(GPT4OMiniJudgeParseError):
        parse_text_judge_response(json.dumps(payload))


def test_unknown_addition_is_recorded_without_becoming_a_delete_operation():
    context, candidate, payload = fixture()
    payload["unverified_additions"] = [{"claim": "The tower is wooden.",
        "evidence": ["text: wooden tower"], "reason": "Material is not established by supplied evidence."}]
    result = GPT4OMiniRankingJudge(Client(payload), use_rubrics=True).judge_text(context, candidate)
    assert result.score.normalized_score == 1
    assert result.score.details["unverified_additions"]
    assert repair_operations(result.observation) == []


def test_absence_does_not_reduce_factual_consistency_and_null_stays_unavailable():
    context, candidate, payload = fixture(values=(2, None, 1, 4), status="missing")
    result = GPT4OMiniRankingJudge(Client(payload), use_rubrics=True).judge_text(context, candidate)
    agg = result.score.details["rubric_aggregation"]
    assert agg["fact_assessment"] == "unavailable"
    assert "verified_fact_consistency" not in agg["caps"]
    assert result.score.normalized_score == .5
    assert result.score.details["rubric_scores"]["verified_fact_consistency"]["score"] is None


def test_empty_check_scope_remains_valid_strict_schema_without_invented_facts():
    context, candidate, payload = fixture(values=(2, None, 1, 4))
    candidate["verification_targets"] = []
    payload["observation"]["fact_checks"] = []
    client = Client(payload)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(context, candidate)
    validate(payload, client.calls[0][1])
    item_schema = client.calls[0][1]["properties"]["observation"]["properties"]["fact_checks"]["items"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"] == {}
    assert result.observation["fact_checks"] == []
    assert result.score.details["rubric_aggregation"]["fact_assessment"] == "unavailable"


def test_conflicting_observation_cannot_keep_a_perfect_factual_score():
    context, candidate, invalid = fixture()
    invalid["observation"]["fact_checks"][1]["status"] = "contradicted"
    fixed = deepcopy(invalid)
    fixed["scores"]["verified_fact_consistency"]["score"] = 2
    client = Client(invalid, fixed)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(context, candidate)
    assert len(client.calls) == 2
    assert "RUBRIC_VALIDATION_ERROR" in client.calls[1][0]
    assert result.score.normalized_score == .5
    assert [op["action"] for op in repair_operations(result.observation)] == ["replace"]


def test_confirmed_extra_conflict_needs_candidate_and_supplied_source_evidence():
    context, candidate, payload = fixture(values=(4, 2, 4, 4))
    refs = candidate["verification_targets"][1]["evidence_refs"]
    payload["observation"]["unsupported_claims"] = [{"claim": "It is blue.",
        "evidence": ["text: blue tower", "context:" + refs[0]], "confidence": .95,
        "reason": "The supplied color fact establishes red."}]
    validate(payload, _rubric_response_schema(candidate["verification_targets"], "text"))
    result = GPT4OMiniRankingJudge(Client(payload), use_rubrics=True).judge_text(context, candidate)
    assert repair_operations(result.observation)[0]["action"] == "remove"
    payload["observation"]["unsupported_claims"][0]["evidence"] = ["text: blue tower"]
    with pytest.raises(GPT4OMiniJudgeParseError):
        GPT4OMiniRankingJudge(Client(payload, payload), use_rubrics=True).judge_text(context, candidate)


def test_unknown_claim_cannot_also_be_executable_removal():
    _, candidate, payload = fixture(values=(4, 2, 4, 4))
    claim = "The tower is wooden."
    payload["unverified_additions"] = [{"claim": claim, "evidence": ["text: wooden"], "reason": "Unknown."}]
    payload["observation"]["unsupported_claims"] = [{"claim": claim, "evidence": ["text: wooden",
        "context:" + candidate["verification_targets"][1]["evidence_refs"][0]], "confidence": .95, "reason": "Conflict."}]
    parsed = judge_rubrics.parse_payload(payload, "text")
    bound = bind_observation(payload["observation"], candidate["verification_targets"], "text", "text_001")
    with pytest.raises(ValueError, match="unverified addition"):
        judge_rubrics.validate_bound_observation(parsed, bound, candidate["verification_targets"])


def test_rubric_mode_does_not_accept_legacy_overall_scores():
    context, candidate, _ = fixture()
    legacy = {"score": 95, "confidence": .99, "hard_reject": False, "hard_rejection_codes": [],
              "soft_violations": [], "evidence": ["text: a tower"], "reason": "Good."}
    assert parse_text_judge_response(json.dumps(legacy)).score.normalized_score == .95
    client = Client(legacy, legacy)
    with pytest.raises(GPT4OMiniJudgeParseError):
        GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(context, candidate)
    assert len(client.calls) == 2


def test_image_uses_pixels_not_generated_prompt_and_keeps_pixels_on_contract_retry():
    context, candidate, invalid = fixture("image")
    candidate["generated_content"]["value"] = "PRIVATE_GENERATION_CLAIM"
    candidate["positive_prompt"] = "PRIVATE_GENERATION_PROMPT"
    invalid["observation"]["fact_checks"][1]["status"] = "contradicted"
    fixed = deepcopy(invalid)
    fixed["scores"]["verified_fact_consistency"]["score"] = 1
    image = ImageInput.from_bytes(entity_key=context.entity_key, candidate_id=candidate["candidate_id"], image_bytes=png())
    prompt = build_image_judge_prompt(context, candidate, image_input=image, use_rubrics=True)
    assert "PRIVATE_GENERATION" not in prompt
    client = Client(invalid, fixed)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).judge_image(context, candidate, image_input=image)
    assert result.score.normalized_score == .25
    assert len(client.calls) == 2
    assert all(call[2] == image.as_data_url() for call in client.calls)


def test_missing_pixels_never_receives_a_numeric_visual_judgment():
    context, candidate, payload = fixture("image")
    client = Client(payload)
    with pytest.raises(GPT4OMiniJudgeParseError, match="actual image pixels"):
        GPT4OMiniRankingJudge(client, use_rubrics=True).judge_image(context, candidate)
    assert not client.calls
    payload["scores"]["visual_usability"]["score"] = None
    with pytest.raises(GPT4OMiniJudgeParseError, match="unavailable"):
        parse_image_judge_response(json.dumps(payload))


@pytest.mark.parametrize("mode", ["text", "image"])
def test_similarity_and_polish_cannot_cancel_verified_error(mode):
    _, _, payload = fixture(mode, values=(4, 1, 4, 4))
    details = judge_rubrics.parse_payload(payload, mode)
    item = {"hard_gate": {"decision": "pass"}, "ranking_reason": "",
            "model_scores": {key: 1.0 for key in ("sentence_bert", "structured_kg_alignment", "clip", "blip_itm", "gpt4omini_judge")},
            "judge_rubrics": details}
    engine = object.__new__(ModelFusionRankingEngine)
    engine._finalize_single_item(mode, item)
    assert item["final_score"] <= .25


def test_factory_enables_new_contract_on_production_path(monkeypatch):
    monkeypatch.setattr("mmkgc.adapters.qwen.load_qwen_config", lambda path: SimpleNamespace(max_token=4096, model_name="test"))
    off = LocalScorerConfig(False, False, None, None)
    profile = Stage4Profile("test", off, off, off, off, GPT4OMiniJudgeConfig(True, Path("unused"), 30))
    scorers = factory._build_scorers(profile, project_root=Path.cwd())
    assert scorers["gpt4omini_judge"].use_rubrics is True


@pytest.mark.parametrize("mode", ["text", "image"])
def test_full_engine_retains_rubrics_and_old_reflection_input(mode):
    context, candidate, payload = fixture(mode, values=(4, 1, 4, 4))
    payload["observation"]["fact_checks"][1]["status"] = "contradicted"
    record, _, knowledge, _, reasoning, _ = setup(mode)
    from mmkgc.stages.plan_context import ranking_knowledge
    from mmkgc.pipeline import _needs_stage4_regeneration
    context = build_ranking_context(record.entity_key, record.label, mode, {},
        ranking_knowledge(knowledge, reasoning), reasoning,
        {"relation_alternation_generation": {mode + "_candidates": [candidate]}},
        {"image_001": {"image_bytes": png()}} if mode == "image" else {})

    class GraphClient(Client):
        def complete(self, prompt):
            assert "Extract factual triples" in prompt
            return '{"triples": []}'

    class Local:
        def score_candidates(self, reference, texts):
            return [ScoreResult(1, 1, "scored", "test-local") for _ in texts]

        def score_image_quality_batch(self, image, texts):
            return [ScoreResult(1, 1, "scored", "test-local") for _ in texts]

    on = LocalScorerConfig(True, True, Path("unused"), "cpu")
    off = LocalScorerConfig(False, False, None, None)
    profile = Stage4Profile("test", on, off, on, on, GPT4OMiniJudgeConfig(True, Path("unused"), 30))
    engine = ModelFusionRankingEngine(profile=profile, scorers={
        "gpt4omini_judge": GPT4OMiniRankingJudge(GraphClient(payload), use_rubrics=True),
        "sentence_bert": Local(), "clip": Local(), "blip_itm": Local()})
    result = engine.rank(context)
    assert result["ranking_status"] == "completed"
    item = result[mode + "_ranking"]["ranked_candidates"][0]
    assert item["judge_rubrics"]["rubric_scores"]["verified_fact_consistency"]["score"] == 1
    assert item["final_score"] <= .25
    assert item["observation"]["violated_facts"]
    assert _needs_stage4_regeneration(result)
