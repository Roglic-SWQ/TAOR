from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmkgc.adapters.evidence_reranker import Qwen3EvidenceReranker, score_knowledge_evidence
from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.relation_planner import GPT4oMiniRelationPlanner
from mmkgc.data import EntityRecord
from mmkgc.io.writer import _stage_artifact
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.ranking.context import build_kg_reference, build_ranking_context
from mmkgc.stages.generation import _relation_sequence, relation_alternation_generation
from mmkgc.stages.modeling import apply_knowledge_response, build_knowledge_input
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.relation_planning import apply_relation_plans, selection_prompt, selection_response_schema
from mmkgc.stages.semantic import StrictJsonStagePlanner


def record() -> EntityRecord:
    return EntityRecord(
        entity_id=1, entity_key="Q1", label="Example Museum", description="SECRET_TARGET_DESCRIPTION",
        neighbors=[], existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "SECRET_TARGET_DESCRIPTION", "image": "SECRET_IMAGE.png"},
    )


def coarse() -> dict:
    return {"selected_relations": [
        {"relation": "located in", "direction": "out", "neighbor_entities": ["Example City"],
         "evidence_triples": ["(Example Museum, located in, Example City)"]},
        {"relation": "located in", "direction": "in", "neighbor_entities": ["Example Gallery"],
         "evidence_triples": ["(Example Gallery, located in, Example Museum)"]},
        {"relation": "architect", "direction": "out", "neighbor_entities": ["Example Architect"],
         "evidence_triples": ["(Example Museum, architect, Example Architect)"]},
    ]}


def model_response(inputs: dict) -> dict:
    return {
        "entity_profile": {
            "primary_type": "museum building", "secondary_types": ["cultural venue"],
            "semantic_domains": ["architecture"], "core_identity": "a museum building",
            "evidence_refs": [inputs["evidence_catalog"][0]["evidence_id"]] if inputs["evidence_catalog"] else [],
        },
        "text_visual_profile": {"visual_scene_cues": [], "context_cues": [], "non_visual_facts": []},
        "key_attributes": [], "uncertain_aspects": [{"aspect": "appearance", "reason": "not provided"}],
        "relation_facts": [
            {"fact_id": item["fact_id"], "claim": f"{item['subject']} has {item['relation']}: {item['object']}"}
            for item in inputs["fact_candidates"]
        ],
    }


class Scorer:
    def __init__(self, value=0.01):
        self.value = value
        self.calls = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


def knowledge() -> dict:
    inputs = build_knowledge_input(record(), "text", coarse())
    return score_knowledge_evidence(apply_knowledge_response(inputs, model_response(inputs)), Scorer())


def decisions(value: dict, status="ready") -> dict:
    return {"status": status, "relation_plans": [
        {
            "fact_id": fact["fact_id"],
            "role": ("primary", "secondary", "excluded")[index] if status == "ready" else "context",
            "candidate_quota": 5 if index == 0 and status == "ready" else 0,
            "evidence_refs": list(fact["evidence_refs"]),
            "supporting_fact_ids": [value["relation_facts"][1]["fact_id"]] if index == 0 and status == "ready" else [],
            "reason": "Evidence and generation context reviewed.",
        }
        for index, fact in enumerate(value["relation_facts"])
    ]}


def slot_response(response: dict) -> dict:
    plan = deepcopy(response)
    anchors = []
    for item in plan["relation_plans"]:
        anchors.extend([item["fact_id"]] * item.pop("candidate_quota"))
    return {"plan": {**plan, "candidate_anchors": anchors}}


def test_model_builds_complete_profile_and_preserves_source_direction():
    value = knowledge()
    assert value["entity_profile"]["entity_type"] == "museum building"
    assert [fact["direction"] for fact in value["relation_facts"][:2]] == ["out", "in"]
    assert value["relation_facts"][1]["subject"] == "Example Gallery"
    for fact in value["relation_facts"]:
        assert not {"allowed_use", "credibility_score", "evidence_prior_score", "generation_use", "support_level"} & fact.keys()


def test_qwen_prompt_is_a_full_build_not_a_patch_and_excludes_target_data():
    target = record()
    target.existing_modalities["text"] = target.description
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        inputs = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])["deterministic_output"]
        return json.dumps(model_response(inputs))

    result = StrictJsonStagePlanner(complete).build_knowledge(target, "text", coarse())
    assert result["relation_facts"]
    assert "SECRET_TARGET_DESCRIPTION" not in prompts[0]
    assert "SECRET_IMAGE" not in prompts[0]
    assert "no rule-generated profile to patch" in prompts[0]


@pytest.mark.parametrize("problem", ["missing", "unknown", "duplicate", "unknown_profile_ref", "unknown_attribute_ref"])
def test_qwen_rejects_incomplete_or_invented_provenance(problem):
    inputs = build_knowledge_input(record(), "image", coarse())
    response = model_response(inputs)
    if problem == "missing":
        response["relation_facts"].pop()
    elif problem == "unknown":
        response["relation_facts"][0]["fact_id"] = "invented"
    elif problem == "duplicate":
        response["relation_facts"].append(response["relation_facts"][0])
    elif problem == "unknown_profile_ref":
        response["entity_profile"]["evidence_refs"] = ["invented"]
    else:
        response["key_attributes"] = [{"value": "marble", "source_ref": "invented"}]
    with pytest.raises(ValueError):
        apply_knowledge_response(inputs, response)


def test_low_matching_score_does_not_exclude_or_cap_a_fact():
    value = knowledge()
    result = apply_relation_plans(value, decisions(value), "image", 5)["fine_grained_relation_selection"]
    assert result["primary_relations"][0]["evidence_match_score"] == 0.01
    assert result["primary_relations"][0]["candidate_quota"] == 5
    assert len(result["secondary_relations"]) == 1
    assert result["quota_summary"]["quota_key"] == "fact_id"


def test_no_evidence_remains_a_fact_with_a_null_score():
    source = {"selected_relations": [{"relation": "architect", "neighbor_entities": ["Someone"]}]}
    inputs = build_knowledge_input(record(), "text", source)
    scorer = Scorer()
    value = score_knowledge_evidence(apply_knowledge_response(inputs, model_response(inputs)), scorer)
    assert len(value["relation_facts"]) == 1
    assert value["relation_facts"][0]["scoring_status"] == "no_evidence"
    assert value["relation_facts"][0]["evidence_match_score"] is None
    assert not scorer.calls


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9"])
def test_invalid_reranker_scores_fail_instead_of_falling_back(score):
    inputs = build_knowledge_input(record(), "text", coarse())
    with pytest.raises(ValueError, match="finite score"):
        score_knowledge_evidence(apply_knowledge_response(inputs, model_response(inputs)), Scorer(score))


def test_reranker_failure_propagates():
    class Failing:
        def score(self, **kwargs):
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        score_knowledge_evidence(knowledge(), Failing())


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "quota_total", "bool_quota", "negative_quota", "context_quota", "foreign_ref", "bad_support", "wrong_status"])
def test_invalid_gpt_plans_are_rejected_without_quota_or_role_repair(problem):
    value = knowledge()
    response = decisions(value)
    first = response["relation_plans"][0]
    if problem == "missing":
        response["relation_plans"].pop()
    elif problem == "duplicate":
        response["relation_plans"].append(deepcopy(first))
    elif problem == "unknown":
        first["fact_id"] = "invented"
    elif problem == "quota_total":
        first["candidate_quota"] = 4
    elif problem == "bool_quota":
        first["candidate_quota"] = True
    elif problem == "negative_quota":
        first["candidate_quota"] = -1
    elif problem == "context_quota":
        first["role"] = "context"
    elif problem == "foreign_ref":
        first["evidence_refs"] = value["relation_facts"][2]["evidence_refs"]
    elif problem == "bad_support":
        first["supporting_fact_ids"] = [value["relation_facts"][2]["fact_id"]]
    else:
        response["status"] = "insufficient_evidence"
    original = deepcopy(response)
    with pytest.raises(ValueError):
        apply_relation_plans(value, response, "image", 5)
    assert response == original


def test_all_facts_reach_gpt_and_prompts_depend_on_entity_context():
    value = knowledge()
    value["relation_facts"] = [dict(value["relation_facts"][0], fact_id=f"rf:{index:04d}") for index in range(20)]
    system, prompt = selection_prompt("Example Museum", "image", value, 5)
    assert len(json.loads(prompt)["relation_facts"]) == 20
    assert selection_response_schema(value)["properties"]["relation_plans"]["minItems"] == 20
    assert "no score thresholds" not in prompt
    changed = deepcopy(value)
    changed["stage1_v3"]["entity_profile"]["primary_type"] = "music artist"
    assert selection_prompt("Example Musician", "image", changed, 5)[1] != prompt
    assert "there are no score thresholds" in system


def test_candidate_contexts_keep_separate_primary_facts_and_link_secondary():
    value = knowledge()
    response = decisions(value)
    response["relation_plans"][0]["candidate_quota"] = 3
    response["relation_plans"][2].update(role="primary", candidate_quota=2)
    selection, context = compile_plan_context(value, apply_relation_plans(value, response, "image", 5), "image")
    primaries = selection["fine_grained_relation_selection"]["primary_relations"]
    first = primaries[0]["generation_context"]
    other = primaries[1]["generation_context"]
    assert primaries[1]["source_fact"]["claim"] not in [item["constraint"] for item in first["must_include"]]
    assert first["should_include"] and not other["should_include"]
    assert len(_relation_sequence(selection, value, context, "image", 5)) == 5
    assert "allowed_use" not in json.dumps(context)


def test_insufficient_plan_generates_no_fallback_candidates():
    value = knowledge()
    selection, context = compile_plan_context(value, apply_relation_plans(value, decisions(value, "insufficient_evidence"), "image", 5), "image")
    assert _relation_sequence(selection, value, context, "image", 5) == []
    result = relation_alternation_generation(record(), "image", value, context, selection, None, None, candidate_budget=5)
    assert result["relation_alternation_generation"]["image_candidates"] == []


def test_ranking_uses_selected_fact_ids_and_keeps_zero_quota_secondary():
    value = knowledge()
    selection, context = compile_plan_context(value, apply_relation_plans(value, decisions(value), "text", 5), "text")
    ranking = build_ranking_context("Q1", "Example Museum", "text", {}, ranking_knowledge(value, context), context, {}, {})
    reference = build_kg_reference(ranking, "text")
    assert value["relation_facts"][0]["claim"] in reference
    assert value["relation_facts"][1]["claim"] in reference
    assert value["relation_facts"][2]["claim"] not in reference
    assert {item["fact_id"]: item["relation_role"] for item in ranking.supported_facts} == {
        "rf:0001": "primary", "rf:0002": "secondary",
    }


def test_stage_artifacts_keep_new_evidence_and_model_plan_fields():
    value = knowledge()
    selection, _ = compile_plan_context(value, apply_relation_plans(value, decisions(value), "text", 5), "text")
    stage1 = _stage_artifact("knowledge_modeling", value)["data"]
    stage2 = _stage_artifact("fine_grained_relation_selection", selection)["data"]
    assert len(stage1["relation_facts"]) == 3
    assert stage1["evidence_catalog"]
    assert stage2["status"] == "ready"
    assert stage2["relation_plans"][0]["candidate_quota"] == 5
    assert "fine_score" not in json.dumps(stage2)


def test_pipeline_builds_scores_plans_and_generates_without_rule_modeling():
    class Builder:
        calls = []

        def refine_coarse(self, target, mode, source):
            return coarse()

        def build_knowledge(self, target, mode, source):
            inputs = build_knowledge_input(target, mode, source)
            return apply_knowledge_response(inputs, model_response(inputs))

    class Planner:
        calls = []

        def plan(self, target, mode, value, budget):
            assert all(fact["scoring_status"] == "scored" for fact in value["relation_facts"])
            return apply_relation_plans(value, decisions(value), mode, budget)

    class Generator:
        def generate(self, label, entity_type, relation, angle, facts):
            return "The museum is located in Example City."

    class Ranking:
        def rank(self, context):
            assert len(context.text_candidates) == 5
            assert "Example Architect" not in build_kg_reference(context, "text")
            return {"ranking_status": "valid", "text_ranking": {"ranked_candidates": list(context.text_candidates)}}

    pipeline = MMKGCPipeline(semantic_planner=Builder(), evidence_reranker=Scorer(), relation_planner=Planner(), text_generator=Generator(), ranking_engine=Ranking())
    result = pipeline.run(record(), "text")
    assert len(result["knowledge_modeling"]["relation_facts"]) == 3
    assert "constraint_compilation" not in result["knowledge_modeling"]
    assert result["final"]["counts"]["text"] == 5


def test_gpt_http_request_uses_existing_config_and_strict_schema(monkeypatch):
    value = knowledge()
    captured = []
    config = QwenConfig("gpt-4o-mini", "http://localhost:12345/v1", "secret-test-key")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(slot_response(decisions(value)))}}]}).encode()

    def request(req, **kwargs):
        captured.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr("mmkgc.adapters.relation_planner.request.urlopen", request)
    planner = GPT4oMiniRelationPlanner(config)
    planner.plan(record(), "text", value, 5)
    assert captured[0]["model"] == "gpt-4o-mini"
    assert captured[0]["response_format"]["json_schema"]["strict"] is True
    assert planner.calls[0]["status"] == "applied"
    assert "secret-test-key" not in json.dumps(planner.calls)


def test_incomplete_reranker_download_is_rejected(tmp_path):
    for name in ("config.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "missing.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="shard"):
        Qwen3EvidenceReranker(tmp_path)


@pytest.mark.parametrize("problem", ["primary_support", "quota_total"])
def test_gpt_plan_contract_repair_uses_original_input_and_revalidates(monkeypatch, problem):
    value = knowledge()
    invalid = decisions(value)
    if problem == "primary_support":
        invalid["relation_plans"][0]["supporting_fact_ids"] = [value["relation_facts"][2]["fact_id"]]
    else:
        invalid["relation_plans"][0]["candidate_quota"] = 6
    responses = iter([json.dumps(slot_response(invalid)), json.dumps(slot_response(decisions(value)))])
    captured = []
    planner = GPT4oMiniRelationPlanner(QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key"))

    def complete(messages, schema):
        captured.append((deepcopy(messages), schema))
        return next(responses)

    monkeypatch.setattr(planner, "_complete", complete)
    result = planner.plan(record(), "image", value, 5)
    assert result["fine_grained_relation_selection"]["quota_summary"]["assigned_total"] == 5
    assert len(captured) == 2
    assert captured[1][0][:2] == captured[0][0]
    assert [message["role"] for message in captured[1][0]] == ["system", "user", "assistant", "user"]
    assert captured[1][0][2]["content"] == json.dumps(slot_response(invalid))
    assert "VALIDATION_ERRORS_JSON:" in captured[1][0][3]["content"]
    assert captured[1][1] == captured[0][1]
    assert planner.calls[0]["contract_repair_retries"] == 1
    assert planner.calls[0]["contract_repair_original_response"] == json.dumps(slot_response(invalid))
    assert planner.calls[0]["status"] == "applied"


def test_gpt_plan_contract_repair_is_bounded_and_does_not_change_invalid_plan(monkeypatch):
    value = knowledge()
    invalid = decisions(value)
    invalid["relation_plans"][0]["candidate_quota"] = 6
    calls = []
    planner = GPT4oMiniRelationPlanner(QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key"))

    def complete(*args):
        calls.append(args)
        return json.dumps(slot_response(invalid))

    monkeypatch.setattr(planner, "_complete", complete)
    with pytest.raises(ValueError, match="candidate_budget") as error:
        planner.plan(record(), "image", value, 5)
    assert len(calls) == 2
    assert invalid["relation_plans"][0]["candidate_quota"] == 6
    assert error.value.model_call["status"] == "failed"
    assert planner.calls[0]["contract_repair_retries"] == 1


def test_gpt_plan_transport_failure_does_not_trigger_contract_repair(monkeypatch):
    planner = GPT4oMiniRelationPlanner(QwenConfig("gpt-4o-mini", "http://localhost/v1", "test-key"))
    calls = []

    def complete(*args):
        calls.append(args)
        raise TimeoutError("test timeout")

    monkeypatch.setattr(planner, "_complete", complete)
    with pytest.raises(TimeoutError):
        planner.plan(record(), "image", knowledge(), 5)
    assert len(calls) == 1
    assert planner.calls[0]["status"] == "failed"


@pytest.mark.parametrize("mode", ["text", "image"])
def test_candidates_without_secondary_facts_reach_stage4(mode):
    value = knowledge()
    response = decisions(value)
    response["relation_plans"][0]["supporting_fact_ids"] = []
    response["relation_plans"][1]["role"] = "context"
    selection, reasoning = compile_plan_context(
        value, apply_relation_plans(value, response, mode, 5), mode,
    )
    generation = relation_alternation_generation(
        record(), mode, value, reasoning, selection,
        SimpleNamespace(generate=lambda *args: "The museum is in Example City."),
        SimpleNamespace(generate=lambda *args: {"positive_prompt": "A museum building."}),
        candidate_budget=5,
    )

    def ranking_context():
        return build_ranking_context(
            "Q1", "Example Museum", mode, {}, ranking_knowledge(value, reasoning),
            reasoning, generation, {},
        )

    candidates = getattr(ranking_context(), f"{mode}_candidates")
    assert len(candidates) == 5
    assert all(candidate["supporting_relations"] == () for candidate in candidates)
    assert all(candidate["expected_relation_coverage_detail"]["supporting"] == () for candidate in candidates)
    assert all(candidate["used_evidence_triples"] for candidate in candidates)
    # Empty optional support is valid, but malformed nonempty provenance still
    # rejects the whole candidate instead of silently filtering its leaves.
    raw = generation["relation_alternation_generation"][f"{mode}_candidates"]
    raw[0]["supporting_relations"] = ["located in", ""]
    raw[1]["expected_relation_coverage_detail"]["supporting"] = [42]
    assert len(getattr(ranking_context(), f"{mode}_candidates")) == 3


@pytest.mark.parametrize("continue_on_error,fail_first", [(False, True), (True, True), (True, False)])
def test_batch_cli_preserves_failures_and_optionally_tests_remaining_entities(monkeypatch, tmp_path, continue_on_error, fail_first):
    from mmkgc import run as run_module

    first, second = record(), replace(record(), entity_key="Q2")
    calls = []

    def execute(target, mode):
        calls.append(target.entity_key)
        if target is first and fail_first:
            raise ValueError("invalid test plan")
        return {"entity": {"entity_key": target.entity_key}, "final": {"counts": {"image": 5}}}

    monkeypatch.setattr(run_module, "validate_dataset_dir", lambda path: tmp_path)
    monkeypatch.setattr(run_module, "build_dataset_store", lambda *args, **kwargs: SimpleNamespace(
        select_records=lambda *args, **kwargs: [first, second],
    ))
    monkeypatch.setattr(run_module, "build_default_mainline_runtime", lambda **kwargs: SimpleNamespace(
        pipeline=SimpleNamespace(run=execute), model_name="gpt-4o-mini", base_url="http://localhost/v1",
        api_key_configured=True, image_generator_mode="sdxl-local",
    ))
    args = ["--limit", "2", "--missing-mode", "image", "--output-root", str(tmp_path), "--run-name", "batch"]
    if continue_on_error:
        args.append("--continue-on-error")
        assert run_module.main(args) == (1 if fail_first else 0)
        assert calls == ["Q1", "Q2"]
    else:
        with pytest.raises(ValueError, match="invalid test plan"):
            run_module.main(args)
        assert calls == ["Q1"]
    summary = json.loads((tmp_path / "batch/metrics_summary.json").read_text())
    assert summary["failed_entities"] == int(fail_first)
    assert summary["processed_entities"] == len(calls) - int(fail_first)
    if fail_first:
        failures = json.loads((tmp_path / "batch/failures.json").read_text())
        assert failures[0]["entity"]["entity_key"] == "Q1"
        assert failures[0]["error"] == "invalid test plan"
