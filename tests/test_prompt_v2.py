from __future__ import annotations

import json
from copy import deepcopy

import pytest

from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator, QwenLocalTextGenerator
from mmkgc.data import EntityRecord
from mmkgc.generation_prompts import PROMPT_VERSION
from mmkgc.stages.generation import relation_alternation_generation, safer_relation_alternation_generation
from mmkgc.stages.modeling import apply_knowledge_response, build_knowledge_input
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.relation_planning import apply_relation_plans
from mmkgc.stages.semantic import SemanticStageParseError, StrictJsonStagePlanner


def _record(mode="image"):
    return EntityRecord(
        entity_id=1, entity_key="Q1", label="Example Person", description="SECRET_TARGET_TEXT",
        neighbors=[], existing_modalities={
            "text": "Example Person is a composer wearing a blue coat." if mode == "image" else None,
            "image": None,
        },
        ground_truth_modalities={"text": "SECRET_TARGET_TEXT", "image": "SECRET_TARGET_IMAGE"},
    )


def _coarse(count=6):
    return {"cleaned_relation_groups": [
        {"relation": f"relation_{index}", "direction": "in" if index % 2 else "out",
         "neighbor_entities": [f"Neighbor {index}"],
         "evidence_triples": [f"source triple {index}"]}
        for index in range(count)
    ]}


def test_stage0_retains_more_than_three_candidates_with_a_visible_budget():
    source = _coarse()
    captured = []

    def complete(prompt):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        captured.append(payload)
        candidates = payload["deterministic_output"]["cleaned_relation_groups"]
        return json.dumps({"selected_relations": [dict(item, reason="useful identity evidence") for item in candidates]})

    result = StrictJsonStagePlanner(complete).refine_coarse(_record(), "image", source)
    assert len(result["selected_relations"]) == 6
    assert captured[0]["deterministic_output"]["max_selected_relations"] == 6
    assert [item["direction"] for item in result["selected_relations"]] == [item["direction"] for item in source["cleaned_relation_groups"]]
    assert result["semantic_refinement"]["prompt_version"] == PROMPT_VERSION


@pytest.mark.parametrize("problem", ["unknown", "direction", "duplicate", "outside_budget"])
def test_stage0_rejects_relations_outside_the_supplied_candidate_boundary(problem):
    source = _coarse(25 if problem == "outside_budget" else 6)
    first = dict(source["cleaned_relation_groups"][0], reason="test")
    if problem == "unknown":
        first["relation"] = "invented"
    elif problem == "direction":
        first["direction"] = "in"
    elif problem == "outside_budget":
        first = dict(source["cleaned_relation_groups"][24], reason="not in prompt")
    selected = [first, first] if problem == "duplicate" else [first]
    planner = StrictJsonStagePlanner(lambda _: json.dumps({"selected_relations": selected}))
    with pytest.raises(SemanticStageParseError, match="unique existing"):
        planner.refine_coarse(_record(), "image", source)


def test_stage0_empty_input_skips_model_and_empty_selection_does_not_invent_a_relation():
    def fail(_):
        raise AssertionError("empty evidence should not call the model")

    planner = StrictJsonStagePlanner(fail)
    result = planner.refine_coarse(_record(), "image", _coarse(0))
    assert result["selected_relations"] == []
    assert result["semantic_refinement"]["status"] == "skipped_no_candidates"
    assert planner.calls == []
    result = StrictJsonStagePlanner(lambda _: '{"selected_relations":[]}').refine_coarse(_record(), "image", _coarse())
    assert result["selected_relations"] == []
    assert len(result["discarded_relations"]) == 6


def test_stage1_models_every_fact_with_source_direction_and_provenance():
    source = {"selected_relations": _coarse(8)["cleaned_relation_groups"]}

    def complete(prompt):
        assert prompt.startswith("Build the complete entity knowledge model")
        assert "SECRET_TARGET_TEXT" not in prompt and "SECRET_TARGET_IMAGE" not in prompt
        inputs = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])["deterministic_output"]
        return json.dumps({
            "entity_profile": {"primary_type": "entity", "core_identity": "Example Person",
                               "secondary_types": [], "semantic_domains": [], "evidence_refs": []},
            "text_visual_profile": {}, "key_attributes": [], "uncertain_aspects": [],
            "relation_facts": [{"fact_id": item["fact_id"],
                                "claim": f"{item['subject']} {item['relation']} {item['object']}"}
                               for item in inputs["fact_candidates"]],
        })

    result = StrictJsonStagePlanner(complete).build_knowledge(_record("text"), "text", source)
    assert len(result["relation_facts"]) == 8
    assert result["relation_facts"][1]["subject"] == "Neighbor 1"
    assert result["relation_facts"][1]["object"] == "Example Person"
    assert all(item["evidence_refs"] for item in result["relation_facts"])


def _generation_fixture(mode):
    target = _record(mode)
    source = {"selected_relations": [
        {"relation": relation, "direction": "out", "neighbor_entities": [value],
         "evidence_triples": [f"(Example Person, {relation}, {value})"]}
        for relation, value in (("birthplace", "Paris"), ("occupation", "composer"), ("award", "Context Prize"))
    ]}
    inputs = build_knowledge_input(target, mode, source)
    refs = [item["evidence_id"] for item in inputs["evidence_catalog"]]
    value = apply_knowledge_response(inputs, {
        "entity_profile": {"primary_type": "person", "core_identity": "a person",
                           "secondary_types": [], "semantic_domains": [], "evidence_refs": refs[:1]},
        "text_visual_profile": {"visual_scene_cues": ["blue coat"], "context_cues": [],
                                "non_visual_facts": ["Example Person was born in Paris."]},
        "key_attributes": ([{"value": "blue coat", "type": "visual", "source_ref": "existing:text"}]
                           if mode == "image" else []),
        "uncertain_aspects": [],
        "relation_facts": [{"fact_id": item["fact_id"],
                            "claim": f"{item['subject']} has {item['relation']}: {item['object']}"}
                           for item in inputs["fact_candidates"]],
    })
    response = {"status": "ready", "relation_plans": [
        {"fact_id": fact["fact_id"], "role": ("primary", "secondary", "context")[index],
         "candidate_quota": 1 if index == 0 else 0, "evidence_refs": fact["evidence_refs"],
         "supporting_fact_ids": ["rf:0002"] if index == 0 else [], "reason": "selected by test"}
        for index, fact in enumerate(value["relation_facts"])
    ]}
    selection, reasoning = compile_plan_context(value, apply_relation_plans(value, response, mode, 1), mode)
    return target, value, selection, reasoning


@pytest.mark.parametrize("mode", ["text", "image"])
@pytest.mark.parametrize("retry", [False, True])
def test_stage3_requests_separate_evidence_requirements_and_rendering(monkeypatch, mode, retry):
    target, value, selection, reasoning = _generation_fixture(mode)
    before = deepcopy(value)
    captured = []
    positive = "A conceptual person in a blue coat, neutral background, soft light, balanced composition"

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            content = "Example Person is a composer born in Paris." if mode == "text" else json.dumps({
                "positive_prompt": positive, "used_text_cues": ["blue coat"],
                "omitted_risky_details": ["birthplace does not establish the depicted setting"],
            })
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()

    def request(req, **kwargs):
        body = json.loads(req.data)
        assert body["model"] == "gpt-4o-mini"
        captured.append(json.loads(body["messages"][1]["content"].split("INPUT_JSON:\n", 1)[1]))
        return Response()

    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", request)
    config = QwenConfig("gpt-4o-mini", "https://gpt.example.test/v1", "test-secret")
    text = QwenLocalTextGenerator(config)
    image = QwenLocalImagePromptGenerator(config)
    args = (target, mode, value, reasoning, selection, text, image)
    if retry:
        result = safer_relation_alternation_generation(*args, candidate_budget=1, failed_ranking={},
            stage3_feedback={"repair_actions": ["remove unsupported details"], "preserve": ["entity identity"]})
    else:
        result = relation_alternation_generation(*args, candidate_budget=1)
    payload = captured[0]
    assert {fact["fact_id"] for fact in payload["source_facts"]} == {"rf:0001", "rf:0002"}
    assert "Context Prize" not in json.dumps(payload["source_facts"])
    assert "Context Prize" in json.dumps(payload["context_only"])
    assert all("Must include:" not in fact["claim"] and "remove unsupported" not in fact["claim"]
               for fact in payload["source_facts"])
    assert payload["must_include"] and payload["should_include"]
    assert payload["task"]["generation_angle"]
    assert "SECRET_TARGET_TEXT" not in json.dumps(payload) and "SECRET_TARGET_IMAGE" not in json.dumps(payload)
    assert value == before
    if retry:
        assert payload["regeneration_feedback"]["repair_actions"] == ["remove unsupported details"]
    if mode == "image":
        assert payload["identity_context"]["non_visual_facts"] == ["Example Person was born in Paris."]
        assert payload["supported_visual_details"] == [{"detail": "blue coat", "evidence_refs": ["existing:text"]}]
        assert payload["rendering_options"]["allowed_choices"]
        evidence = {item["evidence_id"]: item for item in payload["source_evidence"]}
        assert evidence["existing:text"]["content"] == target.existing_modalities["text"]
        candidate = result["relation_alternation_generation"]["image_candidates"][0]
        assert candidate["positive_prompt"] == positive
        assert "Paris" not in candidate["positive_prompt"]
    else:
        assert "non_visual_facts" not in payload["identity_context"]
    assert result["model_calls"][0]["prompt_version"] == PROMPT_VERSION
