"""Image admission precedes extraction; entity features remain sample-scoped."""
from copy import deepcopy
import json

import pytest

from test_image_knowledge import fixture, transport, text_and_image
from test_stage1_multi_image import scene, original_ref, description_ref
from mmkgc.adapters.image_knowledge import (IMAGE_REF, DESCRIPTION_REF,
    parse_image_assessment, with_image_assessment)
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response, KnowledgeResponseValidationError
from mmkgc.stages.triples import model_input_view
from mmkgc.stages.candidate_planning import planning_facts
from mmkgc.stages.semantic import Stage1CompletionError, KNOWLEDGE_REQUEST_CONTEXT
from mmkgc.stages.visual_evidence import ENTITY_VISUAL_POLICY
from mmkgc.stages.generation import _model_generation_context
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.plan_context import ranking_knowledge
from mmkgc.ranking.context import build_ranking_context
from mmkgc.stages.candidate_planning import apply_fact_selection


def assessment(status="compatible"):
    return {"status": status, "visible_subject": "two geometric shapes",
        "description": "A blue circle and red rectangle.",
        "reason": "Visible shapes agree with the target scene.",
        "matching_cues": ["geometric shapes"] if status == "compatible" else [],
        "conflicting_cues": ["a drawing rather than the target animal"] if status == "conflicting" else [],
        "uncertainties": ["Exact identity is not verified from this picture."]}


def feature(obj="blue", kind="entity_feature", refs=None, **kwargs):
    return {"subject": "Scene", "relation": "visual feature", "object": obj,
        "evidence_refs": refs or [IMAGE_REF, DESCRIPTION_REF],
        "visual_support": {"basis": "direct", "scope": "target", "feature_kind": kind,
            "claim_scope": "depicted_instance", "uncertainty": "", "uncertainty_scope": "none", **kwargs}}


def response(rows):
    return {"entity_profile": {"primary_type": "entity", "core_identity": "Scene", "evidence_refs": []},
            "extracted_triples": rows, "uncertain_aspects": []}


def pixel_reply(ref=IMAGE_REF, status="supported"):
    return json.dumps({"checks": [{"observation_id": "mf:0001:visual:0", "status": status,
        "observed": "The blue shape is clearly visible on the left.",
        "reason": "The claimed local property is visible in these pixels.", "evidence_refs": [ref]}]})


def planner(root, model="vision-test"):
    return QwenLocalSemanticStagePlanner(QwenConfig(model, "http://localhost/v1", "TEST_SECRET"), dataset_root=root)


def run(client, target):
    return client.build_knowledge(target, "text", coarse_relation_filtering(target, "text", catalogue_only=True))


def test_admitted_features_are_abstract_scoped_and_not_confidence_weighted(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    rows = [feature(), feature("faces right", "depiction_detail"),
            feature("dirt ground", "context", scope="context"),
            feature("possibly A or B", basis="inferred", uncertainty="Unresolved alternative", uncertainty_scope="feature")]
    requests = transport(monkeypatch, [json.dumps(assessment()), "A blue circle and red rectangle.",
                                      json.dumps(response(rows)), pixel_reply()])
    client = planner(tmp_path, "Qwen/Qwen2-VL-7B-Instruct")
    result = run(client, target)
    assert len(requests) == 4
    assert all("PRIVATE_" not in text_and_image(request)[0] for request in requests)
    assert len({text_and_image(request)[1] for request in requests}) == 1
    assert "TARGET_CONTEXT" not in text_and_image(requests[1])[0]
    assert "geometric shapes" not in text_and_image(requests[2])[0]  # gate reasoning is not a source
    assert requests[0]["guided_json"]["properties"]["status"]["enum"] == [
        "compatible", "unrelated", "conflicting", "unresolved"]
    assert result["image_modeling"]["entity_match"]["identity_verified"] is False
    assert result["stage1_v3"]["visual_fact_policy"]["schema_version"] == ENTITY_VISUAL_POLICY
    assert "min_direct_confidence" not in result["stage1_v3"]["visual_fact_policy"]
    eligible = planning_facts(result)
    assert [item["object"] for item in eligible] == ["blue"]
    assert eligible[0]["source_weight"] == 1
    assert len(result["relation_facts"]) == 4  # excluded details kept for audit
    assert any("identity" in gap["reason"] for gap in result["uncertainties"])
    selection, reasoning = compile_plan_context(result, apply_fact_selection(result,
        {"status": "ready", "selected_fact_ids": [eligible[0]["fact_id"]]}, "text", 8), "text")
    context = _model_generation_context(result, selection["fine_grained_relation_selection"]["candidates"][0],
                                        reasoning, "text", "", "")
    assert context["source_evidence"][0]["image_admission"]["identity_verified"] is False
    assert context["source_facts"][0]["visual_observations"][0]["claim_scope"] == "depicted_instance"
    assert context["source_facts"][0]["visual_source_checks"][0]["status"] == "supported"
    ranking = build_ranking_context("TEST_SCENE", "Scene", "text", result["existing_modalities"],
        ranking_knowledge(result, reasoning), reasoning,
        {"relation_alternation_generation": {"text_candidates": []}}, {})
    assert ranking.to_prompt_payload()["supported_facts"][0]["visual_source_checks"][0]["status"] == "supported"


@pytest.mark.parametrize("status", ["unrelated", "conflicting", "unresolved"])
def test_rejected_or_uncertain_image_never_reaches_extractor_but_graph_and_source_survive(tmp_path, monkeypatch, status):
    target, path = scene(tmp_path, count=1)
    # Single image convention rather than indexed multi-image convention.
    requests = transport(monkeypatch, [json.dumps(assessment(status))])
    result = run(planner(tmp_path), target)
    assert len(requests) == 1
    assert [f["object"] for f in result["relation_facts"]] == ["Town"]
    assert result["image_modeling"]["entity_match"]["status"] == status
    assert {item["evidence_id"] for item in result["evidence_catalog"]} >= {IMAGE_REF, DESCRIPTION_REF}
    assert result["source_inputs"]["has_image"] is True
    assert result["image_modeling"]["original_image"]["asset"] == path[0].name
    assert any(status in item["reason"] for item in result["uncertainties"])


def test_mixed_images_only_send_compatible_originals_to_feature_extraction(tmp_path, monkeypatch):
    target, _ = scene(tmp_path, count=3)
    requests = transport(monkeypatch, [json.dumps(assessment()), "A blue circle and red rectangle.", json.dumps(assessment("conflicting")),
        json.dumps(assessment("unresolved")), json.dumps(response([
            feature(refs=[original_ref(1), description_ref(1)])])), pixel_reply(original_ref(1))])
    result = run(planner(tmp_path), target)
    assert len(requests) == 6
    extraction_prompt = requests[-2]["messages"][-1]["content"][0]["text"]
    assert original_ref(1) in extraction_prompt
    assert original_ref(2) not in extraction_prompt and original_ref(3) not in extraction_prompt
    assert len(result["image_modeling"]["images"]) == 3
    assert len([part for part in requests[-1]["messages"][-1]["content"] if part["type"] == "image_url"]) == 1
    assert {f["object"] for f in planning_facts(result)} == {"Town", "blue"}


def test_invalid_admission_repaired_once_without_switching_image(tmp_path, monkeypatch):
    target, _ = fixture(tmp_path)
    bad = assessment(); bad["matching_cues"] = []
    requests = transport(monkeypatch, [json.dumps(bad), json.dumps(assessment("unresolved"))])
    client = planner(tmp_path)
    result = run(client, target)
    assert len(requests) == 2 and client.calls[0]["contract_repair_retries"] == 1
    assert text_and_image(requests[0])[1] == text_and_image(requests[1])[1]
    assert result["relation_facts"] == []


@pytest.mark.parametrize("reply,code", [("", "empty_response"), (("{", "length"), "output_token_limit")])
def test_incomplete_admission_is_not_json_repair(tmp_path, monkeypatch, reply, code):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, [reply])
    with pytest.raises(Stage1CompletionError) as error:
        run(planner(tmp_path), target)
    assert error.value.code == code and len(requests) == 1
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


@pytest.mark.parametrize("reply,code", [("", "empty_response"), (("A partial description", "length"), "output_token_limit")])
def test_incomplete_independent_description_stops_before_extraction(tmp_path, monkeypatch, reply, code):
    target, _ = fixture(tmp_path)
    requests = transport(monkeypatch, [json.dumps(assessment()), reply])
    client = planner(tmp_path)
    with pytest.raises(Stage1CompletionError) as error:
        run(client, target)
    assert error.value.code == code and len(requests) == 2
    assert client.calls[-1]["stage"] == "stage1_image_description" and client.calls[-1]["status"] == "failed"
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_rejected_citation_cannot_support_facts_even_if_extractor_is_bypassed(tmp_path):
    target, _ = fixture(tmp_path)
    data = with_image_assessment(build_knowledge_input(target, "text", coarse_relation_filtering(target, "text")),
        assessment("conflicting"), {"evidence_id": IMAGE_REF, "asset": "source.png", "sha256": "0" * 64}, "call:1")
    result = apply_knowledge_response(data, response([feature()]))
    assert planning_facts(result) == [] and result["relation_facts"][0]["source_weight"] == 0
    # Universal claims and self-rated confidence are not accepted in the new contract.
    for change in ({"claim_scope": "entity"}, {"confidence": 0.99}):
        with pytest.raises(KnowledgeResponseValidationError):
            apply_knowledge_response(data, response([feature(**change)]))
    before = deepcopy(data)
    view = model_input_view(data)
    assert "source_assertions" not in json.dumps(view)
    assert data == before


def test_identity_uncertainty_does_not_erase_clear_sample_features(tmp_path):
    target, _ = fixture(tmp_path)
    data = with_image_assessment(build_knowledge_input(target, "text", coarse_relation_filtering(target, "text")),
        assessment(), {"evidence_id": IMAGE_REF, "asset": "source.png", "sha256": "0" * 64}, "call:1")
    knowledge = apply_knowledge_response(data, response([
        feature("curved body", uncertainty="Exact identity remains unverified", uncertainty_scope="identity"),
        feature("possible bridge", uncertainty="Component is not clear", uncertainty_scope="feature")]))
    assert [f["object"] for f in planning_facts(knowledge)] == ["curved body"]
    obs = planning_facts(knowledge)[0]["visual_observations"][0]
    assert obs["uncertainty"] == "Exact identity remains unverified"
    assert obs["claim_scope"] == "depicted_instance"
    _, reasoning = compile_plan_context(knowledge, apply_fact_selection(knowledge,
        {"status": "ready", "selected_fact_ids": [planning_facts(knowledge)[0]["fact_id"]]}, "text", 8), "text")
    context = build_ranking_context("TEST_SCENE", "Scene", "text", data["existing_modalities"],
        ranking_knowledge(knowledge, reasoning), reasoning,
        {"relation_alternation_generation": {"text_candidates": []}}, {})
    judged = context.to_prompt_payload()["supported_facts"][0]["visual_observations"][0]
    assert judged["uncertainty_scope"] == "identity" and judged["uncertainty"] == obs["uncertainty"]
    with pytest.raises(KnowledgeResponseValidationError):
        apply_knowledge_response(data, response([feature(uncertainty_scope="identity")]))
