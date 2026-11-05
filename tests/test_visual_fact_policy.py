"""Image guesses remain auditable observations, never mandatory generation facts."""
from copy import deepcopy
import json

import jsonschema
import pytest

from mmkgc.adapters.evidence_reranker import rerank_knowledge_relevance
from mmkgc.adapters.image_batches import merge_image_batches
from mmkgc.adapters.image_knowledge import with_image_description, IMAGE_REF, DESCRIPTION_REF
from mmkgc.data import EntityRecord
from mmkgc.io.writer import _stage_artifact
from mmkgc.ranking.context import build_ranking_context
from mmkgc.stages.candidate_planning import apply_fact_selection, fact_selection_prompt, planning_facts
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response, KnowledgeResponseValidationError
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.generation import _model_generation_context
from mmkgc.generation_prompts import generation_payload
from mmkgc.stages.relation_planning import RelationPlanValidationError
from mmkgc.stages.triples import triple_response_schema, build_triple_prompt, model_input_view
from mmkgc.stages.visual_evidence import VISUAL_FACT_POLICY


def inputs():
    record = EntityRecord(1, "02436645", "water deer", "PRIVATE_TARGET", [],
        {"text": None, "image": "deer.jpg"}, {"text": "PRIVATE_TARGET"})
    coarse = {"selected_relations": [{"relation": "hypernym", "direction": "out",
        "neighbor_entities": ["mouse deer"], "evidence_triples": ["(water deer, hypernym, mouse deer)"]}]}
    data = build_knowledge_input(record, "text", coarse)
    data = with_image_description(data,
        "The animal appears to be a type of rodent or marsupial.",
        {"evidence_id": IMAGE_REF, "asset": "deer.jpg", "sha256": "0" * 64,
         "media_type": "image/jpeg", "byte_count": 1}, "describe:1")
    return record, data


def row(relation="color", obj="brown", **support):
    return {"subject": "water deer", "relation": relation, "object": obj,
        "evidence_refs": [IMAGE_REF, DESCRIPTION_REF],
        "visual_support": {"basis": "direct", "scope": "target", "confidence": 0.95,
                           "uncertainty": "", **support}}


def reply(rows):
    return {"entity_profile": {"primary_type": "entity", "core_identity": "water deer",
        "evidence_refs": []}, "extracted_triples": rows, "uncertain_aspects": []}


class Relevance:
    def score_batch(self, *, documents, **kwargs):
        return [0.99 if 'rodent' in doc else 0.9 for doc in documents]


def test_graph_answers_are_withheld_from_perception_but_preserved_in_merged_catalogue():
    _, data = inputs()
    original = deepcopy(data)
    view = model_input_view(data)
    assert view["fact_candidates"] == [] and view["graph_relation_counts"] == []
    assert "mouse deer" not in json.dumps(view)
    assert "rodent or marsupial" in json.dumps(view)  # raw uncertain source is still visible
    assert all(item["source"] == "existing_modality" for item in view["evidence_catalog"])
    assert "meaningful unresolved source guesses" in build_triple_prompt(data)
    result = apply_knowledge_response(data, reply([row()]))
    assert any(f["object"] == "mouse deer" for f in result["relation_facts"])
    assert data == original
    value = reply([])
    value["entity_profile"]["evidence_refs"] = ["evidence:kg:0001"]
    with pytest.raises(KnowledgeResponseValidationError):
        apply_knowledge_response(data, value)


def test_unselected_caption_guesses_cannot_bypass_fact_selection_into_generation():
    _, data = inputs()
    before = deepcopy(data)
    knowledge = apply_knowledge_response(data, reply([row()]))
    color = next(f for f in knowledge["relation_facts"] if f["relation"] == "color")
    selection, reasoning = compile_plan_context(knowledge, apply_fact_selection(knowledge,
        {"status": "ready", "selected_fact_ids": [color["fact_id"]]}, "text", 8), "text")
    plan = selection["fine_grained_relation_selection"]["candidates"][0]
    context = _model_generation_context(knowledge, plan, reasoning, "text", "", "")
    payload = generation_payload("water deer", "entity", "color", "", [], context)
    assert "rodent or marsupial" not in json.dumps(payload)
    assert payload["source_facts"][0]["object"] == "brown"
    assert payload["source_facts"][0]["source_weight"] == 0.95
    assert DESCRIPTION_REF in {item["evidence_id"] for item in payload["source_evidence"]}
    assert "rodent or marsupial" in json.dumps(knowledge["evidence_catalog"])
    assert data == before


def test_source_alternatives_and_uncertainty_survive_without_becoming_assertive_requirements():
    _, data = inputs()
    value = reply([row("possible category", "rodent or marsupial", basis="inferred", confidence=1,
        uncertainty="appears to be rodent or marsupial; exact category is unresolved"), row()])
    jsonschema.validate(value, triple_response_schema(data))
    before = deepcopy(value)
    knowledge = apply_knowledge_response(data, value)
    assert value == before
    uncertain = next(f for f in knowledge["relation_facts"] if f["relation"] == "possible category")
    assert uncertain["source_weight"] == 0 and uncertain["generation_eligible"] is False
    assert uncertain["visual_observations"][0]["uncertainty"].startswith("appears")
    assert uncertain["visual_observations"][0]["evidence_refs"] == [IMAGE_REF, DESCRIPTION_REF]
    assert any("rodent or marsupial" in gap["aspect"] and "unresolved" in gap["reason"]
               for gap in knowledge["uncertainties"])
    ranked = rerank_knowledge_relevance(knowledge, Relevance(), "text")
    assert uncertain["fact_id"] not in ranked["relevance_selection"]["selected_fact_ids"]
    assert uncertain["fact_id"] in ranked["relevance_selection"]["ineligible_fact_ids"]
    assert uncertain["fact_id"] not in {f["fact_id"] for f in planning_facts(knowledge)}  # no reranker bypass
    with pytest.raises(RelationPlanValidationError):
        apply_fact_selection(ranked, {"status": "ready", "selected_fact_ids": [uncertain["fact_id"]]}, "text", 8)
    ids = [f["fact_id"] for f in planning_facts(ranked)]
    selection, reasoning = compile_plan_context(ranked,
        apply_fact_selection(ranked, {"status": "ready", "selected_fact_ids": ids}, "text", 8), "text")
    checks = selection["fine_grained_relation_selection"]["candidates"][0]["verification_targets"]
    assert all("rodent" not in item["claim"] for item in checks)
    assert reasoning["uncertain_facts"] == knowledge["uncertainties"]
    artifact = _stage_artifact("knowledge_modeling", ranked)["data"]
    assert artifact["visual_fact_policy"]["schema_version"] == VISUAL_FACT_POLICY
    assert artifact["triples"] == ranked["relation_facts"]


@pytest.mark.parametrize("support", [
    {"basis": "inferred", "confidence": 1, "uncertainty": "Possibly a different category"},
    {"uncertainty": "Two views conflict"}, {"confidence": 0.79}, {"scope": "background"},
])
def test_high_relevance_never_overrides_non_assertive_visual_support(support):
    _, data = inputs()
    ranked = rerank_knowledge_relevance(apply_knowledge_response(data, reply([row(**support)])), Relevance(), "text")
    visual = next(f for f in ranked["relation_facts"] if f["relation"] == "color")
    assert visual["relevance_score"] == 0.9
    assert visual["selection_score"] == 0
    assert visual["fact_id"] not in {f["fact_id"] for f in planning_facts(ranked)}


def test_context_is_downweighted_and_confidence_reaches_planner_and_judge():
    _, data = inputs()
    knowledge = apply_knowledge_response(data, reply([row(), row("scene", "vegetation", scope="context")]))
    ranked = rerank_knowledge_relevance(knowledge, Relevance(), "text")
    by_relation = {f["relation"]: f for f in ranked["relation_facts"]}
    assert by_relation["color"]["source_weight"] == 0.95
    assert by_relation["scene"]["source_weight"] == 0.475
    assert by_relation["scene"]["selection_score"] == pytest.approx(0.4275)
    _, raw = fact_selection_prompt("water deer", "text", ranked, 8)
    payload = json.loads(raw)
    assert all("source_weight" in f and "visual_observations" in f for f in payload["triples"])
    selection, reasoning = compile_plan_context(ranked, apply_fact_selection(ranked,
        {"status": "ready", "selected_fact_ids": [by_relation["color"]["fact_id"]]}, "text", 8), "text")
    context = build_ranking_context("02436645", "water deer", "text", data["existing_modalities"],
        ranking_knowledge(ranked, reasoning), reasoning, {"relation_alternation_generation": {"text_candidates": []}}, {})
    positive = context.to_prompt_payload()["supported_facts"][0]
    assert positive["source_weight"] == 0.95
    assert positive["visual_observations"][0]["basis"] == "direct"


@pytest.mark.parametrize("bad", [None, {}, {"confidence": True}, {"confidence": float("nan")},
    {"confidence": 1.1}, {"scope": "everything"}, {"basis": "inferred", "uncertainty": ""}])
def test_missing_or_malformed_visual_metadata_is_not_silently_defaulted(bad):
    _, data = inputs()
    triple = row()
    if isinstance(bad, dict) and bad:
        triple["visual_support"].update(bad)
    else:
        triple["visual_support"] = bad
    with pytest.raises(KnowledgeResponseValidationError):
        apply_knowledge_response(data, reply([triple]))


def test_weak_image_match_cannot_dilute_graph_evidence_or_gain_weight_by_repetition():
    _, data = inputs()
    guess = row("hypernym", "mouse deer", basis="inferred", uncertainty="Only a visual guess")
    batch = apply_knowledge_response(data, reply([guess, deepcopy(guess)]))
    fact = batch["relation_facts"][0]
    assert fact["source_weight"] == 1 and fact["generation_eligible"]
    assert fact["evidence_refs"] == ["evidence:kg:0001"]
    assert len(fact["visual_observations"]) == 1
    assert fact["visual_observations"][0]["uncertainty"] == "Only a visual guess"
    merged = merge_image_batches(data, [batch, deepcopy(batch)])
    assert merged["relation_facts"] == batch["relation_facts"]


def test_multi_batch_uncertainties_are_retained_and_never_sum_into_confirmation():
    _, data = inputs()
    guess = row("possible category", "rodent or marsupial", basis="inferred", confidence=0.7,
                uncertainty="Unresolved alternatives in the source")
    batch = apply_knowledge_response(data, reply([guess]))
    merged = merge_image_batches(data, [batch, deepcopy(batch)])
    fact = next(f for f in merged["relation_facts"] if f["relation"] == "possible category")
    assert not fact["generation_eligible"] and fact["source_weight"] == 0
    assert len(fact["visual_observations"]) == 1
    assert merged["uncertainties"] == batch["uncertainties"]
