from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import replace
import json

import pytest

from mmkgc.data import EntityRecord, MKGWDataStore
from mmkgc.generation_prompts import PROMPT_VERSION, QWEN_PROMPT_VERSION
from mmkgc.io.writer import _compact_stage0_coarse, _compact_stage1_modeling
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.modeling import apply_knowledge_response, build_knowledge_input
from mmkgc.stages.semantic import StrictJsonStagePlanner, _apply_stage0_response, build_stage_prompt


def neighbor(label, relation="depicts", **kwargs):
    return {"entity_key": label, "label": label, "description": label,
            "relation": relation, "direction": "out", **kwargs}


def record(neighbors, text=None):
    return EntityRecord(0, "Q0", "Target", "PRIVATE_TARGET_DESCRIPTION", neighbors,
                        {"text": text, "image": None, "image_summary": None},
                        {"text": "PRIVATE_TARGET_TEXT", "image": "PRIVATE_TARGET_IMAGE"})


def semantic_selection(coarse, groups=None):
    return {"selected_relations": [
        {"relation": item["relation"], "direction": item["direction"], "reason": "useful evidence"}
        for item in (coarse["high_recall_relation_candidates"] if groups is None else groups)]}


def dataset(tmp_path, triples):
    entities = sorted({value for head, _, tail in triples for value in (head, tail)})
    relations = sorted({relation for _, relation, _ in triples})
    for filename, values in (("entity2id.txt", entities), ("relation2id.txt", relations)):
        (tmp_path / filename).write_text("\n".join(f"{value} {index}" for index, value in enumerate(values)), encoding="utf-8")
    (tmp_path / "train.txt").write_text("\n".join(" ".join(row) for row in triples), encoding="utf-8")
    return MKGWDataStore(tmp_path)


def test_default_loader_keeps_edges_beyond_32_and_explicit_limit_still_works(tmp_path):
    store = dataset(tmp_path, [("Q0", "depicts", f"Q{index:03d}") for index in range(1, 46)])
    for name in ("valid.txt", "test.txt"):
        (tmp_path / name).write_text("Q0 depicts Q045\n", encoding="utf-8")
    store = MKGWDataStore(tmp_path)
    entity_id = store.entity_uri_to_id["Q0"]
    target = store.build_record(entity_id, "image")
    assert len(target.neighbors) == 45
    assert len(store.neighbors(entity_id, max_neighbors=32)) == 32
    assert store.neighbors(entity_id, max_neighbors=0) == []
    assert len(coarse_relation_filtering(target, "image")["stage0_v3"]["facts"]) == 45


def test_second_hop_sample_is_stratified_and_independent_of_file_order(tmp_path):
    triples = [("Q0", "link", "Q1"), *[("Q1", "common", f"Q{index}") for index in range(2, 15)],
               ("Q1", "rare", "Q99")]
    store = dataset(tmp_path, triples)
    original = store.neighbors(store.entity_uri_to_id["Q0"])[0]["neighbors"]
    (tmp_path / "train.txt").write_text("\n".join(" ".join(row) for row in reversed(triples)), encoding="utf-8")
    other = MKGWDataStore(tmp_path)
    reordered = other.neighbors(other.entity_uri_to_id["Q0"])[0]["neighbors"]
    assert original == reordered
    assert len(original) == 8
    assert any(item["relation"] == "rare" for item in original)


def test_late_distinctive_fact_reaches_model_without_dropping_catalogue():
    target = record([neighbor(f"A generic object {index}") for index in range(40)] +
                    [neighbor("Z red stone lighthouse tower")], text="Z red stone lighthouse tower")
    coarse = coarse_relation_filtering(target, "image")
    assert len(coarse["stage0_v3"]["facts"]) == 41
    group = coarse["high_recall_relation_candidates"][0]
    assert "Z red stone lighthouse tower" in group["prompt_neighbor_entities"]
    assert len(group["prompt_evidence_triples"]) == 3
    inputs = build_knowledge_input(target, "image", coarse)
    assert len(inputs["fact_candidates"]) == 41
    assert "Z red stone lighthouse tower" in {fact["object"] for fact in inputs["fact_candidates"]}
    sources = {fact["fact_id"]: fact for fact in coarse["stage0_v3"]["facts"]}
    evidence = {item["evidence_id"]: item for item in inputs["evidence_catalog"]}
    for fact in inputs["fact_candidates"]:
        ref = fact["stage0_fact_ids"][0]
        assert fact["object"] == sources[ref]["object"]
        assert evidence[ref]["content"] == sources[ref]["triple"]


def test_sampling_and_fact_ids_are_invariant_to_input_order():
    neighbors = [neighbor(f"object {index}", relation=f"attribute {index % 4}",
                          direction="in" if index % 2 else "out") for index in range(48)]
    first = coarse_relation_filtering(record(neighbors, "object 47"), "image")
    other = coarse_relation_filtering(record(list(reversed(neighbors)), "object 47"), "image")
    assert first == other
    assert build_knowledge_input(record(neighbors), "image", first) == build_knowledge_input(record(neighbors), "image", other)


@pytest.mark.parametrize("profile", [PROMPT_VERSION, QWEN_PROMPT_VERSION])
def test_large_catalogue_has_bounded_prompt_and_knowledge_views(profile):
    target = record([neighbor(f"object {index} of {relation}", relation=f"attribute {relation:02d}")
                     for relation in range(40) for index in range(5)])
    coarse = coarse_relation_filtering(target, "image")
    assert len(coarse["stage0_v3"]["facts"]) == 200
    prompt = build_stage_prompt(stage="stage0_coarse", record=target, missing_mode="image",
                                deterministic=coarse, prior={}, prompt_profile=profile)
    payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
    groups = payload["deterministic_output"]["high_recall_relation_candidates"]
    assert len(groups) == 24
    assert len(payload["record"]["neighbors"]) == 32
    assert all(len(group["evidence_triples"]) <= 3 for group in groups)
    assert all(len(group["neighbor_entities"]) == len(group["evidence_triples"]) for group in groups)
    assert "PRIVATE_TARGET" not in prompt
    assert "no fixed three-item limit" in prompt
    refined = _apply_stage0_response(coarse, semantic_selection(coarse))
    inputs = build_knowledge_input(target, "image", refined)
    assert len(inputs["fact_candidates"]) == 200
    assert len({fact["relation"] for fact in inputs["fact_candidates"]}) == 40
    audit = inputs["evidence_selection"]
    assert audit["catalog_fact_count"] == 200 and audit["modeled_fact_count"] == 200


def test_identity_relation_after_frequent_relations_is_considered():
    target = record([neighbor(f"common {index}", "related to") for index in range(40)] +
                    [neighbor("museum", "instance of")])
    coarse = coarse_relation_filtering(target, "image")
    assert "instance of" in {item["relation"] for item in coarse["high_recall_relation_candidates"]}


@pytest.mark.parametrize("mode,prefixes", [("image", {"X", "Y", "Z"}), ("text", {"A", "B", "C"})])
def test_representatives_follow_target_modality_instead_of_alphabetical_prefix(mode, prefixes):
    target = record([neighbor(label) for label in (
        "A author language field", "B country occupation genre", "C director creator work",
        "X red stone tower", "Y blue metal bridge", "Z green wood forest")])
    coarse = coarse_relation_filtering(target, mode)
    chosen = coarse["high_recall_relation_candidates"][0]["prompt_neighbor_entities"]
    assert {label[0] for label in chosen} == prefixes


def test_duplicate_edges_cannot_consume_representative_slots():
    target = record([neighbor("red tower")] * 10 + [neighbor("river"), neighbor("bridge")], text="red tower")
    coarse = coarse_relation_filtering(target, "image")
    assert len(coarse["stage0_v3"]["facts"]) == 3
    assert set(coarse["high_recall_relation_candidates"][0]["prompt_neighbor_entities"]) == {"red tower", "river", "bridge"}
    assert coarse["evidence_sampling"]["input_edge_count"] == 12


def test_broken_sample_reference_fails_instead_of_silently_losing_evidence():
    target = record([neighbor("museum")])
    coarse = coarse_relation_filtering(target, "image")
    original = build_knowledge_input(target, "image", coarse)
    coarse["stage0_v3"]["relations"][0]["prompt_fact_ids"] = ["fact:missing"]
    assert build_knowledge_input(target, "image", coarse) == original
    from mmkgc.stages.modeling import build_legacy_knowledge_input
    with pytest.raises(ValueError, match="not in its source relation"):
        build_legacy_knowledge_input(target, "image", coarse)


@pytest.mark.parametrize("profile", [PROMPT_VERSION, QWEN_PROMPT_VERSION])
def test_semantic_selection_updates_all_audit_views_without_changing_sources(profile):
    target = record([neighbor("museum", "instance of"), neighbor("stone", "material"),
                     neighbor("Jane", "architect"), neighbor("http://example", "website")])
    coarse = coarse_relation_filtering(target, "image")
    before = deepcopy(coarse)
    chosen = next(item for item in coarse["high_recall_relation_candidates"] if item["relation"] == "architect")
    planner = StrictJsonStagePlanner(lambda _: json.dumps(semantic_selection(coarse, [chosen])), prompt_profile=profile)
    refined = planner.refine_coarse(target, "image", coarse)
    audit = _compact_stage0_coarse(refined)
    decisions = audit["decisions"]
    assert decisions["selected_relation_ids"] == ["rel:architect:out"]
    assert set(decisions["selected_relation_ids"]).isdisjoint(decisions["discarded_relation_ids"])
    assert set(decisions["selected_relation_ids"] + decisions["discarded_relation_ids"]) == {
        item["relation_id"] for item in audit["relations"]}
    for relation in audit["relations"]:
        assert (relation["decision"] == "selected") == (relation["relation"] == "architect")
    assert audit["facts"] == before["stage0_v3"]["facts"]
    assert audit["paths"] == before["stage0_v3"]["paths"]
    assert decisions["rule_baseline_selected_relation_ids"] == before["stage0_v3"]["decisions"]["selected_relation_ids"]
    assert refined["rule_baseline_selected_relations"] == before["selected_relations"]
    assert coarse == before


def test_unshown_relations_are_distinguished_from_model_rejections():
    coarse = coarse_relation_filtering(record([neighbor("object", f"attribute {index:02d}") for index in range(30)]), "text")
    refined = _apply_stage0_response(coarse, {"selected_relations": []})
    reasons = Counter(item["decision_reason"] for item in refined["stage0_v3"]["relations"])
    assert reasons == {"not_selected_by_semantic_stage0": 24, "outside_stage0_prompt_budget": 6}


def test_no_candidate_path_preserves_metadata_catalogue_and_syncs_empty_selection():
    coarse = coarse_relation_filtering(record([neighbor("external identifier", "identifier")]), "text")
    planner = StrictJsonStagePlanner(lambda _: pytest.fail("no model request expected"))
    refined = planner.refine_coarse(record([]), "text", coarse)
    assert refined["stage0_v3"]["decisions"]["selected_relation_ids"] == []
    assert len(refined["stage0_v3"]["facts"]) == 1
    assert refined["stage0_v3"]["relations"][0]["decision_reason"] == "low_value_metadata"


def test_hidden_target_descriptions_do_not_influence_sampling_or_enter_prompts():
    target = record([neighbor("Target", entity_key="Q0", description="PRIVATE_TARGET_SELF"),
                     neighbor("stone tower"), neighbor("museum")])
    coarse = coarse_relation_filtering(target, "text")
    target = replace(target, description="DIFFERENT_PRIVATE_TARGET")
    target.ground_truth_modalities["text"] = "DIFFERENT_PRIVATE_TARGET"
    target.existing_modalities["text"] = "HIDDEN_TARGET_MODALITY"
    assert coarse_relation_filtering(target, "text") == coarse
    prompt = build_stage_prompt(stage="stage0_coarse", record=target, missing_mode="text",
                                deterministic=coarse, prior={})
    # The public record must also mask a target modality in manually built records.
    assert "PRIVATE_TARGET_SELF" not in prompt
    assert "HIDDEN_TARGET_MODALITY" not in prompt


def test_stage1_artifact_keeps_actual_sample_provenance():
    target = record([neighbor(f"object {index}") for index in range(10)])
    inputs = build_knowledge_input(target, "image", coarse_relation_filtering(target, "image"))
    knowledge = apply_knowledge_response(inputs, {
        "entity_profile": {"primary_type": "entity", "core_identity": "Target", "evidence_refs": []},
        "extracted_triples": [], "uncertain_aspects": [],
    })
    assert _compact_stage1_modeling(knowledge)["evidence_selection"] == inputs["evidence_selection"]
