from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mmkgc.io.writer import _stage_artifact
from mmkgc.stages.generation import _relation_sequence
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.relation_planning import (
    RelationPlanValidationError, SLOT_SCHEMA, apply_candidate_slot_plans,
    candidate_slot_response_schema, selection_prompt,
)


def knowledge(count=7):
    return {
        "entity_profile": {"entity_type": "film", "core_identity": "Example Film"},
        "stage1_v3": {"entity_profile": {"primary_type": "film", "core_identity": "Example Film", "evidence_refs": []}},
        "relation_facts": [
            {"fact_id": f"rf:{index:04d}", "relation_group_id": f"rg:{index:04d}",
             "subject": "Example Film", "relation": "cast member", "object": f"Actor {index}",
             "direction": "out", "claim": f"Example Film has cast member Actor {index}.",
             "evidence_refs": [f"fact:{index:04d}"], "evidence_triples": [f"(Example Film, cast member, Actor {index})"],
             "evidence_match_score": 0.9, "scoring_status": "scored"}
            for index in range(1, count + 1)
        ],
    }


def response(source, anchors=None, status="ready"):
    return {"plan": {
        "status": status,
        "relation_plans": [{
            "fact_id": fact["fact_id"], "role": "primary" if index == 0 else "secondary",
            "evidence_refs": list(fact["evidence_refs"]),
            "supporting_fact_ids": ["rf:0003"] if index == 0 else [],
            "reason": "Reviewed supplied evidence for this fact.",
        } for index, fact in enumerate(source["relation_facts"])],
        "candidate_anchors": anchors if anchors is not None else ["rf:0001"] * 3 + ["rf:0002"] * 2,
    }}


def validator(source, budget=5):
    cls = pytest.importorskip("jsonschema").Draft202012Validator
    schema = candidate_slot_response_schema(source, budget)
    cls.check_schema(schema)
    return cls(schema)


@pytest.mark.parametrize("count", [6, 7, 20])
def test_all_facts_are_retained_while_only_five_candidates_are_allocated(count):
    source = knowledge(count)
    value = response(source)
    original = deepcopy((source, value))
    validator(source).validate(value)
    selected = apply_candidate_slot_plans(source, value, "image", 5)["fine_grained_relation_selection"]
    assert selected["quota_summary"]["assigned_total"] == 5
    assert selected["primary_relations"][0]["candidate_quota"] == 3
    assert selected["secondary_relations"][0]["candidate_quota"] == 2
    assert all(item["candidate_quota"] == 0 for item in selected["secondary_relations"][1:])
    assert len(selected["primary_relations"]) + len(selected["secondary_relations"]) == count
    assert selected["primary_relations"][0]["supporting_fact_ids"] == ["rf:0003"]
    assert selected["allocation_method"] == SLOT_SCHEMA
    assert (source, value) == original


@pytest.mark.parametrize("budget", [1, 3, 5, 10])
@pytest.mark.parametrize("mode", ["text", "image"])
def test_budget_is_dynamic_and_repeated_anchors_are_allowed(budget, mode):
    source = knowledge()
    value = response(source, ["rf:0001"] * budget)
    validator(source, budget).validate(value)
    selected = apply_candidate_slot_plans(source, value, mode, budget)["fine_grained_relation_selection"]
    assert selected["primary_relations"][0]["candidate_quota"] == budget
    assert len(selected["candidate_anchors"]) == budget


@pytest.mark.parametrize("count", [0, 4, 6, 7])
def test_wrong_slot_count_is_rejected_without_deleting_or_adding_slots(count):
    source = knowledge()
    value = response(source, ["rf:0001"] * count)
    original = deepcopy(value)
    assert list(validator(source).iter_errors(value))
    with pytest.raises(RelationPlanValidationError) as caught:
        apply_candidate_slot_plans(source, value, "image", 5)
    error = next(item for item in caught.value.errors if item["code"] == "candidate_slot_count")
    assert (error["actual"], error["expected"]) == (count, 5)
    assert error["path"] == "/plan/candidate_anchors"
    assert value == original


@pytest.mark.parametrize("anchor", ["unknown", None, {}, ["rf:0001"], True])
def test_malformed_or_unknown_anchor_is_reported_with_allowed_ids(anchor):
    source = knowledge()
    value = response(source)
    value["plan"]["candidate_anchors"][1] = anchor
    with pytest.raises(RelationPlanValidationError) as caught:
        apply_candidate_slot_plans(source, value, "image", 5)
    error = next(item for item in caught.value.errors if item["code"] == "unknown_anchor")
    assert error["path"] == "/plan/candidate_anchors/1"
    assert error["allowed_values"] == [fact["fact_id"] for fact in source["relation_facts"]]


@pytest.mark.parametrize("role", ["context", "excluded"])
def test_slots_cannot_anchor_context_or_excluded_facts(role):
    source = knowledge()
    value = response(source)
    value["plan"]["relation_plans"][1]["role"] = role
    with pytest.raises(RelationPlanValidationError) as caught:
        apply_candidate_slot_plans(source, value, "image", 5)
    errors = [item for item in caught.value.errors if item["code"] == "invalid_anchor_role"]
    assert len(errors) == 2 and all(item["actual_role"] == role for item in errors)


def test_primary_without_a_slot_must_be_replanned_by_model():
    source = knowledge()
    value = response(source, ["rf:0002"] * 5)
    with pytest.raises(RelationPlanValidationError) as caught:
        apply_candidate_slot_plans(source, value, "image", 5)
    assert any(item["code"] == "unanchored_primary" and item["path"] == "/plan/relation_plans/0/role" for item in caught.value.errors)


@pytest.mark.parametrize("problem", ["primary", "secondary", "slots", "support"])
def test_insufficient_evidence_schema_prevents_contradictory_roles_and_allocation(problem):
    source = knowledge()
    value = response(source, [], "insufficient_evidence")
    for item in value["plan"]["relation_plans"]:
        item.update(role="context", supporting_fact_ids=[])
    validator(source).validate(value)
    assert apply_candidate_slot_plans(source, value, "image", 5)["fine_grained_relation_selection"]["quota_summary"]["assigned_total"] == 0
    if problem in {"primary", "secondary"}:
        value["plan"]["relation_plans"][0]["role"] = problem
    elif problem == "slots":
        value["plan"]["candidate_anchors"] = ["rf:0001"]
    else:
        value["plan"]["relation_plans"][0]["supporting_fact_ids"] = ["rf:0002"]
    assert list(validator(source).iter_errors(value))
    with pytest.raises(RelationPlanValidationError):
        apply_candidate_slot_plans(source, value, "image", 5)


def test_empty_fact_set_accepts_insufficient_evidence_without_candidates():
    source = knowledge(0)
    value = {"plan": {"status": "insufficient_evidence", "relation_plans": [], "candidate_anchors": []}}
    validator(source).validate(value)
    selection = apply_candidate_slot_plans(source, value, "image", 5)
    assert _relation_sequence(selection, source, {}, "image", 5) == []
    invalid = {"plan": {"status": "ready", "relation_plans": [], "candidate_anchors": ["invented"] * 5}}
    assert list(validator(source).iter_errors(invalid))


@pytest.mark.parametrize("problem", ["quota_field", "missing_fact", "duplicate_fact", "foreign_ref", "bad_support", "blank_reason"])
def test_other_fact_contracts_remain_strict_and_all_errors_use_wire_paths(problem):
    source = knowledge()
    value = response(source)
    first = value["plan"]["relation_plans"][0]
    if problem == "quota_field":
        first["candidate_quota"] = 999
    elif problem == "missing_fact":
        value["plan"]["relation_plans"].pop()
    elif problem == "duplicate_fact":
        value["plan"]["relation_plans"][-1] = deepcopy(first)
    elif problem == "foreign_ref":
        first["evidence_refs"] = ["fact:0002"]
    elif problem == "bad_support":
        first["supporting_fact_ids"] = ["rf:0001"]
    else:
        first["reason"] = " "
    with pytest.raises(RelationPlanValidationError) as caught:
        apply_candidate_slot_plans(source, value, "image", 5)
    assert all(item["path"].startswith("/plan") for item in caught.value.errors)
    if problem == "quota_field":
        assert list(validator(source).iter_errors(value))
        assert any(item["code"] == "unexpected_field" for item in caught.value.errors)


def test_slot_order_survives_compilation_generation_and_stage_artifact():
    source = knowledge()
    anchors = ["rf:0002", "rf:0001", "rf:0001", "rf:0002", "rf:0001"]
    selection = apply_candidate_slot_plans(source, response(source, anchors), "image", 5)
    selection, reasoning = compile_plan_context(source, selection, "image")
    sequence = _relation_sequence(selection, source, reasoning, "image", 5)
    assert [item["fact_id"] for item in sequence] == anchors
    artifact = _stage_artifact("fine_grained_relation_selection", selection)["data"]
    assert artifact["candidate_anchors"] == anchors
    assert artifact["allocation_method"] == SLOT_SCHEMA
    assert [item["candidate_quota"] for item in artifact["relation_plans"]] == [3, 2, 0, 0, 0, 0, 0]


def test_prompt_distinguishes_fact_review_from_candidate_allocation():
    system, payload = selection_prompt("Example Film", "image", knowledge(), 5)
    assert "Keeping a fact does NOT require giving it a candidate" in system
    assert "Do not return candidate_quota" in system
    assert "only 5 candidate_anchors" in system
    assert len(json.loads(payload)["relation_facts"]) == 7


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "5"])
def test_invalid_budget_fails_before_request(budget):
    with pytest.raises(ValueError, match="candidate_budget"):
        candidate_slot_response_schema(knowledge(), budget)
