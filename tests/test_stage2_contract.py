from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.relation_planner import GPT4oMiniRelationPlanner
from mmkgc.stages.relation_planning import (
    RelationPlanValidationError,
    apply_relation_plans,
    relation_plan_errors,
    selection_response_schema,
)


@pytest.fixture
def source():
    return {"relation_facts": [
        {
            "fact_id": f"rf:{index:04d}", "relation_group_id": f"rg:{index:04d}",
            "subject": "Museum", "relation": "located in", "object": f"Place {index}",
            "direction": "out", "claim": f"Museum is in Place {index}.",
            "evidence_refs": [f"fact:{index:04d}", "existing:text"],
            "evidence_triples": [f"(Museum, located in, Place {index})"],
            "evidence_match_score": 0.1, "scoring_status": "scored",
        } for index in range(1, 4)
    ]}


@pytest.fixture
def valid(source):
    return {"status": "ready", "relation_plans": [
        {
            "fact_id": fact["fact_id"], "role": ("primary", "secondary", "excluded")[index],
            "candidate_quota": 5 if index == 0 else 0,
            "evidence_refs": list(fact["evidence_refs"]),
            "supporting_fact_ids": ["rf:0002"] if index == 0 else [],
            "reason": "Reviewed the supplied evidence and target modality.",
        } for index, fact in enumerate(source["relation_facts"])
    ]}


def make_planner(**kwargs):
    return GPT4oMiniRelationPlanner(QwenConfig("gpt-4o-mini", "http://localhost/v1", "secret-key"), **kwargs)


def slot_response(response):
    plan = deepcopy(response)
    anchors = []
    for item in plan["relation_plans"]:
        anchors.extend([item["fact_id"]] * item.pop("candidate_quota"))
    return {"plan": {**plan, "candidate_anchors": anchors}}


def schema_validator(source):
    validator = pytest.importorskip("jsonschema").Draft202012Validator
    schema = selection_response_schema(source)
    validator.check_schema(schema)
    return validator(schema)


def test_schema_accepts_valid_reordered_plan(source, valid):
    valid["relation_plans"].reverse()
    schema_validator(source).validate(valid)
    assert not relation_plan_errors(source, valid, "image", 5)


@pytest.mark.parametrize("change", ["foreign_evidence", "empty_reason", "whitespace_reason", "extra_field"])
def test_schema_rejects_fact_specific_contract_errors(source, valid, change):
    first = valid["relation_plans"][0]
    if change == "foreign_evidence":
        first["evidence_refs"] = ["fact:0002"]
    elif change == "empty_reason":
        valid["relation_plans"][2]["reason"] = ""
    elif change == "whitespace_reason":
        valid["relation_plans"][2]["reason"] = " \n\t"
    else:
        first["invented"] = True
    assert list(schema_validator(source).iter_errors(valid))


def test_schema_supports_empty_evidence_and_empty_fact_set(source, valid):
    source["relation_facts"][0]["evidence_refs"] = []
    valid["relation_plans"][0]["evidence_refs"] = []
    schema_validator(source).validate(valid)
    valid["relation_plans"][0]["evidence_refs"] = ["invented"]
    assert list(schema_validator(source).iter_errors(valid))
    empty = {"relation_facts": []}
    response = {"status": "insufficient_evidence", "relation_plans": []}
    schema_validator(empty).validate(response)
    assert not relation_plan_errors(empty, response, "image", 5)


def test_collects_all_failures_with_exact_paths_and_allowed_values(source, valid):
    first, second, third = valid["relation_plans"]
    first.update(candidate_quota=4, evidence_refs=["fact:0003"], supporting_fact_ids=["rf:0003"])
    second["reason"] = " "
    third["reason"] = ""
    before = deepcopy((source, valid))
    errors = relation_plan_errors(source, valid, "image", 5)
    assert {error["code"] for error in errors} == {
        "foreign_evidence_refs", "invalid_support_target", "empty_reason", "quota_total",
    }
    evidence = next(error for error in errors if error["code"] == "foreign_evidence_refs")
    assert evidence["path"] == "/relation_plans/0/evidence_refs"
    assert evidence["allowed_values"] == source["relation_facts"][0]["evidence_refs"]
    assert evidence["owners"] == {"fact:0003": ["rf:0003"]}
    support = next(error for error in errors if error["code"] == "invalid_support_target")
    assert support["actual_role"] == "excluded"
    assert support["allowed_values"] == ["rf:0002"]
    quota = next(error for error in errors if error["code"] == "quota_total")
    assert (quota["actual"], quota["expected"]) == (4, 5)
    assert sum(error["code"] == "empty_reason" for error in errors) == 2
    with pytest.raises(RelationPlanValidationError) as caught:
        apply_relation_plans(source, valid, "image", 5)
    assert caught.value.errors == errors
    assert (source, valid) == before


@pytest.mark.parametrize("field", ["fact_id", "role", "candidate_quota", "evidence_refs", "supporting_fact_ids", "reason"])
@pytest.mark.parametrize("malformed", [None, {}, [1]])
def test_malformed_fields_return_feedback_instead_of_type_errors(source, valid, field, malformed):
    valid["relation_plans"][0][field] = malformed
    with pytest.raises(RelationPlanValidationError):
        apply_relation_plans(source, valid, "text", 5)


def test_missing_duplicate_and_self_references_are_not_silently_accepted(source, valid):
    valid["relation_plans"][0]["supporting_fact_ids"] = ["rf:0001", "rf:0002", "rf:0002", "unknown"]
    valid["relation_plans"][2] = deepcopy(valid["relation_plans"][0])
    errors = relation_plan_errors(source, valid, "image", 5)
    assert {"duplicate_fact", "missing_facts", "invalid_support_target", "duplicate_supporting_ids"} <= {error["code"] for error in errors}


def test_insufficient_evidence_keeps_zero_quota_and_no_fallback(source, valid):
    valid["status"] = "insufficient_evidence"
    for plan in valid["relation_plans"]:
        plan.update(role="context", candidate_quota=0, supporting_fact_ids=[])
    result = apply_relation_plans(source, valid, "image", 5)["fine_grained_relation_selection"]
    assert result["quota_summary"]["assigned_total"] == 0
    assert result["quota_summary"]["fallback_quota"] == 0


def test_two_distinct_repairs_use_full_conversation_and_preserve_input(monkeypatch, source, valid):
    first, second = deepcopy(valid), deepcopy(valid)
    first["relation_plans"][0].update(evidence_refs=["fact:0003"], candidate_quota=4)
    second["relation_plans"][0]["candidate_quota"] = 4
    outputs = iter(json.dumps(slot_response(response)) for response in [first, second, valid])
    captured = []
    planner = make_planner()
    before = deepcopy(source)

    def complete(messages, schema):
        captured.append(deepcopy(messages))
        return next(outputs)

    monkeypatch.setattr(planner, "_complete", complete)
    result = planner.plan(SimpleNamespace(label="Museum"), "image", source, 5)
    assert result["fine_grained_relation_selection"]["quota_summary"]["assigned_total"] == 5
    assert [len(messages) for messages in captured] == [2, 4, 6]
    assert captured[2][:4] == captured[1]
    feedback = json.loads(captured[1][-1]["content"].split("VALIDATION_ERRORS_JSON:\n")[1])
    assert {error["code"] for error in feedback["validation_errors"]} == {"foreign_evidence_refs", "candidate_slot_count"}
    assert json.loads(captured[2][1]["content"])["relation_facts"] == json.loads(captured[0][1]["content"])["relation_facts"]
    assert [len(attempt["errors"]) for attempt in planner.calls[0]["validation_attempts"]] == [2, 1, 0]
    assert planner.calls[0]["contract_repair_retries"] == 2
    assert planner.calls[0]["status"] == "applied"
    assert source == before


def test_repeated_invalid_json_stops_even_with_changed_key_order(monkeypatch, source, valid):
    valid["relation_plans"][0]["candidate_quota"] = 4
    outputs = iter([json.dumps(slot_response(valid)), json.dumps(slot_response(valid), sort_keys=True, indent=2)])
    planner = make_planner()
    monkeypatch.setattr(planner, "_complete", lambda *args: next(outputs))
    with pytest.raises(RelationPlanValidationError) as caught:
        planner.plan(SimpleNamespace(label="Museum"), "image", source, 5)
    assert caught.value.model_call["contract_repair_stopped_reason"] == "repeated_invalid_response"
    assert planner.calls[0]["contract_repair_retries"] == 1
    assert len(planner.calls[0]["validation_attempts"]) == 2


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_distinct_invalid_responses_stop_at_repair_limit(monkeypatch, source, valid, limit):
    responses = []
    for quota in (1, 2, 3):
        changed = deepcopy(valid)
        changed["relation_plans"][0]["candidate_quota"] = quota
        responses.append(json.dumps(slot_response(changed)))
    outputs = iter(responses)
    planner = make_planner(max_contract_repairs=limit)
    monkeypatch.setattr(planner, "_complete", lambda *args: next(outputs))
    with pytest.raises(RelationPlanValidationError):
        planner.plan(SimpleNamespace(label="Museum"), "image", source, 5)
    assert len(planner.calls[0]["validation_attempts"]) == limit + 1
    assert planner.calls[0]["contract_repair_stopped_reason"] == "repair_limit_reached"


@pytest.mark.parametrize("bad_json", ["{", "[]", "null", '"text"'])
def test_malformed_json_and_non_object_response_can_be_repaired(monkeypatch, source, valid, bad_json):
    outputs = iter([bad_json, json.dumps(slot_response(valid))])
    planner = make_planner()
    monkeypatch.setattr(planner, "_complete", lambda *args: next(outputs))
    planner.plan(SimpleNamespace(label="Museum"), "image", source, 5)
    assert planner.calls[0]["validation_attempts"][0]["errors"][0]["path"] == ""
    assert planner.calls[0]["status"] == "applied"


def test_http_request_sends_separate_repair_messages_and_redacts_logs(monkeypatch, source, valid):
    invalid = deepcopy(valid)
    invalid["relation_plans"][0]["reason"] = ""
    outputs = iter([invalid, valid])
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(slot_response(next(outputs)))}}]}).encode()

    def urlopen(req, **kwargs):
        captured.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr("mmkgc.adapters.relation_planner.request.urlopen", urlopen)
    planner = make_planner()
    planner.plan(SimpleNamespace(label="secret-key Museum"), "image", source, 5)
    assert [message["role"] for message in captured[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert captured[0]["response_format"] == captured[1]["response_format"]
    assert captured[1]["response_format"]["json_schema"]["strict"] is True
    assert "secret-key" not in json.dumps(planner.calls)


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), HTTPError("http://localhost", 429, "rate limited", {}, None)])
def test_transport_errors_do_not_consume_contract_repairs(monkeypatch, source, failure):
    calls = []

    def complete(*args):
        calls.append(args)
        raise failure

    planner = make_planner()
    monkeypatch.setattr(planner, "_complete", complete)
    with pytest.raises(type(failure)):
        planner.plan(SimpleNamespace(label="Museum"), "image", source, 5)
    assert len(calls) == 1
    assert not planner.calls[0]["validation_attempts"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_repair_limit_is_rejected(value):
    with pytest.raises(ValueError, match="max_contract_repairs"):
        make_planner(max_contract_repairs=value)
