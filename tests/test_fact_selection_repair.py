"""Selection recovery preserves model decisions and rejects contradictory outputs."""
from copy import deepcopy
import json

import pytest

from mmkgc.adapters.fact_selection_repair import build_fact_selection_repair
from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.stages.candidate_planning import (
    apply_fact_selection, fact_selection_prompt, fact_selection_schema,
)
from mmkgc.stages.relation_planning import RelationPlanValidationError
from test_candidate_fact_plan import source, target


def selection(ids=(), status='ready'):
    return {'status': status, 'selected_fact_ids': list(ids)}


def test_selection_prompt_exposes_only_retrieved_fact_ids_without_a_ready_empty_template():
    knowledge = source()
    knowledge['relevance_selection'] = {'target_modality': 'image', 'selected_fact_ids': ['rf:2', 'rf:3']}
    before = deepcopy(knowledge)
    system, prompt = fact_selection_prompt('Museum', 'image', knowledge, 8)
    payload = json.loads(prompt)
    assert payload['allowed_fact_ids'] == ['rf:2', 'rf:3']
    assert payload['output_contract']['ready'] == {'min_selected_facts': 1, 'max_selected_facts': 2}
    assert 'output_template' not in payload
    assert 'existing:text' in system and 'NOT selectable fact IDs' in system
    assert knowledge == before


@pytest.mark.parametrize('response,valid', [
    (selection(['rf:2']), True),
    (selection([], 'insufficient_evidence'), True),
    (selection(), False),
    (selection(['rf:2'], 'insufficient_evidence'), False),
    (selection(['existing:text']), False),
    (selection(['rf:2', 'rf:2']), False),
    (selection(['rf:1', 'rf:2', 'rf:3']), False),
])
def test_selection_schema_and_application_agree_on_status_ids_uniqueness_and_limit(response, valid):
    schema = fact_selection_schema(source(), 2)
    validator = pytest.importorskip('jsonschema').Draft202012Validator(schema)
    assert validator.is_valid(response) is valid
    if valid:
        apply_fact_selection(source(), response, 'image', 2)
    else:
        with pytest.raises(RelationPlanValidationError):
            apply_fact_selection(source(), response, 'image', 2)


def test_no_retrieved_facts_allows_only_insufficient_evidence():
    knowledge = source()
    knowledge['relevance_selection'] = {'target_modality': 'image', 'selected_fact_ids': []}
    schema = fact_selection_schema(knowledge, 8)
    validator = pytest.importorskip('jsonschema').Draft202012Validator(schema)
    assert validator.is_valid(selection([], 'insufficient_evidence'))
    assert not validator.is_valid(selection())
    assert not validator.is_valid(selection(['rf:1']))


@pytest.mark.parametrize('response,valid', [
    (selection(['rf:2']), True),
    (selection([], 'insufficient_evidence'), True),
    (selection(), False),
    (selection(['existing:text']), False),
    (selection(['rf:1', 'rf:2', 'rf:3']), False),
])
def test_installed_local_decoder_enforces_nonempty_ready_allowed_ids_and_positive_limit(response, valid):
    # Older LMFE does not fully enforce maxItems=0 or uniqueItems. The separate
    # schema/application test above covers the complete contract after decoding.
    parser = pytest.importorskip('lmformatenforcer').JsonSchemaParser(fact_selection_schema(source(), 2))
    accepted = True
    for character in json.dumps(response, separators=(',', ':')):
        if character not in parser.get_allowed_characters():
            accepted = False
            break
        parser = parser.add_character(character)
    assert (accepted and parser.can_end()) is valid


def test_targeted_feedback_does_not_replay_invalid_ids_or_accept_legal_ones_automatically():
    knowledge = source()
    before = deepcopy(knowledge)
    response = selection(['rf:2', 'UNTRUSTED_OUTPUT', 'rf:2'])
    errors = [{'code': 'unknown_fact', 'path': '/selected_fact_ids/1', 'message': 'use a supplied fact ID'},
              {'code': 'duplicate_fact', 'path': '/selected_fact_ids/2', 'message': 'do not repeat selected facts'}]
    repair = build_fact_selection_repair(knowledge, response, errors, 8)
    assert repair['previous_legal_choices'] == ['rf:2']
    payload = json.loads(repair['prompt'].split('SELECTION_REPAIR_JSON:\n')[1])
    assert payload['invalid_choice_positions'] == ['/selected_fact_ids/1', '/selected_fact_ids/2']
    assert payload['allowed_fact_ids'] == [fact['fact_id'] for fact in knowledge['relation_facts']]
    assert 'UNTRUSTED_OUTPUT' not in repair['prompt']
    assert 'NOT automatically accepted facts or mandatory selections' in repair['prompt']
    assert len(payload['repair_actions']) == 2
    assert knowledge == before and response['selected_fact_ids'][-1] == 'rf:2'


def test_legal_choices_survive_an_empty_repair_and_final_choices_are_still_model_authored(monkeypatch):
    responses = iter([selection(['existing:text', 'rf:2']), selection(), selection(['rf:3'])])
    sent = []
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    def complete(messages, schema):
        sent.append(deepcopy(messages))
        return json.dumps(next(responses))
    monkeypatch.setattr(planner, '_complete', complete)
    result = planner.plan(target(), 'image', source(), 3)
    assert result['fine_grained_relation_selection']['selected_fact_ids'] == ['rf:3']
    assert len(sent) == 3
    repairs = planner.calls[-1]['selection_repairs']
    assert [repair['previous_legal_choices'] for repair in repairs] == [['rf:2'], ['rf:2']]
    latest = json.loads(sent[-1][-1]['content'].split('SELECTION_REPAIR_JSON:\n')[1])
    assert latest['error_counts'] == {'selection_status': 1}
    assert 'Read the supplied facts again' in latest['repair_actions'][0]


def test_real_insufficient_evidence_is_preserved_after_reassessment(monkeypatch):
    responses = iter([selection(), selection([], 'insufficient_evidence')])
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    monkeypatch.setattr(planner, '_complete', lambda *args: json.dumps(next(responses)))
    result = planner.plan(target(), 'image', source(), 3)['fine_grained_relation_selection']
    assert result['status'] == 'insufficient_evidence' and result['selected_fact_ids'] == []
    assert planner.calls[-1]['contract_repair_retries'] == 1


def test_repeated_empty_selection_remains_a_bounded_failure_without_top_fact_fallback(monkeypatch):
    calls = []
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    def complete(*args):
        calls.append(True)
        return json.dumps(selection())
    monkeypatch.setattr(planner, '_complete', complete)
    with pytest.raises(RelationPlanValidationError):
        planner.plan(target(), 'image', source(), 3)
    assert len(calls) == 2
    assert planner.calls[-1]['contract_repair_stopped_reason'] == 'repeated_invalid_response'


def test_invalid_json_does_not_reuse_a_stale_parsed_selection(monkeypatch):
    responses = iter([json.dumps(selection(['rf:2', 'unknown'])), '{', json.dumps(selection(['rf:3']))])
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    monkeypatch.setattr(planner, '_complete', lambda *args: next(responses))
    result = planner.plan(target(), 'image', source(), 3)
    assert result['fine_grained_relation_selection']['selected_fact_ids'] == ['rf:3']
    assert planner.calls[-1]['selection_repairs'][-1]['error_counts'] == {'invalid_json': 1}
    assert planner.calls[-1]['selection_repairs'][-1]['previous_legal_choices'] == ['rf:2']
