from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mmkgc.data import EntityRecord
from mmkgc.generation_prompts import PROMPT_VERSION, QWEN_PROMPT_VERSION
from mmkgc.stages.modeling import (
    KnowledgeResponseValidationError, apply_knowledge_response, build_knowledge_input, knowledge_response_errors,
)
from mmkgc.stages.semantic import StrictJsonStagePlanner


def fixture(mode='image'):
    record = EntityRecord(
        entity_id=1, entity_key='Q1', label='Example Player', description='SECRET_TARGET_TEXT',
        neighbors=[], existing_modalities={'text': 'Example Player is an athlete.' if mode == 'image' else None, 'image': None},
        ground_truth_modalities={'text': 'SECRET_TARGET_TEXT', 'image': 'SECRET_TARGET_IMAGE'},
    )
    coarse = {'selected_relations': [
        {'relation': 'member of', 'direction': 'out', 'neighbor_entities': ['Team A', 'Team B'],
         'evidence_triples': ['(Example Player, member of, Team A)', '(Example Player, member of, Team B)']},
        {'relation': 'awarded to', 'direction': 'in', 'neighbor_entities': ['Prize A', 'Prize B'],
         'evidence_triples': ['(Prize A, awarded to, Example Player)', '(Prize B, awarded to, Example Player)']},
    ]}
    inputs = build_knowledge_input(record, mode, coarse)
    response = {
        'entity_profile': {'primary_type': 'person', 'core_identity': 'Example Player',
                           'secondary_types': [], 'semantic_domains': [], 'evidence_refs': []},
        'text_visual_profile': {}, 'key_attributes': [], 'uncertain_aspects': [],
        'relation_facts': [{'fact_id': fact['fact_id'], 'claim': f"{fact['subject']} {fact['relation']} {fact['object']}"}
                           for fact in inputs['fact_candidates']],
    }
    return record, coarse, inputs, response


def invalid(response, problem):
    result = deepcopy(response)
    if problem == 'missing':
        result['relation_facts'] = result['relation_facts'][:1]
    elif problem == 'duplicate':
        result['relation_facts'][1] = deepcopy(result['relation_facts'][0])
    elif problem == 'unknown':
        result['relation_facts'][1]['fact_id'] = 'rf:unknown'
    elif problem == 'empty_claim':
        result['relation_facts'][1]['claim'] = '  '
    elif problem == 'evidence':
        result['entity_profile']['evidence_refs'] = ['made-up-evidence']
    return result


def feedback(prompt):
    return json.loads(prompt.split('VALIDATION_FEEDBACK_JSON:\n', 1)[1].split('\nResponse JSON shape:', 1)[0])


def test_validator_collects_all_id_and_reference_errors_without_mutation():
    _, _, inputs, response = fixture()
    response['relation_facts'] = [response['relation_facts'][0], deepcopy(response['relation_facts'][0]),
                                  {'fact_id': 'invented', 'claim': ''}, {'fact_id': ['not', 'an', 'id'], 'claim': 'x'}]
    response['entity_profile']['evidence_refs'] = ['invented-reference']
    response['key_attributes'] = [{'value': 'invented', 'source_ref': []}]
    original = deepcopy((inputs, response))
    errors = knowledge_response_errors(inputs, response)
    assert {'missing_fact_ids', 'duplicate_fact_id', 'unknown_fact_id', 'empty_claim',
            'unknown_profile_evidence', 'unknown_attribute_evidence'} <= {error['code'] for error in errors}
    missing = next(error for error in errors if error['code'] == 'missing_fact_ids')
    assert missing['missing_fact_ids'] == ['rf:0002', 'rf:0003', 'rf:0004']
    assert missing['expected_count'] == 4
    assert all(error['path'].startswith('/') for error in errors)
    with pytest.raises(KnowledgeResponseValidationError):
        apply_knowledge_response(inputs, response)
    assert (inputs, response) == original


@pytest.mark.parametrize('mode', ['text', 'image'])
@pytest.mark.parametrize('profile', [PROMPT_VERSION, QWEN_PROMPT_VERSION])
@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'unknown', 'empty_claim', 'evidence'])
def test_business_errors_request_complete_model_repair_with_explicit_ids(mode, profile, problem):
    record, coarse, inputs, valid = fixture(mode)
    original = deepcopy((inputs, coarse))
    prompts = []

    def complete(prompt):
        assert 'SECRET_TARGET_TEXT' not in prompt and 'SECRET_TARGET_IMAGE' not in prompt
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps(invalid(valid, problem))
        value = feedback(prompt)
        assert value['expected_fact_count'] == 4
        assert value['expected_fact_ids'] == ['rf:0001', 'rf:0002', 'rf:0003', 'rf:0004']
        assert len(value['required_fact_rows']) == 4
        assert value['required_fact_rows'][2]['subject'] == 'Prize A'
        assert value['required_fact_rows'][2]['object'] == 'Example Player'
        assert value['validation_errors']
        template, _ = json.JSONDecoder().raw_decode(prompt.split('Response JSON shape:\n', 1)[1])
        assert [fact['fact_id'] for fact in template['relation_facts']] == value['expected_fact_ids']
        assert all(fact['claim'] == '' for fact in template['relation_facts'])
        assert 'PREVIOUS_RESPONSE:' not in prompt
        assert 'Never put bare strings in key_attributes' in prompt
        return json.dumps(valid)

    planner = StrictJsonStagePlanner(complete, prompt_profile=profile)
    result = planner.build_knowledge(record, mode, coarse)
    assert len(prompts) == 2
    assert [fact['fact_id'] for fact in result['relation_facts']] == ['rf:0001', 'rf:0002', 'rf:0003', 'rf:0004']
    assert result['relation_facts'][2]['subject'] == 'Prize A'
    assert result['relation_facts'][2]['object'] == 'Example Player'
    assert planner.calls[-1]['status'] == 'applied'
    assert planner.calls[-1]['contract_repair_retries'] == 1
    assert len(planner.calls[-1]['validation_attempts']) == 2
    assert planner.calls[-1]['validation_errors'] == []
    assert planner.calls[-1]['validation_attempts'][0]['errors']
    assert (inputs, coarse) == original


def test_repeated_invalid_response_stops_even_after_whitespace_and_key_reordering():
    record, coarse, _, valid = fixture()
    value = invalid(valid, 'missing')
    responses = [json.dumps(value), json.dumps(value, sort_keys=True, indent=2)]
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return responses[len(prompts) - 1]

    planner = StrictJsonStagePlanner(complete)
    with pytest.raises(KnowledgeResponseValidationError):
        planner.build_knowledge(record, 'image', coarse)
    assert len(prompts) == 2
    assert planner.calls[-1]['contract_repair_stopped_reason'] == 'repeated_invalid_response'


def test_two_different_business_repairs_are_bounded_and_audited():
    record, coarse, _, valid = fixture()
    responses = [invalid(valid, 'missing'), invalid(valid, 'duplicate'), valid]
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return json.dumps(responses[len(prompts) - 1])

    planner = StrictJsonStagePlanner(complete)
    result = planner.build_knowledge(record, 'image', coarse)
    assert len(result['relation_facts']) == 4
    assert planner.calls[-1]['contract_repair_retries'] == 2
    assert len(planner.calls[-1]['validation_attempts']) == 3


def test_unresolved_contract_fails_after_two_repairs_without_filling_missing_facts():
    record, coarse, _, valid = fixture()
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        response = deepcopy(valid)
        response['relation_facts'].pop(len(prompts) - 1)
        return json.dumps(response)

    planner = StrictJsonStagePlanner(complete)
    with pytest.raises(KnowledgeResponseValidationError) as caught:
        planner.build_knowledge(record, 'image', coarse)
    assert len(prompts) == 3
    assert caught.value.model_call is planner.calls[-1]
    assert planner.calls[-1]['contract_repair_stopped_reason'] == 'repair_limit_reached'
    assert planner.calls[-1]['validation_errors'][0]['missing_fact_ids'] == ['rf:0003']


def test_syntax_and_business_errors_share_the_same_repair_budget():
    record, coarse, _, valid = fixture()
    responses = ['broken JSON', json.dumps(invalid(valid, 'missing')), json.dumps(valid)]
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return responses[len(prompts) - 1]

    planner = StrictJsonStagePlanner(complete)
    result = planner.build_knowledge(record, 'image', coarse)
    assert len(result['relation_facts']) == 4
    assert planner.calls[-1]['contract_repair_retries'] == 2
    assert planner.calls[-1]['validation_attempts'][0]['errors'][0]['code'] == 'invalid_json'


def test_repeated_invalid_json_after_original_syntax_repair_gets_one_fresh_source_rebuild():
    record, coarse, _, valid = fixture()
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        if len(prompts) < 3:
            return 'broken JSON'
        assert 'ORIGINAL_INPUT_JSON:' in prompt
        assert 'PREVIOUS_RESPONSE:' not in prompt
        assert feedback(prompt)['validation_errors'][0]['code'] == 'invalid_json'
        return json.dumps(valid)

    planner = StrictJsonStagePlanner(complete, prompt_profile=QWEN_PROMPT_VERSION)
    result = planner.build_knowledge(record, 'image', coarse)
    assert len(result['relation_facts']) == 4
    assert len(prompts) == 3
    assert 'PREVIOUS_RESPONSE:' in prompts[1]
    assert planner.calls[-1]['contract_repair_retries'] == 2


@pytest.mark.parametrize('on_repair', [False, True])
def test_transport_timeout_propagates_without_transport_retries(on_repair):
    record, coarse, _, valid = fixture()
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        if on_repair and len(prompts) == 1:
            return json.dumps(invalid(valid, 'missing'))
        raise TimeoutError('forced timeout')

    planner = StrictJsonStagePlanner(complete)
    with pytest.raises(TimeoutError) as caught:
        planner.build_knowledge(record, 'image', coarse)
    assert len(prompts) == (2 if on_repair else 1)
    assert planner.calls[-1]['contract_repair_retries'] == int(on_repair)
    assert caught.value.model_call['error_type'] == 'TimeoutError'


def test_no_repair_configuration_keeps_strict_failure():
    record, coarse, _, valid = fixture()
    planner = StrictJsonStagePlanner(lambda _: json.dumps(invalid(valid, 'missing')), max_knowledge_repairs=0)
    with pytest.raises(KnowledgeResponseValidationError):
        planner.build_knowledge(record, 'image', coarse)
    assert planner.calls[-1]['contract_repair_retries'] == 0


@pytest.mark.parametrize('value', [-1, True, 1.2, '2'])
def test_invalid_repair_budget_fails_at_construction(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        StrictJsonStagePlanner(lambda _: '{}', max_knowledge_repairs=value)


def test_valid_claims_do_not_override_immutable_source_fields():
    _, _, inputs, valid = fixture()
    valid['relation_facts'][0].update(subject='forged subject', object='forged object', evidence_refs=['forged'])
    result = apply_knowledge_response(inputs, valid)
    assert result['relation_facts'][0]['subject'] == 'Example Player'
    assert result['relation_facts'][0]['object'] == 'Team A'
    assert 'forged' not in result['relation_facts'][0]['evidence_refs']
