"""Length repair preserves prompt text and never accepts a partial image prompt."""
from copy import deepcopy
import json

import pytest

from mmkgc.adapters.image_prompt_repair import compact_repair_prompt, recover_truncated_audit_tail, validate_positive_prompt
from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator
from mmkgc.generation_prompts import IMAGE_PROMPT_VERSION


POSITIVE = 'A close-up of Prem Chopra, centered against a neutral background, soft lighting.'


def truncated(prompt=POSITIVE, tail='"Prem Chopra", "a very long film list'):
    return '{"positive_prompt":' + json.dumps(prompt) + ',"used_text_cues":[' + tail


def test_repair_keeps_all_input_evidence_and_layout_without_competing_output_instructions():
    context = {'source_facts': [{'fact_id': 'f1', 'subject': 'Actor', 'relation': 'cast member',
                'object': 'Film', 'evidence_refs': ['t']}],
        'source_evidence': [{'evidence_id': 't', 'content': 'Original source biography. ' * 100}],
        'must_include': [{'constraint': 'Actor'}], 'should_include': [], 'context_only': [],
        'presentation_context': {'policy_version': 'grounded_image_diversity_v1',
            'previous_candidates': ['OLD_IMAGE_DRAFT'], 'remaining_request_budget': 2,
            'sampling_seed': 17, 'requested_rendering': {'framing': 'close framing'},
            'rewrite_feedback': {'rejected_prompts': ['REJECTED_DRAFT']}},
        'regeneration_feedback': {'repair_actions': ['preserve identity']}}
    before = deepcopy(context)
    prompt = compact_repair_prompt('Actor', 'person', 'cast member', '', [], context,
        'positive_prompt too long', prompt_version=IMAGE_PROMPT_VERSION)
    payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
    assert payload['prompt_version'] == IMAGE_PROMPT_VERSION
    assert payload['source_facts'] == context['source_facts']
    assert payload['source_evidence'] == context['source_evidence']
    assert payload['must_include'] == context['must_include']
    assert payload['presentation_context']['sampling_seed'] == 17
    assert payload['presentation_context']['requested_rendering'] == {'framing': 'close framing'}
    assert payload['regeneration_feedback'] == context['regeneration_feedback']
    assert 'OLD_IMAGE_DRAFT' not in prompt and 'REJECTED_DRAFT' in prompt
    assert 'Both audit arrays MUST be empty' in prompt
    assert '55 words and 400 characters' in prompt
    assert context == before


@pytest.mark.parametrize('tail', ['', '"one full cue"', '"one full cue",',
                                 '"one full cue", "a cut-off cue'])
def test_complete_prompt_is_recovered_verbatim_when_only_audit_list_is_cut(tail):
    value = recover_truncated_audit_tail(truncated(tail=tail))
    assert value['positive_prompt'] == POSITIVE
    assert value['used_text_cues'] == value['omitted_risky_details'] == []
    assert value['audit_recovery']['audit_complete'] is False
    assert value['audit_recovery']['prompt_text_preserved'] is True


def test_second_audit_list_can_be_cut_but_completed_first_list_is_explicitly_discarded():
    value = recover_truncated_audit_tail('{"positive_prompt":' + json.dumps(POSITIVE)
        + ',"used_text_cues":["f1"],"omitted_risky_details":["missing')
    assert value['audit_recovery']['discarded_audit_fields'] == ['used_text_cues', 'omitted_risky_details']


@pytest.mark.parametrize('suffix', ['', ',\n '])
def test_truncation_between_audit_fields_or_before_closing_object_can_be_recovered(suffix):
    text = '{"positive_prompt":' + json.dumps(POSITIVE) + ',"used_text_cues":["one cue"]' + suffix
    value = recover_truncated_audit_tail(text)
    assert value['positive_prompt'] == POSITIVE
    assert value['audit_recovery']['audit_complete'] is False


@pytest.mark.parametrize('text', [
    '{"positive_prompt":"a cut off prompt',
    truncated('word ' * 81),
    truncated('a' * 651),
    '{"used_text_cues":[],"positive_prompt":' + json.dumps(POSITIVE) + ',"omitted_risky_details":["cut',
    '{"positive_prompt":' + json.dumps(POSITIVE) + ',"unexpected_field":["cut',
    '{"positive_prompt":' + json.dumps(POSITIVE) + ',"used_text_cues":[{"nested":"cut',
    '{"positive_prompt":' + json.dumps(POSITIVE) + ',"used_text_cues":["complete" "bad syntax',
    '{"positive_prompt":' + json.dumps(POSITIVE) + ',"used_text_cues":[],"used_text_cues":["cut',
    '{"positive_prompt":' + json.dumps(POSITIVE) + ',["bad key"]:["cut',
    json.dumps({'positive_prompt': POSITIVE, 'used_text_cues': [], 'omitted_risky_details': []}),
])
def test_incomplete_prompt_bad_json_unknown_fields_and_complete_responses_are_not_salvaged(text):
    assert recover_truncated_audit_tail(text) is None


def test_adapter_requires_length_finish_and_records_raw_response_and_incomplete_audit(monkeypatch):
    generator = QwenLocalImagePromptGenerator(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    raw = truncated()
    requests = []
    def complete(prompt, **kwargs):
        requests.append(prompt)
        return {'choices': [{'finish_reason': 'length', 'message': {'content': raw}}]}, 1
    monkeypatch.setattr(generator, '_request_json', complete)
    result = generator.generate('Prem Chopra', 'actor', '', '', [])
    assert len(requests) == 1
    assert result['positive_prompt'] == POSITIVE
    assert result['audit_recovery']['audit_complete'] is False
    call = generator.consume_last_call('Q0', 'image_001')
    assert call['response_text'] == raw
    assert call['initial_finish_reason'] == 'length'
    assert call['audit_recovery']['kind'] == 'truncated_audit_tail'
    assert call['metadata']['contract_repair_retries'] == 0


def test_adapter_does_not_salvage_bad_json_with_normal_finish_or_cut_positive_prompt(monkeypatch):
    generator = QwenLocalImagePromptGenerator(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    replies = iter([
        ('stop', truncated()),
        ('length', '{"positive_prompt":"still cut off'),
        ('stop', json.dumps({'positive_prompt': POSITIVE, 'used_text_cues': [], 'omitted_risky_details': []})),
    ])
    prompts = []
    def complete(prompt, **kwargs):
        prompts.append(prompt)
        finish, raw = next(replies)
        return {'choices': [{'finish_reason': finish, 'message': {'content': raw}}]}, 1
    monkeypatch.setattr(generator, '_request_json', complete)
    result = generator.generate('Prem Chopra', 'actor', '', '', [])
    assert len(prompts) == 3
    assert result['positive_prompt'] == POSITIVE
    assert 'audit_recovery' not in result
    assert generator.consume_last_call('Q0', 'image_001')['metadata']['contract_repair_retries'] == 2


def test_response_placeholder_is_rejected_instead_of_counted_as_a_valid_candidate():
    with pytest.raises(ValueError, match='placeholder'):
        validate_positive_prompt('one short image description')
