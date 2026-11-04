"""Prefer distinct prompts, then fill image slots with audited seed variants."""
from copy import deepcopy
import json

import pytest

from mmkgc.generation_prompts import build_generation_prompt, generation_payload
from mmkgc.stages.generation import relation_alternation_generation
from mmkgc.stages.presentation import duplicate_image_prompt, ImageFocusValidationError
from test_shared_fact_selection import selected, target, Images


FIRST = 'Museum, red brick facade and arched entrance, centered subject, medium framing.'
SECOND = 'Museum, red brick facade and arched entrance, centered subject, close framing.'
THIRD = 'Museum, red brick facade and arched entrance, off-center subject, wide framing, ample negative space.'


class Prompts:
    uses_structured_prompts = True
    max_concurrency = 4
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []
        self.last_call = None
    def generate_with_context(self, *args, context):
        self.inputs.append(deepcopy(context))
        value = next(self.replies)
        self.last_call = {'modality': 'image_prompt', 'status': 'generated', 'response_text': value}
        return {'positive_prompt': value}
    def consume_last_call(self, entity_key, candidate_id):
        result = self.last_call
        self.last_call = None
        return {**result, 'entity_key': entity_key, 'candidate_id': candidate_id}


def generate(replies, budget=3):
    knowledge, selection, reasoning = selected()
    before = deepcopy((knowledge, selection, reasoning))
    prompts = Prompts(replies)
    result = relation_alternation_generation(target(), 'image', knowledge, reasoning, selection,
        None, prompts, Images(), candidate_budget=budget)
    assert (knowledge, selection, reasoning) == before
    return prompts, result


def test_repeated_draft_is_rewritten_and_all_three_images_receive_distinct_prompts():
    prompts, result = generate([FIRST, FIRST, SECOND, THIRD])
    candidates = result['relation_alternation_generation']['image_candidates']
    assert [item['positive_prompt'] for item in candidates] == [FIRST, SECOND, THIRD]
    assert len(result['generated_image_artifacts']) == 3
    assert len(prompts.inputs) == 4
    assert all(context['source_facts'] == prompts.inputs[0]['source_facts'] for context in prompts.inputs)
    assert 'requested_rendering' not in prompts.inputs[1]['presentation_context']
    assert prompts.inputs[2]['presentation_context']['previous_prompt'] == FIRST
    assert prompts.inputs[2]['presentation_context']['rewrite_feedback']['rejected_prompts'] == [FIRST]
    calls = result['model_calls']
    drafts = [call for call in calls if call.get('modality') == 'image_prompt']
    assert [call['presentation_disposition'] for call in drafts] == ['accepted', 'duplicate_rewrite', 'accepted', 'accepted']
    assert drafts[1]['duplicate_of'] == 'image_001'
    assert candidates[1]['metadata']['presentation_rewrites'] == 1


def test_exhausted_rewrites_fill_one_seed_variant_after_later_slots_are_prepared():
    prompts, result = generate([FIRST, FIRST, FIRST, FIRST, THIRD])
    candidates = result['relation_alternation_generation']['image_candidates']
    assert [item['candidate_id'] for item in candidates] == ['image_001', 'image_002', 'image_003']
    assert [item['positive_prompt'] for item in candidates] == [FIRST, FIRST, THIRD]
    assert candidates[1]['metadata']['image_candidate_kind'] == 'seed_variant'
    assert candidates[1]['metadata']['prompt_source_candidate_id'] == 'image_001'
    assert len(result['generated_image_artifacts']) == 3
    drafts = [call for call in result['model_calls'] if call.get('modality') == 'image_prompt']
    assert drafts[3]['presentation_disposition'] == 'duplicate_skipped'
    assert prompts.inputs[-1]['presentation_context']['previous_candidates'] == [FIRST]
    search=result['relation_alternation_generation']['image_prompt_search']
    assert search['unique_prompt_count']==2 and search['seed_variant_count']==1
    assert search['prompt_shortfall']==1 and search['shortfall']==0


@pytest.mark.parametrize('copy', [
    FIRST.upper().replace(',', '，'),
    'Medium framing, arched entrance and red brick facade, Museum centered subject.',
    FIRST.replace('Museum,', 'The Museum,'),
    FIRST + ' candidate 2',
])
def test_cosmetic_changes_are_detected_as_duplicates(copy):
    assert duplicate_image_prompt(copy, [FIRST]) is not None


def test_changed_framing_is_not_rejected_merely_for_sharing_facts():
    assert duplicate_image_prompt(SECOND, [FIRST]) is None
    assert duplicate_image_prompt(THIRD, [FIRST, SECOND]) is None


def test_shortening_a_previous_prompt_is_not_a_new_visual_emphasis():
    earlier='Museum red brick facade and arched entrance, soft natural daylight, architectural photography, uncluttered surroundings.'
    shorter='Museum red brick facade and arched entrance, soft natural daylight, architectural photography.'
    assert duplicate_image_prompt(shorter,[earlier]) == (0,'shortened_copy')
    assert duplicate_image_prompt(earlier,[shorter]) == (0,'shortened_copy')


def test_wire_includes_previous_prompts_without_changing_source_scope_or_fixing_framing():
    prompts, _ = generate([FIRST, FIRST, SECOND, THIRD])
    context = prompts.inputs[1]
    prompt = build_generation_prompt('image', generation_payload('Museum', 'museum', 'facade', '', [], context))
    payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
    assert FIRST in prompt
    assert payload['presentation_context']['previous_candidates'] == [FIRST]
    assert payload['presentation_context']['previous_prompt'] == FIRST
    assert 'requested_rendering' not in payload['presentation_context']
    assert payload['source_facts'] == context['source_facts']
    retry = build_generation_prompt('image', generation_payload('Museum', 'museum', 'facade', '', [], prompts.inputs[2]))
    assert FIRST in retry
    assert 'never evidence' in retry


def test_attempt_budget_is_per_entity_and_not_multiplied_by_candidate_count():
    prompts, result = generate([FIRST] * 7)
    assert len(prompts.inputs) == 7
    assert len(result['generated_image_artifacts']) == 3
    search=result['relation_alternation_generation']['image_prompt_search']
    assert search['unique_prompt_count']==1 and search['seed_variant_count']==2
    assert search['requests']==7  # Filling image slots makes no extra LLM calls.
    assert all(context['presentation_context']['sampling_seed'] != prompts.inputs[0]['presentation_context']['sampling_seed']
               for context in prompts.inputs[1:])


def test_format_repairs_also_consume_the_shared_request_budget():
    class RepairedPrompts(Prompts):
        def generate_with_context(self, *args, context):
            value = super().generate_with_context(*args, context=context)
            self.last_call['sampling_attempts'] = [{'temperature': 0.5}] * 3
            return value
    knowledge, selection, reasoning = selected()
    prompts = RepairedPrompts([FIRST, SECOND, THIRD])
    result = relation_alternation_generation(target(), 'image', knowledge, reasoning, selection,
        None, prompts, None, candidate_budget=3)
    assert len(prompts.inputs) == 3
    assert [item['presentation_context']['remaining_request_budget'] for item in prompts.inputs] == [18, 15, 12]
    summary = result['relation_alternation_generation']['image_prompt_search']
    assert summary['requests'] == 9 and summary['request_budget'] == 18
    assert summary['accepted_candidates'] == 3 and summary['shortfall'] == 0


@pytest.mark.parametrize('failed_call',[0,2])
def test_rejected_focus_is_logged_and_later_slots_only_see_accepted_prompts(failed_call):
    class RejectedPrompts(Prompts):
        def generate_with_context(self,*args,context):
            count=len(self.inputs)
            value=super().generate_with_context(*args,context=context)
            self.last_call['sampling_attempts']=[{'temperature':0.3}]*2
            if count==failed_call:
                self.last_call['status']='failed'
                raise ImageFocusValidationError('Actual output omits the planned difference')
            return value
    knowledge,selection,reasoning=selected()
    prompts=RejectedPrompts([FIRST,FIRST,SECOND,THIRD])
    result=relation_alternation_generation(target(),'image',knowledge,reasoning,selection,
        None,prompts,Images(),candidate_budget=3)
    summary=result['relation_alternation_generation']['image_prompt_search']
    assert summary['focus_rejections']==1
    assert summary['accepted_candidates']==len(result['generated_image_artifacts'])==3
    assert summary['unique_prompt_count']==2 and summary['seed_variant_count']==1
    calls=[c for c in result['model_calls'] if c.get('modality')=='image_prompt']
    assert any(c.get('presentation_disposition')=='focus_rejected' for c in calls)
    assert summary['requests']==2*len(prompts.inputs)
    assert prompts.inputs[-1]['presentation_context']['previous_candidates']==[FIRST]


def test_image_sampling_is_separate_and_seed_is_written_to_request_and_audit(monkeypatch):
    from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator, QwenLocalTextGenerator
    captured = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                'positive_prompt': FIRST, 'used_text_cues': [], 'omitted_risky_details': []})}}]}).encode()
    def send(req, timeout):
        captured.append(json.loads(req.data))
        return Response()
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', send)
    config = QwenConfig('Qwen/test', 'http://localhost/v1', 'test', temperature=0,
                        image_prompt_temperature=0.5, image_prompt_top_p=0.9)
    generator = QwenLocalImagePromptGenerator(config)
    context = {'presentation_context': {'sampling_seed': 321, 'requested_rendering': {'framing': 'medium'}}}
    generator.generate_with_context('Museum', 'museum', 'facade', '', [], context=context)
    call = generator.consume_last_call('Q0', 'image_001')
    assert captured[0]['temperature'] == 0.5 and captured[0]['top_p'] == 0.9 and captured[0]['seed'] == 321
    assert call['sampling_attempts'] == [{'temperature': 0.5, 'top_p': 0.9, 'seed': 321}]
    assert QwenLocalTextGenerator(config).config.temperature == 0


@pytest.mark.parametrize('options', [
    {'image_prompt_temperature': -0.1}, {'image_prompt_temperature': float('nan')},
    {'image_prompt_top_p': 0}, {'image_prompt_top_p': True},
])
def test_invalid_image_sampling_settings_fail_before_requests(options):
    from mmkgc.adapters.qwen import QwenConfig
    with pytest.raises(ValueError):
        QwenConfig('Qwen/test', 'http://localhost/v1', 'test', **options)
