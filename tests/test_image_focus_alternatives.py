"""Allow a new supported emphasis without forcing a new camera or style."""
from copy import deepcopy
import json

import pytest

from mmkgc.adapters.qwen import QwenLocalImagePromptGenerator
from mmkgc.generation_prompts import build_image_focus_prompt
from mmkgc.stages.generation import relation_alternation_generation
from mmkgc.stages.presentation import (
    ImageFocusValidationError, duplicate_image_prompt,
    validate_focus_realization, validate_image_focus,
)
from test_image_prompt_sequential import config, context, focus, install_replies
from test_shared_fact_selection import Images, selected, target


FIRST = (
    'Museum red brick facade, tactile masonry texture emphasized by soft daylight, '
    'arched entrance visible, architectural photography with natural color and restrained surroundings.'
)
SECOND = (
    'Museum arched entrance shape emphasized as the main architectural detail, '
    'red brick facade retained as context, frontal centered architectural photography '
    'in the same soft daylight.'
)
FIRST_DESCRIPTION = (
    'The red brick texture is the main focus of this frontal photograph of Museum. '
    'An arched entrance remains visible within the facade. The building is centered '
    'in soft daylight, with a restrained background.'
)
SECOND_DESCRIPTION = (
    'The arched entrance shape is the main focus of this frontal photograph of Museum. '
    'Its supported outline is emphasized while the red brick facade remains as context. '
    'The building stays centered in the same soft daylight and photographic style.'
)


def entrance_plan():
    plan = focus()
    plan.update(visual_focus='arched entrance shape', supporting_fact_ids=['rf:3'],
                difference_from_previous='Emphasize the supported entrance instead of brick texture')
    return plan


def test_new_visual_emphasis_accepts_identical_rendering():
    previous, plan = focus(), entrance_plan()
    before = deepcopy((previous, plan))
    audit = validate_image_focus(plan, context()['source_facts'], [previous])
    assert audit['changes_from_previous'][0]['focus_changed']
    assert audit['changes_from_previous'][0]['changed_axes'] == {}
    assert (previous, plan) == before


def test_new_focus_may_emphasize_a_feature_already_mentioned_in_the_old_prompt():
    plan = entrance_plan()
    audit = validate_image_focus(plan, context()['source_facts'], [focus()])
    assert 'arched entrance' in FIRST
    validate_focus_realization(SECOND_DESCRIPTION, plan, audit, [FIRST])
    validate_focus_realization(SECOND, plan, audit, [FIRST])


def test_focus_only_change_must_survive_description_and_compression():
    plan = entrance_plan()
    audit = validate_image_focus(plan, context()['source_facts'], [focus()])
    with pytest.raises(ImageFocusValidationError, match='omits'):
        validate_focus_realization(
            'Museum red brick masonry dominates the facade in a frontal photograph under daylight.',
            plan, audit, [FIRST])
    with pytest.raises(ImageFocusValidationError, match='repeats'):
        validate_focus_realization(FIRST, plan, audit, [FIRST])


def test_same_focus_with_a_concrete_rendering_change_is_still_allowed():
    plan = focus()
    plan['rendering']['viewpoint'] = 'oblique'
    audit = validate_image_focus(plan, context()['source_facts'], [focus()])
    assert not audit['changes_from_previous'][0]['focus_changed']
    validate_focus_realization(
        'Museum red brick facade and arched entrance, oblique architectural photograph.',
        plan, audit, [FIRST])


def test_focus_only_change_must_differ_from_every_accepted_plan():
    plan = entrance_plan()
    with pytest.raises(ImageFocusValidationError):
        validate_image_focus(plan, context()['source_facts'], [focus(), deepcopy(plan)])


@pytest.mark.parametrize('axis,value', [
    ('subject_placement', 'centrally positioned subject'),
    ('framing', 'facade view, not a close-up'),
])
def test_paraphrase_or_negated_shot_does_not_create_a_rendering_change(axis, value):
    plan = focus()
    plan['rendering'][axis] = value
    with pytest.raises(ImageFocusValidationError):
        validate_image_focus(plan, context()['source_facts'], [focus()])


@pytest.mark.parametrize('copy', [
    'Museum, red brick facade and arched entrance, centrally positioned subject, medium framing.',
    'Museum, red brick facade and arched entrance, centered subject, medium framing. Not a close-up.',
])
def test_common_position_synonyms_and_negated_shots_cannot_bypass_prompt_dedup(copy):
    previous = 'Museum, red brick facade and arched entrance, centered subject, medium framing.'
    assert duplicate_image_prompt(copy, [previous]) is not None


def test_focus_prompt_offers_either_emphasis_or_rendering_without_forcing_both():
    payload = context()
    payload['presentation_context']['previous_focus_plans'] = [focus()]
    prompt = build_image_focus_prompt(payload)
    assert 'different supported visual emphasis OR a meaningful rendering change' in prompt
    assert 'keep all five rendering values unchanged' in prompt
    assert 'at least two useful rendering dimensions' not in prompt
    assert 'attribute focus and vary at least one' not in prompt


def test_focus_only_candidate_runs_end_to_end_with_shared_facts_and_no_seed_fallback(monkeypatch):
    replies = [
        (focus(), 'stop'), ({'image_description': FIRST_DESCRIPTION}, 'stop'),
        ({'positive_prompt': FIRST, 'used_text_cues': [], 'omitted_risky_details': []}, 'stop'),
        (entrance_plan(), 'stop'), ({'image_description': SECOND_DESCRIPTION}, 'stop'),
        ({'positive_prompt': SECOND, 'used_text_cues': [], 'omitted_risky_details': []}, 'stop'),
    ]
    captured = install_replies(monkeypatch, replies)
    knowledge, selection, reasoning = selected()
    before = deepcopy((knowledge, selection, reasoning))
    result = relation_alternation_generation(
        target(), 'image', knowledge, reasoning, selection, None,
        QwenLocalImagePromptGenerator(config()), Images(), candidate_budget=2)
    stage = result['relation_alternation_generation']
    assert [candidate['positive_prompt'] for candidate in stage['image_candidates']] == [FIRST, SECOND]
    assert len(captured) == stage['image_prompt_search']['requests'] == 6
    assert stage['image_prompt_search']['unique_prompt_count'] == 2
    assert stage['image_prompt_search']['seed_variant_count'] == 0
    assert len(result['generated_image_artifacts']) == 2
    payloads = [json.JSONDecoder().raw_decode(
        request['messages'][-1]['content'].split('INPUT_JSON:\n')[1])[0] for request in captured]
    assert all(payload['source_facts'] == payloads[0]['source_facts'] for payload in payloads)
    assert payloads[4]['presentation_context']['previous_candidates'] == [FIRST]
    assert (knowledge, selection, reasoning) == before


def test_compression_that_loses_focus_is_repaired_without_requiring_new_rendering(monkeypatch):
    replies = [
        (entrance_plan(), 'stop'), ({'image_description': SECOND_DESCRIPTION}, 'stop'),
        ({'positive_prompt': FIRST, 'used_text_cues': [], 'omitted_risky_details': []}, 'stop'),
        ({'positive_prompt': SECOND, 'used_text_cues': [], 'omitted_risky_details': []}, 'stop'),
    ]
    captured = install_replies(monkeypatch, replies)
    inputs = context(4, [FIRST])
    inputs['presentation_context']['previous_focus_plans'] = [focus()]
    generator = QwenLocalImagePromptGenerator(config())
    result = generator.generate_with_context('Museum', 'museum', 'facade', '', [], context=inputs)
    assert result['positive_prompt'] == SECOND
    assert len(captured) == 4
    call = generator.consume_last_call('Q1', 'image_002')
    assert 'repeats' in call['validation_attempts'][0]['error']
