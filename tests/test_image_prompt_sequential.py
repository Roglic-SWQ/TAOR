"""Same facts, previous accepted prompts, free visual emphasis, then compression."""
from copy import deepcopy
import json
import re

import pytest

from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator
from mmkgc.generation_prompts import build_generation_prompt, build_image_description_prompt, generation_payload
from mmkgc.stages.generation import relation_alternation_generation
from mmkgc.io.writer import _stage_artifact
from mmkgc.stages.presentation import (IMAGE_PRESENTATION_POLICY, ImageFocusValidationError,
    validate_image_focus, validate_focus_realization)
from test_shared_fact_selection import selected, target, Images


DESCRIPTIONS = [
    'Museum is presented through the texture of its red brick facade. The arched entrance remains visible '
    'as a distinguishing feature. A soft wash of daylight separates the brick pattern from the entrance '
    'without introducing new objects or changing the supported architectural features.',
    'Museum is presented through the rhythm of its arched entrance and the surrounding red brick facade. '
    'The entrance becomes the main point of attention, with balanced illumination emphasizing the '
    'supported shape. The background stays unobtrusive and does not imply an unseen side of the building.',
    'Museum is shown as a restrained architectural illustration, bringing the red brick facade and arched '
    'entrance together in a clear composition. Line and color organize the same supported features, '
    'giving the third image a different presentation without adding a new object or scene.',
]
PROMPTS = [
    'Museum red brick facade, tactile masonry texture emphasized by soft daylight, arched entrance visible, '
    'architectural photography with natural color and restrained surroundings.',
    'Museum arched entrance as the visual focus, surrounding red brick facade, balanced illumination '
    'emphasizing the supported arch shape, architectural photography with an unobtrusive background.',
    'Museum red brick facade and arched entrance, architectural illustration with restrained linework '
    'and warm brick tones, clear organization of the same supported features.',
]


def config():
    return QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY', image_prompt_max_token=384,
                      image_prompt_temperature=0.3, image_prompt_top_p=0.9)


def install_replies(monkeypatch, replies):
    captured = []
    iterator = iter(replies)
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            package, finish = next(iterator)
            raw = package if isinstance(package, str) else json.dumps(package)
            return json.dumps({'choices':[{'finish_reason':finish,'message':{'content':raw}}]}).encode()
    def send(req, timeout):
        captured.append(json.loads(req.data))
        return Response()
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen',send)
    return captured


def focus(index=0):
    return {'visual_focus': ['red brick texture', 'arched entrance shape', 'architectural illustration'][index],
        'supporting_fact_ids': ['rf:2', 'rf:3'], 'difference_from_previous': '' if index == 0 else 'Change the visual emphasis',
        'rendering': {'framing':'facade view','viewpoint':'frontal','subject_placement':'centered',
            'style':'architectural illustration' if index == 2 else 'architectural photography',
            'lighting':'soft daylight' if index == 0 else 'balanced illumination'}}


def context(budget=6, previous=None):
    return {'source_facts':[{'fact_id':'rf:2'}, {'fact_id':'rf:3'}],
        'presentation_context': {'policy_version':IMAGE_PRESENTATION_POLICY,
            'remaining_request_budget':budget, 'sampling_seed':15,
            'previous_candidates':previous or []}}


def packages(description=DESCRIPTIONS[0], prompt=PROMPTS[0], index=0):
    return [(focus(index),'stop'), ({'image_description':description,'used_text_cues':[],'omitted_risky_details':[]},'stop'),
            ({'positive_prompt':prompt,'used_text_cues':[],'omitted_risky_details':[]},'stop')]


def test_sequential_requests_preserve_facts_and_previous_prompts_without_a_fixed_rendering(monkeypatch):
    captured=install_replies(monkeypatch,[x for i,pair in enumerate(zip(DESCRIPTIONS,PROMPTS)) for x in packages(*pair,index=i)])
    knowledge,selection,reasoning=selected()
    before=deepcopy((knowledge,selection,reasoning))
    result=relation_alternation_generation(target(),'image',knowledge,reasoning,selection,None,
        QwenLocalImagePromptGenerator(config()),Images(),candidate_budget=3)
    assert (knowledge,selection,reasoning)==before
    assert len(captured)==9 and len(result['generated_image_artifacts'])==3
    payloads=[json.JSONDecoder().raw_decode(req['messages'][-1]['content'].split('INPUT_JSON:\n')[1])[0] for req in captured]
    for i in range(3):
        plan,draft,compressed=payloads[3*i:3*i+3]
        assert plan['generation_step']=='focus'
        assert draft['generation_step']=='describe' and compressed['generation_step']=='compress'
        assert compressed['candidate_description']==DESCRIPTIONS[i]
        assert draft['source_facts']==compressed['source_facts']==payloads[0]['source_facts']
        assert compressed['source_evidence_refs']==[e['evidence_id'] for e in draft['source_evidence']]
        assert 'source_evidence' not in compressed
        assert draft['presentation_context']['previous_candidates']==PROMPTS[:i]
        assert draft['presentation_context']['previous_focus_plans']==[focus(j) for j in range(i)]
        assert draft['presentation_context']['previous_descriptions']==DESCRIPTIONS[:i]
        assert draft['focus_plan']==compressed['focus_plan']==focus(i)
        assert draft['presentation_context']['previous_prompt']==(PROMPTS[i-1] if i else None)
        assert 'requested_rendering' not in draft['presentation_context']
        assert 'previous_renderings' not in draft['presentation_context']
        assert all(req['temperature']==0.3 and req['top_p']==0.9 for req in captured[3*i:3*i+3])
        assert 20<=len(re.findall(r"[\w'-]+",PROMPTS[i]))<=70
    candidates=result['relation_alternation_generation']['image_candidates']
    assert [c['positive_prompt'] for c in candidates]==PROMPTS
    assert [c['metadata']['image_description'] for c in candidates]==DESCRIPTIONS
    saved=_stage_artifact('relation_alternation_generation',result['relation_alternation_generation'])
    assert [c['presentation']['focus_plan'] for c in saved['data']['image_candidates']]==[focus(i) for i in range(3)]
    calls=[c for c in result['model_calls'] if c.get('modality')=='image_prompt']
    assert all(c['generation_steps']==['focus','describe','compress'] and len(c['description_calls'])==1 for c in calls)
    search=result['relation_alternation_generation']['image_prompt_search']
    assert search['requests']==9 and search['request_budget']==18


def test_first_and_next_description_choose_focus_from_evidence_and_only_accepted_history():
    payload=generation_payload('Museum','museum','facade','',[],{'presentation_context':{
        'policy_version':IMAGE_PRESENTATION_POLICY,'candidate_index':1,'previous_candidates':[]}})
    first=build_image_description_prompt(payload)
    assert 'FIRST image description' in first
    payload['presentation_context'].update(candidate_index=3,previous_candidates=PROMPTS[:1],previous_prompt=PROMPTS[0])
    second=build_image_description_prompt(payload)
    assert 'There are 1 accepted prompts' in second and 'candidate 3' in second
    assert PROMPTS[0] in second
    assert 'Each new prompt should emphasize different object attributes or scene details' in second
    assert 'NO fixed framing' in second


def test_truncated_description_is_retried_before_compression_and_counts_against_budget(monkeypatch):
    replies=[packages()[0], ('{"image_description":"cut off','length')]+packages()[1:]
    captured=install_replies(monkeypatch,replies)
    inputs=context(4)
    before=deepcopy(inputs)
    generator=QwenLocalImagePromptGenerator(config())
    result=generator.generate_with_context('Museum','museum','facade','',[],context=inputs)
    assert result['positive_prompt']==PROMPTS[0] and inputs==before
    call=generator.consume_last_call('Q1','image_001')
    assert len(captured)==len(call['sampling_attempts'])==4
    assert len(call['description_calls'])==2
    assert 'validation_error' in captured[2]['messages'][-1]['content']


def test_persistent_description_truncation_never_reaches_sdxl_prompt(monkeypatch):
    captured=install_replies(monkeypatch,[packages()[0]]+[('{"image_description":"cut off','length')]*2)
    generator=QwenLocalImagePromptGenerator(config())
    with pytest.raises(ValueError,match='token limit'):
        generator.generate_with_context('Museum','museum','facade','',[],context=context(4))
    call=generator.consume_last_call('Q1','image_001')
    assert len(captured)==3 and call['status']=='failed' and len(call['description_calls'])==2


def test_compression_repair_preserves_description_and_previous_prompt(monkeypatch):
    too_long=' '.join('unique'+str(i) for i in range(75))
    replies=packages(prompt=too_long)+[packages()[2]]
    captured=install_replies(monkeypatch,replies)
    generator=QwenLocalImagePromptGenerator(config())
    inputs=context(4,[PROMPTS[1]])
    inputs['presentation_context']['previous_prompt']=PROMPTS[1]
    result=generator.generate_with_context('Museum','museum','facade','',[],context=inputs)
    assert result['positive_prompt']==PROMPTS[0]
    assert len(captured)==4
    repair=json.JSONDecoder().raw_decode(captured[-1]['messages'][-1]['content'].split('INPUT_JSON:\n')[1])[0]
    assert repair['candidate_description']==DESCRIPTIONS[0]
    assert repair['presentation_context']['previous_prompt']==PROMPTS[1]
    assert 'guided_json' not in captured[-1]


@pytest.mark.parametrize('problem',['unknown_fact','repeated_fact','extra_key','bad_axis','same','synonym'])
def test_focus_plan_rejects_invalid_references_or_superficial_novelty(problem):
    plan=focus()
    previous=[]
    if problem=='unknown_fact': plan['supporting_fact_ids']=['unknown']
    elif problem=='repeated_fact': plan['supporting_fact_ids']*=2
    elif problem=='extra_key': plan['extra']='ignored'
    elif problem=='bad_axis': plan['rendering']['lighting']=[]
    else:
        previous=[deepcopy(plan)]
        plan['difference_from_previous']='This is a completely different image'
        if problem=='synonym':
            previous[0]['visual_focus']='glamorous actress'
            plan['visual_focus']='stylish actress'
    with pytest.raises(ImageFocusValidationError):
        validate_image_focus(plan,context()['source_facts'],previous)


def test_final_prompt_must_keep_the_planned_change_through_compression(monkeypatch):
    inputs=context(4,[PROMPTS[0]])
    inputs['presentation_context']['previous_focus_plans']=[focus()]
    captured=install_replies(monkeypatch,packages(DESCRIPTIONS[1],PROMPTS[0],1)+[packages(prompt=PROMPTS[1])[2]])
    generator=QwenLocalImagePromptGenerator(config())
    result=generator.generate_with_context('Museum','museum','facade','',[],context=inputs)
    assert result['positive_prompt']==PROMPTS[1]
    call=generator.consume_last_call('Q1','image_002')
    assert len(captured)==4 and 'repeats' in call['validation_attempts'][0]['error']
    assert 'focus_plan' in captured[-1]['messages'][-1]['content']


def test_declaring_a_different_plan_without_realizing_it_is_rejected():
    plan=focus()
    plan['rendering']['lighting']='backlit'
    audit=validate_image_focus(plan,context()['source_facts'],[focus()])
    with pytest.raises(ImageFocusValidationError,match='omits'):
        validate_focus_realization('Museum red brick facade in natural colors, an arched entrance in view.',
                                  plan,audit,[PROMPTS[0]])
    validate_focus_realization('Museum red brick facade and arched entrance, backlit architectural photograph.',
                              plan,audit,[PROMPTS[0]])


def test_truncated_focus_plan_is_retried_and_no_description_is_sent_if_it_still_fails(monkeypatch):
    captured=install_replies(monkeypatch,[('{"visual_focus":"cut','length')]*2)
    generator=QwenLocalImagePromptGenerator(config())
    with pytest.raises(ImageFocusValidationError,match='token limit'):
        generator.generate_with_context('Museum','museum','facade','',[],context=context(4))
    call=generator.consume_last_call('Q1','image_001')
    assert len(captured)==2 and not call['description_calls']
    assert 'validation_error' in captured[1]['messages'][-1]['content']


def test_insufficient_budget_starts_no_request(monkeypatch):
    captured=install_replies(monkeypatch,[])
    with pytest.raises(ValueError,match='focus selection'):
        QwenLocalImagePromptGenerator(config()).generate_with_context('Museum','museum','facade','',[],context=context(2))
    assert captured==[]
