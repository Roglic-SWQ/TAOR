from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.generation_prompts import build_generation_prompt, generation_payload
from mmkgc.io.resume import load_saved_image_run
from mmkgc.io.writer import RunWriter, _stage_artifact
from mmkgc.observation import bind_observation, repair_operations
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.plan_schema import FACT_SELECTION_SCHEMA
from mmkgc.ranking.context import build_ranking_context, candidate_ranking_context
from mmkgc.stages.candidate_planning import (
    apply_fact_selection, fact_selection_prompt, fact_selection_schema,
)
from mmkgc.stages.generation import relation_alternation_generation, safer_relation_alternation_generation
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.relation_planning import RelationPlanValidationError
from test_candidate_fact_plan import source, target, Images, PassRanker


def reply(ids=None):
    return {'status': 'ready', 'selected_fact_ids': ['rf:2', 'rf:3'] if ids is None else ids}


def selected(mode='image', ids=None):
    knowledge = source()
    selection = apply_fact_selection(knowledge, reply(ids), mode, 8)
    selection, reasoning = compile_plan_context(knowledge, selection, mode)
    return knowledge, selection, reasoning


class Views:
    uses_structured_prompts = True
    max_concurrency = 4
    def __init__(self, duplicate=False):
        self.calls = []
        self.duplicate = duplicate
    def generate_with_context(self, label, entity_type, relation, angle, facts, *, context):
        self.calls.append(deepcopy(context))
        index = 1 if self.duplicate else len(self.calls)
        return {'positive_prompt': f'Museum red brick facade and arched entrance, composition {index}',
                'used_text_cues': [], 'omitted_risky_details': []}


class Text:
    uses_structured_prompts = True
    def __init__(self): self.calls = []
    def generate_with_context(self, *args, context):
        self.calls.append(deepcopy(context))
        return 'Museum has a red brick facade and an arched entrance.'


def generated(duplicate=False, budget=3):
    knowledge, selection, reasoning = selected()
    views = Views(duplicate)
    result = relation_alternation_generation(target(), 'image', knowledge, reasoning, selection,
        None, views, Images(), candidate_budget=budget)
    return knowledge, selection, reasoning, views, result


def test_selection_may_use_less_than_limit_without_candidate_allocations():
    before = source()
    value = reply(['rf:2'])
    schema = fact_selection_schema(before, 8)
    pytest.importorskip('jsonschema').Draft202012Validator(schema).validate(value)
    result = apply_fact_selection(before, value, 'image', 8)['fine_grained_relation_selection']
    assert result == {'schema_version': FACT_SELECTION_SCHEMA, 'status': 'ready',
        'target_modality': ['image'], 'max_selected_facts': 8, 'selected_fact_ids': ['rf:2']}
    assert before == source()
    _, prompt = fact_selection_prompt('Museum', 'image', before, 8)
    payload = json.loads(prompt)
    assert payload['max_selected_facts'] == 8
    assert 'candidate_budget' not in payload and 'candidates' not in payload['output_template']
    assert 'PRIVATE_TARGET' not in prompt


@pytest.mark.parametrize('problem', ['over_limit', 'unknown', 'duplicate', 'empty', 'type',
    'status', 'extra', 'insufficient_nonempty', 'missing_evidence', 'foreign_evidence', 'outside_view'])
def test_invalid_selection_fails_without_mutating_sources(problem):
    knowledge, value, limit = source(), reply(), 8
    if problem == 'over_limit': limit = 1
    elif problem == 'unknown': value['selected_fact_ids'] = ['unknown']
    elif problem == 'duplicate': value['selected_fact_ids'] *= 2
    elif problem == 'empty': value['selected_fact_ids'] = []
    elif problem == 'type': value['selected_fact_ids'] = [True]
    elif problem == 'status': value['status'] = []
    elif problem == 'extra': value['candidates'] = []
    elif problem == 'insufficient_nonempty': value['status'] = 'insufficient_evidence'
    elif problem == 'missing_evidence': knowledge['relation_facts'][1]['evidence_refs'] = []
    elif problem == 'foreign_evidence': knowledge['relation_facts'][1]['evidence_refs'] = ['missing']
    elif problem == 'outside_view': knowledge['relevance_selection'] = {'selected_fact_ids': ['rf:1']}
    before = deepcopy(knowledge)
    with pytest.raises(RelationPlanValidationError):
        apply_fact_selection(knowledge, value, 'image', limit)
    assert knowledge == before


@pytest.mark.parametrize('local', [False, True])
def test_default_planner_selects_one_set_independent_of_generation_budget(monkeypatch, local):
    planner = OpenAICompatibleRelationPlanner(QwenConfig(
        'Qwen/test' if local else 'test', 'http://localhost/v1', 'private-key'), max_selected_facts=8)
    captured = []
    def complete(messages, schema):
        captured.append((deepcopy(messages), schema))
        return json.dumps(reply())
    monkeypatch.setattr(planner, '_complete', complete)
    first = planner.plan(target(), 'image', source(), 1)
    second = planner.plan(target(), 'image', source(), 17)
    assert first == second
    assert captured[0] == captured[1]
    assert planner.calls[0]['response_schema_version'] == FACT_SELECTION_SCHEMA
    assert planner.calls[0]['max_selected_facts'] == 8
    assert 'private-key' not in json.dumps(planner.calls)


def test_new_local_wire_only_guides_repairs_and_preserves_selection_limit(monkeypatch):
    requests = []
    replies = iter([reply(['MISSING_ID']), reply()])
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, *, json, headers):
            if url.endswith('/tokenize'):
                return httpx.Response(200, json={'max_model_len': 16384, 'count': 1000})
            requests.append(deepcopy(json))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': __import__('json').dumps(next(replies))}}]})
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'test'),
                                              max_selected_facts=8)
    planner.plan(target(), 'image', source(), 3)
    assert 'guided_json' not in requests[0]
    assert set(requests[1]['guided_json']['properties']) == {'status', 'selected_fact_ids'}
    assert 'MISSING_ID' not in json.dumps(requests[1]['messages'])
    assert len(planner.calls[-1]['validation_attempts']) == 2


def test_views_share_facts_and_checks_and_receive_previous_presentations(monkeypatch):
    monkeypatch.setattr('mmkgc.stages.generation._generation_angle_specs',
                        lambda *args: pytest.fail('shared facts must not invoke preset angles'))
    knowledge, selection, reasoning, views, result = generated()
    items = result['relation_alternation_generation']['image_candidates']
    assert len(items) == 3
    assert all(item['source_fact_refs'] == ['rf:2', 'rf:3'] for item in items)
    assert all(item['verification_targets'] == items[0]['verification_targets'] for item in items)
    assert all(item['requirement'] == 'should_include' for item in items[0]['verification_targets'] if item['fact_id'])
    for index, context in enumerate(views.calls):
        assert [fact['fact_id'] for fact in context['source_facts']] == ['rf:2', 'rf:3']
        assert context['presentation_context']['previous_candidates'] == [item['positive_prompt'] for item in items[:index]]
    artifact = _stage_artifact('fine_grained_relation_selection', selection)['data']
    assert artifact['selected_fact_ids'] == ['rf:2', 'rf:3']
    assert 'candidates' not in artifact and 'candidate_budget' not in artifact


def test_duplicate_presentations_do_not_generate_duplicate_images():
    _, _, _, views, result = generated(duplicate=True, budget=4)
    assert len(views.calls) == 4
    assert len(result['relation_alternation_generation']['image_candidates']) == 1
    assert len(result['generated_image_artifacts']) == 1
    assert all(len(call['presentation_context']['previous_candidates']) == 1 for call in views.calls[1:])


@pytest.mark.parametrize('budget', [1, 7])
def test_generation_count_is_independent_of_fact_count_and_not_capped_at_five(budget):
    _, _, _, views, result = generated(budget=budget)
    candidates = result['relation_alternation_generation']['image_candidates']
    assert len(candidates) == budget
    assert len(views.calls) == budget
    assert all(candidate['source_fact_refs'] == ['rf:2', 'rf:3'] for candidate in candidates)


@pytest.mark.parametrize('limit', [0, -1, True, '8'])
def test_invalid_fact_limit_is_rejected_before_model_requests(limit):
    with pytest.raises(ValueError, match='max_selected_facts'):
        OpenAICompatibleRelationPlanner(QwenConfig('test', 'http://localhost/v1', 'test'),
                                         max_selected_facts=limit)


def test_shared_reference_cannot_be_shrunk_by_one_candidate():
    knowledge, _, reasoning, _, generated_result = generated()
    images = {item.candidate_id: {'image_bytes': item.image_bytes, 'media_type': 'image/png'}
              for item in generated_result['generated_image_artifacts']}
    context = build_ranking_context('Q0', 'Museum', 'image', target().existing_modalities,
        ranking_knowledge(knowledge, reasoning), reasoning, generated_result, images)
    candidates = generated_result['relation_alternation_generation']['image_candidates']
    first = candidate_ranking_context(context, candidates[0])
    second = candidate_ranking_context(context, candidates[1])
    assert first.supported_facts == second.supported_facts
    changed = deepcopy(candidates[0])
    changed['source_fact_refs'] = ['rf:2']
    changed['verification_targets'] = [item for item in changed['verification_targets'] if item['fact_id'] != 'rf:3']
    with pytest.raises(ValueError, match='full common fact'):
        candidate_ranking_context(context, changed)


def test_common_image_omission_is_not_a_contradiction_or_required_repair():
    _, selection, _ = selected()
    checks = selection['fine_grained_relation_selection']['candidates'][0]['verification_targets']
    raw = {'fact_checks': [{
        'check_id': item['check_id'], 'status': 'missing' if item['fact_id'] else 'unverifiable',
        'verification_mode': 'direct' if item['fact_id'] else 'unverifiable',
        'observed': '', 'evidence': ['pixel: detail is not presented in this view'],
        'confidence': 0.95, 'reason': 'not depicted',
    } for item in checks], 'unsupported_claims': []}
    observation = bind_observation(raw, checks, 'image', 'image_001')
    assert repair_operations(observation) == []
    assert observation['repair_needed'] is False


def test_text_default_is_one_complete_candidate_with_shared_facts():
    knowledge, selection, reasoning = selected('text')
    text = Text()
    pipeline = MMKGCPipeline(text_generator=text, ranking_engine=PassRanker())
    assert pipeline.generation_budget('text') == 1
    assert pipeline.generation_budget('image') == 3
    result = pipeline.finish(PreparedEntity(target(), 'text', {}, knowledge, selection, reasoning, []))
    assert len(result['relation_alternation_generation']['relation_alternation_generation']['text_candidates']) == 1
    assert [fact['fact_id'] for fact in text.calls[0]['source_facts']] == ['rf:2', 'rf:3']


def test_no_evidence_produces_no_synthetic_candidates():
    knowledge = source()
    selection = apply_fact_selection(knowledge, {'status': 'insufficient_evidence', 'selected_fact_ids': []}, 'image', 8)
    selection, reasoning = compile_plan_context(knowledge, selection, 'image')
    views = Views()
    result = relation_alternation_generation(target(), 'image', knowledge, reasoning, selection,
                                            None, views, Images(), candidate_budget=3)
    assert views.calls == []
    assert result['relation_alternation_generation']['image_candidates'] == []


@pytest.mark.parametrize('duplicate', [False, True])
def test_shared_selection_and_fewer_candidates_resume_original_image_bytes(tmp_path, duplicate):
    knowledge, selection, reasoning = selected()
    call = {'stage': 'stage2_selection', 'status': 'applied', 'response_schema_version': FACT_SELECTION_SCHEMA,
            'max_selected_facts': 8, 'response_text': json.dumps(reply())}
    pipeline = MMKGCPipeline(candidate_budget=3, image_prompt_generator=Views(duplicate),
                             image_generator=Images(), ranking_engine=PassRanker())
    result = pipeline.finish(PreparedEntity(target(), 'image', {}, knowledge, selection, reasoning, [call]))
    writer = RunWriter(tmp_path, 'shared', {})
    writer.write_entity(result)
    entity_dir = next(writer.entities_dir.iterdir())
    restored, generated_result = load_saved_image_run(entity_dir, target(), 3)
    assert restored.selection['fine_grained_relation_selection']['selected_fact_ids'] == ['rf:2', 'rf:3']
    assert len(generated_result['generated_image_artifacts']) == (1 if duplicate else 3)
    for image in generated_result['generated_image_artifacts']:
        assert image.image_bytes == (entity_dir / image.relative_image_path).read_bytes()


def test_repair_keeps_shared_facts_without_running_multiview_planning_again():
    knowledge, selection, reasoning, _, _ = generated()
    generator = Views()
    feedback = {'schema_version': 'candidate_fact_repair_v1', 'candidate_feedback': [{
        'candidate_id': 'image_002', 'operations': [], 'repair_actions': ['remove unsupported detail'],
        'previous_prompt': 'Museum with unsupported detail'}]}
    repaired = safer_relation_alternation_generation(target(), 'image', knowledge, reasoning, selection,
        None, generator, Images(), candidate_budget=3, failed_ranking={}, stage3_feedback=feedback)
    items = repaired['relation_alternation_generation']['image_candidates']
    assert [item['source_candidate_id'] for item in items] == ['image_002']
    assert items[0]['source_fact_refs'] == ['rf:2', 'rf:3']
    assert 'presentation_context' not in generator.calls[0]


def test_previous_presentations_are_task_context_and_not_new_source_facts():
    _, _, _, generator, _ = generated()
    context = generator.calls[1]
    payload = generation_payload('Museum', 'museum', 'facade', 'ignored', [], context)
    prompt = build_generation_prompt('image', payload)
    actual = json.loads(prompt.split('INPUT_JSON:\n', 1)[1])
    assert actual['presentation_context']['previous_candidates'] == context['presentation_context']['previous_candidates']
    assert actual['source_facts'] == context['source_facts']
    assert 'not invented unseen geometry' in prompt
