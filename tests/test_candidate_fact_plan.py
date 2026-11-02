from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import httpx
from PIL import Image
import pytest

from mmkgc.adapters.base import GeneratedImageArtifact
from mmkgc.adapters.qwen import QwenConfig, QwenLocalTextGenerator, QwenLocalImagePromptGenerator
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.data import EntityRecord
from mmkgc.generation_prompts import IMAGE_PROMPT_VERSION, IMAGE_SYSTEM
from mmkgc.io.resume import load_saved_image_run
from mmkgc.io.writer import RunWriter, _stage_artifact
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.ranking.context import (build_ranking_context, build_kg_reference,
    build_structured_kg_triples, build_visual_constraint_text, candidate_ranking_context)
from mmkgc.ranking.contracts import GateResult, ScoreResult
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.profiles import Stage4Profile, LocalScorerConfig, GPT4OMiniJudgeConfig
from mmkgc.ranking.scorers.gpt4omini_judge import TextJudgeResult, ImageJudgeResult
from mmkgc.stages.candidate_planning import (
    PLAN_SCHEMA, LEGACY_PLAN_SCHEMA, apply_candidate_plans, candidate_plan_schema, parse_candidate_response)
from mmkgc.stages.generation import relation_alternation_generation, safer_relation_alternation_generation
from mmkgc.stages.generation import build_stage3_feedback
from mmkgc.observation import bind_observation
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.relation_planning import RelationPlanValidationError, apply_candidate_slot_plans, SLOT_SCHEMA


def source():
    facts = []
    catalog = []
    for index, (relation, value) in enumerate((('instance of', 'museum'), ('facade material', 'red brick'),
                                               ('entrance shape', 'arch'), ('architect', 'Jane')), 1):
        ref = f'evidence:{index}'
        triple = f'(Museum, {relation}, {value})'
        facts.append({'fact_id': f'rf:{index}', 'relation_group_id': f'group:{index}',
            'subject': 'Museum', 'relation': relation, 'object': value, 'direction': 'out',
            'claim': triple, 'fact': triple, 'evidence_refs': [ref], 'evidence_triples': [triple],
            'evidence_match_score': 0.01, 'scoring_status': 'scored'})
        catalog.append({'evidence_id': ref, 'source': 'stage0_kg', 'content': triple})
    profile = {'primary_type': 'museum', 'core_identity': 'a museum', 'evidence_refs': ['evidence:1']}
    visual = {'visual_scene_cues': [], 'context_cues': [], 'non_visual_facts': []}
    return {'schema_version': 'qwen_grounded_knowledge_v1',
        'stage1_v3': {'schema_version': 'qwen_grounded_knowledge_v1', 'entity_profile': profile,
            'text_visual_profile': visual, 'uncertain_aspects': [],
            'modality_state': {'targets': ['image'], 'existing': {'text': None, 'image': None}}},
        'entity_profile': {'entity_type': 'museum', 'core_identity': 'a museum'},
        'text_visual_profile': visual, 'relation_facts': facts, 'evidence_catalog': catalog,
        'existing_modalities': {'text': None, 'image': None}, 'uncertainties': []}


def response():
    return {'status': 'ready', 'candidates': [
        {'fact_ids': ['rf:2', 'rf:3']},
        {'fact_ids': ['rf:2']},
        {'fact_ids': ['rf:3']}]}


def target():
    return EntityRecord(0, 'Q0', 'Museum', 'PRIVATE_TARGET', [],
                        {'text': None, 'image': None}, {'text': 'PRIVATE_TARGET', 'image': 'PRIVATE_IMAGE'})


def planner(local=False, **kwargs):
    kwargs.setdefault('selection_schema', PLAN_SCHEMA)
    return OpenAICompatibleRelationPlanner(QwenConfig(
        'Qwen/Qwen2-VL-7B-Instruct' if local else 'gpt-4o-mini', 'http://localhost:12345/v1', 'private-key'),
        retry_backoff_seconds=0, **kwargs)


def test_small_schema_and_shared_facts_require_no_roles_or_fact_classification():
    schema = candidate_plan_schema(source(), 3)
    pytest.importorskip('jsonschema').Draft202012Validator(schema).validate(response())
    selection = apply_candidate_plans(source(), response(), 'image', 3)
    payload = selection['fine_grained_relation_selection']
    assert set(payload) == {'schema_version', 'status', 'target_modality', 'candidate_budget', 'candidates'}
    assert all(set(item) == {'plan_id', 'fact_ids'} for item in payload['candidates'])
    assert source()['relation_facts'][0]['fact_id'] not in {ref for item in payload['candidates'] for ref in item['fact_ids']}


@pytest.mark.parametrize('problem', ['count', 'unknown', 'duplicate', 'empty', 'focus',
    'quota', 'role', 'wrong_status', 'status_type', 'insufficient_with_candidates', 'no_evidence', 'foreign_evidence'])
def test_invalid_candidate_plans_are_rejected_without_mutating_source(problem):
    knowledge, reply = source(), response()
    if problem == 'count': reply['candidates'].pop()
    elif problem == 'unknown': reply['candidates'][0]['fact_ids'] = ['unknown']
    elif problem == 'duplicate': reply['candidates'][0]['fact_ids'] *= 2
    elif problem == 'empty': reply['candidates'][0]['fact_ids'] = []
    elif problem == 'focus': reply['candidates'][0]['focus'] = 'unrequested framing'
    elif problem in {'quota', 'role'}: reply['candidates'][0][problem] = 1
    elif problem == 'wrong_status': reply['status'] = 'maybe'
    elif problem == 'status_type': reply['status'] = []
    elif problem == 'insufficient_with_candidates': reply['status'] = 'insufficient_evidence'
    elif problem == 'no_evidence': knowledge['relation_facts'][1]['evidence_refs'] = []
    elif problem == 'foreign_evidence': knowledge['relation_facts'][1]['evidence_refs'] = ['missing']
    before = deepcopy(knowledge)
    with pytest.raises(RelationPlanValidationError):
        apply_candidate_plans(knowledge, reply, 'image', 3)
    assert knowledge == before


def test_duplicate_json_keys_are_not_silently_accepted():
    with pytest.raises(ValueError, match='duplicate JSON key'):
        parse_candidate_response('{"status":"ready","status":"insufficient_evidence","candidates":[]}')


@pytest.mark.parametrize('local', [False, True])
def test_candidate_plan_compatibility_returns_only_candidate_decisions(monkeypatch, local):
    client = planner(local)
    captured = []
    def complete(messages, schema):
        captured.append((messages, schema))
        return json.dumps(response())
    monkeypatch.setattr(client, '_complete', complete)
    result = client.plan(target(), 'image', source(), 3)
    assert result['fine_grained_relation_selection']['schema_version'] == PLAN_SCHEMA
    payload = json.loads(captured[0][0][1]['content'])
    assert all(set(item) == {'fact_ids'} for item in payload['output_template']['candidates'])
    assert 'PRIVATE_TARGET' not in json.dumps(captured)
    assert client.calls[-1]['response_schema_version'] == PLAN_SCHEMA
    assert 'private-key' not in json.dumps(client.calls)


def test_contract_repair_uses_original_source_and_preserves_all_attempts(monkeypatch):
    client = planner(True)
    bad = response()
    bad['candidates'][0]['fact_ids'] = ['BAD_RESPONSE_MARKER']
    replies = iter([json.dumps(bad), json.dumps(response())])
    messages_seen = []
    def complete(messages, schema):
        messages_seen.append(deepcopy(messages))
        return next(replies)
    monkeypatch.setattr(client, '_complete', complete)
    client.plan(target(), 'image', source(), 3)
    assert len(messages_seen) == 2
    assert messages_seen[1][:2] == messages_seen[0]
    assert 'BAD_RESPONSE_MARKER' not in json.dumps(messages_seen[1])
    assert len(client.calls[-1]['validation_attempts']) == 2


def test_repeated_bad_plan_stops_and_redacts_attached_error_log(monkeypatch):
    client = planner(max_contract_repairs=4)
    monkeypatch.setattr(client, '_complete', lambda *args: '{"status":"ready","candidates":[]}')
    with pytest.raises(RelationPlanValidationError) as caught:
        client.plan(target(), 'image', source(), 3)
    assert len(client.calls[-1]['validation_attempts']) == 2
    assert client.calls[-1]['contract_repair_stopped_reason'] == 'repeated_invalid_response'
    assert caught.value.model_call is client.calls[-1]


def test_transient_retries_share_existing_stage_budget(monkeypatch):
    client = planner()
    replies = iter([TimeoutError('temporary'), json.dumps(response())])
    def complete(*args):
        item = next(replies)
        if isinstance(item, Exception): raise item
        return item
    monkeypatch.setattr(client, '_complete', complete)
    client.plan(target(), 'image', source(), 3)
    call = client.calls[-1]
    assert call['transport_retries'] == 1 and len(call['network_attempts']) == 2
    assert call['network_attempts'][1]['timeout_seconds'] <= call['network_attempts'][0]['timeout_seconds']


def test_local_wire_uses_simple_template_and_only_guides_repairs(monkeypatch):
    requests = []
    replies = iter([{'status': 'ready', 'candidates': []}, response()])
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
    planner(True).plan(target(), 'image', source(), 3)
    assert 'guided_json' not in requests[0]
    assert set(requests[1]['guided_json']['properties']) == {'status', 'candidates'}
    assert 'candidate_anchors' not in __import__('json').dumps(requests)


class Generator:
    uses_structured_prompts = True
    def __init__(self, image=False): self.calls = []; self.image = image
    def generate_with_context(self, label, entity_type, relation, angle, facts, *, context):
        self.calls.append(deepcopy(context))
        value = '; '.join(facts)
        return {'positive_prompt': value} if self.image else value


class Images:
    def generate(self, key, candidate_id, positive, negative):
        buffer = BytesIO()
        Image.new('RGB', (2, 2)).save(buffer, format='PNG')
        metadata = {'entity_key': key, 'candidate_id': candidate_id, 'positive_prompt': positive,
                    'generator': 'fake', 'model_name': 'fake', 'model_path': 'fake',
                    'width': 2, 'height': 2, 'steps': 1, 'seed': 1}
        return GeneratedImageArtifact(key, candidate_id, f'generated_images/{candidate_id}.png',
            f'generated_images/{candidate_id}.json', buffer.getvalue(), metadata, {})


def generated(mode='text'):
    knowledge = source()
    selection, reasoning = compile_plan_context(knowledge,
        apply_candidate_plans(knowledge, response(), mode, 3), mode)
    text, image = Generator(), Generator(True)
    generation = relation_alternation_generation(target(), mode, knowledge, reasoning, selection,
        text, image, Images(), candidate_budget=3)
    return knowledge, selection, reasoning, generation, text if mode == 'text' else image


def context_for(mode='text'):
    knowledge, selection, reasoning, generation, generator = generated(mode)
    images = {artifact.candidate_id: {'image_bytes': artifact.image_bytes, 'media_type': 'image/png'}
              for artifact in generation['generated_image_artifacts']}
    context = build_ranking_context('Q0', 'Museum', mode, target().existing_modalities,
        ranking_knowledge(knowledge, reasoning), reasoning, generation, images)
    return context, generator


@pytest.mark.parametrize('mode', ['text', 'image'])
def test_generation_and_observation_share_fact_sets_without_preassigned_focus(mode, monkeypatch):
    def reject_old_angles(*args):
        raise AssertionError('fact-only plans must not use preassigned angle templates')
    monkeypatch.setattr('mmkgc.stages.generation._generation_angle_specs', reject_old_angles)
    knowledge, selection, reasoning, generation, generator = generated(mode)
    items = generation['relation_alternation_generation'][f'{mode}_candidates']
    for item, planned, request in zip(items, response()['candidates'], generator.calls):
        assert 'focus' not in item
        assert request['task'] == {'target_modality': mode, 'presentation_policy': 'stage3'}
        assert item['source_fact_refs'] == planned['fact_ids']
        assert [fact['fact_id'] for fact in request['source_facts']] == planned['fact_ids']
        assert {check['fact_id'] for check in item['verification_targets'] if check['fact_id']} == set(planned['fact_ids'])
        assert 'Jane' not in json.dumps(request)


@pytest.mark.parametrize('mode', ['text', 'image'])
def test_qwen_original_profile_uses_fact_only_context_and_stage3_presentation(mode):
    _, _, _, _, fake_generator = generated(mode)
    config = QwenConfig('Qwen/test', 'http://localhost/v1', 'test', prompt_profile='qwen7b_original_v1')
    client = (QwenLocalTextGenerator if mode == 'text' else QwenLocalImagePromptGenerator)(config)
    prompt = client._build_prompt('Museum', 'museum', 'facade material', 'UNREQUESTED_ANGLE_SENTINEL',
                                  [], fake_generator.calls[1])
    payload = json.loads(prompt.split('INPUT_JSON:\n', 1)[1])
    assert payload['task']['presentation_policy'] == 'stage3'
    assert 'generation_angle' not in payload['task'] and 'diversity_axis' not in payload['task']
    assert 'UNREQUESTED_ANGLE_SENTINEL' not in prompt
    assert [fact['fact_id'] for fact in payload['source_facts']] == ['rf:2']
    assert client._prompt_metadata(prompt)['prompt_version'] == (
        'entity_semantics_text_v4' if mode == 'text' else IMAGE_PROMPT_VERSION)


@pytest.mark.parametrize('local_qwen', [True, False])
def test_image_format_repair_preserves_candidate_scope_system_and_version(monkeypatch, local_qwen):
    _, _, _, _, generator = generated('image')
    context = deepcopy(generator.calls[1])
    original = deepcopy(context)
    base_url = 'http://localhost/v1' if local_qwen else 'https://qwen.example.test/v1'
    config = QwenConfig('Qwen/test', base_url, 'test',
                        prompt_profile='qwen7b_original_v1')
    client = QwenLocalImagePromptGenerator(config)
    content = {'positive_prompt': 'Museum, red brick facade, frontal view, neutral background',
               'used_text_cues': ['rf:2'], 'omitted_risky_details': []}
    replies = iter([('length', '{"positive_prompt":"broken'), ('stop', json.dumps(content))])
    requests = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            finish, text = next(replies)
            return json.dumps({'choices': [{'finish_reason': finish,
                                            'message': {'content': text}}]}).encode()

    def urlopen(req, **kwargs):
        requests.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', urlopen)
    result = client.generate_with_context('Museum', 'museum', 'facade material',
                                         'UNREQUESTED_ANGLE_SENTINEL', [], context=context)
    assert {key: result[key] for key in content} == content
    assert len(requests) == 2
    assert 'guided_json' not in requests[0]
    if local_qwen:
        assert requests[1]['guided_json']['properties']['positive_prompt']['maxLength'] == 400
    else:
        assert 'guided_json' not in requests[1]
    payloads = []
    for request in requests:
        assert request['messages'][0]['content'] == IMAGE_SYSTEM
        prompt = request['messages'][1]['content']
        assert 'UNREQUESTED_ANGLE_SENTINEL' not in prompt
        assert 'Jane' not in prompt
        payload = json.loads(prompt.split('INPUT_JSON:\n', 1)[1])
        assert payload['prompt_version'] == IMAGE_PROMPT_VERSION
        assert [fact['fact_id'] for fact in payload['source_facts']] == ['rf:2']
        assert 'generation_angle' not in payload['task']
        payloads.append(payload)
    assert payloads[0]['source_evidence'] == payloads[1]['source_evidence']
    assert context == original
    call = client.consume_last_call('Q0', 'image_002')
    assert call['prompt_version'] == IMAGE_PROMPT_VERSION
    assert call['metadata']['contract_repair_retries'] == 1


@pytest.mark.parametrize('mode', ['text', 'image'])
def test_ranking_references_only_the_current_candidate_plan(mode):
    context, _ = context_for(mode)
    candidates = context.text_candidates if mode == 'text' else context.image_candidates
    second, third = candidates[1:]
    scoped = candidate_ranking_context(context, second)
    assert {fact['fact_id'] for fact in scoped.supported_facts} == {'rf:2'}
    reference = build_kg_reference(scoped, mode)
    assert 'red brick' in reference and 'arch' not in reference and 'Jane' not in reference
    assert {fact['object'] for fact in build_structured_kg_triples(scoped, mode)} == {'red brick'}
    if mode == 'image':
        assert 'arch' not in build_visual_constraint_text(context, second)
        assert 'red brick' not in build_visual_constraint_text(context, third)


def test_candidate_cannot_borrow_another_candidates_fact_checks():
    context, _ = context_for('text')
    candidate = dict(context.text_candidates[1])
    candidate['verification_targets'] = context.text_candidates[2]['verification_targets']
    with pytest.raises(ValueError, match='checks do not match'):
        candidate_ranking_context(context, candidate)


def test_insufficient_evidence_produces_no_synthetic_fallback_candidates():
    selection, reasoning = compile_plan_context(source(), apply_candidate_plans(source(),
        {'status': 'insufficient_evidence', 'candidates': []}, 'text', 3), 'text')
    result = relation_alternation_generation(target(), 'text', source(), reasoning, selection,
        Generator(), Generator(True), None, candidate_budget=3)
    assert result['relation_alternation_generation']['text_candidates'] == []


def test_stage2_artifact_has_only_the_simple_plan_and_basic_metadata():
    _, selection, _, _, _ = generated()
    artifact = _stage_artifact('fine_grained_relation_selection', selection)
    assert artifact['data']['schema_version'] == PLAN_SCHEMA
    assert all(set(candidate) == {'plan_id', 'fact_ids'} for candidate in artifact['data']['candidates'])
    assert 'quota_plan' not in artifact['data'] and 'relation_plans' not in artifact['data']


class PassRanker:
    def rank(self, context):
        mode = context.missing_mode
        candidates = context.text_candidates if mode == 'text' else context.image_candidates
        items = [{**item, 'hard_gate': {'decision': 'pass'}, 'final_score': 0.8} for item in candidates]
        return {'ranking_status': 'completed', f'{mode}_ranking': {
            'best_candidate_id': items[0]['candidate_id'] if items else None, 'ranked_candidates': items}}


@pytest.mark.parametrize('version', [PLAN_SCHEMA, LEGACY_PLAN_SCHEMA, SLOT_SCHEMA])
def test_plan_saves_and_resumes_original_image_bytes_without_replanning(tmp_path, version):
    knowledge = source()
    reply = response()
    if version == SLOT_SCHEMA:
        reply = {'plan': {'status': 'ready', 'candidate_anchors': ['rf:2'] * 3,
            'relation_plans': [{'fact_id': fact['fact_id'],
                'role': 'primary' if fact['fact_id'] == 'rf:2' else 'secondary',
                'evidence_refs': fact['evidence_refs'], 'supporting_fact_ids': [], 'reason': 'source supported'}
                for fact in knowledge['relation_facts']]}}
    elif version == LEGACY_PLAN_SCHEMA:
        for candidate in reply['candidates']:
            candidate['focus'] = 'historical focus'
    selection = (apply_candidate_slot_plans(knowledge, reply, 'image', 3) if version == SLOT_SCHEMA else
                 apply_candidate_plans(knowledge, reply, 'image', 3, schema_version=version))
    selection, reasoning = compile_plan_context(knowledge, selection, 'image')
    call = {'stage': 'stage2_selection', 'status': 'applied', 'response_schema_version': version,
            'response_text': json.dumps(reply)}
    prepared = PreparedEntity(target(), 'image', {}, knowledge, selection, reasoning, [call])
    pipeline = MMKGCPipeline(candidate_budget=3, text_generator=Generator(), image_prompt_generator=Generator(True),
                             image_generator=Images(), ranking_engine=PassRanker())
    result = pipeline.finish(prepared)
    writer = RunWriter(tmp_path, 'simple', {})
    writer.write_entity(result)
    entity_dir = next(writer.entities_dir.iterdir())
    restored, generation = load_saved_image_run(entity_dir, target(), 3)
    assert restored.selection['fine_grained_relation_selection']['schema_version'] == (
        'model_relation_plan_v1' if version == SLOT_SCHEMA else version)
    if version != SLOT_SCHEMA:
        restored_plans = restored.selection['fine_grained_relation_selection']['candidates']
        assert all(('focus' in plan) == (version == LEGACY_PLAN_SCHEMA) for plan in restored_plans)
    for artifact in generation['generated_image_artifacts']:
        assert artifact.image_bytes == (entity_dir / artifact.relative_image_path).read_bytes()
    resumed = pipeline.finish(restored, generation_result=generation)
    assert resumed['final']['best_image_candidate_id'] == result['final']['best_image_candidate_id']


class TextScores:
    def __init__(self): self.references = []
    def score_candidates(self, reference, texts):
        self.references.extend([reference] * len(texts))
        return [ScoreResult(0.8, 0.8, 'scored', 'fake') for _ in texts]


class VisualScores:
    def __init__(self): self.references = []
    def score_image_quality_batch(self, image, texts):
        self.references.extend(texts)
        return [ScoreResult(80, 0.8, 'scored', 'fake') for _ in texts]


class Judge:
    def __init__(self): self.inputs = {}
    def judge_text(self, context, candidate):
        self.inputs[candidate['candidate_id']] = context.to_prompt_payload()
        return TextJudgeResult(ScoreResult(80, 0.8, 'scored', 'fake'), GateResult('pass', 'fake'))
    def judge_image(self, context, candidate, *, image_input):
        self.inputs[candidate['candidate_id']] = context.to_prompt_payload()
        return ImageJudgeResult(ScoreResult(80, 0.8, 'scored', 'fake'), GateResult('pass', 'fake'))
    def extract_text_graph(self, candidate):
        import re
        triples = re.findall(r'\(([^,]+), ([^,]+), ([^)]+)\)', candidate['generated_content']['value'])
        return {'triples': [{'subject': head, 'predicate': relation, 'object': tail, 'confidence': 1.0}
                            for head, relation, tail in triples]}


@pytest.mark.parametrize('mode', ['text', 'image'])
def test_real_ranking_engine_uses_candidate_local_plans_for_all_scorers(mode):
    context, _ = context_for(mode)
    required = LocalScorerConfig(True, True, None, None)
    disabled = LocalScorerConfig(False, False, None, None)
    profile = Stage4Profile('fake', required, disabled, required, required,
        GPT4OMiniJudgeConfig(True, Path('fake.yaml'), 1, max_concurrency=3))
    text, visual, judge = TextScores(), VisualScores(), Judge()
    engine = ModelFusionRankingEngine(profile=profile, scorers={
        'sentence_bert': text, 'clip': visual, 'blip_itm': visual, 'gpt4omini_judge': judge})
    ranking = engine.rank(context)
    assert ranking['ranking_status'] == 'completed'
    assert {fact['fact_id'] for fact in judge.inputs[f'{mode}_002']['supported_facts']} == {'rf:2'}
    if mode == 'text':
        assert all(item['model_scores']['structured_kg_alignment'] == 1
                   for item in ranking['text_ranking']['ranked_candidates'])
        assert any('red brick' in value and 'arch' not in value for value in text.references)
    else:
        assert any('red brick' in value and 'arch' not in value for value in visual.references)


def test_reflection_reuses_original_fact_plan_without_role_or_quota_replanning():
    knowledge, selection, reasoning, generation, _ = generated('text')
    original = generation['relation_alternation_generation']['text_candidates'][0]
    targets = original['verification_targets']
    raw = {'unsupported_claims': [], 'fact_checks': [
        {'check_id': item['check_id'], 'status': 'missing' if item['fact_id'] == 'rf:3' else 'supported',
         'verification_mode': 'direct', 'observed': 'candidate text', 'evidence': ['text: checked span'],
         'confidence': 0.95, 'reason': 'checked'} for item in targets]}
    observed = {**original, 'hard_gate': {'decision': 'pass'}, 'final_score': 0.8,
                'observation': bind_observation(raw, targets, 'text', original['candidate_id'])}
    ranking = {'ranking_status': 'completed', 'text_ranking': {
        'best_candidate_id': original['candidate_id'], 'ranked_candidates': [observed]}}
    feedback = build_stage3_feedback(ranking, attempt=1, generation=generation['relation_alternation_generation'])
    generator = Generator()
    repaired = safer_relation_alternation_generation(target(), 'text', knowledge, reasoning, selection,
        generator, Generator(True), None, candidate_budget=3, failed_ranking=ranking, stage3_feedback=feedback)
    items = repaired['relation_alternation_generation']['text_candidates']
    assert len(items) == 1 and items[0]['source_candidate_id'] == 'text_001'
    assert items[0]['source_fact_refs'] == original['source_fact_refs']
    assert items[0]['verification_targets'] == original['verification_targets']
