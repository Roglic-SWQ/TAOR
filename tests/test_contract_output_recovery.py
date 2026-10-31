"""Regressions for scoring contract failures and recovery without image regeneration."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from mmkgc.adapters.base import GeneratedImageArtifact
from mmkgc.adapters.json_contract import local_semantic_json_schema, local_relation_json_schema
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.io import RunWriter
from mmkgc.io.resume import load_saved_image_run
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.factory import _OpenAICompatibleCompletionClient
from mmkgc.ranking.scorers.gpt4omini_judge import (
    GPT4OMiniJudgeParseError, GPT4OMiniRankingJudge, observation_judgment_schema,
)
from test_fact_observation import setup, raw_observation, judgment, png
from test_judge_concurrency import context, profile, LocalVisualScorer
from test_missing_image_completion import batch, make_job, record, result


def test_wire_schema_rejects_missing_observation_and_illegal_fact_status_pairs():
    *_, targets = setup('image')
    schema = observation_judgment_schema(targets, 'image')
    good = json.loads(judgment(raw_observation(targets)))
    jsonschema.validate(good, schema)
    for problem in ('missing_observation', 'appearance_supported', 'missing_pixel'):
        invalid = deepcopy(good)
        if problem == 'missing_observation':
            invalid.pop('observation')
        elif problem == 'appearance_supported':
            invalid['observation']['fact_checks'][0].update(status='supported', verification_mode='appearance_only')
        else:
            invalid['observation']['fact_checks'][0]['evidence'] = ['KG says so']
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_client_sends_strict_observation_schema_and_same_actual_image(monkeypatch):
    captured = []
    client = _OpenAICompatibleCompletionClient(QwenConfig('gpt-4o-mini', 'http://test.invalid/v1', 'secret'), timeout_seconds=1)
    def respond(req):
        captured.append(json.loads(req.data))
        return {'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(client, '_open_json_with_timeout', respond)
    schema = observation_judgment_schema(setup()[-1], 'image')
    url = 'data:image/png;base64,AAA='
    assert client.complete_multimodal_structured('Check this candidate', url, schema) == '{}'
    payload = captured[0]
    assert payload['response_format']['json_schema'] == {'name':'stage4_judgment','strict':True,'schema':schema}
    assert payload['messages'][1]['content'][1]['image_url']['url'] == url


def test_judge_repairs_missing_checks_then_illegal_semantics_without_weakening_validation():
    *_, targets = setup('image')
    good = raw_observation(targets)
    bad = deepcopy(good)
    bad['fact_checks'][0].update(status='supported', verification_mode='appearance_only')
    legacy = json.loads(judgment(good)); legacy.pop('observation')
    replies = iter([json.dumps(legacy), judgment(bad), judgment(good)])
    calls = []
    class Client:
        def complete_structured(self, prompt, schema):
            calls.append((prompt, schema))
            return next(replies)
    judge = GPT4OMiniRankingJudge(Client())
    scored = judge._judge_with_contract_repair('Check provided facts',
        __import__('mmkgc.ranking.scorers.gpt4omini_judge', fromlist=['parse_image_judge_response']).parse_image_judge_response,
        kind='image', candidate_id='image_001', verification_targets=targets)
    assert len(calls) == 3 and all('observation' in c[1]['required'] for c in calls)
    assert scored.score.details['contract_repair_retries'] == 2
    assert len(scored.observation['fact_checks']) == len(targets)
    assert 'supported/appearance_only' in calls[-1][0]


@pytest.mark.parametrize('malformed', [True, False])
def test_all_failed_judgments_are_distinct_from_real_content_rejection(malformed):
    from mmkgc.ranking.contracts import GateResult, ScoreResult
    class Judge:
        model_name = 'test'
        def judge_image(self, *args, **kwargs):
            if malformed:
                raise GPT4OMiniJudgeParseError('missing factual observation')
            return SimpleNamespace(score=ScoreResult(10, 0.1, 'scored', 'test'),
                gate=GateResult('reject', 'test', hard_violations=('identity_mismatch',)))
    engine = ModelFusionRankingEngine(profile=profile(3), scorers={
        'gpt4omini_judge':Judge(), 'clip':LocalVisualScorer(), 'blip_itm':LocalVisualScorer()})
    ranked = engine.rank(context())
    assert ranked['ranking_status'] == ('evaluation_failed' if malformed else 'no_valid_candidate')
    assert ranked['image_ranking']['best_candidate_id'] is None


def test_batch_preserves_png_but_marks_judge_failure_as_stage4_failure(tmp_path):
    job = make_job(tmp_path, 1)
    writer = RunWriter(job.root/'attempts', 'attempt_0001', {})
    entity = record()
    entry = batch.begin_entity(job, job.manifest[0], 1)
    entry['record'] = entity
    payload = result(entity, passed=False)
    payload['kg_aware_candidate_consistency_ranking']['ranking_status'] = 'evaluation_failed'
    batch.persist_entity_result(job, writer, entry, payload)
    assert entry['row']['status'] == 'failed' and entry['row']['failed_stage'] == 'stage4'
    assert (job.root/'entities/Q1/generated_images/image_001.png').is_file()
    assert not (job.root/'completed_images/Q1/image.png').exists()


def test_supplied_generation_skips_generators_and_keeps_candidate_bytes():
    r, coarse, knowledge, selection, reasoning, _ = setup()
    class Engine:
        def rank(self, context):
            return {'ranking_status':'no_valid_candidate','image_ranking':{'ranked_candidates':[]}}
    pipeline = MMKGCPipeline(ranking_engine=Engine())
    artifact = GeneratedImageArtifact('Q1', 'image_001', 'generated_images/image_001.png',
                                     'generated_images/image_001.json', png(), {}, {})
    prepared = PreparedEntity(r, 'image', coarse, knowledge, selection, reasoning, [])
    output = pipeline.finish(prepared, generation_result={
        'relation_alternation_generation':{'image_candidates':[],'text_candidates':[]},
        'generated_image_artifacts':[artifact], 'model_calls':[]})
    assert output['generated_image_artifacts'][0].image_bytes == artifact.image_bytes


def test_local_stage1_grammar_prevents_wrong_types_without_replacing_fact_claims():
    schema = local_semantic_json_schema('"entity_profile" "text_visual_profile" "relation_facts"')
    good = {'relation_facts':[{'fact_id':'rf:0001','claim':'A model authored claim'}],
        'entity_profile':{'primary_type':'person','secondary_types':[],'semantic_domains':[],
                          'core_identity':'Person','evidence_refs':['fact:0001']},
        'text_visual_profile':{'visual_scene_cues':[],'context_cues':[],'non_visual_facts':['An invisible fact']},
        'key_attributes':[],'uncertain_aspects':[]}
    jsonschema.validate(good, schema)
    bad = deepcopy(good); bad['text_visual_profile']['non_visual_facts'] = 'wrong scalar'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


@pytest.mark.parametrize('repair', [False, True])
def test_local_semantic_timeout_retries_once_and_uses_json_grammar(monkeypatch, repair):
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'{"choices":[{"message":{"content":"{}"}}]}'
    def send(req, **kwargs):
        requests.append(json.loads(req.data))
        if len(requests) == 1: raise TimeoutError('test timeout')
        return Response()
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', send)
    monkeypatch.setattr('mmkgc.adapters.qwen.time.sleep', lambda _:None)
    planner = QwenLocalSemanticStagePlanner(QwenConfig('Qwen/Qwen2-VL-7B-Instruct','http://test/v1','EMPTY'))
    prompt = 'Return an object' + ('\nVALIDATION_FEEDBACK_JSON: []' if repair else '')
    assert planner._complete_prompt(prompt) == '{}'
    assert len(requests) == 2
    if repair:
        assert requests[0]['guided_json'] == {'type':'object'}
    else:
        assert 'guided_json' not in requests[0]


def test_stage2_repair_grammar_prevents_copying_biography_into_reason():
    plan = {'plan':{'status':'ready','candidate_anchors':['rf:0001'] * 3,
        'relation_plans':[{'fact_id':'rf:0001','role':'primary','evidence_refs':[],
                           'supporting_fact_ids':[],'reason':'A concise grounded explanation.'}]}}
    jsonschema.validate(plan, local_relation_json_schema())
    plan['plan']['relation_plans'][0]['reason'] = 'An entire copied biography. ' * 30
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(plan, local_relation_json_schema())


def test_syntax_failure_then_combined_missing_facts_and_slots_use_focused_model_repairs(monkeypatch):
    from test_stage2_qwen_targeted_repair import example, planner
    from mmkgc.adapters.qwen_stage2_repair import TASK_MARKER
    knowledge, full = example(3)
    original = deepcopy(knowledge)
    partial = deepcopy(full)
    partial['plan']['relation_plans'] = partial['plan']['relation_plans'][:1]
    partial['plan']['candidate_anchors'] = ['rf:0002','rf:0003']
    client = planner(); client.max_contract_repairs = 3
    requests = []
    def respond(messages, schema):
        kind = schema.get(TASK_MARKER)
        requests.append(kind)
        if len(requests)==1:
            return '{malformed'
        if kind is None:
            return json.dumps(partial)
        if kind=='missing_facts':
            return json.dumps({'facts':{fid:{'role':'secondary','evidence_refs':[f'fact:{fid[-4:]}'],
                'supporting_fact_ids':[],'reason':'Model reviewed the omitted source fact.'} for fid in ['rf:0002','rf:0003']}})
        assert kind=='candidate_slots'
        return json.dumps({'slots':{f'slot_{i:03d}':fid for i,fid in enumerate(
            ['rf:0001','rf:0002','rf:0003','rf:0001','rf:0002'],1)}})
    monkeypatch.setattr(client,'_complete',respond)
    selected=client.plan(SimpleNamespace(label='City'),'image',knowledge,5)['fine_grained_relation_selection']
    assert requests==[None,None,'missing_facts','candidate_slots']
    assert selected['quota_summary']['assigned_total']==5 and knowledge==original
    assert selected['primary_relations'][0]['fact_id']=='rf:0001'
    assert len(selected['secondary_relations'])==2
