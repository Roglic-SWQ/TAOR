from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from mmkgc.adapters.source_text_triples import SourceTextTask, source_extraction_report
from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.stages.generation import route_image_repairs


def lexical_response():
    return {'entity_profile': {'primary_type': 'verb', 'core_identity': 'stimulate'},
            'extracted_triples': [
                {'subject': '__TARGET__', 'relation': 'means', 'object': 'cause to be alert and energetic'},
                {'subject': 'Coffee and tea', 'relation': 'stimulate', 'object': 'me'}],
            'uncertain_aspects': []}


def test_lexical_usage_keeps_actors_without_becoming_target_properties():
    task = SourceTextTask('stimulate', 'existing:text',
                         'stimulate, cause to be alert and energetic; "Coffee and tea stimulate me"')
    raw = lexical_response();before = deepcopy(raw)
    result = task.decode(raw)
    assert raw == before
    assert len(result['extracted_triples']) == 1
    assert result['extracted_triples'][0]['subject'] == 'stimulate'
    report = source_extraction_report(task, raw, task.inputs())
    assert report['accepted_triple_count'] == 1
    assert report['context_triples'] == [{'triple': raw['extracted_triples'][1],
        'evidence_refs': ['existing:text'], 'scope': 'lexical_source_context'}]


def test_named_entity_background_guard_is_preserved():
    from mmkgc.stages.modeling import KnowledgeResponseValidationError
    task = SourceTextTask('Zhejiang', 'existing:text', 'Hangzhou is in Zhejiang. Jane lives in London.')
    raw = {'entity_profile': {'primary_type': 'place', 'core_identity': 'Zhejiang'},
           'extracted_triples': [{'subject': 'Jane', 'relation': 'lives in', 'object': 'London'}],
           'uncertain_aspects': []}
    with pytest.raises(KnowledgeResponseValidationError):
        task.decode(raw)


def test_image_writer_uses_the_definition_without_changing_archived_facts():
    from mmkgc.generation_prompts import lexical_image_view
    original = {'entity': {'label': 'sun', 'type': 'verb'}, 'identity_context': {'core_identity': 'sun'},
        'source_facts': [{'fact_id': 'mf:1', 'subject': 'sun', 'relation': 'means',
                         'object': 'expose to the rays of the sun', 'direction': 'out'}],
        'source_evidence': [{'content': 'sun, expose to the rays of the sun', 'modality': 'text'}],
        'checks': [{'claim': 'sun'}]}
    before = deepcopy(original)
    view = lexical_image_view(original)
    assert original == before
    assert view['entity']['label'] == 'expose to the rays of the sun'
    assert view['source_facts'][0]['subject'] == 'the target sense'
    assert view['source_facts'][0]['object'] == 'expose to the rays of the sun'
    assert view['source_evidence'][0]['content'] == 'expose to the rays of the sun'
    assert view['checks'] == []
    assert lexical_image_view(view) == view


@pytest.mark.parametrize('review_ready', [True, False])
def test_empty_selection_gets_one_review_without_forcing_ready(monkeypatch, review_ready):
    from test_shared_fact_selection import source
    p = OpenAICompatibleRelationPlanner(QwenConfig('gpt-test', 'https://api.invalid/v1', 'key'))
    sent = []
    def complete(messages, *args):
        sent.append(messages)
        return json.dumps({'status': 'ready', 'selected_fact_ids': ['rf:2']} if
                          len(sent) == 2 and review_ready else
                          {'status': 'insufficient_evidence', 'selected_fact_ids': []})
    monkeypatch.setattr(p, '_complete_with_retries', complete)
    result = p.plan(SimpleNamespace(label='Museum'), 'image', source(), 2)
    assert len(sent) == 2
    assert result['fine_grained_relation_selection']['status'] == ('ready' if review_ready else 'insufficient_evidence')
    assert p.calls[-1]['contract_repair_retries'] == 0
    assert p.calls[-1]['empty_selection_review']['initial_response']


def test_semantic_feedback_goes_to_writer_without_second_judge_gate():
    feedback = {'candidate_feedback': [{'candidate_id': 'image_001', 'modality': 'image',
        'operations': [{'action': 'reframe', 'observed': 'The image depicts metal, not paying attention.'}],
        'judge_feedback': {'rubric_scores': {'entity_semantics': {'score': 0, 'reason': 'Wrong word sense.'}}}}]}
    def never(*_):
        pytest.fail('Semantic feedback must not need a second prompt reviewer')
    routed, calls = route_image_repairs(feedback, never)
    assert calls == [] and routed['repair_needed']
    task = routed['candidate_feedback'][0]
    assert task['repair_route'] == 'edit_prompt'
    assert task['judge_feedback'] == feedback['candidate_feedback'][0]['judge_feedback']


@pytest.mark.parametrize('has_success', [False, True])
def test_repair_service_failure_is_recoverable_without_losing_success(monkeypatch, has_success):
    from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
    from test_shared_fact_selection import generated
    from test_candidate_fact_plan import target, PassRanker
    k, selection, reasoning, _, generation = generated(budget=1)
    class Engine:
        def rank(self, context):
            ranking = PassRanker().rank(context)
            if not has_success:
                ranking['ranking_status'] = 'no_valid_candidate'
                ranking['image_ranking']['best_candidate_id'] = None
            return ranking
    monkeypatch.setattr('mmkgc.pipeline.build_stage3_feedback', lambda *args, **kwargs: {
        'schema_version': 'candidate_fact_repair_v1', 'candidate_feedback': []})
    monkeypatch.setattr('mmkgc.pipeline.route_image_repairs', lambda *args: ({
        'schema_version': 'candidate_fact_repair_v1', 'repair_needed': False,
        'skipped_repairs': [{'review_error_type': 'HTTPError', 'review_error': '400'}]}, []))
    pipeline = MMKGCPipeline(ranking_engine=Engine())
    result = pipeline.finish(PreparedEntity(target(), 'image', {}, k, selection, reasoning, []),
                             generation_result=generation)
    rank = result['kg_aware_candidate_consistency_ranking']
    assert rank['repair_failures']
    assert rank['ranking_status'] == ('completed' if has_success else 'evaluation_failed')
    if has_success:
        assert result['final']['best_image_candidate_id']
