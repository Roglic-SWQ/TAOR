from copy import deepcopy
import json
import pytest

from mmkgc.adapters.visual_projection import apply, prompt
from mmkgc.adapters.qwen import QwenConfig, load_qwen_config


def payload():
    return {'features': [{'attribute': 'support', 'value': 'vertical pole with orange padding',
        'kind': 'entity_feature', 'basis': 'direct', 'uncertainty': '', 'evidence_refs': ['image:2']}],
        'uncertain_aspects': [{'aspect': 'identity', 'reason': 'Not confirmed'}]}


def decision(**kwargs):
    return {'decisions': [{'source_index': 0, 'keep': True, 'attribute': 'support',
        'value': 'vertical pole', 'reason': 'Retain structure, omit incidental padding', **kwargs}]}


def test_abstraction_keeps_sources_and_identity_uncertainty_without_mutation():
    source = payload(); before = deepcopy(source)
    result = apply(source, decision())
    assert source == before
    assert result['features'][0]['value'] == 'vertical pole'
    assert result['features'][0]['evidence_refs'] == ['image:2']
    assert result['uncertain_aspects'] == before['uncertain_aspects']


def test_review_can_abstain_and_cannot_promote_scene_details():
    source = payload(); source['features'][0]['kind'] = 'context'
    assert not apply(source, decision(keep=False, attribute='', value=''))['features']
    with pytest.raises(ValueError, match='Cannot promote'):
        apply(source, decision())


def test_qualified_source_cannot_be_rewritten_as_certain():
    source = payload(); source['features'][0].update(basis='inferred', uncertainty='Structure is unclear')
    with pytest.raises(ValueError, match='qualified'):
        apply(source, decision())
    result = apply(source, decision(value=source['features'][0]['value']))
    assert result['features'][0]['basis'] == 'inferred'
    assert result['features'][0]['uncertainty'] == 'Structure is unclear'


def test_reviews_cannot_reuse_another_features_sources_or_skip_ids():
    source = payload(); source['features'].append(deepcopy(source['features'][0]))
    duplicate = decision()['decisions'] * 2
    with pytest.raises(ValueError, match='exactly once'):
        apply(source, {'decisions': duplicate})


def test_explicit_opt_in_and_roles_are_not_rescued_by_identity_disclaimers():
    assert not QwenConfig('test', 'http://localhost', 'EMPTY').abstract_visual_features
    assert load_qwen_config('configs/qwen35-9b-hybrid.yaml').abstract_visual_features
    with pytest.raises(ValueError):
        QwenConfig('test', 'http://localhost', 'EMPTY', abstract_visual_features=True)
    assert 'identity uncertainty is acknowledged' in prompt('role', payload())


@pytest.mark.parametrize('modality', ['text', 'image'])
def test_semantic_rejection_is_independent_of_factual_accuracy(modality):
    from mmkgc.ranking.scorers.gpt4omini_judge import parse_text_judge_response, parse_image_judge_response
    dimensions = ('entity_semantics', 'verified_fact_consistency',
                  'semantic_coverage' if modality == 'text' else 'visual_semantic_coverage',
                  'textual_usability' if modality == 'text' else 'visual_usability')
    prefix = 'text:' if modality == 'text' else 'pixel:'
    response = {'scores': {d: {'score': 0 if d == 'entity_semantics' else 4,
        'reason': 'Only portrait details are available' if d == 'entity_semantics' else 'Faithful to selected observations',
        'evidence': [prefix + 'The person has brown eyes and is smiling.']} for d in dimensions},
        'observation': {'fact_checks': [], 'unsupported_claims': []},
        'unverified_additions': [], 'repair_suggestions': []}
    parser = parse_text_judge_response if modality == 'text' else parse_image_judge_response
    assert parser(json.dumps(response)).gate.decision == 'reject'


@pytest.mark.parametrize('reply,finish,code', [('', 'stop', 'empty_response'), ('{}', 'length', 'output_token_limit')])
def test_incomplete_projection_is_not_repaired_as_json(reply, finish, code):
    import time
    from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner
    from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT, Stage1CompletionError
    client = QwenLocalSemanticStagePlanner(load_qwen_config('configs/qwen35-9b-hybrid.yaml'),
        visual_verifier_config=QwenConfig('reviewer', 'http://example.invalid', 'EMPTY'))
    calls = []
    def complete(prompt):
        calls.append(prompt)
        KNOWLEDGE_REQUEST_CONTEXT.get()['call']['network_attempts'].append({'finish_reason': finish})
        return reply
    client.visual_verifier._complete_prompt = complete
    with pytest.raises(Stage1CompletionError) as error:
        client._project_visual_payload('object', payload(), time.perf_counter()+60)
    assert error.value.code == code and len(calls) == 1
    assert client.calls[-1]['contract_repair_retries'] == 0
    assert client.calls[-1]['completion_failure'] == code
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_source_review_uses_original_text_and_restores_reviewers_mode(monkeypatch):
    from types import SimpleNamespace
    from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner
    from mmkgc.stages.semantic import StrictJsonStagePlanner
    client = QwenLocalSemanticStagePlanner(load_qwen_config('configs/qwen35-9b-hybrid.yaml'),
        visual_verifier_config=QwenConfig('reviewer', 'http://example.invalid', 'EMPTY'))
    record = SimpleNamespace(label='fish', description='fish, catch or try to catch fish', existing_modalities={}, image_paths=[])
    original = {'relation_facts': [{'relation': 'is', 'object': 'food', 'source_modalities': ['text']}]}
    corrected = {'relation_facts': [{'subject': 'fish', 'relation': 'means', 'object': 'catch or try to catch fish', 'source_modalities': ['text']}]}
    def build(planner, record, mode, coarse):
        if planner is client:
            return original
        assert planner.source_bound_text_extraction and record.description.startswith('fish, catch')
        return corrected
    monkeypatch.setattr(StrictJsonStagePlanner, 'build_knowledge', build)
    assert client.build_knowledge(record, 'image', {}) == corrected
    assert not client.visual_verifier.source_bound_text_extraction
    assert client.calls[-1]['proposed_text_facts'][0]['object'] == 'food'
    assert client.calls[-1]['reviewed_text_facts'][0]['object'] == 'catch or try to catch fish'
    corrected['relation_facts'][0]['object'] = 'food'
    with pytest.raises(ValueError, match='lexical definition'):
        client.build_knowledge(record, 'image', {})
    assert not client.visual_verifier.source_bound_text_extraction


@pytest.mark.parametrize('mode', ['image', 'text'])
def test_judge_preserves_available_source_and_masks_target_modality(mode):
    from mmkgc.ranking.context import build_ranking_context
    context = build_ranking_context('Q0', 'entity', mode,
        {'text': 'Provided source text', 'image': '/private/source-or-target.png'}, {}, {},
        {'relation_alternation_generation': {'text_candidates': [], 'image_candidates': []}}, {})
    assert context.existing_modalities[mode] is None
    if mode == 'image':
        assert context.to_prompt_payload()['existing_modalities'] == {'text': 'Provided source text'}
    else:
        assert context.to_prompt_payload()['existing_modalities'] == {}
        assert context.to_prompt_payload()['has_existing_image']
