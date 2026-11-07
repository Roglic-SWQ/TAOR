"""Evidence provenance, optional coverage and executable semantic repair."""
from copy import deepcopy
import json

import pytest

from mmkgc.io.legacy_evidence import convert_saved_checks
from mmkgc.observation import (bind_observation, candidate_repair_operations, removal_targets,
    repair_is_eligible, repair_operations, validate_verification_targets)
from mmkgc.stages.candidate_planning import apply_fact_selection
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.generation import build_stage3_feedback, route_image_repairs
from test_candidate_fact_plan import source
from test_fact_observation import raw_observation


def selected_sources():
    knowledge = source()
    for index, modality in enumerate(('graph', 'text', 'image')):
        fact = knowledge['relation_facts'][index]
        fact['source_modalities'] = [modality]
        knowledge['evidence_catalog'][index]['modality'] = modality
    image = knowledge['relation_facts'][2]
    image['visual_observations'] = [{'basis': 'direct', 'scope': 'target',
        'feature_kind': 'entity_feature', 'claim_scope': 'depicted_instance',
        'uncertainty_scope': 'identity', 'uncertainty': 'Exact identity is unverified',
        'evidence_refs': image['evidence_refs']}]
    image['source_weight'] = .5
    selected = apply_fact_selection(knowledge, {'status': 'ready',
        'selected_fact_ids': [f['fact_id'] for f in knowledge['relation_facts'][:3]]}, 'image', 4)
    compiled, context = compile_plan_context(knowledge, selected, 'image')
    return knowledge, compiled['fine_grained_relation_selection']['candidates'][0], context


def test_sources_and_visual_uncertainty_survive_selection_and_observation():
    knowledge, candidate, context = selected_sources()
    facts = context['supported_facts']
    assert [f['source_modalities'] for f in facts] == [['graph'], ['text'], ['image']]
    assert facts[2]['source_weight'] == .5
    targets = [item for item in candidate['verification_targets'] if item['fact_id']]
    assert [t['source_modalities'] for t in targets] == [['graph'], ['text'], ['image']]
    observation = bind_observation(raw_observation(targets, status='supported'), targets, 'image', 'image_001')
    assert observation['fact_checks'][2]['visual_observations'] == facts[2]['visual_observations']
    assert knowledge['relation_facts'][2]['visual_observations'][0]['uncertainty']
    for value in (candidate['generation_context'], targets, context):
        serialized = json.dumps(value)
        assert 'must_include' not in serialized and 'should_include' not in serialized
        assert '"requirement"' not in serialized


@pytest.mark.parametrize('mode', ['text', 'image'])
def test_omission_even_at_high_confidence_does_not_become_an_edit(mode):
    _, candidate, _ = selected_sources()
    targets = candidate['verification_targets']
    observation = bind_observation(raw_observation(targets, status='missing', mode=mode,
                                                   confidence=.99), targets, mode, mode + '_001')
    assert not repair_operations(observation)


def test_legacy_conversion_validates_facts_and_does_not_mutate_saved_input():
    _, candidate, _ = selected_sources()
    expected = candidate['verification_targets']
    original = {'verification_targets': [{**{k: deepcopy(v) for k, v in t.items()
        if k not in ('source_modalities', 'visual_observations')}, 'requirement': 'must_include'} for t in expected]}
    saved = deepcopy(original)
    converted = deepcopy(saved)
    convert_saved_checks(converted, expected)
    assert saved == original
    assert converted['verification_targets'] == expected
    broken = deepcopy(saved)
    broken['verification_targets'][0]['evidence_refs'] = ['foreign:source']
    with pytest.raises(ValueError, match='original fact plan'):
        convert_saved_checks(broken, expected)
    with pytest.raises(ValueError, match='boundary'):
        validate_verification_targets(saved['verification_targets'])


def semantic_candidate():
    _, original, _ = selected_sources()
    original.update(candidate_id='image_001', target_modality='image', positive_prompt='An abstract cube.',
        generated_content={'type': 'image', 'value': 'An abstract cube.'},
        hard_gate={'decision': 'reject', 'hard_violations': ['identity_mismatch']},
        judge_rubrics={'rubric_scores': {'entity_semantics': {'score': 0,
            'reason': 'Generic geometry does not depict the target entity.', 'evidence': ['pixel:a cube']}}})
    original['observation'] = bind_observation(raw_observation(original['verification_targets'], status='uncertain'),
                                             original['verification_targets'], 'image', 'image_001')
    return original


def test_semantic_rejection_creates_one_scoped_prompt_review_task():
    original = semantic_candidate()
    ranking = {'image_ranking': {'ranked_candidates': [], 'rejected_candidates': [original],
                                'best_candidate_id': None}}
    feedback = build_stage3_feedback(ranking, attempt=1, generation={'image_candidates': [original]})
    assert feedback['repair_needed']
    operation = feedback['candidate_feedback'][0]['operations'][0]
    assert operation['action'] == 'reframe'
    assert set(operation['evidence_refs']) == {r for t in original['verification_targets'] for r in t['evidence_refs']}
    def reviewer(candidate, operations):
        return {'observation': {'fact_checks': [{'check_id': 'semantic:entity', 'status': 'missing',
                                                'confidence': .95, 'verification_mode': 'direct'}]}}
    routed, _ = route_image_repairs(feedback, reviewer)
    assert routed['candidate_feedback'][0]['repair_route'] == 'edit_prompt'
    assert removal_targets([operation])[0]['issue_id'] == 'semantic:entity'


def test_semantic_repair_requires_explicit_resolution_and_preserves_source_metadata():
    original = semantic_candidate()
    repaired = deepcopy(original)
    repaired['hard_gate'] = {'decision': 'pass'}
    operation = candidate_repair_operations(original)[0]
    repaired['observation']['repair_checks'] = [{'issue_id': 'semantic:entity',
        'claim': operation['observed'], 'status': 'uncertain', 'confidence': .99}]
    assert not repair_is_eligible(original, repaired)
    repaired['observation']['repair_checks'][0]['status'] = 'resolved'
    assert repair_is_eligible(original, repaired)
    repaired['observation']['fact_checks'][1]['source_modalities'] = ['image']
    assert not repair_is_eligible(original, repaired)
