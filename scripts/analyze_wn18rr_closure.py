"""Read-only causal audit of the frozen WN18RR case; no model calls or artifact edits.

Counterfactual labels below are synthetic control inputs, not model judgments.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mmkgc.observation as observation
from mmkgc.pipeline import _needs_stage4_regeneration
from mmkgc.stages.generation import build_stage3_feedback

SOURCE = Path('outputs/wn18rr-onboarding-20261106/attempt-r3/runs')
CLOSURE = Path('outputs/wn18rr-stage4-20261106/attempt-r4/closure')
OUTPUT = Path('outputs/wn18rr-closure-cause-20261106')


def read_stage(root, name):
    return json.loads((root/name).read_text())['data']


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tracked = [p for root in [SOURCE, CLOSURE] for p in root.rglob('*') if p.is_file()]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    source = SOURCE/'text-qwen7b/entities/02436645'
    knowledge = read_stage(source, '1_knowledge_modeling.json')
    selected = read_stage(source, '2_fine_grained_relation_selection.json')
    facts = knowledge.get('triples', knowledge.get('relation_facts'))
    rank_root = CLOSURE/'text/entities/02436645'
    ranking = read_stage(rank_root, '4_kg_aware_candidate_consistency_ranking.json')
    generation = read_stage(rank_root, '3_relation_alternation_generation.json')
    candidate = ranking['text_ranking']['ranked_candidates'][0]
    observed = candidate['observation']
    assert observation.repair_operations(observed) == []
    threshold_results = {}
    old_threshold = observation.REPAIR_CONFIDENCE
    try:
        for threshold in [0.0, 0.5, 0.8, 1.0]:
            observation.REPAIR_CONFIDENCE = threshold
            feedback = build_stage3_feedback(ranking, attempt=1, generation=generation)
            threshold_results[str(threshold)] = {
                'operation_count': len(observation.repair_operations(observed)),
                'regeneration': _needs_stage4_regeneration(ranking, feedback),
            }
    finally:
        observation.REPAIR_CONFIDENCE = old_threshold
    assert not any(row['regeneration'] for row in threshold_results.values())

    # Diagnostic only: if the Judge called this selected fact contradicted,
    # the existing repair would target the SAME selected (bad) expected fact.
    counterfactual = deepcopy(ranking)
    cf_candidate = counterfactual['text_ranking']['ranked_candidates'][0]
    cf_check = next(x for x in cf_candidate['observation']['fact_checks'] if x['fact_id'] == 'mf:0002')
    cf_check['status'] = 'contradicted'
    cf = build_stage3_feedback(counterfactual, attempt=1, generation=generation)
    assert _needs_stage4_regeneration(counterfactual, cf)
    operation = cf['candidate_feedback'][0]['operations'][0]
    assert operation['action'] == 'replace' and 'rodent' in operation['expected']

    # A score alone is deliberately not a factual edit instruction.
    low_score = deepcopy(ranking)
    low_score['text_ranking']['ranked_candidates'][0]['final_score'] = 0.0
    low_feedback = build_stage3_feedback(low_score, attempt=1, generation=generation)
    assert not _needs_stage4_regeneration(low_score, low_feedback)

    # Separate synthetic control: a normal omission can be repaired, but an
    # originally supported wrong reference remains locked by acceptance rules.
    locked_before = deepcopy(candidate)
    locked_before['hard_gate'] = {'decision': 'pass'}
    next(x for x in locked_before['observation']['fact_checks'] if x['fact_id'] is None)['status'] = 'missing'
    repaired_control = deepcopy(locked_before)
    next(x for x in repaired_control['observation']['fact_checks'] if x['fact_id'] is None)['status'] = 'supported'
    assert observation.repair_is_eligible(locked_before, repaired_control)
    source_retraction = deepcopy(repaired_control)
    retract = next(x for x in source_retraction['observation']['fact_checks'] if x['fact_id']=='mf:0002')
    retract.update(status='unverifiable', verification_mode='unverifiable')
    assert not observation.repair_is_eligible(locked_before, source_retraction)

    image_root = CLOSURE/'image/entities/02436645'
    image_rank = read_stage(image_root, '4_kg_aware_candidate_consistency_ranking.json')
    image_candidate = image_rank['image_ranking']['ranked_candidates'][0]
    image_targets = deepcopy(image_candidate['verification_targets'])
    original_targets = deepcopy(image_targets)
    next(x for x in image_targets if x['fact_id'] == 'mf:0001')['verification_mode'] = 'unverifiable'
    raw = {'fact_checks': [{k: x[k] for k in ['check_id','status','verification_mode','observed','evidence','confidence','reason']}
                          for x in image_candidate['observation']['fact_checks']], 'unsupported_claims': []}
    observation.bind_observation(raw, original_targets, 'image', 'image_001')
    try:
        observation.bind_observation(raw, image_targets, 'image', 'image_001')
    except ValueError as exc:
        mode_guard = str(exc)
    else:
        raise AssertionError('Explicit unverifiable target did not reject direct certification')

    calls = [json.loads(line) for line in (rank_root/'ranking_model_calls.jsonl').read_text().splitlines()]
    scores = {x['scorer']:x['normalized_score'] for x in calls}
    payload = json.loads((OUTPUT/'text-judge-payload.json').read_text())
    bundle = json.dumps(payload['EVIDENCE_BUNDLE'])
    report = {
        'case': '02436645', 'audit_type': 'offline trace plus synthetic routing counterfactuals',
        'description': knowledge['image_modeling']['description'],
        'extracted_uncertain_aspects': knowledge.get('uncertain_aspects'),
        'facts': [{key: f.get(key) for key in ['fact_id','subject','relation','object','source_modalities','relevance_score']}
                  for f in facts],
        'selected_fact_ids': selected['selected_fact_ids'],
        'judge_input': {'has_existing_image_flag':payload['EVIDENCE_BUNDLE']['has_existing_image'],
                        'original_caption_hedge_retained': 'appears to be a type of rodent or marsupial' in bundle,
                        'image_bytes_in_payload': 'data:image/' in bundle,
                        'fact_truth_audit_distinct_from_candidate_checks': False},
        'actual_checks': observed['fact_checks'], 'actual_operations': [], 'scores': scores,
        'threshold_counterfactuals': threshold_results,
        'zero_aggregate_score_still_no_fact_repair': not low_feedback['repair_needed'],
        'contradicted_label_counterfactual': {'clearly_synthetic_not_a_model_result': True,
            'would_trigger': cf['repair_needed'], 'replacement_operation': operation},
        'reference_lock_counterfactual': {'clearly_synthetic_not_a_model_result': True,
            'ordinary_omission_repair_eligible': True, 'retract_originally_supported_bad_fact_eligible': False},
        'image_verifiability_counterfactual': {'clearly_synthetic_not_a_model_result': True,
            'actual_target_mode': next(x for x in original_targets if x['fact_id']=='mf:0001')['verification_mode'],
            'explicit_unverifiable_would_reject_actual_reply': mode_guard},
        'not_established': ['Causal effect of using the same judge weights', 'Whether a larger independent judge alone fixes the case',
                            'Whether source-fact revision followed by regeneration produces a correct candidate'],
    }
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    report['original_runs_unchanged'] = True
    (OUTPUT/'causal-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'thresholds':threshold_results,'replacement_expected':operation['expected'],
                      'image_guard':mode_guard,'scores':scores,'original_runs_unchanged':True},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
