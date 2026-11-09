"""Retry unresolved entities with feedback, new seeds and bounded durable rounds."""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, wait
from collections import Counter
from pathlib import Path
import hashlib
import json
import time

from mmkgc.shared_models import SharedLocalModels
from scripts import run_wn18rr_completion_batch as batch
from scripts import run_wn18rr_parallel_completion as parallel

STRATEGIES = [
    'Replan around one defining semantic feature. Change the failed composition; prefer a clear concrete illustrative scene over generic atmosphere.',
    'Use a simple illustration of the intended sense. For motion or change, a short wordless before/after sequence or visible motion cues may help. Avoid the other meaning of a polysemous label.',
    'Use a minimal educational illustration or conceptual diagram grounded in the definition; make the defining distinction visible rather than repeating incidental objects.',
]


def prepare(root, source, *, max_rounds=3, workers=128):
    root, source = Path(root).resolve(), Path(source).resolve()
    parallel.validate_workers(workers)
    if type(max_rounds) is not int or not 1 <= max_rounds <= 5:
        raise ValueError('max_rounds must be in [1, 5]')
    batch.validate(source, check_sources=False)
    summary_path = source / ('combined-summary.json' if (source/'combined-summary.json').exists() else 'summary.json')
    baseline = batch.read(summary_path)
    scope_path = source / ('full-scope-manifest.json' if (source/'full-scope-manifest.json').exists() else 'manifest.json')
    scope = batch.read(scope_path)
    by_key = {row['entity_key']: row for row in baseline['rows']}
    if len(by_key) != len(scope) or set(by_key) != {row['entity_key'] for row in scope}:
        raise ValueError('Baseline summary differs from the frozen full entity scope')
    if any(row['status'] in {'pending','running'} for row in by_key.values()):
        raise ValueError('Retry requires a completed source scope')
    unresolved = [row for row in scope if by_key[row['entity_key']]['status'] != 'exported']
    root.mkdir(parents=True, exist_ok=False)
    policy = {'max_rounds':max_rounds,'workers':workers,'draws_per_round':3,'engineering_attempts_per_round':2,
              'seed_namespace':int(hashlib.sha256(str(root).encode()).hexdigest()[:8],16),
              'source_root':str(source),'source_summary_sha256':batch.digest(summary_path),
              'initial_successes':sum(row['status']=='exported' for row in by_key.values()),
              'initial_unresolved':len(unresolved),'grade_policy':'Existing Stage4 and Reflection; successful exports are immutable',
              'retry_prompt_model':'gpt-6-luna','stage1_model':'Qwen/Qwen3.5-9B'}
    batch.save(root/'baseline.json',baseline)
    batch.save(root/'full-scope-manifest.json',scope)
    batch.save(root/'manifest.json',unresolved)
    batch.save(root/'retry-policy.json',policy)
    binding=parallel.source_binding()
    batch.save(root/'binding.json',{**binding,'manifest_sha256':batch.digest(root/'manifest.json'),
        'files':{**binding['files'],**{str(root/name):batch.digest(root/name) for name in
                 ['baseline.json','full-scope-manifest.json','retry-policy.json']}}})
    snapshot(root)
    return policy


def feedback(row, round_number):
    result={'round':round_number,'previous_status':row['status'],'previous_error':row.get('error'),
            'previous_completion_reason':row.get('completion_reason'),
            'rendering_strategy':STRATEGIES[min(round_number-1,len(STRATEGIES)-1)],
            'provenance':'prior_generated_candidate_diagnostics_not_source_facts','judge_feedback':[]}
    entity=Path(row['entity_dir']) if row.get('entity_dir') else None
    if entity and (entity/'4_kg_aware_candidate_consistency_ranking.json').exists():
        ranking=batch.read(entity/'4_kg_aware_candidate_consistency_ranking.json')['data']['image_ranking']
        for candidate in ranking.get('ranked_candidates',[])[:3]:
            rubrics=candidate.get('judge_rubrics',{}).get('rubric_scores',{})
            result['judge_feedback'].append({
                'candidate_id':candidate['candidate_id'],'previous_prompt':candidate.get('positive_prompt','')[:1200],
                'gate':candidate.get('hard_gate',candidate.get('gate',{})),
                'entity_semantics':rubrics.get('entity_semantics'),
                'repair_suggestions':candidate.get('judge_rubrics',{}).get('repair_suggestions'),
                'ranking_reason':candidate.get('ranking_reason')})
    return result


def snapshot(root):
    root=Path(root)
    baseline=batch.read(root/'baseline.json')
    rows={row['entity_key']:dict(row) for row in baseline['rows']}
    if (root/'combined-summary.json').exists():
        for row in batch.read(root/'combined-summary.json')['rows']:
            if row['status']=='exported':rows[row['entity_key']]=row
    completed_rounds=[]
    for folder in sorted((root/'rounds').glob('*')) if (root/'rounds').exists() else []:
        if not (folder/'retry-configured.json').exists():continue
        try:current=batch.read(folder/'summary.json')
        except (FileNotFoundError,json.JSONDecodeError):
            recovered=[]
            for shard in (folder/'workers').iterdir():
                for item in batch.read(shard/'manifest.json'):
                    state=shard/'state'/f"{item['missing_mode']}-{item['entity_key']}.json"
                    try:recovered.append(batch.read(state))
                    except FileNotFoundError:recovered.append({**item,'status':'pending'})
            current={'rows':recovered}
        for row in current['rows']:
            if rows[row['entity_key']]['status']!='exported':
                rows[row['entity_key']]={**row,'retry_round_root':str(folder)}
        if (folder/'finished.json').exists():completed_rounds.append(folder.name)
    counts=dict(Counter(row['status'] for row in rows.values()))
    summary={'expected_entities':len(rows),'counts':counts,
        'initial_successes':sum(row['status']=='exported' for row in baseline['rows']),
        'new_exports':counts.get('exported',0)-sum(row['status']=='exported' for row in baseline['rows']),
        'completed_retry_rounds':completed_rounds,'remaining':sum(row['status']!='exported' for row in rows.values()),
        'updated_at_unix':time.time(),'rows':list(rows.values())}
    batch.save(root/'combined-summary.json',summary)
    # Account for actual historical draw calls; new round ledgers never replace them.
    policy=batch.read(root/'retry-policy.json')
    actual_draws=Counter()
    source=Path(policy['source_root'])
    if (source/'cumulative-draws.json').exists():
        actual_draws.update(batch.read(source/'cumulative-draws.json'))
    else:
        for ledger in source.glob('workers/*/sdxl-attempts.json'):
            for row in batch.read(ledger):actual_draws[row['entity_key']]+=1
    for ledger in (root/'rounds').glob('*/workers/*/sdxl-attempts.json') if (root/'rounds').exists() else []:
        for row in batch.read(ledger):actual_draws[row['entity_key']]+=1
    wanted={row['entity_key'] for row in batch.read(root/'manifest.json')}
    batch.save(root/'cumulative-draws.json', {key:actual_draws[key] for key in sorted(wanted)})
    return summary


def configure_round(root, round_root, number, previous):
    policy=batch.read(root/'retry-policy.json')
    contexts={row['entity_key']:feedback(row,number) for row in previous['rows'] if row['status']!='exported'}
    for shard in (round_root/'workers').iterdir():
        settings=batch.read(shard/'runtime-settings.json')
        settings.update(max_selected_facts=2 if number==1 else 1)
        settings['model_config']='configs/gpt6-luna-retry-image.yaml'
        batch.save(shard/'runtime-settings.json',settings)
        batch.save(shard/'retry-policy.json',{'round':number,'seed_namespace':policy['seed_namespace']})
        keys={row['entity_key'] for row in batch.read(shard/'manifest.json')}
        batch.save(shard/'retry-contexts.json',{key:contexts[key] for key in keys})
        binding=batch.read(shard/'binding.json')
        batch.save(shard/'binding.json',{**binding,'files':{**binding['files'],**{
            str(shard/name):batch.digest(shard/name) for name in
            ['runtime-settings.json','retry-policy.json','retry-contexts.json']}}})
    config=batch.read(round_root/'orchestration.json')
    batch.save(round_root/'orchestration.json',{**config,'retry_round':number,
        'image_prompt_model':'gpt-6-luna'})
    binding=batch.read(round_root/'binding.json')
    batch.save(round_root/'binding.json',{**binding,'files':{**binding['files'],
        str(round_root/'orchestration.json'):batch.digest(round_root/'orchestration.json')}})
    batch.save(round_root/'entity-concurrency.json',{'active_entities':config['workers']})
    batch.save(round_root/'retry-configured.json',{'round':number,'seed_namespace':policy['seed_namespace']})


def _run(root):
    root=Path(root).resolve()
    batch.validate(root)
    policy=batch.read(root/'retry-policy.json')
    shared=SharedLocalModels()
    for number in range(1,policy['max_rounds']+1):
        batch.validate(root)
        previous=snapshot(root)
        if not previous['remaining']:break
        round_root=root/'rounds'/f'{number:02d}'
        if (round_root/'finished.json').exists():continue
        if not (round_root/'retry-configured.json').exists():
            if round_root.exists():
                archive=root/'interrupted-preparations'/f'{number:02d}-{time.time_ns()}'
                archive.parent.mkdir(parents=True,exist_ok=True)
                round_root.rename(archive)
            keys=[row['entity_key'] for row in previous['rows'] if row['status']!='exported']
            parallel.prepare(round_root,workers=min(policy['workers'],len(keys)),keys=keys)
            configure_round(root,round_root,number,previous)
        batch.save(root/'phase.json',{'phase':'retrying','round':number,'updated_at_unix':time.time()})
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(parallel.run,round_root,shared_local_models=shared)
            while not future.done():
                wait([future],timeout=20)
                summary=snapshot(root)
                print(json.dumps({'event':'retry_progress','round':number,'counts':summary['counts'],
                                  'new_exports':summary['new_exports'],'remaining':summary['remaining']}),flush=True)
            future.result()
        snapshot(root)
    summary=snapshot(root)
    batch.save(root/'finished.json',{key:value for key,value in summary.items() if key!='rows'})
    batch.save(root/'phase.json',{'phase':'completed' if not summary['remaining'] else 'retry_rounds_exhausted',
                                'updated_at_unix':time.time(),'remaining':summary['remaining']})
    return summary


def run(root):
    root=Path(root).resolve()
    with (root/'.retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--max-rounds',type=int,default=3)
    p.add_argument('--workers',type=int,default=128)
    a=p.parse_args()
    if a.prepare:
        if not a.source:p.error('--source is required for preparation')
        print(json.dumps(prepare(a.root,a.source,max_rounds=a.max_rounds,workers=a.workers)),flush=True)
    else:run(a.root)


if __name__=='__main__':main()
