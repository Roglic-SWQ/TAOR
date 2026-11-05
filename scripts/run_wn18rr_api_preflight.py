"""No-GPU API preflight: upstream/text Reflect, image prompt, frozen real-image Judge.

This omits the local relevance reranker and local neural scorers, and does not
generate new pixels. Scores are contract diagnostics, not full fusion benchmarks.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import datetime, timezone
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib import request

from dotenv import load_dotenv
import yaml
import httpx

from mmkgc.adapters.qwen import (
    QwenLocalSemanticStagePlanner, QwenLocalTextGenerator, QwenLocalImagePromptGenerator,
    _coerce_qwen_config,
)
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.data import build_dataset_store
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter, _safe_output
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.image_input import ImageInput
from mmkgc.ranking.factory import _OpenAICompatibleCompletionClient
from mmkgc.ranking.profiles import Stage4Profile, LocalScorerConfig, GPT4OMiniJudgeConfig
from mmkgc.ranking.scorers.gpt4omini_judge import GPT4OMiniRankingJudge
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.generation import _model_generation_context
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.triples import fact_text

ROOT = Path(__file__).resolve().parents[1]


def pace_api_requests(host, interval_ms):
    """Wrap real transports for this process/host; leave request/response data intact."""
    interval = max(0, float(interval_ms)) / 1000
    original_sync, original_async = request.urlopen, httpx.AsyncClient.request
    lock, next_start = threading.Lock(), [0.0]
    def delay(url):
        if urlsplit(str(url)).hostname != host:
            return 0
        with lock:
            now = time.monotonic()
            start = max(now, next_start[0])
            next_start[0] = start + interval
            return max(0, start-now)
    def sync(req, *args, **kwargs):
        wait = delay(req.full_url if hasattr(req, 'full_url') else req)
        if wait:
            time.sleep(wait)
        return original_sync(req, *args, **kwargs)
    async def asynchronous(client, method, url, *args, **kwargs):
        wait = delay(url)
        if wait:
            await asyncio.sleep(wait)
        return await original_async(client, method, url, *args, **kwargs)
    request.urlopen, httpx.AsyncClient.request = sync, asynchronous
    def restore():
        request.urlopen, httpx.AsyncClient.request = original_sync, original_async
    return restore


def resolve_configuration(path, model=None):
    """Report missing names together; never return credential values in status."""
    data = yaml.safe_load(Path(path).read_text())
    if model:
        data['lmm']['model_name'] = model
    lmm = data['lmm']
    missing = []
    for field in ('model_name', 'base_url', 'api_key'):
        literal = str(lmm.get(field) or '').strip()
        variable = str(lmm.get(field+'_env') or '').strip()
        if not literal and not (variable and os.environ.get(variable, '').strip()):
            missing.append(variable or 'lmm.'+field)
    if missing:
        return None, {'status': 'missing_configuration', 'missing': missing}
    config = _coerce_qwen_config(data, path)
    url = urlsplit(config.base_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query:
        return None, {'status': 'invalid_configuration', 'field': 'base_url'}
    return config, {'status': 'configuration_ready', 'model': config.model_name,
                    'api_host': url.hostname, 'api_key_configured': True}


def api_ranker(config, timeout):
    disabled = LocalScorerConfig(required=False, enabled=False, model_dir=None, device='cpu')
    profile = Stage4Profile('api_contract_preflight_only', disabled, disabled, disabled, disabled,
        GPT4OMiniJudgeConfig(True, ROOT/'configs/openai-gpt4o-mini-judge.yaml', timeout))
    judge = GPT4OMiniRankingJudge(_OpenAICompatibleCompletionClient(config, timeout_seconds=timeout),
        model_name=config.model_name, max_retries=0, use_rubrics=True)
    return ModelFusionRankingEngine(profile=profile, scorers={'gpt4omini_judge': judge}), judge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--model-config', type=Path, default=ROOT/'configs/gpt4omini-mainline.yaml')
    parser.add_argument('--model', help='Override model name without writing credentials to config')
    parser.add_argument('--check-only', action='store_true', help='Check configuration; make no API calls')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT/'datasets/wn18rr-import-20261106/WN18RR')
    parser.add_argument('--source-root', type=Path,
        default=ROOT/'outputs/wn18rr-visual-policy-20261106/attempt-r4/runs')
    parser.add_argument('--entity-key', default='02436645')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    load_dotenv(ROOT/'.env', override=False)
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    config, status = resolve_configuration(args.model_config, args.model or os.environ.get('OPENAI_MODEL'))
    print(json.dumps(status), flush=True)
    if config is None:
        raise SystemExit(2)
    if args.check_only:
        return
    output = args.output_root or ROOT/'outputs'/('api-preflight-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    summary = {**status, 'status': 'running', 'local_gpu_inference': False,
        'local_reranker': 'not_run', 'local_neural_scorers': 'not_run',
        'image_generation': 'not_run; frozen real SDXL image used only for Judge transport/contract',
        'quality_acceptance': 'requires_source_and_output_review', 'steps': {}}
    def save(name, value):
        text = json.dumps(_safe_output(value), ensure_ascii=False, indent=2)
        if config.api_key:
            text = text.replace(config.api_key, '<redacted>')
        (output/name).write_text(text)
    save('execution-sha256.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('mmkgc', 'scripts') for p in sorted((ROOT/folder).rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.txt')})
    store = build_dataset_store('wn18rr', args.dataset_dir, graph_split='train')
    planner = QwenLocalSemanticStagePlanner(config, timeout_seconds=args.timeout,
        dataset_root=args.dataset_dir, knowledge_stage_timeout_seconds=180)
    selector = OpenAICompatibleRelationPlanner(config, timeout_seconds=args.timeout, max_selected_facts=8)
    text_generator = QwenLocalTextGenerator(config, timeout_seconds=args.timeout)
    prompt_generator = QwenLocalImagePromptGenerator(config, timeout_seconds=args.timeout)
    ranker, judge = api_ranker(config, args.timeout)
    prompt_call = None
    restore_transport = pace_api_requests(status['api_host'], os.environ.get('OPENAI_MIN_REQUEST_INTERVAL_MS', '0'))
    try:
        record = store.build_record(store.entity_uri_to_id[args.entity_key], 'text')
        coarse = coarse_relation_filtering(record, 'text', catalogue_only=True)
        knowledge = planner.build_knowledge(record, 'text', coarse)
        save('text-knowledge.json', knowledge)
        selected = selector.plan(record, 'text', knowledge, 1)
        selected, reasoning = compile_plan_context(knowledge, selected, 'text')
        save('text-selection.json', selected)
        prepared = PreparedEntity(record, 'text', coarse, knowledge, selected, reasoning,
                                  [*planner.calls, *selector.calls])
        pipeline = MMKGCPipeline(candidate_budget=1, text_generator=text_generator,
                                ranking_engine=ranker, dataset_root=args.dataset_dir)
        result = pipeline.finish(prepared)
        writer = RunWriter(output, 'text-api', {'scope': 'api_contract_preflight_only',
            'model': config.model_name, 'api_host': status['api_host'], 'local_gpu_inference': False})
        writer.write_entity(result)
        writer.write_summary()
        summary['steps']['text_loop'] = result['final']
        if result['final']['ranking_status'] != 'completed':
            raise RuntimeError('API text loop did not complete evaluation')

        image_record = store.build_record(store.entity_uri_to_id[args.entity_key], 'image')
        image_coarse = coarse_relation_filtering(image_record, 'image', catalogue_only=True)
        image_knowledge = planner.build_knowledge(image_record, 'image', image_coarse)
        image_selection = selector.plan(image_record, 'image', image_knowledge, 1)
        image_selection, image_reasoning = compile_plan_context(image_knowledge, image_selection, 'image')
        plans = image_selection['fine_grained_relation_selection']['candidates']
        if not plans:
            raise RuntimeError('API image planner returned insufficient evidence')
        plan = plans[0]
        context = _model_generation_context(image_knowledge, plan, image_reasoning, 'image', '', '')
        prompt = prompt_generator.generate_with_context(image_record.label,
            image_knowledge['entity_profile']['entity_type'], plan['relation'], '',
            [fact_text(fact) for fact in plan['source_facts']], context=context)
        prompt_call = prompt_generator.consume_last_call(image_record.entity_key, 'api_prompt_001')
        save('image-prompt-component.json', {'status': 'prompt_only', 'prompt': prompt,
            'knowledge': image_knowledge, 'selection': image_selection})
        summary['steps']['image_prompt_component'] = 'completed_without_pixel_generation'

        source = args.source_root/'image-qwen7b-sdxl'/'entities'/args.entity_key
        frozen, generation = load_saved_run(source, image_record, 1, missing_mode='image')
        artifact = generation['generated_image_artifacts'][0]
        context = build_ranking_context(image_record.entity_key, image_record.label, 'image',
            image_record.existing_modalities, ranking_knowledge(frozen.knowledge, frozen.reasoning),
            frozen.reasoning, generation, {artifact.candidate_id: {'image_bytes': artifact.image_bytes, 'media_type': 'image/png'}})
        candidate = context.image_candidates[0]
        image_input = ImageInput.from_bytes(entity_key=image_record.entity_key,
            candidate_id=artifact.candidate_id, image_bytes=artifact.image_bytes)
        judgment = judge.judge_image(context, dict(candidate), image_input=image_input)
        save('frozen-image-judge.json', {'source_run': str(source), 'score': judgment.score,
                                       'observation': judgment.observation})
        summary['steps']['frozen_image_judge'] = judgment.score.status
        summary['status'] = 'contract_checks_completed_requires_quality_review'
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        if hasattr(exc, 'details'):
            save('failure-details.json', exc.details)
    finally:
        restore_transport()
        save('summary.json', summary)
        save('model-calls.json', {'knowledge': planner.calls, 'selection': selector.calls,
            'pending_text_generation': text_generator.consume_last_call(args.entity_key, 'api_text_failure'),
            'image_prompt': prompt_call or prompt_generator.consume_last_call(args.entity_key, 'api_prompt_failure')})
    print(json.dumps({'status': summary['status'], 'output_root': str(output)}), flush=True)
    if summary['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
