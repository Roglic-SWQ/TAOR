"""Replay a recorded native failure with identical pixels/caption and a compact contract."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from mmkgc.adapters.image_knowledge import load_source_images, with_image_description
from mmkgc.adapters.json_contract import _payload_from_prompt
from mmkgc.adapters.media_knowledge import SourceMedia
from mmkgc.adapters.evidence_reranker import rerank_knowledge_relevance
from mmkgc.data import build_dataset_store
from mmkgc.defaults import build_default_mainline_runtime
from mmkgc.io.writer import RunWriter, _safe_output
from mmkgc.pipeline import PreparedEntity
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.modeling import build_knowledge_input
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.visual_evidence import ENTITY_VISUAL_POLICY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure-file',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=False)
    digest=hashlib.sha256(args.failure_file.read_bytes()).hexdigest()
    failed=json.loads(args.failure_file.read_text())[0]
    saved=failed['model_call'];view=_payload_from_prompt(saved['prompt'])
    assert failed['error_type']=='Stage1CompletionError' and saved['completion_failure']=='output_token_limit'
    dataset=Path('datasets/wn18rr-import-20261106/WN18RR')
    store=build_dataset_store('wn18rr',dataset,graph_split='train')
    record=store.build_record(store.entity_uri_to_id[failed['entity']['entity_key']],'text')
    coarse=coarse_relation_filtering(record,'text',catalogue_only=True)
    frozen=load_source_images(record,dataset);assert len(frozen)==1
    image,metadata=frozen[0]
    assert metadata['sha256']==saved['image_input_metadata']['sha256']
    caption=view['existing_modalities']['image_summary']
    inputs=with_image_description(build_knowledge_input(record,'text',coarse),caption,metadata,
                                  saved['image_description_model_call_ref'])
    inputs['visual_fact_policy']=ENTITY_VISUAL_POLICY
    # The failed extraction records only this admission view, not the whole gate response.
    admission=deepcopy(view['image_modeling']['entity_match'])
    assert admission['status']=='compatible' and admission['identity_verified'] is False
    admission.update(uncertainties=['Exact identity was not verified in the recorded admission view.'],
                     limitation_origin='program_derived_from_recorded_identity_verified_false')
    inputs['image_modeling']['entity_match']=deepcopy(admission)
    for source in inputs['evidence_catalog']:
        if source.get('modality')=='image':source['entity_match']=deepcopy(admission)
    runtime=build_default_mainline_runtime(missing_mode='text',candidate_budget=1,
        stage1_reranker_model_dir='/medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B',
        stage1_reranker_device='cuda:0',model_call_timeout_seconds=120,dataset_root=dataset,
        stage4_config='configs/stage4-qwen7b-local.yaml')
    pipeline=runtime.pipeline;planner=pipeline.knowledge_planner
    summary={'control':'same_recorded_image_and_caption_compact_transport',
        'source_failure_sha256':digest,'source_image_sha256':metadata['sha256'],
        'source_caption_sha256':hashlib.sha256(caption.encode()).hexdigest(),
        'original_generation_budget':saved['network_attempts'][0]['max_tokens'],
        'current_generation_budget':planner.config.max_token,
        'admission_provenance':'recorded exposed view only; full original gate response unavailable',
        'status':'running'}
    def save(name,value):
        (args.output_root/name).write_text(json.dumps(_safe_output(value),ensure_ascii=False,indent=2))
    save('control-inputs.json',summary)
    writer=RunWriter(args.output_root,'text-qwen7b',summary)
    try:
        sources=[SourceMedia('image',metadata,image=image)]
        context={'media_inputs':sources,'image_input_metadata':metadata,
                 'image_description_model_call_ref':saved['image_description_model_call_ref']}
        deadline=time.perf_counter()+180
        knowledge=planner._compact_visual_knowledge(inputs,context,deadline)
        save('proposed-knowledge.json',knowledge)
        knowledge=planner._verify_visual_features(knowledge,sources,record.entity_key,deadline)
        knowledge['uncertainties'].append({'aspect':'image identity','reason':admission['uncertainties'][0]})
        knowledge['stage1_v3']['uncertain_aspects']=deepcopy(knowledge['uncertainties'])
        save('checked-knowledge.json',knowledge)
        knowledge=rerank_knowledge_relevance(knowledge,pipeline.evidence_reranker,'text',
                                            fact_budget=pipeline.relevance_fact_budget)
        selection=pipeline.relation_planner.plan(record,'text',knowledge,1)
        selection,reasoning=compile_plan_context(knowledge,selection,'text')
        prepared=PreparedEntity(record,'text',coarse,knowledge,selection,reasoning,
            [*planner.calls,*pipeline.relation_planner.calls])
        result=pipeline.finish(prepared)
        writer.write_entity(result);writer.write_summary()
        summary.update(status='completed',final=result['final'])
    except Exception as exc:
        writer.record_failure(record,exc);writer.write_summary()
        summary.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    finally:
        save('model-calls.json',planner.calls);save('summary.json',summary)
        assert hashlib.sha256(args.failure_file.read_bytes()).hexdigest()==digest
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary['status']!='completed' or summary['final']['ranking_status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
