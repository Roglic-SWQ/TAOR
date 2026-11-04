"""Seed variants retain accepted evidence, provenance, and real SDXL RNG inputs."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from mmkgc.adapters.sdxl import SDXLConfig, SDXLImageGenerator
from mmkgc.io.writer import RunWriter, _stage_artifact
from mmkgc.stages.generation import relation_alternation_generation
from mmkgc.stages.presentation import ImageFocusValidationError
from test_shared_fact_selection import selected, target, Images
from test_image_prompt_diversity import FIRST, SECOND, Prompts


class NoisePipeline:
    """Small CPU stand-in that consumes the generator SDXL actually receives."""
    def __init__(self):
        self.seeds=[]
    def __call__(self, **kwargs):
        import torch
        self.seeds.append(kwargs['generator'].initial_seed())
        pixels=torch.randint(256,(8,8,3),generator=kwargs['generator'],dtype=torch.uint8)
        return type('Result',(),{'images':[Image.fromarray(pixels.numpy())]})()


def run(prompts, images=None, budget=3):
    knowledge,selection,reasoning=selected()
    before=deepcopy((knowledge,selection,reasoning))
    result=relation_alternation_generation(target(),'image',knowledge,reasoning,selection,
        None,prompts,images,candidate_budget=budget)
    assert (knowledge,selection,reasoning)==before
    return result


def test_reused_prompt_gets_different_reproducible_sdxl_seeds_and_persisted_provenance(tmp_path):
    pytest.importorskip('torch')
    pipeline=NoisePipeline()
    images=SDXLImageGenerator(SDXLConfig(model_dir=tmp_path,seed=123,device='cpu',width=8,height=8),
        pipeline_factory=lambda _:pipeline)
    result=run(Prompts([FIRST]*7),images)
    artifacts=result['generated_image_artifacts']
    assert pipeline.seeds==[124,125,126]
    assert len({a.image_bytes for a in artifacts})==3
    assert {a.metadata['positive_prompt'] for a in artifacts}=={FIRST}
    assert [a.metadata['image_candidate_kind'] for a in artifacts]==['distinct_prompt','seed_variant','seed_variant']
    assert [a.metadata['seed_variant_index'] for a in artifacts]==[0,1,2]
    for artifact in artifacts:
        assert artifact.model_call['prompt_source_candidate_id']=='image_001'
        assert artifact.metadata['prompt_source_candidate_id']=='image_001'
    second=run(Prompts([FIRST]*7),images)
    assert [a.image_bytes for a in second['generated_image_artifacts']]==[a.image_bytes for a in artifacts]
    stage=result['relation_alternation_generation']
    compact=_stage_artifact('relation_alternation_generation',stage)['data']
    assert compact['image_candidates'][1]['presentation']['image_candidate_kind']=='seed_variant'
    assert compact['image_candidates'][1]['presentation']['prompt_source_candidate_id']=='image_001'
    writer=RunWriter(tmp_path,'seed-variants',{})
    writer.write_entity(result)
    root=next(writer.entities_dir.iterdir())
    for artifact in artifacts:
        saved=json.loads((root/artifact.relative_metadata_path).read_text())
        assert saved['seed']==artifact.metadata['seed']
        assert saved['prompt_source_candidate_id']=='image_001'
        assert (root/artifact.relative_image_path).read_bytes()==artifact.image_bytes


def test_round_robin_fill_uses_original_sources_preserves_their_facts_and_makes_no_new_llm_calls():
    # Two accepted prompts and three rejected slots, each with two rewrites.
    prompts=Prompts([FIRST,SECOND]+[FIRST]*9)
    result=run(prompts,Images(),budget=5)
    assert len(prompts.inputs)==11
    candidates=result['relation_alternation_generation']['image_candidates']
    assert [c['positive_prompt'] for c in candidates]==[FIRST,SECOND,FIRST,SECOND,FIRST]
    assert [c['metadata']['prompt_source_candidate_id'] for c in candidates]==[
        'image_001','image_002','image_001','image_002','image_001']
    assert [c['metadata']['seed_variant_index'] for c in candidates]==[0,0,1,1,2]
    assert all(c['source_fact_refs']==candidates[0]['source_fact_refs'] for c in candidates)
    candidates[2]['verification_targets'].clear()
    assert candidates[0]['verification_targets']
    search=result['relation_alternation_generation']['image_prompt_search']
    assert search['unique_prompt_count']==2 and search['seed_variant_count']==3
    assert search['generated_image_count']==5 and search['requests']==11


def test_zero_accepted_prompts_do_not_create_seed_variants():
    class Rejected:
        uses_structured_prompts=True
        def generate_with_context(self,*args,context):
            raise ImageFocusValidationError('No valid focus')
    result=run(Rejected(),Images())
    assert result['generated_image_artifacts']==[]
    stage=result['relation_alternation_generation']
    assert stage['image_candidates']==[]
    assert stage['image_prompt_search']['unique_prompt_count']==0
    assert stage['image_prompt_search']['seed_variant_count']==0
    assert stage['image_prompt_search']['shortfall']==3


def test_budget_exhaustion_after_one_accepted_prompt_still_fills_the_target_without_more_requests():
    class Exhausted(Prompts):
        def generate_with_context(self,*args,context):
            result=super().generate_with_context(*args,context=context)
            self.last_call['sampling_attempts']=[{'temperature':.3}]*18
            return result
    prompts=Exhausted([FIRST])
    result=run(prompts,Images())
    assert len(prompts.inputs)==1
    stage=result['relation_alternation_generation']
    assert len(stage['image_candidates'])==3
    assert stage['image_prompt_search']['requests']==18
    assert stage['image_prompt_search']['seed_variant_count']==2


def test_prompt_only_mode_plans_variants_but_reports_no_generated_images():
    result=run(Prompts([FIRST]*7))
    assert not result['generated_image_artifacts']
    stage=result['relation_alternation_generation']
    assert len(stage['image_candidates'])==3
    assert all(c['generated_content']['type']=='image_prompt' for c in stage['image_candidates'])
    assert stage['image_prompt_search']['generated_image_count']==0
    assert stage['image_prompt_search']['unique_prompt_count']==1
    assert stage['image_prompt_search']['seed_variant_count']==2


def test_rejected_slot_before_first_accepted_prompt_can_reference_later_original_candidate():
    class FirstRejected(Prompts):
        def generate_with_context(self,*args,context):
            if not self.inputs:
                self.inputs.append(deepcopy(context))
                raise ImageFocusValidationError('Rejected first candidate')
            return super().generate_with_context(*args,context=context)
    result=run(FirstRejected([FIRST,SECOND]),Images())
    stage=result['relation_alternation_generation']
    assert [c['candidate_id'] for c in stage['image_candidates']]==['image_001','image_002','image_003']
    assert stage['image_candidates'][0]['metadata']['prompt_source_candidate_id']=='image_002'
    assert stage['image_candidates'][1]['metadata']['image_candidate_kind']=='distinct_prompt'


def test_image_generation_failure_is_propagated_instead_of_reported_as_a_filled_image():
    class BrokenImages:
        def generate(self,*args):
            raise RuntimeError('SDXL unavailable')
    with pytest.raises(RuntimeError,match='SDXL unavailable'):
        run(Prompts([FIRST]*7),BrokenImages())


def test_empty_prompt_is_not_accepted_as_a_seed_variant_source():
    with pytest.raises(ValueError,match='nonempty'):
        run(Prompts(['']),Images())
