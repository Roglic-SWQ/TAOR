from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from mmkgc.adapters.base import GeneratedImageArtifact
from mmkgc.io.writer import RunWriter
from scripts import run_wn18rr_completion_batch as batch


def image_artifact(entity, candidate, seed):
    output = BytesIO()
    Image.new('RGB', (1024, 1024)).save(output, format='PNG')
    return GeneratedImageArtifact(entity, candidate, f'generated_images/{candidate}.png',
        f'generated_images/{candidate}.json', output.getvalue(), {'seed': seed}, {'seed': seed})


def generation():
    return {'relation_alternation_generation': {'candidate_budget': 1, 'image_candidates': [
        {'candidate_id': 'image_001', 'source_plan_id': 'shared_facts', 'positive_prompt': 'one bird',
         'metadata': {}, 'verification_targets': [{'check_id': 'rf:1'}],
         'generated_content': {'type': 'image', 'value': 'one bird'}}]},
        'model_calls': [], 'generated_image_artifacts': [image_artifact('01234567', 'image_001', 100)]}


def test_seed_variant_uses_same_prompt_and_keeps_checks():
    calls = []
    def draw(*args):
        calls.append(args)
        return image_artifact(args[0], args[1], 101)
    result = batch.add_seed_variant(generation(), SimpleNamespace(entity_key='01234567'), SimpleNamespace(generate=draw))
    first, second = result['relation_alternation_generation']['image_candidates']
    assert calls == [('01234567', 'image_002', 'one bird', '')]
    assert first['positive_prompt'] == second['positive_prompt']
    assert first['verification_targets'] == second['verification_targets']
    assert len(result['generated_image_artifacts']) == 2


def test_second_draw_failure_preserves_first_real_image():
    original = generation()
    png = original['generated_image_artifacts'][0].image_bytes
    def fail(*args):
        raise RuntimeError('second candidate failed')
    result = batch.add_seed_variant(original, SimpleNamespace(entity_key='01234567'), SimpleNamespace(generate=fail))
    assert len(result['relation_alternation_generation']['image_candidates']) == 1
    assert result['generated_image_artifacts'][0].image_bytes == png
    assert result['model_calls'][-1]['status'] == 'failed'


def test_draw_budget_survives_process_recreation_and_failed_draw(tmp_path):
    calls = []
    def draw(*args):
        calls.append(args)
        if args[1] == 'image_safe_001':
            raise RuntimeError('failed repair consumes budget')
        return image_artifact(args[0], args[1], 100)
    delegate = SimpleNamespace(generate=draw)
    ledger = tmp_path/'draws.json'
    first = batch.BudgetedImages(delegate, ledger)
    first.generate('01234567', 'image_001', 'bird', '')
    first.generate('01234567', 'image_002', 'bird', '')
    with pytest.raises(RuntimeError, match='failed repair'):
        first.generate('01234567', 'image_safe_001', 'bird', '')
    resumed = batch.BudgetedImages(delegate, ledger)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        resumed.generate('01234567', 'image_safe_002', 'bird', '')
    assert len(calls) == 3


@pytest.mark.parametrize('gate,expected', [('pass', True), ('reject', False)])
def test_export_validates_actual_persisted_gate(tmp_path, gate, expected):
    entity = tmp_path/'entity'
    entity.mkdir()
    batch.save(entity/'final.json', {'ranking_status': 'completed', 'best_image_candidate_id': 'image_001'})
    batch.save(entity/'4_kg_aware_candidate_consistency_ranking.json', {'data': {'image_ranking': {
        'best_candidate_id': 'image_001', 'ranked_candidates': [
            {'candidate_id': 'image_001', 'gate': {'decision': gate},
             'observation': {'fact_checks': [{'requirement': 'must_include', 'status': 'supported'}]}}]}}})
    image = entity/'generated_images/image_001.png'
    image.parent.mkdir()
    image.write_bytes(image_artifact('01234567', 'image_001', 100).image_bytes)
    destination = tmp_path/'exports/image.png'
    if expected:
        result = batch.export_best(entity, 'image', destination)
        assert result['sha256'] == batch.digest(image)
        assert result['identity_verified'] is False
    else:
        with pytest.raises(ValueError, match='persisted gate'):
            batch.export_best(entity, 'image', destination)
        assert not destination.exists()


def test_text_export_reads_real_writer_format(tmp_path):
    candidate = {'candidate_id': 'text_001', 'target_modality': 'text',
        'generated_content': {'type': 'text', 'value': 'A student is a kind of intellectual.'},
        'hard_gate': {'decision': 'pass'}, 'final_score': 0.8,
        'observation': {'fact_checks': [{'requirement': 'must_include', 'status': 'supported'}]}}
    writer = RunWriter(tmp_path, 'evaluated', {})
    writer.write_entity({'entity': {'entity_key': '10557854'},
        'relation_alternation_generation': {'relation_alternation_generation': {'text_candidates': [candidate]}},
        'kg_aware_candidate_consistency_ranking': {'text_ranking': {
            'best_candidate_id': 'text_001', 'ranked_candidates': [candidate]}},
        'final': {'ranking_status': 'completed', 'best_text_candidate_id': 'text_001'}})
    destination = tmp_path/'export.txt'
    exported = batch.export_best(writer.entities_dir/'10557854', 'text', destination)
    assert destination.read_text() == candidate['generated_content']['value']
    assert exported['candidate_id'] == 'text_001'


@pytest.mark.parametrize('status', ['missing', 'uncertain', 'unverifiable', 'contradicted'])
def test_gate_pass_does_not_publish_unresolved_required_facts(tmp_path, status):
    candidate = {'candidate_id': 'image_001', 'gate': {'decision': 'pass'}, 'observation': {
        'fact_checks': [{'requirement': 'must_include', 'status': status}]}}
    entity = tmp_path/'entity'
    entity.mkdir()
    batch.save(entity/'final.json', {'ranking_status': 'completed', 'best_image_candidate_id': 'image_001'})
    batch.save(entity/'4_kg_aware_candidate_consistency_ranking.json', {'data': {'image_ranking': {
        'best_candidate_id': 'image_001', 'ranked_candidates': [candidate]}}})
    image = entity/'generated_images/image_001.png'
    image.parent.mkdir()
    image.write_bytes(image_artifact('01234567', 'image_001', 100).image_bytes)
    destination = tmp_path/'export.png'
    assert batch.export_best(entity, 'image', destination) is None
    assert not destination.exists()


def test_publication_keeps_optional_uncertainty_but_rejects_unresolved_claims():
    candidate = {'observation': {'fact_checks': [
        {'requirement': 'must_include', 'status': 'supported'},
        {'requirement': 'should_include', 'status': 'unverifiable'}]}}
    assert batch.publication_ready(candidate) is True
    candidate['observation']['unsupported_claims'] = [{'claim': 'invented appearance'}]
    assert batch.publication_ready(candidate) is False
    assert batch.publication_ready({}) is False


def test_resume_checks_artifacts_before_skipping_completed_entity(tmp_path):
    entity = tmp_path/'entity'
    entity.mkdir()
    batch.save(entity/'final.json', {'ranking_status': 'completed'})
    row = {'status': 'needs_review', 'entity_dir': str(entity),
           'artifact_sha256': {str(entity/'final.json'): batch.digest(entity/'final.json')}}
    assert batch.terminal(row, tmp_path)
    batch.save(entity/'final.json', {'ranking_status': 'tampered'})
    with pytest.raises(ValueError, match='artifact changed'):
        batch.terminal(row, tmp_path)


def test_two_seed_checkpoint_restores_without_new_prompt_or_pixels(tmp_path):
    import json
    from mmkgc.io.resume import load_saved_run
    from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
    from mmkgc.plan_schema import FACT_SELECTION_SCHEMA
    from test_candidate_fact_plan import target, Images
    from test_shared_fact_selection import selected, reply, Views
    knowledge, selection, reasoning = selected('image')
    record = target()
    call = {'stage': 'stage2_selection', 'status': 'applied',
            'response_schema_version': FACT_SELECTION_SCHEMA, 'max_selected_facts': 8,
            'response_text': json.dumps(reply())}
    prepared = PreparedEntity(record, 'image', {}, knowledge, selection, reasoning, [call])
    first = batch.relation_alternation_generation(record, 'image', knowledge, reasoning,
        selection, None, Views(), Images(), candidate_budget=1)
    generation = batch.add_seed_variant(first, record, Images())
    pipeline = MMKGCPipeline(candidate_budget=2, skip_stage4=True)
    entity = batch.persist_initial(pipeline, prepared, generation, tmp_path, 'initial')
    before = {str(p): batch.digest(p) for p in entity.rglob('*') if p.is_file()}
    restored, replay = load_saved_run(entity, record, 2, missing_mode='image')
    assert len(replay['generated_image_artifacts']) == 2
    original = generation['relation_alternation_generation']['image_candidates']
    for candidate, initial in zip(replay['relation_alternation_generation']['image_candidates'], original):
        assert candidate['verification_targets'] == initial['verification_targets']
        assert candidate['positive_prompt'] == initial['positive_prompt']
    assert before == {str(p): batch.digest(p) for p in entity.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failed_index', [1, 2])
def test_renderer_exception_keeps_other_candidates_and_failure_audit(failed_index):
    from test_candidate_fact_plan import target, Images
    from test_shared_fact_selection import selected, Views
    knowledge, selection, reasoning = selected('image')
    class FailingImages(Images):
        def generate(self, key, candidate_id, positive, negative):
            if candidate_id == f'image_{failed_index:03d}':
                raise RuntimeError('injected SDXL failure')
            return super().generate(key, candidate_id, positive, negative)
    result = batch.relation_alternation_generation(target(), 'image', knowledge,
        reasoning, selection, None, Views(), FailingImages(), candidate_budget=3)
    assert len(result['generated_image_artifacts']) == 2
    assert len(result['relation_alternation_generation']['image_candidates']) == 2
    failures = [call for call in result['model_calls'] if call.get('status') == 'failed']
    assert failures[-1]['candidate_id'] == f'image_{failed_index:03d}'
    assert failures[-1]['error_type'] == 'RuntimeError'


def test_second_repair_budget_exception_preserves_first_repaired_png(tmp_path):
    from mmkgc.stages.generation import safer_relation_alternation_generation
    from test_candidate_fact_plan import target, Images
    from test_shared_fact_selection import selected, Views
    knowledge, selection, reasoning = selected('image')
    record = target()
    renderer = batch.BudgetedImages(Images(), tmp_path/'ledger.json')
    initial = batch.relation_alternation_generation(record, 'image', knowledge,
        reasoning, selection, None, Views(), renderer, candidate_budget=2)
    feedback = {'schema_version': 'candidate_fact_repair_v1', 'candidate_feedback': []}
    for candidate in initial['relation_alternation_generation']['image_candidates']:
        fact = candidate['verification_targets'][0]
        feedback['candidate_feedback'].append({'candidate_id': candidate['candidate_id'],
            'operations': [{'action': 'replace', 'check_id': fact['check_id'], 'fact_id': fact['fact_id'],
                'expected': fact['claim'], 'observed': 'blue facade',
                'evidence_refs': fact['evidence_refs'], 'evidence': ['pixel: blue facade']}],
            'repair_route': 'resample', 'previous_candidate': deepcopy(candidate),
            'previous_prompt': candidate['positive_prompt']})
    repaired = safer_relation_alternation_generation(record, 'image', knowledge,
        reasoning, selection, None, Views(), renderer, candidate_budget=2,
        failed_ranking={}, stage3_feedback=feedback)
    assert [a.candidate_id for a in repaired['generated_image_artifacts']] == ['image_safe_001']
    assert repaired['generated_image_artifacts'][0].image_bytes
    assert any(call.get('candidate_id') == 'image_safe_002' and call.get('status') == 'failed'
               for call in repaired['model_calls'])
    assert len(batch.read(tmp_path/'ledger.json')) == 3
