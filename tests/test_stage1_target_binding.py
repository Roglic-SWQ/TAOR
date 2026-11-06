"""Target reference direction, abbreviation context and bounded semantic recovery."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mmkgc.adapters.source_text_triples import (
    SCHEMA, TARGET_REF, SourceTextTask, identity_context, merge_source_repair, source_text_task,
    targeted_source_repair, text_source_batches, resplit_text_source_batch,
)
from mmkgc.io.writer import _stage_artifact
from mmkgc.pipeline import _semantic_model_calls
from mmkgc.stages.modeling import KnowledgeResponseValidationError, build_knowledge_input
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT, StrictJsonStagePlanner
from test_triple_modeling import example
from test_stage1_source_bound import reply


def ref(position='subject', relation='occupation', other='film producer'):
    subject, obj = (TARGET_REF, other) if position == 'subject' else (other, TARGET_REF)
    return {'subject': subject, 'relation': relation, 'object': obj}


def response(label='Eric Fellner', triples=None):
    return {'entity_profile': {'primary_type': 'entity', 'core_identity': label},
            'extracted_triples': triples if triples is not None else [ref()], 'uncertain_aspects': []}


@pytest.mark.parametrize('position,other,expected', [
    ('subject', 'film producer', ('Eric Fellner', 'film producer')),
    ('object', 'Jane', ('Jane', 'Eric Fellner')),
])
def test_explicit_target_reference_binds_label_and_source_without_reversing_direction(position, other, expected):
    task = SourceTextTask('Eric Fellner', 'existing:text', 'Eric Nigel Fellner is a film producer.')
    wire = response(triples=[ref(position, 'relation', other)])
    before = deepcopy(wire)
    actual = task.decode(wire)['extracted_triples'][0]
    assert (actual['subject'], actual['object']) == expected
    assert actual['evidence_refs'] == ['existing:text'] and wire == before


@pytest.mark.parametrize('primary_type,expected_refs', [('person', ['existing:text']), ('entity', [])])
def test_profile_target_reference_is_bound_without_leaking_into_identity_checks(primary_type, expected_refs):
    task = SourceTextTask('student', 'existing:text', 'A student is a person who studies.')
    wire = response(label=TARGET_REF, triples=[ref(relation='is', other='person who studies')])
    wire['entity_profile']['primary_type'] = primary_type
    before = deepcopy(wire)
    actual = task.decode(wire)
    assert actual['entity_profile'] == {'primary_type': primary_type, 'core_identity': 'student',
                                        'evidence_refs': expected_refs}
    assert wire == before


@pytest.mark.parametrize('bad', [ref(other=''), ref(other=[]),
    {**ref(), 'subject': 'Eric Nigel Fellner'}, {**ref(), 'evidence_refs': ['invented']},
    {'subject': 'Eric Nigel Fellner', 'relation': 'occupation', 'object': 'film producer'},
    {'target_position': 'subject', 'relation': 'occupation', 'other': 'film producer'},
])
def test_target_binding_never_guesses_aliases_or_accepts_mixed_protocol_and_empty_endpoints(bad):
    task = SourceTextTask('Eric Fellner', 'existing:text', 'Eric Nigel Fellner is a film producer.')
    with pytest.raises(KnowledgeResponseValidationError):
        task.decode(response(triples=[bad]))


@pytest.mark.parametrize('opening', [
    'Mrs. Miniver is a 1942 film. ', 'Dr. J. Smith is a scientist. ',
    'F.I.S.T. is a film. ', 'U.S. Steel is a company. ',
])
def test_opening_context_and_recursive_batches_preserve_abbreviated_target_identity(opening):
    record, coarse, _, _ = example()
    record = replace(record, label=opening.split(' is ')[0])
    text = opening + 'It has a long documented history and many related facts. ' * 35
    record.existing_modalities['text'] = text
    assert identity_context(text) == opening
    inputs = build_knowledge_input(record, 'image', coarse)
    batches = text_source_batches(inputs)
    for batch in batches[1:]:
        assert source_text_task(batch).text.startswith(opening + '\n')
        for child in resplit_text_source_batch(batch):
            assert source_text_task(child).text.startswith(opening + '\n')
    spans = [batch['text_extraction_batch'] for batch in batches]
    assert ''.join(text[span['source_char_start']:span['source_char_end']] for span in spans) == text


def test_background_exclusion_has_a_model_reason_and_freezes_successful_target_facts():
    record, coarse, _, _ = example()
    bad = reply()
    background = {'subject': 'Hangzhou', 'relation': 'lies on', 'object': 'Hangzhou Bay'}
    bad['extracted_triples'].append(background)
    replies = iter([bad, {'fields': {'/extracted_triples/2': {
        'decision': 'background', 'triple': None, 'reason': 'Both endpoints are background places.'}}}])
    planner = StrictJsonStagePlanner(lambda prompt: json.dumps(next(replies)), source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert planner.calls[-1]['source_bound_response'] == reply()
    report = result['source_extraction']
    assert report['status'] == 'applied' and report['excluded_triples'][0]['original_triple'] == background
    assert report['excluded_triples'][0]['reason'] == 'Both endpoints are background places.'
    assert _stage_artifact('1_knowledge_modeling', result)['data']['source_extraction'] == report
    assert _semantic_model_calls(planner, start_index=0)[0]['source_extraction'] == report


def test_explicit_initial_background_outlet_is_audited_without_binding_or_promoting_it():
    record, coarse, _, _ = example()
    data = reply()
    background = {'subject': 'Hangzhou', 'relation': 'lies on', 'object': 'Hangzhou Bay',
                  'reason': 'This assertion concerns background places, not the target.'}
    data['excluded_triples'] = [background]
    planner = StrictJsonStagePlanner(lambda prompt: json.dumps(data), source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert not any(f['subject'] == 'Hangzhou' or f['object'] == 'Hangzhou Bay' for f in result['relation_facts'])
    assert result['source_extraction']['excluded_triples'][0]['original_triple'] == {
        key: background[key] for key in ('subject', 'relation', 'object')}
    assert result['source_extraction']['status'] == 'applied'


def test_empty_object_is_repaired_locally_through_target_reference():
    record, coarse, _, _ = example()
    bad = reply(); bad['extracted_triples'][1]['object'] = ''
    replies = iter([bad, {'fields': {'/extracted_triples/1': ref('subject', 'entrance shape', 'arch')}}])
    planner = StrictJsonStagePlanner(lambda prompt: json.dumps(next(replies)), source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert planner.calls[-1]['source_bound_response']['extracted_triples'][0] == bad['extracted_triples'][0]
    assert any(f['subject'] == 'Museum' and f['object'] == 'arch' for f in result['relation_facts'])


def test_repeated_wrong_alias_changes_to_one_item_reassessment_instead_of_repeating_patch():
    record, coarse, _, _ = example()
    bad = reply(); bad['extracted_triples'][1]['subject'] = 'The Museum'
    prompts = []
    def complete(prompt):
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1]); prompts.append(payload)
        if not payload.get('repair_kind'):
            return json.dumps(bad)
        value = bad['extracted_triples'][1]
        if payload['repair_kind'] == 'stage1_source_review_v1':
            value = {'decision': 'target', 'triple': ref('subject', 'entrance shape', 'arch'),
                     'reason': 'The source describes the target entrance.'}
            return json.dumps(value)
        return json.dumps({'fields': {'/extracted_triples/1': value}})
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert result['source_extraction']['status'] == 'applied'
    assert len(prompts) == 4 and prompts[-1]['repair_kind'] == 'stage1_source_review_v1'
    assert planner.calls[-1]['source_review_request_limit'] == 1


def test_unresolved_rows_are_separate_partial_output_after_bounded_single_item_review():
    record, coarse, inputs, _ = example()
    bad = reply(); bad['extracted_triples'][1]['object'] = ''
    count = 0
    def complete(prompt):
        nonlocal count
        count += 1
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
        return json.dumps({'fields': {'/extracted_triples/1': bad['extracted_triples'][1]}} if payload.get('repair_kind') else bad)
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert count == 4 and result['semantic_refinement']['status'] == 'partial'
    report = result['source_extraction']
    assert report['status'] == 'partial' and report['accepted_triple_count'] == 1
    assert report['rejected_triple_count'] == 1 and report['rejected_triples'][0]['triple']['object'] == ''
    assert all(f['object'] for f in result['relation_facts'])
    assert all(any(all(f[key] == original[key] for key in ('subject','relation','object'))
                   for f in result['relation_facts']) for original in inputs['fact_candidates'])
    assert planner.calls[-1]['status'] == 'partial'


def test_no_valid_rows_still_fails_instead_of_silently_accepting_empty_extraction():
    record, coarse, _, _ = example()
    bad = reply(); bad['extracted_triples'] = [{'subject': 'Third party', 'relation': 'is', 'object': 'person'}]
    def complete(prompt):
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
        return json.dumps({'fields': {'/extracted_triples/0': bad['extracted_triples'][0]}} if payload.get('repair_kind') else bad)
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    with pytest.raises(KnowledgeResponseValidationError):
        planner.build_knowledge(record, 'image', coarse)
    assert len(planner.calls[-1]['validation_attempts']) == 4


def test_single_item_review_survives_deletion_reindexing_and_reviews_each_unresolved_row():
    record, coarse, _, _ = example()
    bad = reply()
    bad['extracted_triples'] += [{'subject': 'A', 'relation': 'is', 'object': 'B'},
                                 {'subject': 'C', 'relation': 'is', 'object': 'D'}]
    seen = []
    def complete(prompt):
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
        if not payload.get('repair_kind'):
            return json.dumps(bad)
        if payload['repair_kind'] != 'stage1_source_review_v1':
            return json.dumps({'fields': {path: item for path, item in payload['invalid_items'].items()}})
        item = payload['item_to_review']; seen.append(item['subject'])
        return json.dumps({'decision': 'background', 'triple': None, 'reason': 'No target endpoint.'})
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert seen == ['A', 'C']
    assert planner.calls[-1]['source_bound_response'] == reply()
    assert len(result['source_extraction']['excluded_triples']) == 2


def test_new_protocol_is_marked_in_prompt_and_source_bound_adapter_keeps_grammar_disabled(monkeypatch):
    from io import BytesIO
    from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
    record, coarse, _, _ = example(); requests = []
    def send(req, timeout):
        body = json.loads(req.data); requests.append(body)
        return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(response('Museum', [ref('object', 'designed', 'Jane')]))}}]}).encode())
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', send)
    planner = QwenLocalSemanticStagePlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    planner.build_knowledge(record, 'image', coarse)
    payload = json.loads(requests[0]['messages'][1]['content'].split('INPUT_JSON:\n')[1])
    assert payload['schema'] == SCHEMA
    assert 'guided_json' not in requests[0] and requests[0]['max_tokens'] == 2048
    assert planner.calls[-1]['network_attempts'][0]['source_bound']


def test_partial_metadata_survives_merging_saving_and_generation_only_resume(tmp_path, monkeypatch):
    from mmkgc.adapters.image_batches import merge_image_batches
    from mmkgc.io import RunWriter
    from mmkgc.io.resume import load_saved_image_run
    from test_generation_only import prepared, pipeline
    record, coarse, inputs, _ = example()
    valid = StrictJsonStagePlanner(lambda prompt: json.dumps(reply()), source_bound_text_extraction=True)
    first = valid.build_knowledge(record, 'image', coarse)
    second = deepcopy(first)
    second['source_extraction'].update(status='partial', rejected_triple_count=1,
        rejected_triples=[{'triple': {'subject': 'Third party', 'relation': 'is', 'object': ''}}])
    merged = merge_image_batches(inputs, [first, second])
    assert merged['source_extraction']['status'] == 'partial'
    assert merged['source_extraction']['rejected_triple_count'] == 1
    assert merged['semantic_refinement']['status'] == 'partial'
    item = prepared()
    item.knowledge['source_extraction'] = deepcopy(merged['source_extraction'])
    result = pipeline(monkeypatch).finish(item)
    writer = RunWriter(tmp_path, 'partial_stage1', {})
    writer.write_entity(result)
    restored, _ = load_saved_image_run(writer.entities_dir / 'Q1', item.record, 3)
    assert restored.knowledge['source_extraction'] == merged['source_extraction']


def test_single_item_reviews_share_stage_deadline_and_timeout_is_not_partial_success(monkeypatch):
    record, coarse, _, _ = example()
    bad = reply(); bad['extracted_triples'][1]['object'] = ''
    clock, calls = [0.0], []
    monkeypatch.setattr('mmkgc.stages.semantic.time.perf_counter', lambda: clock[0])
    def complete(prompt):
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
        calls.append(KNOWLEDGE_REQUEST_CONTEXT.get()['deadline'])
        clock[0] += 3 if payload.get('repair_kind') == 'stage1_source_review_v1' else 1
        return json.dumps({'fields': {'/extracted_triples/1': bad['extracted_triples'][1]}} if payload.get('repair_kind') else bad)
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True, knowledge_stage_timeout_seconds=5)
    with pytest.raises(TimeoutError, match='total time budget'):
        planner.build_knowledge(record, 'image', coarse)
    assert len(calls) == 4 and len(set(calls)) == 1
    assert planner.calls[-1]['status'] == 'failed' and KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_many_unresolved_rows_cannot_exceed_six_final_reviews():
    record, coarse, _, _ = example()
    bad = reply()
    bad['extracted_triples'] = bad['extracted_triples'][:1] + [
        {'subject': f'Third party {index}', 'relation': 'is', 'object': 'background'} for index in range(9)]
    requests = []
    def complete(prompt):
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1]); requests.append(payload)
        if not payload.get('repair_kind'):
            return json.dumps(bad)
        return json.dumps({'fields': {path: bad['extracted_triples'][int(path.rsplit('/', 1)[1])]
                                     for path in payload['fields_to_replace']}})
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert sum(p.get('repair_kind') == 'stage1_source_review_v1' for p in requests) == 6
    assert len(requests) == 9
    assert result['source_extraction']['status'] == 'partial'
    assert result['source_extraction']['rejected_triple_count'] == 9
