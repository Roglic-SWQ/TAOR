"""Regressions for missing patch containers and repeatedly truncated extraction."""
from copy import deepcopy
import json

import pytest

from mmkgc.adapters.source_text_triples import (
    merge_source_repair, parse_source_response, resplit_text_source_batch,
    source_text_task, targeted_source_repair, text_source_batches,
)
from mmkgc.adapters.stage1_repair import KnowledgeRepairTask, merge_knowledge_repair
from mmkgc.stages.modeling import KnowledgeResponseValidationError, build_knowledge_input
from mmkgc.stages.semantic import (
    KNOWLEDGE_REQUEST_CONTEXT, SemanticStageParseError, StrictJsonStagePlanner,
)
from test_stage1_source_bound import reply
from test_triple_modeling import example


def patch_task(bad):
    _, _, inputs, _ = example()
    source = source_text_task(inputs)
    with pytest.raises(KnowledgeResponseValidationError) as caught:
        source.decode(bad)
    return targeted_source_repair(source, bad, caught.value.errors)


@pytest.mark.parametrize('wrapper', ['none', 'response', 'response_template', 'split_response', 'fence'])
def test_missing_empty_facts_is_compatible_but_still_revalidates_values(wrapper):
    bad = reply(); bad['extracted_triples'][1]['object'] = ''
    task = patch_task(bad)
    patch = {'fields': {'/extracted_triples/1': reply()['extracted_triples'][1]}}
    if wrapper in ('response', 'response_template'):
        patch = {wrapper: patch}
    elif wrapper == 'split_response':
        patch = {'response': patch, 'facts': {}}
    raw = json.dumps(patch)
    if wrapper == 'fence':
        raw = '```json\n' + raw + '\n```'
    before = deepcopy(bad)
    merged, info = merge_source_repair(bad, task, raw)
    assert merged == reply() and bad == before
    assert info['missing_fields'] == []
    assert info['normalizations']


@pytest.mark.parametrize('patch', [
    {'fields': {'/extracted_triples/0': None}},
    {'fields': {'/extracted_triples/1': None}, 'facts': {'forged': 'claim'}},
    {'fields': {}, 'other': {}},
    {'response': {'fields': {}}, 'fields': {}},
])
def test_unknown_or_ambiguous_patch_keys_never_mutate_base(patch):
    bad = reply(); bad['extracted_triples'][1]['object'] = ''
    task = patch_task(bad); before = deepcopy(bad)
    with pytest.raises(KnowledgeResponseValidationError):
        merge_source_repair(bad, task, json.dumps(patch))
    assert bad == before


def test_required_facts_container_cannot_be_invented():
    task = KnowledgeRepairTask((), ('rf:0001',), {}, 'repair')
    with pytest.raises(KnowledgeResponseValidationError):
        merge_knowledge_repair({'relation_facts': []}, task, '{"fields":{}}')


def test_partial_patch_preserves_repaired_rows_and_only_requests_missing_row():
    record, coarse, _, _ = example()
    good = reply()
    good['extracted_triples'].append({'subject': 'Museum', 'relation': 'facade material', 'object': 'red brick'})
    bad = deepcopy(good)
    bad['extracted_triples'][1]['object'] = ''
    bad['extracted_triples'][2]['object'] = ''
    replies = iter([bad, {'fields': {'/extracted_triples/1': good['extracted_triples'][1]}},
                    {'fields': {'/extracted_triples/2': good['extracted_triples'][2]}}])
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return json.dumps(next(replies))
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    planner.build_knowledge(record, 'image', coarse)
    last_payload = json.loads(prompts[-1].split('INPUT_JSON:\n')[1])
    assert last_payload['fields_to_replace'] == ['/extracted_triples/2']
    assert planner.calls[-1]['source_bound_response'] == good
    assert planner.calls[-1]['source_patch_results'][0]['missing_fields'] == ['/extracted_triples/2']


def test_malformed_patch_retry_changes_prompt_with_precise_feedback():
    record, coarse, _, _ = example()
    bad = reply(); bad['extracted_triples'][1]['object'] = ''
    replies = iter([bad, {'fields': {'/extracted_triples/0': None}},
                    {'fields': {'/extracted_triples/1': reply()['extracted_triples'][1]}}])
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return json.dumps(next(replies))
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    planner.build_knowledge(record, 'image', coarse)
    assert prompts[1] != prompts[2]
    assert 'Latest errors:' in prompts[2] and '/extracted_triples/1' in prompts[2]
    assert planner.calls[-1]['source_bound_response'] == reply()


def test_many_invalid_fields_are_repaired_in_small_batches_with_bounded_requests():
    record, coarse, _, _ = example()
    good = reply()
    good['extracted_triples'] = [{'subject': 'Museum', 'relation': 'detail', 'object': str(i)} for i in range(9)]
    bad = deepcopy(good)
    for triple in bad['extracted_triples']:
        triple['object'] = ''
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps(bad)
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
        fields = payload['fields_to_replace']
        assert len(fields) <= 3
        return json.dumps({'fields': {p: good['extracted_triples'][int(p.rsplit('/',1)[1])] for p in fields}})
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    planner.build_knowledge(record, 'image', coarse)
    assert len(prompts) == 4
    assert planner.calls[-1]['source_patch_request_limit'] == 4
    assert planner.calls[-1]['source_bound_response'] == good


@pytest.mark.parametrize('raw', [
    '{"entity_profile":{},"entity_profile":{},"extracted_triples":[]}',
    '```json\n{"entity_profile":',
    '{"fields":{}}\n{"fields":{}}',
])
def test_parse_rejects_duplicate_truncated_or_multiple_payloads(raw):
    with pytest.raises(SemanticStageParseError):
        parse_source_response(raw)


def test_resplit_preserves_each_original_character_and_bounded_context():
    record, coarse, _, _ = example()
    text = 'Museum is a museum. ' + 'Museum has a red brick facade with an arch entrance. ' * 40
    record.existing_modalities['text'] = text
    inputs = build_knowledge_input(record, 'image', coarse)
    parent = text_source_batches(inputs)[1]
    children = resplit_text_source_batch(parent)
    parent_span = parent['text_extraction_batch']
    assert children[0]['text_extraction_batch']['source_char_start'] == parent_span['source_char_start']
    assert children[-1]['text_extraction_batch']['source_char_end'] == parent_span['source_char_end']
    spans = [child['text_extraction_batch'] for child in children]
    assert all(a['source_char_end'] == b['source_char_start'] for a, b in zip(spans, spans[1:]))
    for child, span in zip(children, spans):
        assert source_text_task(child).text.endswith(text[span['source_char_start']:span['source_char_end']])
        assert span['source_char_end'] - span['source_char_start'] < parent_span['source_char_end'] - parent_span['source_char_start']
    grandchildren = resplit_text_source_batch(children[-1])
    for child in grandchildren:
        span = child['text_extraction_batch']
        assert source_text_task(child).text.endswith(text[span['source_char_start']:span['source_char_end']])
        assert span['recovery_depth'] == 2


@pytest.mark.parametrize('reason', ['length', 'invalid_json', 'repetition'])
def test_failed_chunk_resplits_without_repeating_successful_chunks_or_losing_graph(reason):
    record, coarse, _, _ = example()
    text = 'Museum is a museum. ' + 'Museum has a red facade. ' * 60
    record.existing_modalities['text'] = text
    prompts, deadlines = [], []
    failed_span = None
    def complete(prompt):
        nonlocal failed_span
        context = KNOWLEDGE_REQUEST_CONTEXT.get(); call = context['call']
        span = call.get('text_extraction_batch', {})
        prompts.append(span); deadlines.append(context['deadline'])
        if span.get('index') == 2 and not span.get('recovery_depth') and failed_span is None:
            failed_span = deepcopy(span)
            if reason == 'length':
                call['network_attempts'].append({'finish_reason': 'length'})
                return '{"entity_profile":'
            if reason == 'invalid_json':
                return 'The extraction is not JSON.'
            data = reply()
            data['extracted_triples'] *= 4
            return json.dumps(data)
        return json.dumps(reply())
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert len(set(deadlines)) == 1
    assert sum(span.get('index') == 1 and not span.get('recovery_depth') for span in prompts) == 1
    failed_call = next(c for c in planner.calls if c.get('source_recovery'))
    assert failed_call['contract_repair_retries'] == 0
    assert failed_call['source_recovery']['status'] == 'recovered'
    from mmkgc.pipeline import _semantic_model_calls
    exported = _semantic_model_calls(planner, start_index=0)
    assert next(c for c in exported if c.get('source_recovery'))['source_recovery'] == failed_call['source_recovery']
    assert next(e['content'] for e in result['evidence_catalog'] if e['evidence_id'] == 'existing:text') == text
    assert all(any(all(g[k] == f[k] for k in ('subject','relation','object'))
                   for f in result['relation_facts']) for g in build_knowledge_input(record,'image',coarse)['fact_candidates'])


def test_persistent_truncation_is_bounded_and_never_returns_partial_facts():
    record, coarse, _, _ = example()
    record.existing_modalities['text'] = 'Museum has an arch entrance and red brick walls. ' * 10
    count = 0
    def complete(prompt):
        nonlocal count
        count += 1
        KNOWLEDGE_REQUEST_CONTEXT.get()['call']['network_attempts'].append({'finish_reason': 'length'})
        return '{"extracted_triples":['
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    with pytest.raises(SemanticStageParseError):
        planner.build_knowledge(record, 'image', coarse)
    assert count <= 3
    assert all(c['status'] == 'failed' for c in planner.calls)
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_nested_wrong_fields_are_reported_and_never_silently_dropped():
    record, coarse, _, _ = example()
    bad = reply()
    bad['extracted_triples'][1]['object_details'] = {'made_up': 'detail'}
    replies = iter([bad, {'fields': {'/extracted_triples/1': bad['extracted_triples'][1]}},
                    {'fields': {'/extracted_triples/1': reply()['extracted_triples'][1]}}])
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return json.dumps(next(replies))
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    planner.build_knowledge(record, 'image', coarse)
    assert 'forbidden keys' in prompts[1] and 'object_details' in prompts[1]
    assert 'Latest errors:' in prompts[2]
    assert planner.calls[-1]['source_bound_response'] == reply()


def test_persistent_patch_shape_error_resplits_and_preserves_valid_rows():
    record, coarse, _, _ = example()
    record.existing_modalities['text'] = ('Jane designed Museum. ' +
        'Museum has a red brick facade and an arch entrance. ' * 5)
    good = reply(); bad = deepcopy(good)
    bad['extracted_triples'][1]['object_details'] = {'unknown': 'value'}
    def complete(prompt):
        call = KNOWLEDGE_REQUEST_CONTEXT.get()['call']
        if call.get('text_extraction_batch', {}).get('recovery_depth'):
            # Valid original row must survive even if child output omits it.
            response = reply(); response['extracted_triples'] = response['extracted_triples'][1:]
            return json.dumps(response)
        payload = json.loads(prompt.split('INPUT_JSON:\n')[1])
        if payload.get('repair_kind'):
            return '{"wrong_fields":{}}'
        return json.dumps(bad)
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    call = next(c for c in planner.calls if c.get('source_recovery'))
    assert call['source_recovery']['status'] == 'recovered'
    assert call['source_recovery']['preserved_valid_rows'] == 1
    assert any(f['subject']=='Jane' and f['relation']=='designed' and f['object']=='Museum'
               for f in result['relation_facts'])


def test_split_recovery_cannot_reset_the_global_stage_deadline(monkeypatch):
    record, coarse, _, _ = example()
    record.existing_modalities['text'] = 'Museum has an arch entrance and red brick walls. ' * 20
    clock = [0.0]; count = [0]
    monkeypatch.setattr('mmkgc.stages.semantic.time.perf_counter', lambda: clock[0])
    def complete(prompt):
        count[0] += 1; clock[0] += 3
        KNOWLEDGE_REQUEST_CONTEXT.get()['call']['network_attempts'].append({'finish_reason': 'length'})
        return '{"extracted_triples":['
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True, knowledge_stage_timeout_seconds=5)
    with pytest.raises(TimeoutError, match='total time budget'):
        planner.build_knowledge(record, 'image', coarse)
    assert count[0] == 2
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None
