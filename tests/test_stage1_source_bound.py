"""Source binding, immutable successful rows, and real engine deadline cancellation."""
import asyncio
from copy import deepcopy
from email.message import Message
from io import BytesIO
import json
from types import SimpleNamespace
from urllib import error

import pytest

from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.adapters.request_deadline import RequestDeadlineMiddleware
from mmkgc.adapters.source_text_triples import source_text_task, targeted_source_repair, text_source_batches
from mmkgc.adapters.stage1_repair import merge_knowledge_repair
from mmkgc.stages.modeling import apply_knowledge_response, KnowledgeResponseValidationError
from mmkgc.stages.semantic import StrictJsonStagePlanner
from test_triple_modeling import example


def reply():
    return {"entity_profile": {"primary_type": "entity", "core_identity": "Museum"},
        "extracted_triples": [{"subject": "Jane", "relation": "designed", "object": "Museum"},
            {"subject": "Museum", "relation": "entrance shape", "object": "arch"}], "uncertain_aspects": []}


def test_source_only_prompt_preserves_refs_direction_and_full_graph():
    _, _, inputs, _ = example()
    original = deepcopy(inputs)
    task = source_text_task(inputs)
    prompt = task.prompt()
    payload = json.loads(prompt.split("INPUT_JSON:\n")[1])
    assert payload["fact_candidates"] == []
    assert payload["evidence_catalog"][0]["evidence_id"] == "existing:text"
    assert "evidence:kg:" not in prompt and "PRIVATE_TARGET" not in prompt
    canonical = task.decode(reply())
    triple = canonical["extracted_triples"][0]
    assert (triple["subject"], triple["object"]) == ("Jane", "Museum")
    assert triple["evidence_refs"] == ["existing:text"]
    result = apply_knowledge_response(inputs, canonical)
    assert inputs == original
    assert all(any(all(f[k] == g[k] for k in ("subject", "relation", "object"))
                   for f in result["relation_facts"]) for g in inputs["fact_candidates"])


def test_complete_json_with_prose_wrapper_does_not_trigger_repair_or_drop_facts():
    record, coarse, _, _ = example()
    planner = StrictJsonStagePlanner(lambda prompt: json.dumps(reply()) + "\n\nThe JSON above lists the target facts.",
                                    source_bound_text_extraction=True)
    planner.build_knowledge(record, "image", coarse)
    call = planner.calls[-1]
    assert call["contract_repair_retries"] == 0 and call["response_wrappers_removed"] == [0]
    assert call["source_bound_response"] == reply()
    from mmkgc.adapters.source_text_triples import parse_source_response
    from mmkgc.stages.semantic import SemanticStageParseError
    with pytest.raises(SemanticStageParseError):
        parse_source_response(json.dumps(reply()) + '\n{"another":"payload"}')


@pytest.mark.parametrize("problem", ["unknown_ref", "graph_ref", "invalid_role"])
def test_source_binding_never_silently_relabels_unsupported_claims(problem):
    _, _, inputs, _ = example()
    data = reply()
    if problem == "unknown_ref":
        data["extracted_triples"][0]["evidence_refs"] = ["invented:source"]
    elif problem == "graph_ref":
        data["extracted_triples"][0] = {"subject": "Museum", "relation": "located in", "object": "Paris",
                                         "evidence_refs": ["fact:0001"]}
    else:
        data["extracted_triples"][0]["subject"] = []
    with pytest.raises(KnowledgeResponseValidationError):
        source_text_task(inputs).decode(data)


def test_partial_repair_freezes_successful_rows_and_revalidates_source_refs():
    record, coarse, inputs, _ = example()
    valid = reply(); invalid = deepcopy(valid)
    invalid["extracted_triples"][1]["evidence_refs"] = ["invented:source"]
    responses = [invalid, {"fields": {"/extracted_triples/1": valid["extracted_triples"][1]}, "facts": {}}]
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return json.dumps(responses[len(prompts) - 1])
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, "image", coarse)
    patch = json.loads(prompts[1].split("INPUT_JSON:\n")[1])
    assert patch["fields_to_replace"] == ["/extracted_triples/1"]
    assert "fact_candidates" not in patch and "response_template" in patch
    call = planner.calls[-1]
    assert call["source_bound_response"] == valid
    assert call["source_bound_response"]["extracted_triples"][0] == invalid["extracted_triples"][0]
    assert call["contract_repair_retries"] == 1
    assert len(result["relation_facts"]) >= len(inputs["fact_candidates"])


def test_partial_repair_can_remove_only_requested_bad_triples_and_rejects_extra_keys():
    _, _, inputs, _ = example()
    task = source_text_task(inputs)
    bad = reply(); bad["extracted_triples"][1]["evidence_refs"] = ["fact:0001"]
    with pytest.raises(KnowledgeResponseValidationError) as error_info:
        task.decode(bad)
    patch_task = targeted_source_repair(task, bad, error_info.value.errors)
    merged = merge_knowledge_repair(bad, patch_task, json.dumps({"fields": {"/extracted_triples/1": None}, "facts": {}}))
    assert merged["extracted_triples"] == bad["extracted_triples"][:1]
    task.decode(merged)
    with pytest.raises(KnowledgeResponseValidationError):
        merge_knowledge_repair(bad, patch_task, json.dumps({"fields": {"/extracted_triples/0": None}, "facts": {}}))


def test_text_batches_cover_full_source_and_merge_into_original_catalogue():
    record, coarse, inputs, _ = example()
    text = 'Museum is a museum. ' + 'Museum has an arch entrance and a red facade. ' * 35
    record.existing_modalities['text'] = text
    from mmkgc.stages.modeling import build_knowledge_input
    inputs = build_knowledge_input(record, 'image', coarse)
    batches = text_source_batches(inputs)
    offsets = [batch['text_extraction_batch'] for batch in batches]
    assert offsets[0]['source_char_start'] == 0
    assert offsets[-1]['source_char_end'] == len(text)
    assert all(left['source_char_end'] == right['source_char_start'] for left, right in zip(offsets, offsets[1:]))
    assert all(text[o['source_char_start']:o['source_char_end']] in source_text_task(batch).text
               for batch, o in zip(batches, offsets))
    deadlines = []
    def complete(prompt):
        from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT
        deadlines.append(KNOWLEDGE_REQUEST_CONTEXT.get()['deadline'])
        return json.dumps(reply())
    planner = StrictJsonStagePlanner(complete, source_bound_text_extraction=True)
    result = planner.build_knowledge(record, 'image', coarse)
    assert len(deadlines) == len(batches) > 1 and len(set(deadlines)) == 1
    assert next(item['content'] for item in result['evidence_catalog'] if item['evidence_id'] == 'existing:text') == text
    assert sum(f['subject'] == 'Jane' and f['relation'] == 'designed' and f['object'] == 'Museum'
               for f in result['relation_facts']) == 1
    assert all(call['text_extraction_batch']['count'] == len(batches) for call in planner.calls)


def test_source_protocol_collects_all_bad_rows_in_one_patch():
    _, _, inputs, _ = example()
    data = reply()
    for triple in data['extracted_triples']:
        triple['evidence_refs'] = ['fact:0001']
    with pytest.raises(KnowledgeResponseValidationError) as error_info:
        source_text_task(inputs).decode(data)
    task = targeted_source_repair(source_text_task(inputs), data, error_info.value.errors)
    assert set(task.fields) == {'/extracted_triples/0', '/extracted_triples/1'}


def test_unambiguous_patch_wrapper_is_unwrapped_and_validated():
    record, coarse, _, _ = example()
    invalid = reply(); invalid['extracted_triples'][1]['object'] = ''
    patch = {'fields': {'/extracted_triples/1': reply()['extracted_triples'][1]}, 'facts': {}}
    responses = iter([invalid, {'response': patch}])
    planner = StrictJsonStagePlanner(lambda prompt: json.dumps(next(responses)), source_bound_text_extraction=True)
    planner.build_knowledge(record, 'image', coarse)
    assert planner.calls[-1]['response_wrappers_removed'] == [1]
    assert planner.calls[-1]['source_bound_response'] == reply()


def test_local_stage1_timeout_does_not_submit_an_overlapping_retry(monkeypatch):
    record, coarse, _, _ = example()
    sent = []
    def send(req, timeout):
        sent.append(req)
        raise TimeoutError("simulated slow request")
    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", send)
    planner = QwenLocalSemanticStagePlanner(QwenConfig("Qwen/test", "http://localhost/v1", "EMPTY"))
    with pytest.raises(TimeoutError):
        planner.build_knowledge(record, "image", coarse)
    assert len(sent) == 1
    body = json.loads(sent[0].data)
    assert body["request_id"] == sent[0].get_header("X-mmkgc-request-id")
    assert "response_format" not in body and "guided_json" not in body


def test_engine_deadline_aborts_only_expired_request_and_next_request_runs():
    async def scenario():
        aborted = []; sent = []; cancellations = []
        class Engine:
            async def abort(self, request_id):
                aborted.append(request_id)
        async def app(scope, receive, send):
            if scope.get("slow"):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancellations.append(True)
            else:
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"OK"})
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        async def send(message):
            sent.append(message)
        scope = {"type": "http", "path": "/v1/chat/completions", "slow": True,
            "app": SimpleNamespace(state=SimpleNamespace(engine_client=Engine())),
            "headers": [(b"x-mmkgc-request-id", b"a" * 32), (b"x-mmkgc-timeout-seconds", b"0.01")]}
        middleware = RequestDeadlineMiddleware(app)
        await middleware(scope, receive, send)
        assert aborted == ["chatcmpl-" + "a" * 32] and cancellations == [True]
        assert sent[0]["status"] == 504
        assert (b"x-mmkgc-aborted", b"true") in sent[0]["headers"]
        sent.clear(); scope["slow"] = False
        await middleware(scope, receive, send)
        assert sent[0]["status"] == 200 and len(aborted) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('aborted', ['true', 'false'])
def test_service_timeout_restarts_only_when_engine_abort_unconfirmed(monkeypatch, aborted):
    record, coarse, _, _ = example()
    sent = []
    def send(req, timeout):
        sent.append(req)
        headers = Message(); headers['X-MMKGC-Aborted'] = aborted
        raise error.HTTPError(req.full_url, 504, 'Gateway Timeout', headers,
                              BytesIO(b'{"error":"deadline exceeded"}'))
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', send)
    planner = QwenLocalSemanticStagePlanner(QwenConfig('Qwen/test', 'http://localhost/v1', 'EMPTY'))
    with pytest.raises(error.HTTPError if aborted == 'true' else TimeoutError):
        planner.build_knowledge(record, 'image', coarse)
    assert len(sent) == 1
    assert planner.calls[-1]['network_attempts'][0]['server_abort_confirmed'] == (aborted == 'true')
