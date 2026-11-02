"""Source preservation and downstream integration for unified graph/text triples."""
from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

from mmkgc.adapters.evidence_reranker import rerank_knowledge_relevance
from mmkgc.adapters.json_contract import local_semantic_json_schema
from mmkgc.adapters.qwen import QwenConfig, QwenLocalTextGenerator, QwenLocalImagePromptGenerator
from mmkgc.generation_prompts import IMAGE_PROMPT_VERSION, QWEN_PROMPT_VERSION, PROMPT_VERSION, TEXT_PROMPT_VERSION, TEXT_SYSTEM
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.data import EntityRecord
from mmkgc.io.resume import load_saved_image_run
from mmkgc.io.writer import RunWriter, _stage_artifact
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity, _semantic_model_calls
from mmkgc.ranking.context import build_ranking_context, candidate_ranking_context, build_kg_reference, build_structured_kg_triples
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.profiles import LocalScorerConfig, Stage4Profile, GPT4OMiniJudgeConfig
from mmkgc.stages.candidate_planning import apply_candidate_plans, candidate_plan_prompt, PLAN_SCHEMA
from mmkgc.stages.generation import relation_alternation_generation
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response, knowledge_response_errors, KnowledgeResponseValidationError
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.semantic import StrictJsonStagePlanner, build_stage_prompt
from mmkgc.stages.triples import TRIPLE_SCHEMA, fact_text, triple_response_schema
from test_candidate_fact_plan import Generator, Images, PassRanker, TextScores, Judge


def example(mode="image", text=True):
    record = EntityRecord(1, "TEST_MUSEUM", "Museum", "PRIVATE_TARGET", [],
        {"text": "Jane designed Museum. Museum has a red-brick facade and an arch entrance." if text and mode == "image" else None,
         "image": "existing.png" if mode == "text" else None,
         "image_summary": "Jane designed Museum. Museum has a red-brick facade and an arch entrance." if text and mode == "text" else None},
        {"text": "PRIVATE_TARGET", "image": "PRIVATE_TARGET_IMAGE"})
    coarse = {"selected_relations": [
        {"relation": "instance of", "direction": "out", "neighbor_entities": ["museum"],
         "evidence_triples": ["(Museum, instance of, museum)"]},
        {"relation": "architect", "direction": "out", "neighbor_entities": ["Jane"],
         "evidence_triples": ["(Museum, architect, Jane)"]},
        {"relation": "located in", "direction": "in", "neighbor_entities": ["Gallery"],
         "evidence_triples": ["(Gallery, located in, Museum)"]},
    ]}
    inputs = build_knowledge_input(record, mode, coarse)
    ref = "existing:text" if mode == "image" else "existing:image_summary"
    triples = ([{"subject": subject, "relation": relation, "object": obj, "evidence_refs": [ref]}
                for subject, relation, obj in (("Museum", "architect", "Jane"),
                    ("Museum", "facade material", "red brick"), ("Museum", "entrance shape", "arch"),
                    ("Jane", "designed", "Museum"))] if text else [])
    reply = {"entity_profile": {"primary_type": "museum", "core_identity": "Museum",
                               "evidence_refs": [inputs["evidence_catalog"][0]["evidence_id"]]},
             "extracted_triples": triples, "uncertain_aspects": []}
    return record, coarse, inputs, reply


def knowledge(mode="image"):
    record, coarse, inputs, reply = example(mode)
    return record, coarse, apply_knowledge_response(inputs, reply)


def planned(mode="text"):
    record, coarse, source = knowledge(mode)
    by_relation = {fact["relation"]: fact for fact in source["relation_facts"]}
    response = {"status": "ready", "candidates": [
        {"fact_ids": [by_relation["facade material"]["fact_id"]]},
        {"fact_ids": [by_relation["entrance shape"]["fact_id"]]},
        {"fact_ids": [by_relation["located in"]["fact_id"]]},
    ]}
    selection, reasoning = compile_plan_context(source, apply_candidate_plans(source, response, mode, 3), mode)
    return record, coarse, source, response, selection, reasoning


def test_graph_triples_are_retained_without_any_model_claim_or_text_output():
    _, _, inputs, reply = example(text=False)
    original = deepcopy(inputs)
    result = apply_knowledge_response(inputs, reply)
    assert inputs == original
    assert len(result["relation_facts"]) == len(inputs["fact_candidates"]) == 3
    for supplied, actual in zip(inputs["fact_candidates"], result["relation_facts"]):
        assert all(actual[key] == supplied[key] for key in ("fact_id", "subject", "relation", "object", "direction", "evidence_refs"))
        assert "claim" not in actual and "fact" not in actual


def test_text_and_graph_share_one_fact_list_and_duplicate_sources_are_merged():
    _, _, inputs, reply = example()
    assert all("existing:text" not in fact["evidence_refs"] for fact in inputs["fact_candidates"])
    result = apply_knowledge_response(inputs, reply)
    facts = result["relation_facts"]
    assert result["schema_version"] == TRIPLE_SCHEMA
    assert len(facts) == 6
    architect = next(fact for fact in facts if fact["relation"] == "architect")
    assert architect["fact_id"] == "rf:0002"
    assert set(architect["evidence_refs"]) == {"evidence:kg:0002", "existing:text"}
    unrelated = next(fact for fact in facts if fact["relation"] == "located in")
    assert unrelated["evidence_refs"] == ["evidence:kg:0003"]
    assert "key_attributes" not in result["stage1_v3"]


def test_incoming_graph_and_text_relations_keep_their_original_subject_and_object():
    _, _, result = knowledge()
    facts = {fact["relation"]: fact for fact in result["relation_facts"]}
    assert (facts["located in"]["subject"], facts["located in"]["object"], facts["located in"]["direction"]) == ("Gallery", "Museum", "in")
    assert (facts["designed"]["subject"], facts["designed"]["object"], facts["designed"]["direction"]) == ("Jane", "Museum", "in")
    assert (facts["architect"]["subject"], facts["architect"]["object"], facts["architect"]["direction"]) == ("Museum", "Jane", "out")


def test_repeated_or_reordered_extracted_triples_do_not_change_ids_or_create_duplicate_facts():
    _, _, inputs, reply = example()
    original = apply_knowledge_response(inputs, reply)
    reply["extracted_triples"] = list(reversed(reply["extracted_triples"])) + [deepcopy(reply["extracted_triples"][1])]
    assert apply_knowledge_response(inputs, reply)["relation_facts"] == original["relation_facts"]


@pytest.mark.parametrize("problem", ["extra_key", "old_claims", "wrong_source", "empty_refs", "duplicate_refs", "blank", "unrelated", "missing_profile", "invalid_gaps", "empty_subject"])
def test_invalid_text_extractions_are_rejected_without_mutating_sources(problem):
    _, _, inputs, reply = example()
    original = deepcopy(inputs)
    triple = reply["extracted_triples"][0]
    if problem == "extra_key": triple["claim"] = "free rewrite"
    elif problem == "old_claims": reply["relation_facts"] = []
    elif problem == "wrong_source": triple["evidence_refs"] = ["evidence:kg:0001"]
    elif problem == "empty_refs": triple["evidence_refs"] = []
    elif problem == "duplicate_refs": triple["evidence_refs"] *= 2
    elif problem == "blank": triple["relation"] = "  "
    elif problem == "unrelated": triple["subject"], triple["object"] = "Jane", "Paris"
    elif problem == "missing_profile": reply["entity_profile"].pop("core_identity")
    elif problem == "invalid_gaps": reply["uncertain_aspects"] = ["guess"]
    elif problem == "empty_subject": triple["subject"] = []
    with pytest.raises(KnowledgeResponseValidationError): apply_knowledge_response(inputs, reply)
    assert inputs == original


def test_no_modality_evidence_requires_an_empty_extraction_and_never_leaks_target_text():
    record, _, inputs, reply = example(mode="text", text=False)
    assert "PRIVATE_TARGET" not in json.dumps(inputs)
    schema = triple_response_schema(inputs)
    jsonschema.validate(reply, schema)
    assert schema["properties"]["extracted_triples"]["maxItems"] == 0
    reply["extracted_triples"] = [{"subject": record.label, "relation": "color", "object": "red", "evidence_refs": ["existing:text"]}]
    assert knowledge_response_errors(inputs, reply)


def test_prompt_and_local_repair_schema_use_only_the_new_extraction_contract():
    record, _, inputs, reply = example()
    prompt = build_stage_prompt(stage="stage1_knowledge", record=record, missing_mode="image", deterministic=inputs, prior={})
    schema = local_semantic_json_schema(prompt)
    jsonschema.validate(reply, schema)
    assert set(schema["properties"]) == {"entity_profile", "extracted_triples", "uncertain_aspects"}
    assert "PRIVATE_TARGET" not in prompt
    assert "DO NOT repeat, rewrite" in prompt


def test_default_planner_repairs_from_sources_and_exports_the_new_contract_marker():
    record, coarse, _, reply = example()
    invalid = deepcopy(reply)
    invalid["extracted_triples"][0]["evidence_refs"] = ["BAD_REFERENCE"]
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return json.dumps(invalid if len(prompts) == 1 else reply)
    planner = StrictJsonStagePlanner(complete, targeted_knowledge_repairs=True)
    result = planner.build_knowledge(record, "image", coarse)
    assert result["schema_version"] == TRIPLE_SCHEMA
    assert len(prompts) == 2 and "BAD_REFERENCE" not in prompts[1]
    assert "VALIDATION_FEEDBACK_JSON" in prompts[1]
    assert planner.calls[-1]["prompt_version"] == TRIPLE_SCHEMA
    assert _semantic_model_calls(planner, start_index=0)[0]["response_schema_version"] == TRIPLE_SCHEMA


def test_reranker_reads_structured_facts_without_requiring_claim_fields():
    _, _, result = knowledge()
    class Capture:
        calls = []
        def score_batch(self, *, queries, documents, instruction):
            self.calls = [{"query": q, "document": d} for q, d in zip(queries, documents)]
            return [0.7] * len(queries)
    scorer = Capture()
    scored = rerank_knowledge_relevance(result, scorer, "image")
    assert all(fact["relevance_score"] == 0.7 for fact in scored["relation_facts"])
    assert any('"subject": "Jane"' in call["document"] and '"object": "Museum"' in call["document"] for call in scorer.calls)
    assert all('"target_modality": "image"' in call["query"] for call in scorer.calls)
    assert all("claim" not in fact for fact in scored["relation_facts"])


def test_stage2_receives_unified_triples_and_selects_text_derived_fact_ids():
    record, _, source, reply, _, _ = planned()
    _, prompt = candidate_plan_prompt(record.label, "text", source, 3)
    payload = json.loads(prompt)
    assert "triples" in payload and "relation_facts" not in payload
    assert all({"subject", "relation", "object"} <= set(fact) and "claim" not in fact for fact in payload["triples"])
    planner = OpenAICompatibleRelationPlanner(QwenConfig("test", "http://localhost/v1", "test"), selection_schema=PLAN_SCHEMA)
    planner._complete = lambda messages, schema: json.dumps(reply)
    assert planner.plan(record, "text", source, 3)["fine_grained_relation_selection"]["schema_version"] == PLAN_SCHEMA


@pytest.mark.parametrize("mode", ["text", "image"])
def test_generation_and_verification_receive_the_same_structured_source_triples(mode):
    record, _, source, _, selection, reasoning = planned(mode)
    text, image = Generator(), Generator(True)
    generated = relation_alternation_generation(record, mode, source, reasoning, selection, text, image, Images(), candidate_budget=3)
    requests = text.calls if mode == "text" else image.calls
    candidates = generated["relation_alternation_generation"][mode + "_candidates"]
    assert requests[0]["source_facts"][0]["object"] == "red brick"
    assert "claim" not in requests[0]["source_facts"][0]
    incoming = candidates[2]["verification_targets"][-1]
    assert (incoming["subject"], incoming["object"], incoming["direction"]) == ("Gallery", "Museum", "in")


def test_real_ranking_engine_preserves_text_extracted_triples_and_candidate_scope():
    record, _, source, _, selection, reasoning = planned("text")
    generated = relation_alternation_generation(record, "text", source, reasoning, selection, Generator(), None, None, candidate_budget=3)
    context = build_ranking_context(record.entity_key, record.label, "text", record.existing_modalities,
        ranking_knowledge(source, reasoning), reasoning, generated, {})
    required = LocalScorerConfig(True, True, None, None)
    disabled = LocalScorerConfig(False, False, None, None)
    profile = Stage4Profile("test", required, disabled, disabled, disabled, GPT4OMiniJudgeConfig(True, Path("fake.yaml"), 1))
    engine = ModelFusionRankingEngine(profile=profile, scorers={"sentence_bert": TextScores(), "gpt4omini_judge": Judge()})
    ranking = engine.rank(context)
    assert ranking["ranking_status"] == "completed"
    assert all(item["model_scores"]["structured_kg_alignment"] == 1 for item in ranking["text_ranking"]["ranked_candidates"])
    scoped = candidate_ranking_context(context, context.text_candidates[0])
    assert "red brick" in build_kg_reference(scoped, "text") and "Jane" not in build_kg_reference(scoped, "text")
    assert build_structured_kg_triples(scoped, "text")[0]["object"] == "red brick"


@pytest.mark.parametrize("mode", ["text", "image"])
def test_default_qwen_generators_receive_structured_sources_for_new_knowledge(monkeypatch, mode):
    record, _, source, _, selection, reasoning = planned(mode)
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            content = ("Museum has a red-brick facade." if mode == "text" else json.dumps({
                "positive_prompt": "Museum building, red brick facade, architectural detail, neutral daylight, clear view",
                "used_text_cues": [], "omitted_risky_details": []}))
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
    def complete(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr("mmkgc.adapters.qwen.request.urlopen", complete)
    config = QwenConfig("Qwen/Qwen2-VL-7B-Instruct", "http://localhost/v1", "test", prompt_profile=QWEN_PROMPT_VERSION)
    result = relation_alternation_generation(record, mode, source, reasoning, selection,
        QwenLocalTextGenerator(config), QwenLocalImagePromptGenerator(config), Images(), candidate_budget=3)
    assert len(requests) == 3
    expected_version = TEXT_PROMPT_VERSION if mode == "text" else IMAGE_PROMPT_VERSION
    if mode == "text":
        assert all(body["messages"][0]["content"] == TEXT_SYSTEM for body in requests)
    prompt = requests[0]["messages"][-1]["content"]
    payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
    assert payload["fact_format"] == TRIPLE_SCHEMA
    assert payload["source_facts"][0]["object"] == "red brick"
    source_ref = "existing:text" if mode == "image" else "existing:image_summary"
    assert source_ref in {item["evidence_id"] for item in payload["source_evidence"]}
    assert "claim" not in payload["source_facts"][0]
    candidates = result["relation_alternation_generation"][mode + "_candidates"]
    for body, candidate in zip(requests, candidates, strict=True):
        request_payload = json.loads(body["messages"][-1]["content"].split("INPUT_JSON:\n", 1)[1])
        assert [fact["fact_id"] for fact in request_payload["source_facts"]] == candidate["source_fact_refs"]
    calls = [call for call in result["model_calls"] if call.get("fact_format") == TRIPLE_SCHEMA]
    assert len(calls) == 3
    assert all(call["prompt_version"] == expected_version and call["fact_format"] == TRIPLE_SCHEMA for call in calls)
    assert all(candidate["metadata"]["prompt_version"] == expected_version
               for candidate in result["relation_alternation_generation"][mode + "_candidates"])


@pytest.mark.parametrize("schema_version", [TRIPLE_SCHEMA, "grounded_triples_v1"])
def test_new_triple_artifact_resumes_original_image_bytes_and_source_bindings(tmp_path, schema_version):
    record, coarse, source, reply, selection, reasoning = planned("image")
    class Relevant:
        def score_batch(self, **kwargs): return [0.6] * len(kwargs["documents"])
    source = rerank_knowledge_relevance(source, Relevant(), "image")
    source["schema_version"] = source["stage1_v3"]["schema_version"] = schema_version
    call = {"stage": "stage2_selection", "status": "applied", "response_schema_version": PLAN_SCHEMA, "response_text": json.dumps(reply)}
    pipeline = MMKGCPipeline(candidate_budget=3, image_prompt_generator=Generator(True), image_generator=Images(), ranking_engine=PassRanker())
    result = pipeline.finish(PreparedEntity(record, "image", coarse, source, selection, reasoning, [call]))
    writer = RunWriter(tmp_path, "triples", {})
    writer.write_entity(result)
    folder = next(writer.entities_dir.iterdir())
    artifact = json.loads((folder / "1_knowledge_modeling.json").read_text())
    assert "triples" in artifact["data"] and "relation_facts" not in artifact["data"]
    assert all("claim" not in triple for triple in artifact["data"]["triples"])
    prepared, generation = load_saved_image_run(folder, record, 3)
    assert prepared.knowledge["schema_version"] == schema_version
    assert prepared.knowledge["relation_facts"] == source["relation_facts"]
    assert prepared.knowledge["relevance_selection"] == source["relevance_selection"]
    for image in generation["generated_image_artifacts"]:
        assert image.image_bytes == (folder / image.relative_image_path).read_bytes()
