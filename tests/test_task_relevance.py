"""Unified source provenance and retrieval behaviour, including facts outside old limits."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mmkgc.adapters.evidence_reranker import rerank_knowledge_relevance, RELEVANCE_INSTRUCTION, _rerank_text
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.stages.candidate_planning import candidate_plan_prompt, candidate_plan_schema, apply_candidate_plans
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response
from mmkgc.stages.semantic import StrictJsonStagePlanner
from mmkgc.stages.sources import public_sources
from mmkgc.stages.triples import model_input_view, triple_response_schema
from mmkgc.stages.generation import _model_generation_context
from mmkgc.stages.plan_context import compile_plan_context
from test_stage0_evidence_selection import record, neighbor
from test_candidate_fact_plan import Generator, PassRanker


def reply(inputs, triples=None):
    return {"entity_profile": {"primary_type": "entity", "core_identity": inputs["entity"]["label"],
                              "evidence_refs": []}, "extracted_triples": triples or [], "uncertain_aspects": []}


def large_knowledge():
    target = record([neighbor(f"object {index}", f"relation {index // 5:02d}") for index in range(200)])
    coarse = coarse_relation_filtering(target, "image")
    inputs = build_knowledge_input(target, "image", coarse)
    return target, coarse, inputs, apply_knowledge_response(inputs, reply(inputs))


class Scorer:
    def __init__(self):
        self.calls, self.batch_sizes = [], []
    def score_batch(self, *, queries, documents, instruction):
        self.batch_sizes.append(len(documents))
        scores = []
        for query, document in zip(queries, documents, strict=True):
            self.calls.append({"query": query, "document": document, "instruction": instruction})
            scores.append(0.9 if json.loads(document)["object"] == "object 199" else 0.1)
        return scores


def test_full_catalogue_survives_semantic_rejection_and_both_old_sampling_limits():
    target, coarse, inputs, knowledge = large_knowledge()
    assert len(coarse["high_recall_relation_candidates"]) == 24
    assert len(inputs["fact_candidates"]) == len(knowledge["relation_facts"]) == 200
    view = model_input_view(inputs)
    assert len(view["fact_candidates"]) == 32
    assert view["graph_catalogue_fact_count"] == 200
    assert len(view["graph_relation_counts"]) == 40
    coarse["selected_relations"] = []
    assert build_knowledge_input(target, "image", coarse) == inputs


def test_all_facts_are_scored_before_top_k_and_late_fact_can_reach_planning():
    _, _, _, knowledge = large_knowledge()
    original = deepcopy(knowledge)
    scorer = Scorer()
    result = rerank_knowledge_relevance(knowledge, scorer, "image", fact_budget=7, batch_size=8)
    assert knowledge == original
    assert len(result["relation_facts"]) == len(scorer.calls) == 200
    assert max(scorer.batch_sizes) == 8
    selected = result["relevance_selection"]["selected_fact_ids"]
    expected = next(fact for fact in result["relation_facts"] if fact["object"] == "object 199")
    assert selected[0] == expected["fact_id"] and len(selected) == 7
    assert len(result["relevance_selection"]["relations"]) == 40
    assert all(call["stage"] == "stage1_relevance_reranking" and "evidence_match_score" not in call for call in scorer.calls)
    _, prompt = candidate_plan_prompt("Target", "image", result, 3)
    payload = json.loads(prompt)
    assert len(payload["triples"]) == 7 and payload["triples"][0]["object"] == "object 199"
    assert payload["retrieval"]["catalog_fact_count"] == 200
    schema = candidate_plan_schema(result, 3)
    assert schema["properties"]["candidates"]["items"]["properties"]["fact_ids"]["items"]["enum"] == selected
    unshown = next(fact["fact_id"] for fact in result["relation_facts"] if fact["fact_id"] not in selected)
    with pytest.raises(ValueError):
        apply_candidate_plans(result, {"status": "ready", "candidates": [{"fact_ids": [unshown]}] * 3}, "image", 3)


def test_query_is_a_shared_task_not_a_copy_of_candidate_or_its_original_evidence():
    _, _, _, knowledge = large_knowledge()
    scorer = Scorer()
    knowledge["stage1_v3"]["entity_profile"]["primary_type"] = "MISCLASSIFIED_TYPE_SENTINEL"
    for fact in knowledge["relation_facts"]:
        fact["evidence_bundle"] = [{"content": "ORIGINAL_BUNDLE_SENTINEL"}]
        fact["source_modalities"] = ["SOURCE_MODALITY_SENTINEL"]
    rerank_knowledge_relevance(knowledge, scorer, "image")
    assert len({call["query"] for call in scorer.calls}) == 1
    assert all("ORIGINAL_BUNDLE_SENTINEL" not in json.dumps(call) for call in scorer.calls)
    assert all("MISCLASSIFIED_TYPE_SENTINEL" not in call["query"] for call in scorer.calls)
    assert all("SOURCE_MODALITY_SENTINEL" not in call["document"] for call in scorer.calls)
    assert all("object 199" not in call["query"] for call in scorer.calls)
    assert all(call["instruction"] == RELEVANCE_INSTRUCTION for call in scorer.calls)
    image_query = scorer.calls[0]["query"]
    other = Scorer()
    rerank_knowledge_relevance(knowledge, other, "text")
    assert image_query != other.calls[0]["query"]
    knowledge["entity"]["label"] = "A different entity"
    changed_entity = Scorer()
    rerank_knowledge_relevance(knowledge, changed_entity, "image")
    assert json.loads(changed_entity.calls[0]["query"])["entity_label"] == "A different entity"


@pytest.mark.parametrize("score", [True, -0.1, 1.1, float("nan"), float("inf"), "0.9"])
def test_invalid_relevance_scores_fail(score):
    _, _, _, knowledge = large_knowledge()
    class Invalid:
        def score_batch(self, **kwargs): return [score] * len(kwargs["documents"])
    with pytest.raises(ValueError, match="finite relevance score"):
        rerank_knowledge_relevance(knowledge, Invalid(), "image")


def test_batch_template_has_same_assistant_scoring_position_as_single_pair():
    text = _rerank_text("task", "triple", RELEVANCE_INSTRUCTION)
    assert text.startswith("<|im_start|>system\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<Query>: task\n<Document>: triple" in text


@pytest.mark.parametrize("mode", ["text", "image"])
def test_media_perception_outputs_share_triples_and_keep_source_positions(mode):
    observations = [{"evidence_id": f"existing:{modality}:segment:1", "modality": modality,
        "content": f"Target has {modality} detail.",
        "locator": {"asset": f"asset.{modality}", "start_seconds": 2.0, "end_seconds": 3.5}}
        for modality in ("image", "video", "audio")]
    target = replace(record([neighbor("entity", "instance of")]),
        existing_modalities={"text": "Target has a red facade.", "observations": observations})
    inputs = build_knowledge_input(target, mode, coarse_relation_filtering(target, mode))
    catalog = {item["evidence_id"]: item for item in inputs["evidence_catalog"]}
    refs = [ref for ref, item in catalog.items() if item["source"] == "existing_modality"]
    assert {catalog[ref]["modality"] for ref in refs} == ({"text", "image", "video", "audio"} - {mode})
    triples = [{"subject": "Target", "relation": "has source detail", "object": catalog[ref]["modality"],
                "evidence_refs": [ref]} for ref in refs]
    result = apply_knowledge_response(inputs, reply(inputs, triples))
    assert len(result["relation_facts"]) == 1 + len(triples)
    assert all(fact["fact_id"].startswith("mf:") for fact in result["relation_facts"][1:])
    selected = result["relation_facts"][-1]
    selection, reasoning = compile_plan_context(result, apply_candidate_plans(result,
        {"status": "ready", "candidates": [{"fact_ids": [selected["fact_id"]]}]}, mode, 1), mode)
    plan = selection["fine_grained_relation_selection"]["candidates"][0]
    context = _model_generation_context(result, plan, reasoning, mode, "view", "view")
    source = context["source_evidence"][0]
    assert source["modality"] == "video" and source["locator"]["start_seconds"] == 2.0


def test_target_derived_observations_are_masked_and_raw_paths_are_not_evidence():
    target = replace(record([]), existing_modalities={"video": "clip.mp4", "audio": "clip.wav", "image": "hidden.png",
        "observations": [{"evidence_id": "existing:audio:derived", "modality": "audio", "content": "PRIVATE",
                          "derived_from_modalities": ["audio", "image"]}]})
    context, catalog = public_sources(target, "image")
    assert context["has_video"] and context["has_audio"]
    assert set(context["unprocessed_modalities"]) == {"video", "audio"}
    assert catalog == [] and "PRIVATE" not in json.dumps(context)


def test_flat_media_summary_keeps_reference_to_the_original_asset():
    target = replace(record([]), existing_modalities={"image": "photo.png", "image_summary": "Target is round."})
    _, catalog = public_sources(target, "text")
    assert catalog[0]["modality"] == "image" and catalog[0]["locator"] == {"asset": "photo.png"}


def test_new_pipeline_skips_stage0_llm_and_sends_retrieval_scores_to_planner():
    target = record([neighbor("museum", "instance of"), neighbor("red brick", "facade material")])
    planner = StrictJsonStagePlanner(lambda prompt: json.dumps(reply(json.loads(prompt.split("INPUT_JSON:\n", 1)[1]))))
    planner.refine_coarse = lambda *args: pytest.fail("Stage 0 LLM filtering must not run")
    class Plans:
        calls = []
        def plan(self, target, mode, knowledge, budget):
            assert len(knowledge["relation_facts"]) == 2
            assert "evidence_match_score" not in json.dumps(knowledge)
            ids = knowledge["relevance_selection"]["selected_fact_ids"]
            return apply_candidate_plans(knowledge, {"status": "ready",
                "candidates": [{"fact_ids": [ids[0]]}] * budget}, mode, budget)
    pipeline = MMKGCPipeline(knowledge_planner=planner, evidence_reranker=Scorer(), relation_planner=Plans(),
                            text_generator=Generator(), ranking_engine=PassRanker(), relevance_fact_budget=1)
    prepared = pipeline.prepare(target, "text")
    assert prepared.coarse["stage0_v3"]["decisions"]["selection_method"] == "catalogue_only"
    assert len(prepared.knowledge["relevance_selection"]["selected_fact_ids"]) == 1
    assert len(planner.calls) == 1 and planner.calls[0]["stage"] == "stage1_knowledge"


def test_catalogue_mode_skips_heuristic_sampling_and_preserves_distinct_relation_names(monkeypatch):
    import mmkgc.stages.coarse as coarse_module
    monkeypatch.setattr(coarse_module, "representative_neighbors", lambda *a: pytest.fail("no heuristic sampling"))
    target = record([neighbor("A", "has-part"), neighbor("B", "has part"), neighbor("C", "has_part")])
    coarse = coarse_relation_filtering(target, "image", catalogue_only=True)
    assert len(coarse["stage0_v3"]["relations"]) == 3
    assert len(coarse["stage0_v3"]["facts"]) == 3
    assert len(build_knowledge_input(target, "image", coarse)["fact_candidates"]) == 3


def test_runtime_and_cli_expose_the_same_retrieval_budget(monkeypatch):
    from types import SimpleNamespace
    import mmkgc.defaults as defaults
    from mmkgc.run import build_parser
    from mmkgc.adapters.qwen import QwenConfig
    monkeypatch.setattr(defaults, "load_runtime_environment", lambda: None)
    monkeypatch.setattr(defaults, "load_qwen_config", lambda _: QwenConfig("test", "http://localhost/v1", "test"))
    monkeypatch.setattr(defaults, "Qwen3EvidenceReranker", lambda *a, **kw: Scorer())
    monkeypatch.setattr(defaults, "MMKGCPipeline", lambda **kwargs: SimpleNamespace(**kwargs))
    runtime = defaults.build_default_mainline_runtime(missing_mode="text", relevance_fact_budget=7,
                                                     max_selected_facts=3, dataset_root=None)
    assert runtime.pipeline.relevance_fact_budget == 7
    assert runtime.pipeline.relation_planner.max_selected_facts == 3
    assert runtime.pipeline.candidate_budget == 1
    assert build_parser().parse_args(["--relevance-fact-budget", "7"]).relevance_fact_budget == 7
    args = build_parser().parse_args(["--relevance-fact-budget", "7", "--max-selected-facts", "3",
                                     "--candidate-budget", "9"])
    assert (args.relevance_fact_budget, args.max_selected_facts, args.candidate_budget) == (7, 3, 9)
