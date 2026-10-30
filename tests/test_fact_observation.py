"""Behavioral checks for factual observation, scoped repair and retained originals."""

from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from mmkgc.adapters import GeneratedImageArtifact
from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator, QwenLocalTextGenerator
from mmkgc.data import EntityRecord
from mmkgc.generation_prompts import QWEN_PROMPT_VERSION
from mmkgc.io.writer import _stage_artifact
from mmkgc.observation import bind_observation, repair_operations
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity, _merge_reflection_ranking, _needs_stage4_regeneration
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.contracts import ScoreResult
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.profiles import GPT4OMiniJudgeConfig, LocalScorerConfig, Stage4Profile
from mmkgc.ranking.scorers.gpt4omini_judge import (
    GPT4OMiniJudgeParseError, GPT4OMiniRankingJudge,
)
from mmkgc.stages.generation import build_stage3_feedback, safer_relation_alternation_generation
from mmkgc.stages.modeling import apply_knowledge_response, build_knowledge_input
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.relation_planning import apply_relation_plans


def setup(mode="image"):
    record = EntityRecord(entity_id=1, entity_key="Q1", label="Example Tower", description="PRIVATE_TARGET",
                          neighbors=[], existing_modalities={"text": None, "image": None},
                          ground_truth_modalities={mode: "PRIVATE_TARGET"})
    coarse = {"selected_relations": [
        {"relation": relation, "direction": "out", "neighbor_entities": [value],
         "evidence_triples": [f"(Example Tower, {relation}, {value})"]}
        for relation, value in (("exterior color", "red"), ("architect", "Architect A"), ("roof feature", "clock"))
    ]}
    inputs = build_knowledge_input(record, mode, coarse)
    claims = [f"Example Tower {fact['relation']} {fact['object']}" for fact in inputs["fact_candidates"]]
    knowledge = apply_knowledge_response(inputs, {
        "entity_profile": {"primary_type": "tower", "core_identity": "a tower", "secondary_types": [],
                           "semantic_domains": [], "evidence_refs": [inputs["evidence_catalog"][0]["evidence_id"]]},
        "text_visual_profile": {"non_visual_facts": [claims[1]], "visual_scene_cues": [], "context_cues": []},
        "key_attributes": [], "uncertain_aspects": [],
        "relation_facts": [{"fact_id": fact["fact_id"], "claim": claim}
                           for fact, claim in zip(inputs["fact_candidates"], claims)],
    })
    plans = {"status": "ready", "relation_plans": [
        {"fact_id": fact["fact_id"], "role": "primary" if index < 2 else "secondary",
         "candidate_quota": 1 if index < 2 else 0, "evidence_refs": fact["evidence_refs"],
         "supporting_fact_ids": ["rf:0003"] if index == 0 else [], "reason": "Use supplied evidence."}
        for index, fact in enumerate(knowledge["relation_facts"])
    ]}
    selection, reasoning = compile_plan_context(knowledge, apply_relation_plans(knowledge, plans, mode, 2), mode)
    targets = selection["fine_grained_relation_selection"]["primary_relations"][0]["verification_targets"]
    return record, coarse, knowledge, selection, reasoning, targets


def raw_observation(targets, *, status="supported", mode="image", confidence=0.95):
    checks = []
    for target in targets:
        state = "unverifiable" if target["verification_mode"] == "unverifiable" else status
        checks.append({
            "check_id": target["check_id"], "status": state,
            "verification_mode": "unverifiable" if state == "unverifiable" else "direct",
            "observed": "candidate content", "evidence": ["pixel: visible tower" if mode == "image" else "text: a tower"],
            "confidence": confidence, "reason": "Checked actual content against the supplied target.",
        })
    return {"fact_checks": checks, "unsupported_claims": []}


def observed_candidate(targets, *, status="missing", mode="image", candidate_id=None, score=0.8):
    candidate_id = candidate_id or f"{mode}_001"
    raw = raw_observation(targets, status=status, mode=mode)
    # Identity is already expressed; modify just the anchor fact.
    raw["fact_checks"][0]["status"] = "supported"
    return {
        "candidate_id": candidate_id, "target_modality": mode, "final_score": score,
        "verification_targets": targets, "hard_gate": {"decision": "pass", "soft_violations": []},
        "observation": bind_observation(raw, targets, mode, candidate_id),
        "generated_content": {"type": mode, "value": "original candidate"},
    }


def ranked(candidate):
    mode = candidate["target_modality"]
    return {"ranking_status": "completed", f"{mode}_ranking": {
        "ranked_candidates": [candidate], "rejected_candidates": [], "best_candidate_id": candidate["candidate_id"],
    }}


@pytest.mark.parametrize("mode", ["text", "image"])
def test_targets_remain_candidate_local_and_keep_fact_identity(mode):
    record, _, knowledge, selection, reasoning, targets = setup(mode)
    plans = selection["fine_grained_relation_selection"]["primary_relations"]
    assert {item["fact_id"] for item in targets} == {None, "rf:0001", "rf:0003"}
    assert {item["fact_id"] for item in plans[1]["verification_targets"]} == {None, "rf:0002"}
    assert targets[1]["direction"] == "out" and targets[1]["object"] == "red"
    candidate = {"candidate_id": f"{mode}_001", "target_modality": mode,
                 "verification_targets": targets, "generated_content": {"type": mode, "value": "candidate"}}
    context = build_ranking_context(record.entity_key, record.label, mode, {}, ranking_knowledge(knowledge, reasoning),
                                    reasoning, {"relation_alternation_generation": {f"{mode}_candidates": [candidate]}}, {})
    values = context.text_candidates if mode == "text" else context.image_candidates
    assert list(values[0]["verification_targets"])[1]["fact_id"] == "rf:0001"
    assert "PRIVATE_TARGET" not in json.dumps(context.to_prompt_payload())


def test_non_visual_fact_is_not_an_image_omission():
    _, _, _, selection, _, _ = setup()
    targets = selection["fine_grained_relation_selection"]["primary_relations"][1]["verification_targets"]
    observation = bind_observation(raw_observation(targets), targets, "image", "image_002")
    assert [item["fact_id"] for item in observation["unverifiable_facts"]] == ["rf:0002"]
    assert observation["missing_facts"] == [] and not observation["repair_needed"]
    feedback = build_stage3_feedback(ranked(observed_candidate(targets, status="supported", candidate_id="image_002")), attempt=1)
    assert not _needs_stage4_regeneration({"ranking_status": "completed"}, feedback)


@pytest.mark.parametrize("mode", ["text", "image"])
def test_passing_candidate_with_required_omission_triggers_scoped_repair(mode):
    *_, targets = setup(mode)
    candidate = observed_candidate(targets, mode=mode)
    feedback = build_stage3_feedback(ranked(candidate), attempt=1, generation={f"{mode}_candidates": [
        {**candidate, "positive_prompt": "original red tower prompt"}
    ]})
    assert _needs_stage4_regeneration(ranked(candidate), feedback)
    task = feedback["candidate_feedback"][0]
    assert [op["fact_id"] for op in task["operations"]] == ["rf:0001"]
    assert task["previous_prompt"] == "original red tower prompt"
    assert task["operations"][0]["evidence_refs"] == targets[1]["evidence_refs"]


@pytest.mark.parametrize("state,verification", [("uncertain", "direct"), ("unverifiable", "unverifiable"),
                                               ("unverifiable", "appearance_only")])
def test_uncertain_or_partial_visual_match_never_requests_factual_repair(state, verification):
    *_, targets = setup()
    raw = raw_observation(targets)
    raw["fact_checks"][1].update(status=state, verification_mode=verification)
    observation = bind_observation(raw, targets, "image", "image_001")
    assert not observation["repair_needed"]


def test_recommended_omission_and_low_confidence_conflict_do_not_trigger_repair():
    *_, targets = setup()
    raw = raw_observation(targets)
    raw["fact_checks"][1].update(status="contradicted", confidence=0.6)
    raw["fact_checks"][2]["status"] = "missing"
    assert not bind_observation(raw, targets, "image", "image_001")["repair_needed"]


@pytest.mark.parametrize("problem", ["unknown", "duplicate", "omitted", "metadata_evidence", "false_visual_proof", "nan"])
def test_invalid_observations_cannot_become_repair_instructions(problem):
    *_, targets = setup()
    raw = raw_observation(targets)
    if problem == "unknown":
        raw["fact_checks"][1]["check_id"] = "invented"
    elif problem == "duplicate":
        raw["fact_checks"].append(deepcopy(raw["fact_checks"][0]))
    elif problem == "omitted":
        raw["fact_checks"].pop()
    elif problem == "metadata_evidence":
        raw["fact_checks"][1]["evidence"] = ["metadata: prompt says red"]
    elif problem == "false_visual_proof":
        raw["fact_checks"][1]["verification_mode"] = "appearance_only"
    else:
        raw["fact_checks"][1]["confidence"] = float("nan")
    with pytest.raises(ValueError):
        bind_observation(raw, targets, "image", "image_001")


def test_unsupported_assertion_has_its_own_id_and_a_remove_operation():
    *_, targets = setup("text")
    raw = raw_observation(targets, mode="text")
    raw["unsupported_claims"] = [{"claim": "Built in 1700", "evidence": ["text: Built in 1700"],
                                  "confidence": 0.95, "reason": "No supplied evidence supports this date."}]
    observation = bind_observation(raw, targets, "text", "text_001")
    assert observation["violated_facts"] == []
    assert repair_operations(observation) == [{"action": "remove", "claim_id": "claim:0001",
                                               "observed": "Built in 1700", "evidence": ["text: Built in 1700"]}]


def test_another_candidates_omission_does_not_repair_an_already_correct_best():
    *_, targets = setup()
    correct = observed_candidate(targets, status="supported")
    other = observed_candidate(targets, candidate_id="image_002")
    ranking = ranked(correct)
    ranking["image_ranking"]["ranked_candidates"].append(other)
    assert not build_stage3_feedback(ranking, attempt=1)["repair_needed"]


def test_all_rejected_selects_repairable_content_instead_of_service_failure():
    *_, targets = setup()
    candidate = observed_candidate(targets)
    candidate["hard_gate"] = {"decision": "reject", "hard_violations": ["missing_required_content"]}
    ranking = ranked(candidate)
    ranking["ranking_status"] = "no_valid_candidate"
    ranking["image_ranking"]["best_candidate_id"] = None
    ranking["image_ranking"]["ranked_candidates"].insert(0, {
        "candidate_id": "image_002", "hard_gate": {"decision": "reject", "hard_violations": ["judge_service_error"]},
    })
    feedback = build_stage3_feedback(ranking, attempt=1)
    assert [task["candidate_id"] for task in feedback["candidate_feedback"]] == ["image_001"]


@pytest.mark.parametrize("mode", ["text", "image"])
def test_real_engine_keeps_structured_observation_and_its_audit_record(mode):
    from pathlib import Path

    record, _, knowledge, _, reasoning, targets = setup(mode)
    candidate = {"candidate_id": f"{mode}_001", "verification_targets": targets, "target_modality": mode,
                 "generated_content": {"type": mode, "value": "a tower"}, "status": "generated"}
    context = build_ranking_context("Q1", record.label, mode, {}, ranking_knowledge(knowledge, reasoning), reasoning,
                                    {"relation_alternation_generation": {f"{mode}_candidates": [candidate]}},
                                    {"image_001": {"image_bytes": png()}} if mode == "image" else {})

    class Client:
        def complete(self, prompt):
            if "Extract factual triples" in prompt:
                return '{"triples": []}'
            return judgment(raw_observation(targets, status="missing", mode=mode))

        def complete_multimodal(self, prompt, image_data_url):
            assert image_data_url.startswith("data:image/")
            return self.complete(prompt)

    class LocalScorer:
        def score_image_quality_batch(self, image, texts):
            return [ScoreResult(80, 0.8, "scored", "local") for _ in texts]

        def score_candidates(self, reference, texts):
            return [ScoreResult(80, 0.8, "scored", "local") for _ in texts]

    enabled = LocalScorerConfig(True, True, Path("unused"), "cpu")
    disabled = LocalScorerConfig(False, False, None, None)
    profile = Stage4Profile("test", enabled, disabled, enabled, enabled,
                            GPT4OMiniJudgeConfig(True, Path("unused"), 1))
    engine = ModelFusionRankingEngine(profile=profile, scorers={
        "gpt4omini_judge": GPT4OMiniRankingJudge(Client()), "clip": LocalScorer(),
        "blip_itm": LocalScorer(), "sentence_bert": LocalScorer(),
    })
    result = engine.rank(context)
    item = result[f"{mode}_ranking"]["ranked_candidates"][0]
    assert item["observation"]["repair_needed"]
    assert result["ranking_model_calls"][0]["observation"] == item["observation"]
    assert _needs_stage4_regeneration(result)


def judgment(raw):
    return json.dumps({"score": 80, "confidence": 0.95, "hard_reject": False, "hard_rejection_codes": [],
                       "soft_violations": [], "evidence": [], "reason": "Checked candidate.", "observation": raw})


@pytest.mark.parametrize("mode", ["text", "image"])
def test_real_judge_binds_fact_checks_and_repairs_missing_checks_with_same_evidence(mode):
    record, _, knowledge, _, reasoning, targets = setup(mode)
    candidate = {"candidate_id": f"{mode}_001", "verification_targets": targets,
                 "target_modality": mode, "generated_content": {"type": mode, "value": "a red tower"}}
    context = build_ranking_context(record.entity_key, record.label, mode, {}, ranking_knowledge(knowledge, reasoning),
                                    reasoning, {"relation_alternation_generation": {f"{mode}_candidates": [candidate]}}, {})
    good = raw_observation(targets, mode=mode)
    bad = deepcopy(good)
    bad["fact_checks"].pop()
    calls, responses = [], [judgment(bad), judgment(good)]

    class Client:
        def complete(self, prompt):
            calls.append((prompt, None))
            return responses.pop(0)

        def complete_multimodal(self, prompt, image_data_url):
            calls.append((prompt, image_data_url))
            return responses.pop(0)

    judge = GPT4OMiniRankingJudge(Client())
    if mode == "image":
        from mmkgc.ranking.image_input import ImageInput
        image_input = ImageInput.from_bytes(entity_key="Q1", candidate_id="image_001", image_bytes=png())
        result = judge.judge_image(context, candidate, image_input=image_input)
        assert all(url and url.startswith("data:image/") for _, url in calls)
    else:
        result = judge.judge_text(context, candidate)
    assert len(calls) == 2
    assert result.observation["fact_checks"][1]["fact_id"] == "rf:0001"
    assert result.score.details["contract_repair_retries"] == 1
    assert "verification_targets" in calls[0][0]


def test_judge_does_not_silently_accept_legacy_scores_when_factual_checks_are_required():
    record, _, knowledge, _, reasoning, targets = setup("text")
    candidate = {"candidate_id": "text_001", "verification_targets": targets,
                 "target_modality": "text", "generated_content": {"type": "text", "value": "a tower"}}
    context = build_ranking_context("Q1", record.label, "text", {}, knowledge, reasoning, {}, {})
    legacy = json.loads(judgment(raw_observation(targets, mode="text")))
    del legacy["observation"]
    judge = GPT4OMiniRankingJudge(SimpleNamespace(complete=lambda _: json.dumps(legacy)))
    with pytest.raises(GPT4OMiniJudgeParseError):
        judge.judge_text(context, candidate)


@pytest.mark.parametrize("mode", ["text", "image"])
def test_qwen_original_profile_receives_original_content_and_only_its_own_fact_edits(mode):
    record, _, knowledge, selection, reasoning, targets = setup(mode)
    candidate = observed_candidate(targets, mode=mode)
    candidate["positive_prompt"] = "original prompt without red color"
    feedback = build_stage3_feedback(ranked(candidate), attempt=1, generation={f"{mode}_candidates": [candidate]})
    config = QwenConfig("Qwen/Qwen2-VL-7B-Instruct", "http://localhost:1", "EMPTY", prompt_profile=QWEN_PROMPT_VERSION)
    captures = []

    class Text(QwenLocalTextGenerator):
        def generate(self, *args, generation_context=None):
            captures.append(self._build_prompt(*args, generation_context))
            return "a red tower"

    class Picture(QwenLocalImagePromptGenerator):
        def generate(self, *args, generation_context=None):
            captures.append(self._build_prompt(*args, generation_context))
            return {"positive_prompt": "a red tower", "used_text_cues": [], "omitted_risky_details": []}

    result = safer_relation_alternation_generation(
        record, mode, knowledge, reasoning, selection, Text(config), Picture(config),
        candidate_budget=2, failed_ranking=ranked(candidate), stage3_feedback=feedback,
    )["relation_alternation_generation"]
    assert len(captures) == 1
    assert "original prompt without red color" in captures[0] and '"action": "add"' in captures[0]
    # A global list of non-visual cautions can mention the architect, but it must
    # not become this candidate's source fact or an editing target.
    assert '"fact_id": "rf:0002"' not in captures[0]
    assert '"expected": "Example Tower architect Architect A"' not in captures[0]
    repaired = result[f"{mode}_candidates"][0]
    assert repaired["candidate_id"] == f"{mode}_safe_001" and repaired["source_candidate_id"] == f"{mode}_001"


@pytest.mark.parametrize("repair_state,repair_score,expected", [
    ("contradicted", 0.99, "image_001"), ("missing", 0.5, "image_001"), ("supported", 0.4, "image_safe_001"),
])
def test_merge_compares_factual_defects_before_quality_and_keeps_originals(repair_state, repair_score, expected):
    *_, targets = setup()
    original = observed_candidate(targets, score=0.8)
    repaired = observed_candidate(targets, candidate_id="image_safe_001", status=repair_state, score=repair_score)
    repaired["source_candidate_id"] = "image_001"
    merged = _merge_reflection_ranking(ranked(original), ranked(repaired))
    assert merged["image_ranking"]["best_candidate_id"] == expected
    assert len(merged["image_ranking"]["ranked_candidates"]) == 2


@pytest.mark.parametrize("regression", ["uncertain", "unverifiable", "lost_supported_fact"])
def test_higher_quality_cannot_hide_unresolved_repairs_or_lose_correct_facts(regression):
    *_, targets = setup()
    original = observed_candidate(targets)
    repaired = observed_candidate(targets, candidate_id="image_safe_001", status="supported", score=0.99)
    raw = raw_observation(targets)
    if regression == "lost_supported_fact":
        raw["fact_checks"][0]["status"] = "uncertain"
    else:
        raw["fact_checks"][1]["status"] = regression
        if regression == "unverifiable":
            raw["fact_checks"][1]["verification_mode"] = "unverifiable"
    repaired["observation"] = bind_observation(raw, targets, "image", "image_safe_001")
    repaired["source_candidate_id"] = "image_001"
    merged = _merge_reflection_ranking(ranked(original), ranked(repaired))
    assert merged["image_ranking"]["best_candidate_id"] == "image_001"


def png():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("mode", ["text", "image"])
@pytest.mark.parametrize("failure", [None, "generation", "ranking"])
def test_pipeline_performs_one_scoped_repair_and_writes_both_observations(mode, failure):
    record, coarse, knowledge, selection, reasoning, _ = setup(mode)
    image_bytes = png()

    class Text:
        def generate(self, label, entity_type, relation, angle, facts):
            if "safe retry" in angle and failure == "generation":
                raise TimeoutError("repair service unavailable")
            return "a red tower"

    class Prompt(Text):
        def generate(self, *args):
            super().generate(*args)
            return {"positive_prompt": "a red tower", "used_text_cues": [], "omitted_risky_details": []}

    class ImageGenerator:
        def generate(self, entity_key, candidate_id, prompt, negative_prompt):
            return GeneratedImageArtifact(entity_key, candidate_id, f"generated_images/{candidate_id}.png",
                                           f"generated_images/{candidate_id}.json", image_bytes,
                                           {"width": 8, "height": 8}, {})

    class Ranker:
        def __init__(self):
            self.calls = 0

        def rank(self, context):
            self.calls += 1
            if self.calls == 2 and failure == "ranking":
                raise TimeoutError("repair judge unavailable")
            candidates = context.text_candidates if mode == "text" else context.image_candidates
            items = []
            for candidate in candidates:
                item = {**candidate, **observed_candidate(candidate["verification_targets"], mode=mode,
                        candidate_id=candidate["candidate_id"], status="supported" if self.calls == 2 else "missing",
                        score=0.7 if candidate["candidate_id"].endswith("002") else 0.8)}
                items.append(item)
            return {"ranking_status": "completed", f"{mode}_ranking": {
                "ranked_candidates": items, "best_candidate_id": items[0]["candidate_id"], "rejected_candidates": []}}

    ranker = Ranker()
    pipeline = MMKGCPipeline(candidate_budget=2, text_generator=Text(), image_prompt_generator=Prompt(),
                             image_generator=ImageGenerator(), ranking_engine=ranker)
    result = pipeline.finish(PreparedEntity(record, mode, coarse, knowledge, selection, reasoning, []))
    ranking = result["kg_aware_candidate_consistency_ranking"]
    assert ranker.calls == (1 if failure == "generation" else 2)
    assert result["final"][f"best_{mode}_candidate_id"] == (f"{mode}_001" if failure else f"{mode}_safe_001")
    assert ranking["stage3_regeneration"]["attempts"] == 1
    assert ranking["stage3_regeneration"].get("status") == ("failed" if failure else None)
    artifact = _stage_artifact("kg_aware_candidate_consistency_ranking", ranking)["data"]
    assert artifact["stage3_regeneration"]["feedback"]["candidate_feedback"][0]["previous_content"]
    assert all(item["observation"] for item in artifact[f"{mode}_ranking"]["ranked_candidates"])
    assert len(result["generated_image_artifacts"]) == (0 if mode == "text" else 2 if failure == "generation" else 3)
    generated = result["relation_alternation_generation"]["relation_alternation_generation"][f"{mode}_candidates"]
    assert len(generated) == (2 if failure == "generation" else 3)
