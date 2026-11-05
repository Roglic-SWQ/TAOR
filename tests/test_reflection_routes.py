"""Reflection routing and fixed-issue completion, independent of model quality."""
from copy import deepcopy

import pytest
from jsonschema import validate

from mmkgc.adapters import GeneratedImageArtifact
from mmkgc.adapters.sdxl import _candidate_seed
from mmkgc.io.writer import _compact_stage4_candidate
from mmkgc.observation import (bind_observation, bind_repair_checks, removal_targets,
    repair_is_eligible, repair_operations, repair_outcomes)
from mmkgc.pipeline import _merge_reflection_ranking
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.profiles import Stage4Profile, LocalScorerConfig, GPT4OMiniJudgeConfig
from mmkgc.ranking.scorers.gpt4omini_judge import GPT4OMiniRankingJudge, GPT4OMiniJudgeParseError
from mmkgc.stages.generation import (build_stage3_feedback, route_image_repairs,
                                    safer_relation_alternation_generation)
from test_fact_observation import setup, raw_observation, observed_candidate, ranked, png
from test_judge_rubrics import fixture, Client


def image_feedback():
    *_, targets = setup()
    candidate = observed_candidate(targets, status="contradicted")
    candidate["positive_prompt"] = "A red tower with a clock on its roof."
    return candidate, build_stage3_feedback(ranked(candidate), attempt=1,
                                           generation={"image_candidates": [candidate]})


def reviewer(state="supported", confidence=.95):
    def review(candidate, operations):
        ids = [operation.get("check_id", operation.get("claim_id")) for operation in operations]
        return {"model_name": "test-judge", "observation": {"fact_checks": [
            {"check_id": issue, "status": state, "confidence": confidence, "verification_mode": "direct"}
            for issue in ids]}}
    return review


def test_cropped_image_missing_even_at_099_never_requests_added_parts():
    target = {"check_id": "windows", "fact_id": "f:windows", "claim": "The tower has exactly two front windows.",
              "requirement": "must_include", "verification_mode": "direct", "evidence_refs": ["e:windows"]}
    for mode in ("image", "text"):
        raw = raw_observation([target], mode=mode, confidence=.99, status="missing")
        raw["fact_checks"][0]["observed"] = "No windows in this doorway crop."
        observation = bind_observation(raw, [target], mode, mode + "_001")
        assert bool(repair_operations(observation)) is (mode == "text")
        assert observation["missing_facts"]


@pytest.mark.parametrize("state,confidence,route", [
    ("supported", .95, "resample"), ("contradicted", .95, "edit_prompt"),
    ("missing", .95, "edit_prompt"), ("uncertain", .99, "skip"), ("supported", .6, "skip")])
def test_image_repair_uses_prompt_observation_to_choose_action(state, confidence, route):
    _, feedback = image_feedback()
    routed, calls = route_image_repairs(feedback, reviewer(state, confidence))
    task = (routed["skipped_repairs"] if route == "skip" else routed["candidate_feedback"])[0]
    assert task["repair_route"] == route
    assert routed["repair_needed"] is (route != "skip")
    assert calls[0]["kind"] == "image_prompt_review"
    assert feedback["candidate_feedback"][0]["repair_route"] == "review_prompt"


@pytest.mark.parametrize("review", [None, lambda *args: {"observation": {"fact_checks": []}}])
def test_unavailable_or_incomplete_prompt_review_abstains(review):
    _, feedback = image_feedback()
    routed, calls = route_image_repairs(feedback, review)
    assert not routed["repair_needed"]
    assert routed["skipped_repairs"][0]["skip_reason"] == "prompt_review_failed"
    assert calls[0]["status"] == "failed"


def test_resample_preserves_exact_prompt_and_facts_without_calling_prompt_generator():
    record, _, knowledge, selection, reasoning, _ = setup()
    candidate, feedback = image_feedback()
    feedback, _ = route_image_repairs(feedback, reviewer())

    class Never:
        def generate(self, *args, **kwargs):
            raise AssertionError("resampling must not rewrite the correct prompt")

    class Images:
        def generate(self, entity_key, candidate_id, prompt, negative):
            assert prompt == candidate["positive_prompt"]
            return GeneratedImageArtifact(entity_key, candidate_id, f"generated_images/{candidate_id}.png",
                f"generated_images/{candidate_id}.json", png(), {"seed": _candidate_seed(42, candidate_id)}, {})

    result = safer_relation_alternation_generation(record, "image", knowledge, reasoning, selection,
        Never(), Never(), Images(), candidate_budget=2, failed_ranking=ranked(candidate), stage3_feedback=feedback)
    repaired = result["relation_alternation_generation"]["image_candidates"]
    assert len(repaired) == 1
    assert repaired[0]["candidate_id"] == "image_safe_001"
    assert repaired[0]["positive_prompt"] == candidate["positive_prompt"]
    assert repaired[0]["verification_targets"] == candidate["verification_targets"]
    assert repaired[0]["metadata"]["repair_route"] == "resample"
    assert repaired[0]["generated_content"]["seed"] != _candidate_seed(42, "image_001")


def test_prompt_review_checks_actual_prompt_and_cannot_invent_a_quote():
    context, candidate, _ = fixture("image")
    candidate["positive_prompt"] = "A red tower with a roof clock."
    candidate["generated_content"]["value"] = "PRIVATE_IMAGE_DESCRIPTION"
    _, feedback = image_feedback()
    operations = feedback["candidate_feedback"][0]["operations"]
    ids = {item["check_id"] for item in operations}
    targets = [item for item in candidate["verification_targets"] if item["check_id"] in ids]
    valid = raw_observation(targets, mode="text")
    for check in valid["fact_checks"]:
        check["observed"] = "red tower" if "0001" in check["check_id"] else "roof clock"
    invalid = deepcopy(valid)
    invalid["fact_checks"][0]["observed"] = "a quote absent from the prompt"
    client = Client(invalid, valid)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).review_image_prompt(context, candidate, operations)
    assert result["contract_repair_retries"] == 1
    assert len(client.calls) == 2
    assert "PRIVATE_IMAGE_DESCRIPTION" not in client.calls[0][0]
    validate(valid, client.calls[0][1])


def removal_fixture():
    context, candidate, payload = fixture()
    raw = payload["observation"]
    raw["unsupported_claims"] = [{"claim": "The tower is blue.", "evidence": ["text: blue tower",
        "context:" + candidate["verification_targets"][1]["evidence_refs"][0]],
        "confidence": .95, "reason": "Contradicts the supplied red exterior."}]
    original = {**candidate, "target_modality": "text", "hard_gate": {"decision": "pass"}, "final_score": .5,
        "observation": bind_observation(raw, candidate["verification_targets"], "text", candidate["candidate_id"])}
    targets = removal_targets(repair_operations(original["observation"]))
    return context, candidate, payload, original, targets


@pytest.mark.parametrize("state,eligible", [("resolved", True), ("unresolved", False), ("uncertain", False)])
def test_removed_assertion_requires_explicit_fixed_issue_review(state, eligible):
    _, candidate, _, original, targets = removal_fixture()
    after = observed_candidate(candidate["verification_targets"], mode="text", status="supported", candidate_id="text_safe_001")
    after["source_candidate_id"] = "text_001"
    after["observation"]["repair_checks"] = bind_repair_checks([{
        "issue_id": targets[0]["issue_id"], "status": state, "observed": "The current description says red tower.",
        "evidence": ["text: red tower"], "confidence": .95, "reason": "Explicitly checked the original assertion."}], targets, "text")
    assert repair_is_eligible(original, after) is eligible
    assert repair_outcomes(original, after)[0]["status"] == state
    result = _merge_reflection_ranking(ranked(original), ranked(after))
    assert result["text_ranking"]["best_candidate_id"] == ("text_safe_001" if eligible else "text_001")
    compact = _compact_stage4_candidate(after, rank=1, fallback_modality="text")
    assert compact["repair_outcomes"][0]["status"] == state


def test_empty_new_error_list_is_not_evidence_that_an_old_claim_was_removed():
    _, candidate, _, original, _ = removal_fixture()
    after = observed_candidate(candidate["verification_targets"], mode="text", status="supported")
    after["final_score"] = .99
    assert after["observation"]["unsupported_claims"] == []
    assert not repair_is_eligible(original, after)


def test_removal_review_survives_context_schema_model_binding_and_artifact():
    context, candidate, payload, _, targets = removal_fixture()
    candidate["repair_targets"] = targets
    candidate["candidate_id"] = "text_safe_001"
    candidate["generated_content"]["value"] = "A red tower with a roof clock."
    payload["observation"]["unsupported_claims"] = []
    payload["observation"]["repair_checks"] = [{"issue_id": targets[0]["issue_id"], "status": "resolved",
        "observed": "The text says red, not blue.", "evidence": ["text: red tower"], "confidence": .95,
        "reason": "The original conflicting assertion is no longer made."}]
    real_context = build_ranking_context(context.entity_key, context.entity_label, "text", {}, {}, {},
        {"relation_alternation_generation": {"text_candidates": [candidate]}}, {})
    safe_candidate = dict(real_context.text_candidates[0])
    client = Client(payload)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(real_context, safe_candidate)
    validate(payload, client.calls[0][1])
    assert "REPAIR_TARGETS" in client.calls[0][0]
    assert result.observation["repair_checks"][0]["claim"] == targets[0]["claim"]
    assert result.score.details["rubric_scores"]["verified_fact_consistency"]["score"] == 4


def test_omitted_issue_review_cannot_pass_a_removal_review():
    context, candidate, payload, _, targets = removal_fixture()
    candidate["repair_targets"] = targets
    payload["observation"]["unsupported_claims"] = []
    client = Client(payload, payload)
    with pytest.raises(GPT4OMiniJudgeParseError):
        GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(context, candidate)
    assert len(client.calls) == 2


def test_fact_fix_that_makes_the_image_unusable_cannot_replace_the_original():
    *_, targets = setup()
    original = observed_candidate(targets, status="contradicted")
    after = observed_candidate(targets, status="supported")
    original["judge_rubrics"] = {"rubric_scores": {"visual_usability": {"score": 4}}}
    after["judge_rubrics"] = {"rubric_scores": {"visual_usability": {"score": 1}}}
    assert not repair_is_eligible(original, after)


@pytest.mark.parametrize("route", ["resample", "edit_prompt", "skip_omission", "skip_uncertain"])
def test_full_pipeline_routes_one_repair_and_keeps_all_candidates(route):
    from pathlib import Path
    record, coarse, knowledge, selection, reasoning, targets = setup()
    _, _, initial = fixture("image", values=(3, 2, 3, 4))
    if route == "skip_omission":
        initial["scores"]["verified_fact_consistency"]["score"] = 4
        initial["observation"]["fact_checks"][1]["status"] = "missing"
    else:
        initial["observation"]["fact_checks"][1]["status"] = "contradicted"
    _, _, fixed = fixture("image")
    review = raw_observation([targets[1]], mode="text",
        status="supported" if route == "resample" else "uncertain" if route == "skip_uncertain" else "contradicted")
    review["fact_checks"][0]["observed"] = "blue tower" if route == "edit_prompt" else "red tower"
    responses = [initial] if route == "skip_omission" else [initial, review]
    if route in {"resample", "edit_prompt"}:
        responses.append(fixed)
    client = Client(*responses)
    off = LocalScorerConfig(False, False, None, None)
    engine = ModelFusionRankingEngine(profile=Stage4Profile("test", off, off, off, off,
        GPT4OMiniJudgeConfig(True, Path("unused"), 30)), scorers={
            "gpt4omini_judge": GPT4OMiniRankingJudge(client, use_rubrics=True)})

    class Prompts:
        def __init__(self):
            self.calls = []
        def generate_with_context(self, *args, context):
            self.calls.append(context)
            color = "blue" if route == "edit_prompt" and len(self.calls) == 1 else "red"
            return {"positive_prompt": f"A {color} tower with a roof clock."}

    class Images:
        def __init__(self):
            self.calls = []
        def generate(self, entity_key, candidate_id, prompt, negative):
            self.calls.append((candidate_id, prompt))
            return GeneratedImageArtifact(entity_key, candidate_id, f"generated_images/{candidate_id}.png",
                f"generated_images/{candidate_id}.json", png(), {"seed": _candidate_seed(42, candidate_id)}, {})

    prompts, images = Prompts(), Images()
    pipeline = MMKGCPipeline(candidate_budget=1, image_prompt_generator=prompts,
        image_generator=images, ranking_engine=engine)
    result = pipeline.finish(PreparedEntity(record, "image", coarse, knowledge, selection, reasoning, []))
    repaired = route in {"resample", "edit_prompt"}
    assert result["final"]["best_image_candidate_id"] == ("image_safe_001" if repaired else "image_001")
    assert len(images.calls) == (2 if repaired else 1)
    assert len(prompts.calls) == (2 if route == "edit_prompt" else 1)
    ranking = result["kg_aware_candidate_consistency_ranking"]
    assert len(ranking["image_ranking"]["ranked_candidates"]) == (2 if repaired else 1)
    if repaired:
        assert ranking["stage3_regeneration"]["attempts"] == 1
        item = next(item for item in ranking["image_ranking"]["ranked_candidates"] if item["candidate_id"] == "image_safe_001")
        assert item["repair_eligible"]
        assert item["repair_outcomes"] == [{"issue_id": targets[1]["check_id"], "action": "replace", "status": "resolved"}]
    elif route == "skip_uncertain":
        assert ranking["stage3_regeneration"]["attempted"] is False
    if route == "resample":
        assert images.calls[0][1] == images.calls[1][1]


@pytest.mark.parametrize("state", ["resolved", "unresolved", "uncertain"])
def test_text_removal_is_rechecked_in_the_same_pipeline_judge_call(state):
    from pathlib import Path
    record, coarse, knowledge, selection, reasoning, targets = setup("text")
    _, _, initial, _, removal_checks = removal_fixture()
    initial["scores"]["verified_fact_consistency"]["score"] = 2
    _, _, fixed = fixture()
    fixed["observation"]["repair_checks"] = [{"issue_id": removal_checks[0]["issue_id"], "status": state,
        "observed": "The current candidate was inspected for the original blue assertion.",
        "evidence": ["text: red tower" if state == "resolved" else "text: blue tower"], "confidence": .95,
        "reason": "Explicit review of the fixed original issue."}]
    fixed["scores"]["verified_fact_consistency"]["score"] = 2 if state == "unresolved" else 4

    class GraphClient(Client):
        def complete(self, prompt):
            assert "Extract factual triples" in prompt
            return '{"triples": []}'

    class Text:
        def __init__(self):
            self.calls = 0
        def generate_with_context(self, *args, context):
            self.calls += 1
            return "A red tower with a roof clock." + (" It is blue." if self.calls == 1 or state != "resolved" else "")

    client = GraphClient(initial, fixed)
    off = LocalScorerConfig(False, False, None, None)
    engine = ModelFusionRankingEngine(profile=Stage4Profile("test", off, off, off, off,
        GPT4OMiniJudgeConfig(True, Path("unused"), 30)), scorers={
            "gpt4omini_judge": GPT4OMiniRankingJudge(client, use_rubrics=True)})
    text = Text()
    pipeline = MMKGCPipeline(candidate_budget=1, text_generator=text, ranking_engine=engine)
    result = pipeline.finish(PreparedEntity(record, "text", coarse, knowledge, selection, reasoning, []))
    assert text.calls == 2
    assert result["final"]["best_text_candidate_id"] == ("text_safe_001" if state == "resolved" else "text_001")
    assert "REPAIR_TARGETS" in client.calls[1][0]
    ranking = result["kg_aware_candidate_consistency_ranking"]
    repaired = next(item for item in ranking["text_ranking"]["ranked_candidates"] if item["candidate_id"] == "text_safe_001")
    assert repaired["repair_targets"] == removal_checks
    assert repaired["repair_outcomes"][0]["status"] == state
    assert repaired["repair_eligible"] is (state == "resolved")
    assert bool(repaired["observation"]["repair_needed"]) is (state == "unresolved")


def test_perfect_fact_score_cannot_hide_an_explicitly_unresolved_removal():
    context, candidate, payload, _, targets = removal_fixture()
    candidate["repair_targets"] = targets
    payload["observation"]["unsupported_claims"] = []
    payload["observation"]["repair_checks"] = [{"issue_id": targets[0]["issue_id"], "status": "unresolved",
        "observed": "The blue exterior assertion is still made.", "evidence": ["text: blue tower"],
        "confidence": .95, "reason": "The original conflicting claim remains."}]
    corrected = deepcopy(payload)
    corrected["scores"]["verified_fact_consistency"]["score"] = 2
    client = Client(payload, corrected)
    result = GPT4OMiniRankingJudge(client, use_rubrics=True).judge_text(context, candidate)
    assert result.score.details["contract_repair_retries"] == 1
    assert result.observation["repair_needed"]


@pytest.mark.parametrize("change", ["claim", "source", "orphan"])
def test_repair_cannot_change_its_target_fact_or_escape_its_parent(change):
    *_, targets = setup()
    original = observed_candidate(targets, status="contradicted")
    after = observed_candidate(targets, status="supported", candidate_id="image_safe_001", score=.99)
    after["source_candidate_id"] = "image_001"
    if change == "claim":
        after["observation"]["fact_checks"][1]["expected"] = "Changed target fact"
    elif change == "source":
        after["observation"]["fact_checks"][1]["evidence_refs"] = ["another-source"]
    else:
        after["source_candidate_id"] = "another-candidate"
    merged = _merge_reflection_ranking(ranked(original), ranked(after))
    assert merged["image_ranking"]["best_candidate_id"] == "image_001"
    assert after["repair_eligible"] is False
