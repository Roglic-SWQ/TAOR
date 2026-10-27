"""Stage 4 model-fusion ranking engine."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any

from mmkgc.ranking.context import (
    RankingContext,
    build_kg_reference,
    build_structured_kg_triples,
    build_visual_constraint_text,
    candidate_generated_text,
)
from mmkgc.ranking.contracts import (
    GateResult,
    RequiredScorerError,
    ScoreResult,
)
from mmkgc.ranking.fusion import (
    IMAGE_CANDIDATE_WEIGHTS,
    TEXT_CANDIDATE_WEIGHTS,
    fuse_image_candidate_score,
    fuse_text_candidate_score,
    normalize_visual_quality,
)
from mmkgc.ranking.hard_gate import (
    deterministic_candidate_gate,
)
from mmkgc.ranking.graph_alignment import (
    structured_kg_alignment,
)
from mmkgc.ranking.image_input import ImageInput, ImageInputError
from mmkgc.ranking.profiles import Stage4Profile, required_scorers_for_mode
from mmkgc.ranking.scorers.gpt4omini_judge import GPT4OMiniJudgeParseError


SCORE_VERSION = "stage4_image_fusion_v4_no_visual_graph"
RANKER_NAME = "model_fusion"
_JUDGE_SCORER_KEY = "gpt4omini_judge"
JUDGE_OUTPUT_SCORER_NAME = "gpt4omini_judge"


class ModelFusionRankingEngine:
    """Coordinate Stage 4 ranker modules without owning model logic."""

    def __init__(
        self,
        *,
        profile: Stage4Profile,
        scorers: dict[str, Any],
    ) -> None:
        self.profile = profile
        self.scorers = dict(scorers)

    def rank(self, context: RankingContext) -> dict[str, Any]:
        required = sorted(
            required_scorers_for_mode(
                self.profile,
                context.missing_mode,
                has_existing_image=bool(
                    context.existing_image_input is not None
                    or context.existing_modalities.get("image") is not None
                ),
                has_existing_text=bool(context.existing_modalities.get("text")),
            )
        )
        self._ensure_required_scorers(required)
        required_set = set(required)
        scorer_context = _ScorerSafeContext(context)
        optional = self._optional_scorers_for_context(context, required_set)
        optional_set = set(optional)
        failed_optional_scorers: list[str] = []
        model_calls: list[dict[str, Any]] = []

        text_items = self._rank_single_modality(
            context,
            scorer_context,
            "text",
            model_calls,
            required_set,
            optional_set,
            failed_optional_scorers,
        )
        image_items = self._rank_single_modality(
            context,
            scorer_context,
            "image",
            model_calls,
            required_set,
            optional_set,
            failed_optional_scorers,
        )
        pair_ranking = _empty_pair_ranking()
        status = self._ranking_status(
            text_items,
            image_items,
            failed_optional_scorers,
        )

        return {
            "ranking_status": status,
            "ranker": RANKER_NAME,
            "score_version": SCORE_VERSION,
            "profile_name": self.profile.profile_name,
            "required_scorers": required,
            "optional_scorers": optional,
            "failed_optional_scorers": failed_optional_scorers,
            "candidate_score_weights": {
                "text": dict(TEXT_CANDIDATE_WEIGHTS),
                "image": dict(IMAGE_CANDIDATE_WEIGHTS),
            },
            "ranking_model_calls": model_calls,
            "kg_aware_candidate_consistency_ranking": {
                "ground_truth_used": context.ground_truth_used,
            },
            "text_ranking": self._candidate_ranking_payload(text_items),
            "image_ranking": self._candidate_ranking_payload(image_items),
            "pair_ranking": pair_ranking,
        }

    def _ensure_required_scorers(self, required: list[str]) -> None:
        missing = [
            name
            for name in required
            if name not in self.scorers
        ]
        if missing:
            raise RequiredScorerError(f"missing required scorer: {missing[0]}")

    def _optional_scorers_for_context(
        self,
        context: RankingContext,
        required: set[str],
    ) -> list[str]:
        candidates: list[str] = []
        if context.missing_mode == "text" and self.profile.sentence_bert.enabled:
            candidates.append("sentence_bert")
        if context.missing_mode == "text" and self.profile.bertscore.enabled:
            candidates.append("bertscore")
        needs_visual = context.missing_mode == "image" or (
            context.missing_mode == "text"
            and bool(
                context.existing_image_input is not None
                or context.existing_modalities.get("image") is not None
            )
        )
        if needs_visual:
            if self.profile.clip.enabled:
                candidates.append("clip")
            if self.profile.blip_itm.enabled:
                candidates.append("blip_itm")
        if not self.profile.gpt4omini_judge.required:
            candidates.append("gpt4omini_judge")
        return sorted(
            name
            for name in candidates
            if name not in required and name in self.scorers
        )

    def _rank_single_modality(
        self,
        context: RankingContext,
        scorer_context: Any,
        modality: str,
        model_calls: list[dict[str, Any]],
        required_scorers: set[str],
        optional_scorers: set[str],
        failed_optional_scorers: list[str],
    ) -> list[dict[str, Any]]:
        if context.missing_mode != modality:
            return []
        source_candidates = (
            list(context.text_candidates)
            if modality == "text"
            else list(context.image_candidates)
        )
        image_inputs = [
            self._image_input_for(context, candidate) if modality == "image" else None
            for candidate in source_candidates
        ]
        deterministic_gates = []
        for candidate, image_input in zip(source_candidates, image_inputs, strict=True):
            if modality == "image" and image_input is None:
                deterministic_gates.append(
                    GateResult(
                        "reject",
                        "deterministic",
                        hard_violations=("missing_image_input",),
                        reason="generated ImageInput is required for image scoring",
                    )
                )
                continue
            deterministic_gates.append(
                deterministic_candidate_gate(candidate, modality, image_input)
            )
        items = [
            self._candidate_shell(candidate, deterministic_gate)
            for candidate, deterministic_gate in zip(
                source_candidates,
                deterministic_gates,
                strict=True,
            )
        ]
        survivors = [
            (index, candidate)
            for index, (candidate, gate) in enumerate(
                zip(source_candidates, deterministic_gates, strict=True)
            )
            if gate.decision == "pass"
        ]
        if survivors:
            self._apply_judge_scores(
                scorer_context,
                modality,
                survivors,
                items,
                model_calls,
                required_scorers,
                optional_scorers,
                failed_optional_scorers,
            )
            live_survivors = [
                (index, candidate)
                for index, candidate in survivors
                if items[index]["hard_gate"]["decision"] == "pass"
            ]
            if modality == "text":
                self._apply_text_scores(
                    context,
                    live_survivors,
                    items,
                    model_calls,
                    required_scorers,
                    optional_scorers,
                    failed_optional_scorers,
                )
            else:
                self._apply_image_scores(
                    context,
                    live_survivors,
                    items,
                    model_calls,
                    required_scorers,
                    optional_scorers,
                    failed_optional_scorers,
                )
        for item in items:
            self._finalize_single_item(modality, item)
        return sorted(
            items,
            key=lambda item: (
                item["hard_gate"]["decision"] != "pass",
                -float(item["final_score"]),
                str(item.get("candidate_id", "")),
            ),
        )

    def _candidate_shell(
        self,
        candidate: dict[str, Any],
        deterministic_gate: GateResult,
    ) -> dict[str, Any]:
        return {
            **_plain(candidate),
            "model_scores": {},
            "auxiliary_metrics": {},
            "hard_gate": _gate_payload(deterministic_gate),
            "soft_violations": list(deterministic_gate.soft_violations),
            "candidate_score": 0.0,
            "fused_score": 0.0,
            "final_score": 0.0,
            "score": 0.0,
            "ranking_reason": deterministic_gate.reason,
        }

    def _apply_judge_scores(
        self,
        context: Any,
        modality: str,
        survivors: list[tuple[int, dict[str, Any]]],
        items: list[dict[str, Any]],
        model_calls: list[dict[str, Any]],
        required_scorers: set[str],
        optional_scorers: set[str],
        failed_optional_scorers: list[str],
    ) -> None:
        judge = self.scorers.get(_JUDGE_SCORER_KEY)
        if judge is None:
            return
        outcomes = self._judge_candidate_outcomes(judge, context, modality, survivors)
        for (index, candidate), (result, request_error, latency_ms) in zip(survivors, outcomes, strict=True):
            try:
                if request_error is not None:
                    raise request_error
                _validate_score_result(JUDGE_OUTPUT_SCORER_NAME, result.score)
            except Exception as exc:
                if _JUDGE_SCORER_KEY in optional_scorers:
                    _record_optional_failure(
                        JUDGE_OUTPUT_SCORER_NAME,
                        failed_optional_scorers,
                        model_calls,
                        kind=modality,
                        candidate_id=str(candidate.get("candidate_id", "")),
                        result=getattr(locals().get("result", None), "score", None),
                        exc=exc,
                    )
                    model_calls[-1]["latency_ms"] = latency_ms
                    continue
                if isinstance(exc, GPT4OMiniJudgeParseError):
                    gate = GateResult(
                        "reject",
                        JUDGE_OUTPUT_SCORER_NAME,
                        hard_violations=("malformed_candidate",),
                        reason=f"{JUDGE_OUTPUT_SCORER_NAME} parse error: {type(exc).__name__}",
                    )
                    items[index]["hard_gate"] = _gate_payload(gate)
                    items[index]["ranking_reason"] = gate.reason
                    model_calls.append(
                        _model_call(
                            scorer=JUDGE_OUTPUT_SCORER_NAME,
                            kind=modality,
                            candidate_id=str(candidate.get("candidate_id", "")),
                            result=ScoreResult(
                                raw_score=None,
                                normalized_score=None,
                                status="failed",
                                model_name=getattr(judge, "model_name", JUDGE_OUTPUT_SCORER_NAME),
                                details=_judge_parse_error_details(exc),
                            ),
                            status="failed",
                        )
                    )
                    model_calls[-1]["latency_ms"] = latency_ms
                    continue
                if isinstance(exc, RequiredScorerError):
                    raise
                failed_call = _judge_failed_call(
                    judge,
                    kind=modality,
                    candidate_id=str(candidate.get("candidate_id", "")),
                    exc=exc,
                )
                model_calls.append(failed_call)
                model_calls[-1]["latency_ms"] = latency_ms
                gate = GateResult(
                    "reject",
                    JUDGE_OUTPUT_SCORER_NAME,
                    hard_violations=("judge_service_error",),
                    reason=f"{JUDGE_OUTPUT_SCORER_NAME} service error: {type(exc).__name__}",
                )
                items[index]["hard_gate"] = _gate_payload(gate)
                items[index]["ranking_reason"] = gate.reason
                continue
            score = _result_score(result.score)
            if modality == "image":
                score = _normalized_visual_score(result.score)
            gate = result.gate
            items[index]["model_scores"][JUDGE_OUTPUT_SCORER_NAME] = score
            if modality == "image":
                items[index]["model_scores"][f"{JUDGE_OUTPUT_SCORER_NAME}_raw_score"] = _raw_visual_score_0_100(
                    result.score
                )
                items[index]["model_scores"][f"{JUDGE_OUTPUT_SCORER_NAME}_normalized_score"] = _result_score(
                    result.score
                )
            items[index]["hard_gate"] = _gate_payload(gate)
            items[index]["soft_violations"] = list(gate.soft_violations)
            items[index]["ranking_reason"] = gate.reason
            model_calls.append(
                _model_call(
                    scorer=JUDGE_OUTPUT_SCORER_NAME,
                    kind=modality,
                    candidate_id=str(candidate.get("candidate_id", "")),
                    result=result.score,
                    status="rejected" if gate.decision == "reject" else "scored",
                    gate=gate,
                )
            )
            model_calls[-1]["latency_ms"] = latency_ms

    def _judge_candidate_outcomes(
        self,
        judge: Any,
        context: Any,
        modality: str,
        survivors: list[tuple[int, dict[str, Any]]],
    ):
        """Run independent remote calls concurrently; merge results in input order."""
        def invoke(entry):
            _index, candidate = entry
            started = time.perf_counter()
            try:
                if modality == "text":
                    result = judge.judge_text(context, _plain(candidate))
                else:
                    result = judge.judge_image(
                        context, _plain(candidate),
                        image_input=self._image_input_for(context, candidate),
                    )
                return result, None, int((time.perf_counter() - started) * 1000)
            except Exception as exc:
                return None, exc, int((time.perf_counter() - started) * 1000)

        workers = self.profile.gpt4omini_judge.max_concurrency
        if workers == 1 or len(survivors) < 2:
            for entry in survivors:
                yield invoke(entry)
            return
        with ThreadPoolExecutor(
            max_workers=min(workers, len(survivors)), thread_name_prefix="mmkgc-gpt-judge",
        ) as executor:
            # Workers never modify candidate scores, logs, or local GPU models.
            yield from executor.map(invoke, survivors)

    def _apply_text_scores(
        self,
        context: RankingContext,
        survivors: list[tuple[int, dict[str, Any]]],
        items: list[dict[str, Any]],
        model_calls: list[dict[str, Any]],
        required_scorers: set[str],
        optional_scorers: set[str],
        failed_optional_scorers: list[str],
    ) -> None:
        texts = [candidate_generated_text(candidate) for _index, candidate in survivors]
        reference = build_kg_reference(context, "text")
        if "sentence_bert" in self.scorers and texts:
            try:
                scores = self.scorers["sentence_bert"].score_candidates(reference, texts)
                _validate_score_batch("sentence_bert", scores, len(texts))
            except Exception as exc:
                if "sentence_bert" in optional_scorers:
                    _record_optional_failure(
                        "sentence_bert",
                        failed_optional_scorers,
                        model_calls,
                        kind="text",
                        candidate_id=_joined_candidate_ids(survivors),
                        result=_first_score_result(locals().get("scores", None)),
                        exc=exc,
                    )
                elif isinstance(exc, RequiredScorerError):
                    raise
                else:
                    raise RequiredScorerError(
                        f"sentence_bert failed: {type(exc).__name__}"
                    ) from exc
            else:
                for (index, candidate), score in zip(survivors, scores, strict=True):
                    items[index]["model_scores"]["sentence_bert"] = _result_score(score)
                    model_calls.append(
                        _model_call(
                            scorer="sentence_bert",
                            kind="text",
                            candidate_id=str(candidate.get("candidate_id", "")),
                            result=score,
                        )
                    )
        if "bertscore" in self.scorers and texts:
            try:
                scores = self.scorers["bertscore"].score_candidates(reference, texts)
                _validate_score_batch("bertscore", scores, len(texts))
            except Exception as exc:
                if "bertscore" in optional_scorers:
                    _record_optional_failure(
                        "bertscore",
                        failed_optional_scorers,
                        model_calls,
                        kind="text",
                        candidate_id=_joined_candidate_ids(survivors),
                        result=_first_score_result(locals().get("scores", None)),
                        exc=exc,
                    )
                elif isinstance(exc, RequiredScorerError):
                    raise
                else:
                    raise RequiredScorerError(
                        f"bertscore failed: {type(exc).__name__}"
                    ) from exc
            else:
                for (index, candidate), score in zip(survivors, scores, strict=True):
                    items[index]["auxiliary_metrics"]["bertscore"] = _bertscore_payload(
                        score
                    )
                    model_calls.append(
                        _model_call(
                            scorer="bertscore",
                            kind="text",
                            candidate_id=str(candidate.get("candidate_id", "")),
                            result=score,
                        )
                    )
        target_triples = build_structured_kg_triples(context, "text")
        extractor = self.scorers.get(_JUDGE_SCORER_KEY)
        for index, candidate in survivors:
            if extractor is None or not hasattr(extractor, "extract_text_graph"):
                raise RequiredScorerError(
                    "gpt4omini_judge must support extract_text_graph for structured KG alignment"
                )
            try:
                candidate_graph = extractor.extract_text_graph(_plain(candidate))
            except Exception as exc:
                raise RequiredScorerError(
                    f"text graph extraction failed: {type(exc).__name__}"
                ) from exc
            alignment = structured_kg_alignment(
                target_triples,
                candidate_graph.get("triples", []),
            )
            score = float(alignment["score"])
            items[index]["auxiliary_metrics"]["structured_kg_alignment"] = {
                **_plain(alignment),
                "target_triples": _plain(target_triples),
                "candidate_triples": _plain(candidate_graph.get("triples", [])),
            }
            items[index]["model_scores"]["structured_kg_alignment"] = score
            model_calls.append(
                _model_call(
                    scorer="structured_kg_alignment",
                    kind="text",
                    candidate_id=str(candidate.get("candidate_id", "")),
                    result=ScoreResult(
                        raw_score=score,
                        normalized_score=score,
                        status="scored",
                        model_name=getattr(extractor, "model_name", "graph-extractor"),
                        details={
                            "target_count": alignment["target_count"],
                            "candidate_count": alignment["candidate_count"],
                            "precision": alignment["precision"],
                            "recall": alignment["recall"],
                            "f1": alignment["f1"],
                        },
                    ),
                )
            )

    def _apply_image_scores(
        self,
        context: RankingContext,
        survivors: list[tuple[int, dict[str, Any]]],
        items: list[dict[str, Any]],
        model_calls: list[dict[str, Any]],
        required_scorers: set[str],
        optional_scorers: set[str],
        failed_optional_scorers: list[str],
    ) -> None:
        for index, candidate in survivors:
            image_input = self._image_input_for(context, candidate)
            texts = [build_visual_constraint_text(context, candidate, clip_safe=True)]
            for scorer_name in ("clip", "blip_itm"):
                scorer = self.scorers.get(scorer_name)
                if scorer is None:
                    continue
                try:
                    scores = scorer.score_image_quality_batch(image_input, texts)
                    _validate_score_batch(scorer_name, scores, len(texts))
                except Exception as exc:
                    if scorer_name in optional_scorers:
                        _record_optional_failure(
                            scorer_name,
                            failed_optional_scorers,
                            model_calls,
                            kind="image",
                            candidate_id=str(candidate.get("candidate_id", "")),
                            result=_first_score_result(locals().get("scores", None)),
                            exc=exc,
                        )
                        continue
                    if isinstance(exc, RequiredScorerError):
                        raise
                    raise RequiredScorerError(
                        f"{scorer_name} failed: {type(exc).__name__}"
                    ) from exc
                score = scores[0]
                items[index]["model_scores"][scorer_name] = _normalized_visual_score(score)
                items[index]["model_scores"][f"{scorer_name}_raw_score"] = _raw_visual_score_0_100(
                    score
                )
                items[index]["model_scores"][f"{scorer_name}_normalized_score"] = _result_score(
                    score
                )
                model_calls.append(
                    _model_call(
                        scorer=scorer_name,
                        kind="image",
                        candidate_id=str(candidate.get("candidate_id", "")),
                        result=score,
                    )
                )


    def _finalize_single_item(
        self,
        modality: str,
        item: dict[str, Any],
    ) -> None:
        if item["hard_gate"]["decision"] == "reject":
            item["candidate_score"] = 0.0
            item["fused_score"] = 0.0
            item["final_score"] = 0.0
            item["score"] = 0.0
            return

        if modality == "text":
            structured = item["model_scores"].get("structured_kg_alignment", 0.0)
            sentence_bert = item["model_scores"].get("sentence_bert", structured)
            judge = _judge_model_score(item["model_scores"], structured)
            final = fuse_text_candidate_score(
                sentence_bert=sentence_bert,
                structured_kg_alignment=structured,
                judge=judge,
            )
        else:
            clip = item["model_scores"].get("clip", 0.0)
            blip = item["model_scores"].get("blip_itm", clip)
            judge = _judge_model_score(item["model_scores"], clip)
            final = fuse_image_candidate_score(
                clip=clip,
                blip=blip,
                judge=judge,
            )
        item["model_scores"]["candidate_score"] = final
        item["candidate_score"] = final
        item["fused_score"] = final
        item["final_score"] = final
        item["score"] = final
        if not item["ranking_reason"]:
            item["ranking_reason"] = f"flat {modality} candidate score"

    def _image_input_for(
        self,
        context: RankingContext,
        candidate: dict[str, Any],
    ) -> ImageInput | None:
        candidate_id = str(candidate.get("candidate_id", ""))
        image_input = context.generated_image_inputs.get(candidate_id)
        if isinstance(image_input, ImageInput):
            return image_input
        if isinstance(image_input, dict):
            return _image_input_from_transport(
                context.entity_key,
                candidate_id,
                image_input,
            )
        return None

    def _candidate_ranking_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        pass_items = [
            item for item in items if item["hard_gate"]["decision"] == "pass"
        ]
        top_5 = pass_items[:5]
        return {
            "best_candidate_id": top_5[0].get("candidate_id") if top_5 else None,
            "ranked_candidates": items,
            "top_5_candidates": top_5,
            "rejected_candidates": [
                item for item in items if item["hard_gate"]["decision"] == "reject"
            ],
        }

    def _ranking_status(
        self,
        text_items: list[dict[str, Any]],
        image_items: list[dict[str, Any]],
        failed_optional_scorers: list[str],
    ) -> str:
        if not _has_survivor(text_items) and not _has_survivor(image_items):
            return "no_valid_candidate"
        if failed_optional_scorers:
            return "degraded"
        return "completed"


def _gate_payload(gate: GateResult) -> dict[str, Any]:
    return {
        "decision": gate.decision,
        "source": gate.source,
        "hard_violations": list(gate.hard_violations),
        "soft_violations": list(gate.soft_violations),
        "judge_confidence": gate.judge_confidence,
        "evidence": list(gate.evidence),
        "reason": gate.reason,
    }


def _result_score(score: ScoreResult) -> float:
    if score.normalized_score is None:
        return 0.0
    return float(score.normalized_score)


def _raw_visual_score_0_100(score: ScoreResult) -> float:
    if score.raw_score is not None:
        return float(score.raw_score)
    return _result_score(score) * 100.0


def _normalized_visual_score(score: ScoreResult) -> float:
    return normalize_visual_quality(_raw_visual_score_0_100(score))


def _validate_score_batch(
    scorer_name: str,
    scores: Any,
    expected_count: int,
) -> None:
    if not isinstance(scores, list) or len(scores) != expected_count:
        raise RequiredScorerError(
            f"{scorer_name} returned {len(scores) if isinstance(scores, list) else 'non-list'} "
            f"scores for {expected_count} candidates"
        )
    for score in scores:
        _validate_score_result(scorer_name, score)


def _validate_score_result(scorer_name: str, score: Any) -> None:
    if not isinstance(score, ScoreResult):
        raise RequiredScorerError(f"{scorer_name} returned an invalid score result")
    if score.status != "scored" or score.normalized_score is None:
        raise RequiredScorerError(
            f"{scorer_name} returned {score.status} without a normalized score"
        )


def _record_optional_failure(
    scorer_name: str,
    failed_optional_scorers: list[str],
    model_calls: list[dict[str, Any]],
    *,
    kind: str,
    candidate_id: str,
    result: Any,
    exc: Exception,
) -> None:
    if scorer_name not in failed_optional_scorers:
        failed_optional_scorers.append(scorer_name)
    if isinstance(result, ScoreResult):
        model_calls.append(
            _model_call(
                scorer=scorer_name,
                kind=kind,
                candidate_id=candidate_id,
                result=result,
                status="optional_failed",
            )
        )
        return
    model_calls.append(
        _model_call(
            scorer=scorer_name,
            kind=kind,
            candidate_id=candidate_id,
            result=ScoreResult(
                raw_score=None,
                normalized_score=None,
                status="failed",
                model_name=scorer_name,
                details={"error_type": type(exc).__name__},
            ),
            status="optional_failed",
        )
    )


def _judge_parse_error_details(exc: GPT4OMiniJudgeParseError) -> dict[str, Any]:
    details = {"error_type": type(exc).__name__}
    error_details = getattr(exc, "details", {})
    if isinstance(error_details, dict):
        details.update(error_details)
    return details


def _judge_service_error_details(exc: Exception) -> dict[str, Any]:
    details: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    diagnostic = getattr(exc, "gpt4omini_judge_diagnostic", None)
    if isinstance(diagnostic, dict):
        details.update(_safe_details(diagnostic))
    return details


def _judge_failed_call(
    judge: Any,
    *,
    kind: str,
    candidate_id: str,
    exc: Exception,
) -> dict[str, Any]:
    details = _judge_service_error_details(exc)
    latency_ms = details.get("latency_ms", 0)
    if not isinstance(latency_ms, int) or latency_ms < 0:
        latency_ms = 0
    return _model_call(
        scorer=JUDGE_OUTPUT_SCORER_NAME,
        kind=kind,
        candidate_id=candidate_id,
        result=ScoreResult(
            raw_score=None,
            normalized_score=None,
            status="failed",
            model_name=getattr(judge, "model_name", JUDGE_OUTPUT_SCORER_NAME),
            latency_ms=latency_ms,
            details=details,
        ),
        status="failed",
    )


def _judge_model_score(model_scores: dict[str, Any], fallback: float) -> float:
    if JUDGE_OUTPUT_SCORER_NAME in model_scores:
        return model_scores.get(JUDGE_OUTPUT_SCORER_NAME, fallback)
    return model_scores.get(_JUDGE_SCORER_KEY, fallback)


def _raise_required_scorer_error_with_diagnostics(
    message: str,
    model_calls: list[dict[str, Any]],
    failed_call: dict[str, Any],
    exc: Exception,
) -> None:
    error = RequiredScorerError(message)
    error.diagnostic_events = [*model_calls, failed_call]
    raise error from exc


def _first_score_result(scores: Any) -> ScoreResult | None:
    if isinstance(scores, list) and scores and isinstance(scores[0], ScoreResult):
        return scores[0]
    return None


def _bertscore_payload(score: ScoreResult) -> dict[str, Any]:
    details = dict(score.details)
    return {
        "precision": details.get("precision"),
        "recall": details.get("recall"),
        "f1": details.get("f1", score.raw_score),
        "normalized_score": score.normalized_score,
        "reference_kind": details.get("reference_kind"),
        "model_name": score.model_name,
        "model_version": score.model_version,
    }


def _joined_candidate_ids(survivors: list[tuple[int, dict[str, Any]]]) -> str:
    return ",".join(str(candidate.get("candidate_id", "")) for _index, candidate in survivors)


def _model_call(
    *,
    scorer: str,
    kind: str,
    candidate_id: str,
    result: ScoreResult,
    status: str | None = None,
    gate: GateResult | None = None,
) -> dict[str, Any]:
    call = {
        "scorer": scorer,
        "kind": kind,
        "candidate_id": candidate_id,
        "status": status or result.status,
        "raw_score": result.raw_score,
        "normalized_score": result.normalized_score,
        "model_name": result.model_name,
        "model_version": result.model_version,
        "latency_ms": result.latency_ms,
        "details": _safe_details(result.details),
    }
    if gate is not None:
        call["gate"] = _gate_payload(gate)
    return call


def _safe_details(details: Any) -> dict[str, Any]:
    safe = _scrub_sentinels(_plain(details))
    return safe if isinstance(safe, dict) else {}


def _image_input_from_transport(
    entity_key: str,
    candidate_id: str,
    image_input: dict[str, Any],
) -> ImageInput | None:
    image_bytes = image_input.get("image_bytes", image_input.get("bytes"))
    if isinstance(image_bytes, bytes):
        try:
            return ImageInput.from_bytes(
                entity_key=entity_key,
                candidate_id=candidate_id,
                image_bytes=image_bytes,
            )
        except ImageInputError:
            return None
    image_path = image_input.get("image_path", image_input.get("path"))
    if isinstance(image_path, Path):
        try:
            return ImageInput(
                entity_key=entity_key,
                candidate_id=candidate_id,
                media_type=str(image_input.get("media_type", "image/png")),
                image_path=image_path,
            )
        except ImageInputError:
            return None
    return None


def _has_survivor(items: list[dict[str, Any]]) -> bool:
    return any(item["hard_gate"]["decision"] == "pass" for item in items)


def _empty_pair_ranking() -> dict[str, Any]:
    return {
        "ranked_pairs": [],
        "top_5_pairs": [],
        "best_pair_id": None,
        "best_pair": None,
    }


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


class _ScorerSafeContext:
    def __init__(self, context: RankingContext) -> None:
        self._context = context

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)

    def to_prompt_payload(self) -> dict[str, Any]:
        return _scrub_sentinels(self._context.to_prompt_payload())


def _scrub_sentinels(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            scrubbed = _scrub_sentinels(item)
            if scrubbed is not _DROP:
                result[key] = scrubbed
        return result
    if isinstance(value, list):
        return [
            scrubbed
            for item in value
            if (scrubbed := _scrub_sentinels(item)) is not _DROP
        ]
    if isinstance(value, str) and any(
        marker in value
        for marker in ("ZZZ_", "GROUND_TRUTH", "INSTRUCTION_ONLY", "RATIONALE_ONLY")
    ):
        return _DROP
    return value


_DROP = object()
