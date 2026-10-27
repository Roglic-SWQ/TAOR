"""End-to-end MMKGC pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from mmkgc.adapters import (
    ImageGenerator,
    ImagePromptGenerator,
    TextGenerator,
)
from mmkgc.data import EntityRecord
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.image_input import ImageInput, ImageInputError
from mmkgc.stages import (
    build_stage3_feedback,
    coarse_relation_filtering,
    knowledge_modeling,
    relation_alternation_generation,
    safer_relation_alternation_generation,
)
from mmkgc.stages.semantic import StageSemanticPlanner
from mmkgc.adapters.evidence_reranker import score_knowledge_evidence
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge


class MMKGCPipeline:
    """Run the five-stage MMKGC mainline with injectable adapters."""

    @classmethod
    def from_default_mainline(cls, **kwargs) -> "MMKGCPipeline":
        from mmkgc.defaults import build_default_mainline_pipeline

        return build_default_mainline_pipeline(**kwargs)

    def __init__(
        self,
        candidate_budget: int = 5,
        text_generator: TextGenerator | None = None,
        image_prompt_generator: ImagePromptGenerator | None = None,
        image_generator: ImageGenerator | None = None,
        *,
        ranking_engine: Any | None = None,
        semantic_planner: StageSemanticPlanner | None = None,
        knowledge_planner: StageSemanticPlanner | None = None,
        evidence_reranker: Any | None = None,
        relation_planner: Any | None = None,
        stage4_config: str | Path | None = None,
        dataset_root: str | Path | None = None,
    ):
        self.candidate_budget = candidate_budget
        self.text_generator = text_generator
        self.image_prompt_generator = image_prompt_generator
        self.image_generator = image_generator
        self.semantic_planner = semantic_planner
        self.knowledge_planner = knowledge_planner if knowledge_planner is not None else semantic_planner
        self.evidence_reranker = evidence_reranker
        self.relation_planner = relation_planner
        self.stage4_config = Path(stage4_config or "configs/stage4-model-fusion.yaml")
        self.dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        self.project_root = Path(__file__).resolve().parents[1]
        self.ranking_engine = ranking_engine
        if self.ranking_engine is None:
            from mmkgc.ranking import factory

            self.ranking_engine = factory.build_model_fusion_engine(
                self.project_root,
                self.stage4_config,
            )

    def run(self, record: EntityRecord, missing_mode: str) -> dict[str, Any]:
        if missing_mode not in {"text", "image"}:
            raise ValueError("missing_mode must be one of: text, image")
        self._validate_mainline_components(missing_mode)
        if type(self.candidate_budget) is not int or self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be a positive integer")
        semantic_call_start = _semantic_call_count(self.semantic_planner)
        knowledge_call_start = _semantic_call_count(self.knowledge_planner)
        evidence_call_start = len(getattr(self.evidence_reranker, "calls", []))
        relation_call_start = len(getattr(self.relation_planner, "calls", []))
        coarse = coarse_relation_filtering(record, missing_mode)
        coarse = self.semantic_planner.refine_coarse(record, missing_mode, coarse)
        knowledge = knowledge_modeling(record, missing_mode, coarse, planner=self.knowledge_planner)
        knowledge = score_knowledge_evidence(knowledge, self.evidence_reranker)
        selection = self.relation_planner.plan(record, missing_mode, knowledge, self.candidate_budget)
        selection, reasoning = compile_plan_context(knowledge, selection, missing_mode)
        generation = relation_alternation_generation(
            record,
            missing_mode,
            knowledge,
            reasoning,
            selection,
            self.text_generator,
            self.image_prompt_generator,
            self.image_generator,
            candidate_budget=self.candidate_budget,
        )
        model_calls = _semantic_model_calls(
            self.semantic_planner,
            start_index=semantic_call_start,
        )
        if self.knowledge_planner is not self.semantic_planner:
            model_calls.extend(_semantic_model_calls(self.knowledge_planner, start_index=knowledge_call_start))
        model_calls.extend(getattr(self.evidence_reranker, "calls", [])[evidence_call_start:])
        model_calls.extend(getattr(self.relation_planner, "calls", [])[relation_call_start:])
        generation_model_calls, image_model_calls = _split_generation_model_calls(
            generation.get("model_calls", [])
        )
        model_calls.extend(generation_model_calls)
        generated_image_artifacts = list(generation.get("generated_image_artifacts", []))
        generation = {"relation_alternation_generation": generation["relation_alternation_generation"]}
        context = build_ranking_context(
            record.entity_key,
            record.label,
            missing_mode,
            record.existing_modalities,
            ranking_knowledge(knowledge, reasoning),
            reasoning,
            generation,
            _generated_image_inputs(generated_image_artifacts),
            existing_image_input=_existing_image_input(record, missing_mode, self.dataset_root),
        )
        ranking = self.ranking_engine.rank(context)
        ranking_model_calls = list(ranking.get("ranking_model_calls", []))
        stage3_feedback = build_stage3_feedback(ranking, attempt=1)
        if _needs_stage4_regeneration(ranking, stage3_feedback):
            original_ranking = ranking
            retry_generation = safer_relation_alternation_generation(
                record,
                missing_mode,
                knowledge,
                reasoning,
                selection,
                self.text_generator,
                self.image_prompt_generator,
                self.image_generator,
                candidate_budget=self.candidate_budget,
                failed_ranking=original_ranking,
                attempt=1,
                stage3_feedback=stage3_feedback,
            )
            retry_model_calls, retry_image_model_calls = _split_generation_model_calls(
                retry_generation.get("model_calls", [])
            )
            model_calls.extend(retry_model_calls)
            image_model_calls.extend(retry_image_model_calls)
            generated_image_artifacts = list(
                retry_generation.get("generated_image_artifacts", [])
            )
            generation = {
                "relation_alternation_generation": retry_generation[
                    "relation_alternation_generation"
                ]
            }
            context = build_ranking_context(
                record.entity_key,
                record.label,
                missing_mode,
                record.existing_modalities,
                ranking_knowledge(knowledge, reasoning),
                reasoning,
                generation,
                _generated_image_inputs(generated_image_artifacts),
                existing_image_input=_existing_image_input(record, missing_mode, self.dataset_root),
            )
            retry_ranking = self.ranking_engine.rank(context)
            ranking_model_calls.extend(retry_ranking.get("ranking_model_calls", []))
            ranking = {
                **retry_ranking,
                "ranking_model_calls": ranking_model_calls,
                "stage3_regeneration": {
                    "attempted": True,
                    "attempts": 1,
                    "initial_status": original_ranking.get("ranking_status"),
                    "final_status": retry_ranking.get("ranking_status"),
                    "triggered_by": "stage4_model_fusion",
                },
            }

        return {
            "entity": asdict(record),
            "coarse_relation_filtering": coarse,
            "knowledge_modeling": knowledge,
            "fine_grained_relation_selection": selection,
            "relation_alternation_generation": generation,
            "kg_aware_candidate_consistency_ranking": ranking,
            "model_calls": model_calls,
            "image_model_calls": image_model_calls,
            "ranking_model_calls": ranking_model_calls,
            "generated_image_artifacts": generated_image_artifacts,
            "final": _final_result(record, missing_mode, ranking),
        }

    def _validate_mainline_components(self, missing_mode: str) -> None:
        missing = []
        if self.semantic_planner is None:
            missing.append("semantic_planner")
        if self.evidence_reranker is None:
            missing.append("evidence_reranker")
        if self.relation_planner is None:
            missing.append("relation_planner")
        if missing_mode == "text" and self.text_generator is None:
            missing.append("text_generator")
        if missing_mode == "image":
            if self.image_prompt_generator is None:
                missing.append("image_prompt_generator")
            if self.image_generator is None:
                missing.append("image_generator")
        if self.ranking_engine is None:
            missing.append("ranking_engine")
        if missing:
            raise RuntimeError(
                "fixed mainline is missing required components: " + ", ".join(missing)
            )



def _needs_stage4_regeneration(
    ranking: dict[str, Any],
    stage3_feedback: dict[str, Any] | None = None,
) -> bool:
    if ranking.get("ranking_status") != "no_valid_candidate":
        return False
    if stage3_feedback is None:
        stage3_feedback = build_stage3_feedback(ranking, attempt=1)
    return bool(
        stage3_feedback.get("hard_violations")
        or stage3_feedback.get("soft_violations")
    )


def _semantic_call_count(planner: StageSemanticPlanner | None) -> int:
    calls = getattr(planner, "calls", None)
    if isinstance(calls, list):
        return len(calls)
    return 0


def _semantic_model_calls(
    planner: StageSemanticPlanner | None,
    *,
    start_index: int,
) -> list[dict[str, Any]]:
    calls = getattr(planner, "calls", None)
    if not isinstance(calls, list):
        return []
    planner_name = str(getattr(planner, "planner_name", "semantic-planner"))
    rows = []
    for call in calls[start_index:]:
        if not isinstance(call, dict):
            continue
        row = {
            "provider": planner_name,
            "stage": call.get("stage"),
            "status": call.get("status"),
            "prompt": call.get("prompt", ""),
            "response_text": call.get("response_text", ""),
            "error_type": call.get("error_type"),
            "error_message": call.get("error_message"),
            "ground_truth_used": False,
        }
        for key in (
            "contract_repair_retries",
            "contract_repair_error_type",
            "contract_repair_error",
            "contract_repair_prompt",
            "contract_repair_original_response_preview",
            "contract_repair_original_response_truncated",
            "contract_repair_response_text",
            "contract_local_salvage",
            "contract_local_salvage_error_type",
            "contract_local_salvage_error",
            "contract_local_salvage_original_response_preview",
            "contract_local_salvage_original_response_truncated",
            "contract_local_salvage_response_text",
            "contract_local_salvage_failed",
            "contract_local_salvage_failure_error_type",
            "contract_local_salvage_failure_error",
        ):
            if key in call:
                row[key] = call[key]
        rows.append(row)
    return rows


def _stage2_model_calls(selection: dict[str, Any], entity_key: str) -> list[dict[str, Any]]:
    rows = selection.get("model_calls", [])
    if not isinstance(rows, list):
        return []
    model_calls = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        enriched = dict(row)
        enriched.setdefault("entity_key", entity_key)
        model_calls.append(enriched)
    return model_calls


def _split_generation_model_calls(rows: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_calls = []
    image_model_calls = []
    if not isinstance(rows, list):
        return model_calls, image_model_calls
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("modality") == "image":
            image_model_calls.append(row)
        else:
            model_calls.append(row)
    return model_calls, image_model_calls


def _final_result(
    record: EntityRecord,
    missing_mode: str,
    ranking: dict[str, Any],
) -> dict[str, Any]:
    text_candidates = ranking.get("text_ranking", {}).get("ranked_candidates", [])
    image_candidates = ranking.get("image_ranking", {}).get("ranked_candidates", [])
    ranked_pairs = ranking.get("pair_ranking", {}).get("ranked_pairs", [])
    top_5_pairs = ranking.get("pair_ranking", {}).get("top_5_pairs", [])
    best_pair = ranking.get("pair_ranking", {}).get("best_pair")

    return {
        "entity": asdict(record),
        "missing_mode": missing_mode,
        "best_text_candidate_id": ranking.get("text_ranking", {}).get("best_candidate_id"),
        "best_image_candidate_id": ranking.get("image_ranking", {}).get("best_candidate_id"),
        "best_pair_id": ranking.get("pair_ranking", {}).get("best_pair_id"),
        "best_pair": best_pair,
        "top_5_pairs": top_5_pairs,
        "counts": {
            "text": len(text_candidates),
            "image": len(image_candidates),
            "pair": len(ranked_pairs),
        },
        "ranking_status": ranking.get("ranking_status"),
        "stage4_ranker": "model-fusion",
        "stage4_profile_name": ranking.get("profile_name"),
        "stage4_score_version": ranking.get("score_version"),
        "ground_truth_used": bool(
            ranking.get("kg_aware_candidate_consistency_ranking", {}).get(
                "ground_truth_used",
                False,
            )
        ),
    }


def _generated_image_inputs(artifacts: list[Any]) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        candidate_id = _artifact_value(artifact, "candidate_id")
        image_bytes = _artifact_value(artifact, "image_bytes")
        if not candidate_id or not isinstance(image_bytes, bytes) or not image_bytes:
            continue
        inputs[str(candidate_id)] = {"image_bytes": bytes(image_bytes)}
    return inputs


def _artifact_value(artifact: Any, key: str) -> Any:
    if isinstance(artifact, dict):
        return artifact.get(key)
    return getattr(artifact, key, None)


def _existing_image_input(
    record: EntityRecord,
    missing_mode: str,
    dataset_root: Path | None,
) -> ImageInput | None:
    if missing_mode != "text":
        return None
    image_path = record.existing_modalities.get("image")
    if not isinstance(image_path, str) or not image_path.strip():
        return None
    if dataset_root is None:
        raise ValueError("dataset_root is required to pass existing image input safely")

    root = dataset_root.resolve()
    candidate = Path(image_path)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        try:
            relative_path = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("existing image path must stay under dataset_root") from exc
    else:
        resolved_from_cwd = candidate.resolve(strict=False)
        try:
            relative_path = resolved_from_cwd.relative_to(root)
        except ValueError:
            relative_path = candidate

    try:
        return ImageInput.from_allowed_path(
            entity_key=record.entity_key,
            candidate_id="existing_image",
            allowed_root=root,
            relative_path=relative_path,
        )
    except ImageInputError:
        return None
