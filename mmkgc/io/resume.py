"""Rebuild owned pipeline state from persisted stages without regenerating images."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from mmkgc.adapters.base import GeneratedImageArtifact
from mmkgc.data import EntityRecord, existing_image_paths
from mmkgc.pipeline import PreparedEntity
from mmkgc.io.legacy_evidence import convert_saved_checks
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.relation_planning import apply_candidate_slot_plans
from mmkgc.stages.candidate_planning import PLAN_SCHEMAS, LEGACY_PLAN_SCHEMA, apply_candidate_plans
from mmkgc.stages.candidate_planning import FACT_SELECTION_SCHEMA, apply_fact_selection


def _validate_saved_media_sources(knowledge: dict[str, Any], record: EntityRecord) -> None:
    from mmkgc.adapters.image_knowledge import image_modeling_entries
    images = image_modeling_entries(knowledge.get("image_modeling", {}))
    if images and existing_image_paths(record) != [entry["original_image"]["asset"] for entry in images]:
        raise ValueError("Current source image assets differ from the saved generation inputs")
    for entry in images:
        original = entry["original_image"]
        path = Path(original["resolved_path"])
        if (not path.is_file() or path.stat().st_size != original["byte_count"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != original["sha256"]):
            raise ValueError("Current source image bytes differ from the saved generation inputs")
    for modality, entry in knowledge.get("media_modeling", {}).items():
        original = entry["original_media"]
        if str(record.existing_modalities.get(modality)) != original["asset"]:
            raise ValueError(f"Current source {modality} asset differs from the saved generation inputs")
        path = Path(original["resolved_path"])
        if (not path.is_file() or path.stat().st_size != original["byte_count"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != original["sha256"]):
            raise ValueError(f"Current source {modality} bytes differ from the saved generation inputs")


def load_saved_image_run(entity_dir: str | Path, record: EntityRecord, candidate_budget: int) -> tuple[PreparedEntity, dict[str, Any]]:
    """Backward-compatible image-only entry point."""
    return load_saved_run(entity_dir, record, candidate_budget, missing_mode="image")


def load_saved_run(entity_dir: str | Path, record: EntityRecord, candidate_budget: int,
                   *, missing_mode: str) -> tuple[PreparedEntity, dict[str, Any]]:
    """Validate saved inputs and reuse Stage 0-3 for text or image evaluation."""
    if missing_mode not in {"text", "image"}:
        raise ValueError("missing_mode must be text or image")
    root = Path(entity_dir).resolve(strict=True)
    def read(path):
        return json.loads(path.read_text())
    def stage(name):
        return read(root / name)["data"]
    calls = [json.loads(line) for line in (root / "model_calls.jsonl").read_text().splitlines()]
    v3 = stage("1_knowledge_modeling.json")
    if v3.get("schema_version") not in {"qwen_grounded_knowledge_v1", "grounded_triples_v1", "grounded_triples_v2"}:
        raise ValueError("Saved image run does not contain full grounded knowledge")
    profile = v3["entity_profile"]
    knowledge = {
        "schema_version": v3["schema_version"], "stage1_v3": {
            key: deepcopy(value) for key, value in v3.items()
            if key not in {"relation_facts", "triples", "evidence_catalog", "entity", "relevance_selection", "image_modeling", "media_modeling", "source_inputs", "source_extraction"}},
        "entity_profile": {"entity_type": profile["primary_type"], "type": profile["primary_type"],
            "secondary_types": profile.get("secondary_types", []), "semantic_domain": profile.get("semantic_domains", []),
            "core_identity": profile["core_identity"], "type_evidence": profile.get("evidence_refs", []),
            "uncertain_aspects": v3.get("uncertain_aspects", [])},
        "text_visual_profile": v3.get("text_visual_profile", {}),
        "relation_facts": v3.get("triples", v3.get("relation_facts", [])),
        "evidence_catalog": v3["evidence_catalog"], "existing_modalities": v3["modality_state"]["existing"],
        "uncertainties": v3.get("uncertain_aspects", []),
    }
    if v3.get("schema_version") in {"grounded_triples_v1", "grounded_triples_v2"}:
        knowledge["entity"] = deepcopy(v3["entity"])
        if knowledge["entity"] != {"entity_key": record.entity_key, "label": record.label}:
            raise ValueError("Saved triple knowledge belongs to another entity")
    if "relevance_selection" in v3:
        knowledge["relevance_selection"] = deepcopy(v3["relevance_selection"])
    for key in ("image_modeling", "media_modeling", "source_inputs", "source_extraction", "generation_task"):
        if key in v3:
            knowledge[key] = deepcopy(v3[key])
    # Early opt-in description runs wrote their explicit task marker only in
    # Stage 3. Recover that declared policy, then retain the exact check match.
    from mmkgc.visual_description import TASK
    saved_candidates = stage('3_relation_alternation_generation.json').get('text_candidates', [])
    if (missing_mode == 'text' and not knowledge.get('generation_task') and saved_candidates
            and all(c.get('description_task') == TASK for c in saved_candidates)):
        knowledge['generation_task'] = knowledge['stage1_v3']['generation_task'] = TASK
    _validate_saved_media_sources(knowledge, record)
    original_sources = knowledge.get("source_inputs", knowledge["existing_modalities"])
    if original_sources.get("text") != record.existing_modalities.get("text"):
        raise ValueError("Current entity text differs from the saved generation inputs")
    if v3.get("schema_version") == "grounded_triples_v2":
        from mmkgc.stages.sources import public_sources
        if original_sources != public_sources(record, missing_mode)[0]:
            raise ValueError("Current source modalities differ from the saved generation inputs")
    old_record_path = root / "entity_description.json"
    if old_record_path.exists():
        old = read(old_record_path)
        if (old["entity_key"], old["entity_id"]) != (record.entity_key, record.entity_id):
            raise ValueError("Saved entity identity differs from current record")
    plans = [call for call in calls if call.get("stage") == "stage2_selection" and call.get("status") == "applied"]
    if len(plans) != 1:
        raise ValueError("Expected one successful saved Stage 2 model plan")
    response = json.loads(plans[0]["response_text"])
    plan_schema = plans[0].get("response_schema_version")
    shared = plan_schema == FACT_SELECTION_SCHEMA
    simple = shared or plan_schema in PLAN_SCHEMAS
    selection = (apply_fact_selection(knowledge, response, missing_mode, plans[0]["max_selected_facts"])
                 if shared else apply_candidate_plans(knowledge, response, missing_mode, candidate_budget, schema_version=plan_schema)
                 if simple else apply_candidate_slot_plans(knowledge, response, missing_mode, candidate_budget))
    selection, reasoning = compile_plan_context(knowledge, selection, missing_mode)
    payload = selection["fine_grained_relation_selection"]
    by_plan = ({plan["plan_id"]: plan for plan in payload["candidates"]} if simple else
               {plan["plan_id"]: plan for role in ("primary", "secondary", "context", "excluded")
                for plan in payload[role + "_relations"]})
    generation = stage("3_relation_alternation_generation.json")
    if generation.get("candidate_budget") != candidate_budget:
        raise ValueError("Saved candidate budget differs from the current run")
    if generation.get("target_modalities") != [missing_mode]:
        raise ValueError("Saved generation target modality differs from requested evaluation")
    artifacts = []
    seen = set()
    for candidate in generation.get("text_candidates", []):
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            raise ValueError("Duplicate saved text candidate")
        seen.add(candidate_id)
        plan = by_plan[candidate["source_plan_id"]]
        convert_saved_checks(candidate, plan["verification_targets"])
        content = candidate.get("generated_content", {})
        if content.get("type") != "text" or not isinstance(content.get("value"), str) or not content["value"].strip():
            raise ValueError("Saved text candidate has no usable content")
        context = plan["generation_context"]
        candidate.update(used_evidence_triples=deepcopy(plan["evidence_triples"]),
            used_constraints=deepcopy(context["checks"]),
            generation_context=deepcopy(context), metadata={"status": "generated"})
        if simple:
            candidate.update(plan_schema_version=plan_schema, source_fact_refs=list(plan["fact_ids"]))
    for candidate in generation.get("image_candidates", []):
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            raise ValueError("Duplicate saved image candidate")
        seen.add(candidate_id)
        image_path = (root / "generated_images" / (candidate_id + ".png")).resolve(strict=True)
        metadata_path = image_path.with_suffix(".json")
        image_path.relative_to(root)
        metadata_path.resolve(strict=True).relative_to(root)
        metadata = read(metadata_path)
        if (metadata["entity_key"], metadata["candidate_id"]) != (record.entity_key, candidate_id):
            raise ValueError("Saved image metadata belongs to another candidate")
        plan = by_plan[candidate["source_plan_id"]]
        convert_saved_checks(candidate, plan["verification_targets"])
        context = plan["generation_context"]
        candidate.update(
            generated_content={"type": "image", "value": metadata["positive_prompt"],
                "status": "generated", "image_artifact_id": f"{record.entity_key}:{candidate_id}"},
            positive_prompt=metadata["positive_prompt"], negative_prompt=metadata.get("negative_prompt", ""),
            used_evidence_triples=deepcopy(plan["evidence_triples"]),
            used_constraints=deepcopy(context["checks"]),
            metadata=deepcopy(metadata),
        )
        if simple:
            candidate.update(plan_schema_version=plan_schema, source_fact_refs=list(plan["fact_ids"]))
            if plan_schema == LEGACY_PLAN_SCHEMA:
                candidate["focus"] = plan["focus"]
        candidate.pop("artifact", None)
        artifacts.append(GeneratedImageArtifact(record.entity_key, candidate_id,
            f"generated_images/{candidate_id}.png", f"generated_images/{candidate_id}.json",
            image_path.read_bytes(), metadata, {}))
    if shared:
        initial = [ref for ref in seen if not ref.startswith(f"{missing_mode}_safe_")]
        if len(initial) > candidate_budget or (payload["status"] == "ready" and not initial):
            raise ValueError("Saved candidates do not match the shared-fact generation budget")
    elif len(seen) < candidate_budget:
        raise ValueError("Saved Stage 3 is missing initial candidate images")
    preparation_stages = {"stage0_coarse", "stage1_image_description", "stage1_video_description", "stage1_audio_description",
                          "stage1_knowledge", "stage1_evidence_reranking", "stage1_relevance_reranking", "stage2_selection"}
    prepared = PreparedEntity(record, missing_mode, {"stage0_v3": stage("0_coarse_relation_filtering.json")},
        knowledge, selection, reasoning, [call for call in calls if call.get("stage") in preparation_stages])
    image_calls_path = root / "image_model_calls.jsonl"
    image_calls = [json.loads(line) for line in image_calls_path.read_text().splitlines()] if image_calls_path.exists() else []
    return prepared, {"relation_alternation_generation": generation, "generated_image_artifacts": artifacts,
                      "model_calls": [call for call in calls if call.get("stage") not in preparation_stages] + image_calls}
