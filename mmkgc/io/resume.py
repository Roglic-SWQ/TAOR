"""Rebuild owned pipeline state from persisted stages without regenerating images."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from mmkgc.adapters.base import GeneratedImageArtifact
from mmkgc.data import EntityRecord
from mmkgc.pipeline import PreparedEntity
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.relation_planning import apply_candidate_slot_plans


def load_saved_image_run(entity_dir: str | Path, record: EntityRecord, candidate_budget: int) -> tuple[PreparedEntity, dict[str, Any]]:
    """Validate saved facts/plan and original candidate bytes, then reuse Stage 3."""
    root = Path(entity_dir).resolve(strict=True)
    def read(path):
        return json.loads(path.read_text())
    def stage(name):
        return read(root / name)["data"]
    calls = [json.loads(line) for line in (root / "model_calls.jsonl").read_text().splitlines()]
    v3 = stage("1_knowledge_modeling.json")
    if v3.get("schema_version") != "qwen_grounded_knowledge_v1":
        raise ValueError("Saved image run does not contain full grounded knowledge")
    profile = v3["entity_profile"]
    knowledge = {
        "schema_version": v3["schema_version"], "stage1_v3": {
            key: deepcopy(value) for key, value in v3.items() if key not in {"relation_facts", "evidence_catalog"}},
        "entity_profile": {"entity_type": profile["primary_type"], "type": profile["primary_type"],
            "secondary_types": profile.get("secondary_types", []), "semantic_domain": profile.get("semantic_domains", []),
            "core_identity": profile["core_identity"], "type_evidence": profile.get("evidence_refs", []),
            "uncertain_aspects": v3.get("uncertain_aspects", [])},
        "text_visual_profile": v3["text_visual_profile"], "relation_facts": v3["relation_facts"],
        "evidence_catalog": v3["evidence_catalog"], "existing_modalities": v3["modality_state"]["existing"],
        "uncertainties": v3.get("uncertain_aspects", []),
    }
    if knowledge["existing_modalities"].get("text") != record.existing_modalities.get("text"):
        raise ValueError("Current entity text differs from the saved generation inputs")
    old_record_path = root / "entity_description.json"
    if old_record_path.exists():
        old = read(old_record_path)
        if (old["entity_key"], old["entity_id"]) != (record.entity_key, record.entity_id):
            raise ValueError("Saved entity identity differs from current record")
    plans = [call for call in calls if call.get("stage") == "stage2_selection" and call.get("status") == "applied"]
    if len(plans) != 1:
        raise ValueError("Expected one successful saved Stage 2 model plan")
    selection = apply_candidate_slot_plans(knowledge, json.loads(plans[0]["response_text"]), "image", candidate_budget)
    selection, reasoning = compile_plan_context(knowledge, selection, "image")
    by_plan = {plan["plan_id"]: plan for role in ("primary", "secondary", "context", "excluded")
               for plan in selection["fine_grained_relation_selection"][role + "_relations"]}
    generation = stage("3_relation_alternation_generation.json")
    if generation.get("candidate_budget") != candidate_budget:
        raise ValueError("Saved candidate budget differs from the current run")
    artifacts = []
    seen = set()
    for candidate in generation["image_candidates"]:
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
        if candidate.get("verification_targets") != plan["verification_targets"]:
            raise ValueError("Saved candidate checks differ from its original fact plan")
        context = plan["generation_context"]
        candidate.update(
            generated_content={"type": "image", "value": metadata["positive_prompt"],
                "status": "generated", "image_artifact_id": f"{record.entity_key}:{candidate_id}"},
            positive_prompt=metadata["positive_prompt"], negative_prompt=metadata.get("negative_prompt", ""),
            used_evidence_triples=deepcopy(plan["evidence_triples"]),
            used_constraints=deepcopy(context["must_include"] + context["should_include"]),
            metadata=deepcopy(metadata),
        )
        candidate.pop("artifact", None)
        artifacts.append(GeneratedImageArtifact(record.entity_key, candidate_id,
            f"generated_images/{candidate_id}.png", f"generated_images/{candidate_id}.json",
            image_path.read_bytes(), metadata, {}))
    if len(seen) < candidate_budget:
        raise ValueError("Saved Stage 3 is missing initial candidate images")
    preparation_stages = {"stage0_coarse", "stage1_knowledge", "stage1_evidence_reranking", "stage2_selection"}
    prepared = PreparedEntity(record, "image", {"stage0_v3": stage("0_coarse_relation_filtering.json")},
        knowledge, selection, reasoning, [call for call in calls if call.get("stage") in preparation_stages])
    image_calls_path = root / "image_model_calls.jsonl"
    image_calls = [json.loads(line) for line in image_calls_path.read_text().splitlines()] if image_calls_path.exists() else []
    return prepared, {"relation_alternation_generation": generation, "generated_image_artifacts": artifacts,
                      "model_calls": [call for call in calls if call.get("stage") not in preparation_stages] + image_calls}
