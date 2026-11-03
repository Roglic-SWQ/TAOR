"""Scope image extraction batches and merge their source-bound triples."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from mmkgc.adapters.image_knowledge import image_modeling_entries
from mmkgc.stages.triples import apply_triple_response


def image_batch_input(inputs: dict[str, Any], original_refs: set[str]) -> dict[str, Any]:
    """Expose only descriptions and image IDs whose originals accompany this request."""
    result = deepcopy(inputs)
    entries = image_modeling_entries(inputs.get("image_modeling", {}))
    inactive = {ref for entry in entries if entry["original_image_ref"] not in original_refs
                for ref in (entry["original_image_ref"], entry["description_evidence_ref"])}
    result["evidence_catalog"] = [item for item in result["evidence_catalog"]
                                  if item["evidence_id"] not in inactive]
    modeling = result.get("image_modeling", {})
    if "images" in modeling:
        modeling["images"] = [entry for entry in modeling["images"] if entry["original_image_ref"] in original_refs]
        context = result["existing_modalities"]
        context["image_summaries"] = [entry for entry in context["image_summaries"]
                                      if entry["original_image_ref"] in original_refs]
        context["image_paths"] = [entry["original_image"]["asset"] for entry in modeling["images"]]
    return result


def merge_image_batches(inputs: dict[str, Any], batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebind all batches once, assigning stable fact IDs and unioning exact duplicates."""
    if len(batches) == 1:
        return batches[0]
    source_refs = {item["evidence_id"] for item in inputs["evidence_catalog"]
                   if item.get("source") == "existing_modality"}
    triples, gaps, seen_gaps = [], [], set()
    profile = deepcopy(batches[0]["stage1_v3"]["entity_profile"])
    for batch in batches:
        for fact in batch["relation_facts"]:
            refs = [ref for ref in fact["evidence_refs"] if ref in source_refs]
            if refs:
                triples.append({**{key: fact[key] for key in ("subject", "relation", "object")},
                                "evidence_refs": refs})
        for gap in batch["uncertainties"]:
            key = json.dumps(gap, sort_keys=True, ensure_ascii=False)
            if key not in seen_gaps:
                gaps.append(deepcopy(gap))
                seen_gaps.add(key)
        other = batch["stage1_v3"]["entity_profile"]
        if all(other[key] == profile[key] for key in ("primary_type", "core_identity")):
            profile["evidence_refs"] = sorted(set(profile["evidence_refs"] + other["evidence_refs"]))
    result = apply_triple_response(inputs, {"entity_profile": profile,
        "extracted_triples": triples, "uncertain_aspects": gaps})
    if "semantic_refinement" in batches[0]:
        result["semantic_refinement"] = deepcopy(batches[0]["semantic_refinement"])
    return result
