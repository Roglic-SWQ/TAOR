"""Conservative use of image observations; weights are policy, not truth probabilities."""
from __future__ import annotations

from copy import deepcopy
import json
import math

VISUAL_FACT_POLICY = "visual_observation_policy_v1"
ENTITY_VISUAL_POLICY = "entity_visual_features_v2"
VISUAL_POLICIES = {VISUAL_FACT_POLICY, ENTITY_VISUAL_POLICY}
MIN_VISUAL_CONFIDENCE = 0.8


def visual_support_schema(policy=VISUAL_FACT_POLICY):
    properties = {
        "basis": {"type": "string", "enum": ["direct", "inferred"]},
        "scope": {"type": "string", "enum": ["target", "context", "background"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "string"},
    }
    if policy == ENTITY_VISUAL_POLICY:
        properties.pop("confidence")
        properties["feature_kind"] = {"type": "string", "enum": [
            "entity_feature", "depiction_detail", "context"]}
        properties["claim_scope"] = {"type": "string", "enum": ["depicted_instance"]}
        properties["uncertainty_scope"] = {"type": "string", "enum": ["none", "feature", "identity"]}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def valid_visual_support(value, policy=VISUAL_FACT_POLICY):
    if not isinstance(value, dict) or set(value) != set(visual_support_schema(policy)["properties"]):
        return False
    if policy == ENTITY_VISUAL_POLICY:
        return (value["basis"] in ("direct", "inferred")
                and value["scope"] in ("target", "context", "background")
                and value["feature_kind"] in ("entity_feature", "depiction_detail", "context")
                and value["claim_scope"] == "depicted_instance"
                and isinstance(value["uncertainty"], str)
                and value["uncertainty_scope"] in ("none", "feature", "identity")
                and ((value["uncertainty_scope"] == "none") == (not value["uncertainty"].strip()))
                and (value["basis"] != "inferred" or bool(value["uncertainty"].strip())))
    confidence = value["confidence"]
    return (value["basis"] in ("direct", "inferred")
            and value["scope"] in ("target", "context", "background")
            and type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1
            and isinstance(value["uncertainty"], str)
            and (value["basis"] != "inferred" or bool(value["uncertainty"].strip())))


def observation_weight(observation):
    """Never let a high self-rating turn a guess into an assertive generation fact."""
    if "feature_kind" in observation:
        return float(observation["basis"] == "direct" and observation["scope"] == "target"
                     and observation["feature_kind"] == "entity_feature"
                     and observation["uncertainty_scope"] in ("none", "identity"))
    if (observation["basis"] != "direct" or observation["uncertainty"].strip()
            or observation["scope"] == "background"
            or observation["confidence"] < MIN_VISUAL_CONFIDENCE):
        return 0.0
    return observation["confidence"] * (0.5 if observation["scope"] == "context" else 1.0)


def bind_visual_observations(fact, observations, catalog, policy=VISUAL_FACT_POLICY):
    """Keep each source assessment, without counting captions as extra corroboration."""
    observations = [{**item, "evidence_refs": sorted(item["evidence_refs"])} for item in observations]
    unique = {json.dumps(item, sort_keys=True, ensure_ascii=False): item for item in observations}
    observations = [deepcopy(unique[key]) for key in sorted(unique)]
    fact["visual_observations"] = observations
    # Text/graph assertions retain their own source policy. A weak image cannot
    # downgrade them or promote its own citation into positive support for them.
    other_refs = [ref for ref in fact["evidence_refs"] if catalog[ref].get("modality") != "image"]
    def admitted(item):
        if policy != ENTITY_VISUAL_POLICY:
            return True
        originals = [catalog[ref] for ref in item["evidence_refs"]
                     if catalog[ref].get("representation") == "original_image"]
        return bool(originals) and all(source.get("entity_match", {}).get("status") == "compatible"
                                       for source in originals)
    eligible = [item for item in observations if observation_weight(item) > 0 and admitted(item)]
    weight = 1.0 if other_refs else max((observation_weight(item) for item in eligible), default=0.0)
    positive_refs = set(other_refs) | {ref for item in eligible for ref in item["evidence_refs"]}
    if positive_refs:
        fact["evidence_refs"] = sorted(positive_refs)
    fact["source_weight"] = weight
    fact["generation_eligible"] = weight > 0


def generation_eligible(fact):
    return fact.get("generation_eligible", True) is True


def selection_priority(fact):
    return fact.get("selection_score", fact.get("relevance_score", 0))
