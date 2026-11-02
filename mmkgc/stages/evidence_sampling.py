"""Bound model views while keeping the complete Stage 0 fact catalogue."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from mmkgc.utils import token_cosine, tokenize


SAMPLING_VERSION = "stage0_representative_evidence_v1"
RELATION_BUDGET = 24
FACTS_PER_RELATION = 3
RECORD_NEIGHBOR_BUDGET = 32
KNOWLEDGE_FACT_BUDGET = 32
_VISUAL_CUES = set("color material shape depicts image flag logo cover building tower bridge "
                   "river mountain water forest portrait facade red blue green stone wood metal".split())
_TEXT_CUES = set("instance genre occupation author creator director country origin subject "
                 "part type organization work language field".split())


def neighbor_key(item: dict[str, Any]) -> tuple[str, ...]:
    """Order by source identity, never by triple-file position."""
    return tuple(str(item[key]) if item.get(key) is not None else "" for key in
                 ("relation", "direction", "entity_key", "entity_id", "label", "description"))


def available_context(record: Any, missing_mode: str) -> str:
    key = "text" if missing_mode == "image" else "image_summary"
    return str(record.existing_modalities.get(key) or "")


def public_neighbor(record: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Do not use a target's hidden description through a self-neighbor."""
    result = dict(item)
    if ((record.entity_key and item.get("entity_key") == record.entity_key)
            or (record.label and item.get("label") == record.label)):
        result["description"] = result["neighbor_description"] = None
    return result


def representative_neighbors(
    neighbors: list[dict[str, Any]], missing_mode: str, context: str, limit: int = FACTS_PER_RELATION,
) -> list[dict[str, Any]]:
    """Heuristic relevance/rarity plus greedy diversity; scores are not confidence."""
    ordered = sorted(neighbors, key=neighbor_key)
    texts = [" ".join(str(item.get(key) or "") for key in ("label", "description"))
             for item in ordered]
    tokens = [set(tokenize(text)) for text in texts]
    frequencies = Counter(token for values in tokens for token in values)
    cue_terms = _VISUAL_CUES if missing_mode == "image" else _TEXT_CUES
    scores = []
    for item, text, values in zip(ordered, texts, tokens):
        relevance = token_cosine(context, text) if context.strip() else 0.0
        modality_fit = min(1.0, len(values & cue_terms) / 2)
        degree = item.get("graph_degree")
        if isinstance(degree, int) and not isinstance(degree, bool) and degree >= 0:
            distinctiveness = 1 / (1 + math.log1p(degree))
        else:
            distinctiveness = sum(1 / frequencies[token] for token in values) / max(1, len(values))
        representation = sum(frequencies[token] / max(1, len(ordered)) for token in values) / max(1, len(values))
        scores.append(0.5 * relevance + 0.2 * modality_fit + 0.15 * distinctiveness + 0.15 * representation)
    selected: list[int] = []
    remaining = list(range(len(ordered)))
    while remaining and len(selected) < limit:
        def utility(index: int) -> float:
            overlap = max((len(tokens[index] & tokens[other]) / max(1, len(tokens[index] | tokens[other]))
                           for other in selected), default=0.0)
            return scores[index] + (0.25 * (1 - overlap) if selected else 0.0)

        best = max(remaining, key=utility)
        selected.append(best)
        remaining.remove(best)
    return [ordered[index] for index in selected]


def representative_relations(groups: list[dict[str, Any]], missing_mode: str) -> list[dict[str, Any]]:
    """Prefer useful evidence, with complementary relation summaries under a budget."""
    remaining = sorted(groups, key=lambda item: (item["relation"], item["direction"]))
    selected: list[dict[str, Any]] = []
    priority = {"low": 0.0, "medium": 0.5, "high": 1.0}
    while remaining and len(selected) < RELATION_BUDGET:
        def utility(item: dict[str, Any]) -> float:
            terms = set(tokenize(item["relation"]))
            overlap = max((len(terms & set(tokenize(other["relation"]))) /
                           max(1, len(terms | set(tokenize(other["relation"]))))
                           for other in selected), default=0.0)
            return (float(item["coarse_score"]) + float(item.get("context_relevance", 0))
                    + priority.get(item.get("target_modality_priority", {}).get(missing_mode), 0)
                    + 0.5 * (1 - overlap))

        best = max(remaining, key=utility)
        selected.append(best)
        remaining.remove(best)
    return selected


def compact_stage0_evidence(coarse: dict[str, Any], *, include_selection_limit: bool) -> dict[str, Any]:
    """Serialize the same sampled candidate set for Qwen and GPT prompt profiles."""
    candidates = coarse["high_recall_relation_candidates"]
    result = {
        "high_recall_relation_candidates": [{
            "relation": item["relation"], "direction": item["direction"],
            "coarse_score": item["coarse_score"], "frequency": item["frequency"],
            "neighbor_entities": item["prompt_neighbor_entities"],
            "evidence_triples": item["prompt_evidence_triples"],
            "target_modality_priority": item["target_modality_priority"],
            "relation_semantic_summary": item["relation_semantic_summary"],
            "path_context": item["path_context"][:2],
        } for item in candidates],
        "cleaned_relation_groups": [{"relation": item["relation"], "direction": item["direction"]}
                                    for item in candidates],
        "rule_baseline_selected_relations": [
            {"relation": item["relation"], "direction": item["direction"]}
            for item in coarse.get("rule_baseline_selected_relations", coarse.get("selected_relations", []))[:8]],
        "evidence_sampling": dict(coarse["evidence_sampling"]),
    }
    if include_selection_limit:
        result["max_selected_relations"] = len(candidates)
    return result
