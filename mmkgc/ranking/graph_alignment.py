"""Structured text-KG and visual-scene-graph alignment for Stage 4."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mmkgc.utils import clamp01, token_cosine


Similarity = Callable[[str, str], float]


_VISUAL_CANONICAL_GROUPS = (
    ("scene", {"scene", "view", "setting", "location", "place", "landscape"}),
    ("environment", {"environment", "environmental", "natural", "nature", "landscape"}),
    ("vegetation", {"vegetation", "greenery", "tree", "trees", "plant", "plants", "forest"}),
    ("building", {"building", "house", "hut", "architecture", "structure"}),
    ("water", {"water", "river", "lake", "sea", "ocean", "coastline"}),
    ("road", {"road", "street", "path", "highway"}),
    ("person", {"person", "people", "man", "woman", "portrait"}),
)


def structured_kg_alignment(
    target_triples: Sequence[Mapping[str, Any]],
    candidate_triples: Sequence[Mapping[str, Any]],
    *,
    similarity: Similarity = token_cosine,
) -> dict[str, Any]:
    """Return confidence-weighted one-to-one triple matching metrics."""
    targets = [_text_triple(item) for item in target_triples]
    candidates = [_text_triple(item) for item in candidate_triples]
    targets = [item for item in targets if item is not None]
    candidates = [item for item in candidates if item is not None]
    result = _alignment_result(
        targets,
        candidates,
        lambda left, right: _triple_similarity(left, right, similarity),
    )
    return {"score": result["f1"], **result}


def visual_graph_alignment(
    target_graph: Mapping[str, Any],
    observed_graph: Mapping[str, Any],
    *,
    similarity: Similarity = token_cosine,
) -> dict[str, Any]:
    """Align visual graph nodes, attributes, edges, and OCR independently."""
    target_nodes = _visual_nodes(target_graph.get("nodes"))
    observed_nodes = _visual_nodes(observed_graph.get("nodes"))
    node_result = _alignment_result(
        target_nodes,
        observed_nodes,
        lambda left, right: _visual_item_similarity(left, right, "label", similarity),
    )

    target_attributes = _visual_attributes(target_graph.get("attributes"), target_nodes)
    observed_attributes = _visual_attributes(
        observed_graph.get("attributes"), observed_nodes
    )
    attribute_result = _alignment_result(
        target_attributes,
        observed_attributes,
        lambda left, right: _visual_item_similarity(left, right, "text", similarity),
    )

    target_edges = _visual_edges(target_graph.get("edges"), target_nodes)
    observed_edges = _visual_edges(observed_graph.get("edges"), observed_nodes)
    edge_result = _alignment_result(
        target_edges,
        observed_edges,
        lambda left, right: _triple_similarity(left, right, _visual_similarity),
    )

    target_ocr = _visual_ocr(target_graph.get("ocr"))
    observed_ocr = _visual_ocr(observed_graph.get("ocr"))
    ocr_result = _alignment_result(
        target_ocr,
        observed_ocr,
        lambda left, right: _visual_similarity(left["text"], right["text"], similarity),
    )

    configured = {
        "nodes": (0.50, node_result),
        "attributes": (0.30, attribute_result),
        "edges": (0.20, edge_result),
        "ocr": (0.0, ocr_result),
    }
    active = {
        name: (weight, result)
        for name, (weight, result) in configured.items()
        if result["target_count"] > 0
    }
    total_weight = sum(weight for weight, _result in active.values())
    if total_weight:
        score = sum(
            weight * float(result["f1"])
            for weight, result in active.values()
        ) / total_weight
    else:
        score = 0.0
    return {
        "score": round(clamp01(score), 6),
        "component_weights": {
            name: round(weight / total_weight, 6)
            for name, (weight, _result) in active.items()
        }
        if total_weight
        else {},
        "components": {
            "nodes": node_result,
            "attributes": attribute_result,
            "edges": edge_result,
            "ocr": ocr_result,
        },
    }


def _alignment_result(
    targets: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    pair_score: Callable[[dict[str, Any], dict[str, Any]], float],
) -> dict[str, Any]:
    pairs = sorted(
        (
            (
                clamp01(pair_score(target, candidate)),
                target_index,
                candidate_index,
            )
            for target_index, target in enumerate(targets)
            for candidate_index, candidate in enumerate(candidates)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_targets: set[int] = set()
    used_candidates: set[int] = set()
    matches: list[dict[str, Any]] = []
    matched_mass = 0.0
    for score, target_index, candidate_index in pairs:
        if target_index in used_targets or candidate_index in used_candidates:
            continue
        used_targets.add(target_index)
        used_candidates.add(candidate_index)
        target_weight = _target_weight(targets[target_index])
        confidence = _confidence(candidates[candidate_index])
        weighted_score = score * confidence * target_weight
        matched_mass += weighted_score
        matches.append(
            {
                "target_index": target_index,
                "candidate_index": candidate_index,
                "similarity": round(score, 6),
                "target_weight": round(target_weight, 6),
                "candidate_confidence": round(confidence, 6),
                "weighted_similarity": round(weighted_score, 6),
            }
        )

    target_mass = sum(_target_weight(item) for item in targets)
    candidate_mass = sum(_confidence(item) for item in candidates)
    recall = matched_mass / target_mass if target_mass else 0.0
    precision = matched_mass / candidate_mass if candidate_mass else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(clamp01(precision), 6),
        "recall": round(clamp01(recall), 6),
        "f1": round(clamp01(f1), 6),
        "target_count": len(targets),
        "candidate_count": len(candidates),
        "target_mass": round(target_mass, 6),
        "candidate_mass": round(candidate_mass, 6),
        "matches": matches,
    }

def _triple_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    similarity: Similarity,
) -> float:
    if left.get("polarity", "positive") != right.get("polarity", "positive"):
        return 0.0
    return clamp01(
        0.30 * similarity(str(left["subject"]), str(right["subject"]))
        + 0.40 * similarity(str(left["predicate"]), str(right["predicate"]))
        + 0.30 * similarity(str(left["object"]), str(right["object"]))
    )


def _visual_item_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    field: str,
    base_similarity: Similarity,
) -> float:
    left_values = [str(left.get(field) or ""), *_strings(left.get("aliases"))]
    right_values = [str(right.get(field) or ""), *_strings(right.get("aliases"))]
    scores = [
        _visual_similarity(left_value, right_value, base_similarity)
        for left_value in left_values
        for right_value in right_values
        if left_value.strip() and right_value.strip()
    ]
    score = max(scores, default=0.0)
    left_role = str(left.get("role") or "").casefold()
    right_role = str(right.get("role") or "").casefold()
    if left_role and right_role and left_role != right_role:
        score *= 0.75
    return round(clamp01(score), 6)


def _target_weight(value: Mapping[str, Any]) -> float:
    raw = value.get("weight", 1.0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 1.0
    return clamp01(float(raw))


def _visual_similarity(
    left: str,
    right: str,
    base_similarity: Similarity = token_cosine,
) -> float:
    """Match visual concepts despite harmless wording differences."""
    left_text = _strip_visual_prefixes(left)
    right_text = _strip_visual_prefixes(right)
    direct = base_similarity(left_text, right_text)
    left_tokens = set(_visual_tokens(left_text))
    right_tokens = set(_visual_tokens(right_text))
    if not left_tokens or not right_tokens:
        return direct
    canonical_overlap = len(left_tokens & right_tokens) / max(
        1,
        max(len(left_tokens), len(right_tokens)),
    )
    return round(clamp01(max(direct, canonical_overlap)), 6)


def _strip_visual_prefixes(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    lowered = text.casefold()
    for prefix in (
        "entity type:",
        "text visual cue:",
        "visual cue:",
        "text context cue:",
        "context cue:",
    ):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _visual_tokens(value: str) -> list[str]:
    tokens = set(value.casefold().replace("-", " ").split())
    canonical = set(tokens)
    for label, members in _VISUAL_CANONICAL_GROUPS:
        if tokens & members:
            canonical.add(label)
    return sorted(canonical)


def _text_triple(value: Mapping[str, Any]) -> dict[str, Any] | None:
    subject = _text(value.get("subject"))
    predicate = _text(value.get("predicate"))
    object_value = _text(value.get("object"))
    if not subject or not predicate or not object_value:
        return None
    polarity = _text(value.get("polarity")) or "positive"
    if polarity not in {"positive", "negative"}:
        polarity = "positive"
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "polarity": polarity,
        "confidence": _confidence(value),
    }


def _visual_nodes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        label = _text(item.get("label"))
        if not label:
            continue
        result.append(
            {
                "id": _text(item.get("id")) or f"node_{index}",
                "label": label,
                "aliases": _strings(item.get("aliases")),
                "role": _text(item.get("role")),
                "attributes": _strings(item.get("attributes")),
                "weight": _target_weight(item),
                "confidence": _confidence(item),
            }
        )
    return result

def _visual_attributes(value: Any, nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping):
                text = _text(item.get("text") or item.get("attribute"))
                confidence = _confidence(item)
                weight = _target_weight(item)
                aliases = _strings(item.get("aliases"))
            else:
                text = _text(item)
                confidence = 1.0
                weight = 1.0
                aliases = []
            if text:
                result.append(
                    {
                        "text": text,
                        "aliases": aliases,
                        "confidence": confidence,
                        "weight": weight,
                    }
                )
    for node in nodes:
        for attribute in _strings(node.get("attributes")):
            result.append(
                {
                    "text": f"{node['label']} {attribute}",
                    "confidence": _confidence(node),
                    "weight": _target_weight(node),
                }
            )
    return result

def _visual_edges(value: Any, nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    labels = {str(node["id"]): str(node["label"]) for node in nodes}
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        subject = _text(item.get("subject"))
        predicate = _text(item.get("predicate"))
        object_value = _text(item.get("object"))
        if not subject or not predicate or not object_value:
            continue
        result.append(
            {
                "subject": labels.get(subject, subject),
                "predicate": predicate,
                "object": labels.get(object_value, object_value),
                "polarity": "positive",
                "confidence": _confidence(item),
            }
        )
    return result


def _visual_ocr(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if isinstance(item, Mapping):
            text = _text(item.get("text"))
            confidence = _confidence(item)
        else:
            text = _text(item)
            confidence = 1.0
        if text:
            result.append({"text": text, "confidence": confidence})
    return result


def _confidence(value: Mapping[str, Any]) -> float:
    raw = value.get("confidence", 1.0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 1.0
    return clamp01(float(raw))


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := _text(item))]


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""
