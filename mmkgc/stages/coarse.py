"""Stage 0: high-recall relation filtering."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from mmkgc.data import EntityRecord
from mmkgc.utils import token_cosine, tokenize

LOW_VALUE_TERMS = {
    "id",
    "identifier",
    "url",
    "website",
    "wikimedia",
    "commons",
    "category",
    "catalog",
    "archive",
    "freebase",
    "imdb",
    "viaf",
}

HIGH_VALUE_RELATIONS = {
    "instance of",
    "subclass of",
    "genre",
    "occupation",
    "country",
    "country of origin",
    "part of",
    "located in",
    "creator",
    "author",
    "director",
    "performer",
    "depicts",
    "influenced by",
    "main subject",
    "made from material",
    "use",
}

VISUAL_RELATION_TERMS = {
    "color",
    "material",
    "shape",
    "depicts",
    "image",
    "flag",
    "coat",
    "logo",
    "cover",
    "located",
}

TEXT_RELATION_TERMS = {
    "instance",
    "genre",
    "occupation",
    "author",
    "creator",
    "director",
    "country",
    "origin",
    "subject",
    "part",
}


def _priority_label(score: int) -> str:
    if score >= 2:
        return "high"
    if score == 1:
        return "medium"
    return "low"


def _relation_key(relation: str) -> str:
    return " ".join(tokenize(relation))


def _is_low_value_relation(relation: str) -> bool:
    return bool(set(tokenize(relation)) & LOW_VALUE_TERMS)


def _triple(record: EntityRecord, neighbor: dict[str, Any]) -> str:
    relation = neighbor.get("relation", "")
    neighbor_label = neighbor.get("label") or neighbor.get("entity_key") or "unknown"
    if neighbor.get("direction") == "in":
        return f"({neighbor_label}, {relation}, {record.label})"
    return f"({record.label}, {relation}, {neighbor_label})"


def _score_group(record: EntityRecord, missing_mode: str, relation: str, neighbors: list[dict[str, Any]]) -> float:
    relation_key = _relation_key(relation)
    relation_tokens = set(tokenize(relation))
    score = min(3.0, len(neighbors))

    if relation_key in HIGH_VALUE_RELATIONS:
        score += 2.2
    elif relation_tokens & (VISUAL_RELATION_TERMS | TEXT_RELATION_TERMS):
        score += 1.2

    neighbor_text = " ".join(
        str(value or "")
        for neighbor in neighbors
        for value in (
            neighbor.get("label"),
            neighbor.get("description"),
            neighbor.get("entity_key"),
        )
    )
    if neighbor_text.strip():
        score += 0.6

    if missing_mode == "image" and relation_tokens & VISUAL_RELATION_TERMS:
        score += 0.7
    if missing_mode == "text" and relation_tokens & TEXT_RELATION_TERMS:
        score += 0.7

    existing_text = " ".join(
        value
        for value in (
            record.existing_modalities.get("text"),
            record.existing_modalities.get("image_summary"),
        )
        if isinstance(value, str) and value.strip()
    )
    if existing_text:
        score += token_cosine(existing_text, f"{relation} {neighbor_text}")

    return round(score, 6)


def _target_modality_priority(
    record: EntityRecord,
    missing_mode: str,
    relation: str,
    neighbors: list[dict[str, Any]],
) -> dict[str, str]:
    relation_tokens = set(tokenize(relation))
    neighbor_tokens = set(
        token
        for neighbor in neighbors
        for value in (
            neighbor.get("label"),
            neighbor.get("description"),
            neighbor.get("entity_key"),
        )
        if isinstance(value, str)
        for token in tokenize(value)
    )
    text_score = 0
    image_score = 0
    if relation_tokens & TEXT_RELATION_TERMS:
        text_score += 2
    if relation_tokens & VISUAL_RELATION_TERMS:
        image_score += 2
    if neighbor_tokens & TEXT_RELATION_TERMS:
        text_score += 1
    if neighbor_tokens & VISUAL_RELATION_TERMS:
        image_score += 1
    existing_text = " ".join(
        value
        for value in (
            record.existing_modalities.get("text"),
            record.existing_modalities.get("image_summary"),
        )
        if isinstance(value, str) and value.strip()
    )
    relation_context = " ".join([relation, *[str(neighbor.get("label") or "") for neighbor in neighbors]])
    if existing_text and token_cosine(existing_text, relation_context) > 0:
        text_score += 2
    if missing_mode == "text" and text_score == 0:
        text_score = 1
    if missing_mode == "image" and image_score == 0 and text_score > 0:
        image_score = 1
    return {
        "text": _priority_label(text_score),
        "image": _priority_label(image_score),
    }


def _relation_semantic_summary(
    record: EntityRecord,
    relation: str,
    direction: str,
    neighbors: list[dict[str, Any]],
    priority: dict[str, str],
) -> str:
    labels = [
        str(neighbor.get("label") or neighbor.get("entity_key") or "unknown")
        for neighbor in neighbors[:3]
    ]
    direction_text = "incoming" if direction == "in" else "outgoing"
    modality_text = ", ".join(
        f"{modality}:{level}"
        for modality, level in priority.items()
        if level != "low"
    ) or "modality:low"
    return (
        f"{direction_text} relation '{relation}' links {record.label} with "
        f"{', '.join(labels)}; semantic priority {modality_text}."
    )


def _neighbor_modality_priority(neighbor: dict[str, Any]) -> dict[str, str]:
    text = " ".join(
        str(value or "")
        for value in (
            neighbor.get("relation"),
            neighbor.get("label"),
            neighbor.get("description"),
            neighbor.get("entity_key"),
        )
    )
    tokens = set(tokenize(text))
    return {
        "text": _priority_label(1 if tokens & TEXT_RELATION_TERMS else 0),
        "image": _priority_label(1 if tokens & VISUAL_RELATION_TERMS else 0),
    }


def _neighbor_label(neighbor: dict[str, Any]) -> str:
    return str(neighbor.get("label") or neighbor.get("entity_key") or "unknown")


def _two_hop_pattern(first_direction: str, second_direction: str) -> str:
    if first_direction == "in" and second_direction == "in":
        return "neighbor_to_target_from_neighbor"
    if first_direction == "in":
        return "neighbor_to_target_to_neighbor"
    if second_direction == "in":
        return "target_to_neighbor_from_neighbor"
    return "target_to_neighbor_to_neighbor"


def _second_hop_neighbors(neighbor: dict[str, Any]) -> list[dict[str, Any]]:
    raw = neighbor.get("neighbors")
    if not isinstance(raw, list):
        raw = neighbor.get("second_hop_neighbors")
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, dict)
        and not _is_low_value_relation(str(item.get("relation") or "related to"))
    ]


def _two_hop_path_context(
    item: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paths = []
    seen = set()
    for neighbor in neighbors:
        via_entity = _neighbor_label(neighbor)
        for second_hop in _second_hop_neighbors(neighbor):
            relation = str(second_hop.get("relation") or "related to")
            direction = str(second_hop.get("direction") or "out")
            terminal = _neighbor_label(second_hop)
            key = (via_entity, relation, direction, terminal)
            if key in seen:
                continue
            seen.add(key)
            paths.append(
                {
                    "path_depth": 2,
                    "path_pattern": _two_hop_pattern(item["direction"], direction),
                    "relations": [item["relation"], relation],
                    "directions": [item["direction"], direction],
                    "via_entity": via_entity,
                    "terminal_entities": [terminal],
                    "evidence_triples": [
                        f"({via_entity}, {relation}, {terminal})"
                        if direction != "in"
                        else f"({terminal}, {relation}, {via_entity})"
                    ],
                }
            )
            if len(paths) >= 6:
                return paths
    return paths


def _semantic_candidate(
    record: EntityRecord,
    item: dict[str, Any],
    neighbors: list[dict[str, Any]],
    missing_mode: str,
) -> dict[str, Any]:
    priority = _target_modality_priority(record, missing_mode, item["relation"], neighbors)
    candidate = dict(item)
    two_hop_paths = _two_hop_path_context(item, neighbors)
    candidate["target_modality_priority"] = priority
    semantic_summary = _relation_semantic_summary(
        record,
        item["relation"],
        item["direction"],
        neighbors,
        priority,
    )
    if two_hop_paths:
        terminal_labels = [
            terminal
            for path in two_hop_paths[:3]
            for terminal in path.get("terminal_entities", [])
        ]
        semantic_summary = (
            f"{semantic_summary} Second-hop context reaches "
            f"{', '.join(terminal_labels)}."
        )
    candidate["relation_semantic_summary"] = semantic_summary
    candidate["path_context"] = [
        {
            "path_depth": 1,
            "path_pattern": "neighbor_to_target" if item["direction"] == "in" else "target_to_neighbor",
            "relation": item["relation"],
            "direction": item["direction"],
            "neighbor_entities": list(candidate.get("neighbor_entities", []))[:5],
        }
    ] + two_hop_paths
    candidate["neighbor_modality_priority"] = [
        {
            "neighbor": str(neighbor.get("label") or neighbor.get("entity_key") or "unknown"),
            "priority": _neighbor_modality_priority(neighbor),
        }
        for neighbor in neighbors[:5]
    ]
    return candidate


def _v3_relation_id(item: dict[str, Any]) -> str:
    relation = "_".join(tokenize(item.get("relation"))) or "related_to"
    direction = str(item.get("direction") or (item.get("directions") or ["out"])[0] or "out")
    return f"rel:{relation}:{direction}"


def _v3_relation_ids(items: list[dict[str, Any]]) -> list[str]:
    return [_v3_relation_id(item) for item in items]


def _v3_fact(
    fact_id: str,
    record: EntityRecord,
    relation_id: str,
    item: dict[str, Any],
    neighbor_label: str,
) -> dict[str, Any]:
    relation = str(item.get("relation") or "related to")
    direction = str(item.get("direction") or (item.get("directions") or ["out"])[0] or "out")
    if direction == "in":
        subject = neighbor_label
        obj = record.label
    else:
        subject = record.label
        obj = neighbor_label
    return {
        "fact_id": fact_id,
        "subject": subject,
        "predicate": relation,
        "object": obj,
        "relation_id": relation_id,
        "hop": 1,
        "triple": f"({subject}, {relation}, {obj})",
    }


def _build_stage0_v3(
    record: EntityRecord,
    cleaned: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    discarded: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_ids = set(_v3_relation_ids(kept))
    relation_items = []
    seen_relations = set()
    for item in [*cleaned, *discarded]:
        relation_id = _v3_relation_id(item)
        if relation_id in seen_relations:
            continue
        seen_relations.add(relation_id)
        relation_items.append(item)

    facts: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    path_ids_by_relation: dict[str, list[str]] = {}
    relations: list[dict[str, Any]] = []
    fact_index = 1
    seen_paths = set()
    for item in relation_items:
        relation_id = _v3_relation_id(item)
        fact_ids = []
        neighbors = list(item.get("neighbor_entities", []))
        evidence = list(item.get("evidence_triples", []))
        for index, _triple_text in enumerate(evidence):
            neighbor_label = str(neighbors[index]) if index < len(neighbors) else "unknown"
            fact_id = f"fact:{fact_index:04d}"
            fact_index += 1
            facts.append(_v3_fact(fact_id, record, relation_id, item, neighbor_label))
            fact_ids.append(fact_id)
        for path in item.get("path_context", []):
            if not isinstance(path, dict) or path.get("path_depth") != 2:
                continue
            key = (
                path.get("via_entity"),
                tuple(path.get("terminal_entities", [])),
                tuple(path.get("relations", [])),
                tuple(path.get("directions", [])),
            )
            if key in seen_paths:
                continue
            seen_paths.add(key)
            path_id = f"path:{len(paths) + 1:04d}"
            path_ids_by_relation.setdefault(relation_id, []).append(path_id)
            paths.append(
                {
                    "path_id": path_id,
                    "path_depth": 2,
                    "via_entity": path.get("via_entity"),
                    "terminal_entities": list(path.get("terminal_entities", [])),
                    "relations": list(path.get("relations", [])),
                    "directions": list(path.get("directions", [])),
                    "fact_ids": [],
                }
            )

        decision = "selected" if relation_id in selected_ids else "discarded"
        relations.append(
            {
                "relation_id": relation_id,
                "relation": item.get("relation"),
                "direction": str(item.get("direction") or (item.get("directions") or ["out"])[0] or "out"),
                "score": item.get("coarse_score"),
                "frequency": item.get("frequency"),
                "neighbor_entities": neighbors,
                "fact_ids": fact_ids,
                "path_ids": path_ids_by_relation.get(relation_id, []),
                "decision": decision,
                "decision_reason": (
                    "selected_by_stage0"
                    if decision == "selected"
                    else str(item.get("discard_reason") or "not_selected_by_stage0")
                ),
            }
        )

    return {
        "relations": relations,
        "facts": facts,
        "paths": paths,
        "decisions": {
            "selected_relation_ids": _v3_relation_ids(kept),
            "discarded_relation_ids": _v3_relation_ids(discarded),
            "high_recall_relation_ids": _v3_relation_ids(cleaned[:24]),
        },
    }


def coarse_relation_filtering(record: EntityRecord, missing_mode: str) -> dict[str, Any]:
    """Clean and compress graph neighbors into relation-level evidence.

    The rule stage keeps a deterministic ``selected_relations`` baseline for
    offline/debug runs, while ``cleaned_relation_groups`` is the candidate set
    intended for the LMM semantic coarse-filtering mainline.
    """
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for neighbor in record.neighbors:
        relation = str(neighbor.get("relation") or "related to")
        direction = str(neighbor.get("direction") or "out")
        grouped.setdefault((relation, direction), []).append(neighbor)

    cleaned = []
    discarded = []
    for (relation, direction), neighbors in grouped.items():
        evidence = [_triple(record, neighbor) for neighbor in neighbors[:3]]
        item = {
            "relation": relation,
            "direction": direction,
            "coarse_score": _score_group(record, missing_mode, relation, neighbors),
            "frequency": len(neighbors),
            "directions": [direction],
            "neighbor_entities": [
                neighbor.get("label") or neighbor.get("entity_key") or "unknown" for neighbor in neighbors
            ],
            "evidence_triples": evidence,
        }
        if _is_low_value_relation(relation):
            item["discard_reason"] = "low_value_metadata"
            discarded.append(item)
        else:
            cleaned.append(_semantic_candidate(record, item, neighbors, missing_mode))

    cleaned.sort(key=lambda item: (-item["coarse_score"], item["relation"].lower(), item["direction"]))
    kept = cleaned[:8]
    kept = [item for item in kept if item["coarse_score"] >= 3]
    kept_groups = {(item["relation"], item["direction"]) for item in kept}
    for item in cleaned:
        if (item["relation"], item["direction"]) not in kept_groups:
            item = dict(item)
            item["discard_reason"] = "below_stage_0_threshold"
            discarded.append(item)

    compressed_evidence = [
        triple for item in cleaned for triple in item["evidence_triples"]
    ]
    return {
        "cleaned_relation_groups": cleaned,
        "high_recall_relation_candidates": cleaned[:24],
        "compressed_evidence": compressed_evidence,
        "selected_relations": kept,
        "discarded_relations": discarded,
        "filtered_triples": [
            triple for item in kept for triple in item["evidence_triples"]
        ],
        "stage0_v3": _build_stage0_v3(record, cleaned, kept, discarded),
    }
