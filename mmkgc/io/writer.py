"""Run output writer for the MMKGC MVP pipeline."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mmkgc.utils import tokenize
from mmkgc.utils import write_json


STAGE_FILES = [
    ("coarse_relation_filtering", "0_coarse_relation_filtering.json"),
    ("knowledge_modeling", "1_knowledge_modeling.json"),
    ("fine_grained_relation_selection", "2_fine_grained_relation_selection.json"),
    ("relation_alternation_generation", "3_relation_alternation_generation.json"),
    (
        "kg_aware_candidate_consistency_ranking",
        "4_kg_aware_candidate_consistency_ranking.json",
    ),
]
_STAGE2_SCORE_COMPONENT_KEYS = (
    "modality_fit",
    "constraint_alignment",
    "evidence_strength",
    "generation_utility",
    "risk_penalty",
)
_SENSITIVE_KEY_PATTERN = (
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|token|password|secret)"
)

def _json_ready(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"redacted_bytes": _bytes_length(value)}
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _safe_output(value: Any) -> Any:
    ready = _json_ready(value)
    if isinstance(ready, dict):
        return {
            key: _safe_output(item)
            for key, item in ready.items()
            if not _is_secret_key(key)
        }
    if isinstance(ready, list):
        return [_safe_output(item) for item in ready]
    if isinstance(ready, str):
        return _redact_output_string(ready)
    return ready


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe_output(row), ensure_ascii=False, separators=(",", ":")) + "\n")


class RunWriter:
    """Write per-run, per-entity, and summary artifacts."""

    def __init__(self, output_root: str | Path, run_name: str, config: dict[str, Any]):
        self.output_root = Path(output_root)
        self.run_name = run_name
        self.config = _safe_output(dict(config))
        self.run_dir = self.output_root / run_name
        self.entities_dir = self.run_dir / "entities"
        self.results_path = self.run_dir / "results.jsonl"
        self._processed: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._fallback_counts: Counter = Counter()

        if self.run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists for run_name={run_name!r}: {self.run_dir}. "
                "Choose a new run name; existing runs are not overwritten."
            )
        self.run_dir.mkdir(parents=True)
        self.entities_dir.mkdir()
        write_json(self.run_dir / "config.json", self.config)
        self.results_path.write_text("", encoding="utf-8")

    def write_entity(self, result: dict[str, Any]) -> None:
        entity = result.get("entity", {})
        entity_key = str(entity.get("entity_key") or entity.get("label") or entity.get("entity_id") or "entity")
        entity_dir = self.entities_dir / _entity_dir_name(entity_key)
        entity_dir.mkdir(parents=True, exist_ok=True)

        image_artifacts = _transport_ready(result.get("generated_image_artifacts", []))
        for artifact in image_artifacts:
            _write_image_artifact(entity_dir, artifact)

        for key, filename in STAGE_FILES:
            write_json(entity_dir / filename, _stage_artifact(key, result.get(key, {})))
        write_json(entity_dir / "final.json", _safe_output(result.get("final", {})))
        raw_model_calls = _json_ready(result.get("model_calls", []))
        if not isinstance(raw_model_calls, list):
            raw_model_calls = []
        raw_model_calls = [row for row in raw_model_calls if isinstance(row, dict)]
        model_calls = [row for row in raw_model_calls if row.get("modality") != "image"]
        embedded_image_model_calls = [
            row for row in raw_model_calls if row.get("modality") == "image"
        ]
        if model_calls:
            _write_jsonl(entity_dir / "model_calls.jsonl", model_calls)
        image_model_calls = _json_ready(result.get("image_model_calls", []))
        if not isinstance(image_model_calls, list):
            image_model_calls = []
        image_model_calls = [row for row in image_model_calls if isinstance(row, dict)]
        if not image_model_calls:
            image_model_calls = embedded_image_model_calls
        if image_model_calls:
            _write_jsonl(entity_dir / "image_model_calls.jsonl", image_model_calls)
        ranking_model_calls = _json_ready(result.get("ranking_model_calls", []))
        if ranking_model_calls:
            _write_jsonl(
                entity_dir / "ranking_model_calls.jsonl",
                _normalize_stage4_ranking_model_calls(ranking_model_calls),
            )

        final = _safe_output(result.get("final", {}))
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._processed.append(final)
        self._record_fallback_counts(result)

    def record_failure(
        self,
        entity: Any | None = None,
        error: BaseException | str | None = None,
        stage: str | None = None,
    ) -> None:
        failure = {
            "entity": _safe_output(entity),
            "stage": stage or "pipeline",
            "error": str(error) if error is not None else None,
            "error_type": type(error).__name__ if isinstance(error, BaseException) else None,
        }
        model_call = getattr(error, "model_call", None) if error is not None else None
        if isinstance(model_call, dict):
            failure["model_call"] = _safe_output(model_call)
            if stage is None and isinstance(model_call.get("stage"), str):
                failure["stage"] = model_call["stage"]
        self._failures.append(failure)

    def write_summary(self) -> dict[str, Any]:
        text_counts = [int(item.get("counts", {}).get("text", 0) or 0) for item in self._processed]
        image_counts = [int(item.get("counts", {}).get("image", 0) or 0) for item in self._processed]
        pair_counts = [int(item.get("counts", {}).get("pair", 0) or 0) for item in self._processed]

        summary = {
            "processed_entities": len(self._processed),
            "failed_entities": len(self._failures),
            "avg_text_candidates": _average(text_counts),
            "avg_image_candidates": _average(image_counts),
            "avg_pair_candidates": _average(pair_counts),
            "fallback_counts": dict(self._fallback_counts),
            "stage_failure_counts": dict(Counter(item["stage"] for item in self._failures)),
        }
        write_json(self.run_dir / "metrics_summary.json", _safe_output(summary))
        if self._failures:
            write_json(self.run_dir / "failures.json", _safe_output(self._failures))
        return summary

    def _record_fallback_counts(self, result: dict[str, Any]) -> None:
        ranking = result.get("kg_aware_candidate_consistency_ranking", {})
        for modality, ranking_key in (
            ("text", "text_ranking"),
            ("image", "image_ranking"),
        ):
            candidates = ranking.get(ranking_key, {}).get("ranked_candidates", [])
            count = _unique_fallback_candidate_count(candidates)
            if count:
                self._fallback_counts[modality] += count


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _stage_artifact(stage: str, payload: Any) -> dict[str, Any]:
    data = _compact_stage_data(stage, _safe_output(payload))
    return {
        "stage": stage,
        "summary": _stage_summary(stage, data),
        "data": data,
    }


def _compact_stage_data(stage: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if stage == "coarse_relation_filtering":
        return _compact_stage0_coarse(payload)
    if stage == "knowledge_modeling":
        return _compact_stage1_modeling(payload)
    if stage == "fine_grained_relation_selection":
        return _compact_stage2_selection(
            dict(payload.get("fine_grained_relation_selection", payload))
        )
    if stage == "relation_alternation_generation":
        return _compact_stage3_generation(
            dict(payload.get("relation_alternation_generation", payload))
        )
    if stage == "kg_aware_candidate_consistency_ranking":
        return _compact_stage4_ranking(payload)
    return dict(payload)


def _compact_stage0_coarse(payload: dict[str, Any]) -> dict[str, Any]:
    explicit_v3 = payload.get("stage0_v3")
    if isinstance(explicit_v3, dict):
        return _normalize_stage0_v3(explicit_v3)
    if {"relations", "facts", "paths", "decisions"}.issubset(payload):
        return _normalize_stage0_v3(payload)

    relation_groups = list(payload.get("cleaned_relation_groups", []))
    selected_ids = set(_relation_ids(payload.get("selected_relations", [])))
    high_recall_ids = _relation_ids(payload.get("high_recall_relation_candidates", []))
    discarded_items = list(payload.get("discarded_relations", []))
    discarded_ids = set(_relation_ids(discarded_items))
    relation_items = _unique_relations([*relation_groups, *discarded_items])
    paths, path_ids_by_relation = _paths_from_legacy_relations(relation_items)

    facts: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    fact_index = 1
    for item in relation_items:
        if not isinstance(item, dict):
            continue
        relation_id = _relation_id(item)
        fact_ids = []
        for triple in item.get("evidence_triples", []) if isinstance(item.get("evidence_triples"), list) else []:
            fact_id = f"fact:{fact_index:04d}"
            fact_index += 1
            facts.append(_fact_from_triple(fact_id, relation_id, triple))
            fact_ids.append(fact_id)
        decision = "selected" if relation_id in selected_ids else "discarded"
        reason = (
            "selected_by_stage0"
            if decision == "selected"
            else str(item.get("discard_reason") or "not_selected_by_stage0")
        )
        relations.append(
            {
                "relation_id": relation_id,
                "relation": item.get("relation"),
                "direction": _relation_direction(item),
                "score": item.get("coarse_score"),
                "frequency": item.get("frequency"),
                "neighbor_entities": list(item.get("neighbor_entities", []))
                if isinstance(item.get("neighbor_entities"), list)
                else [],
                "fact_ids": fact_ids,
                "path_ids": path_ids_by_relation.get(relation_id, []),
                "decision": decision,
                "decision_reason": reason,
            }
        )

    return _normalize_stage0_v3(
        {
            "relations": relations,
            "facts": facts,
            "paths": paths,
            "decisions": {
                "selected_relation_ids": list(selected_ids),
                "discarded_relation_ids": list(discarded_ids),
                "high_recall_relation_ids": high_recall_ids,
            },
        }
    )


def _compact_stage1_modeling(payload: dict[str, Any]) -> dict[str, Any]:
    explicit_v3 = payload.get("stage1_v3")
    if isinstance(explicit_v3, dict):
        compact = dict(explicit_v3)
        if payload.get("schema_version") == "qwen_grounded_knowledge_v1":
            compact["relation_facts"] = payload.get("relation_facts", [])
            compact["evidence_catalog"] = payload.get("evidence_catalog", [])
        constraints = payload.get("constraint_compilation")
        if isinstance(constraints, dict):
            compact["constraint_compilation"] = constraints
        diagnostics = payload.get("semantic_fact_assessment_diagnostics")
        if isinstance(diagnostics, dict):
            compact["semantic_fact_assessment_diagnostics"] = diagnostics
        return compact
    return dict(payload)


def _normalize_stage0_v3(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "relations": list(payload.get("relations", []))
        if isinstance(payload.get("relations"), list)
        else [],
        "facts": list(payload.get("facts", [])) if isinstance(payload.get("facts"), list) else [],
        "paths": list(payload.get("paths", [])) if isinstance(payload.get("paths"), list) else [],
        "decisions": dict(payload.get("decisions", {}))
        if isinstance(payload.get("decisions"), dict)
        else {
            "selected_relation_ids": [],
            "discarded_relation_ids": [],
            "high_recall_relation_ids": [],
        },
    }


def _relation_id(item: dict[str, Any]) -> str:
    relation = "_".join(tokenize(item.get("relation"))) or "related_to"
    return f"rel:{relation}:{_relation_direction(item)}"


def _relation_direction(item: dict[str, Any]) -> str:
    directions = item.get("directions")
    first_direction = directions[0] if isinstance(directions, list) and directions else None
    return str(item.get("direction") or first_direction or "out")


def _relation_ids(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [_relation_id(item) for item in items if isinstance(item, dict)]


def _unique_relations(items: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        relation_id = _relation_id(item)
        if relation_id in seen:
            continue
        seen.add(relation_id)
        result.append(item)
    return result


def _fact_from_triple(fact_id: str, relation_id: str, triple: Any) -> dict[str, Any]:
    triple_text = str(triple)
    subject = predicate = obj = None
    if triple_text.startswith("(") and triple_text.endswith(")"):
        parts = triple_text[1:-1].split(", ", 2)
        if len(parts) == 3:
            subject, predicate, obj = parts
    return {
        "fact_id": fact_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "relation_id": relation_id,
        "hop": 1,
        "triple": triple_text,
    }


def _paths_from_legacy_relations(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    paths = []
    path_ids_by_relation: dict[str, list[str]] = {}
    seen = set()
    for item in items:
        relation_id = _relation_id(item)
        if not isinstance(item.get("path_context"), list):
            continue
        for path in item["path_context"]:
            if not isinstance(path, dict) or path.get("path_depth") != 2:
                continue
            key = json.dumps(path, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            path_id = f"path:{len(paths) + 1:04d}"
            path_ids_by_relation.setdefault(relation_id, []).append(path_id)
            paths.append(
                {
                    "path_id": path_id,
                    "path_depth": 2,
                    "via_entity": path.get("via_entity"),
                    "terminal_entities": list(path.get("terminal_entities", []))
                    if isinstance(path.get("terminal_entities"), list)
                    else [],
                    "relations": list(path.get("relations", []))
                    if isinstance(path.get("relations"), list)
                    else [],
                    "directions": list(path.get("directions", []))
                    if isinstance(path.get("directions"), list)
                    else [],
                    "fact_ids": [],
                }
            )
    return paths, path_ids_by_relation


def _relation_refs(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    refs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation") or "").strip()
        directions = item.get("directions")
        first_direction = directions[0] if isinstance(directions, list) and directions else None
        direction = str(item.get("direction") or first_direction or "out")
        if relation:
            refs.append(f"{relation}:{direction}")
    return refs


def _compact_stage2_selection(payload: dict[str, Any]) -> dict[str, Any]:
    relation_plans, global_constraint_refs = _compact_stage2_relation_plans(payload)
    compact: dict[str, Any] = {
        "target_modalities": list(payload.get("target_modality") or payload.get("target_modalities") or []),
        "global_constraint_refs_by_bucket": global_constraint_refs,
        "relation_plans": relation_plans,
        "quota_plan": _compact_stage2_quota(payload.get("quota_summary", {})),
    }
    if payload.get("schema_version") == "model_relation_plan_v1":
        compact["schema_version"] = payload["schema_version"]
        compact["status"] = payload["status"]
        if "candidate_anchors" in payload:
            compact["candidate_anchors"] = list(payload["candidate_anchors"])
            compact["allocation_method"] = payload["allocation_method"]
    if payload.get("semantic_refinement") is not None:
        compact["semantic_refinement"] = payload.get("semantic_refinement")
    return compact


def _compact_stage2_relation_plans(payload: dict[str, Any]) -> tuple[list[Any], dict[str, list[str]]]:
    plans: list[Any] = []
    global_constraint_refs = _empty_stage2_constraint_refs()
    index = 1
    for key in (
        "primary_relations",
        "secondary_relations",
        "context_relations",
        "excluded_relations",
    ):
        for item in payload.get(key, []):
            compact, item_global_refs = _compact_stage2_relation(item, index)
            plans.append(compact)
            _merge_stage2_constraint_refs(global_constraint_refs, item_global_refs)
            index += 1
    return plans, global_constraint_refs


def _compact_stage2_relation(item: Any, index: int) -> tuple[Any, dict[str, list[str]]]:
    if not isinstance(item, dict):
        return item, _empty_stage2_constraint_refs()
    if item.get("fact_id") and item.get("score_method") == "gpt4omini_global_relation_plan":
        return {
            "plan_id": item["plan_id"], "fact_id": item["fact_id"],
            "relation": item["relation"], "direction": item["direction"],
            "relation_group_id": item["relation_group_id"], "role": item["relation_role"],
            "target_modalities": item["target_modality"],
            "evidence_match_score": item["evidence_match_score"],
            "source_fact_refs": item["source_fact_refs"],
            "evidence_refs": item["selected_evidence_refs"],
            "supporting_fact_ids": item["supporting_fact_ids"],
            "candidate_quota": item["candidate_quota"],
            "candidate_quota_by_modality": item["candidate_quota_by_modality"],
            "selection_reason": item["role_decision"],
            "generation_context": item.get("generation_context", {}),
        }, _empty_stage2_constraint_refs()
    role = str(item.get("relation_role") or "unassigned")
    global_refs = _normalize_stage2_constraint_refs(
        item.get("global_constraint_refs_by_bucket")
    )
    relation_refs = _normalize_stage2_constraint_refs(
        item.get("relation_constraint_refs_by_bucket")
    )
    score_breakdown = item.get("score_breakdown", {})
    components = (
        {
            key: score_breakdown[key]
            for key in _STAGE2_SCORE_COMPONENT_KEYS
            if key in score_breakdown
        }
        if isinstance(score_breakdown, dict)
        else {}
    )
    compact: dict[str, Any] = {
        "plan_id": item.get("plan_id") or f"rp:{index:04d}",
        "relation": item.get("relation"),
        "role": role,
        "status": _stage2_relation_status(item, role),
        "target_modalities": list(item.get("target_modality", [])),
        "source_fact_refs": list(item.get("source_fact_refs", [])),
        "evidence_refs": _stage2_evidence_refs(item),
        "score": {
            "fine_score": item.get("fine_score"),
            "method": item.get("score_method"),
            "components": components,
        },
        "selection_reasons": _stage2_selection_reasons(item, role),
        "relation_constraint_refs_by_bucket": relation_refs,
        "candidate_quota": _stage2_candidate_quota(item),
        "candidate_quota_by_modality": _stage2_candidate_quota_by_modality(item),
    }
    source_fact = item.get("source_fact")
    if isinstance(source_fact, dict) and not compact["source_fact_refs"]:
        compact["source_fact_summary"] = {
            key: source_fact.get(key)
            for key in (
                "relation",
                "fact",
                "support_level",
                "credibility_score",
                "credibility_label",
                "allowed_use",
            )
            if source_fact.get(key) is not None
        }
    llm_assessment = _compact_stage2_llm_assessment(item.get("llm_assessment"))
    if llm_assessment is not None:
        compact["llm_assessment"] = llm_assessment
    if item.get("model_call_ref"):
        compact["model_call_ref"] = str(item["model_call_ref"])
    return compact, global_refs


def _compact_stage2_llm_assessment(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    assessment = {
        "status": value.get("status"),
        "component_adjustments": value.get("component_adjustments", {}),
        "risk_penalty_adjustment": value.get("risk_penalty_adjustment"),
        "confidence": value.get("confidence"),
        "reason": _compact_stage2_llm_reason(value.get("reason")),
    }
    recommended_role = str(value.get("recommended_role") or "").strip().lower()
    if recommended_role in {"primary", "secondary", "context", "excluded"}:
        assessment["recommended_role"] = recommended_role
    return assessment


def _compact_stage2_llm_reason(value: Any) -> str:
    text = _redact_output_string(str(value or ""))
    text = re.sub(r"(?i)\b(prompt|response|raw_response)\b", "[redacted]", text)
    return text[:240]


def _redact_stage2_url_userinfo(value: str) -> str:
    return re.sub(
        r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@",
        r"\1[REDACTED]@",
        value,
    )


def _redact_output_string(value: str) -> str:
    text = _redact_stage2_url_userinfo(value)
    quoted_pattern = re.compile(
        rf"(?i)(?P<prefix>(?:\\?[\"']){_SENSITIVE_KEY_PATTERN}"
        r"(?:\\?[\"'])\s*:\s*(?P<quote>\\?[\"']))(?P<secret>.*?)(?P=quote)"
    )

    def replace_quoted(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{match.group('prefix')}[REDACTED]{quote}"

    text = quoted_pattern.sub(replace_quoted, text)
    return re.sub(
        r"(?i)\b(?P<key>" + _SENSITIVE_KEY_PATTERN + r")\s*[:=]\s*(?!\\?[\"'])[^,;}&\r\n]+",
        r"\g<key>=[REDACTED]",
        text,
    )


def _stage2_relation_status(item: dict[str, Any], role: str) -> str:
    if item.get("stage2_hard_excluded") or role == "excluded":
        return "excluded"
    if item.get("stage2_context_only") or role == "context":
        return "context_only"
    return "selected"


def _stage2_selection_reasons(item: dict[str, Any], role: str) -> list[str]:
    reasons = list(item.get("stage2_hard_exclusion_reasons", []))
    reasons.extend(item.get("stage2_context_only_reasons", []))
    if reasons:
        return reasons
    score_breakdown = item.get("score_breakdown", {})
    if score_breakdown.get("evidence_strength", 0) >= 0.85:
        reasons.append("strong_evidence")
    if score_breakdown.get("modality_fit", 0) >= 0.75:
        reasons.append("high_modality_fit")
    if score_breakdown.get("constraint_alignment", 0) >= 0.75:
        reasons.append("constraint_aligned")
    if role in {"primary", "secondary"}:
        reasons.append(f"selected_as_{role}")
    return reasons


def _stage2_evidence_refs(item: dict[str, Any]) -> list[str]:
    source_fact = item.get("source_fact")
    if isinstance(source_fact, dict):
        refs = source_fact.get("evidence_refs") or source_fact.get("stage0_fact_ids")
        if isinstance(refs, list):
            return [str(ref) for ref in refs]
    refs = item.get("evidence_refs")
    if isinstance(refs, list):
        return [str(ref) for ref in refs]
    return []


def _empty_stage2_constraint_refs() -> dict[str, list[str]]:
    return {key: [] for key in ("must_include", "should_include", "context_only")}


def _merge_stage2_constraint_refs(
    target: dict[str, list[str]],
    source: dict[str, list[str]],
) -> None:
    for bucket, refs in source.items():
        target.setdefault(bucket, [])
        for ref in refs:
            if ref not in target[bucket]:
                target[bucket].append(ref)


def _normalize_stage2_constraint_refs(value: Any) -> dict[str, list[str]]:
    buckets = ("must_include", "should_include", "context_only")
    if not isinstance(value, dict):
        return {bucket: [] for bucket in buckets}
    return {
        bucket: [str(ref) for ref in value.get(bucket, [])]
        if isinstance(value.get(bucket, []), list)
        else []
        for bucket in buckets
    }


def _stage2_candidate_quota(item: dict[str, Any]) -> int:
    return int(item.get("candidate_quota", 0) or 0)


def _stage2_candidate_quota_by_modality(item: dict[str, Any]) -> dict[str, int]:
    quota = item.get("candidate_quota_by_modality")
    if not isinstance(quota, dict):
        return {}
    return {str(key): int(value) for key, value in quota.items()}


def _compact_stage2_quota(quota: Any) -> dict[str, Any]:
    if not isinstance(quota, dict):
        return {}
    compact = {
        key: quota.get(key)
        for key in (
            "candidate_budget",
            "quota_key",
            "quota_mode",
            "target_modalities",
            "primary_quota",
            "secondary_quota",
            "relation_quotas",
            "assigned_total",
            "fallback_quota",
            "global_profile_fallback",
        )
        if key in quota
    }
    target_modalities = quota.get("target_modalities", [])
    keep_modality_quota = (
        quota.get("quota_mode") == "per_modality"
        or (isinstance(target_modalities, list) and len(target_modalities) > 1)
    )
    if keep_modality_quota and "modality_quota" in quota:
        compact["modality_quota"] = quota.get("modality_quota")
    return compact


def _compact_stage3_generation(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    for key in ("text_candidates", "image_candidates"):
        candidates = []
        for item in compact.get(key, []):
            candidate, _unused_refs = _compact_stage3_candidate(item)
            candidates.append(candidate)
        compact[key] = candidates
    return compact


def _compact_stage3_candidate(item: Any) -> tuple[Any, list[str]]:
    if not isinstance(item, dict):
        return item, []
    coverage = _stage3_candidate_coverage(item)
    compact: dict[str, Any] = {
        "candidate_id": item.get("candidate_id"),
        "target_modality": item.get("target_modality"),
        "status": _stage3_candidate_status(item),
        "source_plan_id": item.get("source_plan_id"),
        "primary_relation": item.get("primary_relation"),
        "generation_angle": item.get("generation_angle"),
        "fallback": bool(item.get("metadata", {}).get("fallback", item.get("fallback", False)))
        if isinstance(item.get("metadata"), dict)
        else bool(item.get("fallback", False)),
        "coverage": coverage,
    }
    if item.get("source_candidate_id"):
        compact["source_candidate_id"] = item.get("source_candidate_id")
    if item.get("verification_targets") is not None:
        compact["verification_targets"] = item["verification_targets"]
    content = _stage3_compact_generated_content(item.get("generated_content"))
    if content:
        if content.get("type") == "text":
            compact["generated_content"] = content
        else:
            compact["artifact"] = content
    model_call_ref = _stage3_model_call_ref(item)
    if model_call_ref:
        compact["model_call_ref"] = model_call_ref
    return compact, []


def _constraint_refs(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    refs = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            ref = item.get("constraint_id") or item.get("constraint_ref") or item.get("id")
            if ref:
                refs.append(str(ref))
            elif item.get("constraint"):
                refs.append(str(item["constraint"]))
        elif isinstance(item, str) and item.strip():
            refs.append(item)
    return refs


def _stage3_candidate_status(item: dict[str, Any]) -> Any:
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata.get("status") is not None:
        return metadata.get("status")
    return item.get("status")


def _stage3_candidate_coverage(item: dict[str, Any]) -> dict[str, Any]:
    relation_refs = list(item.get("expected_relation_coverage", []))
    if not relation_refs:
        relation_refs = list(item.get("supporting_relations", []))
    return {
        "relation_refs": relation_refs,
        "source_fact_refs": list(item.get("source_fact_refs", [])),
        "evidence_refs": list(item.get("evidence_refs", [])),
        "constraint_refs": _stage3_existing_or_compacted_constraint_refs(item, "constraint_refs", "used_constraints"),
    }


def _stage3_existing_or_compacted_constraint_refs(
    item: dict[str, Any],
    ref_key: str,
    legacy_key: str,
) -> list[str]:
    refs = item.get(ref_key)
    if isinstance(refs, list) and refs:
        return [str(ref) for ref in refs]
    return _constraint_refs(item.get(legacy_key, []))


def _stage3_compact_generated_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, dict):
        return {}
    content_type = content.get("type")
    if content_type == "text":
        return {"type": "text", "value": content.get("value")}
    if content_type == "image":
        artifact = {"type": "image"}
        for key in ("image_path", "metadata_path"):
            if content.get(key) is not None:
                artifact[key] = content.get(key)
        return artifact
    if content_type == "image_prompt":
        return {"type": "image_prompt", "status": "planned"}
    return {"type": content_type} if content_type else {}


def _stage3_model_call_ref(item: dict[str, Any]) -> str | None:
    candidate_id = item.get("candidate_id")
    modality = item.get("target_modality")
    status = _stage3_candidate_status(item)
    if candidate_id and modality and status in {"generated", "failed"}:
        return f"{modality}_call:{candidate_id}"
    return None


def _compact_stage4_ranking(payload: dict[str, Any]) -> dict[str, Any]:
    reference = payload.get("kg_aware_candidate_consistency_ranking")
    reference = reference if isinstance(reference, dict) else {}
    text_ranking = _compact_stage4_candidate_ranking(payload.get("text_ranking"), "text")
    image_ranking = _compact_stage4_candidate_ranking(payload.get("image_ranking"), "image")
    pair_ranking = _compact_stage4_pair_ranking(payload.get("pair_ranking"))
    return {
        "ranking_status": payload.get("ranking_status"),
        "observation_version": payload.get("observation_version"),
        "stage3_regeneration": payload.get("stage3_regeneration"),
        "target_modalities": _stage4_target_modalities(reference, text_ranking, image_ranking),
        "candidate_budget": reference.get("candidate_num") or payload.get("candidate_budget"),
        "evaluated_text_candidate_count": len(text_ranking["ranked_candidates"]),
        "evaluated_image_candidate_count": len(image_ranking["ranked_candidates"]),
        "ranked_pair_count": len(pair_ranking["ranked_pairs"]),
        "ground_truth_used": reference.get("ground_truth_used", payload.get("ground_truth_used")),
        "ranking_profile": _compact_stage4_profile(payload),
        "text_ranking": text_ranking,
        "image_ranking": image_ranking,
        "pair_ranking": pair_ranking,
    }


def _compact_stage4_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profile = {
        "ranker": payload.get("ranker"),
        "score_version": payload.get("score_version"),
        "profile_name": payload.get("profile_name"),
    }
    if isinstance(payload.get("candidate_score_weights"), dict):
        profile["candidate_score_weights"] = payload["candidate_score_weights"]
    return profile


def _stage4_target_modalities(
    reference: dict[str, Any],
    text_ranking: dict[str, Any],
    image_ranking: dict[str, Any],
) -> list[str]:
    target = str(reference.get("target_modality") or "").strip().lower()
    if target in {"text", "image"}:
        return [target]
    modalities = []
    if text_ranking.get("ranked_candidates"):
        modalities.append("text")
    if image_ranking.get("ranked_candidates"):
        modalities.append("image")
    return modalities


def _compact_stage4_candidate_ranking(value: Any, modality: str) -> dict[str, Any]:
    ranking = value if isinstance(value, dict) else {}
    ranked = [
        _compact_stage4_candidate(candidate, rank=index, fallback_modality=modality)
        for index, candidate in enumerate(_dict_items(ranking.get("ranked_candidates")), start=1)
    ]
    rejected = [
        _compact_stage4_candidate(candidate, rank=None, fallback_modality=modality)
        for candidate in _dict_items(ranking.get("rejected_candidates"))
    ]
    top_ids = ranking.get("top_5_candidate_ids")
    if not isinstance(top_ids, list):
        top_ids = [
            str(candidate.get("candidate_id"))
            for candidate in _dict_items(ranking.get("top_5_candidates"))
            if candidate.get("candidate_id")
        ]
    if not top_ids:
        top_ids = [item["candidate_id"] for item in ranked[:5] if item.get("candidate_id")]
    return {
        "best_candidate_id": ranking.get("best_candidate_id"),
        "ranked_candidate_ids": [item["candidate_id"] for item in ranked if item.get("candidate_id")],
        "ranked_candidates": ranked,
        "rejected_candidate_ids": [
            item["candidate_id"] for item in rejected if item.get("candidate_id")
        ],
        "rejected_candidates": rejected,
        "top_5_candidate_ids": [str(item) for item in top_ids if item],
    }


def _compact_stage4_candidate(
    item: dict[str, Any],
    *,
    rank: int | None,
    fallback_modality: str,
) -> dict[str, Any]:
    candidate_id = str(item.get("candidate_id") or "")
    compact: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_candidate_id": item.get("source_candidate_id"),
        "verification_targets": item.get("verification_targets"),
        "observation": item.get("observation"),
        "candidate_ref": f"stage3:{candidate_id}" if candidate_id else None,
        "rank": rank,
        "target_modality": item.get("target_modality") or fallback_modality,
        "primary_relation": item.get("primary_relation"),
        "generation_angle": item.get("generation_angle"),
    }
    artifact_ref = _stage4_artifact_ref(item)
    if artifact_ref:
        compact["artifact_ref"] = artifact_ref
    metadata_ref = _stage4_metadata_ref(item)
    if metadata_ref:
        compact["metadata_ref"] = metadata_ref
    generated_content = _stage4_compact_generated_content(item.get("generated_content"))
    if generated_content:
        compact["generated_content"] = generated_content
    for key in (
        "expected_relation_coverage",
        "relation_coverage_score",
        "must_term_score",
        "pair_similarity_score",
        "kg_anchor_score",
    ):
        if item.get(key) is not None:
            compact[key] = item.get(key)
    scores = _stage4_scores(item)
    compact["scores"] = scores
    compact["final_score"] = scores["final"]
    gate = _stage4_gate(item.get("hard_gate"))
    if gate:
        compact["gate"] = gate
    refs = _stage4_ranking_call_refs(candidate_id, item.get("model_scores"))
    if refs:
        compact["ranking_call_refs"] = refs
    return {key: value for key, value in compact.items() if value is not None}


def _compact_stage4_pair_ranking(value: Any) -> dict[str, Any]:
    ranking = value if isinstance(value, dict) else {}
    ranked = [
        _compact_stage4_pair(pair, rank=index)
        for index, pair in enumerate(_dict_items(ranking.get("ranked_pairs")), start=1)
    ]
    top_ids = ranking.get("top_5_pair_ids")
    if not isinstance(top_ids, list):
        top_ids = [
            str(pair.get("pair_id"))
            for pair in _dict_items(ranking.get("top_5_pairs"))
            if pair.get("pair_id")
        ]
    if not top_ids:
        top_ids = [item["pair_id"] for item in ranked[:5] if item.get("pair_id")]
    return {
        "best_pair_id": ranking.get("best_pair_id"),
        "ranked_pair_ids": [item["pair_id"] for item in ranked if item.get("pair_id")],
        "ranked_pairs": ranked,
        "top_5_pair_ids": [str(item) for item in top_ids if item],
    }


def _compact_stage4_pair(item: dict[str, Any], *, rank: int) -> dict[str, Any]:
    pair_id = str(item.get("pair_id") or "")
    text_candidate = item.get("text_candidate") if isinstance(item.get("text_candidate"), dict) else {}
    image_candidate = item.get("image_candidate") if isinstance(item.get("image_candidate"), dict) else {}
    text_id = item.get("text_candidate_id") or text_candidate.get("candidate_id")
    image_id = item.get("image_candidate_id") or image_candidate.get("candidate_id")
    compact: dict[str, Any] = {
        "pair_id": pair_id,
        "pair_ref": f"stage4:{pair_id}" if pair_id else None,
        "rank": rank,
        "text_candidate_id": text_id,
        "image_candidate_id": image_id,
        "text_candidate_ref": f"stage3:{text_id}" if text_id else None,
        "image_candidate_ref": f"stage3:{image_id}" if image_id else None,
        "text_candidate": _stage4_score_stub(text_candidate),
        "image_candidate": _stage4_score_stub(image_candidate),
    }
    for key in ("pair_similarity_score", "kg_anchor_score"):
        if item.get(key) is not None:
            compact[key] = item.get(key)
    scores = _stage4_scores(item)
    compact["scores"] = scores
    compact["final_score"] = scores["final"]
    gate = _stage4_gate(item.get("hard_gate"))
    if gate:
        compact["gate"] = gate
    refs = _stage4_ranking_call_refs(pair_id, item.get("model_scores"))
    if refs:
        compact["ranking_call_refs"] = refs
    return {key: value for key, value in compact.items() if value not in (None, {}, [])}


def _stage4_score_stub(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    candidate_id = candidate.get("candidate_id")
    if not candidate_id:
        return {}
    return {
        "candidate_id": candidate_id,
        "final_score": _stage4_final_score(candidate),
    }


def _stage4_scores(item: dict[str, Any]) -> dict[str, Any]:
    model_scores = item.get("model_scores") if isinstance(item.get("model_scores"), dict) else {}
    return {
        "final": _stage4_final_score(item),
        "candidate_score": _stage4_number(
            model_scores.get("candidate_score", item.get("candidate_score"))
        ),
        "components": _stage4_score_components(model_scores),
    }


def _stage4_score_components(model_scores: dict[str, Any]) -> dict[str, Any]:
    components = {}
    for key, value in model_scores.items():
        key = str(key)
        if key in {"candidate_score", "quality", "kg_consistency"}:
            continue
        if key.endswith("_raw_score") or key.endswith("_normalized_score"):
            continue
        number = _stage4_number(value)
        if number is not None:
            components[_stage4_scorer_name(key)] = number
    return components


def _stage4_final_score(item: dict[str, Any]) -> float | None:
    for key in ("final_score", "score", "fused_score"):
        number = _stage4_number(item.get(key))
        if number is not None:
            return number
    return 0.0


def _stage4_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage4_gate(value: Any) -> dict[str, Any]:
    gate = value if isinstance(value, dict) else {}
    if not gate:
        return {}
    return {
        "decision": gate.get("decision"),
        "source": gate.get("source"),
        "hard_violation_codes": list(gate.get("hard_violations", []))
        if isinstance(gate.get("hard_violations"), list)
        else [],
        "soft_violation_codes": list(gate.get("soft_violations", []))
        if isinstance(gate.get("soft_violations"), list)
        else [],
        "confidence": gate.get("judge_confidence", gate.get("confidence")),
    }


def _stage4_ranking_call_refs(candidate_id: str, model_scores: Any) -> list[str]:
    if not candidate_id or not isinstance(model_scores, dict):
        return []
    refs = []
    for key in model_scores:
        key = str(key)
        if key in {"candidate_score", "quality", "kg_consistency"}:
            continue
        if key.endswith("_raw_score") or key.endswith("_normalized_score"):
            continue
        refs.append(f"ranking_call:{candidate_id}:{_stage4_scorer_name(key)}")
    return refs


def _stage4_scorer_name(name: str) -> str:
    return str(name)


def _normalize_stage4_ranking_model_calls(value: Any) -> list[dict[str, Any]]:
    rows = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        scorer = normalized.get("scorer")
        if isinstance(scorer, str):
            normalized["scorer"] = _stage4_scorer_name(scorer)
        rows.append(normalized)
    return rows


def _stage4_artifact_ref(item: dict[str, Any]) -> str | None:
    content = item.get("generated_content")
    if isinstance(content, dict) and content.get("image_path"):
        return str(content["image_path"])
    artifact = item.get("artifact")
    if isinstance(artifact, dict) and artifact.get("image_path"):
        return str(artifact["image_path"])
    return None


def _stage4_metadata_ref(item: dict[str, Any]) -> str | None:
    content = item.get("generated_content")
    if isinstance(content, dict) and content.get("metadata_path"):
        return str(content["metadata_path"])
    artifact = item.get("artifact")
    if isinstance(artifact, dict) and artifact.get("metadata_path"):
        return str(artifact["metadata_path"])
    return None


def _stage4_compact_generated_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, dict):
        return {}
    compact = {}
    for key in ("type", "value", "image_path", "metadata_path"):
        if content.get(key) is not None:
            compact[key] = content.get(key)
    return compact


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _replace_top_candidates_with_refs(ranking: dict[str, Any]) -> None:
    top_candidates = ranking.pop("top_5_candidates", None)
    if isinstance(top_candidates, list):
        ranking["top_5_candidate_ids"] = [
            str(candidate.get("candidate_id"))
            for candidate in top_candidates
            if isinstance(candidate, dict) and candidate.get("candidate_id")
        ]


def _replace_top_pairs_with_refs(ranking: dict[str, Any]) -> None:
    top_pairs = ranking.pop("top_5_pairs", None)
    if isinstance(top_pairs, list):
        ranking["top_5_pair_ids"] = [
            str(pair.get("pair_id"))
            for pair in top_pairs
            if isinstance(pair, dict) and pair.get("pair_id")
        ]


def _stage_summary(stage: str, data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"payload_type": type(data).__name__}
    if stage == "coarse_relation_filtering":
        decisions = data.get("decisions", {})
        decisions = decisions if isinstance(decisions, dict) else {}
        return {
            "relation_count": len(data.get("relations", [])),
            "fact_count": len(data.get("facts", [])),
            "path_count": len(data.get("paths", [])),
            "selected_relation_count": len(decisions.get("selected_relation_ids", [])),
            "discarded_relation_count": len(decisions.get("discarded_relation_ids", [])),
            "high_recall_relation_count": len(decisions.get("high_recall_relation_ids", [])),
        }
    if stage == "knowledge_modeling":
        if data.get("schema_version") == "qwen_grounded_knowledge_v1":
            facts = data.get("relation_facts", [])
            return {
                "primary_type": data.get("entity_profile", {}).get("primary_type"),
                "relation_fact_count": len(facts),
                "scored_fact_count": sum(fact.get("scoring_status") == "scored" for fact in facts),
                "unscored_fact_count": sum(fact.get("evidence_match_score") is None for fact in facts),
                "evidence_count": len(data.get("evidence_catalog", [])),
                "target_modalities": data.get("modality_state", {}).get("targets", []),
            }
        profile = data.get("entity_profile", {})
        constraints = data.get("constraint_compilation", {})
        constraints = constraints if isinstance(constraints, dict) else {}
        supported_facts = data.get("supported_facts", [])
        supported_facts = supported_facts if isinstance(supported_facts, list) else []
        graph_relation_count = sum(
            1
            for item in supported_facts
            if isinstance(item, dict) and item.get("fact_type") == "graph_relation"
        )
        entity_fact_count = sum(
            1
            for item in supported_facts
            if isinstance(item, dict) and item.get("fact_type") == "entity_type"
        )
        summary = {
            "supported_fact_count": len(supported_facts),
            "supported_graph_relation_count": graph_relation_count,
            "supported_entity_fact_count": entity_fact_count,
            "supported_other_fact_count": len(supported_facts)
            - graph_relation_count
            - entity_fact_count,
            "safe_inference_count": len(data.get("safe_inferences", [])),
            "text_must_include_count": len(
                constraints.get("text", {}).get("must_include", [])
                if isinstance(constraints.get("text"), dict)
                else []
            ),
            "image_must_include_count": len(
                constraints.get("image", {}).get("must_include", [])
                if isinstance(constraints.get("image"), dict)
                else []
            ),
        }
        profile = data.get("entity_profile", {})
        if isinstance(profile, dict) and "primary_type" in profile:
            summary.update(
                {
                    "primary_type": profile.get("primary_type"),
                    "secondary_type_count": len(profile.get("secondary_types", [])),
                    "attribute_count": len(data.get("key_attributes", [])),
                    "target_modalities": list(data.get("modality_state", {}).get("targets", []))
                    if isinstance(data.get("modality_state"), dict)
                    else [],
                    "uncertain_aspect_count": len(data.get("uncertain_aspects", [])),
                }
            )
        elif isinstance(profile, dict):
            summary.update(
                {
                    "entity_type": profile.get("entity_type") or profile.get("type"),
                    "relation_fact_count": len(data.get("relation_facts", [])),
                    "uncertainty_count": len(data.get("uncertainties", [])),
                }
            )
        return summary
    if stage == "fine_grained_relation_selection":
        relation_plans = data.get("relation_plans", [])
        if isinstance(relation_plans, list):
            roles = Counter(
                str(item.get("role"))
                for item in relation_plans
                if isinstance(item, dict)
            )
            quota_plan = data.get("quota_plan", {}) if isinstance(data.get("quota_plan"), dict) else {}
            return {
                "input_relation_fact_count": len(relation_plans),
                "planned_relation_count": len(relation_plans),
                "primary_relation_count": roles.get("primary", 0),
                "secondary_relation_count": roles.get("secondary", 0),
                "context_relation_count": roles.get("context", 0),
                "excluded_relation_count": roles.get("excluded", 0),
                "candidate_budget": quota_plan.get("candidate_budget"),
                "assigned_candidate_quota": quota_plan.get("assigned_total"),
                "fallback_quota": quota_plan.get("fallback_quota"),
                "target_modalities": list(data.get("target_modalities", [])),
            }
        return {
            "primary_relation_count": len(data.get("primary_relations", [])),
            "secondary_relation_count": len(data.get("secondary_relations", [])),
            "context_relation_count": len(data.get("context_relations", [])),
            "excluded_relation_count": len(data.get("excluded_relations", [])),
        }
    if stage == "relation_alternation_generation":
        candidates = [
            item
            for item in list(data.get("text_candidates", [])) + list(data.get("image_candidates", []))
            if isinstance(item, dict)
        ]
        return {
            "target_modalities": list(data.get("target_modalities", [])),
            "text_candidate_count": len(data.get("text_candidates", [])),
            "image_candidate_count": len(data.get("image_candidates", [])),
            "candidate_budget": data.get("candidate_budget"),
            "generated_candidate_count": sum(
                1 for item in candidates if item.get("status") == "generated"
            ),
            "failed_candidate_count": sum(
                1 for item in candidates if str(item.get("status", "")).startswith("failed")
            ),
            "fallback_candidate_count": sum(1 for item in candidates if item.get("fallback")),
        }
    if stage == "kg_aware_candidate_consistency_ranking":
        profile = data.get("ranking_profile", {}) if isinstance(data.get("ranking_profile"), dict) else {}
        return {
            "ranking_status": data.get("ranking_status"),
            "ranker": profile.get("ranker", data.get("ranker")),
            "candidate_budget": data.get("candidate_budget"),
            "text_candidate_count": len(
                data.get("text_ranking", {}).get("ranked_candidates", [])
                if isinstance(data.get("text_ranking"), dict)
                else []
            ),
            "image_candidate_count": len(
                data.get("image_ranking", {}).get("ranked_candidates", [])
                if isinstance(data.get("image_ranking"), dict)
                else []
            ),
            "pair_count": len(
                data.get("pair_ranking", {}).get("ranked_pairs", [])
                if isinstance(data.get("pair_ranking"), dict)
                else []
            ),
            "best_text_candidate_id": data.get("text_ranking", {}).get("best_candidate_id")
            if isinstance(data.get("text_ranking"), dict)
            else None,
            "best_image_candidate_id": data.get("image_ranking", {}).get("best_candidate_id")
            if isinstance(data.get("image_ranking"), dict)
            else None,
            "best_pair_id": data.get("pair_ranking", {}).get("best_pair_id")
            if isinstance(data.get("pair_ranking"), dict)
            else None,
        }
    return {}


def _entity_dir_name(entity_key: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in entity_key)
    return safe.strip("._") or "entity"


def _write_image_artifact(entity_dir: Path, artifact: dict[str, Any]) -> None:
    image_path = None
    if artifact.get("relative_image_path") is not None:
        image_path = _safe_entity_relative_path(entity_dir, artifact.get("relative_image_path"))

    metadata_path = None
    if "metadata" in artifact and artifact["metadata"] is not None:
        metadata_path = _safe_entity_relative_path(entity_dir, artifact.get("relative_metadata_path"))

    image_bytes = artifact.get("image_bytes", b"")
    if image_bytes:
        if image_path is None:
            raise ValueError("Generated image artifact path must stay inside the entity directory")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

    if metadata_path is not None:
        write_json(metadata_path, _safe_output(artifact["metadata"]))


def _safe_entity_relative_path(entity_dir: Path, relative_path: Any) -> Path:
    if relative_path is None:
        raise ValueError("Generated image artifact path must stay inside the entity directory")

    path = Path(str(relative_path))
    if not str(relative_path) or path.is_absolute() or ".." in path.parts:
        raise ValueError("Generated image artifact path must stay inside the entity directory")

    entity_root = entity_dir.resolve()
    resolved = (entity_dir / path).resolve(strict=False)
    try:
        resolved.relative_to(entity_root)
    except ValueError as exc:
        raise ValueError("Generated image artifact path must stay inside the entity directory") from exc
    return resolved


def _unique_fallback_candidate_count(candidates: list[dict[str, Any]]) -> int:
    seen_candidate_ids = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id or candidate_id in seen_candidate_ids:
            continue
        if _is_fallback_candidate(candidate):
            seen_candidate_ids.add(candidate_id)
    return len(seen_candidate_ids)


def _is_fallback_candidate(candidate: dict[str, Any]) -> bool:
    metadata = candidate.get("metadata", {})
    coverage_detail = candidate.get("expected_relation_coverage_detail", {})
    return (
        candidate.get("primary_relation") == "global_profile"
        or metadata.get("fallback") is True
        or candidate.get("relation_role") == "fallback"
        or coverage_detail.get("primary") == "global_profile"
    )


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return (
        normalized
        in {
            "api_key",
            "apikey",
            "authorization",
            "token",
            "access_token",
            "refresh_token",
            "client_secret",
            "password",
            "secret",
            "headers",
        }
        or normalized.endswith("_token")
        or normalized.endswith("_secret")
        or normalized in {"token", "secret", "refresh_token", "access_token"}
    )


def _bytes_length(value: bytes | bytearray | memoryview) -> int:
    if isinstance(value, memoryview):
        return value.nbytes
    return len(value)


def _transport_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _transport_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _transport_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_transport_ready(item) for item in value]
    return value
