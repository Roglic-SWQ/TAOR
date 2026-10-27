#!/usr/bin/env python
"""Compare deterministic Stage 0-2 output with semantic Stage 0-2 output."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--semantic-run-dir", required=True)
    parser.add_argument("--sample-manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_dir = Path(args.baseline_run_dir)
    semantic_dir = Path(args.semantic_run_dir)
    output_path = Path(args.output)
    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    _validate_output(output_path, args.overwrite)
    _validate_output(csv_path, args.overwrite)

    entity_keys = _entity_keys(baseline_dir, semantic_dir, args.sample_manifest)
    rows = [
        _compare_entity(baseline_dir, semantic_dir, entity_key)
        for entity_key in entity_keys
    ]
    report = {
        "baseline_run_dir": str(baseline_dir),
        "semantic_run_dir": str(semantic_dir),
        "sample_manifest": args.sample_manifest,
        "entity_count": len(rows),
        "summary": _summary(rows),
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "csv_output": str(csv_path),
                "entity_count": len(rows),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _validate_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _entity_keys(
    baseline_dir: Path,
    semantic_dir: Path,
    manifest_path: str | None,
) -> list[str]:
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        rows = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("sample manifest must contain samples list")
        keys = [str(row.get("entity_key") or "").strip() for row in rows]
    else:
        baseline_keys = {path.name for path in (baseline_dir / "entities").iterdir()}
        semantic_keys = {path.name for path in (semantic_dir / "entities").iterdir()}
        keys = sorted(baseline_keys & semantic_keys)
    keys = [key for key in keys if key]
    missing = [
        key
        for key in keys
        if not (baseline_dir / "entities" / key).is_dir()
        or not (semantic_dir / "entities" / key).is_dir()
    ]
    if missing:
        raise ValueError(f"entity keys missing from one run: {', '.join(missing[:10])}")
    return keys


def _compare_entity(baseline_dir: Path, semantic_dir: Path, entity_key: str) -> dict[str, Any]:
    baseline = _load_entity(baseline_dir, entity_key)
    semantic = _load_entity(semantic_dir, entity_key)
    calls = _semantic_calls(semantic_dir / "entities" / entity_key / "model_calls.jsonl")
    baseline_stage0 = _relations(baseline["stage0"].get("selected_relations"))
    semantic_stage0 = _relations(semantic["stage0"].get("selected_relations"))
    baseline_stage1_facts = _fact_set(baseline["stage1"].get("relation_facts"))
    semantic_stage1_facts = _fact_set(semantic["stage1"].get("relation_facts"))
    baseline_stage2 = _selected_relations(baseline["stage2"])
    semantic_stage2 = _selected_relations(semantic["stage2"])
    baseline_stage3 = baseline["stage3"].get("relation_alternation_generation", {})
    semantic_stage3 = semantic["stage3"].get("relation_alternation_generation", {})
    baseline_final = baseline["final"]
    semantic_final = semantic["final"]
    baseline_score = _best_pair_score(baseline_final)
    semantic_score = _best_pair_score(semantic_final)
    return {
        "entity_key": entity_key,
        "entity_label": str(baseline_final.get("entity_label") or baseline["entity"].get("label") or ""),
        "semantic_statuses": {
            call["stage"]: call["status"]
            for call in calls
        },
        "stage0_baseline_relations": baseline_stage0,
        "stage0_semantic_relations": semantic_stage0,
        "stage0_relation_set_changed": set(baseline_stage0) != set(semantic_stage0),
        "stage0_relation_order_changed": baseline_stage0 != semantic_stage0,
        "stage0_top_relation_changed": _first(baseline_stage0) != _first(semantic_stage0),
        "stage1_entity_type_baseline": _entity_type(baseline["stage1"]),
        "stage1_entity_type_semantic": _entity_type(semantic["stage1"]),
        "stage1_entity_type_changed": _entity_type(baseline["stage1"]) != _entity_type(semantic["stage1"]),
        "stage1_relation_facts_changed": baseline_stage1_facts != semantic_stage1_facts,
        "stage1_relation_fact_count_delta": len(semantic_stage1_facts) - len(baseline_stage1_facts),
        "stage2_relation_selection_changed": baseline_stage2 != semantic_stage2,
        "stage2_relation_count_delta": len(semantic_stage2) - len(baseline_stage2),
        "stage3_text_primary_changed": _candidate_primary(baseline_stage3, "text_candidates") != _candidate_primary(semantic_stage3, "text_candidates"),
        "stage3_image_primary_changed": _candidate_primary(baseline_stage3, "image_candidates") != _candidate_primary(semantic_stage3, "image_candidates"),
        "baseline_best_pair_id": baseline_final.get("best_pair_id"),
        "semantic_best_pair_id": semantic_final.get("best_pair_id"),
        "best_pair_changed": baseline_final.get("best_pair_id") != semantic_final.get("best_pair_id"),
        "baseline_best_text_candidate_id": baseline_final.get("best_text_candidate_id"),
        "semantic_best_text_candidate_id": semantic_final.get("best_text_candidate_id"),
        "best_text_changed": baseline_final.get("best_text_candidate_id") != semantic_final.get("best_text_candidate_id"),
        "baseline_best_image_candidate_id": baseline_final.get("best_image_candidate_id"),
        "semantic_best_image_candidate_id": semantic_final.get("best_image_candidate_id"),
        "best_image_changed": baseline_final.get("best_image_candidate_id") != semantic_final.get("best_image_candidate_id"),
        "baseline_best_pair_score": baseline_score,
        "semantic_best_pair_score": semantic_score,
        "best_pair_score_delta": (
            None
            if baseline_score is None or semantic_score is None
            else round(semantic_score - baseline_score, 6)
        ),
    }


def _load_entity(run_dir: Path, entity_key: str) -> dict[str, Any]:
    entity_dir = run_dir / "entities" / entity_key
    return {
        "entity": _read_json(entity_dir / "final.json").get("entity", {}),
        "stage0": _read_json(entity_dir / "0_coarse_relation_filtering.json"),
        "stage1": _read_json(entity_dir / "1_knowledge_modeling.json"),
        "stage2": _read_json(entity_dir / "2_fine_grained_relation_selection.json"),
        "stage3": _read_json(entity_dir / "3_relation_alternation_generation.json"),
        "final": _read_json(entity_dir / "final.json"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("provider") == "qwen-local-stage03-semantic-planner":
            calls.append(row)
    return calls


def _relations(items: Any) -> list[str]:
    return [
        f"{item.get('relation')}::{item.get('direction', 'out')}"
        for item in _list(items)
        if isinstance(item, dict)
    ]


def _fact_set(items: Any) -> list[str]:
    facts = []
    for item in _list(items):
        if isinstance(item, dict):
            fact = str(item.get("fact") or "").strip()
            if fact:
                facts.append(fact)
    return sorted(set(facts))


def _selected_relations(stage2: dict[str, Any]) -> list[str]:
    relations = []
    for bucket in ("primary_relations", "secondary_relations", "context_relations"):
        for item in _list(stage2.get(bucket)):
            if isinstance(item, dict) and item.get("relation"):
                relations.append(str(item["relation"]))
    return relations


def _candidate_primary(stage3: dict[str, Any], key: str) -> list[str]:
    return [
        str(item.get("primary_relation") or "")
        for item in _list(stage3.get(key))
        if isinstance(item, dict)
    ]


def _entity_type(stage1: dict[str, Any]) -> str:
    profile = stage1.get("entity_profile")
    if not isinstance(profile, dict):
        return ""
    return str(profile.get("entity_type") or profile.get("type") or "")


def _best_pair_score(final: dict[str, Any]) -> float | None:
    pair = final.get("best_pair")
    if not isinstance(pair, dict):
        return None
    value = pair.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = Counter()
    status_counts = Counter()
    deltas = []
    for row in rows:
        for field in (
            "stage0_relation_set_changed",
            "stage0_relation_order_changed",
            "stage0_top_relation_changed",
            "stage1_entity_type_changed",
            "stage1_relation_facts_changed",
            "stage2_supported_facts_changed",
            "stage3_text_primary_changed",
            "stage3_image_primary_changed",
            "best_pair_changed",
            "best_text_changed",
            "best_image_changed",
        ):
            counters[field] += int(bool(row[field]))
        for stage, status in row["semantic_statuses"].items():
            status_counts[f"{stage}:{status}"] += 1
        delta = row.get("best_pair_score_delta")
        if isinstance(delta, (int, float)):
            deltas.append(float(delta))
    summary = dict(counters)
    summary["semantic_call_status_counts"] = dict(status_counts)
    summary["avg_best_pair_score_delta"] = (
        round(sum(deltas) / len(deltas), 6) if deltas else None
    )
    summary["positive_best_pair_score_delta"] = sum(1 for delta in deltas if delta > 0)
    summary["negative_best_pair_score_delta"] = sum(1 for delta in deltas if delta < 0)
    summary["zero_best_pair_score_delta"] = sum(1 for delta in deltas if delta == 0)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "entity_key",
        "entity_label",
        "stage0_relation_set_changed",
        "stage0_relation_order_changed",
        "stage0_top_relation_changed",
        "stage1_entity_type_changed",
        "stage1_relation_facts_changed",
        "stage2_supported_facts_changed",
        "stage3_text_primary_changed",
        "stage3_image_primary_changed",
        "best_pair_changed",
        "best_text_changed",
        "best_image_changed",
        "baseline_best_pair_score",
        "semantic_best_pair_score",
        "best_pair_score_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
