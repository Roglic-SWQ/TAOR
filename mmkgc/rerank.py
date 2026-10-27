"""Replay Stage 4 model-fusion ranking for an existing MMKGC run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

from mmkgc.io.stage_artifacts import unwrap_stage_artifact
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.image_input import ImageInput, ImageInputError
from mmkgc.utils import write_json


DEFAULT_OUTPUT_DIR_NAME = "stage4_model_fusion_rerank"
SCORE_VERSION = "stage4_structured_graph_v3"


class RerankError(RuntimeError):
    """Raised when an existing run cannot be safely replay-ranked."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Stage 4 model-fusion ranking.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-manifest", default=None)
    parser.add_argument("--stage4-config", default="configs/stage4-model-fusion.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    engine_factory: Callable[..., Any] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if engine_factory is None:
        from mmkgc.ranking.factory import build_model_fusion_engine

        engine_factory = build_model_fusion_engine

    project_root = Path(__file__).resolve().parents[1]
    engine = engine_factory(project_root, args.stage4_config)
    summary = rerank_run(
        args.run_dir,
        sample_manifest=args.sample_manifest,
        stage4_config=args.stage4_config,
        output_dir=args.output_dir,
        limit=args.limit,
        engine=engine,
        project_root=project_root,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def rerank_run(
    run_dir: str | Path,
    *,
    sample_manifest: str | Path | None = None,
    stage4_config: str | Path = "configs/stage4-model-fusion.yaml",
    output_dir: str | Path | None = None,
    limit: int | None = None,
    engine: Any | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rerank selected entities from an existing run into an isolated directory."""
    source_run = Path(run_dir).resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {source_run}")
    entities_dir = source_run / "entities"
    if not entities_dir.is_dir():
        raise RerankError(f"run_dir is missing entities directory: {entities_dir}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else source_run / DEFAULT_OUTPUT_DIR_NAME
    )
    if destination.exists():
        raise FileExistsError(f"rerank output directory already exists: {destination}")
    destination.mkdir(parents=True)
    (destination / "entities").mkdir()

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    if engine is None:
        from mmkgc.ranking.factory import build_model_fusion_engine

        engine = build_model_fusion_engine(root, stage4_config)

    source_config = _read_json_object(source_run / "config.json", required=False)
    entity_index = _index_source_entities(entities_dir)
    manifest_payload, requested_keys = _load_manifest(sample_manifest)
    selected = _select_entities(entity_index, requested_keys, limit)

    config = {
        "source_run_dir": str(source_run),
        "sample_manifest": str(Path(sample_manifest).resolve()) if sample_manifest else None,
        "stage4_config": str(stage4_config),
        "ranker": "model-fusion",
        "stage4_ranker": "model-fusion",
        "stage4_profile_name": _profile_name(engine),
        "stage4_score_version": SCORE_VERSION,
        "ground_truth_used": False,
        "limit": limit,
    }
    write_json(destination / "config.json", _safe_output(config))
    write_json(
        destination / "sample_manifest.json",
        _safe_output(
            manifest_payload
            if manifest_payload is not None
            else {"samples": [{"entity_key": key} for key in selected]}
        ),
    )
    results_path = destination / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    processed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entity_key in selected:
        entity_dir = entity_index[entity_key]
        try:
            result = _rerank_entity(
                entity_key,
                entity_dir,
                source_run=source_run,
                source_config=source_config,
                engine=engine,
            )
        except RerankError:
            raise
        except Exception as exc:
            failure = {
                "entity_key": entity_key,
                "stage": "rerank",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            diagnostics = getattr(exc, "diagnostic_events", None)
            if isinstance(diagnostics, list) and diagnostics:
                failure["diagnostic_events"] = diagnostics
            failures.append(failure)
            continue

        final = result["final"]
        processed.append(final)
        output_entity_dir = destination / "entities" / _entity_dir_name(entity_key)
        output_entity_dir.mkdir(parents=True)
        write_json(
            output_entity_dir / "4_kg_aware_candidate_consistency_ranking.json",
            _safe_output(result["kg_aware_candidate_consistency_ranking"]),
        )
        write_json(output_entity_dir / "final.json", _safe_output(final))
        if result["ranking_model_calls"]:
            _write_jsonl(
                output_entity_dir / "ranking_model_calls.jsonl",
                result["ranking_model_calls"],
            )
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_safe_output(final), ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = _summary(processed, failures)
    write_json(destination / "metrics_summary.json", _safe_output(summary))
    if failures:
        write_json(destination / "failures.json", _safe_output(failures))
    return summary


def _rerank_entity(
    entity_key: str,
    entity_dir: Path,
    *,
    source_run: Path,
    source_config: Mapping[str, Any],
    engine: Any,
) -> dict[str, Any]:
    final = _read_json_object(entity_dir / "final.json")
    entity = _entity_payload(final, entity_key)
    missing_mode = str(final.get("missing_mode") or source_config.get("missing_mode") or "text")
    knowledge = unwrap_stage_artifact(_read_json_object(entity_dir / "1_knowledge_modeling.json"))
    reasoning = knowledge.get("constraint_compilation", {})
    generation = _stage4_generation_payload(
        unwrap_stage_artifact(_read_json_object(entity_dir / "3_relation_alternation_generation.json"))
    )
    image_inputs = _generated_image_inputs_from_generation(
        entity_key,
        entity_dir,
        generation,
    )
    context = build_ranking_context(
        entity_key=entity_key,
        entity_label=str(entity.get("label") or entity_key),
        missing_mode=missing_mode,
        existing_modalities=_existing_modalities(entity),
        knowledge=knowledge,
        reasoning=reasoning,
        generation=generation,
        generated_image_inputs=image_inputs,
    )
    ranking = engine.rank(context)
    ranking_model_calls = list(ranking.get("ranking_model_calls", []))
    return {
        "entity": entity,
        "kg_aware_candidate_consistency_ranking": ranking,
        "ranking_model_calls": ranking_model_calls,
        "final": _final_result(entity, missing_mode, ranking),
    }


def _stage4_generation_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {"relation_alternation_generation": {}}
    if isinstance(payload.get("relation_alternation_generation"), Mapping):
        return payload
    return {"relation_alternation_generation": dict(payload)}


def _generated_image_inputs_from_generation(
    entity_key: str,
    entity_dir: Path,
    generation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    unwrapped = unwrap_stage_artifact(generation)
    stage_four = unwrapped.get("relation_alternation_generation", unwrapped)
    if not isinstance(stage_four, Mapping):
        return {}
    candidates = stage_four.get("image_candidates", [])
    if not isinstance(candidates, list):
        return {}
    inputs: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("candidate_id")
        generated = candidate.get("generated_content", {})
        if not isinstance(candidate_id, str) or not isinstance(generated, Mapping):
            continue
        relative_path = generated.get("image_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            continue
        try:
            image_input = ImageInput.from_entity_path(
                entity_key=entity_key,
                candidate_id=candidate_id,
                entity_dir=entity_dir,
                relative_path=relative_path,
            )
        except ImageInputError as exc:
            raise RerankError(
                f"unsafe or unreadable image path for {entity_key}/{candidate_id}: {relative_path}"
            ) from exc
        inputs[candidate_id] = {
            "image_path": image_input.image_path,
            "media_type": image_input.media_type,
        }
    return inputs


def _index_source_entities(entities_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    root = entities_dir.resolve()
    for entity_dir in sorted(entities_dir.iterdir()):
        if not entity_dir.is_dir():
            continue
        resolved = entity_dir.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RerankError(f"entity directory escapes run entities root: {entity_dir}") from exc
        final = _read_json_object(entity_dir / "final.json", required=False)
        key = str(
            final.get("entity", {}).get("entity_key")
            if isinstance(final.get("entity"), Mapping)
            else ""
        ).strip() or entity_dir.name
        if key in index:
            raise RerankError(f"duplicate entity key in source run: {key}")
        index[key] = entity_dir
    if not index:
        raise RerankError(f"no rerankable entities found in {entities_dir}")
    return index


def _load_manifest(path: str | Path | None) -> tuple[Any | None, list[str] | None]:
    if path is None:
        return None, None
    manifest_path = Path(path)
    payload = _read_json_object(manifest_path)
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, Mapping):
        samples = payload.get("samples", payload.get("entities", []))
    else:
        raise RerankError("sample manifest must be a JSON object or array")
    if not isinstance(samples, list):
        raise RerankError("sample manifest samples/entities must be a list")
    keys: list[str] = []
    for sample in samples:
        if isinstance(sample, str):
            key = sample
        elif isinstance(sample, Mapping):
            key = str(sample.get("entity_key") or sample.get("entity") or "")
        else:
            key = ""
        key = key.strip()
        if not key:
            raise RerankError("sample manifest contains an entity without entity_key")
        keys.append(key)
    return payload, keys


def _select_entities(
    entity_index: Mapping[str, Path],
    requested_keys: list[str] | None,
    limit: int | None,
) -> list[str]:
    selected = requested_keys if requested_keys is not None else sorted(entity_index)
    missing = [key for key in selected if key not in entity_index]
    if missing:
        raise RerankError(f"sample manifest references unknown entity: {missing[0]}")
    if limit is not None:
        selected = selected[:limit]
    return list(selected)


def _final_result(entity: Mapping[str, Any], missing_mode: str, ranking: Mapping[str, Any]) -> dict[str, Any]:
    text_candidates = ranking.get("text_ranking", {}).get("ranked_candidates", [])
    image_candidates = ranking.get("image_ranking", {}).get("ranked_candidates", [])
    ranked_pairs = ranking.get("pair_ranking", {}).get("ranked_pairs", [])
    top_5_pairs = ranking.get("pair_ranking", {}).get("top_5_pairs", [])
    best_pair = ranking.get("pair_ranking", {}).get("best_pair")
    return {
        "entity": dict(entity),
        "missing_mode": missing_mode,
        "best_text_candidate_id": ranking.get("text_ranking", {}).get("best_candidate_id"),
        "best_image_candidate_id": ranking.get("image_ranking", {}).get("best_candidate_id"),
        "best_pair_id": ranking.get("pair_ranking", {}).get("best_pair_id"),
        "best_pair": best_pair,
        "top_5_pairs": top_5_pairs,
        "counts": {
            "text": len(text_candidates) if isinstance(text_candidates, list) else 0,
            "image": len(image_candidates) if isinstance(image_candidates, list) else 0,
            "pair": len(ranked_pairs) if isinstance(ranked_pairs, list) else 0,
        },
        "ranking_status": ranking.get("ranking_status"),
        "stage4_ranker": "model-fusion",
        "stage4_profile_name": ranking.get("profile_name"),
        "stage4_score_version": ranking.get("score_version", SCORE_VERSION),
        "ground_truth_used": bool(
            ranking.get("kg_aware_candidate_consistency_ranking", {}).get(
                "ground_truth_used",
                False,
            )
        ),
    }


def _summary(processed: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    text_counts = [int(item.get("counts", {}).get("text", 0) or 0) for item in processed]
    image_counts = [int(item.get("counts", {}).get("image", 0) or 0) for item in processed]
    pair_counts = [int(item.get("counts", {}).get("pair", 0) or 0) for item in processed]
    return {
        "processed_entities": len(processed),
        "failed_entities": len(failures),
        "avg_text_candidates": _average(text_counts),
        "avg_image_candidates": _average(image_counts),
        "avg_pair_candidates": _average(pair_counts),
        "stage_failure_counts": dict(Counter(item["stage"] for item in failures)),
    }


def _read_json_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing required rerank input: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RerankError(f"invalid JSON input: {path}") from exc
    if not isinstance(payload, dict):
        raise RerankError(f"rerank input must be a JSON object: {path}")
    return payload


def _entity_payload(final: Mapping[str, Any], fallback_key: str) -> dict[str, Any]:
    entity = final.get("entity")
    if isinstance(entity, Mapping):
        return dict(entity)
    return {"entity_key": fallback_key, "label": fallback_key, "existing_modalities": {}}


def _existing_modalities(entity: Mapping[str, Any]) -> dict[str, Any]:
    existing = entity.get("existing_modalities")
    return dict(existing) if isinstance(existing, Mapping) else {"text": None, "image": None}


def _profile_name(engine: Any) -> str | None:
    return getattr(getattr(engine, "profile", None), "profile_name", None)


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_safe_output(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _safe_output(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"redacted_bytes": len(value)}
    if is_dataclass(value):
        return _safe_output(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_output(item)
            for key, item in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_output(item) for item in value]
    return value


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return (
        normalized in {"api_key", "authorization", "headers"}
        or normalized.endswith("_token")
        or normalized.endswith("_secret")
        or normalized in {"token", "secret", "refresh_token", "access_token"}
    )


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _entity_dir_name(entity_key: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in entity_key)
    return safe.strip("._") or "entity"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
