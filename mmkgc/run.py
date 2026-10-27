"""Command line entry point for MMKGC runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mmkgc.data import DATASET_STORE_TYPES, DatasetStore, build_dataset_store
from mmkgc.defaults import build_default_mainline_runtime
from mmkgc.io import RunWriter


REQUIRED_DATASET_FILES = ("entity2id.txt", "relation2id.txt", "train.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MMKGC MVP pipeline.")
    parser.add_argument(
        "--dataset-type",
        choices=DATASET_STORE_TYPES,
        default="mkgw",
        help="Dataset adapter type. Default 'mkgw' preserves the current MKG-W loader.",
    )
    parser.add_argument("--dataset-dir", default="datasets/MKG-W")
    parser.add_argument(
        "--graph-split",
        choices=["train", "train-valid", "all"],
        default="train",
        help=(
            "KG split used to build neighbor evidence. Default 'train' avoids "
            "valid/test graph leakage; 'all' preserves exploratory transductive runs."
        ),
    )
    parser.add_argument(
        "--missing-mode",
        choices=["text", "image"],
        default="text",
        help="Single modality to generate: text or image.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--entity-manifest",
        default=None,
        help=(
            "Optional JSON/JSONL manifest with entity_key rows. When set, records "
            "are selected in manifest order before applying --limit."
        ),
    )
    parser.add_argument("--candidate-budget", type=int, default=5)
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Record failed entities and test the remaining records; exit nonzero if any fail.",
    )
    parser.add_argument(
        "--stage1-reranker-model-dir",
        default="Qwen3-Reranker-4B",
        help="Local Qwen3-Reranker directory used for Stage 1 evidence matching.",
    )
    parser.add_argument(
        "--stage1-reranker-device", default="cuda:0",
        help="Explicit device for the local Stage 1 reranker; no automatic device fallback.",
    )
    parser.add_argument("--stage1-reranker-max-length", type=int, default=8192)
    parser.add_argument("--stage1-model-config", default="configs/gpt4omini-mainline.yaml")
    parser.add_argument("--stage2-model-config", default="configs/openai-gpt4o-mini-judge.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-config", default="configs/gpt4omini-mainline.yaml")
    parser.add_argument("--model-call-timeout-seconds", type=float, default=60)
    parser.add_argument("--stage4-config", default="configs/stage4-model-fusion.yaml")
    parser.add_argument("--sdxl-model-dir", default="SDXL/stable-diffusion-xl-base-1.0")
    parser.add_argument("--sdxl-width", type=int, default=1024)
    parser.add_argument("--sdxl-height", type=int, default=1024)
    parser.add_argument("--sdxl-steps", type=int, default=20)
    parser.add_argument("--sdxl-seed", type=int, default=20260708)
    parser.add_argument("--sdxl-guidance-scale", type=float, default=7.5)
    parser.add_argument("--sdxl-device", default="cuda")
    return parser


def validate_dataset_dir(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"dataset_dir is not a directory: {path}")

    invalid_files = []
    for filename in REQUIRED_DATASET_FILES:
        file_path = path / filename
        if not file_path.exists():
            invalid_files.append(f"{filename} (missing)")
        elif not file_path.read_text(encoding="utf-8").strip():
            invalid_files.append(f"{filename} (empty)")

    if invalid_files:
        raise ValueError(
            "dataset_dir is missing required non-empty files: "
            + ", ".join(invalid_files)
        )
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_generator_mode = (
        "sdxl-local" if args.missing_mode == "image" else "not-required"
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = validate_dataset_dir(args.dataset_dir)
    store = build_dataset_store(
        args.dataset_type,
        dataset_dir,
        graph_split=args.graph_split,
    )
    if args.entity_manifest:
        manifest_keys = load_entity_manifest(args.entity_manifest)
        records = select_records_from_manifest(
            store,
            args.missing_mode,
            manifest_keys[: args.limit] if args.limit is not None else manifest_keys,
        )
    else:
        records = store.select_records(args.missing_mode, limit=args.limit)
    if args.limit > 0 and not records:
        raise ValueError(
            f"selected records is empty for dataset_dir={dataset_dir}, "
            f"missing_mode={args.missing_mode}, limit={args.limit}"
        )

    runtime = build_default_mainline_runtime(
        missing_mode=args.missing_mode,
        candidate_budget=args.candidate_budget,
        stage1_reranker_model_dir=args.stage1_reranker_model_dir,
        stage1_reranker_device=args.stage1_reranker_device,
        stage1_reranker_max_length=args.stage1_reranker_max_length,
        stage1_model_config=args.stage1_model_config,
        stage2_model_config=args.stage2_model_config,
        model_config=args.model_config,
        model_call_timeout_seconds=args.model_call_timeout_seconds,
        stage4_config=args.stage4_config,
        dataset_root=dataset_dir,
        sdxl_model_dir=args.sdxl_model_dir,
        sdxl_width=args.sdxl_width,
        sdxl_height=args.sdxl_height,
        sdxl_steps=args.sdxl_steps,
        sdxl_seed=args.sdxl_seed,
        sdxl_guidance_scale=args.sdxl_guidance_scale,
        sdxl_device=args.sdxl_device,
    )
    pipeline = runtime.pipeline
    model_name = runtime.model_name
    base_url = runtime.base_url
    api_key_configured = runtime.api_key_configured
    image_generator_mode = runtime.image_generator_mode
    profile = getattr(getattr(pipeline, "ranking_engine", None), "profile", None)
    stage4_profile_name = getattr(profile, "profile_name", None)
    stage4_score_version = "stage4_structured_graph_v3"

    config: dict[str, Any] = {
        "dataset_dir": args.dataset_dir,
        "dataset_type": args.dataset_type,
        "graph_split": args.graph_split,
        "missing_mode": args.missing_mode,
        "limit": args.limit,
        "entity_manifest": args.entity_manifest,
        "selected_entity_count": len(records),
        "candidate_budget": args.candidate_budget,
        "continue_on_error": args.continue_on_error,
        "stage1_reranker": "qwen3-reranker-local",
        "stage1_reranker_model_dir": args.stage1_reranker_model_dir,
        "stage1_reranker_device": args.stage1_reranker_device,
        "stage1_reranker_max_length": args.stage1_reranker_max_length,
        "stage1_model_config": args.stage1_model_config,
        "stage2_model_config": args.stage2_model_config,
        "output_root": args.output_root,
        "run_name": run_name,
        "text_generator": "openai-compatible",
        "image_prompt_generator": "openai-compatible",
        "stage2_planner": "gpt4omini-global-relation-planner",
        "model_config": args.model_config,
        "model_call_timeout_seconds": args.model_call_timeout_seconds,
        "model_name": model_name,
        "base_url": base_url,
        "api_key_configured": api_key_configured,
        "image_generator": image_generator_mode,
        "stage4_ranker": "model-fusion",
        "stage4_config": args.stage4_config,
        "stage4_profile_name": stage4_profile_name,
        "stage4_judge_max_concurrency": getattr(
            getattr(profile, "gpt4omini_judge", None), "max_concurrency", None,
        ),
        "stage4_score_version": stage4_score_version,
        "ground_truth_used": False,
        "sdxl_model_dir": args.sdxl_model_dir if args.missing_mode == "image" else None,
        "sdxl_width": args.sdxl_width if args.missing_mode == "image" else None,
        "sdxl_height": args.sdxl_height if args.missing_mode == "image" else None,
        "sdxl_steps": args.sdxl_steps if args.missing_mode == "image" else None,
        "sdxl_seed": args.sdxl_seed if args.missing_mode == "image" else None,
        "sdxl_guidance_scale": (
            args.sdxl_guidance_scale if args.missing_mode == "image" else None
        ),
        "sdxl_device": args.sdxl_device if args.missing_mode == "image" else None,
    }

    writer = RunWriter(args.output_root, run_name, config)

    for record in records:
        try:
            writer.write_entity(pipeline.run(record, args.missing_mode))
        except Exception as exc:
            writer.record_failure(record, exc)
            writer.write_summary()
            if not args.continue_on_error:
                raise

    summary = writer.write_summary()
    print(
        json.dumps(
            {
                "run_dir": str(writer.run_dir),
                "processed_entities": summary["processed_entities"],
                "failed_entities": summary["failed_entities"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["failed_entities"] else 0


def load_entity_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"entity_manifest does not exist: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"entity_manifest is empty: {manifest_path}")
    if manifest_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows = payload.get("samples")
        else:
            rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("entity_manifest must contain a non-empty list of rows")
    keys = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"entity_manifest row {index} must be an object")
        key = str(row.get("entity_key") or "").strip()
        if not key:
            raise ValueError(f"entity_manifest row {index} is missing entity_key")
        if key in seen:
            raise ValueError(f"entity_manifest contains duplicate entity_key: {key}")
        seen.add(key)
        keys.append(key)
    return keys


def select_records_from_manifest(
    store: DatasetStore,
    missing_mode: str,
    entity_keys: list[str],
) -> list[Any]:
    try:
        return store.select_records_by_keys(missing_mode, entity_keys)
    except ValueError as exc:
        raise ValueError(f"entity_manifest contains {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
