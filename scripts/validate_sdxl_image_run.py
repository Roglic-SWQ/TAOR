"""Validate engineering artifacts from a local SDXL image smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def validate_run(
    run_dir: str | Path,
    expected_entities: int,
    expected_images_per_entity: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, Any]:
    """Validate SDXL run artifacts and return a compact success summary."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise AssertionError(f"run_dir does not exist: {run_path}")
    if not run_path.is_dir():
        raise AssertionError(f"run_dir is not a directory: {run_path}")

    metrics_path = run_path / "metrics_summary.json"
    if not metrics_path.exists():
        raise AssertionError(f"metrics_summary.json is missing: {metrics_path}")
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, dict):
        raise AssertionError("metrics_summary.json must contain a JSON object")
    if metrics.get("processed_entities") != expected_entities:
        raise AssertionError(
            "metrics_summary.json processed_entities "
            f"expected {expected_entities}, got {metrics.get('processed_entities')}"
        )
    if metrics.get("failed_entities") != 0:
        raise AssertionError(
            "metrics_summary.json failed_entities expected 0, "
            f"got {metrics.get('failed_entities')}"
        )

    entities_dir = run_path / "entities"
    if not entities_dir.exists():
        raise AssertionError(f"entities directory is missing: {entities_dir}")
    entity_dirs = sorted(path for path in entities_dir.iterdir() if path.is_dir())
    if len(entity_dirs) != expected_entities:
        raise AssertionError(
            f"entities directory expected {expected_entities} entity dirs, got {len(entity_dirs)}"
        )

    images_checked = 0
    for entity_dir in entity_dirs:
        image_paths = _validate_entity(
            entity_dir,
            expected_images_per_entity=expected_images_per_entity,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        images_checked += len(image_paths)

    return {"ok": True, "entities_checked": len(entity_dirs), "images_checked": images_checked}


def _validate_entity(
    entity_dir: Path,
    *,
    expected_images_per_entity: int,
    expected_width: int,
    expected_height: int,
) -> list[Path]:
    images_dir = entity_dir / "generated_images"
    if not images_dir.exists():
        raise AssertionError(f"generated_images directory is missing: {images_dir}")

    image_paths = sorted(images_dir.glob("*.png"))
    if len(image_paths) != expected_images_per_entity:
        raise AssertionError(
            f"{entity_dir.name} expected {expected_images_per_entity} PNG files, got {len(image_paths)}"
        )

    calls_path = entity_dir / "image_model_calls.jsonl"
    if not calls_path.exists():
        raise AssertionError(f"image_model_calls.jsonl is missing: {calls_path}")
    image_call_rows = _read_jsonl(calls_path)
    if len(image_call_rows) < expected_images_per_entity:
        raise AssertionError(
            f"{calls_path} expected at least {expected_images_per_entity} image rows, "
            f"got {len(image_call_rows)}"
        )

    for image_path in image_paths:
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists():
            raise AssertionError(f"metadata JSON is missing for PNG: {metadata_path}")
        _read_json(metadata_path)
        _validate_image_size(image_path, expected_width, expected_height)

    return image_paths


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"invalid JSONL row in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AssertionError(f"{path}:{line_number} must contain a JSON object")
        if row.get("modality") == "image":
            rows.append(row)
    return rows


def _validate_image_size(image_path: Path, expected_width: int, expected_height: int) -> None:
    try:
        with Image.open(image_path) as image:
            if image.size != (expected_width, expected_height):
                raise AssertionError(
                    f"{image_path} expected size {(expected_width, expected_height)}, got {image.size}"
                )
            image.load()
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"could not open PNG with Pillow: {image_path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate artifacts from an MMKGC SDXL image run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-entities", required=True, type=int)
    parser.add_argument("--expected-images-per-entity", required=True, type=int)
    parser.add_argument("--expected-width", required=True, type=int)
    parser.add_argument("--expected-height", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate_run(
            args.run_dir,
            expected_entities=args.expected_entities,
            expected_images_per_entity=args.expected_images_per_entity,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
        )
    except AssertionError as exc:
        print(f"SDXL image run validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
