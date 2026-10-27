#!/usr/bin/env python3
"""Prepare a Knowledge-Bridger (src/) COCO-like input from an MMKGC run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=5)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    manifest = read_json(args.run_dir / "original_images_manifest.json")
    descriptions = read_json(args.run_dir / "entity_descriptions.json")
    description_rows = {
        str(row["entity_key"]): row
        for row in descriptions["entities"]
    }

    metadata: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    missing_details: dict[str, list[int]] = {}
    entity_rows: list[dict[str, Any]] = []

    for idx, entity in enumerate(manifest["entities"]):
        entity_key = str(entity["entity_key"])
        description_row = description_rows.get(entity_key)
        if description_row is None:
            raise ValueError(f"missing description for {entity_key}")
        images = entity.get("images") or []
        if not images:
            raise ValueError(f"missing source image for {entity_key}")

        # Use the exact image copied into this MMKGC run. This avoids relying
        # on the manifest's repository-relative source_path convention.
        source_image = args.run_dir / entity["copied_directory"] / images[0]["file_name"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        metadata.append(
            {
                "entity_key": entity_key,
                "label": entity["label"],
                "category": entity.get("category"),
                "captions": [description_row["description"]],
                "file_name": str(source_image.resolve()),
            }
        )
        missing_indices.append(idx)
        # src/knowledge_extracting.py interprets [1, 0] as image missing,
        # therefore the available modality passed to the model is text.
        missing_details[str(idx)] = [1, 0]
        entity_rows.append(
            {
                "idx": idx,
                "entity_key": entity_key,
                "label": entity["label"],
                "category": entity.get("category"),
                "description": description_row["description"],
                "source_image": str(source_image.resolve()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_anno.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "missing-image-50.json").write_text(
        json.dumps(
            {
                "type": "image",
                "missing_ratio": 1.0,
                "num_missing": len(missing_indices),
                "total": len(metadata),
                "missing_indices": missing_indices,
                "missing_details": missing_details,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "entity_manifest.json").write_text(
        json.dumps(
            {
                "source_run": str(args.run_dir.resolve()),
                "source_dataset": str(args.dataset_dir.resolve()),
                "entity_count": len(entity_rows),
                "num_candidates": args.num_candidates,
                "input_type": "text",
                "missing_type": "image",
                "entities": entity_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "entity_count": len(metadata),
                "num_candidates": args.num_candidates,
                "input_type": "text",
                "missing_type": "image",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
