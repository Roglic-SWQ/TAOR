from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy generated entity images from an MMKGC run into an MKG-W-style dataset copy."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--image-name", default="image_001.png")
    parser.add_argument(
        "--selection",
        choices=["named", "best"],
        default="named",
        help="Select a fixed image name or the best image candidate recorded in final.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    dataset_dir = Path(args.dataset_dir)
    entity_root = run_dir / "entities"
    if not entity_root.is_dir():
        raise FileNotFoundError(f"missing run entity directory: {entity_root}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"missing dataset directory: {dataset_dir}")

    rows = []
    for entity_dir in sorted(entity_root.iterdir()):
        if not entity_dir.is_dir():
            continue
        entity_key = entity_dir.name
        generated_dir = entity_dir / "generated_images"
        source_image, selection_status = select_source_image(
            entity_dir=entity_dir,
            generated_dir=generated_dir,
            image_name=args.image_name,
            selection=args.selection,
        )

        target_dir = dataset_dir / "images" / entity_key
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image = target_dir / "image.png"
        shutil.copy2(source_image, target_image)

        source_metadata = source_image.with_suffix(".json")
        rows.append(
            {
                "entity_key": entity_key,
                "source_image": str(source_image),
                "target_image": str(target_image),
                "source_metadata": str(source_metadata) if source_metadata.exists() else "",
                "run_dir": str(run_dir),
                "candidate_id": source_image.stem,
                "selection": args.selection,
                "selection_status": selection_status,
            }
        )

    manifest_jsonl = dataset_dir / "generated_image_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_tsv = dataset_dir / "generated_image_manifest.tsv"
    fields = [
        "entity_key",
        "source_image",
        "target_image",
        "source_metadata",
        "run_dir",
        "candidate_id",
        "selection",
        "selection_status",
    ]
    with manifest_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(row[field] for field in fields) + "\n")

    readme = dataset_dir / "README.generated-images.md"
    readme.write_text(
        "\n".join(
            [
                "# Generated Image Completion Export",
                "",
                "Source dataset: datasets/MKG-W",
                f"Source run: {run_dir}",
                f"Export target: {dataset_dir}",
                f"Generated image entities exported: {len(rows)}",
                "Image placement: images/<QID>/image.png",
                "Manifest: generated_image_manifest.jsonl and generated_image_manifest.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "exported": len(rows),
                "dataset_dir": str(dataset_dir),
                "manifest": str(manifest_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def select_source_image(
    *,
    entity_dir: Path,
    generated_dir: Path,
    image_name: str,
    selection: str,
) -> tuple[Path, str]:
    if selection == "best":
        final_path = entity_dir / "final.json"
        if final_path.exists():
            final = read_json(final_path)
            candidate_id = final.get("best_image_candidate_id")
            if candidate_id:
                best_image = generated_dir / f"{candidate_id}.png"
                if best_image.exists():
                    return best_image, "best_image_candidate_id"
                return fallback_first_png(generated_dir), "best_missing_fallback_first_png"
            return fallback_first_png(generated_dir), "fallback_first_png"
        return fallback_first_png(generated_dir), "missing_final_fallback_first_png"

    source_image = generated_dir / image_name
    if source_image.exists():
        return source_image, "named_image"
    return fallback_first_png(generated_dir), "named_missing_fallback_first_png"


def fallback_first_png(generated_dir: Path) -> Path:
    candidates = sorted(generated_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"missing generated image in: {generated_dir}")
    return candidates[0]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
