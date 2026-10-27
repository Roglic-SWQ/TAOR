from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export only generated images with best_image_candidate_id into a dataset copy."
    )
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--output-dataset-dir", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def export_valid_best_images(
    *,
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    run_dirs: list[Path],
    limit: int | None = None,
) -> dict[str, Any]:
    if not source_dataset_dir.is_dir():
        raise FileNotFoundError(f"missing source dataset: {source_dataset_dir}")
    if output_dataset_dir.exists():
        shutil.rmtree(output_dataset_dir)
    shutil.copytree(source_dataset_dir, output_dataset_dir)

    rows = collect_best_rows(run_dirs)
    selected_rows = rows[:limit] if limit is not None else rows
    for row in selected_rows:
        target_dir = output_dataset_dir / "images" / row["entity_key"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image = target_dir / "image.png"
        shutil.copy2(row["source_image"], target_image)
        row["target_image"] = str(target_image)

    manifest_path = output_dataset_dir / "generated_image_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_tsv(output_dataset_dir / "generated_image_manifest.tsv", selected_rows)
    write_readme(output_dataset_dir, source_dataset_dir, run_dirs, selected_rows, len(rows))

    return {
        "ok": True,
        "source_dataset_dir": str(source_dataset_dir),
        "output_dataset_dir": str(output_dataset_dir),
        "available_best": len(rows),
        "exported": len(selected_rows),
        "manifest": str(manifest_path),
    }


def collect_best_rows(run_dirs: list[Path]) -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        entities_dir = run_dir / "entities"
        if not entities_dir.is_dir():
            raise FileNotFoundError(f"missing run entities dir: {entities_dir}")
        for final_path in sorted(entities_dir.glob("*/final.json")):
            entity_key = final_path.parent.name
            if entity_key in seen:
                continue
            final = read_json(final_path)
            candidate_id = final.get("best_image_candidate_id")
            if not candidate_id:
                continue
            source_image = final_path.parent / "generated_images" / f"{candidate_id}.png"
            if not source_image.exists():
                continue
            seen.add(entity_key)
            rows.append(
                {
                    "entity_key": entity_key,
                    "run_dir": str(run_dir),
                    "candidate_id": candidate_id,
                    "source_image": str(source_image),
                    "target_image": "",
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["entity_key", "run_dir", "candidate_id", "source_image", "target_image"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(row[field] for field in fields) + "\n")


def write_readme(
    output_dataset_dir: Path,
    source_dataset_dir: Path,
    run_dirs: list[Path],
    rows: list[dict[str, str]],
    available_best: int,
) -> None:
    lines = [
        "# Generated Valid-Best Image Completion Export",
        "",
        f"Source dataset: {source_dataset_dir}",
        f"Export target: {output_dataset_dir}",
        f"Available valid-best entities: {available_best}",
        f"Exported entities: {len(rows)}",
        "Image placement: images/<QID>/image.png",
        "Manifest: generated_image_manifest.jsonl and generated_image_manifest.tsv",
        "",
        "Source runs:",
        *[f"- {run_dir}" for run_dir in run_dirs],
        "",
    ]
    (output_dataset_dir / "README.generated-images.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    summary = export_valid_best_images(
        source_dataset_dir=Path(args.source_dataset_dir),
        output_dataset_dir=Path(args.output_dataset_dir),
        run_dirs=[Path(run_dir) for run_dir in args.run_dir],
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
