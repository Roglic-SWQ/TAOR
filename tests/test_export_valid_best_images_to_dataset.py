from __future__ import annotations

import json
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_valid_best_images_to_dataset.py"
spec = importlib.util.spec_from_file_location("export_valid_best_images_to_dataset", SCRIPT_PATH)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
export_valid_best_images = module.export_valid_best_images


def write_candidate(entity_dir: Path, candidate_id: str, content: bytes = b"png") -> None:
    image_dir = entity_dir / "generated_images"
    image_dir.mkdir(parents=True)
    (image_dir / f"{candidate_id}.png").write_bytes(content)


def write_final(entity_dir: Path, candidate_id: str | None) -> None:
    entity_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    if candidate_id is not None:
        payload["best_image_candidate_id"] = candidate_id
    (entity_dir / "final.json").write_text(json.dumps(payload), encoding="utf-8")


def test_exports_only_best_images_from_multiple_runs_with_limit(tmp_path: Path) -> None:
    source_dataset = tmp_path / "MKG-W"
    (source_dataset / "images").mkdir(parents=True)
    (source_dataset / "entity2id.txt").write_text("Q1\t0\nQ2\t1\nQ3\t2\n", encoding="utf-8")

    run_a = tmp_path / "run_a"
    q1 = run_a / "entities" / "Q1"
    write_candidate(q1, "image_003", b"q1-best")
    write_final(q1, "image_003")

    q2 = run_a / "entities" / "Q2"
    write_candidate(q2, "image_001", b"q2-fallback")
    write_final(q2, None)

    run_b = tmp_path / "run_b"
    q3 = run_b / "entities" / "Q3"
    write_candidate(q3, "image_002", b"q3-best")
    write_final(q3, "image_002")

    output_dataset = tmp_path / "MKG-W-filled"
    summary = export_valid_best_images(
        source_dataset_dir=source_dataset,
        output_dataset_dir=output_dataset,
        run_dirs=[run_a, run_b],
        limit=1,
    )

    assert summary["exported"] == 1
    assert summary["available_best"] == 2
    assert (output_dataset / "images" / "Q1" / "image.png").read_bytes() == b"q1-best"
    assert not (output_dataset / "images" / "Q2").exists()
    assert not (output_dataset / "images" / "Q3").exists()

    manifest_rows = [
        json.loads(line)
        for line in (output_dataset / "generated_image_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows == [
        {
            "entity_key": "Q1",
            "run_dir": str(run_a),
            "candidate_id": "image_003",
            "source_image": str(q1 / "generated_images" / "image_003.png"),
            "target_image": str(output_dataset / "images" / "Q1" / "image.png"),
        }
    ]
