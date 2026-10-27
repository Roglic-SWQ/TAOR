import json
from pathlib import Path

from mmkgc.data import EntityRecord, MKGWDataStore
from mmkgc.io import RunWriter
from mmkgc.io.writer import _stage_artifact
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.run import build_parser, main, load_entity_manifest, validate_dataset_dir

try:
    from tests.test_data import make_fixture
except ImportError:

    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def make_fixture(tmp_path: Path) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        _write(
            tmp_path / "entity2id.txt",
            "\n".join(
                [
                    "http://www.wikidata.org/entity/Q1 0",
                    "http://www.wikidata.org/entity/Q2 1",
                    "http://www.wikidata.org/entity/Q3 2",
                ]
            )
            + "\n",
        )
        _write(
            tmp_path / "relation2id.txt",
            "\n".join(
                [
                    "http://www.wikidata.org/entity/P31 0",
                    "http://www.wikidata.org/entity/P17 1",
                ]
            )
            + "\n",
        )
        _write(
            tmp_path / "train.txt",
            "\n".join(
                [
                    "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .",
                    "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P17 http://www.wikidata.org/entity/Q3 .",
                ]
            )
            + "\n",
        )
        _write(tmp_path / "valid.txt", "")
        _write(tmp_path / "test.txt", "")
        (tmp_path / "entity_info.json").write_text(
            json.dumps(
                {
                    "Q1": {"label": "Example Film", "description": "a fictional film"},
                    "Q2": {"label": "film", "description": "type of creative work"},
                    "Q3": {"label": "Exampleland", "description": "a country"},
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "relation_info.json").write_text(
            json.dumps({"P31": "instance of", "P17": "country"}),
            encoding="utf-8",
        )
        image_dir = tmp_path / "images" / "Example Film"
        image_dir.mkdir(parents=True)
        (image_dir / "image.jpg").write_bytes(b"fake")
        return tmp_path














def test_stage3_single_modality_quota_plan_omits_duplicate_modality_quota():
    artifact = _stage_artifact(
        "fine_grained_relation_selection",
        {
            "fine_grained_relation_selection": {
                "target_modality": ["image"],
                "primary_relations": [],
                "secondary_relations": [],
                "context_relations": [],
                "excluded_relations": [],
                "quota_summary": {
                    "candidate_budget": 3,
                    "quota_mode": "single_modality",
                    "target_modalities": ["image"],
                    "primary_quota": {"composer": 1},
                    "secondary_quota": {},
                    "relation_quotas": {"composer": 1},
                    "assigned_total": 1,
                    "fallback_quota": 2,
                    "global_profile_fallback": True,
                    "modality_quota": {
                        "image": {
                            "candidate_budget": 3,
                            "primary_quota": {"composer": 1},
                            "secondary_quota": {},
                            "relation_quotas": {"composer": 1},
                            "assigned_total": 1,
                            "fallback_quota": 2,
                            "global_profile_fallback": True,
                        }
                    },
                },
            }
        },
    )

    quota_plan = artifact["data"]["quota_plan"]

    assert "schema_version" not in artifact
    assert quota_plan["quota_mode"] == "single_modality"
    assert quota_plan["target_modalities"] == ["image"]
    assert quota_plan["relation_quotas"] == {"composer": 1}
    assert "modality_quota" not in quota_plan














def test_run_writer_refuses_existing_run_and_preserves_results(tmp_path):
    output_root = tmp_path / "outputs"
    writer = RunWriter(output_root, "same_name", {"run": 1})
    writer.results_path.write_text("already here\n", encoding="utf-8")

    try:
        RunWriter(output_root, "same_name", {"run": 2})
    except FileExistsError as exc:
        assert "same_name" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError for existing run directory")

    assert writer.results_path.read_text(encoding="utf-8") == "already here\n"


def test_validate_dataset_dir_rejects_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    try:
        validate_dataset_dir(missing_dir)
    except FileNotFoundError as exc:
        assert "dataset_dir does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing dataset directory")


def test_validate_dataset_dir_rejects_missing_or_empty_required_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "entity2id.txt").write_text("http://www.wikidata.org/entity/Q1 0\n", encoding="utf-8")
    (dataset_dir / "relation2id.txt").write_text("", encoding="utf-8")

    try:
        validate_dataset_dir(dataset_dir)
    except ValueError as exc:
        message = str(exc)
        assert "relation2id.txt" in message
        assert "train.txt" in message
    else:
        raise AssertionError("Expected ValueError for missing or empty required dataset files")


def test_cli_rejects_positive_limit_when_no_records_selected(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "entity2id.txt").write_text("malformed\n", encoding="utf-8")
    (dataset_dir / "relation2id.txt").write_text("malformed\n", encoding="utf-8")
    (dataset_dir / "train.txt").write_text("malformed\n", encoding="utf-8")
    output_root = tmp_path / "outputs"

    try:
        main(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--limit",
                "1",
                "--output-root",
                str(output_root),
                "--run-name",
                "empty_selection",
                "--stage4-ranker",
                "legacy",
            ]
        )
    except ValueError as exc:
        assert "selected records is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError when positive limit selects no records")

    assert not (output_root / "empty_selection").exists()




def test_cli_help_includes_dataset_type_option():
    help_text = build_parser().format_help()

    assert "--dataset-type" in help_text


def test_load_entity_manifest_rejects_duplicate_keys(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"entity_key": "Q1"}\n{"entity_key": "Q1"}\n',
        encoding="utf-8",
    )

    try:
        load_entity_manifest(manifest_path)
    except ValueError as exc:
        assert "duplicate entity_key" in str(exc)
    else:
        raise AssertionError("Expected duplicate manifest keys to be rejected")












def test_writer_rejects_generated_image_artifact_paths_outside_entity_dir(tmp_path):
    output_root = tmp_path / "outputs"
    writer = RunWriter(output_root, "unsafe_artifact", {"image_generator": "sdxl-local"})
    absolute_escape = tmp_path / "escape.png"

    for relative_image_path, outside_path in (
        ("../escape.png", writer.run_dir / "entities" / "escape.png"),
        (str(absolute_escape), absolute_escape),
    ):
        result = {
            "entity": {"entity_key": "Q1"},
            "final": {"entity_key": "Q1", "counts": {"text": 0, "image": 0, "pair": 0}},
            "generated_image_artifacts": [
                {
                    "entity_key": "Q1",
                    "candidate_id": "image_001",
                    "relative_image_path": relative_image_path,
                    "relative_metadata_path": "generated_images/image_001.json",
                    "image_bytes": b"unsafe",
                    "metadata": {"status": "generated"},
                    "model_call": {"modality": "image"},
                }
            ],
        }

        try:
            writer.write_entity(result)
        except ValueError as exc:
            assert "entity directory" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unsafe generated image artifact path")

        assert not outside_path.exists()


def test_writer_rejects_unsafe_generated_image_metadata_path_before_writing(tmp_path):
    writer = RunWriter(tmp_path / "outputs", "unsafe_metadata_artifact", {"image_generator": "sdxl-local"})
    entity_dir = writer.run_dir / "entities" / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.png"
    escaped_metadata_path = writer.run_dir / "entities" / "escape.json"
    result = {
        "entity": {"entity_key": "Q1"},
        "final": {"entity_key": "Q1", "counts": {"text": 0, "image": 0, "pair": 0}},
        "generated_image_artifacts": [
            {
                "entity_key": "Q1",
                "candidate_id": "image_001",
                "relative_image_path": "generated_images/image_001.png",
                "relative_metadata_path": "../escape.json",
                "image_bytes": b"unsafe",
                "metadata": {"status": "generated"},
                "model_call": {"modality": "image"},
            }
        ],
    }

    try:
        writer.write_entity(result)
    except ValueError as exc:
        assert "entity directory" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsafe generated image metadata path")

    assert not image_path.exists()
    assert not escaped_metadata_path.exists()


def _empty_record() -> EntityRecord:
    return EntityRecord(
        entity_id=99,
        entity_key="empty",
        label="Empty Entity",
        description="",
        neighbors=[],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )
