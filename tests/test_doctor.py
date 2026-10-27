import json
from pathlib import Path

from mmkgc.doctor import main, run_doctor


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write(
        tmp_path / "entity2id.txt",
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 0",
                "http://www.wikidata.org/entity/Q2 1",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "relation2id.txt",
        "http://www.wikidata.org/entity/P31 0\n",
    )
    _write(
        tmp_path / "train.txt",
        (
            "http://www.wikidata.org/entity/Q1 "
            "http://www.wikidata.org/entity/P31 "
            "http://www.wikidata.org/entity/Q2 .\n"
        ),
    )
    _write(tmp_path / "valid.txt", "")
    _write(tmp_path / "test.txt", "")
    _write(
        tmp_path / "entity_info.json",
        json.dumps(
            {
                "Q1": {"label": "Example Film", "description": "a fictional film"},
                "Q2": {"label": "film", "description": "type of creative work"},
            }
        ),
    )
    _write(tmp_path / "relation_info.json", json.dumps({"P31": "instance of"}))
    return tmp_path


def _qwen_config(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "configs" / "qwen.yaml",
        """
lmm:
  model_name: Qwen/Qwen2-VL-7B-Instruct
  base_url: http://127.0.0.1:12345/v1
  api_key: local-key
inference:
  max_token: 128
  temperature: 0.1
""",
    )


def _stage4_profile(tmp_path: Path, qwen_config: Path) -> Path:
    return _write(
        tmp_path / "configs" / "stage4.yaml",
        f"""
profile_name: test_local
sentence_bert:
  required: true
  enabled: true
  model_dir: models/mpnet
  device: cuda:1
clip:
  required: true
  enabled: true
  model_dir: models/clip
  device: cuda:1
blip_itm:
  required: true
  enabled: true
  model_dir: models/blip
  device: cuda:1
gpt4omini_judge:
  required: true
  model_config: {qwen_config.relative_to(tmp_path)}
  timeout_seconds: 2
""",
    )


def _check(result: dict, name: str) -> dict:
    for item in result["checks"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"missing doctor check: {name}")


def test_doctor_minimal_run_is_ok(tmp_path):
    dataset_dir = make_fixture(tmp_path / "dataset")

    result = run_doctor(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    assert result["overall_status"] == "ok"
    assert _check(result, "dataset_dir")["status"] == "ok"
    assert _check(result, "output_root")["status"] == "ok"
    assert _check(result, "qwen_config")["status"] == "skipped"
    assert _check(result, "stage4_profile")["status"] == "skipped"


def test_doctor_qwen_service_failure_is_reported(tmp_path, monkeypatch):
    dataset_dir = make_fixture(tmp_path / "dataset")
    config = _qwen_config(tmp_path)

    def fake_probe(_config, timeout_seconds):
        assert timeout_seconds == 0.25
        return False, "connection refused"

    monkeypatch.setattr("mmkgc.doctor.probe_qwen_service", fake_probe)

    result = run_doctor(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "outputs"),
            "--model-config",
            str(config),
            "--qwen-timeout-seconds",
            "0.25",
        ]
    )

    assert result["overall_status"] == "error"
    assert _check(result, "qwen_config")["status"] == "ok"
    assert _check(result, "qwen_service") == {
        "name": "qwen_service",
        "status": "error",
        "message": "connection refused",
    }


def test_doctor_stage4_profile_reports_missing_required_resources(tmp_path):
    dataset_dir = make_fixture(tmp_path / "dataset")
    qwen_config = _qwen_config(tmp_path)
    stage4_config = _stage4_profile(tmp_path, qwen_config)

    result = run_doctor(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "outputs"),
            "--missing-mode",
            "text",
            "--stage4-config",
            str(stage4_config),
            "--model-config",
            str(qwen_config),
            "--project-root",
            str(tmp_path),
            "--skip-qwen-service",
        ]
    )

    stage4 = _check(result, "stage4_profile")
    assert result["overall_status"] == "error"
    assert stage4["status"] == "error"
    assert "sentence_bert.model_dir" in stage4["message"]


def test_doctor_main_prints_json_and_returns_nonzero_on_error(tmp_path, capsys):
    missing_dataset = tmp_path / "missing-dataset"

    exit_code = main(
        [
            "--dataset-dir",
            str(missing_dataset),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["overall_status"] == "error"
    assert _check(payload, "dataset_dir")["status"] == "error"
