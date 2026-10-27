from __future__ import annotations

from pathlib import Path

import pytest

from mmkgc.ranking.profiles import (
    load_stage4_profile,
    preflight_stage4_profile,
    required_scorers_for_mode,
    validate_stage4_mode,
)


def test_profile_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "stage4.yaml"
    config_path.write_text("profile_name: test\nunknown_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown top-level key"):
        load_stage4_profile(config_path, project_root=tmp_path)


def test_profile_loader_rejects_removed_calibration_block(tmp_path: Path) -> None:
    config_path = _write_profile(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "calibration:\n  model_deltas:\n    clip: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="calibration"):
        load_stage4_profile(config_path, project_root=tmp_path)


def test_profile_loader_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    config_path = _write_profile(
        tmp_path,
        sentence_bert_extra="  mystery: true\n",
    )

    with pytest.raises(ValueError, match="sentence_bert.mystery"):
        load_stage4_profile(config_path, project_root=tmp_path)


def test_profile_loader_rejects_secret_like_yaml_keys(tmp_path: Path) -> None:
    config_path = _write_profile(tmp_path, judge_extra="  api_key: should-not-be-here\n")

    with pytest.raises(ValueError, match="secret"):
        load_stage4_profile(config_path, project_root=tmp_path)


def test_profile_loader_rejects_paths_outside_project_root(tmp_path: Path) -> None:
    external = tmp_path.parent / "outside-model"
    config_path = _write_profile(
        tmp_path,
        sentence_bert_model_dir=external,
    )

    with pytest.raises(ValueError, match="project root"):
        load_stage4_profile(config_path, project_root=tmp_path)



def test_load_profile_and_mode_required_scorers(tmp_path: Path) -> None:
    config_path = _write_profile(tmp_path)

    profile = load_stage4_profile(config_path, project_root=tmp_path)

    assert profile.profile_name == "full_local_v1"
    assert profile.sentence_bert.device == "cuda:1"
    assert profile.bertscore.enabled is False
    assert profile.gpt4omini_judge.max_retries == 0
    assert profile.gpt4omini_judge.retry_backoff_seconds == 0.0
    assert profile.gpt4omini_judge.max_concurrency == 1
    assert required_scorers_for_mode(profile, "text", has_existing_image=False) == {
        "sentence_bert",
        "gpt4omini_judge",
    }
    assert required_scorers_for_mode(profile, "text", has_existing_image=True) == {
        "sentence_bert",
        "clip",
        "blip_itm",
        "gpt4omini_judge",
    }
    assert required_scorers_for_mode(profile, "image", has_existing_text=False) == {
        "clip",
        "blip_itm",
        "gpt4omini_judge",
    }



def test_load_profile_with_optional_bertscore(tmp_path: Path) -> None:
    config_path = _write_profile(
        tmp_path,
        bertscore_section="""
bertscore:
  required: false
  enabled: true
  model_dir: models/bertscore/roberta-large
  device: cpu
""",
    )

    profile = load_stage4_profile(config_path, project_root=tmp_path)

    assert profile.bertscore.enabled is True
    assert profile.bertscore.required is False
    assert profile.bertscore.model_dir == tmp_path / "models/bertscore/roberta-large"
    assert profile.bertscore.device == "cpu"
    assert "bertscore" not in required_scorers_for_mode(profile, "text")


def test_load_profile_with_gpt4omini_retry_policy(tmp_path: Path) -> None:
    config_path = _write_profile(
        tmp_path,
        judge_extra="  max_retries: 2\n  retry_backoff_seconds: 0.5\n",
    )

    profile = load_stage4_profile(config_path, project_root=tmp_path)

    assert profile.gpt4omini_judge.max_retries == 2
    assert profile.gpt4omini_judge.retry_backoff_seconds == 0.5


def test_load_profile_rejects_invalid_gpt4omini_retry_policy(tmp_path: Path) -> None:
    config_path = _write_profile(tmp_path, judge_extra="  max_retries: -1\n")

    with pytest.raises(ValueError, match="gpt4omini_judge.max_retries"):
        load_stage4_profile(config_path, project_root=tmp_path)


def test_load_profile_with_gpt4omini_concurrency(tmp_path: Path) -> None:
    profile = load_stage4_profile(
        _write_profile(tmp_path, judge_extra="  max_concurrency: 3\n"), project_root=tmp_path,
    )
    assert profile.gpt4omini_judge.max_concurrency == 3


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5", "null", "'3'"])
def test_load_profile_rejects_invalid_gpt4omini_concurrency(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="gpt4omini_judge.max_concurrency"):
        load_stage4_profile(
            _write_profile(tmp_path, judge_extra=f"  max_concurrency: {value}\n"), project_root=tmp_path,
        )


def test_validate_stage4_mode_rejects_unknown_mode() -> None:
    assert validate_stage4_mode("text") == "text"
    assert validate_stage4_mode("image") == "image"

    with pytest.raises(ValueError, match="mode"):
        validate_stage4_mode("video")


def test_preflight_stage4_profile_checks_required_local_paths_only(
    tmp_path: Path,
) -> None:
    config_path = _write_profile(tmp_path)
    profile = load_stage4_profile(config_path, project_root=tmp_path)

    (tmp_path / "models" / "mpnet").mkdir(parents=True)
    (tmp_path / "models" / "clip").mkdir()
    (tmp_path / "models" / "blip").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "mkgw-knowledge-extraction.yaml").write_text(
        "model: qwen-local\n",
        encoding="utf-8",
    )
    assert preflight_stage4_profile(profile, project_root=tmp_path) == []


def test_preflight_stage4_profile_reports_missing_required_paths(tmp_path: Path) -> None:
    config_path = _write_profile(tmp_path)
    profile = load_stage4_profile(config_path, project_root=tmp_path)

    failures = preflight_stage4_profile(profile, project_root=tmp_path)

    assert any("sentence_bert.model_dir" in failure for failure in failures)
    assert any("gpt4omini_judge.model_config" in failure for failure in failures)



def _write_profile(
    tmp_path: Path,
    *,
    sentence_bert_model_dir: Path | str = "models/mpnet",
    sentence_bert_extra: str = "",
    judge_extra: str = "",
    bertscore_section: str = "",
) -> Path:
    config_path = tmp_path / "stage4.yaml"
    config_path.write_text(
        f"""
profile_name: full_local_v1
sentence_bert:
  required: true
  enabled: true
  model_dir: {sentence_bert_model_dir}
  device: cuda:1
{sentence_bert_extra}clip:
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
  model_config: configs/mkgw-knowledge-extraction.yaml
  timeout_seconds: 120
{judge_extra}{bertscore_section}""".lstrip(),
        encoding="utf-8",
    )
    return config_path
