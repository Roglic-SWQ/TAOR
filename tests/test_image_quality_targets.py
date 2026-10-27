from __future__ import annotations

from pathlib import Path

import pytest

from mmkgc.evaluation.image_quality_targets import (
    ImageQualityTargetError,
    collect_entity_image_quality_targets,
    collect_image_quality_targets,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(0, 128, 255)).save(path)


def _minimal_artifacts(entity_dir: Path) -> RunEntityArtifacts:
    _write_png(entity_dir / "generated_images" / "image_001.png")
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key="Q42",
        knowledge={
            "entity_profile": {
                "entity_label": "Ada Lovelace",
                "entity_type": "person",
            },
            "relation_facts": [
                {
                    "relation_label": "occupation",
                    "neighbor_label": "mathematician",
                }
            ],
        },
        reasoning={
            "modality_constraints": {
                "image": {
                    "must_include": ["Victorian portrait"],
                    "should_include": ["analytical engine notes"],
                }
            }
        },
        ranking={
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Ada Lovelace was a mathematician and computing pioneer."
                        },
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image_001",
                        "positive_prompt": "portrait of Ada Lovelace as a mathematician",
                        "generated_content": {
                            "type": "image",
                            "value": "fallback prompt should not win",
                            "image_path": "generated_images/image_001.png",
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "pair_id": "pair_001",
                        "text_candidate_id": "text_001",
                        "image_candidate": {"candidate_id": "image_001"},
                    }
                ]
            },
        },
        final={
            "entity": {
                "entity_key": "Q42",
                "label": "Ada Lovelace",
            },
            "missing_mode": "both",
        },
    )


def test_collect_entity_image_quality_targets_extracts_one_minimal_target(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")

    targets = collect_entity_image_quality_targets(artifacts)

    assert len(targets) == 1
    target = targets[0]
    assert target.entity_key == "Q42"
    assert target.label == "Ada Lovelace"
    assert target.image_candidate_id == "image_001"
    assert target.relative_image_path == "generated_images/image_001.png"
    assert target.image_path == (
        artifacts.entity_dir / "generated_images" / "image_001.png"
    ).resolve()


def test_collect_entity_image_quality_targets_includes_rejected_image_candidates(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")
    _write_png(artifacts.entity_dir / "generated_images" / "image_002.png")
    artifacts.ranking["image_ranking"]["rejected_candidates"] = [
        {
            "candidate_id": "image_002",
            "positive_prompt": "rejected visual candidate",
            "generated_content": {
                "type": "image",
                "image_path": "generated_images/image_002.png",
            },
            "hard_gate": {"decision": "reject"},
        }
    ]

    targets = collect_entity_image_quality_targets(artifacts)

    assert [target.image_candidate_id for target in targets] == [
        "image_001",
        "image_002",
    ]


def test_collect_entity_image_quality_targets_includes_unranked_generated_images(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")
    _write_png(artifacts.entity_dir / "generated_images" / "image_002.png")
    (artifacts.entity_dir / "generated_images" / "image_002.json").write_text(
        '{"candidate_id":"image_002","positive_prompt":"orphan generated image"}',
        encoding="utf-8",
    )

    targets = collect_entity_image_quality_targets(artifacts)

    assert [target.image_candidate_id for target in targets] == [
        "image_001",
        "image_002",
    ]
    assert targets[1].prompt_text == "orphan generated image"


def test_collect_entity_image_quality_targets_uses_positive_prompt_for_prompt_text(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")

    target = collect_entity_image_quality_targets(artifacts)[0]

    assert target.prompt_text == "portrait of Ada Lovelace as a mathematician"


def test_collect_entity_image_quality_targets_finds_paired_generated_text(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")

    target = collect_entity_image_quality_targets(artifacts)[0]

    assert (
        target.candidate_text
        == "Ada Lovelace was a mathematician and computing pioneer."
    )


def test_collect_entity_image_quality_targets_builds_evidence_summary_from_entity_and_constraints(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")

    target = collect_entity_image_quality_targets(artifacts)[0]

    assert "Entity: Ada Lovelace" in target.evidence_summary_text
    assert "Type: person" in target.evidence_summary_text
    assert "occupation: mathematician" in target.evidence_summary_text
    assert "Visual requirements" in target.evidence_summary_text
    assert "Victorian portrait" in target.evidence_summary_text
    assert "analytical engine notes" in target.evidence_summary_text


def test_collect_entity_image_quality_targets_rejects_image_path_escape(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")
    _write_png(tmp_path / "outside.png")
    image_candidate = artifacts.ranking["image_ranking"]["ranked_candidates"][0]
    image_candidate["generated_content"]["image_path"] = "../outside.png"

    with pytest.raises(ImageQualityTargetError, match="image path must stay under"):
        collect_entity_image_quality_targets(artifacts)


def test_collect_image_quality_targets_alias_preserves_existing_api(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q42")

    alias_target = collect_image_quality_targets(artifacts)[0]
    public_target = collect_entity_image_quality_targets(artifacts)[0]

    assert alias_target == public_target
