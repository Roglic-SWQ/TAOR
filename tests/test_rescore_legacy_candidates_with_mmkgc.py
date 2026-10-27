from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rescore_legacy_candidates_with_mmkgc.py"
)
spec = importlib.util.spec_from_file_location(
    "rescore_legacy_candidates_with_mmkgc",
    SCRIPT_PATH,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FixedScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = iter(scores)
        self.calls: list[tuple[Path, list[str]]] = []

    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        self.calls.append((image_path, texts))
        return [next(self.scores)]


def _entity() -> dict[str, Any]:
    return {
        "idx": 0,
        "entity_key": "Q1",
        "label": "Example",
        "category": "person",
    }


def _candidate_pairs(tmp_path: Path) -> list[dict[str, Any]]:
    pairs = []
    for candidate_index, prompt in enumerate(("first prompt", "second prompt")):
        image_path = tmp_path / f"id_{candidate_index}.jpg"
        image_path.write_bytes(f"image-{candidate_index}".encode())
        pairs.append(
            {
                "candidate_index": candidate_index,
                "prompt": prompt,
                "image_path": image_path,
            }
        )
    return pairs


def test_scores_each_legacy_candidate_against_its_own_prompt(tmp_path: Path) -> None:
    pairs = _candidate_pairs(tmp_path)
    clip = FixedScorer([25.0, 50.0])
    blip = FixedScorer([75.0, 25.0])

    result = module.score_entity(
        _entity(),
        pairs,
        clip_scorer=clip,
        blip_scorer=blip,
        clip_weight=0.2,
        blip_weight=0.3,
        signature={"score_version": "test"},
    )

    assert clip.calls == [
        (pairs[0]["image_path"], ["first prompt"]),
        (pairs[1]["image_path"], ["second prompt"]),
    ]
    assert blip.calls == clip.calls
    assert result["candidates"][0]["clip_normalized"] == 0.25
    assert result["candidates"][0]["blip_itm_normalized"] == 0.75
    assert result["candidates"][0]["clip_blip_contribution"] == 0.275
    assert result["candidates"][0]["clip_blip_score"] == 0.55
    assert result["candidates"][0]["rank"] == 1
    assert result["summary"]["best_candidate_index"] == 0


def test_loads_id_images_in_prompt_order(tmp_path: Path) -> None:
    entity_dir = tmp_path / "0"
    entity_dir.mkdir()
    (entity_dir / "prompts.json").write_text(
        json.dumps(["prompt zero", "prompt one"]),
        encoding="utf-8",
    )
    (entity_dir / "id_0.jpg").write_bytes(b"zero")
    (entity_dir / "id_1.jpg").write_bytes(b"one")

    pairs = module.load_candidate_pairs(_entity(), tmp_path)

    assert [(row["candidate_index"], row["prompt"]) for row in pairs] == [
        (0, "prompt zero"),
        (1, "prompt one"),
    ]
    assert [Path(row["image_path"]).name for row in pairs] == ["id_0.jpg", "id_1.jpg"]


def test_resume_rejects_changed_inputs(tmp_path: Path) -> None:
    pairs = _candidate_pairs(tmp_path)
    result = module.score_entity(
        _entity(),
        pairs,
        clip_scorer=FixedScorer([20.0, 30.0]),
        blip_scorer=FixedScorer([40.0, 50.0]),
        clip_weight=0.2,
        blip_weight=0.3,
        signature={"score_version": "test"},
    )
    result_path = tmp_path / "Q1.json"
    module.write_json(result_path, result)
    pairs[0]["prompt"] = "changed prompt"

    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        module.load_resumable_result(
            result_path,
            entity=_entity(),
            pairs=pairs,
            signature={"score_version": "test"},
        )


def test_fusion_matches_mmkgc_without_judge_normalization() -> None:
    contribution, normalized = module.fuse_clip_blip(
        0.4,
        0.8,
        clip_weight=0.2,
        blip_weight=0.3,
    )

    assert contribution == pytest.approx(0.32)
    assert normalized == pytest.approx(0.64)
    assert normalized == pytest.approx(0.4 * 0.4 + 0.6 * 0.8)
