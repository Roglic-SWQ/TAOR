# MKG-W Qwen-VL Judge + BLIP/ITM Image Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict Stage 8 image evaluation on the existing Stage 7 run, using Qwen2-VL as a VLM judge and local BLIP/ITM as an image-text matching evaluator.

**Architecture:** Extend the existing `mmkgc.evaluate` post-run evaluation path with two independent optional providers: `qwen-vl-judge` and `blip-itm-local`. Reuse a shared image-quality target collector so CLIP, BLIP, and Qwen-VL judge evaluate the same image, prompt, generated text, and KG evidence text. Write separate per-provider summaries plus a comparison summary under the existing run `evaluation/` directory.

**Tech Stack:** Python 3.11, existing `mmkgc.evaluate`, local vLLM OpenAI-compatible Qwen2-VL service, `requests`, `transformers`, `torch`, `Pillow`, local `blip-itm-large-coco`, pytest, compileall.

---

## Controller Rules For This Plan

- 主 Codex 会话作为 controller。
- 每个 task 派一个 implementer subagent。
- 每个 implementer 必须收到本 task 的写入范围、验证命令、禁止 commit 指令。
- 每个任务完成后派两个 reviewer：
  - spec reviewer：检查是否符合 `docs/superpowers/specs/7 mkgw-qwen-vl-blip-image-evaluation-design.zh.md` 和用户确认方案。
  - quality reviewer：检查代码质量、测试质量、职责边界、路径/模板/打包风险。
- reviewer 发现问题时，必须回给 implementer 修，不跳过。
- 两个 reviewer 都 approve 后才标记 task 完成。
- 最后派 final reviewer 审完整变更。
- 最后运行完整验证命令并汇报证据。
- 工作区已有用户未提交改动和前阶段改动；不要 revert 任何非本任务改动。
- 不自动 commit。

## Existing Dirty Workspace Notes

执行前必须保留以下既有状态：

```text
Stage 6 image quality code/tests are already dirty.
Stage 7 spec/plan/report docs are already dirty or untracked.
User renamed older plan/spec docs into numbered filenames.
Qwen/
SDXL/
CLIP/
datasets/
outputs/
```

本计划新增或修改文件时，要和这些变动共存，不恢复旧文件名，不清理用户工作。

## Files And Responsibilities

- Modify `.gitignore`
  - Add `blip-itm-large-coco/`.
- Create `mmkgc/evaluation/image_quality_targets.py`
  - Shared extraction of image candidate path, prompt text, generated text, and KG evidence summary.
- Modify `mmkgc/evaluation/image_quality.py`
  - Reuse shared target collector for CLIP without changing CLIP output schema.
- Create `mmkgc/evaluation/blip_itm.py`
  - Local BLIP/ITM model-dir validation, scorer, entity evaluator, and fake scorer for tests.
- Create `mmkgc/evaluation/qwen_vl_judge.py`
  - Qwen-VL judge config, prompt construction, image data URL encoding, HTTP client, JSON parsing, score validation, and entity evaluator.
- Create `mmkgc/evaluation/multi_image_quality_writer.py`
  - Write Qwen-VL summary/CSV/entity JSON, BLIP/ITM summary/CSV/entity JSON, and comparison summary.
- Modify `mmkgc/evaluate.py`
  - Add CLI parameters, initialize optional providers, run them after offline evaluation, and include counts in stdout JSON.
- Test `tests/test_image_quality_targets.py`
  - Shared target extraction and path safety.
- Test `tests/test_blip_itm.py`
  - BLIP path validation, fake scorer, scoring shape, score ranges.
- Test `tests/test_qwen_vl_judge.py`
  - Prompt, data URL, response parsing, error handling, fake judge.
- Test `tests/test_multi_image_quality_writer.py`
  - Writer output files, CSV rows, overwrite safety, comparison summary.
- Modify `tests/test_evaluate_cli.py`
  - CLI help/default/fake provider integration/fail-fast tests.
- Create `docs/evaluation/2026-07-09-qwen-vl-blip-image-evaluation-validation.zh.md`
  - Stage 8 validation report after real smoke succeeds.
- Modify `docs/project-progress-summary.zh.md`
  - Add Stage 8 only after real smoke succeeds.
- Modify `README.md` and `README.zh.md`
  - Briefly document Stage 8 evaluator usage after real smoke succeeds.

## Task 1: Shared Image Quality Target Extraction

**Files:**
- Create: `mmkgc/evaluation/image_quality_targets.py`
- Modify: `mmkgc/evaluation/image_quality.py`
- Test: `tests/test_image_quality_targets.py`
- Test: `tests/test_image_quality.py`

**Goal:** Make CLIP, BLIP/ITM, and Qwen-VL Judge consume the same image-level targets.

**Implementer scope:**
- Extract existing target-building logic from `image_quality.py` without changing CLIP output field names.
- Do not touch BLIP/Qwen code in this task.
- Do not commit.

- [ ] **Step 1: Add failing target extraction tests**

Create `tests/test_image_quality_targets.py` with these tests:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mmkgc.evaluation.image_quality import ClipImageQualityError
from mmkgc.evaluation.image_quality_targets import collect_entity_image_quality_targets
from mmkgc.evaluation.loader import iter_entity_artifact_results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_with_one_image(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    entity_dir = run_dir / "entities" / "Q1"
    image_dir = entity_dir / "generated_images"
    image_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_dir / "image_001.png")
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {
            "entity_profile": {
                "entity_key": "Q1",
                "entity_label": "Ada Lovelace",
                "entity_type": "person",
            },
            "relation_facts": [
                {"relation_label": "occupation", "neighbor_label": "mathematician"}
            ],
        },
    )
    _write_json(
        entity_dir / "2_graph_constrained_reasoning.json",
        {
            "modality_constraints": {
                "image": {"must_include": ["portrait", "mathematician"]}
            }
        },
    )
    _write_json(
        entity_dir / "5_kg_aware_pair_ranking.json",
        {
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Ada Lovelace was a mathematician."
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
                            "image_path": "generated_images/image_001.png"
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "text_candidate_id": "text_001",
                        "image_candidate_id": "image_001",
                    }
                ]
            },
        },
    )
    _write_json(
        entity_dir / "final.json",
        {
            "entity": {
                "entity_key": "Q1",
                "label": "Ada Lovelace",
            },
            "missing_mode": "both",
        },
    )
    return run_dir


def _artifacts(run_dir: Path):
    result = next(iter_entity_artifact_results(run_dir))
    assert result.artifacts is not None
    return result.artifacts


def test_collect_entity_image_quality_targets_extracts_shared_text_targets(tmp_path: Path) -> None:
    artifacts = _artifacts(_run_with_one_image(tmp_path))

    targets = collect_entity_image_quality_targets(artifacts)

    assert len(targets) == 1
    target = targets[0]
    assert target.entity_key == "Q1"
    assert target.label == "Ada Lovelace"
    assert target.image_candidate_id == "image_001"
    assert target.relative_image_path == "generated_images/image_001.png"
    assert target.prompt_text == "portrait of Ada Lovelace as a mathematician"
    assert target.candidate_text == "Ada Lovelace was a mathematician."
    assert "occupation: mathematician" in target.evidence_summary_text
    assert "Visual requirements: portrait, mathematician" in target.evidence_summary_text


def test_collect_entity_image_quality_targets_rejects_image_path_escape(tmp_path: Path) -> None:
    run_dir = _run_with_one_image(tmp_path)
    ranking_path = run_dir / "entities" / "Q1" / "5_kg_aware_pair_ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["image_ranking"]["ranked_candidates"][0]["generated_content"][
        "image_path"
    ] = "../outside.png"
    ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
    artifacts = _artifacts(run_dir)

    with pytest.raises(ClipImageQualityError, match="image path must stay under"):
        collect_entity_image_quality_targets(artifacts)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality_targets.py -v -p no:cacheprovider
```

Expected:

```text
ModuleNotFoundError: No module named 'mmkgc.evaluation.image_quality_targets'
```

- [ ] **Step 3: Implement `image_quality_targets.py`**

Create `mmkgc/evaluation/image_quality_targets.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmkgc.evaluation.loader import RunEntityArtifacts


@dataclass(frozen=True)
class ImageQualityTarget:
    entity_key: str
    label: str
    missing_mode: str
    image_candidate_id: str
    image_path: Path
    relative_image_path: str
    prompt_text: str
    candidate_text: str
    evidence_summary_text: str


class ImageQualityTargetError(RuntimeError):
    """Raised when an image quality target cannot be constructed safely."""


def collect_entity_image_quality_targets(
    artifacts: RunEntityArtifacts,
) -> list[ImageQualityTarget]:
    image_candidates = _as_list(
        _as_dict(artifacts.ranking.get("image_ranking")).get("ranked_candidates")
    )
    final_entity = _as_dict(artifacts.final.get("entity"))
    profile = _as_dict(artifacts.knowledge.get("entity_profile"))
    entity_key = _first_text(final_entity.get("entity_key"), artifacts.entity_key)
    label = _first_text(
        final_entity.get("label"),
        profile.get("entity_label"),
        profile.get("label"),
    )
    missing_mode = _first_text(artifacts.final.get("missing_mode"))
    evidence_summary_text = build_entity_evidence_summary(artifacts)

    targets: list[ImageQualityTarget] = []
    for image_candidate in image_candidates:
        if not isinstance(image_candidate, dict):
            continue
        image_candidate_id = _candidate_id(image_candidate)
        image_path = _resolve_entity_image_path(artifacts, image_candidate)
        targets.append(
            ImageQualityTarget(
                entity_key=entity_key,
                label=label,
                missing_mode=missing_mode,
                image_candidate_id=image_candidate_id,
                image_path=image_path,
                relative_image_path=_relative_path(image_path, artifacts.entity_dir),
                prompt_text=_prompt_text(image_candidate),
                candidate_text=_candidate_text_for_image(artifacts, image_candidate_id),
                evidence_summary_text=evidence_summary_text,
            )
        )
    return targets
```

Move or copy the helper functions currently used by `image_quality.py` into this module:

```text
build_entity_evidence_summary
_as_dict
_as_list
_first_text
_clean_text
_text_list
_dedupe
_relation_evidence_snippets
_relation_fact_values
_image_constraint_terms
_constraint_terms
_constraint_dict_terms
_constraint_label_values
_constraint_evidence_triple_terms
_candidate_id
_resolve_entity_image_path
_relative_path
_prompt_text
_candidate_text_for_image
_ranked_text_candidates_by_id
_text_from_candidate
```

To preserve existing exception behavior, make `image_quality.py` import and alias:

```python
from mmkgc.evaluation.image_quality_targets import ImageQualityTargetError


class ClipImageQualityError(ImageQualityTargetError):
    """Raised when CLIP image quality evaluation cannot run safely."""
```

Then update `image_quality_targets.py` to raise `ImageQualityTargetError`; in `image_quality.py`, catch it around `collect_entity_image_quality_targets()` and re-raise `ClipImageQualityError(str(exc))` so existing CLIP tests still match `ClipImageQualityError`.

- [ ] **Step 4: Update CLIP evaluator to use the shared targets**

In `mmkgc/evaluation/image_quality.py`, update `evaluate_entity_image_quality()` to:

```python
targets = collect_entity_image_quality_targets(artifacts)
for target in targets:
    target_pairs = [
        ("clip_prompt_similarity", target.prompt_text),
        ("clip_text_similarity", target.candidate_text),
        ("clip_evidence_similarity", target.evidence_summary_text),
    ]
    available_targets = [
        (field, text) for field, text in target_pairs if _clean_text(text)
    ]
    texts = [text for _, text in available_targets]
    scores = scorer.score_image_texts(target.image_path, texts)
    score_by_field = {
        "clip_prompt_similarity": None,
        "clip_text_similarity": None,
        "clip_evidence_similarity": None,
    }
    for (field, _text), score in zip(available_targets, scores):
        score_by_field[field] = round(float(score), 6)
    image_scores.append({
        "entity_key": target.entity_key,
        "image_candidate_id": target.image_candidate_id,
        "image_path": target.relative_image_path,
        "clip_model_name": clip_model_name,
        "clip_weight_path": str(clip_weight_path),
        "clip_prompt_similarity": score_by_field["clip_prompt_similarity"],
        "clip_text_similarity": score_by_field["clip_text_similarity"],
        "clip_evidence_similarity": score_by_field["clip_evidence_similarity"],
        "clip_overall_image_quality_score": overall,
        "text_targets": {
            "prompt_text": _clean_text(target.prompt_text),
            "candidate_text": _clean_text(target.candidate_text),
            "evidence_summary_text": _clean_text(target.evidence_summary_text),
        },
    })
```

Keep existing field names and writer contract unchanged.

- [ ] **Step 5: Run target and CLIP tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality_targets.py tests/test_image_quality.py tests/test_image_quality_writer.py -v -p no:cacheprovider
```

Expected:

```text
all selected tests pass
```

## Task 2: BLIP/ITM Local Evaluator

**Files:**
- Create: `mmkgc/evaluation/blip_itm.py`
- Test: `tests/test_blip_itm.py`
- Modify: `.gitignore`

**Goal:** Add local BLIP/ITM image-text scoring on prompt, generated text, and KG evidence targets.

**Implementer scope:**
- Implement provider internals only; do not wire CLI in this task.
- Add `blip-itm-large-coco/` to `.gitignore`.
- Do not commit.

- [ ] **Step 1: Add `.gitignore` rule**

Add this line near model/data directories in `.gitignore`:

```text
blip-itm-large-coco/
```

- [ ] **Step 2: Add failing BLIP tests**

Create `tests/test_blip_itm.py` with tests covering:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mmkgc.evaluation.blip_itm import (
    BLIPITMError,
    BlipITMConfig,
    BlipITMLocalScorer,
    FixedBLIPITMScorer,
    evaluate_entity_blip_itm,
    validate_blip_model_dir,
)
from mmkgc.evaluation.loader import iter_entity_artifact_results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def minimal_image_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    entity_dir = run_dir / "entities" / "Q1"
    image_dir = entity_dir / "generated_images"
    image_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_dir / "image_001.png")
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {
            "entity_profile": {
                "entity_key": "Q1",
                "entity_label": "Ada Lovelace",
                "entity_type": "person",
            },
            "relation_facts": [
                {"relation_label": "occupation", "neighbor_label": "mathematician"}
            ],
        },
    )
    _write_json(
        entity_dir / "2_graph_constrained_reasoning.json",
        {
            "modality_constraints": {
                "image": {"must_include": ["portrait", "mathematician"]}
            }
        },
    )
    _write_json(
        entity_dir / "5_kg_aware_pair_ranking.json",
        {
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Ada Lovelace was a mathematician."
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
                            "image_path": "generated_images/image_001.png"
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "text_candidate_id": "text_001",
                        "image_candidate_id": "image_001",
                    }
                ]
            },
        },
    )
    _write_json(entity_dir / "final.json", {"entity": {"entity_key": "Q1", "label": "Ada Lovelace"}})
    return run_dir


def test_validate_blip_model_dir_accepts_project_local_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "blip-itm-large-coco"
    for filename in (
        "config.json",
        "pytorch_model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (model_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        (model_dir / filename).write_text("{}", encoding="utf-8")

    resolved = validate_blip_model_dir(model_dir, project_root=tmp_path)

    assert resolved == model_dir.resolve()


def test_validate_blip_model_dir_rejects_external_path(tmp_path: Path) -> None:
    external = tmp_path.parent / "external-blip"
    external.mkdir(exist_ok=True)

    with pytest.raises(BLIPITMError, match="blip_model_dir must be under"):
        validate_blip_model_dir(external, project_root=tmp_path)


def test_fixed_blip_itm_scorer_returns_one_score_per_text(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image_path)
    scorer = FixedBLIPITMScorer([[10, 20, 30]])

    scores = scorer.score_image_texts(image_path, ["prompt", "text", "evidence"])

    assert scores == [10.0, 20.0, 30.0]
    assert scorer.calls[0]["texts"] == ["prompt", "text", "evidence"]


def test_evaluate_entity_blip_itm_scores_three_targets(minimal_image_run) -> None:
    artifacts = next(iter_entity_artifact_results(minimal_image_run)).artifacts
    assert artifacts is not None

    result = evaluate_entity_blip_itm(
        artifacts,
        scorer=FixedBLIPITMScorer([[11, 22, 33]]),
        model_dir=Path("blip-itm-large-coco"),
    )

    assert result["image_count"] == 1
    image_score = result["image_scores"][0]
    assert image_score["blip_prompt_itm_score"] == 11.0
    assert image_score["blip_text_itm_score"] == 22.0
    assert image_score["blip_evidence_itm_score"] == 33.0
    assert image_score["blip_overall_itm_score"] == 22.0
    assert result["averages"]["blip_overall_itm_score"] == 22.0
```

- [ ] **Step 3: Run BLIP tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_blip_itm.py -v -p no:cacheprovider
```

Expected:

```text
ModuleNotFoundError: No module named 'mmkgc.evaluation.blip_itm'
```

- [ ] **Step 4: Implement BLIP/ITM module**

Create `mmkgc/evaluation/blip_itm.py` with these public objects:

```python
BLIP_ITM_SCORE_FIELDS = (
    "blip_prompt_itm_score",
    "blip_text_itm_score",
    "blip_evidence_itm_score",
    "blip_overall_itm_score",
)

class BLIPITMError(RuntimeError):
    """Raised when local BLIP/ITM evaluation cannot run safely."""

@dataclass
class BlipITMConfig:
    model_dir: Path = Path("blip-itm-large-coco")
    device: str = "cuda"
    project_root: Path = MMKGC_ROOT

class FixedBLIPITMScorer:
    """Test scorer that returns configured scores and records calls."""

class BlipITMLocalScorer:
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        """Return one 0-100 ITM score for each text target."""

def validate_blip_model_dir(
    model_dir: str | Path,
    *,
    project_root: str | Path = MMKGC_ROOT,
) -> Path:
    """Return a safe project-local BLIP model directory."""

def evaluate_entity_blip_itm(
    artifacts: RunEntityArtifacts,
    *,
    scorer: ImageTextSimilarityScorer,
    model_dir: Path,
) -> dict[str, Any]:
    """Score all generated image candidates for one entity."""
```

Model scoring implementation:

```python
processor = BlipProcessor.from_pretrained(str(model_dir), local_files_only=True)
model = BlipForImageTextRetrieval.from_pretrained(str(model_dir), local_files_only=True)
model.to(device)
model.eval()
```

For each text target:

```python
inputs = processor(images=image, text=text, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    outputs = model(**inputs)
```

Use `outputs.itm_score` when present; if the tensor has two logits, apply softmax and take the positive/matching class probability:

```python
probability = torch.softmax(outputs.itm_score, dim=-1)[0, 1].item()
score = round(max(0.0, min(1.0, probability)) * 100.0, 6)
```

If `outputs.itm_score` is unavailable, raise `BLIPITMError("BLIP output does not contain itm_score")`.

- [ ] **Step 5: Run BLIP tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_blip_itm.py -v -p no:cacheprovider
```

Expected:

```text
all tests pass
```

## Task 3: Qwen-VL Judge Provider

**Files:**
- Create: `mmkgc/evaluation/qwen_vl_judge.py`
- Test: `tests/test_qwen_vl_judge.py`

**Goal:** Add a strict image-level Qwen2-VL judge provider over the local OpenAI-compatible service.

**Implementer scope:**
- Implement provider internals only; do not wire CLI in this task.
- Use base64 data URLs for local images.
- Do not commit.

- [ ] **Step 1: Add failing Qwen-VL judge tests**

Create `tests/test_qwen_vl_judge.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mmkgc.evaluation.loader import iter_entity_artifact_results
from mmkgc.evaluation.qwen_vl_judge import (
    FixedQwenVLJudge,
    QwenVLJudgeError,
    parse_qwen_vl_judge_response,
    image_to_data_url,
    build_qwen_vl_judge_prompt,
    evaluate_entity_qwen_vl_judge,
)
from mmkgc.evaluation.image_quality_targets import collect_entity_image_quality_targets


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def minimal_image_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    entity_dir = run_dir / "entities" / "Q1"
    image_dir = entity_dir / "generated_images"
    image_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_dir / "image_001.png")
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {
            "entity_profile": {
                "entity_key": "Q1",
                "entity_label": "Ada Lovelace",
                "entity_type": "person",
            },
            "relation_facts": [
                {"relation_label": "occupation", "neighbor_label": "mathematician"}
            ],
        },
    )
    _write_json(
        entity_dir / "2_graph_constrained_reasoning.json",
        {
            "modality_constraints": {
                "image": {"must_include": ["portrait", "mathematician"]}
            }
        },
    )
    _write_json(
        entity_dir / "5_kg_aware_pair_ranking.json",
        {
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Ada Lovelace was a mathematician."
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
                            "image_path": "generated_images/image_001.png"
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "text_candidate_id": "text_001",
                        "image_candidate_id": "image_001",
                    }
                ]
            },
        },
    )
    _write_json(entity_dir / "final.json", {"entity": {"entity_key": "Q1", "label": "Ada Lovelace"}})
    return run_dir


def test_image_to_data_url_encodes_png(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)

    data_url = image_to_data_url(image_path)

    assert data_url.startswith("data:image/png;base64,")


def test_build_qwen_vl_judge_prompt_contains_required_context(minimal_image_run) -> None:
    artifacts = next(iter_entity_artifact_results(minimal_image_run)).artifacts
    assert artifacts is not None
    target = collect_entity_image_quality_targets(artifacts)[0]

    prompt = build_qwen_vl_judge_prompt(target)

    assert "entity_match_score" in prompt
    assert "prompt_alignment_score" in prompt
    assert "kg_evidence_score" in prompt
    assert "artifact_score" in prompt
    assert target.prompt_text in prompt
    assert target.candidate_text in prompt
    assert target.evidence_summary_text in prompt


def test_parse_qwen_vl_judge_response_accepts_json_object() -> None:
    parsed = parse_qwen_vl_judge_response(
        '{"entity_match_score": 80, "prompt_alignment_score": 70, '
        '"kg_evidence_score": 60, "artifact_score": 90, '
        '"overall_judge_score": 75, "rationale": "图像基本符合。"}'
    )

    assert parsed["overall_judge_score"] == 75.0
    assert parsed["rationale"] == "图像基本符合。"


def test_parse_qwen_vl_judge_response_rejects_missing_rationale() -> None:
    with pytest.raises(QwenVLJudgeError, match="missing required field"):
        parse_qwen_vl_judge_response(
            '{"entity_match_score": 80, "prompt_alignment_score": 70, '
            '"kg_evidence_score": 60, "artifact_score": 90, '
            '"overall_judge_score": 75}'
        )


def test_parse_qwen_vl_judge_response_rejects_out_of_range_score() -> None:
    with pytest.raises(QwenVLJudgeError, match="must be between 0 and 100"):
        parse_qwen_vl_judge_response(
            '{"entity_match_score": 101, "prompt_alignment_score": 70, '
            '"kg_evidence_score": 60, "artifact_score": 90, '
            '"overall_judge_score": 75, "rationale": "bad"}'
        )


def test_evaluate_entity_qwen_vl_judge_scores_each_image(minimal_image_run) -> None:
    artifacts = next(iter_entity_artifact_results(minimal_image_run)).artifacts
    assert artifacts is not None
    judge = FixedQwenVLJudge(
        [
            {
                "entity_match_score": 80,
                "prompt_alignment_score": 70,
                "kg_evidence_score": 60,
                "artifact_score": 90,
                "overall_judge_score": 75,
                "rationale": "图像基本符合。",
            }
        ]
    )

    result = evaluate_entity_qwen_vl_judge(
        artifacts,
        judge=judge,
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        base_url="http://localhost:12345/v1",
    )

    assert result["image_count"] == 1
    image_score = result["image_scores"][0]
    assert image_score["status"] == "judged"
    assert image_score["fallback_used"] is False
    assert image_score["overall_judge_score"] == 75.0
    assert result["averages"]["overall_judge_score"] == 75.0
```

- [ ] **Step 2: Run Qwen-VL tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_qwen_vl_judge.py -v -p no:cacheprovider
```

Expected:

```text
ModuleNotFoundError: No module named 'mmkgc.evaluation.qwen_vl_judge'
```

- [ ] **Step 3: Implement Qwen-VL judge module**

Create `mmkgc/evaluation/qwen_vl_judge.py` with:

```python
QWEN_VL_JUDGE_SCORE_FIELDS = (
    "entity_match_score",
    "prompt_alignment_score",
    "kg_evidence_score",
    "artifact_score",
    "overall_judge_score",
)

class QwenVLJudgeError(RuntimeError):
    """Raised when Qwen-VL judge configuration, HTTP, or parsing fails."""

@dataclass
class QwenVLJudgeConfig:
    model_name: str
    base_url: str
    api_key: str
    max_tokens: int = 512
    temperature: float = 0.0
    timeout_seconds: int = 120

class FixedQwenVLJudge:
    """Test judge that returns configured judge JSON objects and records calls."""

class QwenVLJudgeClient:
    """HTTP client for the OpenAI-compatible Qwen2-VL service."""

def image_to_data_url(image_path: Path) -> str:
    """Encode a local PNG/JPEG as a base64 image data URL."""

def build_qwen_vl_judge_prompt(target: ImageQualityTarget) -> str:
    """Build the strict JSON-only Chinese judge prompt."""

def parse_qwen_vl_judge_response(response_text: str) -> dict[str, Any]:
    """Parse and validate the Qwen-VL judge JSON response."""

def evaluate_entity_qwen_vl_judge(
    artifacts: RunEntityArtifacts,
    *,
    judge: QwenVLJudgeProtocol,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    """Judge all generated image candidates for one entity."""
```

`QwenVLJudgeClient.judge_image()` must call:

```text
POST {base_url.rstrip("/")}/chat/completions
Authorization: Bearer token-abc123
```

Message content shape:

```json
[
  {"type": "text", "text": "请根据图像、实体、prompt、generated text 和 KG evidence 输出评分 JSON。"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
]
```

The prompt must require JSON only:

```text
请只输出一个 JSON object，不要使用 Markdown 代码块。
字段必须是 entity_match_score, prompt_alignment_score, kg_evidence_score,
artifact_score, overall_judge_score, rationale。
所有分数为 0-100。
rationale 使用简短中文。
```

`parse_qwen_vl_judge_response()` must strip Markdown fences if the model returns them, then parse JSON and validate required fields.

- [ ] **Step 4: Run Qwen-VL tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_qwen_vl_judge.py -v -p no:cacheprovider
```

Expected:

```text
all tests pass
```

## Task 4: Writers And Comparison Summary

**Files:**
- Create: `mmkgc/evaluation/multi_image_quality_writer.py`
- Test: `tests/test_multi_image_quality_writer.py`

**Goal:** Persist Qwen-VL Judge, BLIP/ITM, and comparison summaries safely under `evaluation/`.

**Implementer scope:**
- Writer only; do not wire CLI in this task.
- Preserve existing CLIP writer behavior.
- Do not commit.

- [ ] **Step 1: Add failing writer tests**

Create `tests/test_multi_image_quality_writer.py` with tests for:

```python
def test_write_qwen_vl_judge_outputs_writes_summary_csv_and_entity_files(tmp_path):
    summary = write_qwen_vl_judge_outputs(tmp_path / "evaluation", tmp_path / "run", [qwen_result])
    assert summary["evaluated_entities"] == 1
    assert summary["evaluated_images"] == 1
    assert (tmp_path / "evaluation" / "qwen_vl_judge_summary.json").is_file()
    assert (tmp_path / "evaluation" / "qwen_vl_judge_entity_scores.csv").is_file()

def test_write_blip_itm_outputs_writes_summary_csv_and_entity_files(tmp_path):
    summary = write_blip_itm_outputs(tmp_path / "evaluation", tmp_path / "run", [blip_result])
    assert summary["evaluated_entities"] == 1
    assert summary["evaluated_images"] == 1
    assert (tmp_path / "evaluation" / "blip_itm_summary.json").is_file()
    assert (tmp_path / "evaluation" / "blip_itm_entity_scores.csv").is_file()

def test_write_image_quality_model_comparison_summary_reads_all_three_summaries(tmp_path):
    summary = write_image_quality_model_comparison_summary(
        tmp_path / "evaluation",
        clip_summary=clip_summary,
        blip_summary=blip_summary,
        qwen_vl_summary=qwen_summary,
        overwrite=True,
    )
    assert summary["evaluated_images"] == 1
    assert summary["clip"]["clip_overall_image_quality_score"] == 44.0
    assert summary["blip_itm"]["blip_overall_itm_score"] == 22.0
    assert summary["qwen_vl_judge"]["overall_judge_score"] == 75.0

def test_writer_refuses_existing_owned_outputs_without_overwrite(tmp_path):
    output_dir = tmp_path / "evaluation"
    write_qwen_vl_judge_outputs(output_dir, tmp_path / "run", [qwen_result])
    with pytest.raises(FileExistsError):
        write_qwen_vl_judge_outputs(output_dir, tmp_path / "run", [qwen_result])

def test_writer_overwrite_only_removes_owned_stage8_outputs(tmp_path):
    output_dir = tmp_path / "evaluation"
    clip_file = output_dir / "clip_image_quality_summary.json"
    clip_file.parent.mkdir(parents=True)
    clip_file.write_text("{}", encoding="utf-8")
    write_qwen_vl_judge_outputs(output_dir, tmp_path / "run", [qwen_result], overwrite=True)
    assert clip_file.is_file()
```

Use concrete fake result dictionaries:

```python
qwen_result = {
    "entity": {"entity_key": "Q1", "label": "Ada", "missing_mode": "both"},
    "image_count": 1,
    "model_name": "Qwen/Qwen2-VL-7B-Instruct",
    "base_url": "http://localhost:12345/v1",
    "averages": {
        "entity_match_score": 80.0,
        "prompt_alignment_score": 70.0,
        "kg_evidence_score": 60.0,
        "artifact_score": 90.0,
        "overall_judge_score": 75.0,
    },
    "best_image_candidate_id": "image_001",
    "worst_image_candidate_id": "image_001",
    "image_scores": [
        {
            "entity_key": "Q1",
            "image_candidate_id": "image_001",
            "image_path": "generated_images/image_001.png",
            "status": "judged",
            "fallback_used": False,
            "entity_match_score": 80.0,
            "prompt_alignment_score": 70.0,
            "kg_evidence_score": 60.0,
            "artifact_score": 90.0,
            "overall_judge_score": 75.0,
            "rationale": "图像基本符合。",
        }
    ],
}
```

and:

```python
blip_result = {
    "entity": {"entity_key": "Q1", "label": "Ada", "missing_mode": "both"},
    "image_count": 1,
    "model_dir": "blip-itm-large-coco",
    "averages": {
        "blip_prompt_itm_score": 11.0,
        "blip_text_itm_score": 22.0,
        "blip_evidence_itm_score": 33.0,
        "blip_overall_itm_score": 22.0,
    },
    "best_image_candidate_id": "image_001",
    "worst_image_candidate_id": "image_001",
    "image_scores": [
        {
            "entity_key": "Q1",
            "image_candidate_id": "image_001",
            "image_path": "generated_images/image_001.png",
            "status": "scored",
            "fallback_used": False,
            "blip_prompt_itm_score": 11.0,
            "blip_text_itm_score": 22.0,
            "blip_evidence_itm_score": 33.0,
            "blip_overall_itm_score": 22.0,
        }
    ],
}
```

- [ ] **Step 2: Run writer tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_multi_image_quality_writer.py -v -p no:cacheprovider
```

Expected:

```text
ModuleNotFoundError: No module named 'mmkgc.evaluation.multi_image_quality_writer'
```

- [ ] **Step 3: Implement writer module**

Create `mmkgc/evaluation/multi_image_quality_writer.py` with functions:

```python
def write_qwen_vl_judge_outputs(
    output_dir: str | Path,
    run_dir: str | Path,
    entity_judge_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write Qwen-VL judge summary, CSV, and per-entity JSON."""

def write_blip_itm_outputs(
    output_dir: str | Path,
    run_dir: str | Path,
    entity_itm_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write BLIP/ITM summary, CSV, and per-entity JSON."""

def write_image_quality_model_comparison_summary(
    output_dir: str | Path,
    *,
    clip_summary: dict[str, Any] | None,
    blip_summary: dict[str, Any] | None,
    qwen_vl_summary: dict[str, Any] | None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one comparison summary from available CLIP, BLIP, and Qwen-VL summaries."""
```

Owned output names:

```text
qwen_vl_judge_summary.json
qwen_vl_judge_entity_scores.csv
blip_itm_summary.json
blip_itm_entity_scores.csv
image_quality_model_comparison_summary.json
entities/Q1__example/qwen_vl_judge.json
entities/Q1__example/blip_itm.json
```

Use the same entity directory naming scheme as `image_quality_writer.py`:

```python
slug = sanitized entity key
digest = sha256(entity_key)[:8]
return f"{slug}__{digest}"
```

Do not delete CLIP outputs, offline evaluation outputs, or generated inputs when `overwrite=True`; only remove Stage 8 owned outputs.

- [ ] **Step 4: Run writer tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_multi_image_quality_writer.py -v -p no:cacheprovider
```

Expected:

```text
all tests pass
```

## Task 5: CLI Integration In `mmkgc.evaluate`

**Files:**
- Modify: `mmkgc/evaluate.py`
- Modify: `tests/test_evaluate_cli.py`

**Goal:** Wire BLIP/ITM and Qwen-VL Judge into the existing evaluation CLI.

**Implementer scope:**
- Add CLI flags and fake-provider tests.
- Do not run real Qwen or real BLIP in unit tests.
- Do not commit.

- [ ] **Step 1: Add failing CLI tests**

Modify `tests/test_evaluate_cli.py`:

```python
def test_cli_help_includes_stage8_image_evaluator_options() -> None:
    help_text = evaluate_cli._build_parser().format_help()

    assert "--blip-itm-evaluator" in help_text
    assert "--blip-model-dir" in help_text
    assert "--blip-device" in help_text
    assert "--vlm-judge-evaluator" in help_text
    assert "--vlm-judge-model-config" in help_text
    assert "--vlm-judge-timeout-seconds" in help_text


def test_cli_default_does_not_run_stage8_evaluators(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["blip_itm_evaluator"] == "none"
    assert stdout_summary["blip_evaluated_entities"] == 0
    assert stdout_summary["blip_evaluated_images"] == 0
    assert stdout_summary["vlm_judge_evaluator"] == "none"
    assert stdout_summary["vlm_judged_entities"] == 0
    assert stdout_summary["vlm_judged_images"] == 0
```

Add fake provider integration test:

```python
def test_cli_writes_stage8_outputs_with_fake_evaluators(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"

    monkeypatch.setattr(evaluate_cli, "validate_blip_model_dir", lambda model_dir: tmp_path / "blip-itm-large-coco")
    monkeypatch.setattr(evaluate_cli, "BlipITMLocalScorer", lambda config: FixedBLIPITMScorer([[10, 20, 30]]))
    class FakeQwenConfig:
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        base_url = "http://localhost:12345/v1"
        api_key = "token-abc123"
        max_tokens = 512
        temperature = 0.0
        timeout_seconds = 120

    qwen_scores = [
        {
            "entity_match_score": 80,
            "prompt_alignment_score": 70,
            "kg_evidence_score": 60,
            "artifact_score": 90,
            "overall_judge_score": 75,
            "rationale": "图像基本符合。",
        }
    ]
    monkeypatch.setattr(evaluate_cli, "load_qwen_vl_judge_config", lambda path, timeout_seconds: FakeQwenConfig())
    monkeypatch.setattr(evaluate_cli, "QwenVLJudgeClient", lambda config: FixedQwenVLJudge(qwen_scores))

    exit_code = main([
        "--run-dir", str(run_dir),
        "--output-dir", str(output_dir),
        "--blip-itm-evaluator", "blip-itm-local",
        "--blip-model-dir", "blip-itm-large-coco",
        "--blip-device", "cpu",
        "--vlm-judge-evaluator", "qwen-vl-judge",
        "--vlm-judge-model-config", "configs/mkgw-knowledge-extraction.yaml",
    ])

    assert exit_code == 0
    assert (output_dir / "blip_itm_summary.json").is_file()
    assert (output_dir / "qwen_vl_judge_summary.json").is_file()
    assert (output_dir / "image_quality_model_comparison_summary.json").is_file()
```

Use actual imports from `mmkgc.evaluation.blip_itm` and `mmkgc.evaluation.qwen_vl_judge` after Task 2/3 are complete.

- [ ] **Step 2: Run CLI tests and verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluate_cli.py -v -p no:cacheprovider
```

Expected:

```text
new CLI option assertions fail
```

- [ ] **Step 3: Modify CLI parser**

In `mmkgc/evaluate.py`, add imports:

```python
from mmkgc.evaluation.blip_itm import (
    BlipITMConfig,
    BlipITMLocalScorer,
    evaluate_entity_blip_itm,
    validate_blip_model_dir,
)
from mmkgc.evaluation.multi_image_quality_writer import (
    write_blip_itm_outputs,
    write_image_quality_model_comparison_summary,
    write_qwen_vl_judge_outputs,
)
from mmkgc.evaluation.qwen_vl_judge import (
    QwenVLJudgeClient,
    evaluate_entity_qwen_vl_judge,
    load_qwen_vl_judge_config,
)
```

Add parser args:

```python
parser.add_argument("--blip-itm-evaluator", choices=("none", "blip-itm-local"), default="none")
parser.add_argument("--blip-model-dir", default="blip-itm-large-coco")
parser.add_argument("--blip-device", default="cuda")
parser.add_argument("--vlm-judge-evaluator", choices=("none", "qwen-vl-judge"), default="none")
parser.add_argument("--vlm-judge-model-config", default="configs/mkgw-knowledge-extraction.yaml")
parser.add_argument("--vlm-judge-timeout-seconds", type=int, default=120)
```

- [ ] **Step 4: Wire evaluation flow**

In `main()`:

```python
blip_results: list[dict] = []
qwen_vl_results: list[dict] = []
blip_summary = None
qwen_vl_summary = None
comparison_summary = None
```

Initialize BLIP only when requested:

```python
if args.blip_itm_evaluator == "blip-itm-local":
    blip_model_dir = validate_blip_model_dir(args.blip_model_dir)
    blip_scorer = BlipITMLocalScorer(
        BlipITMConfig(model_dir=Path(args.blip_model_dir), device=args.blip_device)
    )
```

Initialize Qwen-VL only when requested:

```python
if args.vlm_judge_evaluator == "qwen-vl-judge":
    qwen_vl_config = load_qwen_vl_judge_config(
        args.vlm_judge_model_config,
        timeout_seconds=args.vlm_judge_timeout_seconds,
    )
    qwen_vl_judge = QwenVLJudgeClient(qwen_vl_config)
```

After offline evaluation loop, run optional evaluators over loaded artifacts:

```python
if args.blip_itm_evaluator == "blip-itm-local":
    for load_result in load_results:
        if load_result.error is None and load_result.artifacts is not None:
            blip_results.append(
                evaluate_entity_blip_itm(
                    load_result.artifacts,
                    scorer=blip_scorer,
                    model_dir=blip_model_dir,
                )
            )
```

Do the same for Qwen-VL.

After `write_evaluation_outputs()` and optional CLIP writer:

```python
if args.blip_itm_evaluator == "blip-itm-local":
    blip_summary = write_blip_itm_outputs(
        output_dir=output_dir,
        run_dir=run_dir,
        entity_itm_results=blip_results,
        overwrite=args.overwrite,
    )
if args.vlm_judge_evaluator == "qwen-vl-judge":
    qwen_vl_summary = write_qwen_vl_judge_outputs(
        output_dir=output_dir,
        run_dir=run_dir,
        entity_judge_results=qwen_vl_results,
        overwrite=args.overwrite,
    )
if clip_summary or blip_summary or qwen_vl_summary:
    comparison_summary = write_image_quality_model_comparison_summary(
        output_dir=output_dir,
        clip_summary=clip_summary,
        blip_summary=blip_summary,
        qwen_vl_summary=qwen_vl_summary,
        overwrite=args.overwrite,
    )
```

Add stdout fields:

```python
"blip_itm_evaluator": args.blip_itm_evaluator,
"blip_evaluated_entities": blip_summary["evaluated_entities"] if blip_summary else 0,
"blip_evaluated_images": blip_summary["evaluated_images"] if blip_summary else 0,
"vlm_judge_evaluator": args.vlm_judge_evaluator,
"vlm_judged_entities": qwen_vl_summary["evaluated_entities"] if qwen_vl_summary else 0,
"vlm_judged_images": qwen_vl_summary["evaluated_images"] if qwen_vl_summary else 0,
```

- [ ] **Step 5: Run CLI and focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest \
  tests/test_evaluate_cli.py \
  tests/test_blip_itm.py \
  tests/test_qwen_vl_judge.py \
  tests/test_multi_image_quality_writer.py \
  -v -p no:cacheprovider
```

Expected:

```text
all selected tests pass
```

## Task 6: Stage 8 Real Smoke And Documentation

**Files:**
- Create: `docs/evaluation/2026-07-09-qwen-vl-blip-image-evaluation-validation.zh.md`
- Modify: `docs/project-progress-summary.zh.md`
- Modify: `README.md`
- Modify: `README.zh.md`

**Goal:** Run Stage 8 on the Stage 7 output and record real Qwen-VL/BLIP/CLIP comparison evidence.

**Implementer scope:**
- Run real models only after controller confirms Qwen/vLLM is running.
- Use escalated execution if sandbox cannot reach localhost/GPU.
- Do not commit.

- [ ] **Step 1: Preflight input run and local models**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test -d outputs/qwen_sdxl_clip_joint_smoke_20260709
test -f outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/clip_image_quality_summary.json
test -d blip-itm-large-coco
test -f blip-itm-large-coco/pytorch_model.bin
test -f configs/mkgw-knowledge-extraction.yaml
```

Expected:

```text
exit code 0
```

- [ ] **Step 2: Run Qwen health check**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
QWEN_HEALTH_TIMEOUT_SECONDS=30 /home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py
```

Expected:

```json
{"ok": true}
```

If normal sandbox cannot access localhost, controller reruns with escalated execution.

- [ ] **Step 3: Run Stage 8 evaluation**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_sdxl_clip_joint_smoke_20260709 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda:1 \
  --blip-itm-evaluator blip-itm-local \
  --blip-model-dir blip-itm-large-coco \
  --blip-device cuda:1 \
  --vlm-judge-evaluator qwen-vl-judge \
  --vlm-judge-model-config configs/mkgw-knowledge-extraction.yaml \
  --vlm-judge-timeout-seconds 120 \
  --overwrite
```

Expected stdout JSON contains:

```json
{
  "evaluated_entities": 5,
  "failed_entities": 0,
  "clip_evaluated_images": 25,
  "blip_evaluated_images": 25,
  "vlm_judged_images": 25
}
```

- [ ] **Step 4: Validate Stage 8 strict success**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -c "import csv,json; from pathlib import Path; run=Path('outputs/qwen_sdxl_clip_joint_smoke_20260709'); e=run/'evaluation'; q=json.load(open(e/'qwen_vl_judge_summary.json', encoding='utf-8')); b=json.load(open(e/'blip_itm_summary.json', encoding='utf-8')); c=json.load(open(e/'image_quality_model_comparison_summary.json', encoding='utf-8')); assert q['evaluated_entities']==5 and q['evaluated_images']==25 and q['failed_images']==0 and q['fallback_images']==0 and q['parse_failed_images']==0, q; assert b['evaluated_entities']==5 and b['evaluated_images']==25 and b['failed_images']==0 and b['fallback_images']==0, b; assert c['evaluated_images']==25, c; q_rows=list(csv.DictReader(open(e/'qwen_vl_judge_entity_scores.csv', encoding='utf-8'))); b_rows=list(csv.DictReader(open(e/'blip_itm_entity_scores.csv', encoding='utf-8'))); assert len(q_rows)==5 and len(b_rows)==5, (len(q_rows), len(b_rows)); print(json.dumps({'qwen_vl': {'entities': q['evaluated_entities'], 'images': q['evaluated_images'], 'overall': q['averages']['overall_judge_score']}, 'blip_itm': {'entities': b['evaluated_entities'], 'images': b['evaluated_images'], 'overall': b['averages']['blip_overall_itm_score']}, 'comparison_images': c['evaluated_images']}, ensure_ascii=False))"
```

Expected:

```text
exit code 0
printed JSON has qwen_vl.entities = 5
printed JSON has qwen_vl.images = 25
printed JSON has qwen_vl.overall as a real numeric value
printed JSON has blip_itm.entities = 5
printed JSON has blip_itm.images = 25
printed JSON has blip_itm.overall as a real numeric value
printed JSON has comparison_images = 25
```

- [ ] **Step 5: Write Stage 8 validation report**

Create `docs/evaluation/2026-07-09-qwen-vl-blip-image-evaluation-validation.zh.md` with:

```markdown
# Qwen-VL Judge + BLIP/ITM 图像评价 Smoke 验收报告

日期：2026-07-09
输入 run：`outputs/qwen_sdxl_clip_joint_smoke_20260709`

## 范围

记录本阶段只评价阶段 7 的 5 个实体、25 张 PNG，不重新生成 text/image，不扩大规模。

## Qwen-VL Judge 结果

- `evaluated_entities = 5`
- `evaluated_images = 25`
- `failed_images = 0`
- `fallback_images = 0`
- `parse_failed_images = 0`
保存报告前，从 Step 4 的实际输出中复制 Qwen-VL summary 的真实均值。报告必须包含：

- `entity_match_score`
- `prompt_alignment_score`
- `kg_evidence_score`
- `artifact_score`
- `overall_judge_score`

每个字段后面直接写 Step 4 输出中的真实数字。

## BLIP/ITM 结果

- `evaluated_entities = 5`
- `evaluated_images = 25`
- `failed_images = 0`
- `fallback_images = 0`
保存报告前，从 Step 4 的实际输出中复制 BLIP/ITM summary 的真实均值。报告必须包含：

- `blip_prompt_itm_score`
- `blip_text_itm_score`
- `blip_evidence_itm_score`
- `blip_overall_itm_score`

每个字段后面直接写 Step 4 输出中的真实数字。

## CLIP / BLIP / Qwen-VL 对比

记录 comparison summary 中的真实数值，并说明三者衡量不同信号，不能直接当作同一指标。

## 结论

如果 Qwen-VL 和 BLIP 均 25/25 成功，则记录阶段 8 通过；否则记录 blocked 或 failed smoke。
```

报告保存前必须确认上述字段都带有真实数字；如果仍有示例词或空值，停止并修正文档。

- [ ] **Step 6: Update progress summary and README**

Update:

```text
docs/project-progress-summary.zh.md
README.md
README.zh.md
```

Add concise Stage 8 status only after real smoke success. Include:

```text
Stage 8 evaluates the Stage 7 25 PNGs with Qwen-VL Judge and BLIP/ITM.
It does not expand scale or add new generation models.
Qwen-VL is a model judge, BLIP/ITM is image-text matching, CLIP remains cross-modal similarity.
```

- [ ] **Step 7: Run documentation checks**

Run:

```bash
rg -n "Qwen-VL Judge|BLIP/ITM|qwen_vl_judge_summary|blip_itm_summary|image_quality_model_comparison" docs/evaluation/2026-07-09-qwen-vl-blip-image-evaluation-validation.zh.md docs/project-progress-summary.zh.md README.md README.zh.md
```

Expected:

```text
matches in report, progress summary, and README files
```

## Task 7: Final Verification

**Files:**
- No intended writes.

**Goal:** Prove Stage 8 implementation and smoke are complete without pycache or model/output commit risk.

**Implementer scope:**
- Run verification only.
- Do not commit.

- [ ] **Step 1: Run full tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
```

Expected:

```text
all tests pass
```

- [ ] **Step 2: Run compileall without writing pycache into repo**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/mmkgc_stage8_pycache /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall -q mmkgc tests scripts
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: Check ignored model/data/output paths**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
git check-ignore -v outputs/qwen_sdxl_clip_joint_smoke_20260709 Qwen/Qwen2-VL-7B-Instruct SDXL/stable-diffusion-xl-base-1.0 CLIP/ViT-B-32.pt datasets/MKG-W configs/mkgw-knowledge-extraction.yaml blip-itm-large-coco
```

Expected:

```text
all listed paths are ignored by .gitignore
```

- [ ] **Step 4: Check no pycache tracked changes**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
git ls-files -m -- mmkgc tests scripts | rg "__pycache__|\\.pyc"
```

Expected:

```text
no output
exit code 1 from rg is acceptable because it means no matches
```

- [ ] **Step 5: Capture final git status**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
git status --short | sed -n '1,220p'
```

Expected:

```text
Stage 8 code/docs changes visible
No model weights, datasets, outputs, local configs, or pycache are listed for commit
Existing unrelated dirty files may remain
```

## Final Reviewer Handoff

After all tasks and task-level reviewers pass, dispatch a final reviewer with:

- Stage 8 spec path.
- Stage 8 plan path.
- Stage 8 report path.
- Real smoke command.
- Strict validation output from Task 6 Step 4.
- Full pytest output.
- compileall exit code.
- git ignore evidence.
- git status summary.

Final reviewer must approve before reporting Stage 8 complete.
