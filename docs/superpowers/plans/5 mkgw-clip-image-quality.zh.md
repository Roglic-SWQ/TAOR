# MKG-W CLIP Image Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional CLIP-based real PNG image quality evaluation to `python -m mmkgc.evaluate`, using local `CLIP/ViT-B-32.pt` without changing existing offline evaluation scores.

**Architecture:** Keep the existing offline evaluation path unchanged by default. Add a focused image-quality evaluation module that builds text targets from run artifacts, calls a local CLIP scorer only when requested, aggregates per-image/per-entity/run-level scores, and writes `clip_image_quality_*` outputs into the run evaluation directory. Use fail-fast behavior for missing CLIP weights, missing PNGs, unreadable images, and invalid local model paths.

**Tech Stack:** Python 3.11, pytest, Pillow, torch, installed `clip` package, existing `mmkgc.evaluation` loader/writer patterns.

---

## Controller Rules For This Plan

- 主 Codex 会话作为 controller。
- 每个 task 派一个 implementer subagent。
- 每个 implementer 必须收到本 task 的写入范围、验证命令、禁止 commit 指令。
- 每个 task 完成后派两个 reviewer：
  - spec reviewer：检查是否符合 `docs/superpowers/specs/5 mkgw-clip-image-quality-design.zh.md` 和用户确认方案。
  - quality reviewer：检查代码质量、测试质量、职责边界、路径/模板/打包风险。
- reviewer 发现问题时，必须回给 implementer 修，不跳过。
- 两个 reviewer 都 approve 后才标记 task 完成。
- 最后派 final reviewer 审完整变更。
- 最后运行完整验证命令并汇报证据。
- 工作区已有用户重命名文档的未提交改动，不 revert 任何非本任务改动。
- 不自动 commit。所有任务的 commit 步骤均被用户要求覆盖为“不要 commit”。

## Existing Dirty Workspace Notes

开始执行前，controller 必须确认并保留以下用户改动：

```text
docs/superpowers/plans/2026-*.md -> docs/superpowers/plans/<stage> *.md
docs/superpowers/specs/2026-*.md -> docs/superpowers/specs/<stage> *.md
CLIP/
```

这些是用户整理阶段文档和模型权重目录产生的改动。实现阶段不要恢复旧文件名，不要提交 `CLIP/` 权重。

## Files And Responsibilities

- Modify `.gitignore`
  - Add `CLIP/` so local CLIP weights are never committed.

- Create `mmkgc/evaluation/image_quality.py`
  - Owns CLIP model path validation, lazy CLIP loading, score computation, text target construction, per-image evaluation, and per-entity aggregation.
  - Must not write files.
  - Must not import `clip` at module import time; import lazily in `ClipLocalImageQualityEvaluator`.

- Create `mmkgc/evaluation/image_quality_writer.py`
  - Owns `clip_image_quality_summary.json`, `clip_image_quality_entity_scores.csv`, and per-entity `clip_image_quality.json` writing.
  - Keeps the already large `mmkgc/evaluation/writer.py` from growing further.

- Modify `mmkgc/evaluate.py`
  - Add CLI args:
    - `--image-quality-evaluator {none,clip-local}`
    - `--clip-model-name`
    - `--clip-model-dir`
    - `--clip-device`
  - Default remains `none`; old evaluation behavior unchanged.
  - When enabled, run CLIP image quality evaluation fail-fast and then write CLIP output files.

- Modify `mmkgc/evaluation/__init__.py`
  - Export image quality helpers only if existing package pattern requires it.

- Create `tests/test_image_quality.py`
  - Unit tests for CLIP path validation, model name to weight filename mapping, text target construction, score aggregation, and fail-fast behavior.

- Create or extend `tests/test_evaluate_cli.py`
  - CLI tests for default no-CLIP behavior, missing weight failure, and fake CLIP output writing.

- Create `tests/test_image_quality_writer.py`
  - Writer tests for summary JSON, CSV, per-entity JSON, overwrite behavior, and entity directory naming.

- Modify `README.md`, `README.zh.md`, `docs/project-progress-summary.zh.md`
  - Document Stage 6 command and limitation.

- Create `docs/evaluation/2026-07-09-clip-image-quality-smoke-validation.zh.md`
  - Record final smoke evidence after real run.

## Task 1: CLIP Path Validation And Scorer Boundary

**Files:**
- Create: `mmkgc/evaluation/image_quality.py`
- Test: `tests/test_image_quality.py`
- Modify: `.gitignore`

**Implementer scope:**
- Only create the image quality module skeleton, path helpers, model-name mapping, local CLIP evaluator boundary, and tests for these pieces.
- Do not integrate CLI yet.
- Do not write output files yet.
- Do not commit.

- [ ] **Step 1: Add failing tests for local CLIP path validation**

Add `tests/test_image_quality.py` with these tests:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from mmkgc.evaluation.image_quality import (
    ClipImageQualityError,
    clip_weight_filename,
    validate_clip_model_dir,
    validate_clip_weight_path,
)


def test_clip_weight_filename_maps_model_names() -> None:
    assert clip_weight_filename("ViT-B/32") == "ViT-B-32.pt"
    assert clip_weight_filename("ViT-L/14") == "ViT-L-14.pt"


def test_clip_weight_filename_rejects_unknown_model() -> None:
    with pytest.raises(ClipImageQualityError, match="unsupported clip_model_name"):
        clip_weight_filename("RN50")


def test_validate_clip_model_dir_accepts_project_local_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)

    resolved = validate_clip_model_dir("CLIP", project_root=project_root)

    assert resolved == model_dir.resolve()


def test_validate_clip_model_dir_rejects_external_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    external = tmp_path / "outside" / "CLIP"
    external.mkdir(parents=True)

    with pytest.raises(ClipImageQualityError, match="clip_model_dir must be under"):
        validate_clip_model_dir(external, project_root=project_root)


def test_validate_clip_model_dir_rejects_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project_root / "CLIP"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ClipImageQualityError, match="clip_model_dir must be under"):
        validate_clip_model_dir("CLIP", project_root=project_root)


def test_validate_clip_weight_path_requires_expected_file(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)

    with pytest.raises(ClipImageQualityError, match="missing CLIP weight"):
        validate_clip_weight_path(model_dir, "ViT-B/32")

    weight = model_dir / "ViT-B-32.pt"
    weight.write_bytes(b"fake")

    assert validate_clip_weight_path(model_dir, "ViT-B/32") == weight.resolve()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality.py -v -p no:cacheprovider
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'mmkgc.evaluation.image_quality'
```

- [ ] **Step 3: Implement path helpers and evaluator boundary**

Create `mmkgc/evaluation/image_quality.py` with these public APIs:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


MMKGC_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_CLIP_MODELS = {
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-L/14": "ViT-L-14.pt",
}


class ClipImageQualityError(RuntimeError):
    """Raised when CLIP image quality evaluation cannot run safely."""


class ImageTextSimilarityScorer(Protocol):
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        """Return one non-negative percentage score per text target."""


def clip_weight_filename(model_name: str) -> str:
    try:
        return SUPPORTED_CLIP_MODELS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_CLIP_MODELS))
        raise ClipImageQualityError(
            f"unsupported clip_model_name: {model_name}; supported: {supported}"
        ) from exc


def validate_clip_model_dir(
    model_dir: str | Path,
    *,
    project_root: str | Path = MMKGC_ROOT,
) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(model_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ClipImageQualityError(
            f"clip_model_dir must be under {root}: {resolved}"
        ) from exc
    if not resolved.exists():
        raise ClipImageQualityError(f"clip_model_dir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ClipImageQualityError(f"clip_model_dir is not a directory: {resolved}")
    return resolved


def validate_clip_weight_path(model_dir: str | Path, model_name: str) -> Path:
    directory = Path(model_dir).resolve()
    weight_path = directory / clip_weight_filename(model_name)
    if not weight_path.is_file():
        raise ClipImageQualityError(
            f"missing CLIP weight for {model_name}: expected {weight_path}"
        )
    resolved_weight = weight_path.resolve()
    try:
        resolved_weight.relative_to(directory)
    except ValueError as exc:
        raise ClipImageQualityError(
            f"CLIP weight must stay under {directory}: {resolved_weight}"
        ) from exc
    return resolved_weight
```

Also add a lazy evaluator class skeleton in the same file:

```python
@dataclass(frozen=True)
class ClipLocalConfig:
    model_name: str = "ViT-B/32"
    model_dir: Path = Path("CLIP")
    device: str = "cuda"


class ClipLocalImageQualityEvaluator:
    def __init__(
        self,
        config: ClipLocalConfig,
        *,
        project_root: str | Path = MMKGC_ROOT,
        clip_module: Any | None = None,
        model: Any | None = None,
        preprocess: Any | None = None,
    ):
        self.config = config
        self.project_root = Path(project_root)
        self.model_dir = validate_clip_model_dir(config.model_dir, project_root=project_root)
        self.weight_path = validate_clip_weight_path(self.model_dir, config.model_name)
        self._clip_module = clip_module
        self._model = model
        self._preprocess = preprocess

    def _load_clip(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._preprocess is not None:
            if self._clip_module is None:
                raise ClipImageQualityError("clip_module is required with injected model")
            return self._clip_module, self._model, self._preprocess
        try:
            import clip  # type: ignore[import-not-found]
        except Exception as exc:
            raise ClipImageQualityError("Python package 'clip' is required") from exc
        model, preprocess = clip.load(
            str(self.weight_path),
            device=self.config.device,
            jit=False,
        )
        model.eval()
        self._clip_module = clip
        self._model = model
        self._preprocess = preprocess
        return clip, model, preprocess
```

Do not implement real scoring in this task beyond a stub that raises:

```python
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        raise NotImplementedError("Task 2 implements CLIP scoring")
```

- [ ] **Step 4: Add `.gitignore` entry**

Modify `.gitignore`:

```gitignore
CLIP/
```

Do not change existing ignore entries.

- [ ] **Step 5: Run Task 1 tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality.py -v -p no:cacheprovider
```

Expected:

```text
9 passed
```

- [ ] **Step 6: Task 1 reviewer gates**

Spec reviewer checks:

- `CLIP/` is gitignored.
- CLIP model dir must stay inside `MMKGC/`.
- Weight mapping uses `ViT-B/32 -> ViT-B-32.pt`.
- No CLI or writer behavior is implemented in this task.

Quality reviewer checks:

- `clip` is not imported at module import time.
- Symlink escape is rejected.
- No weight files are staged.
- No commit was made.

## Task 2: CLIP Scoring, Text Targets, And Entity Aggregation

**Files:**
- Modify: `mmkgc/evaluation/image_quality.py`
- Test: `tests/test_image_quality.py`

**Implementer scope:**
- Implement actual scoring with injectable fake scorer support.
- Implement text target construction and per-entity image quality aggregation.
- Do not modify CLI.
- Do not write files.
- Do not commit.

- [ ] **Step 1: Add tests for score aggregation and text target construction**

Append to `tests/test_image_quality.py`:

```python
import json

from mmkgc.evaluation.image_quality import (
    FixedImageTextSimilarityScorer,
    build_entity_evidence_summary,
    evaluate_entity_image_quality,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)


def _artifacts_with_generated_image(entity_dir: Path) -> RunEntityArtifacts:
    _write_png(entity_dir / "generated_images" / "image_001.png")
    knowledge = {
        "entity_profile": {
            "entity_type": "film",
            "entity_label": "Example Film",
        }
    }
    reasoning = {
        "modality_constraints": {
            "must_include": ["film poster"],
            "forbidden": ["watermark"],
        }
    }
    ranking = {
        "text_ranking": {
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "generated_content": {
                        "value": "Example Film is a drama film set in Exampleland."
                    },
                    "offline_evaluation_score": 0.8,
                }
            ]
        },
        "image_ranking": {
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "positive_prompt": "a cinematic poster for Example Film",
                    "generated_content": {
                        "type": "image",
                        "value": "a cinematic poster for Example Film",
                        "image_path": "generated_images/image_001.png",
                    },
                }
            ]
        },
        "pair_ranking": {
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "text_candidate": {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Example Film is a drama film set in Exampleland."
                        },
                    },
                    "image_candidate": {"candidate_id": "image_001"},
                    "offline_evaluation_score": 0.9,
                }
            ]
        },
    }
    final = {
        "entity": {
            "entity_key": "Q1",
            "label": "Example Film",
            "description": "ground truth must not be copied into evidence summary",
        },
        "missing_mode": "both",
    }
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key="Q1",
        knowledge=knowledge,
        reasoning=reasoning,
        ranking=ranking,
        final=final,
    )


def test_build_entity_evidence_summary_uses_entity_and_kg_signals(tmp_path: Path) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")

    summary = build_entity_evidence_summary(artifacts)

    assert "Example Film" in summary
    assert "film" in summary
    assert "film poster" in summary
    assert "ground truth must not be copied" not in summary


def test_evaluate_entity_image_quality_scores_three_targets(tmp_path: Path) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    scorer = FixedImageTextSimilarityScorer([10.0, 20.0, 30.0])

    result = evaluate_entity_image_quality(
        artifacts,
        scorer=scorer,
        clip_model_name="ViT-B/32",
        clip_weight_path=Path("CLIP/ViT-B-32.pt"),
    )

    assert result["entity"]["entity_key"] == "Q1"
    assert result["image_count"] == 1
    image_score = result["image_scores"][0]
    assert image_score["image_candidate_id"] == "image_001"
    assert image_score["clip_prompt_similarity"] == 10.0
    assert image_score["clip_text_similarity"] == 20.0
    assert image_score["clip_evidence_similarity"] == 30.0
    assert image_score["clip_overall_image_quality_score"] == 20.0
    assert result["averages"]["clip_overall_image_quality_score"] == 20.0
    assert scorer.calls[0]["texts"][0] == "a cinematic poster for Example Film"
```

Add fail-fast tests:

```python
def test_evaluate_entity_image_quality_rejects_missing_png(tmp_path: Path) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    (artifacts.entity_dir / "generated_images" / "image_001.png").unlink()

    with pytest.raises(ClipImageQualityError, match="image file does not exist"):
        evaluate_entity_image_quality(
            artifacts,
            scorer=FixedImageTextSimilarityScorer([1.0, 2.0, 3.0]),
            clip_model_name="ViT-B/32",
            clip_weight_path=Path("CLIP/ViT-B-32.pt"),
        )


def test_evaluate_entity_image_quality_rejects_entity_without_text_targets(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    artifacts.ranking["image_ranking"]["ranked_candidates"][0].pop("positive_prompt")
    artifacts.ranking["image_ranking"]["ranked_candidates"][0]["generated_content"] = {
        "type": "image",
        "image_path": "generated_images/image_001.png",
    }
    artifacts.ranking["text_ranking"]["ranked_candidates"] = []
    artifacts.ranking["pair_ranking"]["ranked_pairs"] = []
    artifacts.knowledge["entity_profile"] = {}
    artifacts.reasoning["modality_constraints"] = {}
    artifacts.final["entity"] = {"entity_key": "Q1"}

    with pytest.raises(ClipImageQualityError, match="no text targets available"):
        evaluate_entity_image_quality(
            artifacts,
            scorer=FixedImageTextSimilarityScorer([]),
            clip_model_name="ViT-B/32",
            clip_weight_path=Path("CLIP/ViT-B-32.pt"),
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality.py -v -p no:cacheprovider
```

Expected:

```text
ImportError or NotImplementedError for FixedImageTextSimilarityScorer/build_entity_evidence_summary/evaluate_entity_image_quality
```

- [ ] **Step 3: Implement fake scorer and CLIP scoring**

Add to `mmkgc/evaluation/image_quality.py`:

```python
import contextlib


@dataclass
class FixedImageTextSimilarityScorer:
    scores: list[float]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        self.calls.append({"image_path": image_path, "texts": texts})
        if len(self.scores) != len(texts):
            raise ClipImageQualityError(
                f"fake scorer expected {len(self.scores)} texts, got {len(texts)}"
            )
        return [round(float(score), 6) for score in self.scores]
```

Implement `ClipLocalImageQualityEvaluator.score_image_texts`:

```python
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        if not texts:
            return []
        path = Path(image_path)
        if not path.is_file():
            raise ClipImageQualityError(f"image file does not exist: {path}")
        clip, model, preprocess = self._load_clip()
        try:
            from PIL import Image
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise ClipImageQualityError("Pillow and torch are required for CLIP scoring") from exc

        image = Image.open(path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(self.config.device)
        text_input = clip.tokenize(texts, truncate=True).to(self.config.device)
        with torch.no_grad():
            image_features = F.normalize(model.encode_image(image_input).float(), dim=-1)
            text_features = F.normalize(model.encode_text(text_input).float(), dim=-1)
            similarities = image_features @ text_features.T
        return [
            round(max(0.0, float(value)) * 100.0, 6)
            for value in similarities.squeeze(0).detach().cpu().tolist()
        ]
```

- [ ] **Step 4: Implement text target extraction and aggregation**

Add helper functions:

```python
def evaluate_entity_image_quality(
    artifacts: RunEntityArtifacts,
    *,
    scorer: ImageTextSimilarityScorer,
    clip_model_name: str,
    clip_weight_path: Path,
) -> dict[str, Any]:
    ...
```

Required behavior:

- Use ranked image candidates from `artifacts.ranking["image_ranking"]["ranked_candidates"]`.
- For each candidate, resolve image path from `generated_content.image_path`.
- Reject missing or unsafe image paths:
  - relative path must stay under `artifacts.entity_dir`;
  - file must exist.
- Build text targets in this order:
  - prompt from `positive_prompt`, `generated_content.prompt`, or `generated_content.value`;
  - candidate text from matching pair candidate by image candidate id, else best ranked text candidate;
  - evidence summary from `build_entity_evidence_summary`.
- Score only non-empty text targets.
- Map returned score list back to nullable fields.
- Compute per-image overall as average of available scores.
- Compute entity-level averages across per-image score fields.

Use field names from the spec exactly:

```python
CLIP_SCORE_FIELDS = (
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
)
```

Use this entity-level shape:

```python
{
    "entity": {"entity_key": ..., "label": ..., "missing_mode": ...},
    "image_count": 5,
    "clip_model_name": "ViT-B/32",
    "clip_weight_path": "CLIP/ViT-B-32.pt",
    "averages": {...},
    "best_image_candidate_id": "image_001",
    "worst_image_candidate_id": "image_003",
    "image_scores": [...]
}
```

- [ ] **Step 5: Run Task 2 tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality.py -v -p no:cacheprovider
```

Expected:

```text
all tests in tests/test_image_quality.py pass
```

- [ ] **Step 6: Task 2 reviewer gates**

Spec reviewer checks:

- Three targets are evaluated: prompt, text candidate, evidence summary.
- Ground truth description is not used as primary evidence summary.
- Missing PNG and missing all text targets fail-fast.
- Old offline scores are not touched.

Quality reviewer checks:

- CLIP scorer is injectable for tests.
- Real `clip` import is lazy.
- Image paths cannot escape entity directory.
- Aggregation handles `None` only where spec allows.
- No commit was made.

## Task 3: CLIP Image Quality Output Writer

**Files:**
- Create: `mmkgc/evaluation/image_quality_writer.py`
- Test: `tests/test_image_quality_writer.py`

**Implementer scope:**
- Only write CLIP-specific output writer and tests.
- Do not modify CLI.
- Do not run real CLIP.
- Do not commit.

- [ ] **Step 1: Add failing writer tests**

Create `tests/test_image_quality_writer.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

from mmkgc.evaluation.image_quality_writer import write_clip_image_quality_outputs


def _quality(entity_key: str, overall: float) -> dict:
    return {
        "entity": {
            "entity_key": entity_key,
            "label": f"Entity {entity_key}",
            "missing_mode": "both",
        },
        "image_count": 2,
        "clip_model_name": "ViT-B/32",
        "clip_weight_path": "CLIP/ViT-B-32.pt",
        "averages": {
            "clip_prompt_similarity": overall + 1,
            "clip_text_similarity": overall + 2,
            "clip_evidence_similarity": overall + 3,
            "clip_overall_image_quality_score": overall,
        },
        "best_image_candidate_id": f"{entity_key}-best",
        "worst_image_candidate_id": f"{entity_key}-worst",
        "image_scores": [
            {
                "image_candidate_id": f"{entity_key}-best",
                "clip_overall_image_quality_score": overall,
            }
        ],
    }


def test_write_clip_image_quality_outputs_writes_summary_csv_and_entities(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    summary = write_clip_image_quality_outputs(
        output_dir=output_dir,
        run_dir=tmp_path / "run",
        entity_quality_results=[_quality("Q1", 10.0), _quality("Q/Unsafe", 30.0)],
    )

    assert summary["evaluated_entities"] == 2
    assert summary["evaluated_images"] == 4
    assert summary["averages"]["clip_overall_image_quality_score"] == 20.0
    assert summary["best_entities"][0]["entity_key"] == "Q/Unsafe"
    assert summary["worst_entities"][0]["entity_key"] == "Q1"

    summary_path = output_dir / "clip_image_quality_summary.json"
    csv_path = output_dir / "clip_image_quality_entity_scores.csv"
    assert summary_path.is_file()
    assert csv_path.is_file()

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary["output_files"]["entity_scores_csv"] == (
        "clip_image_quality_entity_scores.csv"
    )

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["entity_key"] == "Q1"
    assert rows[1]["entity_key"] == "Q/Unsafe"
    assert rows[1]["clip_overall_image_quality_score"] == "30.0"

    entity_files = sorted((output_dir / "entities").glob("*/clip_image_quality.json"))
    assert len(entity_files) == 2
```

Add overwrite preservation test:

```python
def test_write_clip_outputs_overwrite_only_owned_clip_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "caller_notes.txt").write_text("keep", encoding="utf-8")

    write_clip_image_quality_outputs(
        output_dir=output_dir,
        run_dir=tmp_path / "run",
        entity_quality_results=[_quality("Q1", 10.0)],
        overwrite=True,
    )
    write_clip_image_quality_outputs(
        output_dir=output_dir,
        run_dir=tmp_path / "run",
        entity_quality_results=[_quality("Q2", 20.0)],
        overwrite=True,
    )

    assert (output_dir / "caller_notes.txt").read_text(encoding="utf-8") == "keep"
    entity_files = sorted((output_dir / "entities").glob("*/clip_image_quality.json"))
    assert len(entity_files) == 1
    persisted = json.loads(entity_files[0].read_text(encoding="utf-8"))
    assert persisted["entity"]["entity_key"] == "Q2"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality_writer.py -v -p no:cacheprovider
```

Expected:

```text
ModuleNotFoundError: No module named 'mmkgc.evaluation.image_quality_writer'
```

- [ ] **Step 3: Implement writer**

Create `mmkgc/evaluation/image_quality_writer.py` with:

```python
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from mmkgc.utils import write_json


SUMMARY_FILENAME = "clip_image_quality_summary.json"
ENTITY_SCORES_CSV_FILENAME = "clip_image_quality_entity_scores.csv"
ENTITY_FILENAME = "clip_image_quality.json"
CLIP_SCORE_FIELDS = (
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
)
```

Implement:

```python
def write_clip_image_quality_outputs(
    *,
    output_dir: str | Path,
    run_dir: str | Path,
    entity_quality_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    ...
```

Required behavior:

- Creates `output_dir` and `output_dir/entities`.
- On `overwrite=True`, remove:
  - `clip_image_quality_summary.json`
  - `clip_image_quality_entity_scores.csv`
  - existing `*/clip_image_quality.json` directories under `entities`.
- Preserve unrelated files such as `caller_notes.txt`.
- Write summary JSON.
- Write CSV with fields:

```python
ENTITY_SCORE_FIELDS = [
    "entity_key",
    "label",
    "missing_mode",
    "image_count",
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
    "best_image_candidate_id",
    "worst_image_candidate_id",
]
```

- Write one `clip_image_quality.json` per entity using the same hashed directory naming as `writer.py`.

- [ ] **Step 4: Run writer tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_image_quality_writer.py -v -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Task 3 reviewer gates**

Spec reviewer checks:

- Output filenames exactly match `clip_image_quality_summary.json`, `clip_image_quality_entity_scores.csv`, and `clip_image_quality.json`.
- Existing offline evaluation files are not renamed or overwritten.
- Summary includes evaluated entity/image counts and averages.

Quality reviewer checks:

- Writer does not import CLIP or torch.
- Overwrite is limited to owned CLIP files.
- Entity directory naming cannot collide.
- No commit was made.

## Task 4: `mmkgc.evaluate` CLI Integration

**Files:**
- Modify: `mmkgc/evaluate.py`
- Modify: `mmkgc/evaluation/__init__.py` only if imports need package-level export
- Test: `tests/test_evaluate_cli.py`

**Implementer scope:**
- Integrate optional `clip-local` mode into CLI.
- Preserve default evaluation behavior.
- Use fake evaluator in tests; do not load real CLIP in tests.
- Do not commit.

- [ ] **Step 1: Add CLI tests**

Extend `tests/test_evaluate_cli.py` with:

```python
def test_cli_help_includes_image_quality_options() -> None:
    help_text = evaluate_cli._build_parser().format_help()

    assert "--image-quality-evaluator" in help_text
    assert "--clip-model-name" in help_text
    assert "--clip-model-dir" in help_text
    assert "--clip-device" in help_text
```

If `_build_parser` does not exist yet, Task 4 should split `_parse_args` into `_build_parser` plus `_parse_args`.

Add fake CLIP integration test:

```python
def test_cli_clip_local_writes_clip_outputs_with_fake_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    image_dir = run_dir / "entities" / "Q1" / "generated_images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_dir / "image-1.png")

    ranking_path = run_dir / "entities" / "Q1" / "5_kg_aware_pair_ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["image_ranking"]["ranked_candidates"][0]["candidate_id"] = "image-1"
    ranking["image_ranking"]["ranked_candidates"][0]["generated_content"] = {
        "type": "image",
        "value": "portrait of Ada Lovelace as a person",
        "image_path": "generated_images/image-1.png",
    }
    ranking_path.write_text(json.dumps(ranking), encoding="utf-8")

    class FakeEvaluator:
        def __init__(self, config, project_root=None):
            self.config = config
            self.weight_path = Path("CLIP/ViT-B-32.pt")

        def score_image_texts(self, image_path, texts):
            return [10.0 + index for index, _ in enumerate(texts)]

    monkeypatch.setattr(evaluate_cli, "ClipLocalImageQualityEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        evaluate_cli,
        "validate_clip_model_dir",
        lambda model_dir: tmp_path / "CLIP",
    )
    monkeypatch.setattr(
        evaluate_cli,
        "validate_clip_weight_path",
        lambda model_dir, model_name: Path("CLIP/ViT-B-32.pt"),
    )

    exit_code = main(
        [
            "--run-dir",
            str(run_dir),
            "--image-quality-evaluator",
            "clip-local",
            "--clip-model-name",
            "ViT-B/32",
            "--clip-model-dir",
            "CLIP",
            "--clip-device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert (run_dir / "evaluation" / "evaluation_summary.json").is_file()
    assert (run_dir / "evaluation" / "clip_image_quality_summary.json").is_file()
```

Add default no-CLIP test:

```python
def test_cli_default_does_not_require_clip_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    assert not (run_dir / "evaluation" / "clip_image_quality_summary.json").exists()
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluate_cli.py -v -p no:cacheprovider
```

Expected before implementation:

```text
AttributeError: module 'mmkgc.evaluate' has no attribute '_build_parser'
or argparse rejects --image-quality-evaluator
```

- [ ] **Step 3: Implement parser changes**

Modify `mmkgc/evaluate.py`:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an existing MMKGC run offline.")
    parser.add_argument("--run-dir", required=True, help="Path to an existing MMKGC run directory.")
    parser.add_argument("--output-dir", help="Directory for evaluation artifacts. Defaults to <run-dir>/evaluation.")
    parser.add_argument("--case-limit", type=int, default=20, help="Number of best, worst, and fallback cases to export.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite evaluation artifacts in an existing output directory.")
    parser.add_argument("--image-quality-evaluator", choices=["none", "clip-local"], default="none")
    parser.add_argument("--clip-model-name", default="ViT-B/32")
    parser.add_argument("--clip-model-dir", default="CLIP")
    parser.add_argument("--clip-device", default="cuda")
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)
```

- [ ] **Step 4: Implement optional CLIP flow**

At top of `mmkgc/evaluate.py`, import:

```python
from mmkgc.evaluation.image_quality import (
    ClipLocalConfig,
    ClipLocalImageQualityEvaluator,
    evaluate_entity_image_quality,
    validate_clip_model_dir,
    validate_clip_weight_path,
)
from mmkgc.evaluation.image_quality_writer import write_clip_image_quality_outputs
```

In `main`, after `entity_evaluations` and load results are collected, collect CLIP results only when requested:

```python
    clip_quality_results: list[dict] = []
    if args.image_quality_evaluator == "clip-local":
        clip_model_dir = validate_clip_model_dir(args.clip_model_dir)
        clip_weight_path = validate_clip_weight_path(clip_model_dir, args.clip_model_name)
        clip_evaluator = ClipLocalImageQualityEvaluator(
            ClipLocalConfig(
                model_name=args.clip_model_name,
                model_dir=Path(args.clip_model_dir),
                device=args.clip_device,
            )
        )
        for load_result in loaded_results:
            if load_result.artifacts is None:
                continue
            clip_quality_results.append(
                evaluate_entity_image_quality(
                    load_result.artifacts,
                    scorer=clip_evaluator,
                    clip_model_name=args.clip_model_name,
                    clip_weight_path=Path("CLIP") / clip_weight_path.name,
                )
            )
```

Implementation detail: avoid iterating `iter_entity_artifact_results(run_dir)` twice by storing the load results:

```python
load_results = list(iter_entity_artifact_results(run_dir))
```

Then use `load_results` for both existing evaluation and CLIP evaluation.

After `write_evaluation_outputs(...)`, write CLIP outputs when enabled:

```python
    if args.image_quality_evaluator == "clip-local":
        write_clip_image_quality_outputs(
            output_dir=output_dir,
            run_dir=run_dir,
            entity_quality_results=clip_quality_results,
            overwrite=args.overwrite,
        )
```

Stdout JSON should include:

```python
"image_quality_evaluator": args.image_quality_evaluator,
"clip_evaluated_entities": len(clip_quality_results),
"clip_evaluated_images": sum(result["image_count"] for result in clip_quality_results),
```

Only include CLIP counts when enabled, or include zero for `none`; either is acceptable if tests assert the chosen behavior.

- [ ] **Step 5: Run CLI tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluate_cli.py tests/test_image_quality.py tests/test_image_quality_writer.py -v -p no:cacheprovider
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 6: Task 4 reviewer gates**

Spec reviewer checks:

- Default evaluation does not require CLIP.
- `clip-local` writes CLIP outputs into same evaluation directory.
- Existing evaluation summary still writes.
- CLI args match spec.

Quality reviewer checks:

- No real CLIP load in tests.
- Fail-fast errors are not swallowed as failed entities.
- Output overwrite behavior does not delete run inputs.
- No commit was made.

## Task 5: Documentation, Smoke Validation Scriptlet, And Final Report

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `docs/project-progress-summary.zh.md`
- Create: `docs/evaluation/2026-07-09-clip-image-quality-smoke-validation.zh.md`
- Modify: `.gitignore` only if Task 1 did not already add `CLIP/`

**Implementer scope:**
- Update docs after code tasks pass.
- Run the real CLIP smoke only if controller confirms `CLIP/ViT-B-32.pt` and `outputs/sdxl_image_smoke_20260708` exist.
- Do not commit.

- [ ] **Step 1: Add README command**

In `README.md` and `README.zh.md`, add a "Local CLIP Image Quality Evaluation" / "本地 CLIP 图像质量评价" section with this command:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda \
  --overwrite
```

Text must state:

- `CLIP/` is local and gitignored.
- CLIP computes cross-modal similarity, not VLM judging.
- CLIP scores do not replace `calibrated_image_best_offline_evaluation_score`.

- [ ] **Step 2: Update project progress summary**

Append Stage 6 to `docs/project-progress-summary.zh.md`:

```markdown
## 阶段 6：CLIP 真实图像质量评价

目标是在 SDXL 已生成真实 PNG 后，用本地 OpenAI CLIP ViT-B-32 计算 PNG 与 image prompt、text candidate、entity evidence summary 的跨模态相似度。

第一版不接 BLIP/VLM judge，不替换旧 offline evaluation score，只新增 `clip_*` image quality 指标。
```

If smoke has run, include actual result numbers. If smoke has not run yet, state that implementation supports the command but真实 smoke证据由 controller 最终验证补充.

- [ ] **Step 3: Run real smoke**

Controller or implementer runs:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test -f CLIP/ViT-B-32.pt
test -d outputs/sdxl_image_smoke_20260708
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda \
  --overwrite
```

Expected:

```text
exit code 0
stdout JSON includes evaluated_entities=5
stdout JSON includes clip_evaluated_entities=5
stdout JSON includes clip_evaluated_images=25
```

If CUDA is unavailable in the Codex sandbox but CPU can run, retry with:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cpu \
  --overwrite
```

If both fail due environment limitation, stop and report exact error; do not fabricate smoke success.

- [ ] **Step 4: Validate CLIP output files**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test -f outputs/sdxl_image_smoke_20260708/evaluation/clip_image_quality_summary.json
test -f outputs/sdxl_image_smoke_20260708/evaluation/clip_image_quality_entity_scores.csv
/home/lirunjin/miniconda3/envs/kb/bin/python -c "import json; p='outputs/sdxl_image_smoke_20260708/evaluation/clip_image_quality_summary.json'; d=json.load(open(p, encoding='utf-8')); assert d['evaluated_entities'] == 5; assert d['evaluated_images'] == 25; print(json.dumps({'evaluated_entities': d['evaluated_entities'], 'evaluated_images': d['evaluated_images'], 'averages': d['averages']}, ensure_ascii=False))"
```

Expected:

```text
{"evaluated_entities": 5, "evaluated_images": 25, "averages": {...}}
```

- [ ] **Step 5: Write validation report**

Create `docs/evaluation/2026-07-09-clip-image-quality-smoke-validation.zh.md` with:

```markdown
# CLIP 真实图像质量评价 Smoke 验收报告

日期：2026-07-09
工作目录：`/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC`

## 范围

本报告验证阶段 6：使用本地 OpenAI CLIP `ViT-B-32` 对 `outputs/sdxl_image_smoke_20260708` 中的真实 PNG 做跨模态相似度评价。

## 验收命令

```bash
...
```

## 结果

从 `clip_image_quality_summary.json` 抄录以下实际数值：

- `evaluated_entities`
- `evaluated_images`
- `averages.clip_prompt_similarity`
- `averages.clip_text_similarity`
- `averages.clip_evidence_similarity`
- `averages.clip_overall_image_quality_score`

## 限制

- CLIP 分数是跨模态相似度，不是 VLM judge。
- 本阶段不证明 KG 关系严格成立。
- 本阶段不替换旧 offline evaluation 或 calibrated score。
```

报告必须包含真实 smoke 输出中的实际数值，不允许留下尖括号模板值或示例数字。

- [ ] **Step 6: Task 5 reviewer gates**

Spec reviewer checks:

- Docs mention local `CLIP/ViT-B-32.pt`.
- Docs state CLIP does not replace calibrated score.
- Validation report has actual numbers and no placeholders.

Quality reviewer checks:

- README commands are copy-paste runnable.
- No generated outputs or CLIP weights are staged.
- No commit was made.

## Task 6: Full Verification And Final Review

**Files:**
- No intended code writes.
- Possible docs fix only if verification reveals documentation mismatch.

**Implementer scope:**
- This is controller-owned unless explicitly delegated.
- Do not commit.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest \
  tests/test_image_quality.py \
  tests/test_image_quality_writer.py \
  tests/test_evaluate_cli.py \
  -v -p no:cacheprovider
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Run full tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: Run compileall outside repo pycache**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONPYCACHEPREFIX=/tmp/mmkgc-clip-compileall-pycache \
  /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

Expected:

```text
exit code 0
```

- [ ] **Step 4: Check ignored local weights and outputs**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
git check-ignore -v CLIP/ViT-B-32.pt CLIP/ViT-L-14.pt outputs/sdxl_image_smoke_20260708
git status --short
```

Expected:

```text
CLIP/ViT-B-32.pt is ignored by .gitignore
CLIP/ViT-L-14.pt is ignored by .gitignore
outputs/sdxl_image_smoke_20260708 is ignored by .gitignore
```

`git status --short` may still show the user's stage-document rename changes. It must not show CLIP weights as staged or untracked once `.gitignore` is updated.

- [ ] **Step 5: Run final reviewer**

Dispatch final reviewer with this checklist:

- All spec requirements in `docs/superpowers/specs/5 mkgw-clip-image-quality-design.zh.md` are implemented.
- Default `mmkgc.evaluate` behavior remains compatible.
- CLIP mode fail-fast behavior is clear.
- No generated outputs, model weights, local configs, or pycache files are staged.
- Existing user file renames were not reverted.
- No commit was made.

- [ ] **Step 6: Final report to user**

Final response must include:

- What was implemented.
- Files changed.
- Verification commands and evidence.
- Whether real CLIP smoke ran on GPU or CPU.
- Any limitation or blocker.
- Confirmation that no commit was made.
