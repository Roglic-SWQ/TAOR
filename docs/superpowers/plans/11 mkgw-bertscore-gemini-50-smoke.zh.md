# MKG-W BERTScore + Gemini 50 Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BERTScore and Gemini Embedding as independent evaluation signals, then run a 50-entity MKG-W smoke test without replacing existing CLIP, BLIP/ITM, Qwen-VL, Stage 5, or calibrated offline scores.

**Architecture:** Extend the existing `mmkgc.evaluate` provider pattern. Add one text-quality evaluator for BERTScore, one Gemini embedding evaluator for text/image/pair semantic alignment, provider-specific writers, comparison summary integration, and a 50-entity validation report. Real BERTScore/Gemini calls must sit behind injectable scorer/client boundaries so unit tests never require model downloads or network.

**Tech Stack:** Python 3.11, existing `mmkgc.evaluate`, existing image quality target extraction, optional `bertscore`, Gemini API REST `gemini-embedding-2`, pytest, compileall.

## Global Constraints

- Gemini must not replace CLIP, BLIP/ITM, Qwen-VL Judge, Stage 5 ranking, or calibrated offline evaluation.
- BERTScore must be used for text candidate quality only in this stage.
- Gemini must be added as a parallel evaluator for text, image, and pair semantic alignment.
- First real smoke scale is 50 entities, `missing_mode=both`, expected 250 PNG images.
- Do not auto commit.
- Do not write API keys, Authorization headers, model weights, datasets, or outputs into git.
- Tests must use fake BERTScore/Gemini providers and must not call network.
- Real Gemini provider must fail fast when `GEMINI_API_KEY` or configured API key env var is missing.
- Real BERTScore provider must fail fast when dependency/model loading fails.

---

## File Structure

- Create `MMKGC/mmkgc/evaluation/text_quality.py`
  - BERTScore provider boundary, fake scorer, text target/reference extraction, per-entity aggregation.
- Create `MMKGC/mmkgc/evaluation/text_quality_writer.py`
  - Writes `bertscore_text_quality_summary.json`, `bertscore_text_quality_entity_scores.csv`, and per-entity `bertscore_text_quality.json`.
- Create `MMKGC/mmkgc/evaluation/gemini_embedding.py`
  - Gemini config loading, REST client, fake client, cosine scoring, text/image/pair target evaluation.
- Create `MMKGC/mmkgc/evaluation/gemini_embedding_writer.py`
  - Writes Gemini summary, entity CSV, image CSV, pair CSV, and per-entity JSON.
- Modify `MMKGC/mmkgc/evaluate.py`
  - Add CLI options and call the new evaluators/writers.
- Modify `MMKGC/mmkgc/evaluation/multi_image_quality_writer.py`
  - Add optional Gemini/BERTScore fields to comparison summary without changing existing CLIP/BLIP/Qwen output contracts.
- Create `MMKGC/configs/gemini-embedding.example.yaml`
  - Safe template only.
- Modify `MMKGC/.gitignore`
  - Ignore `configs/gemini-embedding.yaml`.
- Create tests:
  - `MMKGC/tests/test_text_quality.py`
  - `MMKGC/tests/test_text_quality_writer.py`
  - `MMKGC/tests/test_gemini_embedding.py`
  - `MMKGC/tests/test_gemini_embedding_writer.py`
  - Extend `MMKGC/tests/test_evaluate_cli.py`
  - Extend `MMKGC/tests/test_multi_image_quality_writer.py`
- Create final report after real run:
  - `MMKGC/docs/evaluation/2026-07-15-bertscore-gemini-50-smoke-validation.zh.md`

## Task 1: BERTScore Text Quality Evaluator

**Files:**
- Create: `MMKGC/mmkgc/evaluation/text_quality.py`
- Test: `MMKGC/tests/test_text_quality.py`

**Interfaces:**
- Produces: `evaluate_entity_text_quality(artifacts, scorer, model_type) -> dict[str, Any]`
- Produces: `FixedBERTScoreScorer(scores: list[Any])`
- Produces: `BERTScoreLocalScorer(model_type: str, device: str | None)`
- Consumes: `RunEntityArtifacts`

- [ ] **Step 1: Write failing tests for reference selection and aggregation**

Add tests that build a minimal `RunEntityArtifacts` with two text candidates. One test uses `final.entity.description` as `ground_truth`; another uses `build_entity_evidence_summary()` when description is empty. Use `FixedBERTScoreScorer` with deterministic precision/recall/f1 triples.

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_text_quality.py -v
```

Expected before implementation: import or attribute failure for `mmkgc.evaluation.text_quality`.

- [ ] **Step 2: Implement the fake scorer and target extraction**

Create:

```python
class BERTScoreTextQualityError(RuntimeError):
    pass

@dataclass
class FixedBERTScoreScorer:
    scores: list[Any]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def score_pairs(self, candidates: list[str], references: list[str]) -> list[dict[str, float]]:
        self.calls.append({"candidates": list(candidates), "references": list(references)})
        if len(candidates) != len(references):
            raise BERTScoreTextQualityError("candidates and references length mismatch")
        rows = self.scores
        if rows and isinstance(rows[0], dict):
            selected = rows
        else:
            selected = [
                {"precision": row[0], "recall": row[1], "f1": row[2]}
                for row in rows
            ]
        if len(selected) != len(candidates):
            raise BERTScoreTextQualityError("fake scorer row count mismatch")
        return [
            {
                "precision": round(float(row["precision"]), 6),
                "recall": round(float(row["recall"]), 6),
                "f1": round(float(row["f1"]), 6),
            }
            for row in selected
        ]
```

Implement `evaluate_entity_text_quality()` so each text candidate receives:

```text
bertscore_precision
bertscore_recall
bertscore_f1
bertscore_reference_kind
bertscore_overall_text_score = bertscore_f1
```

- [ ] **Step 3: Implement real BERTScore scorer boundary**

`BERTScoreLocalScorer.score_pairs()` imports `bertscore.score`, calls it with candidate/reference lists, and returns float rows. It must catch import/model errors and raise `BERTScoreTextQualityError` with a concise message.

- [ ] **Step 4: Verify Task 1**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_text_quality.py -v
```

Expected: all `test_text_quality.py` tests pass. No network or real BERTScore model call occurs in tests.

## Task 2: BERTScore Writer

**Files:**
- Create: `MMKGC/mmkgc/evaluation/text_quality_writer.py`
- Test: `MMKGC/tests/test_text_quality_writer.py`

**Interfaces:**
- Consumes: `evaluate_entity_text_quality()` result dictionaries.
- Produces: `write_bertscore_text_quality_outputs(output_dir, run_dir, entity_text_quality_results, overwrite=False) -> dict[str, Any]`

- [ ] **Step 1: Write failing writer tests**

Tests must assert:

- summary JSON is written.
- entity CSV has one header plus one row per entity.
- per-entity `bertscore_text_quality.json` is written under `evaluation/entities/<entity_key>/`.
- overwrite only removes BERTScore-owned files.
- existing output without overwrite raises `FileExistsError`.

- [ ] **Step 2: Implement writer**

Use the existing CLIP writer style. Owned filenames:

```python
SUMMARY_FILENAME = "bertscore_text_quality_summary.json"
ENTITY_SCORES_CSV_FILENAME = "bertscore_text_quality_entity_scores.csv"
ENTITY_FILENAME = "bertscore_text_quality.json"
```

Summary fields:

```text
run_dir
output_dir
evaluator = bertscore-local
model_type
evaluated_entities
evaluated_text_candidates
averages.bertscore_precision
averages.bertscore_recall
averages.bertscore_f1
averages.bertscore_overall_text_score
best_entities
worst_entities
output_files
```

- [ ] **Step 3: Verify Task 2**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_text_quality.py tests/test_text_quality_writer.py -v
```

Expected: all tests pass.

## Task 3: Gemini Embedding Evaluator

**Files:**
- Create: `MMKGC/mmkgc/evaluation/gemini_embedding.py`
- Create: `MMKGC/configs/gemini-embedding.example.yaml`
- Modify: `MMKGC/.gitignore`
- Test: `MMKGC/tests/test_gemini_embedding.py`

**Interfaces:**
- Produces: `GeminiEmbeddingConfig`
- Produces: `load_gemini_embedding_config(path: str | Path) -> GeminiEmbeddingConfig`
- Produces: `GeminiEmbeddingClient.embed_text(text: str) -> list[float]`
- Produces: `GeminiEmbeddingClient.embed_image(image_path: Path) -> list[float]`
- Produces: `FixedGeminiEmbeddingClient(embeddings_by_key: dict[str, list[float]])`
- Produces: `evaluate_entity_gemini_embedding(artifacts, client, model_name) -> dict[str, Any]`

- [ ] **Step 1: Write failing config and cosine tests**

Tests must cover:

- `configs/gemini-embedding.yaml` outside project root is rejected.
- missing API key env var raises `GeminiEmbeddingError`.
- fake client returns deterministic vectors.
- cosine maps identical vectors to `100.0` and orthogonal vectors to `0.0`.

- [ ] **Step 2: Add safe config template and gitignore**

Create `configs/gemini-embedding.example.yaml`:

```yaml
gemini:
  api_key_env: GEMINI_API_KEY
  base_url: https://generativelanguage.googleapis.com/v1beta
  model_name: gemini-embedding-2
  output_dimensionality: 768
  timeout_seconds: 120
  max_retries: 2
```

Add to `.gitignore`:

```text
MMKGC/configs/gemini-embedding.yaml
```

- [ ] **Step 3: Implement REST client boundary**

Use `urllib.request` and no new required runtime dependency. For text, POST to:

```text
{base_url}/models/{model_name}:embedContent
```

Payload for text:

```json
{
  "content": {"parts": [{"text": "task: sentence similarity | query: ..."}]},
  "output_dimensionality": 768
}
```

Payload for image:

```json
{
  "content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": "<base64>"}}]},
  "output_dimensionality": 768
}
```

Never write API key or raw headers into results.

- [ ] **Step 4: Implement entity evaluation**

For each entity:

- text candidates get `gemini_text_ground_truth_similarity`, `gemini_text_evidence_similarity`, `gemini_overall_text_embedding_score`.
- image candidates get `gemini_prompt_similarity`, `gemini_text_similarity`, `gemini_evidence_similarity`, `gemini_overall_multimodal_score`.
- pair candidates get `gemini_pair_image_text_similarity`, `gemini_pair_image_evidence_similarity`, `gemini_pair_text_evidence_similarity`, `gemini_overall_pair_embedding_score`.

The evaluator may cache embeddings by stable content key during one entity evaluation so repeated prompt/text/evidence embeddings are not recomputed.

- [ ] **Step 5: Verify Task 3**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_gemini_embedding.py -v
```

Expected: all tests pass with fake client only.

## Task 4: Gemini Writer

**Files:**
- Create: `MMKGC/mmkgc/evaluation/gemini_embedding_writer.py`
- Test: `MMKGC/tests/test_gemini_embedding_writer.py`

**Interfaces:**
- Consumes: `evaluate_entity_gemini_embedding()` result dictionaries.
- Produces: `write_gemini_embedding_outputs(output_dir, run_dir, entity_embedding_results, overwrite=False) -> dict[str, Any]`

- [ ] **Step 1: Write failing writer tests**

Tests must assert all output files exist:

```text
gemini_embedding_summary.json
gemini_embedding_entity_scores.csv
gemini_embedding_image_scores.csv
gemini_embedding_pair_scores.csv
entities/<entity_key>/gemini_embedding.json
```

- [ ] **Step 2: Implement writer**

Summary fields:

```text
run_dir
output_dir
evaluator = gemini-api
model_name
evaluated_entities
evaluated_text_candidates
evaluated_images
evaluated_pairs
averages.gemini_overall_text_embedding_score
averages.gemini_overall_multimodal_score
averages.gemini_overall_pair_embedding_score
best_entities
worst_entities
output_files
```

CSV files must be row-per-entity, row-per-image, and row-per-pair respectively.

- [ ] **Step 3: Verify Task 4**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_gemini_embedding.py tests/test_gemini_embedding_writer.py -v
```

Expected: all tests pass.

## Task 5: CLI Integration And Comparison Summary

**Files:**
- Modify: `MMKGC/mmkgc/evaluate.py`
- Modify: `MMKGC/mmkgc/evaluation/multi_image_quality_writer.py`
- Test: `MMKGC/tests/test_evaluate_cli.py`
- Test: `MMKGC/tests/test_multi_image_quality_writer.py`

**Interfaces:**
- Adds CLI args:
  - `--text-quality-evaluator {none,bertscore-local}`
  - `--bertscore-model-type roberta-large`
  - `--bertscore-device`
  - `--gemini-embedding-evaluator {none,gemini-api}`
  - `--gemini-embedding-config configs/gemini-embedding.yaml`

- [ ] **Step 1: Write failing CLI tests with monkeypatched fake evaluators**

Tests must assert:

- default `mmkgc.evaluate` still does not import/load BERTScore or Gemini.
- enabling `bertscore-local` calls text-quality writer.
- enabling `gemini-api` calls Gemini writer.
- comparison summary can include Gemini and BERTScore records while preserving existing CLIP/BLIP/Qwen fields.

- [ ] **Step 2: Implement CLI wiring**

Initialize optional providers only when explicitly enabled. Reuse `load_results` already collected in `evaluate.py`. Write new outputs after existing offline evaluation outputs, following the CLIP/BLIP/Qwen pattern.

- [ ] **Step 3: Extend comparison summary**

Add optional keyword arguments:

```python
bertscore_summary: dict[str, Any] | None = None
gemini_summary: dict[str, Any] | None = None
```

Do not change existing callers; default `None` keeps current behavior.

- [ ] **Step 4: Verify Task 5**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluate_cli.py tests/test_multi_image_quality_writer.py -v
```

Expected: all tests pass and default CLI behavior remains unchanged.

## Task 6: Full Regression Verification

**Files:**
- No source changes unless verification finds a task-specific bug.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest \
  tests/test_text_quality.py \
  tests/test_text_quality_writer.py \
  tests/test_gemini_embedding.py \
  tests/test_gemini_embedding_writer.py \
  tests/test_evaluate_cli.py \
  tests/test_multi_image_quality_writer.py \
  -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run all tests**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Compile package and tests**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

Expected: exit code 0.

## Task 7: 50 Entity Real Smoke Run

**Files:**
- Create output directory under `MMKGC/outputs/`
- Create report: `MMKGC/docs/evaluation/2026-07-15-bertscore-gemini-50-smoke-validation.zh.md`
- Do not commit.

**Preflight Requirements:**
- Qwen local service is healthy.
- `SDXL/stable-diffusion-xl-base-1.0` exists.
- `CLIP/ViT-B-32.pt` exists if CLIP comparison is enabled.
- `blip-itm-large-coco/` exists if BLIP comparison is enabled.
- `configs/gemini-embedding.yaml` exists or `GEMINI_API_KEY` is set according to config.
- BERTScore dependency and local model cache are available.

- [ ] **Step 1: Check model/service readiness**

Run:

```bash
cd MMKGC
nvidia-smi
curl -sS --max-time 10 http://localhost:12345/v1/models
/bin/bash -lc "QWEN_HEALTH_TIMEOUT_SECONDS=30 /home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py --config configs/mkgw-knowledge-extraction.yaml"
```

Expected: GPU is visible, Qwen model endpoint responds, health check exits 0.

- [ ] **Step 2: Run 50 entity Qwen + SDXL generation**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 50 \
  --text-generator qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260715 \
  --sdxl-guidance-scale 7.5 \
  --sdxl-device cuda:1 \
  --output-root outputs \
  --run-name qwen_sdxl_gemini_bertscore_50_20260715
```

Expected: run completes with `processed_entities = 50`, `failed_entities = 0`, and 250 PNG files.

- [ ] **Step 3: Run evaluation with existing and new evaluators**

Run:

```bash
cd MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_sdxl_gemini_bertscore_50_20260715 \
  --overwrite \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda:1 \
  --blip-itm-evaluator blip-itm-local \
  --blip-model-dir blip-itm-large-coco \
  --blip-device cuda:1 \
  --vlm-judge-evaluator qwen-vl-judge \
  --vlm-judge-model-config configs/mkgw-knowledge-extraction.yaml \
  --text-quality-evaluator bertscore-local \
  --bertscore-model-type roberta-large \
  --bertscore-device cuda:1 \
  --gemini-embedding-evaluator gemini-api \
  --gemini-embedding-config configs/gemini-embedding.yaml
```

Expected: offline evaluation, CLIP, BLIP, Qwen-VL, BERTScore, and Gemini outputs are all written.

- [ ] **Step 4: Validate counts and sensitive data**

Run a small read-only validation script or shell checks to confirm:

```text
evaluation_summary.json evaluated_entities = 50
clip_image_quality_summary.json evaluated_images = 250
blip_itm_summary.json evaluated_images = 250
qwen_vl_judge_summary.json evaluated_images = 250
gemini_embedding_summary.json evaluated_images = 250
bertscore_text_quality_summary.json evaluated_entities = 50
no output file contains the Gemini API key value
```

- [ ] **Step 5: Write validation report**

Create `docs/evaluation/2026-07-15-bertscore-gemini-50-smoke-validation.zh.md` with:

- exact run command and evaluation command.
- processed/failed counts.
- PNG count.
- BERTScore summary.
- Gemini text/image/pair summary.
- CLIP/BLIP/Qwen/Gemini disagreement notes.
- statement that Gemini and BERTScore are not yet promoted to production fusion weights.

## Task 8: Final Review Package

**Files:**
- Review all changed source, tests, config template, docs, and report.

- [ ] **Step 1: Check changed files**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main
git status --short
```

Expected: no model weights, generated images, datasets, private configs, or outputs are staged or committed.

- [ ] **Step 2: Run final verification**

Run:

```bash
cd MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

Expected: tests pass and compileall exits 0.

- [ ] **Step 3: Final reviewer checklist**

Final reviewer must confirm:

- Gemini did not replace CLIP/BLIP/Qwen-VL outputs.
- BERTScore did not replace existing offline text scores.
- No secrets are written to git-tracked files.
- 50-entity report cites real output counts.
- No automatic commit was made.
