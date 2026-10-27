# MKG-W SDXL 图像生成接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MMKGC 内接入本地 SDXL Base 1.0，把 Stage 4 的 5 个 image prompt candidates 生成为每实体 5 张真实 PNG，并保留规则/Qwen 文本链路兼容性。

**Architecture:** 保留现有 Stage 0-5 契约，新增可选 `ImageGenerator` 边界。`sdxl-local` 模式在 Stage 4 image prompt 生成后调用 SDXL，pipeline 返回图片 artifact，`RunWriter` 负责把 PNG、sidecar metadata、`image_model_calls.jsonl` 写入每个实体目录；默认 `rule` 模式不生成图片文件。

**Tech Stack:** Python 3.11、Diffusers、PyTorch CUDA、Pillow、现有 MMKGC pipeline、pytest、compileall、现有 offline evaluation。

---

## Controller 规则

执行时必须遵守用户确认的流程：

1. 主 Codex 会话作为 controller。
2. 每个 plan task 派一个 implementer subagent。
3. 每个 implementer 必须有明确写入范围、验证命令、禁止 commit。
4. 每个任务完成后派两个 reviewer：
   - spec reviewer：检查是否符合 `docs/superpowers/specs/2026-07-08-mkgw-sdxl-image-generation-design.zh.md` 和用户确认方案。
   - quality reviewer：检查代码质量、测试质量、职责边界、路径/模板/打包/GPU 风险。
5. reviewer 发现问题时，必须回给 implementer 修复。
6. 两个 reviewer 都 approve 后才标记任务完成。
7. 所有任务完成后派 final reviewer。
8. final reviewer approve 后，controller 运行完整验证命令并汇报证据。
9. 工作区可能有用户未提交改动，不要 revert 任何非本任务改动。
10. 不要自动 commit。

## 已确认范围

```text
image_generator_mode: sdxl-local
text_generator_mode_for_smoke: rule
sdxl_model_dir: SDXL/stable-diffusion-xl-base-1.0
sdxl_refiner: not used
missing_mode: both
limit: 5
candidate_budget: 5
generated_images: 25
width: 1024
height: 1024
steps: 20
seed: 20260708
output_run_name: sdxl_image_smoke_20260708
image_output_location: per entity generated_images/
image_call_log: per entity image_model_calls.jsonl
evaluation_scope: engineering validation only
```

## 文件与所有权

新增：

```text
MMKGC/mmkgc/adapters/sdxl.py
MMKGC/scripts/validate_sdxl_image_run.py
MMKGC/tests/test_sdxl_adapter.py
MMKGC/tests/test_sdxl_cli_integration.py
MMKGC/docs/evaluation/2026-07-08-sdxl-image-smoke-validation.zh.md
```

修改：

```text
MMKGC/mmkgc/adapters/__init__.py
MMKGC/mmkgc/adapters/base.py
MMKGC/mmkgc/stages/generation.py
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/io/writer.py
MMKGC/mmkgc/run.py
MMKGC/tests/test_pipeline_outputs.py
MMKGC/tests/test_stages.py
MMKGC/README.md
MMKGC/README.zh.md
```

只读：

```text
MMKGC/docs/superpowers/specs/2026-07-08-mkgw-sdxl-image-generation-design.zh.md
MMKGC/SDXL/stable-diffusion-xl-base-1.0/
MMKGC/datasets/MKG-W/
MMKGC/outputs/qwen_text_real_7b_20260708/
MMKGC/outputs/scale_500_both_20260707/
```

禁止修改或覆盖：

```text
MMKGC/SDXL/stable-diffusion-xl-base-1.0/
MMKGC/SDXL/stable-diffusion-xl-refiner-1.0/
MMKGC/outputs/qwen_text_real_7b_20260708/
MMKGC/outputs/scale_500_both_20260707/
```

## Task 1：SDXL adapter 与协议边界

**目标：** 建立不在单元测试中加载真实 SDXL 的 `ImageGenerator` 协议、artifact 数据结构和 SDXL adapter。

**写入范围：**

```text
MMKGC/mmkgc/adapters/base.py
MMKGC/mmkgc/adapters/sdxl.py
MMKGC/mmkgc/adapters/__init__.py
MMKGC/tests/test_sdxl_adapter.py
```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_adapter.py -v -p no:cacheprovider
```

**禁止 commit。**

- [ ] **Step 1：写失败测试，验证 SDXL 配置和 fake pipeline**

创建 `MMKGC/tests/test_sdxl_adapter.py`，覆盖：

```python
from pathlib import Path

from PIL import Image

from mmkgc.adapters.sdxl import SDXLConfig, SDXLImageGenerator


class FakeImageResult:
    def __init__(self, image):
        self.images = [image]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeImageResult(Image.new("RGB", (kwargs["width"], kwargs["height"]), "white"))


def test_sdxl_generator_uses_prompt_parameters_and_returns_png(tmp_path):
    fake_pipeline = FakePipeline()
    config = SDXLConfig(
        model_dir=tmp_path,
        width=64,
        height=64,
        steps=3,
        seed=123,
        guidance_scale=7.5,
        device="cpu",
    )
    generator = SDXLImageGenerator(config, pipeline_factory=lambda _: fake_pipeline)

    artifact = generator.generate(
        entity_key="Q1",
        candidate_id="image_001",
        positive_prompt="Ada Lovelace portrait, mathematician",
        negative_prompt="unsupported detail",
    )

    assert artifact.relative_image_path == "generated_images/image_001.png"
    assert artifact.relative_metadata_path == "generated_images/image_001.json"
    assert artifact.metadata["width"] == 64
    assert artifact.metadata["height"] == 64
    assert artifact.metadata["seed"] == 123
    assert artifact.model_call["modality"] == "image"
    assert artifact.model_call["status"] == "generated"
    assert fake_pipeline.calls[0]["prompt"] == "Ada Lovelace portrait, mathematician"
    assert fake_pipeline.calls[0]["negative_prompt"] == "unsupported detail"
    assert artifact.image_bytes.startswith(b"\\x89PNG")
```

运行验证命令。期望：失败，原因是 `mmkgc.adapters.sdxl` 尚不存在。

- [ ] **Step 2：实现协议与 artifact 数据结构**

修改 `MMKGC/mmkgc/adapters/base.py`，新增：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedImageArtifact:
    entity_key: str
    candidate_id: str
    relative_image_path: str
    relative_metadata_path: str
    image_bytes: bytes
    metadata: dict[str, Any]
    model_call: dict[str, Any]


class ImageGenerator(Protocol):
    """Protocol for optional real image generators used after image prompt planning."""

    def generate(
        self,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
    ) -> GeneratedImageArtifact:
        """Generate an image artifact from a planned prompt package."""
```

- [ ] **Step 3：实现 `mmkgc.adapters.sdxl`**

创建 `MMKGC/mmkgc/adapters/sdxl.py`。实现要求：

```python
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from mmkgc.adapters.base import GeneratedImageArtifact


@dataclass(frozen=True)
class SDXLConfig:
    model_dir: str | Path
    width: int = 1024
    height: int = 1024
    steps: int = 20
    seed: int = 20260708
    guidance_scale: float = 7.5
    device: str = "cuda"


class SDXLImageGenerator:
    def __init__(self, config: SDXLConfig, pipeline_factory: Callable[[SDXLConfig], Any] | None = None):
        self.config = config
        self._pipeline_factory = pipeline_factory or _load_diffusers_pipeline
        self._pipeline = None

    def generate(
        self,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
    ) -> GeneratedImageArtifact:
        started = time.perf_counter()
        relative_image_path = f"generated_images/{candidate_id}.png"
        relative_metadata_path = f"generated_images/{candidate_id}.json"
        pipeline = self._get_pipeline()
        try:
            kwargs = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": self.config.width,
                "height": self.config.height,
                "num_inference_steps": self.config.steps,
                "guidance_scale": self.config.guidance_scale,
            }
            _attach_generator(kwargs, self.config)
            result = pipeline(**kwargs)
            image = result.images[0]
            image_bytes = _png_bytes(image)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            metadata = self._metadata(
                entity_key,
                candidate_id,
                relative_image_path,
                relative_metadata_path,
                positive_prompt,
                negative_prompt,
                "generated",
                elapsed_ms,
            )
            model_call = dict(metadata)
            model_call["modality"] = "image"
            return GeneratedImageArtifact(
                entity_key=entity_key,
                candidate_id=candidate_id,
                relative_image_path=relative_image_path,
                relative_metadata_path=relative_metadata_path,
                image_bytes=image_bytes,
                metadata=metadata,
                model_call=model_call,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            metadata = self._metadata(
                entity_key,
                candidate_id,
                relative_image_path,
                relative_metadata_path,
                positive_prompt,
                negative_prompt,
                "failed_fallback",
                elapsed_ms,
                error=exc,
            )
            model_call = dict(metadata)
            model_call["modality"] = "image"
            return GeneratedImageArtifact(
                entity_key=entity_key,
                candidate_id=candidate_id,
                relative_image_path=relative_image_path,
                relative_metadata_path=relative_metadata_path,
                image_bytes=b"",
                metadata=metadata,
                model_call=model_call,
            )

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(self.config)
        return self._pipeline
```

完整实现还必须包含：

- `_load_diffusers_pipeline(config)`：校验 `model_dir` 存在，懒加载 `diffusers.StableDiffusionXLPipeline` 和 `torch`，使用 `torch.float16`，`local_files_only=True`，加载后 `.to(config.device)`。
- `_get_pipeline()` 的模型目录、Diffusers import、CUDA 初始化和模型加载错误不得被 candidate fallback 吞掉；这些错误应直接抛出，让 `sdxl-local` run 失败并暴露环境问题。
- `_attach_generator(kwargs, config)`：如 `torch` 可用，用 `torch.Generator(device=config.device).manual_seed(config.seed)`；CPU fake 测试允许不传 generator。
- `_png_bytes(image)`：把 Pillow image 保存为 PNG bytes。
- `_metadata(...)`：输出 `generator=sdxl-local`、`model_name=stable-diffusion-xl-base-1.0`、`model_path`、prompt、negative prompt、seed、width、height、steps、guidance scale、status、fallback_used、elapsed_ms、错误字段。

- [ ] **Step 4：导出 adapter**

修改 `MMKGC/mmkgc/adapters/__init__.py`，导出：

```python
GeneratedImageArtifact
ImageGenerator
SDXLConfig
SDXLImageGenerator
```

- [ ] **Step 5：运行 Task 1 验证**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_adapter.py -v -p no:cacheprovider
```

期望：全部通过。

## Task 2：Stage 4 与 pipeline 集成

**目标：** 让 Stage 4 在有 `ImageGenerator` 时为每个 image candidate 生成 artifact，并把 image model calls 合并到 pipeline 结果。

**写入范围：**

```text
MMKGC/mmkgc/stages/generation.py
MMKGC/mmkgc/pipeline.py
MMKGC/tests/test_sdxl_cli_integration.py
MMKGC/tests/test_stages.py
```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_cli_integration.py tests/test_stages.py -v -p no:cacheprovider
```

**禁止 commit。**

- [ ] **Step 1：写失败测试，验证 fake image generator 集成**

在 `MMKGC/tests/test_sdxl_cli_integration.py` 添加：

```python
from mmkgc.adapters import GeneratedImageArtifact
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.data import EntityRecord


class RecordingImageGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, entity_key, candidate_id, positive_prompt, negative_prompt):
        self.calls.append((entity_key, candidate_id, positive_prompt, negative_prompt))
        return GeneratedImageArtifact(
            entity_key=entity_key,
            candidate_id=candidate_id,
            relative_image_path=f"generated_images/{candidate_id}.png",
            relative_metadata_path=f"generated_images/{candidate_id}.json",
            image_bytes=b"png-bytes",
            metadata={
                "entity_key": entity_key,
                "candidate_id": candidate_id,
                "status": "generated",
                "generator": "sdxl-local",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "seed": 20260708,
            },
            model_call={
                "entity_key": entity_key,
                "candidate_id": candidate_id,
                "modality": "image",
                "status": "generated",
            },
        )


def test_pipeline_collects_sdxl_image_artifacts_and_model_calls():
    record = EntityRecord(
        entity_id=1,
        entity_key="Q1",
        label="Ada Lovelace",
        description="mathematician",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "mathematician",
                "label": "mathematician",
                "description": "a person whose work uses mathematics",
                "relation_id": 10,
                "relation": "occupation",
                "direction": "out",
            }
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "Ada Lovelace was a mathematician.", "image": None},
    )
    image_generator = RecordingImageGenerator()

    result = MMKGCPipeline(candidate_budget=2, image_generator=image_generator).run(record, "both")

    generated = result["relation_alternation_generation"]["relation_alternation_generation"]
    assert len(generated["image_candidates"]) == 2
    assert len(result["generated_image_artifacts"]) == 2
    assert [row["candidate_id"] for row in result["model_calls"]] == ["image_001", "image_002"]
    assert generated["image_candidates"][0]["generated_content"]["type"] == "image"
    assert generated["image_candidates"][0]["generated_content"]["image_path"] == "generated_images/image_001.png"
    assert generated["image_candidates"][0]["metadata"]["generator"] == "sdxl-local"
```

运行 Task 2 验证命令。期望：失败，原因是 pipeline 还不接受 `image_generator`。

- [ ] **Step 2：修改 Stage 4 函数签名**

修改 `MMKGC/mmkgc/stages/generation.py`：

- 引入 `ImageGenerator` 和 `GeneratedImageArtifact`。
- `_generate_modality_candidates(...)` 增加 `image_generator: ImageGenerator | None`。
- 返回值从 `tuple[list[dict[str, Any]], list[dict[str, Any]]]` 改为 `tuple[list[dict[str, Any]], list[dict[str, Any]], list[GeneratedImageArtifact]]`。
- `relation_alternation_generation(...)` 增加 `image_generator: ImageGenerator | None = None`，并返回 `generated_image_artifacts`。

image 分支生成 prompt 后，如果 `image_generator is None`，保持当前行为：

```python
base["generated_content"] = {"type": "image_prompt", "value": base["positive_prompt"]}
base["metadata"]["status"] = "planned"
```

如果 `image_generator` 存在，调用：

```python
artifact = image_generator.generate(
    entity_key=entity_key,
    candidate_id=base["candidate_id"],
    positive_prompt=base["positive_prompt"],
    negative_prompt=base["negative_prompt"],
)
```

成功时更新 candidate：

```python
if artifact.image_bytes:
    base["generated_content"] = {
        "type": "image",
        "value": base["positive_prompt"],
        "image_path": artifact.relative_image_path,
        "metadata_path": artifact.relative_metadata_path,
        "prompt": base["positive_prompt"],
        "negative_prompt": base["negative_prompt"],
        "width": artifact.metadata.get("width"),
        "height": artifact.metadata.get("height"),
        "steps": artifact.metadata.get("steps"),
        "seed": artifact.metadata.get("seed"),
    }
    base["metadata"].update(
        {
            "status": "generated",
            "generator": artifact.metadata.get("generator"),
            "model_name": artifact.metadata.get("model_name"),
            "model_path": artifact.metadata.get("model_path"),
            "fallback_used": False,
        }
    )
    image_artifacts.append(artifact)
```

失败 fallback 时：

```python
else:
    base["generated_content"] = {"type": "image_prompt", "value": base["positive_prompt"]}
    base["metadata"].update(
        {
            "status": "failed_fallback",
            "generator": artifact.metadata.get("generator"),
            "fallback_used": True,
            "error_type": artifact.metadata.get("error_type"),
            "error_message": artifact.metadata.get("error_message"),
        }
    )
```

无论成功或失败，`artifact.model_call` 都加入 `model_calls`。

- [ ] **Step 3：修改 pipeline 构造函数和返回值**

修改 `MMKGC/mmkgc/pipeline.py`：

- `MMKGCPipeline.__init__` 增加 `image_generator: ImageGenerator | None = None`。
- 调用 `relation_alternation_generation(...)` 时传入 `self.image_generator`。
- 从 generation 中取出 `generated_image_artifacts`。
- 返回值顶层增加：

```python
"generated_image_artifacts": generated_image_artifacts,
```

保留：

```python
"model_calls": model_calls,
```

不要把 `generated_image_artifacts` 放入 `final`。

- [ ] **Step 4：运行 Task 2 验证**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_cli_integration.py tests/test_stages.py -v -p no:cacheprovider
```

期望：全部通过。

## Task 3：RunWriter 保存 PNG、metadata 和 image call log

**目标：** 把 pipeline 返回的 image artifacts 写入每个实体目录的 `generated_images/`，并拆分写出 `image_model_calls.jsonl`。

**写入范围：**

```text
MMKGC/mmkgc/io/writer.py
MMKGC/tests/test_pipeline_outputs.py
MMKGC/tests/test_sdxl_cli_integration.py
```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_pipeline_outputs.py tests/test_sdxl_cli_integration.py -v -p no:cacheprovider
```

**禁止 commit。**

- [ ] **Step 1：写失败测试，验证 writer 写图片产物**

在 `MMKGC/tests/test_sdxl_cli_integration.py` 添加：

```python
import json

from mmkgc.io import RunWriter


def test_writer_persists_generated_images_and_image_model_calls(tmp_path):
    record = EntityRecord(
        entity_id=1,
        entity_key="Q1",
        label="Ada Lovelace",
        description="mathematician",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "mathematician",
                "label": "mathematician",
                "description": "a person whose work uses mathematics",
                "relation_id": 10,
                "relation": "occupation",
                "direction": "out",
            }
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "Ada Lovelace was a mathematician.", "image": None},
    )
    image_generator = RecordingImageGenerator()
    result = MMKGCPipeline(candidate_budget=1, image_generator=image_generator).run(record, "image")

    writer = RunWriter(tmp_path / "outputs", "sdxl_writer", {"image_generator": "sdxl-local"})
    writer.write_entity(result)

    entity_dir = writer.run_dir / "entities" / "Q1"
    image_path = entity_dir / "generated_images" / "image_001.png"
    metadata_path = entity_dir / "generated_images" / "image_001.json"
    image_calls_path = entity_dir / "image_model_calls.jsonl"

    assert image_path.read_bytes() == b"png-bytes"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["candidate_id"] == "image_001"
    rows = [json.loads(line) for line in image_calls_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["modality"] == "image"
```

运行 Task 3 验证命令。期望：失败，原因是 writer 尚不保存 image artifacts。

- [ ] **Step 2：实现 image artifact 写入**

修改 `MMKGC/mmkgc/io/writer.py`：

- 在 `write_entity` 创建 `entity_dir` 后，读取：

```python
image_artifacts = _json_ready(result.get("generated_image_artifacts", []))
```

- 对每个 artifact：

```python
image_path = entity_dir / artifact["relative_image_path"]
metadata_path = entity_dir / artifact["relative_metadata_path"]
image_path.parent.mkdir(parents=True, exist_ok=True)
image_path.write_bytes(artifact["image_bytes"])
write_json(metadata_path, artifact["metadata"])
```

- 对空 `image_bytes` 的失败 artifact，不写 PNG，但仍可写 metadata。
- `relative_image_path` 和 `relative_metadata_path` 必须保持在实体目录内。实现一个 `_safe_entity_relative_path(entity_dir, relative_path)`，拒绝绝对路径和包含 `..` 的路径。

- [ ] **Step 3：拆分写 image model calls**

在 `write_entity` 中，当前 `model_calls.jsonl` 保持写全部 model calls。新增：

```python
image_model_calls = [row for row in model_calls if row.get("modality") == "image"]
if image_model_calls:
    _write_jsonl(entity_dir / "image_model_calls.jsonl", image_model_calls)
```

规则 pipeline 没有 image model calls 时不得写 `image_model_calls.jsonl`。

- [ ] **Step 4：运行 Task 3 验证**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_pipeline_outputs.py tests/test_sdxl_cli_integration.py -v -p no:cacheprovider
```

期望：全部通过。

## Task 4：CLI 参数、SDXL 构造和 run 配置

**目标：** 通过 `python -m mmkgc.run --image-generator sdxl-local ...` 构造真实 SDXL generator，并在 `config.json` 中记录参数。

**写入范围：**

```text
MMKGC/mmkgc/run.py
MMKGC/tests/test_sdxl_cli_integration.py
```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_cli_integration.py tests/test_qwen_cli_integration.py -v -p no:cacheprovider
```

**禁止 commit。**

- [ ] **Step 1：写失败测试，验证 CLI help 和 safe config**

在 `MMKGC/tests/test_sdxl_cli_integration.py` 添加 CLI 测试，使用 monkeypatch fake 掉真实 `SDXLImageGenerator`：

```python
from pathlib import Path

from mmkgc import run as run_module


def test_cli_help_includes_sdxl_options(capsys):
    parser = run_module.build_parser()
    help_text = parser.format_help()
    assert "--image-generator" in help_text
    assert "--sdxl-model-dir" in help_text
    assert "--sdxl-width" in help_text
    assert "--sdxl-height" in help_text
    assert "--sdxl-steps" in help_text
    assert "--sdxl-seed" in help_text


def test_cli_sdxl_local_writes_config_without_loading_real_model(tmp_path, monkeypatch):
    class FakeStore:
        def __init__(self, dataset_dir):
            self.dataset_dir = dataset_dir

        def select_records(self, missing_mode, limit):
            return [
                EntityRecord(
                    entity_id=1,
                    entity_key="Q1",
                    label="Ada Lovelace",
                    description="mathematician",
                    neighbors=[
                        {
                            "entity_id": 2,
                            "entity_key": "mathematician",
                            "label": "mathematician",
                            "description": "a person whose work uses mathematics",
                            "relation_id": 10,
                            "relation": "occupation",
                            "direction": "out",
                        }
                    ],
                    existing_modalities={"text": None, "image": None},
                    ground_truth_modalities={"text": "Ada Lovelace was a mathematician.", "image": None},
                )
            ]

    monkeypatch.setattr(run_module, "MKGWDataStore", FakeStore)
    monkeypatch.setattr(run_module, "validate_dataset_dir", lambda path: Path(path))
    monkeypatch.setattr(run_module, "validate_sdxl_model_dir", lambda path: Path(path))
    monkeypatch.setattr(run_module, "SDXLImageGenerator", lambda config: RecordingImageGenerator())

    exit_code = run_module.main(
        [
            "--dataset-dir",
            "datasets/MKG-W",
            "--missing-mode",
            "image",
            "--limit",
            "1",
            "--candidate-budget",
            "1",
            "--image-generator",
            "sdxl-local",
            "--sdxl-model-dir",
            "SDXL/stable-diffusion-xl-base-1.0",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-name",
            "sdxl_cli",
        ]
    )

    assert exit_code == 0
    config = json.loads((tmp_path / "outputs" / "sdxl_cli" / "config.json").read_text(encoding="utf-8"))
    assert config["image_generator"] == "sdxl-local"
    assert config["sdxl_model_dir"] == "SDXL/stable-diffusion-xl-base-1.0"
    assert config["sdxl_width"] == 1024
    assert config["sdxl_height"] == 1024
    assert config["sdxl_steps"] == 20
    assert config["sdxl_seed"] == 20260708
```

运行 Task 4 验证命令。期望：失败，原因是 CLI 尚未增加 SDXL 参数。

- [ ] **Step 2：修改 CLI parser**

修改 `MMKGC/mmkgc/run.py`：

```python
parser.add_argument("--image-generator", choices=["rule", "sdxl-local"], default="rule")
parser.add_argument("--sdxl-model-dir", default="SDXL/stable-diffusion-xl-base-1.0")
parser.add_argument("--sdxl-width", type=int, default=1024)
parser.add_argument("--sdxl-height", type=int, default=1024)
parser.add_argument("--sdxl-steps", type=int, default=20)
parser.add_argument("--sdxl-seed", type=int, default=20260708)
parser.add_argument("--sdxl-guidance-scale", type=float, default=7.5)
parser.add_argument("--sdxl-device", default="cuda")
```

- [ ] **Step 3：新增 SDXL 模型目录预检**

在 `MMKGC/mmkgc/run.py` 中新增：

```python
def validate_sdxl_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"sdxl_model_dir does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"sdxl_model_dir is not a directory: {path}")
    model_index = path / "model_index.json"
    if not model_index.exists():
        raise FileNotFoundError(f"sdxl_model_dir is missing model_index.json: {model_index}")
    return path
```

`sdxl-local` 模式必须在创建 `RunWriter` 之前调用该预检，避免输出目录已创建后才发现模型路径错误。

- [ ] **Step 4：构造 SDXL generator**

在 `main()` 中：

```python
image_generator = None
if args.image_generator == "sdxl-local":
    sdxl_model_dir = validate_sdxl_model_dir(args.sdxl_model_dir)
    sdxl_config = SDXLConfig(
        model_dir=sdxl_model_dir,
        width=args.sdxl_width,
        height=args.sdxl_height,
        steps=args.sdxl_steps,
        seed=args.sdxl_seed,
        guidance_scale=args.sdxl_guidance_scale,
        device=args.sdxl_device,
    )
    image_generator = SDXLImageGenerator(sdxl_config)
```

传入 pipeline：

```python
pipeline = MMKGCPipeline(
    candidate_budget=args.candidate_budget,
    text_generator=text_generator,
    image_generator=image_generator,
)
```

`config` 中记录：

```python
"image_generator": args.image_generator,
"sdxl_model_dir": args.sdxl_model_dir if args.image_generator == "sdxl-local" else None,
"sdxl_width": args.sdxl_width if args.image_generator == "sdxl-local" else None,
"sdxl_height": args.sdxl_height if args.image_generator == "sdxl-local" else None,
"sdxl_steps": args.sdxl_steps if args.image_generator == "sdxl-local" else None,
"sdxl_seed": args.sdxl_seed if args.image_generator == "sdxl-local" else None,
"sdxl_guidance_scale": args.sdxl_guidance_scale if args.image_generator == "sdxl-local" else None,
"sdxl_device": args.sdxl_device if args.image_generator == "sdxl-local" else None,
```

- [ ] **Step 5：运行 Task 4 验证**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_cli_integration.py tests/test_qwen_cli_integration.py -v -p no:cacheprovider
```

期望：全部通过。

## Task 5：SDXL run 验收脚本与文档

**目标：** 提供工程验收脚本和 README 说明，确保 25 张真实图片可以被自动检查。

**写入范围：**

```text
MMKGC/scripts/validate_sdxl_image_run.py
MMKGC/tests/test_sdxl_cli_integration.py
MMKGC/README.md
MMKGC/README.zh.md
```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_cli_integration.py -v -p no:cacheprovider
```

**禁止 commit。**

- [ ] **Step 1：写失败测试，验证验收脚本函数**

在 `MMKGC/tests/test_sdxl_cli_integration.py` 添加：

```python
from PIL import Image

from scripts.validate_sdxl_image_run import validate_run


def test_validate_sdxl_image_run_accepts_expected_artifacts(tmp_path):
    run_dir = tmp_path / "outputs" / "sdxl"
    entity_dir = run_dir / "entities" / "Q1"
    images_dir = entity_dir / "generated_images"
    images_dir.mkdir(parents=True)
    (run_dir / "metrics_summary.json").write_text(
        json.dumps({"processed_entities": 1, "failed_entities": 0}),
        encoding="utf-8",
    )
    image_path = images_dir / "image_001.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    (images_dir / "image_001.json").write_text(
        json.dumps({"candidate_id": "image_001", "width": 32, "height": 32}),
        encoding="utf-8",
    )
    (entity_dir / "image_model_calls.jsonl").write_text(
        json.dumps({"candidate_id": "image_001", "modality": "image", "status": "generated"}) + "\\n",
        encoding="utf-8",
    )

    summary = validate_run(run_dir, expected_entities=1, expected_images_per_entity=1, expected_width=32, expected_height=32)

    assert summary["entities_checked"] == 1
    assert summary["images_checked"] == 1
```

运行 Task 5 验证命令。期望：失败，原因是脚本尚不存在。

- [ ] **Step 2：创建验收脚本**

创建 `MMKGC/scripts/validate_sdxl_image_run.py`，实现：

- `validate_run(run_dir, expected_entities, expected_images_per_entity, expected_width, expected_height) -> dict`
- CLI 参数：

```text
--run-dir
--expected-entities
--expected-images-per-entity
--expected-width
--expected-height
```

验收逻辑：

```python
metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
if metrics.get("processed_entities") != expected_entities:
    raise AssertionError(...)
if metrics.get("failed_entities") != 0:
    raise AssertionError(...)
for entity_dir in sorted((run_dir / "entities").iterdir()):
    image_paths = sorted((entity_dir / "generated_images").glob("*.png"))
    if len(image_paths) != expected_images_per_entity:
        raise AssertionError(...)
    calls_path = entity_dir / "image_model_calls.jsonl"
    if not calls_path.exists():
        raise AssertionError(...)
    for image_path in image_paths:
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists():
            raise AssertionError(...)
        with Image.open(image_path) as image:
            if image.size != (expected_width, expected_height):
                raise AssertionError(...)
```

CLI 成功时输出 JSON：

```json
{
  "ok": true,
  "entities_checked": 5,
  "images_checked": 25
}
```

- [ ] **Step 3：更新 README**

修改 `MMKGC/README.zh.md` 和 `MMKGC/README.md`，新增 SDXL image smoke 小节。中文 README 至少包含：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator rule \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260708 \
  --output-root outputs \
  --run-name sdxl_image_smoke_20260708

/home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --expected-entities 5 \
  --expected-images-per-entity 5 \
  --expected-width 1024 \
  --expected-height 1024
```

说明本阶段只做工程验收，不使用 CLIP/BLIP 评价质量。

- [ ] **Step 4：运行 Task 5 验证**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_sdxl_cli_integration.py -v -p no:cacheprovider
```

期望：全部通过。

## Task 6：真实 SDXL smoke run 与报告

**目标：** 在用户已停止 Qwen/vLLM 且 GPU 可用的情况下，运行 5 实体、25 图片真实 SDXL 烟测，生成验收报告。

**写入范围：**

```text
MMKGC/outputs/sdxl_image_smoke_20260708/
MMKGC/docs/evaluation/2026-07-08-sdxl-image-smoke-validation.zh.md
```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/mmkgc-sdxl-compileall-pycache /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

**禁止 commit。**

- [ ] **Step 1：预检 SDXL 模型与 GPU**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test -d SDXL/stable-diffusion-xl-base-1.0
test -f SDXL/stable-diffusion-xl-base-1.0/model_index.json
nvidia-smi
```

期望：模型目录存在，GPU 可见。若 GPU 不可见或显存被 Qwen 占用，报告 `BLOCKED`，提醒用户停止 Qwen/vLLM 后继续。

- [ ] **Step 2：确认输出目录不存在**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test ! -e outputs/sdxl_image_smoke_20260708
```

如果目录已存在，报告 `BLOCKED`，不覆盖。

- [ ] **Step 3：运行真实 SDXL smoke**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator rule \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260708 \
  --output-root outputs \
  --run-name sdxl_image_smoke_20260708
```

期望输出 JSON 中：

```json
{
  "run_dir": "outputs/sdxl_image_smoke_20260708",
  "processed_entities": 5,
  "failed_entities": 0
}
```

- [ ] **Step 4：运行图片工程验收**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --expected-entities 5 \
  --expected-images-per-entity 5 \
  --expected-width 1024 \
  --expected-height 1024
```

期望输出 JSON：

```json
{
  "ok": true,
  "entities_checked": 5,
  "images_checked": 25
}
```

- [ ] **Step 5：运行 offline evaluation**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --output-dir outputs/sdxl_image_smoke_20260708/evaluation \
  --overwrite
```

期望：`evaluation/evaluation_summary.json` 存在，且 `failed_entities=0`。

- [ ] **Step 6：写验收报告**

创建 `MMKGC/docs/evaluation/2026-07-08-sdxl-image-smoke-validation.zh.md`，包含：

- 运行命令。
- 模型路径：`SDXL/stable-diffusion-xl-base-1.0`。
- 参数：1024x1024、20 steps、seed 20260708。
- `metrics_summary.json` 的 processed/failed 结果。
- 图片验收脚本输出。
- evaluation 执行结果。
- 限制：本阶段只证明真实图片生成链路和工程产物完整，不证明图文语义质量。

- [ ] **Step 7：运行完整验证**

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/mmkgc-sdxl-compileall-pycache /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

期望：pytest 全部通过，compileall exit code 为 `0`。

## Final Review 与最终验收

所有 task 完成且每个 task 的 spec reviewer、quality reviewer 都 approve 后，controller 派 final reviewer 检查完整变更。

final reviewer 需要确认：

- spec 中已确认的 A/A/B/A/A/A 方案全部落地。
- 默认 `rule` 模式不生成图片，不写 `image_model_calls.jsonl`。
- `sdxl-local` 只使用 `MMKGC/SDXL/stable-diffusion-xl-base-1.0`。
- 图片保存在每个实体目录下的 `generated_images/`。
- 每实体 5 张图，总计 25 张图。
- 每张图有 sidecar metadata。
- `image_model_calls.jsonl` 存在且包含 image 调用记录。
- evaluation 可以继续运行。
- 没有自动 commit。

controller 最终运行并汇报：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/mmkgc-sdxl-final-compileall-pycache /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py --run-dir outputs/sdxl_image_smoke_20260708 --expected-entities 5 --expected-images-per-entity 5 --expected-width 1024 --expected-height 1024
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/sdxl_image_smoke_20260708 --output-dir outputs/sdxl_image_smoke_20260708/evaluation --overwrite
```

最终汇报必须包含：

- 测试通过数量。
- compileall 结果。
- SDXL smoke run 的 processed/failed。
- 图片验收的 entities/images 数量。
- evaluation 的 evaluated/failed。
- 真实图像质量尚未由 CLIP/BLIP/VLM 评价的限制说明。
