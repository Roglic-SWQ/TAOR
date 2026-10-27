# MKG-W 本地 SDXL 图像生成接入设计文档

## 目标

在已经完成离线 MVP、evaluation 校准、500 实体扩大运行和本地 Qwen 文本生成烟测之后，接入真实图像生成的第一版：使用本地 `Stable Diffusion XL 1.0 Base`，把现有 Stage 4 的 `image_candidates` 从规则 image prompt 扩展为可落盘的真实图片产物。

本阶段只做小样本工程烟测，目标是回答：

- 现有六阶段 pipeline 能否在不破坏规则 baseline 和 Qwen text 链路的前提下调用本地 SDXL 生成图片。
- 每个 image candidate 的 prompt、negative prompt、seed、尺寸、steps、模型路径、耗时和失败信息是否可追溯。
- 生成图片能否保存到每个实体目录下，并被后续验收脚本检查。
- 现有 Stage 5 ranking 和 offline evaluation 能否继续消费 image candidates。

## 已确认方案

brainstorming 阶段已确认：

- 运行方式：先停掉 Qwen，只跑 SDXL，避免 GPU 显存冲突。
- 图像模型：本地 `Stable Diffusion XL 1.0 Base`。
- 模型路径：`MMKGC/SDXL/stable-diffusion-xl-base-1.0`。
- Refiner：第一版不接入。
- 运行规模：`missing_mode=both`、`limit=5`、`candidate_budget=5`。
- 生成粒度：每个实体 5 张图，共 25 张真实图片。
- 图片参数：`1024x1024`、`20` steps、固定 seed。
- 输出位置：图片保存在每个实体目录下。
- 调用记录：每个实体目录写 `image_model_calls.jsonl`，并保留通用 `model_calls.jsonl`。
- 图片 metadata：每张图片写同名 JSON sidecar，记录 prompt、negative prompt、candidate id、seed、尺寸、steps、generator、模型路径和生成状态。
- 评价范围：第一版只做工程验收，不接 CLIP、BLIP、VLM judge 或图文一致性真实模型。
- 不自动 commit。

## 非目标

本阶段不做以下事情：

- 不接 SDXL Refiner、SD3.5、Flux 或其它图像模型。
- 不接 CLIP、BLIP、Sentence-BERT、VLM judge 或 LLM judge。
- 不修改 Stage 0、Stage 1、Stage 2、Stage 3 的规则。
- 不重写 Stage 5 ranking 公式。
- 不修改 offline evaluation 的打分公式和 calibration 规则。
- 不跑 500 实体或全量 MKG-W。
- 不让 `mmkgc.run` 自动启动或停止 Qwen/vLLM。
- 不依赖 `MMKGC/` 外部的 SDXL 模型路径。
- 不覆盖已有输出目录。
- 不自动 commit。

## 自包含目录结构

本阶段使用以下 MMKGC 内部路径：

```text
MMKGC/
  SDXL/
    stable-diffusion-xl-base-1.0/
      model_index.json
      scheduler/
      text_encoder/
      text_encoder_2/
      tokenizer/
      tokenizer_2/
      unet/
      vae/
      other SDXL Base files
    stable-diffusion-xl-refiner-1.0/
      unused in this stage
  datasets/
    MKG-W/
  outputs/
```

实施阶段只校验 `SDXL/stable-diffusion-xl-base-1.0` 的关键文件存在，不读取 `MMKGC/` 外部模型目录。

## 架构设计

现有 pipeline 的 Stage 4 已经通过 `ImagePromptGenerator` 生成 `image_candidates`。本阶段不替换这层 prompt planning，而是在 prompt 生成后增加一个可选的真实图像生成边界：

```text
EntityRecord
  -> Stage 0-3
  -> Stage 4 image prompt candidates
  -> optional SDXL image generator
  -> image candidates with generated image metadata
  -> Stage 5 ranking
  -> RunWriter writes JSON + PNG + sidecar metadata
```

新增 `ImageGenerator` 协议，与 `TextGenerator` 和 `ImagePromptGenerator` 并列。默认规则模式不提供 `ImageGenerator`，因此现有行为保持不变。只有 CLI 参数 `--image-generator sdxl-local` 打开时，Stage 4 才会把每个 image candidate 的 `positive_prompt` 和 `negative_prompt` 交给 SDXL。

为了避免 pipeline 直接依赖实体输出目录，SDXL adapter 返回内存中的图片 artifact。artifact 包含图片 bytes、推荐相对路径、metadata 和 model call。`RunWriter` 负责把 artifact 写入对应实体目录：

```text
outputs/<run_name>/entities/<entity_key>/
  0_coarse_relation_filtering.json
  1_knowledge_modeling.json
  2_graph_constrained_reasoning.json
  3_fine_grained_relation_selection.json
  4_relation_alternation_generation.json
  5_kg_aware_pair_ranking.json
  final.json
  model_calls.jsonl
  image_model_calls.jsonl
  generated_images/
    image_001.png
    image_001.json
    image_002.png
    image_002.json
    image_003.png
    image_003.json
    image_004.png
    image_004.json
    image_005.png
    image_005.json
```

## Candidate 输出结构

规则模式下，image candidate 保持当前结构：

```json
{
  "candidate_id": "image_001",
  "generated_content": {
    "type": "image_prompt",
    "value": "..."
  },
  "metadata": {
    "status": "planned"
  }
}
```

`sdxl-local` 模式下，image candidate 仍保留 `positive_prompt` 和 `negative_prompt`，并扩展 `generated_content`：

```json
{
  "candidate_id": "image_001",
  "positive_prompt": "...",
  "negative_prompt": "...",
  "generated_content": {
    "type": "image",
    "value": "...",
    "image_path": "generated_images/image_001.png",
    "metadata_path": "generated_images/image_001.json",
    "prompt": "...",
    "negative_prompt": "...",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "seed": 20260708
  },
  "metadata": {
    "status": "generated",
    "generator": "sdxl-local",
    "model_name": "stable-diffusion-xl-base-1.0",
    "model_path": "SDXL/stable-diffusion-xl-base-1.0",
    "fallback_used": false
  }
}
```

`generated_content.value` 继续保存 prompt 文本，保证 Stage 5 ranking 和 offline evaluation 的文本化逻辑仍可工作。

## 失败策略

如果单个 image candidate 生成失败：

- 不中断整个 run。
- 该 candidate 仍保留原始 image prompt。
- `generated_content.type` 保持或回退为 `image_prompt`。
- `metadata.status` 写为 `failed_fallback`。
- `metadata.fallback_used` 写为 `true`。
- `metadata.error_type` 和 `metadata.error_message` 记录失败原因。
- `image_model_calls.jsonl` 记录本次失败调用。

如果 SDXL 模型目录不存在、Diffusers 无法导入、GPU 不可用或 pipeline 初始化失败：

- `sdxl-local` 模式启动时直接报错。
- 不创建半成品 run 作为成功产物。
- 错误信息必须包含缺失路径或缺失依赖名称。

## CLI 设计

默认行为保持规则模式：

```bash
python -m mmkgc.run
```

新增参数：

```text
--image-generator {rule,sdxl-local}
--sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0
--sdxl-width 1024
--sdxl-height 1024
--sdxl-steps 20
--sdxl-seed 20260708
--sdxl-guidance-scale 7.5
--sdxl-device cuda
```

第一版烟测命令：

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

本命令要求用户已停止 Qwen/vLLM 服务，或至少保证 SDXL 可用 GPU 显存充足。

## 工程验收

新增验收脚本：

```text
MMKGC/scripts/validate_sdxl_image_run.py
```

验收命令：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --expected-entities 5 \
  --expected-images-per-entity 5 \
  --expected-width 1024 \
  --expected-height 1024
```

验收脚本需要检查：

- run 目录存在。
- `metrics_summary.json` 中 `processed_entities=5`、`failed_entities=0`。
- 每个实体目录有 5 张 PNG。
- 每张 PNG 可被 Pillow 打开。
- 每张 PNG 尺寸为 `1024x1024`。
- 每张 PNG 有同名 JSON metadata。
- 每个实体目录有 `image_model_calls.jsonl`。
- `image_model_calls.jsonl` 中每个实体至少 5 条 image 调用记录。
- 原 offline evaluation 可以继续跑通。

## 成功标准

本阶段完成时必须满足：

- `rule` 默认模式测试仍通过。
- `qwen-local` 文本生成相关单元测试仍通过。
- `sdxl-local` 代码路径有 fake adapter 单元测试覆盖，不要求单元测试加载真实 SDXL。
- 真实 SDXL smoke run 生成 5 个实体、25 张 `1024x1024` PNG。
- 每张图片有 sidecar metadata。
- 每个实体目录有 `image_model_calls.jsonl`。
- 运行配置记录 SDXL 参数和模型路径，且路径在 `MMKGC/` 内。
- `scripts/validate_sdxl_image_run.py` 验收通过。
- offline evaluation 对该 run 执行成功。
- 不自动 commit。
