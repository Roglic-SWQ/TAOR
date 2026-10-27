# MKG-W Qwen-VL Judge + BLIP/ITM 图像评价设计文档

## 目标

阶段 8 的目标是在阶段 7 已生成的真实 PNG 上新增两类更强的图像评价模型：

- Qwen2-VL VLM Judge：对每张图做开放式、可解释的视觉质量判断。
- BLIP/ITM：对每张图与 prompt、generated text、KG evidence 做图文匹配评分。

本阶段回答的问题是：

```text
CLIP 之外，能否用 VLM judge 和 BLIP/ITM 对真实 PNG 做更可解释、更接近视觉理解的评价？
```

阶段 8 不重新生成 text 或 image，不扩大数据规模，不替换现有 offline evaluation 和 CLIP image quality。它只在阶段 7 的 5 实体、25 张 PNG 上补充评价模型产物，为后续 50、100 或 500 实体联合真实模型实验建立更强的图像质量验收标准。

## 已确认方案

brainstorming 阶段已确认：

- 阶段名称：Qwen-VL Judge + BLIP/ITM 双评价模型。
- 输入 run 固定为：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709
```

- 输入规模：
  - 5 个实体。
  - 25 张 SDXL PNG。
- 不重新运行 `mmkgc.run`。
- 不重新生成 text candidates。
- 不重新生成 PNG。
- 不扩大到 50、100 或 500 实体。
- 保留阶段 6/7 的 CLIP 评价产物，新增评价结果与 CLIP 并排展示。
- Qwen2-VL Judge 使用本地 vLLM OpenAI-compatible 服务：
  - `base_url = http://localhost:12345/v1`
  - `api_key = token-abc123`
  - `model_name = Qwen/Qwen2-VL-7B-Instruct`
- Qwen2-VL Judge 输出粒度：
  - 每张图片 1 条 judge 结果。
  - 25 张图片共 25 条 image-level judge。
- Qwen2-VL Judge 评分范围：
  - `0-100`
- Qwen2-VL Judge 评分字段：

```text
entity_match_score
prompt_alignment_score
kg_evidence_score
artifact_score
overall_judge_score
rationale
```

- BLIP/ITM 使用本地权重目录：

```text
blip-itm-large-coco
```

- BLIP/ITM 评价三类文本目标：
  - image prompt。
  - generated text。
  - KG evidence summary。
- BLIP/ITM 输出字段：

```text
blip_prompt_itm_score
blip_text_itm_score
blip_evidence_itm_score
blip_overall_itm_score
```

- 严格成功标准：
  - Qwen-VL Judge 必须 25/25 成功。
  - BLIP/ITM 必须 25/25 成功。
  - 任一模型全 fallback、部分失败或 parse failure，都不能算阶段 8 完成。

## 非目标

本阶段不做以下事情：

- 不新增生成模型。
- 不接 SDXL Refiner、Stable Diffusion 3.5、Flux 或其它图像生成模型。
- 不扩大真实模型 run。
- 不重跑阶段 7 joint smoke。
- 不修改 Qwen text generation prompt。
- 不修改 SDXL image generation prompt。
- 不修改 CLIP scoring 公式。
- 不把 Qwen-VL、BLIP 或 CLIP 分数融合进现有 calibrated offline evaluation score。
- 不自动 commit。
- 不把 `blip-itm-large-coco/`、`Qwen/`、`SDXL/`、`CLIP/`、`datasets/` 或 `outputs/` 加入 git。

## 运行前置条件

### 输入 run

阶段 7 run 必须存在：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709/
```

并且必须满足：

```text
processed_entities = 5
failed_entities = 0
generated PNG count = 25
evaluation/clip_image_quality_summary.json exists
```

### Qwen2-VL Judge 服务

用户需要保持本地 Qwen/vLLM 服务可用。服务不由 `mmkgc.evaluate` 管理生命周期。

第一版只支持 OpenAI-compatible HTTP 调用，不在 evaluation 进程中直接加载 Qwen 权重。

本地运行配置复用：

```text
configs/mkgw-knowledge-extraction.yaml
```

该文件是 gitignored 的本机配置。阶段 8 只读取其中的 `lmm.model_name`、`lmm.base_url`、`lmm.api_key`、`inference.max_token` 和 `inference.temperature`。

### BLIP/ITM 权重

本地 BLIP/ITM 权重目录固定为：

```text
blip-itm-large-coco/
```

该目录必须位于 `MMKGC/` 项目根目录内，且至少包含：

```text
config.json
pytorch_model.bin
preprocessor_config.json
tokenizer.json
tokenizer_config.json
vocab.txt
```

`config.json` 的 architecture 应为：

```text
BlipForImageTextRetrieval
```

阶段 8 应把 `blip-itm-large-coco/` 加入 `.gitignore`。

### 依赖

当前本地环境已具备阶段 8 所需基础依赖：

```text
transformers = installed
torch = installed
PIL = installed
requests = installed
```

阶段 8 不允许自动联网下载模型权重。BLIP/ITM 必须从本地 `blip-itm-large-coco/` 加载。

## 数据流

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709
  -> existing offline evaluation
  -> existing CLIP image quality evaluation
  -> Qwen-VL Judge image-level evaluation
  -> BLIP/ITM image-text matching evaluation
  -> image quality model comparison summary
  -> Stage 8 validation report
```

每个 image candidate 使用同一套文本目标：

```text
prompt_text          = image candidate positive prompt
candidate_text       = paired or best available generated text candidate
evidence_summary     = entity profile + KG relation facts + visual constraints
```

`prompt_text`、`candidate_text`、`evidence_summary` 的构造应复用或抽取现有 `mmkgc.evaluation.image_quality` 中的逻辑，避免 CLIP、BLIP、Qwen-VL 三套评价器各自拼文本。

## Qwen-VL Judge 设计

### Provider

新增 provider 名称：

```text
qwen-vl-judge
```

CLI 通过显式参数启用，不作为默认行为。

### 输入

每张图片的 judge prompt 至少包含：

- entity key。
- entity label。
- entity type。
- image candidate id。
- image prompt。
- generated text candidate。
- KG evidence summary。
- 图像文件。

图片通过 OpenAI-compatible message 的 `image_url` content 传入。第一版使用本地图片 base64 data URL，避免要求服务能访问本地文件路径。

### 输出 schema

Qwen-VL Judge 必须返回可解析 JSON object。每个 image-level result 包含：

```json
{
  "entity_key": "Q123",
  "image_candidate_id": "image_001",
  "image_path": "generated_images/image_001.png",
  "provider": "qwen-vl-judge",
  "model_name": "Qwen/Qwen2-VL-7B-Instruct",
  "status": "judged",
  "fallback_used": false,
  "entity_match_score": 0,
  "prompt_alignment_score": 0,
  "kg_evidence_score": 0,
  "artifact_score": 0,
  "overall_judge_score": 0,
  "rationale": "简短中文理由",
  "latency_ms": 0,
  "raw_response": {}
}
```

评分字段必须满足：

```text
0 <= score <= 100
```

`rationale` 必须是非空字符串。

`overall_judge_score` 第一版保存 Qwen-VL 自己给出的综合判断，不强制由四个分项公式覆盖。报告中同时记录四个分项均值，便于后续校准。

### 错误处理

阶段 8 严格成功标准下：

- HTTP error 不 fallback 为成功。
- JSON parse failure 不 fallback 为成功。
- 缺字段不 fallback 为成功。
- score 越界不 fallback 为成功。
- 空 rationale 不 fallback 为成功。

这些情况应记录为 failed judge，并让真实 smoke 验收失败。

## BLIP/ITM 设计

### Provider

新增 provider 名称：

```text
blip-itm-local
```

CLI 通过显式参数启用，不作为默认行为。

### 模型加载

第一版使用 `transformers` 从本地目录加载：

```python
BlipProcessor.from_pretrained("blip-itm-large-coco", local_files_only=True)
BlipForImageTextRetrieval.from_pretrained("blip-itm-large-coco", local_files_only=True)
```

模型目录必须在项目根目录内，拒绝外部路径、`..` traversal 和 symlink escape。

### 输入

每张图片分别与三类文本目标计算 ITM 分数：

```text
prompt_text
candidate_text
evidence_summary_text
```

如果某个文本目标为空，该目标分数记为 `null`；但在阶段 7 joint smoke 输入中，严格验收要求 25 张图片三类目标都存在。

### 输出 schema

每个 image-level result 包含：

```json
{
  "entity_key": "Q123",
  "image_candidate_id": "image_001",
  "image_path": "generated_images/image_001.png",
  "provider": "blip-itm-local",
  "model_dir": "blip-itm-large-coco",
  "status": "scored",
  "fallback_used": false,
  "blip_prompt_itm_score": 0,
  "blip_text_itm_score": 0,
  "blip_evidence_itm_score": 0,
  "blip_overall_itm_score": 0,
  "text_targets": {
    "prompt_text": "portrait of Ada Lovelace as a mathematician",
    "candidate_text": "Ada Lovelace was a mathematician.",
    "evidence_summary_text": "Entity: Ada Lovelace. Type: person. Relations: occupation: mathematician. Visual requirements: portrait, mathematician."
  }
}
```

分数范围统一为 `0-100`。`blip_overall_itm_score` 是可用三类分数的算术平均。

### 错误处理

阶段 8 严格成功标准下：

- 模型目录不存在，fail-fast。
- 模型目录在项目根之外，fail-fast。
- 权重缺失，fail-fast。
- `transformers` 无法导入，fail-fast。
- 任一图片评分失败，本阶段 smoke 失败。

## CLI 设计

扩展现有 `python -m mmkgc.evaluate`。

保留已有参数：

```text
--image-quality-evaluator {none,clip-local}
```

新增两个独立开关，避免把 CLIP、BLIP、Qwen-VL 混成一个互斥选项：

```text
--blip-itm-evaluator {none,blip-itm-local}
--blip-model-dir blip-itm-large-coco
--blip-device cuda:1

--vlm-judge-evaluator {none,qwen-vl-judge}
--vlm-judge-model-config configs/mkgw-knowledge-extraction.yaml
--vlm-judge-timeout-seconds 120
```

阶段 8 smoke 命令：

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

普通 Codex sandbox 可能无法访问 CUDA 和 localhost；执行真实 smoke 时，controller 可以使用已获批准的非 sandbox/escalated 命令。

## 输出设计

阶段 8 写入现有 run 的 evaluation 目录：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/
  qwen_vl_judge_summary.json
  qwen_vl_judge_entity_scores.csv
  blip_itm_summary.json
  blip_itm_entity_scores.csv
  image_quality_model_comparison_summary.json
  entities/Q123__abcdef12/
    qwen_vl_judge.json
    blip_itm.json
```

### Qwen-VL run summary

`qwen_vl_judge_summary.json` 包含：

- `run_dir`
- `output_dir`
- `evaluator = qwen-vl-judge`
- `model_name`
- `base_url`
- `evaluated_entities`
- `evaluated_images`
- `failed_images`
- `fallback_images`
- `parse_failed_images`
- `averages.entity_match_score`
- `averages.prompt_alignment_score`
- `averages.kg_evidence_score`
- `averages.artifact_score`
- `averages.overall_judge_score`
- `best_entities`
- `worst_entities`

### BLIP/ITM run summary

`blip_itm_summary.json` 包含：

- `run_dir`
- `output_dir`
- `evaluator = blip-itm-local`
- `model_dir`
- `evaluated_entities`
- `evaluated_images`
- `failed_images`
- `fallback_images`
- `averages.blip_prompt_itm_score`
- `averages.blip_text_itm_score`
- `averages.blip_evidence_itm_score`
- `averages.blip_overall_itm_score`
- `best_entities`
- `worst_entities`

### Comparison summary

`image_quality_model_comparison_summary.json` 包含：

- CLIP summary 引用：
  - `clip_overall_image_quality_score`
  - `clip_prompt_similarity`
  - `clip_text_similarity`
  - `clip_evidence_similarity`
- BLIP/ITM summary 引用：
  - `blip_overall_itm_score`
  - `blip_prompt_itm_score`
  - `blip_text_itm_score`
  - `blip_evidence_itm_score`
- Qwen-VL Judge summary 引用：
  - `overall_judge_score`
  - `entity_match_score`
  - `prompt_alignment_score`
  - `kg_evidence_score`
  - `artifact_score`
- `evaluated_images = 25`
- `notes`：说明 CLIP、BLIP 和 Qwen-VL 衡量的是不同信号，不应直接当作同一模型分数。

## 成功标准

阶段 8 成功必须同时满足：

- 输入 run 验证：
  - `outputs/qwen_sdxl_clip_joint_smoke_20260709` 存在。
  - 5 个实体。
  - 25 张 PNG。
  - 已有 CLIP summary 可读。
- Qwen-VL Judge：
  - 25/25 images judged successfully。
  - `failed_images = 0`。
  - `fallback_images = 0`。
  - `parse_failed_images = 0`。
  - 每条 image-level result 都有 5 个 score 字段。
  - 每个 score 都在 `0-100`。
  - 每条 image-level result 都有非空 `rationale`。
- BLIP/ITM：
  - 25/25 images scored successfully。
  - `failed_images = 0`。
  - `fallback_images = 0`。
  - 每条 image-level result 都有 prompt/text/evidence 三类分数。
  - 每个 score 都在 `0-100`。
- 输出：
  - `qwen_vl_judge_summary.json` 存在。
  - `qwen_vl_judge_entity_scores.csv` 有 header + 5 行。
  - `blip_itm_summary.json` 存在。
  - `blip_itm_entity_scores.csv` 有 header + 5 行。
  - `image_quality_model_comparison_summary.json` 存在。
  - 每个实体目录都有 `qwen_vl_judge.json` 和 `blip_itm.json`。
- 验证：
  - `pytest tests -v -p no:cacheprovider` 通过。
  - `compileall mmkgc tests scripts` 通过，pycache 写入 `/tmp`。
  - `git status` 中没有模型权重、数据集、outputs 或 pycache 待提交。
- 报告：
  - 新增阶段 8 验收报告。
  - 报告记录真实 Qwen-VL Judge、BLIP/ITM、CLIP 对比数值。

任何一个条件不满足，阶段 8 应记录为 blocked 或 failed smoke，不进入扩大规模。

## 测试策略

单元测试：

- BLIP 模型目录路径安全。
- BLIP 本地权重缺失 fail-fast。
- BLIP fake scorer 生成 prompt/text/evidence 三类分数。
- Qwen-VL Judge prompt 构造包含实体、prompt、generated text、KG evidence。
- Qwen-VL Judge JSON parse 成功。
- Qwen-VL Judge 缺字段、越界分数、空 rationale fail-fast。
- writer 输出 summary、CSV 和 per-entity JSON。
- comparison summary 能读取 CLIP、BLIP、Qwen-VL 三类 summary 并并排输出。

CLI 测试：

- help text 包含新增参数。
- 默认不启用 BLIP/ITM 或 Qwen-VL Judge。
- fake BLIP 和 fake Qwen-VL Judge 能写出完整产物。
- validate failure 时 fail-fast，不写半成品 summary。

真实 smoke 验证：

- 对 `outputs/qwen_sdxl_clip_joint_smoke_20260709` 执行阶段 8 命令。
- 验证 Qwen-VL 25/25。
- 验证 BLIP/ITM 25/25。
- 验证 comparison summary。
- 写阶段 8 中文验收报告。

## 风险与约束

- Qwen-VL Judge 与 Qwen text generation 使用同一个 Qwen2-VL 模型，可能存在模型自我偏向。报告必须明确这一点。
- BLIP/ITM 与 CLIP 都是图文匹配信号，不等于 KG 关系真正可见。
- Qwen-VL Judge 是开放式视觉判断，但仍不是人工金标准。
- 25 张图片 smoke 不能代表 500 实体规模分布。
- 普通 Codex sandbox 可能无法访问 localhost、CUDA 和 GPU。
- Qwen/vLLM 服务由用户管理，不由本阶段代码启动或停止。
- BLIP 与 CLIP 可共用 GPU 1；如显存不足，BLIP 可用 CPU 调试，但阶段 8 真实 smoke 优先使用 `cuda:1`。
- `blip-itm-large-coco/` 必须加入 `.gitignore`。

## 后续路线

阶段 8 通过后，建议下一步：

1. 人工抽样复核 Qwen-VL Judge 的 best/worst cases，确认 judge rubric 是否偏松或偏严。
2. 扩大到 50 或 100 实体联合真实模型 run。
3. 对比 CLIP、BLIP/ITM、Qwen-VL Judge 的相关性和冲突样本。
4. 在评价标准稳定后，再接 SDXL Refiner、SD3.5 或 Flux 做生成模型对比。
