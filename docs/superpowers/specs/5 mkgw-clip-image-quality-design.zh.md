# MKG-W CLIP 真实图像质量评价设计文档

## 目标

在已经完成本地 SDXL Base 1.0 图像生成 smoke 之后，接入真实图像质量评价的第一版：使用本地 OpenAI CLIP `ViT-B-32` 权重，对 SDXL 生成的 PNG 与文本目标之间做跨模态相似度评价。

本阶段目标是回答：

- SDXL 生成的真实 PNG 是否符合 Stage 4 的 image prompt。
- PNG 是否和对应 text candidate / generated text 语义一致。
- PNG 是否贴近实体 label、entity type 和 KG relation evidence summary。
- CLIP 图像质量指标能否作为独立新增指标进入现有 evaluation 输出，而不破坏旧的 offline evaluation 与 calibrated score 语义。

## 已确认方案

brainstorming 阶段已确认：

- 阶段名称：CLIP 真实图像质量评价。
- 第一版模型：OpenAI CLIP `ViT-B-32`。
- 权重路径：`MMKGC/CLIP/ViT-B-32.pt`。
- 备用权重：`MMKGC/CLIP/ViT-L-14.pt` 已存在，但第一版不默认使用。
- 加载库：当前环境已安装的 Python `clip` 包。
- 不使用 `open_clip_torch`，不自动联网下载权重。
- 运行入口：扩展现有 `python -m mmkgc.evaluate`。
- 评价 run：复用 `outputs/sdxl_image_smoke_20260708`。
- 评价对象：
  - PNG vs image prompt。
  - PNG vs text candidate / generated text。
  - PNG vs entity evidence summary。
- 接入方式：新增独立 CLIP image quality 指标，不覆盖旧 offline evaluation score。
- 输出位置：被评价 run 的 `evaluation/` 目录。
- 失败策略：第一版 fail-fast。
- 不自动 commit。

说明：早期讨论中曾使用 `OpenCLIP` 作为阶段简称。用户已将权重目录改为 `MMKGC/CLIP/`，且当前可用权重与加载库是 OpenAI CLIP 格式。因此第一版实现、文档和字段统一使用 `CLIP` / `clip-local` / `clip_*` 命名。后续如果改用 LAION OpenCLIP 或 `open_clip_torch`，应作为新的 evaluator provider 接入，而不是混淆第一版语义。

## 非目标

本阶段不做以下事情：

- 不接 `open_clip_torch` 或 LAION OpenCLIP 权重。
- 不接 BLIP、BLIP-2、InstructBLIP、ImageReward、HPS、aesthetic predictor 或 VLM judge。
- 不重新生成 SDXL 图片。
- 不跑 100、500 或全量 MKG-W。
- 不修改 Stage 0-5 生成逻辑。
- 不修改 Stage 5 ranking 公式。
- 不把 CLIP 分数融合进 `image_best_offline_evaluation_score` 或 `calibrated_image_best_offline_evaluation_score`。
- 不覆盖已有 evaluation 输出目录，除非用户显式传入现有 `--overwrite` 行为。
- 不读取 `MMKGC/` 外部的 CLIP 权重。
- 不提交 `CLIP/` 权重目录。
- 不自动 commit。

## 自包含目录结构

本阶段使用以下 MMKGC 内部路径：

```text
MMKGC/
  CLIP/
    ViT-B-32.pt
    ViT-L-14.pt
  outputs/
    sdxl_image_smoke_20260708/
      entities/
        <entity_dir>/
          0_coarse_relation_filtering.json
          1_knowledge_modeling.json
          2_graph_constrained_reasoning.json
          3_fine_grained_relation_selection.json
          4_relation_alternation_generation.json
          5_kg_aware_pair_ranking.json
          final.json
          generated_images/
            image_001.png
            image_001.json
            ...
```

实施阶段需要把 `CLIP/` 加入 `.gitignore`，避免提交 338M/890M 权重文件。

## 架构设计

现有 `python -m mmkgc.evaluate` 已能读取 run directory，生成 run-level summary、entity-level score、CSV 和 case exports。本阶段在这个 evaluation 流程上增加一个可选的 image quality evaluator：

```text
run_dir
  -> iter entity artifacts
  -> existing offline evaluation
  -> optional clip-local image quality evaluation
  -> write existing evaluation outputs
  -> write CLIP image quality outputs
```

默认不启用 CLIP，因此现有 evaluation 行为、输出字段和测试应保持不变。只有显式传入 `--image-quality-evaluator clip-local` 时，才加载 CLIP 模型并读取 PNG 文件。

新增边界建议：

```text
mmkgc/evaluation/image_quality.py
  ImageQualityEvaluator protocol
  ClipLocalImageQualityEvaluator
  build_entity_evidence_summary()
  evaluate_entity_image_quality()

mmkgc/evaluation/writer.py
  write_clip_image_quality_outputs()

mmkgc/evaluate.py
  CLI 参数解析与 fail-fast 调用
```

`ClipLocalImageQualityEvaluator` 只负责模型加载和相似度计算，不直接写文件。文本目标构造、entity aggregation 和 run-level summary 由 evaluation 层完成。这样后续接 BLIP、VLM judge 或 Gemini Embedding 2 时，可以增加新的 provider，而不改现有 CLIP 实现。

## 数据流

对每个实体目录：

```text
final.json / stage_4 / stage_5
  -> 找到 image candidates
  -> 找到每个 image candidate 的 generated_content.image_path
  -> 读取 PNG
  -> 构造三类 text targets
  -> CLIP image encoder + text encoder
  -> 计算 cosine similarity
  -> 写 per-image scores
  -> 聚合 entity-level scores
```

三类 text target：

1. `prompt_text`
   - 优先使用 image candidate 的 `positive_prompt`。
   - 如果不存在，使用 `generated_content.prompt`。
   - 再 fallback 到 `generated_content.value`。

2. `candidate_text`
   - 优先使用同一个 pair candidate 中关联的 text candidate。
   - 如果无法从 pair 找到对应 text，则使用 Stage 4 / Stage 5 的 best text candidate。
   - 如果没有 text candidate，则该项记录为不可用，不参与 entity-level 平均。

3. `evidence_summary_text`
   - 由 entity label、entity type、high-value relation evidence、modality constraints 拼成短文本。
   - 不使用 ground truth description 作为主要证据，避免把评价目标变成对 ground truth 的泄漏式匹配。
   - 可以包含关系标签和邻居 label，例如：`Entity: X. Type: person. Evidence: occupation musician; country United States.`

## 输出指标

第一版新增字段使用 `clip_*` 前缀：

```text
clip_prompt_similarity
clip_text_similarity
clip_evidence_similarity
clip_overall_image_quality_score
```

每张图片输出：

```json
{
  "entity_key": "...",
  "image_candidate_id": "image_001",
  "image_path": "generated_images/image_001.png",
  "clip_model_name": "ViT-B/32",
  "clip_weight_path": "CLIP/ViT-B-32.pt",
  "clip_prompt_similarity": 31.42,
  "clip_text_similarity": 27.85,
  "clip_evidence_similarity": 24.18,
  "clip_overall_image_quality_score": 28.96,
  "text_targets": {
    "prompt_text": "...",
    "candidate_text": "...",
    "evidence_summary_text": "..."
  }
}
```

分数定义：

- 先计算 CLIP image embedding 与 text embedding 的 cosine similarity。
- 第一版输出 `max(0, cosine_similarity) * 100`，保持与当前项目已有 `src/metrics/CLIPScore.py` 风格接近。
- `clip_overall_image_quality_score` 是可用三类分数的算术平均。
- 如果某类 text target 不可用，则该类字段为 `null`，不参与 overall 平均。
- 如果三类 text target 全部不可用，则 fail-fast，因为该实体缺少可评价文本目标。

## 输出文件

启用 CLIP 后，输出写入被评价 run 的 `evaluation/` 目录：

```text
outputs/sdxl_image_smoke_20260708/evaluation/
  clip_image_quality_summary.json
  clip_image_quality_entity_scores.csv
  entities/
    <entity_dir>/
      clip_image_quality.json
```

`clip_image_quality_summary.json` 包含：

- `run_dir`
- `evaluator`
- `clip_model_name`
- `clip_weight_path`
- `evaluated_entities`
- `evaluated_images`
- `averages.clip_prompt_similarity`
- `averages.clip_text_similarity`
- `averages.clip_evidence_similarity`
- `averages.clip_overall_image_quality_score`
- `best_entities`
- `worst_entities`

`clip_image_quality_entity_scores.csv` 每行一个实体，包含：

- entity key / label / missing mode
- image count
- entity-level average prompt/text/evidence/overall score
- best image candidate id
- worst image candidate id

每个实体的 `clip_image_quality.json` 包含：

- entity metadata
- text target summary
- per-image score list
- entity-level aggregate

## CLI 设计

默认 evaluation 行为不变：

```bash
python -m mmkgc.evaluate --run-dir outputs/sdxl_image_smoke_20260708
```

启用 CLIP 图像质量评价：

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

新增 CLI 参数：

```text
--image-quality-evaluator {none,clip-local}
--clip-model-name ViT-B/32
--clip-model-dir CLIP
--clip-device cuda
```

`--image-quality-evaluator` 默认值为 `none`。如果为 `clip-local`：

- 校验 `--clip-model-dir` 必须位于 `MMKGC/` 内。
- 校验 `CLIP/ViT-B-32.pt` 存在。
- 调用 `clip.load(str(CLIP/ViT-B-32.pt), jit=False, device=...)`，显式加载本地权重文件。
- 禁止使用模型名触发 `clip` 包的自动下载逻辑；如权重缺失，直接报错。

## 失败策略

第一版采用 fail-fast：

- `clip` 包无法导入：退出非 0，并提示缺失依赖。
- `CLIP/ViT-B-32.pt` 不存在：退出非 0，并提示需要放到 `MMKGC/CLIP/ViT-B-32.pt`。
- `--clip-model-dir` 指向 `MMKGC/` 外部：退出非 0。
- run directory 不存在或不是 MMKGC run：沿用现有 evaluation 错误。
- 任一 image candidate 声明了 PNG 但文件缺失：退出非 0。
- PNG 无法被 Pillow 打开：退出非 0。
- text target 三类都不可用：退出非 0。
- 任何实体失败时，不写半截 CLIP summary。

后续扩大到 100/500 实体时，可以新增 partial continue 模式，但不在第一版实现。

## 测试与验收

单元测试：

- `clip_model_name` 到权重文件名的映射：`ViT-B/32 -> ViT-B-32.pt`。
- `clip_model_dir` 路径安全：拒绝外部路径、`..` traversal、symlink escape。
- evidence summary 构造稳定、不会使用 ground truth description 作为主证据。
- 三类 text target 缺失时的处理。
- score aggregation 忽略 `null` 分数，并在全部缺失时 fail-fast。
- writer 输出 summary、CSV 和 per-entity JSON。

CLI 测试：

- 默认 `python -m mmkgc.evaluate` 不加载 CLIP、不要求 `CLIP/` 存在。
- `--image-quality-evaluator clip-local` 缺失权重时 fail-fast。
- 使用 fake CLIP evaluator 时，可以写出 `clip_image_quality_summary.json`、`clip_image_quality_entity_scores.csv` 和 per-entity `clip_image_quality.json`。
- 旧 evaluation 输出字段保持兼容。

真实 smoke：

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

验收标准：

- 5 个实体全部有 `clip_image_quality.json`。
- 共 25 张 PNG 都产生 CLIP 分数。
- summary 中 `evaluated_entities=5`，`evaluated_images=25`。
- 三类平均分和 overall 平均分存在。
- 现有 evaluation summary 仍能生成，旧字段不消失。
- 完整测试通过：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/mmkgc-clip-compileall-pycache \
  /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

## 文档更新

实施阶段需要更新：

- `README.md`
- `README.zh.md`
- `docs/project-progress-summary.zh.md`
- 新增阶段 6 validation report，例如：
  - `docs/evaluation/2026-07-09-clip-image-quality-smoke-validation.zh.md`

文档必须说明：

- 阶段 6 使用 CLIP 跨模态相似度评价真实 PNG。
- CLIP 分数不是 VLM judge，不直接证明 KG 关系严格成立。
- 第一版不会替换旧 offline evaluation 和 calibrated score。
- `CLIP/` 是本地权重目录，已加入 gitignore，不提交。

## 后续扩展

本阶段完成后，下一步可以考虑：

- 用 `ViT-L-14` 对同一 run 复评，比较 `ViT-B-32` 与 `ViT-L-14` 的稳定性。
- 接入 BLIP / BLIP-2 caption，把 PNG 转成 caption 后做文本一致性分析。
- 用 Qwen2-VL 或 Gemini 做小样本 VLM judge，补充可解释判断。
- 在 Qwen + SDXL 联合真实模型 run 上复用同一套 CLIP image quality evaluation。
- 扩大到 100 或 500 实体，并增加 partial continue 模式。
