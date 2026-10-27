# MKG-W Qwen + SDXL + CLIP 联合真实模型 Smoke 设计文档

## 目标

阶段 7 的目标是在同一次 MKG-W run 中联合验证三个已经分别跑通的真实模型链路：

- Qwen2-VL-7B 生成 text candidates。
- SDXL Base 1.0 生成真实 PNG image candidates。
- OpenAI CLIP ViT-B-32 对真实 PNG 做跨模态相似度评价。

本阶段回答的问题是：

```text
真实 text generation、真实 image generation 和真实 image quality evaluation 能否在同一个 5 实体 smoke 中稳定串起来？
```

这不是扩大规模实验，也不是模型效果最终结论。它是进入 50、100 或 500 实体真实模型实验之前的联合链路验收。

## 已确认方案

brainstorming 阶段已确认：

- 阶段名称：Qwen + SDXL + CLIP 联合真实模型 5 实体 smoke。
- 不新增代码，主要做实验运行、验收和报告。
- Qwen/vLLM 由用户在终端手动启动。
- 资源分工：
  - GPU 0：Qwen/vLLM 服务。
  - GPU 1：SDXL 生成和 CLIP 评价。
- Qwen 服务参数：
  - `base_url = http://localhost:12345/v1`
  - `api_key = token-abc123`
  - 模型路径优先使用 `MMKGC/Qwen/Qwen2-VL-7B-Instruct`。
- 样本规模：
  - `limit = 5`
  - `candidate_budget = 5`
  - 预期 25 个 Qwen text candidates。
  - 预期 25 张 SDXL PNG。
  - 预期 25 张 PNG 进入 CLIP image quality evaluation。
- SDXL 参数沿用阶段 5：
  - `1024x1024`
  - `20` steps
  - `seed = 20260708`
  - `guidance_scale = 7.5`
  - `sdxl-device = cuda:1`
- CLIP 参数沿用阶段 6：
  - `clip_model_name = ViT-B/32`
  - `clip_model_dir = CLIP`
  - `clip-device = cuda:1`
- 失败处理：
  - 严格成功标准。
  - Qwen fallback 不算成功。
  - SDXL fallback 不算成功。
  - CLIP 未评价完整 25 张图片不算成功。
  - offline evaluation 失败不算成功。
- 预期 run 名称：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709
```

## 非目标

本阶段不做以下事情：

- 不新增 pipeline 代码、adapter 代码、evaluation 代码或测试。
- 不新增 joint smoke 脚本。
- 不做 50、100、500 或全量 MKG-W 真实模型运行。
- 不接 BLIP、VLM judge、ImageReward、HPS、aesthetic predictor 或人工视觉验收流程。
- 不接 SDXL Refiner、Stable Diffusion 3.5 或 Flux。
- 不修改 Qwen prompt、SDXL prompt 或 CLIP scoring 公式。
- 不把 CLIP 分数融合进 offline/calibrated score。
- 不把 fallback 结果当作真实联合 smoke 成功。
- 不自动 commit。

## 运行前置条件

### 本地配置

`configs/mkgw-knowledge-extraction.yaml` 必须存在，并指向本地 Qwen OpenAI-compatible 服务：

```yaml
lmm:
  model_name: ./Qwen/Qwen2-VL-7B-Instruct
  base_url: http://localhost:12345/v1
  api_key: token-abc123

dataset:
  name: mkgw
  data_dir: ./datasets/MKG-W
  entity_info_path: ./datasets/MKG-W/entity_info.json

inference:
  save_path: ./outputs/qwen_text_smoke
  max_token: 512
  temperature: 0.1
  num_workers: 1
```

### Qwen/vLLM 服务

用户在终端启动 Qwen/vLLM，建议命令：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC/Qwen/Qwen2-VL-7B-Instruct \
  --dtype auto \
  --api-key token-abc123 \
  --max_model_len 8192 \
  --port 12345 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --disable-log-stats \
  --trust-remote-code \
  --limit-mm-per-prompt image=4
```

如果用户已经在其它路径成功启动同一个 7B 模型，也可以保持已有服务，只要健康检查通过即可。

### 本地模型与数据

以下路径必须存在：

```text
datasets/MKG-W/
Qwen/Qwen2-VL-7B-Instruct/
SDXL/stable-diffusion-xl-base-1.0/
CLIP/ViT-B-32.pt
```

`Qwen/`、`SDXL/`、`CLIP/`、`outputs/` 中的权重和生成物不应进入 git。

## 实验数据流

```text
MKG-W selected 5 entities
  -> Stage 0-3 rule pipeline
  -> Stage 4:
       Qwen local service generates 5 text candidates per entity
       rule pipeline builds 5 image prompts per entity
       SDXL local generates 5 PNG images per entity
  -> Stage 5 ranks text/image/pair candidates
  -> run artifacts written under outputs/qwen_sdxl_clip_joint_smoke_20260709
  -> offline evaluation
  -> CLIP image quality evaluation
  -> validation report
```

## 成功标准

阶段 7 成功必须同时满足：

- Qwen health check 成功。
- `metrics_summary.json` 中：
  - `processed_entities = 5`
  - `failed_entities = 0`
- Qwen 调用日志：
  - `model_calls.jsonl` 保留全部模型调用；joint run 中总行数为 50，其中 25 行是 Qwen text 调用，25 行是 SDXL image 调用。
  - 过滤 `provider = qwen-local-openai-compatible` 后，Qwen text 调用行数为 25。
  - Qwen text 调用每行 `status = generated`。
  - Qwen text 调用每行 `fallback_used = false`。
  - Qwen text 调用每行 `metadata.model_status = generated`。
- SDXL 产物：
  - 5 个 entity 目录。
  - 每个 entity 目录 5 张 PNG。
  - 总 PNG 数 25。
  - 每张 PNG 有同名 JSON sidecar。
  - `image_model_calls.jsonl` 总 image 行数为 25。
  - 每行 `status = generated`。
  - 每行 `fallback_used = false`。
  - 每行尺寸为 `1024x1024`。
- offline evaluation：
  - `evaluation/evaluation_summary.json` 存在。
  - `evaluated_entities = 5`
  - `failed_entities = 0`
- CLIP image quality：
  - `evaluation/clip_image_quality_summary.json` 存在。
  - `evaluated_entities = 5`
  - `evaluated_images = 25`
  - `clip_image_quality_entity_scores.csv` 有 6 行，包含 header + 5 entities。
- 验收报告记录真实命令、真实输出和真实分数。

任何一个条件不满足，阶段 7 应记录为失败，并停止扩大规模。

## 输出与报告

运行输出目录：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709/
```

验收报告：

```text
docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md
```

报告必须记录：

- Qwen/vLLM 启动方式。
- Qwen health check 输出。
- joint run 命令。
- run metrics。
- Qwen model call 统计。
- SDXL PNG 和 image model call 统计。
- offline evaluation 统计。
- CLIP image quality 统计。
- GPU 分工。
- 是否使用 fallback。
- 失败时的错误信息和处理建议。

## 风险与约束

- 普通 Codex sandbox 看不到 GPU；需要用户终端执行 GPU 命令，或 controller 使用已获批准的非 sandbox/escalated 命令。
- Qwen/vLLM 是长驻服务，不由本阶段代码管理生命周期。
- GPU 0/1 分工依赖 `CUDA_VISIBLE_DEVICES=0` 和 `cuda:1` 约定；如果用户机器 GPU 编号变化，需要先用 `nvidia-smi` 确认。
- SDXL 和 CLIP 复用 GPU 1；二者不需要同时常驻，通常先生成 PNG，再运行 CLIP。
- Qwen 失败 fallback 是代码支持的容错行为，但本阶段验收不接受 fallback 作为成功。
- CLIP 是跨模态相似度，不是 VLM judge，不能替代开放式视觉理解或人工验收。

## 后续路线

阶段 7 成功后，建议下一步再做：

1. 联合结果的 best/worst/low-CLIP/high-CLIP 样本复核。
2. 与 rule-only、Qwen-only、SDXL-only、joint run 的指标对比表。
3. 50 或 100 实体联合真实模型 run。
4. 引入 BLIP/VLM judge 或人工视觉验收流程。
