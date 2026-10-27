# MMKGC 项目当前进展总结

更新时间：2026-07-23

## 总体状态

MMKGC 目前已经从最初的规则 MVP，推进到“可离线评估、可扩大样本、可接入本地真实模型”的阶段。

当前已完成：

- MKG-W 数据集自包含到 `MMKGC/datasets/MKG-W`。
- 建立六阶段 multimodal missing KG completion pipeline。
- 支持 `text`、`image`、`both` 三种缺失模态。
- 默认规则模式可稳定生成 text candidates、image prompt candidates 和 text-image pair ranking。
- 建立 offline evaluation、calibration、人工/多 agent 验收样本与 500 样本扩大运行。
- 接入本地 Qwen2-VL-7B 文本生成。
- 接入本地 SDXL Base 1.0 图像生成，并完成 5 实体、25 图片真实 smoke。
- 接入本地 OpenAI CLIP ViT-B-32 真实 PNG 跨模态相似度评价。
- 完成 Qwen + SDXL + CLIP 联合真实模型 5 实体 smoke，Qwen text、SDXL PNG 和 CLIP 评价在同一次 run 后验收通过。
- 完成 Qwen-VL Judge + BLIP/ITM 图像评价，并对 500 实体 run 的 2500 张真实 PNG 严格评价通过。
- 完成 Stage 9 三人 consensus 视觉评价校准复核；25 图 smoke、100 实体/500 图、500 实体/2500 图均已有校准输出。500 run 中 CLIP 按平均绝对误差最接近 reviewer consensus 但整体偏低，Qwen-VL Judge 明显偏高且平均绝对误差最大。
- 完成 Stage 10 自动视觉评价校准与风险门控；`outputs/qwen_sdxl_500_20260712` 的 2500 张图已写出 `accept/review/reject` 决策，其中 accept=0、review=1404、reject=1096。
- 完成 500 run Stage5 高风险人工复核包与 gate 校准闭环工具；已从 2500 张图抽取 60 条高风险样本，并用 Stage9 reviewer consensus 先生成候选 gate profile，Qwen-VL 权重保持上限 0.15。已完成三 agent 只读校准审计，并先收紧 Qwen image/pair judge rubric 的视觉证据来源约束；另完成 60 条 high-risk packet 的 multi-agent 三审模拟标签与 gate 校准回放，simulated consensus 为 reject=42、review=18。
- 完成 Stage 12 BERTScore 文本语义联合评价；500 实体、2500 个文本候选已写入独立评价目录，平均 F1 为 `0.842020`。
- 完成 Stage 0-3 LMM 语义主链路入口、默认主链路切换、Stage 0-2 fixed50 对照、100 实体验收与 500 实体 clean 验收；Stage 0 规则阶段负责清噪和压缩，默认 `--stage02-planner qwen-local --stage3-planner qwen-local` 让本地 Qwen 对 Stage 0 relation filtering、Stage 1 knowledge modeling、Stage 2 constrained reasoning 和 Stage 3 relation selection 做严格 JSON refinement，失败时回退 deterministic 输出并记录 `semantic_refinement` / `model_calls.jsonl`。Stage 0-2 sanitized fixed50 版已达到 `150/150 applied`、semantic fallback `0/150`、prompt leak `0/150`；Stage 3 prompt compact 后，100 实体验收中 Stage 0/1/2/3 为 `400/400 applied`；500 clean combined 中 Stage 0/1/2/3 为 `2000/2000 applied`、严格缺失实体 `0/500`。
- 完成 Stage 13 Stage 5 model-fusion ranking，并已把 model-fusion 切为默认 Stage 5 主链路；包括在线 5 实体 smoke、replay / rerank CLI、fixed50 manifest、legacy/model-fusion held-out 对比、BERTScore optional auxiliary scorer、fixed50 calibration v2、Stage 4 safer candidate regeneration，以及 Qwen judge retry/坏候选隔离。
- README 中英文文档、spec、implementation plan 和 evaluation 报告已经按阶段保存。

当前仍未完成：

- 500 实体 Stage 9/10 高风险人工复核包已生成，并已完成 multi-agent 三审模拟校准；但还没有新一轮真人填写 `review_template.jsonl` 并据此覆盖默认 gate 配置。
- 还没有进入全量 MKG-W 真实模型运行；不建议在 Stage 10 `accept = 0` 且 review/reject 高风险结构密集的情况下直接全量运行。
- 已用 Stage9 consensus 跑通 Stage5 gate 候选校准闭环，但还没有基于新人工复核结果调整 Qwen-VL Judge rubric、prompt rubric 和默认视觉评价阈值。
- Stage 0–3 已切换为默认 Qwen LMM 语义主链路，500 级真实 LMM Stage 0-3 质量验收已完成；但质量信号仍偏混合，legacy best-pair 平均 delta 为 `-0.002588`，需要对正负 delta 样本做人审判断。
- 还没有接入 SDXL Refiner、SD3.5 或 Flux。

## 阶段 1：MKG-W 双模态 MVP

目标是先把项目计划中的六阶段流程落地到 MKG-W，并保持完全离线、可复现。

已实现的核心流程：

```text
Stage 0: coarse_relation_filtering
Stage 1: knowledge_modeling
Stage 2: graph_constrained_reasoning
Stage 3: fine_grained_relation_selection
Stage 4: relation_alternation_generation
Stage 5: kg_aware_pair_ranking
```

默认行为：

- `candidate_budget=5`。
- `text` 模式生成 5 个 text candidates。
- `image` 模式生成 5 个 image candidates；默认 model-fusion 主链路会通过 `--image-generator auto` 解析到 SDXL 真实图像 artifact，legacy/rule baseline 仍生成 image prompts。
- `both` 模式生成 5 个 text candidates、5 个 image candidates，并由 Top-3 x Top-3 生成 9 个 pair candidates。
- 输出按实体目录保存完整六阶段 JSON 与 `final.json`。

相关文档：

- `docs/superpowers/specs/0 mkgw-both-modal-mvp-design.zh.md`
- `docs/superpowers/plans/0 mkgw-both-modal-mvp.zh.md`

## 阶段 2：Offline Evaluation 与校准

目标是先建立“如何判断生成结果好不好”的离线评价标准，再扩大规模或接真实模型。

已实现：

- `python -m mmkgc.evaluate` 离线评估 CLI。
- 对 text candidates、image prompt candidates、pair ranking 分别打分。
- 输出 `evaluation_summary.json`、`summary.csv`、`entity_scores.csv`、best/worst/fallback cases。
- 引入 calibrated scores，对 fallback、generic entity type、template image prompt、弱关系证据、pair 单侧偏弱等情况扣分。
- 保留 Stage 5 ranking score 与 offline evaluation score 的命名边界，避免混淆。

相关文档与报告：

- `docs/superpowers/specs/1 mkgw-offline-evaluation-design.zh.md`
- `docs/superpowers/plans/1 mkgw-offline-evaluation.zh.md`
- `docs/superpowers/specs/2 mkgw-evaluation-calibration-design.zh.md`
- `docs/superpowers/plans/1 mkgw-evaluation-calibration.zh.md`
- `docs/evaluation/2026-07-07-calibration-validation.zh.md`
- `docs/evaluation/2026-07-07-three-agent-acceptance-review.zh.md`

## 阶段 3：500 样本扩大运行

目标是验证规则 MVP 和 evaluation 在更大样本上是否稳定。

已完成：

- 运行 500 实体规模的规则 baseline。
- 检查 processed/failed、fallback 分布、calibrated score 分布和典型 best/worst cases。
- 确认扩大运行阶段不接真实 LMM、SDXL、CLIP、BLIP 或 embedding 模型。

相关文档与报告：

- `docs/superpowers/specs/2 mkgw-scale-500-design.zh.md`
- `docs/superpowers/plans/2 mkgw-scale-500.zh.md`
- `docs/evaluation/2026-07-07-scale-500-validation.zh.md`

## 阶段 4：本地 Qwen2-VL-7B 文本生成

目标是先接入真实 text generation，同时保持 image candidates 仍为规则 prompt。

已实现：

- `configs/mkgw-knowledge-extraction.example.yaml`：脱敏的本地 Qwen OpenAI-compatible 服务配置模板。
- `configs/mkgw-knowledge-extraction.yaml`：本机运行配置，包含本地 key，已加入 gitignore，不提交。
- `scripts/start_qwen_local_server.sh`：本地服务启动辅助脚本。
- `scripts/check_qwen_local_server.py`：健康检查脚本。
- `mmkgc/adapters/qwen.py`：Qwen text generator adapter。
- CLI 参数：
  - `--text-generator {rule,qwen-local}`
  - `--model-config`
  - `--model-call-timeout-seconds`
- `model_calls.jsonl`：每个实体记录模型调用、状态、fallback 和耗时。
- `--text-generator` 默认已切到 `qwen-local`，作为 Stage 4 LMM relation-alternation 文本候选主链路；`rule` 保留为离线 baseline/debug 模式。
- Python API 新增 `build_default_mainline_pipeline()` / `MMKGCPipeline.from_default_mainline()`，用于构建与 CLI 默认一致的 Qwen Stage 0-4 + auto image + model-fusion 主链路；helper 默认带上 `datasets/MKG-W` dataset root，以支持 text-only model-fusion 安全读取已有图像输入；裸 `MMKGCPipeline()` 仍保留为 adapter injection 构造器。
- Stage 4 候选新增 `generation_brief`，包含 generation intent、semantic focus、diversity axis、grounded evidence policy、must/should/context/avoid 约束和 evidence facts；text prompt 与 image positive prompt 已注入 composition focus、must include 和 avoid 信息。当前代码状态下已完成 5 实体 Qwen+SDXL 真实质量 smoke：25/25 text candidates、25/25 image candidates、25/25 PNG、Qwen text fallback 0、SDXL image fallback 0；CLIP/BLIP/Qwen-VL/BERTScore 均已写出评价，图像自动质量仍偏低，报告见 `docs/evaluation/2026-07-22-stage4-qwen-sdxl-quality-smoke.zh.md`。
- `qwen-local` 失败时 fallback 到规则文本候选，并记录 `failed_fallback`。

已完成两类 smoke：

- fallback-only smoke：服务未连通时，25 次调用全部 fallback，用于验证 fallback 与日志链路。
- real Qwen smoke：用户手动部署 Qwen2-VL-7B 后，5 实体、25 次 text candidate 调用全部真实生成，fallback 0。

真实 Qwen smoke 验证结果：

```text
outputs/qwen_text_real_7b_20260708
processed_entities = 5
failed_entities = 0
model_calls = 25
generated = 25
failed_fallback = 0
evaluation evaluated_entities = 5
evaluation failed_entities = 0
```

相关文档与报告：

- `docs/superpowers/specs/3 mkgw-qwen-text-generation-design.zh.md`
- `docs/superpowers/plans/3 mkgw-qwen-text-generation.zh.md`
- `docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md`
- `docs/evaluation/2026-07-08-qwen-real-7b-smoke-validation.zh.md`

## 阶段 5：本地 SDXL Base 1.0 图像生成

目标是把 Stage 4 的 image prompt candidates 送入本地 SDXL，生成真实 PNG 文件。

用户确认的第一版范围：

- 先停 Qwen，只跑 SDXL，避免 GPU 冲突。
- 只用 SDXL Base 1.0，不接 Refiner。
- 模型路径固定为 `SDXL/stable-diffusion-xl-base-1.0`。
- 5 个实体，每个实体 5 张图，共 25 张。
- 参数：`1024x1024`、`20` steps、固定 seed `20260708`。
- 图片保存到每个实体目录下的 `generated_images/`。
- 每张图片有同名 JSON sidecar metadata。
- 每个实体有 `image_model_calls.jsonl`。
- 第一版只做工程验收，不接 CLIP/BLIP 图像质量评价。

已实现：

- `mmkgc/adapters/sdxl.py`：SDXL image generator adapter。
- `ImageGenerator` 和 `GeneratedImageArtifact` 协议边界。
- Stage 4 可选调用 SDXL，并保留规则 image prompt 模式兼容。
- `RunWriter` 保存 PNG、sidecar metadata 和 `image_model_calls.jsonl`。
- `scripts/validate_sdxl_image_run.py` 工程验收脚本。
- CLI 参数：
  - `--image-generator {auto,rule,sdxl-local}`，默认 `auto` 会在 `image`/`both` + model-fusion 时解析为 `sdxl-local`，其它模式解析为 `rule`。
  - `--sdxl-model-dir`
  - `--sdxl-width`
  - `--sdxl-height`
  - `--sdxl-steps`
  - `--sdxl-seed`
  - `--sdxl-guidance-scale`
  - `--sdxl-device`
- 路径安全：
  - 只允许 `SDXL/stable-diffusion-xl-base-1.0`。
  - 拒绝 Refiner、外部路径、`..` traversal 和 symlink escape。

真实 SDXL smoke 验证结果：

```text
outputs/sdxl_image_smoke_20260708
processed_entities = 5
failed_entities = 0
PNG files = 25
metadata sidecars = 25
image_model_calls.jsonl files = 5
image model call rows = 25
image size = 1024x1024
evaluation evaluated_entities = 5
evaluation failed_entities = 0
```

最终验证：

```text
pytest tests -v -p no:cacheprovider
126 passed

compileall mmkgc tests scripts
exit code 0

validate_sdxl_image_run.py
ok = true
entities_checked = 5
images_checked = 25
```

相关文档与报告：

- `docs/superpowers/specs/4 mkgw-sdxl-image-generation-design.zh.md`
- `docs/superpowers/plans/4 mkgw-sdxl-image-generation.zh.md`
- `docs/evaluation/2026-07-08-sdxl-image-smoke-validation.zh.md`

## 阶段 6：CLIP 真实图像质量评价

目标是让 evaluation 能直接消费 SDXL 生成的真实 PNG，用本地 OpenAI CLIP ViT-B-32 计算图片与 prompt、实体文本、KG evidence 的跨模态相似度。

本阶段范围：

- 使用本地权重目录 `CLIP/`，该目录已加入 gitignore。
- 默认评估 `outputs/sdxl_image_smoke_20260708` 中的 5 个实体、25 张 PNG。
- 只做 CLIP similarity，不接 BLIP、VLM judge 或人工视觉验收。
- CLIP 分数作为额外图像质量产物保存，不替换 `calibrated_image_best_offline_evaluation_score` 或既有 offline evaluation。

已实现：

- `python -m mmkgc.evaluate` 支持 `--image-quality-evaluator clip-local`。
- 支持 `--clip-model-name ViT-B/32`、`--clip-model-dir CLIP` 和 `--clip-device`。
- 输出 run 级 summary、entity CSV 和每实体 `clip_image_quality.json`。
- 评价器继续保留原有 offline evaluation 产物，并额外写入 CLIP image quality 产物。

真实 CLIP smoke 验证结果：

```text
outputs/sdxl_image_smoke_20260708
first CUDA attempt = RuntimeError: No CUDA GPUs are available
successful device = cpu
evaluated_entities = 5
failed_entities = 0
clip_evaluated_entities = 5
clip_evaluated_images = 25
clip_prompt_similarity = 26.945619
clip_text_similarity = 28.143429
clip_evidence_similarity = 25.113275
clip_overall_image_quality_score = 26.734108
clip_model_name = ViT-B/32
clip_weight_path = /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC/CLIP/ViT-B-32.pt
clip_image_quality_entity_scores.csv rows = 6
```

本阶段限制：

- 当前 Codex 进程不可见 CUDA，所以 smoke 改用 CPU 完成；这不改变评价逻辑，只影响运行设备和速度。
- CLIP 是跨模态相似度模型，不是 VLM judge，不能替代人工判断或开放式视觉问答。
- 当前结果是 5 实体、25 图片 smoke，不代表大规模真实模型 run 的质量分布。

相关文档与报告：

- `docs/evaluation/2026-07-09-clip-image-quality-smoke-validation.zh.md`

## 阶段 7：Qwen + SDXL + CLIP 联合真实模型 smoke

目标是在同一次 MKG-W 5 实体真实模型 smoke 中，验证 Qwen2-VL-7B 生成 text candidates、SDXL Base 1.0 生成真实 PNG、OpenAI CLIP ViT-B-32 评价真实 PNG 的完整链路。

资源分工：

- GPU 0：用户手动部署的 Qwen/vLLM OpenAI-compatible 服务。
- GPU 1：SDXL 图像生成和 CLIP 图像质量评价。
- Qwen/vLLM 不由 `mmkgc.run` 管理生命周期；controller 通过 health check 验证服务。

Qwen health check：

```json
{"base_url":"http://localhost:12345/v1","latency_ms":129,"model_name":"Qwen/Qwen2-VL-7B-Instruct","ok":true,"response_preview":"OK."}
```

联合 run：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709
processed_entities = 5
failed_entities = 0
model_calls_total = 50
qwen_text_calls = 25
image_calls_in_model_calls = 25
qwen_generated = 25
qwen_fallback_used = 0
```

说明：`model_calls_total = 50` 是因为日志保留 text 与 image model calls；过滤 `provider = qwen-local-openai-compatible` 后，Qwen text 调用为 25 行，不是 Qwen 调用了 50 次。

SDXL 产物：

```text
entities_checked = 5
images_checked = 25
image_model_calls rows = 25
generated = 25
fallback_used = 0
image size = 1024x1024
```

CLIP 评价：

```text
evaluated_entities = 5
failed_entities = 0
clip_evaluated_entities = 5
clip_evaluated_images = 25
clip_prompt_similarity = 26.94383
clip_text_similarity = 27.602151
clip_evidence_similarity = 25.091251
clip_overall_image_quality_score = 26.545744
clip_image_quality_entity_scores.csv rows = 6
```

本阶段结论：

- Qwen text generation 与 SDXL image generation 已经在同一次真实模型 run 中组合成功。
- Qwen、SDXL 均未使用 fallback。
- Offline evaluation 与 CLIP image quality evaluation 均完整处理 5 个实体。
- 该结果是 5 实体联合 smoke 验收，不是 500 实体规模结论，也不是开放式视觉质量判断。

相关文档与报告：

- `docs/superpowers/specs/6 mkgw-qwen-sdxl-clip-joint-smoke-design.zh.md`
- `docs/superpowers/plans/6 mkgw-qwen-sdxl-clip-joint-smoke.zh.md`
- `docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md`

## 阶段 8：Qwen-VL Judge + BLIP/ITM 图像评价

目标是在 Stage 7 已生成的 25 张真实 PNG 上，补充 CLIP 之外的两类图像评价信号：

- Qwen-VL Judge：开放式视觉判断，输出 entity match、prompt alignment、KG evidence、artifact 和 overall 分数。
- BLIP/ITM：图像与 prompt、generated text、KG evidence 的图文匹配分数。

运行对象：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709
```

真实 smoke 结果：

```text
qwen_vl.evaluated_entities = 5
qwen_vl.evaluated_images = 25
qwen_vl.failed_images = 0
qwen_vl.fallback_images = 0
qwen_vl.parse_failed_images = 0
qwen_vl.overall_judge_score = 43.2

blip_itm.evaluated_entities = 5
blip_itm.evaluated_images = 25
blip_itm.failed_images = 0
blip_itm.fallback_images = 0
blip_itm.blip_overall_itm_score = 23.495848

comparison.evaluated_images = 25
clip_overall_image_quality_score = 26.545744
```

本阶段结论：

- Qwen-VL Judge 25/25 成功，fallback 0，parse failure 0。
- BLIP/ITM 25/25 成功，fallback 0。
- 已写出 `qwen_vl_judge_summary.json`、`blip_itm_summary.json` 和 `image_quality_model_comparison_summary.json`。
- CLIP、BLIP/ITM、Qwen-VL 衡量不同信号，不能直接当作同一种指标。
- Qwen-VL Judge 仍不是人工金标准，后续需要人工抽样复核 best/worst cases。

相关文档与报告：

- `docs/superpowers/specs/7 mkgw-qwen-vl-blip-image-evaluation-design.zh.md`
- `docs/superpowers/plans/7 mkgw-qwen-vl-blip-image-evaluation.zh.md`
- `docs/evaluation/2026-07-09-qwen-vl-blip-image-evaluation-validation.zh.md`

## 已移除：原 Stage 9-11B 人工校准闭环

截至 2026-09-24，项目采用方案二：删除视觉人工复审、模型偏差拟合、风险门控校准和配置回写闭环。Stage 4 不再读取 reviewer consensus、calibration profile 或人工拟合偏差；CLIP、BLIP/ITM 和 Visual Judge 的原始 `0-100` 分数直接除以 `100` 并截断到 `[0, 1]`。

相关脚本、模块、测试、专属评估文档和 `outputs/` 校准产物已删除。现行 Stage 4 分数契约为 `stage4_structured_graph_v3`，旧校准结果仅属于已废弃实验，不再构成项目运行依赖。

## 阶段 12A：Stage 0-2 LMM 语义主链路入口

目标是对齐 MMKG Stage 0 和 Stage 3 设计：规则阶段负责清噪、压缩和 deterministic fine scoring，LMM Prompt/Qwen 阶段负责语义粗筛与 relation-selection refinement；同时保留显式 rule baseline 作为离线复现/debug 模式。

```bash
--stage02-planner qwen-local
--stage3-planner qwen-local
```

已实现：

- `mmkgc.run` 新增 `--stage02-planner {rule,qwen-local}`，默认 `qwen-local`。
- `mmkgc.run` 新增 `--stage3-planner {rule,qwen-local}`，默认 `qwen-local`，让 Stage 3 relation selection 进入 LMM 主链路，同时仍可显式回到 rule baseline。
- 旧参数 `--stage03-planner qwen-local` 保留为兼容别名，等价于同时启用 Stage 0-2 和 Stage 3；`--stage03-planner rule` 等价于同时关闭两者。
- Pipeline 支持只 refine Stage 0-2，不触碰 Stage 3。
- Stage 0 rule 阶段输出 `cleaned_relation_groups` 和 `compressed_evidence`；Qwen Stage 0 默认从这些清噪/压缩候选中生成最终 `selected_relations`，而不是只重排 rule baseline。
- Stage 0 已补 `high_recall_relation_candidates`，为每个关系候选提供 relation-level semantic summary、一跳/二跳 path context、target modality priority 和 neighbor modality priority；MKG-W data adapter 会为一跳邻居带入 bounded second-hop facts，Stage 0 过滤低价值二跳 metadata relation 后压缩进候选池，Qwen prompt 优先消费该高召回语义候选池。
- Stage 1/2 planner 先运行 deterministic stage，再用 Qwen 严格 JSON 结果丰富白名单业务字段；Stage 1 prompt 已明确禁止 `allowlisted_semantic_fields`、meta 字段和 schema placeholder，避免 Qwen 复述 contract。
- Stage 1 entity_profile 已补齐原始 Knowledge Modeling 设计中的 `secondary_types`、`type_confidence`、列表型 `semantic_domain`、结构化 `key_attributes`、`conflicting_evidence` 和 `uncertain_aspects`；Stage 5 ranking context 已同步白名单化这些字段。
- Stage 3 先运行 deterministic fine scoring/quota，再允许 Qwen 在既有 relation set 内 refine role、target modality 和 candidate quota。
- Qwen 非 JSON、parse error 或服务异常时，保留 deterministic 输出并写入 `semantic_refinement.status=fallback`。
- 严格 JSON parse 失败时先执行本地 contract salvage，回收截断响应中已经完整的 allowlisted 字段；salvage 不足时再走一次模型 contract repair。
- 每个语义 stage 调用会进入 `model_calls.jsonl`，用于审计是否真的走了 LMM。
- Prompt 不包含 ground truth modalities、缺失 description、人工 reviewer 分数或 evaluation 结果。

当前边界：

- 离线 deterministic baseline 仍可通过 `--stage02-planner rule` 显式启用。
- fixed50 deterministic vs Qwen semantic 对照已完成；Stage 3 prompt compact 后，100 实体验收已完成，Stage 0/1/2/3 为 `400/400 applied`。
- 500 实体 Stage 0-3 LMM 质量验收已完成：clean combined 目录 `outputs/stage03_qwen_500_20260723_combined_real_applied`，严格口径 `2000/2000` semantic calls applied，`remaining_entities = 0`。对照结果为 `stage0_relation_set_changed = 61`、`stage2_supported_facts_changed = 102`、`best_pair_changed = 31`、legacy best-pair 平均 delta `-0.002588`。
- Stage 0-2 LMM 输出仍受 deterministic evidence boundary 约束，不能引入原图谱证据中不存在的新 KG relation；当前 Stage 0 已支持 bounded 二跳 KG path evidence，但不会做更深层 path expansion 或跨实体全图搜索。

相关报告：

- `docs/evaluation/2026-07-23-stage03-qwen-500-comparison.zh.md`

## 阶段 12：BERTScore 文本语义联合评价

输入 run：`outputs/qwen_sdxl_500_20260712`

输出目录：`outputs/qwen_sdxl_500_20260712/evaluation_bertscore_20260715/`

本阶段补充本地 BERTScore 文本语义评价，不使用 Gemini，不覆盖既有 `evaluation/` 目录。

核心结论：

- 500 个实体、2500 个文本候选全部完成评分。
- 平均 BERTScore F1 为 `0.842020`。
- BERTScore 已在 Stage 13 中接入为 optional auxiliary scorer；它不参与 model-fusion ranking weight，只写入辅助指标和 optional failure/degraded 状态。

## 阶段 13：Stage 5 model-fusion ranking

目标是把原先的 legacy 词法 Stage 5 排序升级为默认启用的 model-fusion ranker，对齐原始 MMKG Stage 5 hard gate、多视角评分和 KG-aware fusion 设计；`legacy` 仅保留为显式 baseline/debug 行为。

已完成能力：

- `--stage5-ranker legacy|model-fusion` 和 `--stage5-config`；CLI 默认 `model-fusion`，`legacy` 需显式指定。
- `--image-generator auto` 为默认值；`image`/`both` + model-fusion 自动使用 `sdxl-local`，显式 `rule` 组合会在创建输出目录前被拒绝。
- 共享 `ModelFusionRankingEngine`，支持在线 pipeline 和已有 run 的 `mmkgc.rerank`。
- `RankingContext` 白名单化，排序上下文记录 `ground_truth_used=false`，不把 ground truth、人工评分或 evaluation 结果送入 Judge prompt。
- 接入 deterministic gate、Sentence-BERT、CLIP、BLIP/ITM、Qwen/Qwen-VL Judge。
- BERTScore 接入为 optional auxiliary scorer，不进入 final ranking weight。
- 视觉评分使用原始 `0-100` 分数除以 `100` 的无人工校准归一化；融合权重由 Stage 4 `candidate_score_weights` 明确记录。
- Stage 4 safer candidate regeneration：当 Stage 5 返回 `no_valid_candidate` 或 `no_valid_pair` 时，自动生成更安全候选并重跑一次 Stage 5。
- Qwen judge retry/坏候选隔离：超时和连接类异常可有限重试；重试后仍失败的单候选标记 `qwen_service_error` 并剔除，pair 失败则跳过该 pair。

5 实体真实 smoke：

```text
outputs/stage5_model_fusion_5_smoke_qwenvl_pixels_20260720
processed_entities = 5
failed_entities = 0
ranking_status = completed for 5/5
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
ranking_model_calls = 260
```

旧的 fixed50 人工 consensus held-out 对比已随校准闭环删除，不再作为当前 Stage 4 验收依据。

当前设计与报告：

- `docs/five-stage-pipeline-current-design.zh.md`
- `docs/five-stage-pipeline-explanation.zh.md`
- `docs/evaluation/2026-07-20-stage13-model-fusion-ranking.zh.md`

## 当前运行方式速查

规则 baseline：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m mmkgc.run --dataset-dir datasets/MKG-W --missing-mode both --limit 100
```

Qwen text smoke：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --output-root outputs \
  --run-name qwen_text_smoke_20260707
```

SDXL image smoke：

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

SDXL run 验收：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --expected-entities 5 \
  --expected-images-per-entity 5 \
  --expected-width 1024 \
  --expected-height 1024
```

CLIP image quality：

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

如果当前进程看不到 CUDA，把 `--clip-device cuda` 改为 `--clip-device cpu`。

Qwen + SDXL joint smoke：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260708 \
  --sdxl-guidance-scale 7.5 \
  --sdxl-device cuda:1 \
  --output-root outputs \
  --run-name qwen_sdxl_clip_joint_smoke_20260709
```

Qwen-VL Judge + BLIP/ITM image evaluation：

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

Stage 5 model-fusion fixed50 rerank：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.rerank \
  --run-dir outputs/qwen_sdxl_500_20260712 \
  --sample-manifest docs/evaluation/2026-07-17-stage5-model-fusion-fixed-50-manifest.json \
  --stage5-config configs/stage5-model-fusion-fixed50-v2.yaml \
  --output-dir outputs/qwen_sdxl_500_20260712/stage5_model_fusion_rerank_fixed50_v2
```

Stage 0-2 LMM 语义主链路：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --stage02-planner qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --output-root outputs \
  --run-name stage02_qwen_semantic_smoke_compact_20260721
```

当前 smoke 验收：`processed_entities = 5`，`failed_entities = 0`，Stage 0/1/2 语义调用 `15/15 applied`，semantic prompt 泄漏检查 `0/15`。报告见 `docs/evaluation/2026-07-21-stage02-qwen-semantic-smoke.zh.md`。

离线 rule baseline 需显式添加：

```bash
--stage02-planner rule
--stage3-planner rule
```

Stage 0-2 fixed50 对照：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/compare_stage02_semantic.py \
  --baseline-run-dir outputs/stage02_rule_fixed50_20260721 \
  --semantic-run-dir outputs/stage02_qwen_fixed50_stage0_sanitized_20260722 \
  --sample-manifest docs/evaluation/2026-07-17-stage5-model-fusion-fixed-50-manifest.json \
  --output docs/evaluation/2026-07-21-stage02-qwen-fixed50-comparison.json \
  --csv-output docs/evaluation/2026-07-21-stage02-qwen-fixed50-comparison.csv \
  --overwrite
```

当前 fixed50 对照：两组 run 均 `50/50` 成功；semantic calls `150/150 applied`，semantic fallback `0/150`，本地 salvage/repair `0/150`，prompt 泄漏检查 `0/150`；Stage4 text/image primary relation `4/50` 变化，Stage5 best pair `1/50` 变化，best-pair 均值 delta `+0.014313`。报告见 `docs/evaluation/2026-07-21-stage02-qwen-fixed50-comparison.zh.md`。

Stage 0-3 Qwen 100 实体验收：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/compare_stage02_semantic.py \
  --baseline-run-dir outputs/stage03_rule_100_20260722 \
  --semantic-run-dir outputs/stage03_qwen_100_stage3_compact_20260722 \
  --output docs/evaluation/2026-07-22-stage03-qwen-100-stage3-compact-comparison.json \
  --csv-output docs/evaluation/2026-07-22-stage03-qwen-100-stage3-compact-comparison.csv \
  --overwrite
```

当前 100 对照：两组 run 均 `100/100` 成功；Stage 0/1/2/3 semantic calls `400/400 applied`，Stage 3 fallback 从上一轮 `45/100` 降为 `0/100`；best-pair 均值 delta `+0.011724`，正/负/零 delta 为 `37/10/53`，best pair `10/100` 变化。报告见 `docs/evaluation/2026-07-22-stage03-qwen-100-stage3-compact-comparison.zh.md`。上一轮未 compact 的失败记录见 `docs/evaluation/2026-07-22-stage03-qwen-100-comparison.zh.md`。

离线评估：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m mmkgc.evaluate --run-dir <run_dir> --overwrite
```

## 后续建议

建议按 500 实体 Stage 9 校准、Stage 10 风险门控和 Stage 13 fixed50 结果放缓进入全量：

1. 不要直接进入全量 MKG-W 真实模型运行。
2. 下一步先填写 `outputs/qwen_sdxl_500_20260712/evaluation/stage5_high_risk_review_packet/review_template.jsonl` 中的 60 条高风险样本，重点复核 Qwen-VL 高估、BLIP 高估、CLIP 低估和 reviewer disagreement。
3. 基于新人工复核结果复跑 `scripts/calibrate_stage5_gate_from_review.py`，再调整 Qwen-VL judge rubric 与图像生成 prompt，重点处理 `entity_type_too_generic`、`relation_evidence_weak`、`kg_evidence_not_visually_checkable` 和 `needs_human_review`。
4. 调整后重新跑一轮 100 或 500 实体对照实验，确认 `needs_human_review` 多数标记和 reviewer disagreement 是否下降。
5. Stage 0-3 LMM 语义主链路已切换为默认 Qwen，并已完成 500 clean 验收：Stage0/1/2/3 工程 fallback 清零。下一步应对 500 对照中的正负 delta 样本做人工抽样判断，再决定是否扩大到全量。
6. 自动评价与人工抽样明显收敛后，再考虑全量 MKG-W 真实模型运行。
7. SDXL Refiner、SD3.5 或 Flux 可以作为后续图像模型对比实验，但不应替代当前评价校准问题。

当前最重要的限制是：500 实体 run 已完成生成、自动评价、Stage 9 三人 consensus 校准、Stage 10 风险门控、Stage 13 fixed50 held-out 对比，以及 60 条高风险人工复核包；Stage 13 v2 的 paired delta 为 `+0.013647`，但 `needs_human_review_majority_count = 2500/2500`，Stage 10 `accept = 0`，Qwen-VL Judge 明显偏高，且 777/2500 图触发 reviewer disagreement。因此，全量前仍需要填写高风险人工复核包，并基于新人工复核结果做更严格的 prompt/评价校准。
