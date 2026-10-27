# Stage 13 Model-Fusion Ranking Smoke 验收报告

日期：2026-07-20

## 范围

本报告记录 Stage 13 在线 model-fusion ranking v1 的当前验收状态。

本次只确认核心在线闭环：

```text
Stage 0-4
-> generated text candidates
-> generated SDXL PNG image bytes
-> RankingContext
-> deterministic / Qwen Judge gate
-> Sentence-BERT + CLIP + BLIP/ITM + Qwen/Qwen-VL Judge
-> fusion ranking
-> final.json / ranking_model_calls.jsonl
```

本报告的原始 smoke 不验收 replay / rerank CLI、fixed 50 held-out 质量对比、BERTScore 主排序、Gemini Embedding、全量 MKG-W run 或 Stage 0-3 语义重写。后续同日增量已补 BERTScore optional auxiliary scorer；它不参与 ranking weight。

## 对应文档

- Spec: `docs/superpowers/specs/13 mkgw-stage5-model-fusion-ranking-design.zh.md`
- Plan: `docs/superpowers/plans/13 mkgw-stage5-model-fusion-ranking.zh.md`
- Config: `configs/stage5-model-fusion.yaml`
- Run: `outputs/stage5_model_fusion_5_smoke_qwenvl_pixels_20260720`

## 运行配置

本次 smoke 使用：

```text
dataset_dir = datasets/MKG-W
missing_mode = both
limit = 5
candidate_budget = 5
text_generator = qwen-local
image_generator = sdxl-local
stage5_ranker = model-fusion
stage5_profile_name = full_local_v1
stage5_score_version = stage5_model_fusion_v1
ground_truth_used = false
sdxl_device = cuda:1
```

Stage 5 profile 中 required scorer 为：

```text
deterministic
sentence_bert
clip
blip_itm
gpt4omini_judge
```

## Run 级结果

`metrics_summary.json`：

```text
processed_entities = 5
failed_entities = 0
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
fallback_counts = {}
stage_failure_counts = {}
```

`final.json` 汇总：

```text
ranking_status completed = 5 / 5
stage5_ranker model-fusion = 5 / 5
candidate hard gate pass = 50 / 50
failed_optional_scorers = 0
ranking_model_calls = 260
```

逐模型调用计数：

```text
gpt4omini_judge text scored = 25
sentence_bert text scored = 25
gpt4omini_judge image scored = 25
clip image scored = 25
blip_itm image scored = 25
gpt4omini_judge pair scored = 45
clip pair scored = 45
blip_itm pair scored = 45
```

5 个实体的 best pair：

```text
Q1356749 -> text_002__image_002
Q1373204 -> text_005__image_002
Q234900  -> text_003__image_004
Q370252  -> text_001__image_001
Q3857815 -> text_001__image_004
```

## 验收点

本次 smoke 满足 Stage 13 第一版最低验收：

- 原始 smoke 证明显式 `--stage5-ranker model-fusion` 新链路可运行；后续默认主链路已切换为 `model-fusion`，`legacy` 仅保留为显式 baseline/debug。
- model-fusion 使用 Stage 1-4 输出和在线 SDXL image bytes，不依赖写盘后的 evaluation 产物。
- `config.json` 与 `final.json` 均记录 `ground_truth_used=false`。
- 所有 5 个实体均输出 `5_kg_aware_pair_ranking.json`、`final.json` 和 `ranking_model_calls.jsonl`。
- required scorer 均成功参与评分，未出现 optional scorer 降级。
- text/image 单候选先分别排序，再由 Top-3 x Top-3 形成 9 个 pair 并完成 pair ranking。
- `ranking_model_calls.jsonl` 记录模型调用状态；API key/token 等 secret-like 字段由 writer 脱敏策略排除。

## 已知限制

- 这只是 5 实体 smoke，不是 fixed held-out 质量结论。
- 当前结果证明链路可运行，不证明 model-fusion 一定优于 legacy。
- 本次原始 smoke 运行时 `model-fusion` 仍是显式启用路径；后续代码已将 Stage 5 默认主链路切换为 `model-fusion`，`legacy` 仅作为显式 baseline/debug。
- 本次原始 smoke 未覆盖 Stage 0-3 默认 Qwen 语义主链路；后续代码已将 Stage 0-3 默认 planner 切换为 `qwen-local`，相关大样本验收见后续 Stage 0-3 对照报告。
- 输出文件中的 `entity` 仍保留 `description` 和 `ground_truth_modalities` 供后评估使用；排序上下文通过 `RankingContext` 白名单避免读取这些字段。
- replay / rerank CLI、fixed 50 manifest、legacy 对比脚本后续已补；BERTScore 已作为 optional auxiliary scorer 接入，但主排序权重仍未使用 BERTScore。
- 当前 pair fusion 使用实现版冻结公式 `0.23 Q_text + 0.23 Q_image + 0.26 C_missing + 0.28 C_KG_pair - pair_penalty`。由于 `both` 模式没有稳定 existing modality，公式有意移除原始草案中的 `C_existing` 并重新归一化；这属于 Stage 5 model-fusion v1 的合理适配。

## 结论

Stage 13 在线 model-fusion ranking v1 可以记为工程 smoke 通过。

建议下一步进入 fixed 50 held-out 或最小 replay 验收，用同一批实体对比 `legacy` 与 `model-fusion` 的 Top-1、review/reject 风险和人工抽样一致性，再决定是否把 `model-fusion` 从 opt-in 推向默认候选路径。
