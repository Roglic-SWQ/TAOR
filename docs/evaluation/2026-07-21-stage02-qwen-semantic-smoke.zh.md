# Stage 0-2 Qwen semantic mainline smoke 验收

日期：2026-07-21

## 范围

本次验证 Stage 0-2 是否能通过 `--stage02-planner qwen-local` 接入 Qwen 语义 planner，并保持确定性 Stage 0-2 作为 fallback 与证据边界。

最终验收 run：

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

Qwen health check：

```text
ok = true
base_url = http://localhost:12345/v1
model_name = Qwen/Qwen2-VL-7B-Instruct
```

## 结果

```text
run_dir = outputs/stage02_qwen_semantic_smoke_compact_20260721
processed_entities = 5
failed_entities = 0
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
stage_failure_counts = {}
```

Stage 0-2 semantic model calls：

```text
stage0_coarse     applied = 5 / 5
stage1_knowledge  applied = 5 / 5
stage2_reasoning  applied = 5 / 5
total             applied = 15 / 15
```

实体覆盖：

```text
Q1356749  stage0/stage1/stage2 applied
Q1373204  stage0/stage1/stage2 applied
Q234900   stage0/stage1/stage2 applied
Q370252   stage0/stage1/stage2 applied
Q3857815  stage0/stage1/stage2 applied
```

Prompt 泄漏检查：

```text
checked_semantic_calls = 15
semantic_prompt_leaks = []
```

检查项覆盖 `ground_truth_modalities`、测试密钥/令牌片段、以及 record 级 target description。未发现进入 semantic prompt。

## 修复点

- 将 Stage 0-2 Qwen prompt 从整包 JSON 请求改成“指令 + Response JSON shape + INPUT_JSON”，避免模型把 `contract`、`deterministic_output`、`prior_stage_outputs` 原样复述。
- Stage 2 prompt 改为紧凑事实摘要，不再把完整 `modality_constraints` / constraint matrix 发给 Qwen，避免长响应被截断。
- Stage 2 语义输出只作为紧凑增量更新 `supported_facts`、`safe_inferences`、`uncertain_facts`、`reasoning_confidence` 和 `reasoning_summary`；完整约束矩阵仍由 deterministic Stage 2 维护。
- 新增测试覆盖 prompt envelope 回显时必须 fallback，防止错误格式被静默接收。

## 中间失败记录

- `stage02_qwen_semantic_smoke_20260721`：15/15 semantic calls fallback。原因是 Qwen 原样复述整包输入。
- `stage02_qwen_semantic_smoke_promptfix_20260721`：Stage 0/1 applied，Stage 2 仍 5/5 fallback。原因是 Stage 2 返回完整约束矩阵并被 token 截断。
- `stage02_qwen_semantic_smoke_contractfix_20260721`：Stage 0/1 applied，Stage 2 2/5 applied。剩余失败仍由 Stage 2 长响应截断触发。

## 结论

Stage 0-2 Qwen semantic mainline 的工程接入、fallback 记录、model call 留痕和 prompt 泄漏边界通过 5 实体 smoke 验收。当前仍需要更大样本质量对照，建议下一步跑 fixed50 或 50/100 deterministic vs qwen-local Stage 0-2 对比。
