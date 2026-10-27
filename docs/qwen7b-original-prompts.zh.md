# 千问 7B 原始提示词恢复记录

2026-10-27 按用户要求，从当天修改 GPT 提示词前的完整备份中恢复 Stage 0、Stage 1、
Stage 3 文本生成和 Stage 3 图像提示词生成四处模板。

原始模板集中保存在 `mmkgc/qwen7b_prompts.py`。千问配置显式设置：

```yaml
inference:
  prompt_profile: qwen7b_original_v1
```

默认 CLI、Python API 和 4 / 10 / 50 实体脚本使用的 `qwen7b-mainline.yaml`、
`qwen7b-stage1.yaml`，以及可选的 `qwen-local.yaml` 均已启用该设置。
调用日志和候选元数据记录 `qwen7b_original_v1`，便于区分之前的 GPT 实验。

| 位置 | 恢复内容 |
| --- | --- |
| Stage 0 | 原始关系粗筛任务说明、system 指令和输入 JSON；恢复紧凑列表最多 3 项的文字要求 |
| Stage 1 | 原始知识建模说明、system 指令及格式修正提示；仍要求每个输入 fact_id 返回一个 claim |
| Stage 3 文本 | 原始百科描述 system、实体／类型／关系／角度／Grounded facts 模板，输出 1～2 句英文 |
| Stage 3 图像 | 原始 SDXL planner system 与图像 JSON 模板，保留 positive_prompt、used_text_cues、omitted_risky_details |

Stage 3 同时恢复原来的 generation brief 输入方式、各 brief 列表取前 4 项、图像
grounded_facts 取前 8 项，以及程序在 SDXL 正向提示词后附加的 brief 片段。
Stage 4 反馈触发重生成时，继续使用原来的反馈事实前缀和提示词形式。

这次恢复保留当前的 Stage 1 Qwen3-Reranker-4B、Stage 2 GPT 槽位方案及严格校验、
Stage 3 并发和线程日志隔离、Stage 4 评分配置。千问 Stage 3 默认并发 1，仍可配置。
显式使用 `gpt4omini-mainline.yaml` 时继续使用 GPT v2 提示词。

`tests/fixtures/qwen7b_original_prompts.json` 的参考摘要由修改前备份的真实 Python
实现离线生成，记录源文件 SHA-256。回归测试逐条检查原始 HTTP 请求体（含 system
和 user 消息），覆盖文本／图像模式下 Stage 0 / 1 的首次调用和格式修正，以及
Stage 3 的初次生成、反馈重生成与最终 SDXL 提示词。另验证新 Stage 2 槽位顺序在
原始千问提示词并发调用后仍被保留，且候选与日志正确对应。

本次验证不调用外部 API 或启动本地权重推理；实际效果需启动千问服务后复测。
