# 剩余 8 个失败实体的恢复优化

原测试为 `outputs/mkgw_image_test_50_qwen7b_stage1repair_20261028_135438`：
42 个实体完成，8 个失败。Stage 1 全部通过，剩余问题是 Stage 2 的 5 个网络故障、
2 个支持关系错误，以及 Stage 3 的 1 个提示词重复扩写／截断。

## Stage 2 网络恢复

每次模型调用默认最多额外网络重试 1 次，与两次业务契约修正分别计数，
整个 Stage 2 的所有调用共用总时间预算。
超时、连接故障、HTTP 408/429/500/502/503/504 可以重试，其他 4xx 不重试。
短退避与最长 60 秒的数值 Retry-After 均受阶段剩余时间限制。

HTTPX 异步取消限制整个请求耗时；连接阶段最多等待 10 秒，单次总时限沿用
`timeout_seconds`，Stage 2 默认总预算 300 秒，包含重试、退避和模型修正。
Python API 可配置 `max_transport_retries`、`retry_backoff_seconds`、`stage_timeout_seconds`。
审计记录各轮请求编号、服务端返回的请求编号、状态、耗时和异常类型链，不记录密钥。

## Stage 2 支持关系修正

初次提示词版本为 `gpt4omini_stage2_candidate_slots_v4`，明确链接方向：
A 为 primary、B 为 secondary 时，A 可引用 B，B 不能反向引用 A。

修正请求使用原始输入、全部错误和完整事实模板，不再追加错误 assistant 响应。
仅有 `invalid_support_target` 错误时，保留已通过校验的状态、角色和候选槽位顺序，
按这些角色生成合法 secondary 支持 ID 枚举并收紧修正 Schema。本地也校验冻结决策。
模型重新选择支持项或明确返回空列表；程序不直接删链接后接受计划。
其他错误仍允许完整重新规划；重复无效响应和修正耗尽继续严格退出。

## Stage 3 本地千问截断恢复

初次千问请求和 SDXL 参数保持原样。positive_prompt 超过 80 词／650 字符，或相同
四词短语重复至少 3 次，触发修正。finish_reason=length 也视为未正常完成。

本地 Qwen 修正使用原始事实、身份和渲染约束构造独立短请求，不引用错误原文。
修正启用 vLLM guided_json，指定已安装的 lm-format-enforcer 后端，重复惩罚为 1.1。
Schema 限制 positive_prompt 最多 400 字符，辅助列表各最多 3 项，每项最多 60 字符。
最多额外修正两次；保存各轮原文、结束原因与错误，不在本地补齐 JSON 或编造提示词。
当前千问的 384 token 输出预算不变；其他接口保留原来的一次修正行为。
JSON 结构限制不保证语义正确，最终仍需 Stage 4 评分。

## 验证

313 项相关离线测试通过，覆盖网络取消和重试、阶段预算、永久错误不重试、
动态支持 Schema、冻结决策、图像截断恢复及原始千问 HTTP 请求参考摘要。
JUnit：`outputs/runtime_checks/20261028_failed8_recovery_tests.xml`。

真实接口定向验证覆盖原来的 8 个案例：Stage 2 的 7 个通过；Cluj-Napoca 和 Bologna
复用原始错误初次响应后，各由真实 GPT 修正一次通过。Delhi 首次连接超时后由重试恢复，
Cluj-Napoca 的修正请求约 120 秒超时后也由重试恢复。A King in New York 复用原始
截断响应，真实千问修正一次返回完整 JSON，结束原因 stop。

产物目录：

- `outputs/runtime_checks/mkgw_failed8_recovery_stage_checks_20261028_200904/`：7 个 Stage 2 案例。
- `outputs/runtime_checks/mkgw_failed8_recovery_stage_checks_20261028_202613/`：最终千问截断复测。

中间验证发现旧千问推理进程退出，以及默认 Outlines 缺少 pyairports；这些失败记录保留。
重启服务并选择已有的 lm-format-enforcer 后，最终定向验证通过，没有安装或替换环境依赖。
定向验证不包含后续绘图，网络故障仍可能再次出现，不能据此保证后续请求全部成功。

完整八实体五阶段重测：`outputs/mkgw_failed8_recovery_full_20261028_202827/`。
该轮使用 GPT-4o-mini 规划 Stage 2，重新调用全部阶段，不复用定向验证响应。
最终 5/8 完成五阶段，成功实体平均 325.777 秒，中位数 301.579 秒。
Delhi 在 300 秒 Stage 2 总预算内遇到两次网络超时和事实／支持关系契约错误，最终预算耗尽；
Bologna、City College of New York 均为初次请求和额外重试各约 120 秒超时，没有获得模型计划。
Cheyenne Autumn 的网络重试及 A King in New York 的提示词修正在实际完整流程中成功。
该结果说明恢复机制有效，但持续接口故障仍会退出，不能宣称八实体已全部恢复。
查看 timing_summary.zh.md、entity_timing.csv、comparison.zh.md 和 final_verification.json。

图片统计口径：`final.counts.image` 表示参与排序的候选数，包含被拒绝的候选；
实际通过筛选数量应读取 `image_ranking.top_5_candidate_ids`。
本次复测保存并排序 25 张候选图，实际通过筛选 20 张；Stage 4 有 5 条候选级 GPT 评分失败记录。
耗时报告已据 top_5_candidate_ids 更正入选数量，原计数保留于 timing_progress.original_candidate_counts.json；
逐实体数量见 quality_summary.zh.md。实体完成不代表其全部候选均已通过评分。
