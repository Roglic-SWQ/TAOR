# Stage 1 事实 ID 漏项与修正机制

## 与旧版千问流程的差异

2026-10-26 的代码备份中，`knowledge_modeling` 先按选定关系组构建事实并生成
`rf:*` ID，千问返回画像补丁和事实评估，原有事实表继续保留。它不要求模型重新
逐条返回全部事实，因此不会因漏写某条模型 claim 而出现当前的完整性校验错误。

当前 `qwen_grounded_knowledge_v1` 将关系组展开为逐个邻居／三元组的原子事实，
要求模型为每个 `fact_candidate` 返回且仅返回一条 claim。原始模型提示词恢复后，
该数据契约仍然保留。例如本轮 Farrah Fawcett 输入 4 个事实 ID，模型只返回
`rf:0001`；Danny Markov 输入 4 个 ID，模型返回 9 行并重复使用 ID。

原来的修正入口只识别 JSON 解析错误。上述业务校验抛出 `ValueError`，不会触发
修正请求，所以本轮最初 13 次 Stage 1 漏项／重复 ID 失败的修正次数均为 0。

## 修复

- `KnowledgeResponseValidationError` 以结构化列表保存模型输出错误；校验一次收集
  缺失 ID、重复／未知 ID、空 claim、错误字段类型及无效证据引用，并给出 JSON Pointer。
- Stage 1 初次请求继续使用所选提示词配置，千问原始提示词保持不变。
- 业务校验失败时，修正请求明确列出完整 ID 集合、实际缺失 ID、目标数量、每个 ID
  对应的原始主语／关系／宾语／方向、所有校验错误及原始证据，要求完整重写知识模型。
  输出模板按真实输入展开，每个实际 ID 占一行，模型只需逐行填写 claim，避免把单行示例当作目标数量。
  修正请求不包含错误响应原文，避免模型照抄错误 ID；原始响应完整保存在调用审计中。
  同时明确 key_attributes 的 value/type/source_ref 对象格式及可用证据 ID。
- 默认最多额外请求两次；JSON 解析错误和业务校验错误共享同一预算。重复无效业务 JSON
  即使只变了空格或对象键顺序，也会提前停止。初次 JSON 解析错误仍先使用原始格式修正；
  若该请求仍返回无效 JSON，剩余一次预算改用完整源数据重建。网络错误直接记录，不额外进行传输重试。
- 程序仍要求完整、一一对应的事实表；不补写 claim、不重编号、不删事实，不修改
  来源主客体、方向或证据绑定。最终响应通过同一套严格校验后才进入 Reranker。
- 模型调用日志新增 `knowledge_repair_version: stage1_complete_fact_repair_v3`，
  `validation_attempts` 保留各轮响应和错误，`contract_repair_retries` 记录实际修正次数。

`StrictJsonStagePlanner.max_knowledge_repairs` 可在 Python API 中配置预算；设为 0
表示业务契约失败时直接终止。修复不改变 Stage 2 槽位、Stage 3 模板或 Stage 4 配置。

## 验证与实验边界

290 项相关离线测试通过，包括原始提示词 HTTP 请求参考摘要、修正成功／失败、
重复输出停止、预算耗尽、网络超时、来源字段不变以及 Stage 2 槽位和 Stage 3 并发。

`mkgw_image_test_50_qwen7b_original_20261027_225729` 已结束，原版本结果为
19 个成功、31 个失败，其中 29 个在 Stage 1 知识建模失败，2 个在 Stage 2 失败。
该轮使用启动时加载的原版本，保留原始时间基线。独立重放任务在该轮结束后才调用本地千问：
初次输出使用这轮实际失败响应的缓存，修正请求调用真实 Qwen2-VL-7B-Instruct。
这验证 Stage 1 修正能力，不代表重新跑完五阶段或重新生成图片。

重放产物位于 `outputs/runtime_checks/stage1_fact_id_repair_live_20261028_000743/`，
查看 `summary.zh.md` 和 `progress.json`；逐实体完整调用记录在 `entities/`。

## 2026-10-28 真实千问验证结果

原 50 实体实验中全部 29 个 Stage 1 失败实体均通过修正重放（29/29），
严格校验 153 条事实的 ID 和来源绑定。修正请求次数分布为：1 次请求 26 个实体、2 次请求 3 个实体。
逐实体修正耗时平均 11.59 秒、中位数 7.96 秒，
范围 5.27～35.70 秒。耗时只含修正请求及本地校验。

290 项相关离线测试通过，JUnit 为
`outputs/runtime_checks/20261028_stage1_fact_id_repair_tests.xml`。
机器复核凭据为上述重放目录下的 `verification.json`。
本结果不包含 Reranker、Stage 2～4、绘图或新的首次模型生成；原 50 实体报告仍是原版本基线。
