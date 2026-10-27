# Stage 2 Schema 与修正反馈改进（2026-10-27）

Stage 2 使用 `gpt4omini_stage2_constraints_v2`，针对已停止的 50 实体测试中出现的
证据串用、理由为空、配额总和错误和支持角色错误进行约束与反馈改进。

## 修改内容

- `mmkgc/stages/relation_planning.py`：在数组元素内部按事实建立 `anyOf` 分支，
  用单一 `fact_id` 和该事实自己的证据 `enum` 绑定引用范围。没有证据时只允许
  空列表；所有角色的理由要求至少有一个非空白字符。支持 ID 的全集用 `$defs`
  共享，避免每个事实都重复展开全集。
- `relation_plan_errors()`：一次返回能够独立判断的全部错误，附 JSON Pointer、
  实际值、完整合法值、错误证据所属事实、支持目标当前角色以及配额明细。
  `apply_relation_plans()` 汇总报错后才转换计划，继续拒绝无效的事实、角色、
  引用和配额，不在本地删除引用、改变角色或补配额。
- `mmkgc/adapters/relation_planner.py`：修正请求保留最初输入，并将上一轮输出和
  完整错误列表分别作为 assistant / user 消息。默认最多两轮修正；重复无效
  JSON（忽略空白及对象键顺序）提前停止，网络异常不进入契约修正循环。
- 每轮响应和校验错误保存在 `validation_attempts`；停止原因保存在
  `contract_repair_stopped_reason`。原有 API 密钥日志脱敏继续生效。

保留 `model_relation_plan_v1` 和 `candidate_quota`，未迁移为候选槽位格式。
Stage 3 的输入、每实体候选预算和 Stage 4 的并发设置均保持现有接口。

## 验证结果

164 项相关离线测试通过，涵盖严格 Schema、完整反馈、多轮修正、有界停止、
原始事实不变、HTTP 消息结构、日志脱敏及 Stage 3 / Stage 4 的相关回归。
JUnit 记录：`outputs/runtime_checks/20261027_stage2_contract_tests.xml`。

离线回放 `mkgw_image_test_50_gpt4omini_v2_20261027_141437` 的已有响应：

| 检查 | 结果 |
| --- | --- |
| 失败实体的无效计划被业务校验拒绝 | 16 / 16 |
| 失败实体仅由新 Schema 即可拒绝 | 13 / 16 |
| 原成功实体继续通过 Schema 和业务校验 | 4 / 4 |
| 原分析发现的问题类别均能检出 | 是 |
| 实际 GPT 请求 / 图片生成 / 流水线重启 | 0 / 0 / 否 |

按实体计数且允许重叠：跨事实证据 10、配额错误 4、支持角色错误 3、空理由 5。
完整反馈示例及源记录 SHA-256 保存在
`outputs/runtime_checks/20261027_stage2_contract_replay.json`。

这些结果验证了错误检测、反馈内容和修正请求机制。尚未测量真实 GPT-4o-mini
接口的新修正成功率及中转接口对新 Schema 的实际支持情况；离线回放不代表
16 个失败实体已经由模型重新生成合格计划。原运行结果未覆盖。

## 定向测试入口

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_stage2_contract.py tests/test_grounded_model_mainline.py
```
