# 千问 Stage 2：漏写事实与候选槽位定向修正

先前实测的失败记录位于 `outputs/runtime_checks/stage2_qwen_switch_20261028_223014/`。
Delhi 输入 12 条事实、Bologna 输入 9 条，千问反复只返回前 5 条；Billy Talent
初次把 `fact:` / `path:` 证据 ID 用作候选锚点，完整重生成后又只返回 3 个槽位。
该记录为 4/7 实体通过，另两个原始错误回放修正案例为 1/2 通过。

## 修改方式

初次千问规划提示词 `qwen7b_stage2_candidate_slots_v3` 和模型路由不变。
对独立可修正的错误，使用 `qwen_stage2_targeted_repair_v1`，不再要求千问重复输出整个计划。

- 仅漏写事实时，保留已有审查结果和候选锚点，只请求遗漏事实的角色、原证据引用、支持项和理由。
  每批至多 4 条，输出以实际事实 ID 为键的 `facts` 对象。没有槽位的事实禁止选择 primary，
  但 secondary / context / excluded 仍由千问按证据决定，不自动补上角色或 claim。
- 仅槽位错误时，保留全部事实审查结果，请求 `slots` 对象，键为 `slot_001` 至预算对应的最后一项。
  每个值由千问从合法 primary / secondary 事实 ID 中选择，允许重复；证据 ID 不属于合法选项。
  程序按槽位编号转为数组，不自动延长、复制或均分模型的配额。
- 仅支持引用错误时，只请求出错事实的 `supports`，限制为其他 secondary 事实；所有角色、
  证据、理由和候选槽位保留。context / excluded 事实误写支持链接也属于此类，
  其合法支持列表为空，仍由模型显式返回 []，程序不直接删除链接。
- 其他混合错误继续采用完整规划修正；不能孤立修正时不冻结已知无效决策。

完整初次规划仍使用普通生成接口，远程 GPT 对照继续使用原有逐事实 Schema。
只有定向修正的小响应使用 lm-format-enforcer JSON 约束，字段直接按事实或槽位键展开，
不使用全局逐事实 anyOf；reason 最长 160 字符，不组合非空白正则和长度限制。
最终非空白要求及跨字段业务规则继续本地校验。

每个模型补答必须精确返回所请求的全部键；多键、少键、重复键、错误字段均拒绝。
程序只合并千问实际返回的字段，合并后再调用完整的 `apply_candidate_slot_plans`，检查事实完整性、
证据归属、角色与配额、支持方向及冻结决策。补答失败或完整校验失败不会作为成功结果输出。

默认仍为首次请求外最多 2 轮业务修正；遗漏项一轮可能分多个批次调用。所有批次、重试、修正
共用 Stage 2 的 300 秒预算，每次网络调用仍只额外重试一次。
日志 `targeted_repairs` 保存批次、所需键、提示词、原始补答；`validation_attempts` 标记
`full_model_response` 或 `model_patch_merge`，每次网络调用另记 `targeted_repair_kind`。
`merged_pending_full_validation` 只表示结构上可合并，不能视为整体计划通过。

## 验证范围

离线测试新增 `tests/test_stage2_qwen_targeted_repair.py`，覆盖 7 条缺失事实分批补答、
5 个明确槽位、证据与候选 ID 混淆、已有决策不被覆盖、错误补答严格拒绝、
完整校验不被绕过、修正上限以及仅定向请求启用小 Schema。

真实验证脚本：`outputs/runtime_checks/verify_stage2_qwen_targeted_20261028.py`。
7 个原始实体重新调用 Stage 2；另回放 2 个原始支持关系错误和 3 个千问 v3 错误首次响应，
后续修正均调用真实千问。验证不执行绘图和 Stage 4，不代表已重跑新的 50 实体五阶段测试。

## 最终验证结果

相关离线测试 336 项通过。真实重新规划 7/7 个实体通过，历史错误回放后的真实修正 5/5 个案例通过。
成功的新规划平均 22.393 秒，中位数 9.606 秒，包含初次规划及修正；网络重试 0 次。

| 实体 | 状态 | 事实数 | 槽位数 | 秒 | 修正轮数 |
| --- | --- | ---: | ---: | ---: | ---: |
| Faye Dunaway (Q168721) | passed | 3 | 5 | 6.519 | 0 |
| Cheyenne Autumn (Q1071402) | passed | 5 | 5 | 9.606 | 0 |
| Delhi (Q1353) | passed | 12 | 5 | 67.571 | 2 |
| Cluj-Napoca (Q100188) | passed | 3 | 5 | 6.407 | 0 |
| Bologna (Q1891) | passed | 9 | 5 | 49.676 | 2 |
| Billy Talent (Q154815) | passed | 3 | 5 | 12.241 | 1 |
| City College of New York (Q1093910) | passed | 2 | 5 | 4.732 | 0 |

完整原始响应及修正审计：`outputs/runtime_checks/stage2_qwen_targeted_20261028_225811/`。
源码、测试摘要与计时验证：`outputs/runtime_checks/20261028_stage2_qwen_targeted_verification.json`。
本次仅验证 Stage 2，未执行新的 50 实体完整流程；结果不能保证后续请求全部通过。
