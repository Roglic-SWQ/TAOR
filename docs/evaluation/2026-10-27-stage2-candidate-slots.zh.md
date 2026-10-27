# Stage 2 将事实保留与候选配额分开

## 改动目的

旧接口要求模型为每个事实直接填写 `candidate_quota`。部分模型响应把 6 或 7 个
保留事实各分配一个候选，超过每实体 5 个候选的预算，即使修正后仍重复错误。

新提示词为 `gpt4omini_stage2_candidate_slots_v3`。模型分别完成两项决定：

- `relation_plans`：每个输入事实恰好保留一条角色、证据引用、支持关系和理由，
  不再填写 `candidate_quota`。
- `candidate_anchors`：恰好选择 `candidate_budget` 个事实 ID，每个 ID 对应一个
  候选。允许重复；不出现在该数组中的事实仍可保留为零配额的辅助事实。

例如保留 7 个事实，候选分配为：

```json
{
  "candidate_anchors": ["rf:0001", "rf:0001", "rf:0001", "rf:0002", "rf:0002"]
}
```

程序只统计模型所选 ID 的次数：`rf:0001` 为 3、`rf:0002` 为 2，其余为 0。
候选分配决策由模型给出，程序不擅自增加、删除或重分配候选。

## Schema 与校验

完整模型响应为 `{"plan": {"status": ..., "relation_plans": [...], "candidate_anchors": [...]}}`。
`plan` 下使用两个 Schema 分支：

| 状态 | 候选槽位 | 事实角色与支持关系 |
| --- | --- | --- |
| `ready` | 长度恰好等于预算，ID 必须来自输入事实 | 本地检查至少一个主事实，每个主事实至少有一个槽位；槽位仅能指向主／辅助事实 |
| `insufficient_evidence` | 必须为空 | Schema 限制角色只能为 `context` / `excluded`，支持关系必须为空 |

输入没有事实时 Schema 只提供 `insufficient_evidence` 分支。
每个事实的证据枚举、非空理由和完整事实覆盖仍保留；非法支持角色、重复事实、
未知 ID、槽位数量错误或遗留的 `candidate_quota` 字段均会被拒绝并反馈。

正常输出仍转换为下游使用的 `model_relation_plan_v1`，保持派生配额字段。
额外保存 `allocation_method: model_candidate_slots_v2` 和原始 `candidate_anchors`。
Stage 3 按槽位数组顺序生成，Stage 2 JSON 产物记录槽位和派生配额。
旧配额 Schema 与校验函数保留用于历史响应验证；当前 GPT 规划器只接受新槽位接口。

## 验证

新增 41 项槽位相关测试，与现有定向回归合计 228 项通过：

- 保留 6、7、20 个事实，仍只分配 5 个候选，零配额辅助事实不被删除。
- 预算 1 / 3 / 5 / 10，支持重复锚点，文本与图像两种模态。
- 错误数量、未知 ID、状态矛盾、遗漏／重复事实、空理由和错误证据／支持链接。
- 模型输入与输出不被改写，原始候选顺序传递到 Stage 3，审计产物保存槽位。

JUnit：`outputs/runtime_checks/20261027_stage2_candidate_slots_tests.xml`。

另外，仅对此前重复返回配额错误的两个实体重测 Stage 2，复用原始事实输入：

| 实体 | 保留事实数 | 候选数 | 零配额辅助事实数 | 模型修正次数 | Stage 2 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1068968 / A King in New York | 6 | 5 | 1 | 0 | 11.567 秒 |
| Q1071402 / Cheyenne Autumn | 7 | 5 | 2 | 0 | 10.829 秒 |

两者均首次返回即通过。输入 payload 与原记录一致，未重新执行 Stage 0 / 1，
未生成图片，也未运行 Stage 4。完整调用记录：
`outputs/runtime_checks/20261027_stage2_candidate_slots_live.json`。

该验证针对配额表示及两例真实请求；不代表解决所有语义错误或网络超时。
另外 3 个修正超时实体未在本次重测，网络重试策略也未改动。
