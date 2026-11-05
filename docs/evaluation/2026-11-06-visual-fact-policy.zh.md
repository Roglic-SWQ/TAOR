# 视觉观察的相关性、来源权重与不确定性

本次改动处理 Stage 1 将图片猜测升级为事实、Stage 2 又将其设为必写参照的问题。图片描述不再要求穷举场景，优先保留目标的区分性属性及理解目标所必需的场景信息。没有要求抽满固定数量。

## 输入合同与使用规则

新生成的图片描述通过 `with_image_description` 启用 `visual_observation_policy_v1`。每个图片抽取项增加必填 `visual_support`：

- `basis`：`direct`（整个断言可直接观察）或 `inferred`（推断）；
- `scope`：`target`（目标属性／动作）、`context`（必要场景）或 `background`（无关背景）；
- `confidence`：有限的 0～1 模型自评数值；
- `uncertainty`：保留猜测、或关系、冲突及限定条件；推断必须填写非空说明。

同一抽取请求内的非图片项使用 `visual_support: null`。缺失或不合法元数据会进入原有有界合同修复，不默认补成高置信度。提示词明确禁止将“可能 A 或 B”拆成确定的 A 和 B，也不能将未核验视觉猜测放入实体身份摘要。

每条来源判断连同证据 ID 保存在 `visual_observations` 中；原图与描述仍在 evidence catalogue 中追溯。推断、非空不确定性、背景项以及自评低于 0.8 的观察，不进入正向生成事实。它们留在 Stage 1 完整目录与审计记录中，不因被筛掉而删除。程序也补入未被模型主动填写的不确定性记录。

可用直接目标观察的 `source_weight=confidence`，必要场景为 `0.5×confidence`。已有图谱／文本／音视频断言沿用原有来源策略，基准权重为 1；这不宣称它们绝对真实。相关性原分 `relevance_score` 保留，检索使用 `selection_score=relevance_score×source_weight`，再由 Stage 2 选择一个精简、互补的事实集合。权重和元数据继续传给生成器及语义 Judge，不把权重命名成真实性。

**0.8 和 0.5 是首版保守策略参数，未经样本集校准。** 同图描述不是第二份独立证据，多图也只保留各自判断、取可用最大权重，不按重复次数累加。弱视觉猜测若恰好与图谱重复，不降低图谱自身权重，也不将该猜测引用并入其正向支持引用。

过滤同时发生于相关性检索和 Stage 2 的可选 ID 构建；不运行 reranker 也不能绕过。编译生成任务时再次拒绝不可用观察，避免其成为 must_include 或 Stage 4 正向参照。未启用该策略的历史产物保持原解释，不回写或重新标注旧结果。

## 验证范围

针对性回归覆盖源描述的猜测／或关系、置信度为 1 的推断仍被隔离、高相关性不能覆盖不确定性、必要场景降权、无 reranker 路径、错误元数据、跨批次合并与同源重复、图谱重复事实的来源分离，以及权重进入 Stage 2／Judge／落盘记录。

本实现不声称程序能够从像素判断所有模型自评是否诚实：若模型把错误分类标成 direct 且无不确定性，仍可能漏过。后续需在冻结实体上校准和人工核验；独立 Judge、原始来源争议复核与上游事实版本修订仍是单独工作，尚未由本改动解决。

实际单卡入口：`scripts/run_wn18rr_visual_policy_cluster.sh`。它新建 attempt，先固定旧失败描述和同一原图，只重做 Stage 1 抽取，再运行单实体 Qwen 文本全流程以及真实 SDXL＋Stage 4。固定描述控制用于区分“描述变好了”和“抽取保留了不确定性”，不能单独作为全闭环质量改善证明。

自动回归：图片／多图／音视频／三元组／共享选择／相关性／事实反思和新增策略测试 180 passed，1 skipped（当前推理环境未装 lm-format-enforcer）。另查 ranking context、Judge 和候选计划相关 266 项：当前与修改前 `2116d0b` 均为 202 passed、64 failed，逐项失败集合完全相同，无新增失败。63 项旧 context 测试主要仍使用已被主线移除的 missing_mode=both，另 1 项旧 Qwen Judge 摘要预期不匹配。保留该现状，不以修改旧测试预期掩盖失败。基线与当前 JUnit／日志在 `outputs/wn18rr-visual-policy-20261106/verification/`。

首次单卡尝试 `taor-visual-policy-1106-r1`：真实 Qwen 初次输出缺少新增字段，触发合同修复；vLLM 0.11 默认 auto backend 未按请求中的旧 guided_decoding_backend 字段选用 LM Format Enforcer，语法编译以 `Unimplemented keys: uniqueItems` 返回 400。该次保留为失败，停止的仅是本次诊断作业。新的单卡入口显式指定服务端 `structured-outputs-config.backend=lm-format-enforcer`；完整 schema 与 Python 校验均保持，未删除去重约束或给缺失元数据补默认高置信度。

第二次尝试 `r2`：固定旧描述控制成功执行，但发现模型直接复制可见的图谱 mouse deer，把来源误标为 image direct/confidence=1，并未保留旧描述中的相关猜测。这不算来源语义保留成功。文本全流程结果已保存，随后仅停止本诊断作业，保留所有产物。由此增加输入隔离：新视觉策略的 perception payload 不展示图谱实例、关系计数和图谱证据内容；实体 profile 同样不能引用未展示图谱。完整图谱继续由程序保留、合并，未丢弃。明确要求将有意义的未决源猜测写入 uncertain_aspects，即使不作为抽取三元组。该改动避免照抄图谱后再伪装成视觉证据，不保证消除由实体名称或模型记忆产生的推断。

同时复核 `r2` 已完成的文本结果，发现 Stage 3 将整段原描述中的 rodent／marsupial／bandicoot 和背景细节重新写入，绕过仅含 mouse deer 的精选事实；因此只隔离 Stage 2 仍不够。新策略下，Stage 3 仍收到所选事实、逐源判断和来源 ID，但不再收到完整机器图片描述的 content。原描述只在 Stage 1 完整保留；图谱和其他原始来源的内容策略保持。新增回归确认未选猜测不会从原描述绕回生成 payload。

`r3` 固定描述控制达到原脚本默认的 60 秒单请求上限而被服务中止，不算语义成功或失败。控制脚本单请求预算改为与完整 CLI 一致的 120 秒，Stage 1 总预算仍为 180 秒，原生输出 token 预算不变。新视觉合同从第一条本地 Qwen 抽取请求即采用完整 JSON schema，减少先遗漏元数据、再重做整段抽取的额外调用；源描述请求仍是自然语言输出。最终针对性回归 183 passed，1 skipped；已另在真实服务环境验证完整 LM Format Enforcer grammar。

`r4`：固定旧描述与完整文本／真实 SDXL 均完成生成，原 rodent／marsupial／bandicoot 未进入精选事实或最终文本，8 条事实中选择 6 条，未选岩石地形和围栏。但模型仍普遍自评 1，且 uncertain_aspects=[]；仅能确认危险猜测未被升级，不能宣称已可靠结构化所有来源不确定性。固定描述控制中还出现 has body→dirt ground、has body→standing on 等关系表达不佳，说明单有权重并不能保证三元组语义正确，原始记录保留以供后续核验。

该轮 Stage 4 在 LM Format Enforcer 下提前终止 JSON，真实 ranking_status=evaluation_failed；CLI 返回 0 不代表评审成功。为冒烟入口增加 final.json 的严格状态断言。定位到混合整数／null enum 不被该后端支持，将评分改为等价 anyOf，合法值仍仅为 0～4 或 null；证据前缀 regex 改写成能整串匹配任意非空后文的等价表达，避免只允许前缀加单个字符。严格语义解析和来源验收不变。实际服务环境对 r4 真实检查目标的文本／图片完整 grammar（含事实评分 null）均通过。下一轮只复评 r4 冻结候选，避免重新生成掩盖失败。

LMFE 的 grammar 测试通过后，r5 真实长评审仍进入很长的生成／修复，未计为成功，停止的仍仅为本诊断作业。最终传输兼容方案恢复服务默认 auto 后端：Stage 1 的仅供解码 schema 移除 auto 后端不支持的 uniqueItems，完整业务 schema 和 Python 来源去重校验保持；重复引用仍被拒绝，不能靠重复引用加权。Stage 4 保留等价 nullable 分支与完整证据后文表达。已在实际服务环境编译 Stage 1 及 r4 真实 text/image 检查的 xgrammar，均通过；针对性回归 216 passed、3 skipped。r6 将先在默认后端验证固定旧描述，再独立复评 r4 冻结候选；控制实验失败也不阻断无依赖的 Stage 4 复评。
