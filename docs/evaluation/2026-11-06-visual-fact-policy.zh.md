# 视觉事实的来源权重、范围与不确定性

已实现第一版来源使用策略：图片抽取只看该来源，相关性与来源权重分开，Stage 2 不必选择全部画面信息，Stage 3 不再从完整机器图片描述带回未选猜测。原始描述、原图引用、被隔离的观察和所有失败尝试均保留。

这项改动不能把模型自评变成真实概率。实际 Qwen 仍可能过度自信、遗漏不确定性记录或抽取语义不佳的关系；不能因此宣称已得到可靠的自动事实闭环。

## 最终规则

新图片描述通过 `with_image_description` 启用 `visual_observation_policy_v1`，每个图片抽取项增加必填 `visual_support`：

| 字段 | 含义 |
| --- | --- |
| basis | direct：整个断言可直接观察；inferred：推断 |
| scope | target：目标属性／动作；context：必要场景；background：无关背景 |
| confidence | 有限的 0～1 模型自评分值，未经校准 |
| uncertainty | 保留猜测、替代项、冲突和限定条件；inferred 必须有非空说明 |

同一请求的非图片项使用 `visual_support: null`。元数据缺失或不合法会被原有有界合同修复拒绝，不补成默认高置信度。提示词禁止把“可能 A 或 B”拆成确定 A 和 B，也要求将有意义的未决源猜测写入 uncertain_aspects，即使不作为抽取三元组。

抽取时不展示图谱事实、图谱关系计数或图谱证据内容，profile 同样不能引用未展示图谱。完整图谱由程序保存，抽取后合并，既不丢弃，也不让模型照抄图谱后宣称是图片直接支持。目标名称仍是公开输入，无法据此保证模型不会使用记忆推断。

每条来源判断及引用保存在 visual_observations。图片推断、非空不确定性、background、自评低于 0.8 的观察，其视觉贡献权重为 0，不成为 assertive generation 的可选事实；仍留在 Stage 1 目录与审计记录中。程序也为被隔离的观察补入不确定性记录，不仅依靠模型的顶层列表。

可用 direct/target 的 `source_weight=confidence`；direct/context 为 `0.5×confidence`。已有图谱／文本／音视频断言沿用原有来源策略，基准权重 1，不表示绝对真实。保留相关性原分 relevance_score，检索使用 `selection_score=relevance_score×source_weight`，Stage 2 再选精简、互补的一组。

**0.8 和 0.5 是首版保守策略参数，未经样本集校准。** 同图描述不是独立证据，多图取可用最大权重，不按出现次数累加。弱视觉观察若与图谱重复，保留逐源判断，但不把弱视觉引用并入图谱的正向支持引用。

过滤同时约束相关性检索、Stage 2 可选 ID 和生成任务编译，不运行 reranker 也不能绕过。权重和逐源判断继续传给生成器及语义 Judge。Stage 3 使用所选视觉事实，不再收到完整机器图片描述的 content，避免未选猜测从 source_evidence 绕回生成。完整描述仍在 Stage 1，未改成伪造的“原文摘录”。未启用新策略的历史产物保持原解释，不回写旧结果。

## 实际单实体证据

数据为 WN18RR train，实体 02436645，Qwen2-VL-7B-Instruct、Qwen3-Reranker-4B 与真实 SDXL，单 A100 80GB。固定描述控制使用旧失败描述和原始同一 JPEG，SHA256 为 `0dc11e1b26f68c44209be5b764a02441ad1fc89a5030a98a501d6fb4899cc456`，不通过换描述隐藏原错误。

r4 的控制中，mouse deer 仅作为图谱事实保留，visual_observations=[]；rodent/marsupial 未变成确定抽取项。完整文本流程中，源描述仍包含 rodent/marsupial/bandicoot，但 Stage 2 从 8 条选了 6 条，未选 rocky terrain 与 wire mesh fence，两者的来源权重均为 0.5。最终文本不再出现这些猜测分类。

同时必须保留反例：模型对直接观察普遍自评 1，并返回 uncertain_aspects=[]；还有 has body→dirt ground、has body→standing on 等语义表达不佳。原始猜测保留在来源记录，但模型没有可靠地将其全部结构化。新文本还增加了缺少明确支撑的 natural setting。因此只能确认本例的来源隔离和生成范围有所改善，不能宣称全部断言获核验、权重已校准或 Reflect 已成功纠正内容。

r4 同时实际生成 1024×1024 SDXL PNG，20 steps、guidance 7.5、候选 seed=20260709；文件在 `outputs/wn18rr-visual-policy-20261106/attempt-r4/runs/image-qwen7b-sdxl/entities/02436645/generated_images/image_001.png`。已查看真实像素，是湿地中的鹿状动物；不能仅凭这一张图宣称精确物种身份正确。

## 解码与验证

新视觉合同从首条本地 Qwen 抽取请求使用完整字段 grammar；图片描述请求仍为自然语言。vLLM 0.11 的旧 guided_decoding_backend 参数不能可靠选择请求后端，最终服务恢复默认 auto。仅供解码的 Stage 1 schema 移除 auto 不支持的 uniqueItems；完整业务 schema 与 Python 对重复／未知引用的拒绝保持，不允许重复引用加权。

Stage 4 的混合整数/null enum 改为等价 anyOf，仍仅允许 0～4 或 null；证据前缀 regex 改为能整串匹配完整后文的等价写法，避免 LMFE 限制到单个字符。严格语义解析、来源绑定和修复验收不变。实际服务环境已验证 Stage 1 与真实 text/image 检查的 xgrammar 编译，也验证了 LMFE 完整评审 grammar；后者不等于实际模型稳定完成长评审。

针对性回归：216 passed、3 skipped。跳过的是推理环境未安装的 LM Format Enforcer 检查，已在实际服务环境另做 grammar 验证。扩展回归：208 passed、2 skipped、64 failed；修改前 2116d0b 为 202 passed、64 failed，失败项集合完全相同，无新增失败。63 项旧 context 测试主要仍使用主线不支持的 missing_mode=both，另 1 项是旧 Qwen Judge 摘要预期不匹配。未修改旧测试预期掩盖这些失败。JUnit／日志在 `outputs/wn18rr-visual-policy-20261106/verification/`。

可复跑入口：`scripts/run_wn18rr_visual_policy_cluster.sh`；冻结候选复评入口：`scripts/run_wn18rr_stage4_cluster.sh`。后者可指定 TAOR_STAGE4_SOURCE_ROOT、TAOR_STAGE4_ARTIFACT_ROOT、TAOR_STAGE4_ATTEMPT，以及可选的 TAOR_STAGE4_VISUAL_CONTROL=1。每次新建 attempt；Stage 4 复评校验源文件哈希，原候选不回写。完整冒烟入口检查 final.json 的 ranking_status，不能把 CLI 返回 0 当作评审成功。

## 尝试记录

| 尝试 | 真实情况 |
| --- | --- |
| r1 | 新字段遗漏后进入合同修复，默认解码后端拒绝 uniqueItems；失败保留 |
| r2 | 固定描述中复制图谱为视觉高置信支持，文本又从原描述带回猜测；不能算来源保留成功，促成来源隔离和生成范围约束 |
| r3 | 固定描述控制达到脚本默认 60 秒请求上限；之后与完整 CLI 对齐到 120 秒，Stage 1 总预算仍 180 秒，原生 token 预算未提高 |
| r4 | 文本与真实 SDXL 均保存，Stage 4 因混合 enum 在 LMFE 下提前结束 JSON，ranking_status=evaluation_failed；未计为评审成功 |
| r5 | LMFE 实际长评审进入长回复／修复；仅停止本诊断作业，候选不变，未计成功 |
| r6 | 默认后端固定描述控制＋r4 冻结候选复评，结果见后续审计记录 |

所有尝试均申请一张卡，未启动全量任务，停止的仅是本轮自己的诊断作业。先前工程冒烟及 Stage 4 真实运行的 49 个原始文件已核对哈希不变。

## 后续改为 API 优先

用户随后要求先用 API 联调，稳定后再申请卡。本轮自己的 GPU 诊断作业均已结束，后续流程见 `docs/api-preflight.zh.md`。r6 固定原图／旧描述控制完成，但文本 Stage 4 因缺少 fact check reason、只返回部分检查并重复输出多个 JSON 而失败；未计为闭环成功。之后仅做 CPU grammar 回放及 API 传输修复：本地解码使用语法视图、Python 保留语义验收，远程 API 继续接收完整 schema；空响应与截断也在解析前分类。相关回归更新为 232 passed、3 skipped。
