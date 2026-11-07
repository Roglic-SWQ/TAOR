# WN18RR 原生接入与 Qwen2-VL-7B／SDXL 冒烟

## 2026-11-07：快速修复 Reflection 的意见传递

当前策略为 `targeted_reflection_v4`：把原 Judge 的分项原因、证据、具体建议、未确认新增信息与诊断传给现有修复模型；语义修复直接指出实际问题。建议只作为参考，不成为必须补齐的清单，也不单独触发重写。旧兼容路径清除低关系覆盖导致的补写触发，并将“仅限图谱事实”改为已有来源支持。没有增加严格门控或自动上游循环。

新自动批次无导出时明确记录未补全（`no_usable_candidate`）及原因，不再挂成无人接手的 needs_review；评审运行失败单独记 failed，历史记录保持可读。GPT API 图片提示词预算改为 2048，本次实际 belief 回复用 477 token，超过此前 384 的上限；本地 Qwen 配置未改。

491 项相关回归通过。两例真实失败记录的 API 提示词修复均收到完整意见并完成复查。waterway 改为可见通道及连续水流；belief 仍依赖抽象符号，不能把提示词 supported 当成出图验证成功。未新增图片或 GPU 作业。源码、回归和复核记录见 `outputs/reflection-feedback-20261107-r1/` 及 [Stage4 文档](stage4-judge-rubrics.zh.md)。

## 2026-11-07：上游重提小对照，暂不接入自动回退

用户要求先试验并判断是否值得增加设计。本轮仅做 CPU/API 诊断，复用现有 compact 提取、语义筛选、原图像素核验、事实选择、文本生成和 Stage4。没有修改生产流水线，没有申请 GPU，没有新增 SDXL 图片，也没有增加门控阈值或 must/should 规则。

在调用前固定 student、bludgeon、mullah 三个案例，分别作为漏提疑似例、已有可用结果的物体正例、肖像不足以表达角色的负例；共 7 张已准入原图。保留原准入、盲描述、图片字节和 train 图谱，隐藏目标文本。比较同一个 gpt-6-luna、温度 0、相同来源与下游处理的两组：

- repeat：使用当前提取说明重新看图。
- targeted：在相同说明后追加旧提议、旧筛选结果和一次问题指向，询问是否主体定位错误、漏掉可见行为或结构；允许空结果，不要求发现特定事实。

旧 Qwen 输出只用于定位问题，不作为同模型对照。两组执行顺序事先交替；没有根据结果继续改提示词直到成功。

| 案例 | 普通重提 | 带失败反馈重提 | 本轮判断 |
| --- | --- | --- | --- |
| student | 提取 0 项，仍不足 | 多提取一项“右侧人物手持纸张和话筒”；语义筛选认为该动作不足以表达 student，删除后仍不足 | 没有恢复文本补全；原图有课堂线索，不能由弃答推断完全无信息 |
| mullah | 提取 0 项，弃答 | 提取 0 项，弃答 | 没有把头饰、胡须升级成角色身份，也没有新增可用信息 |
| bludgeon | 提议 4 项、筛选后 2 项，两项像素核验均支持；选中形状和木质外观，生成文本并通过评审 | 提议 4 项、筛选后 3 项；把不同图片的木质／纹理合在一起的提议、被手遮挡的柄端均未获确认，只保留形状；生成文本并通过评审 | 两组都可用，定向反馈没有显示额外收益；未确认的提议不等于已证明错误 |

普通重提的木棒文本：

> The pictured subject associated with “bludgeon” appears to be an elongated, roughly cylindrical wooden object with visible grain and a rough surface. One end is thicker, while the body tapers slightly toward the other.

定向重提的木棒文本：

> The pictured subject associated with “bludgeon” appears to have an elongated cylindrical body with one thicker, enlarged end; its exact identity and function are not verified.

后一组形状提议原引两图，像素复核明确只确认第二张原图；原提议引用与实际支持引用分别保留，不能宣称多图独立互证。两组所选事实及核验 panel 不同，内部评分不可直接用于量化写作质量收益。生成和评审仍同用 gpt-6-luna，不是独立真值认证。

六组均执行完成，共记录 17 次上游网络尝试、2 次文本生成，两个文本候选均通过 Stage4，没有触发内容 Reflection。对 student，定向反馈额外产生一次语义筛选调用，仍未恢复补全。三个诊断案例中没有出现“普通重提失败、定向重提成功”的新增完成案例；样本很小，每组只运行一次，不能据此断言回退永远无用。

结论：暂不把自动返回上游接入主流程。当前保留既有诊断入口；后续若出现明确可定位的上游漏提，再针对原始来源验证。生产问题仍包括主体定位及角色相关性判断，不能靠泛化的重试指令保证解决。

结果根目录：`outputs/upstream-reextract-probe-20261107-r1/`。`manifest.json` 与 `*-frozen.json` 保存固定输入；`attempt-r2/` 保存六组提议、筛选、像素核验、选择及完整文本结果；`verified-summary.json` 核对两组提取提示词仅追加反馈、schema／原图元数据相同、原始文件／train 指纹不变、选择仅使用图片来源、执行源码未改。`probe.py` 和 `verify.py` 保存可复核实验代码。

最初实验封装遗漏内部调用审计字段，六组均在发出网络请求前失败，记录单独保留在根目录的 `arms/` 和 `results.json`；修正封装后运行 `attempt-r2`，没有将封装错误计入模型质量结论。未重跑全仓软件测试，本轮验证的是实际 API 对照与数据／来源一致性。

## 2026-11-07：Judge 格式性失败修复

当前 Judge 调用协议为 `candidate_observation_contract_v3`：空 evidence、缺少 text:／pixel: 前缀、事实分与逐项观察不一致不再触发整份评分失败。保留原始分数与 null，记录 `rubric_diagnostics`；不补满分、不伪造引用。观察不全只列为未评估；依据不足不执行自动删改，固定修复问题未解决不能替换原候选。Judge 无效响应标为 `judge_response_invalid`，与候选内容缺陷区分。

历史 22 条无效响应回放有 18 条恢复；剩余 4 条是真正的 JSON／检查项结构损坏。新 API 在两张已有 SDXL 修复图上首轮返回有效评分：水道修复可采用；belief 分为 E=1／F=null，原语义问题仍 unresolved，合并后不采用。它此次普通 gate 为 pass，说明模型评分仍波动，不能据此宣称质量完全解决。

468 项针对性回归通过，另 64 项兼容测试通过；4 项修改前已失败的旧协议测试保留基线记录。未重跑 GPU 或扩大批量。回归、历史回放、原始响应和修复资格合并结果位于 `outputs/judge-contract-relaxation-20261107-r1/`，详细规则及已知基线测试失败见 [Stage4 文档](stage4-judge-rubrics.zh.md)。

## 2026-11-07：清除覆盖等级，保留来源区别（当前）

主体重构提交 `bb05866`；实测后补充了下述校验与恢复边界修复。新运行删除 must／should、逐检查 requirement、文本必需遗漏补写，以及批处理“必须有 must 才导出”的门槛；没有把这些规则换名继续执行。候选上下文只有同一组来源明确的事实检查。旧字段只在只读导入模块 `mmkgc/io/legacy_evidence.py` 中识别、核对和转换，不进入新执行协议。

| 来源 | 保留的信息 | 核验边界 |
| --- | --- | --- |
| 已有文本 | 原文证据 ID、提取断言、主客体／方向及限定 | 核对与原文是否一致；模型提取结果不自动成为独立事实来源 |
| 原始图片 | 原图引用、直接／推断依据、样本范围、身份／特征不确定性 | 核对可见属性；看图生成的 caption 仍是图片派生信息，不作为新增独立文本证据 |
| 图谱 | train 图谱三元组、关系含义、方向、原始引用 | 作为结构化断言；不能用图谱关系证明像素中存在相应外观或动作 |

来源通过 `source_modalities`、`evidence_refs` 和证据目录区分；图片的 `visual_observations`、既有 `source_weight` 在选中事实中保留，不把权重当作已校准的正确概率。当前视觉描述配置只将图片相关事实用于补文本，graph 仍留在 Stage1 审计；缺图规划可使用已有文本和 train 图谱。来源未知时不由后续阶段猜测或升级身份确定性。

观察协议为 `fact_observation_v2`，上下文为 `selected_evidence_context_v2`／`ranking_context_v2`，反思策略为 `targeted_reflection_v3`。Stage4 统一决定候选状态：已确认且有具体证据的事实冲突拒绝并进入修复，合理遗漏与不可验证内容不强制补齐。语义为 0 且有来源和候选证据时产生独立 `reframe`，先审原提示词，再改写、采样或因不确定而跳过。固定 issue_id 的复核负责判断原问题是否解决；修复失败保留原候选，不能把未解决的冲突当成成功候选导出。

验证产物位于 `outputs/evidence-context-refactor-20261107-r1/`：

- **436 项针对性回归通过**，见 `regression-final-r5.xml`。覆盖来源／不确定性保留、无覆盖等级、事实与语义修复、候选隔离、保存恢复和 SDXL 路径；不是全仓测试通过声明。
- 旧 20 case、25 候选全部只读转换成功，原始文件指纹不变，见 `historical-migration.json`。
- 如果仅删除旧导出门槛、完全复用旧评分，机械可导出数会从 7 变成 15；这是规则变化，**不是质量提升**，见 `historical-export-comparison.json`。
- 真实 API 对 8 张已保存 SDXL 图片重新评分：水道与蚝鹬四张通过，belief／freedom 四张拒绝且形成反馈。初次提示词审核误选重新采样；改进后，两例均明确进入 `edit_prompt`，见 `api-preflight/`、`api-routing-r2/`。这些 CPU/API 对照没有生成新图片，也未运行本地神经评分器。
- 已冻结 161 个执行源码／配置／提示词文件，单卡作业 `taor-evidence4-1107-r1` 已成功结束，GPU 已释放。实际运行 waterway、belief 两个真实缺图实体，以及 bludgeon、student 两个模拟缺文本实体，共 6 张不同的真实 SDXL 图片、1 段原生 9B 文本。每个缺图实体初始两个种子，最多追加一次修复；未扩大到 100 或全量。


本轮四例的最终复核如下。作业成功表示执行完成，不代表四例质量都通过。

| 实体 | 实际过程 | 最终处理 |
| --- | --- | --- |
| waterway | 原两图只显示水面，无通道；Stage4 语义为 0，审核原提示词后选择 edit_prompt；9B 改为可见混凝土水渠，SDXL 第三图呈现通道；固定语义问题复核 resolved | 暂保留修复图，精确地点／真实实例身份未认证 |
| belief | 原两图为几何体；实际改写后第三图为玻璃状气泡及乱码，依旧无法表达被视为真实的认知内容 | 同图复评语义为 0、事实分 null、固定语义问题 unresolved；拒绝采用 |
| bludgeon | 9B 从原图所选木质、粗糙表面特征写作，保留低分辨率限制；原评审因冻结元数据复制错误失败 | 修复后对同一段文本、同一组来源做真实 API 复评，通过；只作为所示木棒的外观描述 |
| student | 图像提取没有留下足够相关的所选特征，未生成文本 | 保留弃答。课堂画面仍有可进一步提取的学习场景线索，不能宣称原图完全无信息 |

实测后修复了三处边界：冻结后的图片观察元数据先还原为普通 JSON 数据再校验；修复问题明确区分 `issue_kind=semantic` 与事实问题，未解决的语义问题不冒充事实冲突；Stage3 保存固定修复问题，旧产物缺失时从 Stage4 按候选父子关系、检查范围与来源引用核对恢复。

原 GPU 批次记录仍为 exported、needs_review、failed、no_usable_candidate 各一例，两个候选评审错误均保留，未改写历史。后续只对已有文本／图片重新调用 API：`reassessment-r2/` 验证图片来源元数据可正常评分；`reassessment-r3/` 在保留原固定语义问题的情况下复评 belief，得到正常拒绝、unresolved，评审错误数为 0。没有再次生成文本或图片，**也没有重跑整个修复后 GPU 批次**。

最终暂保留 1 张修复水道图、1 段木棒文本，见 `reviewed-r2/exports/`；完整原始状态、来源、后续复评和执行版本差异见 `reviewed-r2/summary.json`。复核为助手判断，未经独立人工标注；四例不代表准确率或全量质量验收。

### 当前四例复查：Reflection 的实际分工与剩余问题

实际日志中的模型名为准：`relation_alternation_generation_gpt_prompt` 是历史阶段名，本轮该阶段实际调用 Qwen/Qwen3.5-9B；Judge 类名中的 GPT4OMini 也不表示用了 GPT-4o-mini，本轮配置及记录为 gpt-6-luna。

| 环节 | 实际执行者 |
| --- | --- |
| 初次／修复后内容评审、实体语义与事实观察 | gpt-6-luna |
| 从观察构造修复操作、管理一轮预算和候选保留 | Python 控制器 |
| 检查原提示词是否已经正确表达相关内容 | gpt-6-luna |
| 根据该检查选择改写、采样或跳过 | Python 控制器 |
| 需要改写时，修改文本或图片提示词 | Qwen3.5-9B |
| 原提示词采样或改写后重新出图 | SDXL |
| 固定问题复核与候选替换资格 | gpt-6-luna 复评，Python 控制器核验；GPU 图片分支另有本地 CLIP／BLIP 评分 |

waterway：来源文本仅定义水流通道，选中该定义；初始 Qwen 提示词主动排除岸边与地景，只画水面。两个种子均被 GPT 以没有通道线索判语义 0。GPT 审核提示词后选择改写，Qwen 加入可见混凝土水渠与边界，SDXL 第三图通过固定问题复核。剩余问题首先在上游提示词：关键可见结构被当成无关环境排除，依靠出图后纠正，初始两个种子重复了同一方案错误。

belief：来源为文本定义和图谱 mental object 上位关系，均被正确提取和选择；Qwen 先用几何体象征信念，GPT 拒绝后改成含书面命题的气泡，SDXL 实际生成玻璃状物体与乱码。后续同图完整问题复评为语义 0、事实分 null、semantic 问题 unresolved。流程能拒绝失败结果，但还缺少上游对可视表达方案的评估；泛化的 mental object 对画面帮助有限，写一个概念词不等于表达了该概念。

bludgeon：原四图中两张准入、一张 unresolved、一张 unrelated。第四张被模型以细长、像枪管且没有粗重端部为由排除；这个粗重端部条件不是来源提供的规则，存在误筛风险，需要独立复核。9B 提议的整体形状混合了细长杆身、锥形把手及圆形端部，像素核验只能确认其中部分，整条属性被判 uncertain，最终只选木质和粗糙纹理。材质事实原提议引用第二、三张，而核验明确只引用第三张；目前可追溯二者，但还需把原提议来源与实际支持来源分清，避免多图共同支持的误读。最终文本可限于木棒样本使用，未执行内容 Reflection；最初的运行失败是冻结元数据校验错误，修复后是同文本 API 复评通过，未重跑完整 GPU 融合评审。

student：三图两张准入，一张 unresolved；原图包含课堂和展示活动。9B 实际提取第一张中站在前方人物的站姿、头发、衣裤、面对观众等五项，GPT 抽象全部删除，Stage2 返回证据不足。没有文本候选，也没有候选 Reflection。当前反思仅作用于 Stage3，没有定向回到 Stage1 重新定位目标和提取可见行为的路径，因此无法恢复这种上游漏提；合理弃答避免了错误描述，但不代表原图完全不可用。

## 2026-11-07：9B／API 混合方案的 20 实体闭环（此前）

**这轮完成 20 实体、20 张真实 SDXL 图片；仍未通过全量自动采用的质量验收。** 实验目录为 `outputs/wn18rr-qwen35-hybrid-20-20261107-r1/`，完整批次为 `batch-r2`。测试集共有 5323 个实体，其中 4493 个真实缺图、0 个真实缺文本。本轮冻结 10 个真实缺图实体和 10 个遮蔽已有文本的实体；后者是模拟缺失测试。样本用于诊断，不是随机准确率评测。图谱只读取 train，test 关系不进入生成，遮蔽文本不提供给模型。

本轮修复了两个相互叠加的问题：

- `fish` 的原始词义是“捕捉或尝试捕捉鱼／贝类”，9B 曾将其提取成食物。现在原始词典文本由独立 API 重新提取，保留定义原句及“尝试”等限定；新配置显式启用，失败不回退到错误提取。初始 9B 提取和审核调用均保留。
- Stage4 原先按候选模态过滤已有来源，导致缺图时连合法的已有文本也被清空，评审只能循环信任错误事实。现在只遮蔽目标模态，保留相反模态作为来源。对同两张历史鱼肉图片重放，修复后均拒绝；两张历史水道正例仍通过。新生成的 fish 两个种子均表现为钓鱼活动，不宣称已经钓到鱼。

上游增加可选的视觉语义抽象：9B 提议特征，API 只保留／简化同一来源的属性，再用原图独立核验。不能提升背景属性为实体属性，不能合并来源、改写带不确定性的值或消除来源限定；允许全部丢弃。提取／筛选／写作不要求每张图都有特征，不提供图谱边来补写视觉描述。surface lift 只留下缆线与支撑杆，不再把橙色护垫、人物衣着写进去；五个角色实体没有足够视觉语义时直接弃答。没有新增置信度阈值；相关性仍是模型判断，不是真值概率。

实际分工是 **Qwen3.5-9B 提议视觉属性、初次提取文本、写生图提示词和文本；gpt-6-luna 复核原始文本、抽象／筛选视觉属性、核验像素并执行 Stage4；SDXL 实际出图**，不是纯 9B 离线方案。配置为 `configs/qwen35-9b-hybrid.yaml`、`configs/wn18rr-qwen35-hybrid-pilot.yaml`。使用一张 A100 80GB，成功作业 `taor-q35-hybrid20-1107-r2` 已结束。每个缺图实体同提示词两个种子，1024×1024、20 步、guidance 7.5；共 20 次生成、20 个不同像素哈希，无额外图像反思生成。

| 结果层次 | 缺图 10 实体 | 模拟缺文本 10 实体 |
| --- | --- | --- |
| 实际生成 | 20 张图片 | 5 段文本，5 个证据不足弃答 |
| 原始自动导出 | 2 个实体 | 5 个实体 |
| 自动规则与本次助手逐例复核均接受 | 1 个实体：fish | 4 个实体：ganoid fish、bludgeon、screen、surface lift |

没有技术运行失败。鸟类文本的精细条纹表述仍不够确定，保持原文待审。鱼鳍文本信息稀疏；上述四段仅作为带来源限制的样本外观描述，不是实体身份认证或通用定义。haematopus、syringa、waterway 的图片概念上相容，但未通过现有严格发布条件，未为提高覆盖率放宽规则。色谱图虽然自动导出，逐图复核未确认分离过程，因此不进入最终接受集。这些复核是助手定性判断，未经独立人工标注，不能作为准确率。

批次完成后，对同 20 张已保存图片做 API 审核对照，无新增 GPU 或图片。更明确的独立视觉语义规则拒绝 belief 的普通几何图案和 freedom 的无关人像，保留 fish、haematopus、waterway 正例，随后合入图片评审提示词。**这不是再跑一轮生成**；规则仍会放行部分普通办公人物、油状纹理、比例失真的迁移旅客与含糊色谱图，助手待审决定继续有效。因此尚不允许 100 或全量扩展。

回归：`regression-final.xml` 记录 **221 项通过**，包括来源模态遮蔽、空响应／截断分类、来源及不确定性保留、语义拒绝与事实支持独立、已有候选隔离和 SDXL 路径。158 个执行源码已归档核对；20 个终态的文件指纹、数据与原图绑定、20 张图片均验证。后加入的审核提示词与执行源码差异另记，未改写历史调用／候选。早期错误鱼肉产物位于 `batch-r1`，该探索作业由我们停止，未覆盖。

优先查看：

- `batch-r2/verification/measured-results.json`：分模态结果、逐实体限制、来源验证及后加入的规则。
- `batch-r2/reviewed/accepted_completions.jsonl` 与 `reviewed/exports/`：最终暂保留的 4 段文本和 1 张图片；其余原始产物仍在 `attempts/`。
- `source-sense-preflight-r3/`：原词义、相同错误图片及正例重放。
- `image-semantics-control/`：同像素 API 审核、完整提示词／响应和判断。
- `batch-r2/verification/executed-sources.tar.gz`、`binding.json`、`manifest.json`：执行版本与输入；`run_cluster_r2.sh` 为实际入口。

### 逐 case 复查：闭环为何没有触发

以下复查只读取 `batch-r2` 原始调用、来源、两个种子的图片和保存的候选，没有新增模型请求或 GPU 作业。结构化全过程见 `batch-r2/verification/case-traces.json`。不能把“最终暂保留 5/20”直接当作模型准确率：其中包含 5 个上游弃答、发布规则限制、证据强度限制和助手定性复核。

**本轮 25 个生成候选（20 图、5 文）原始门控全部 pass、repair_needed 全部 false、没有反思重生成。** 图像语义只要不是 0 分就可能通过；freedom 的语义得分为 1，某候选视觉语义覆盖为 0，仍为 pass。`uncertain`／`unverifiable` 也不直接产生修复操作。离线把后续严格审核中 belief、freedom 的四个真实 reject 结果接入反馈构建函数，仍得到 0 个修复操作、repair_needed=false，记录在 `semantic-repair-routing-analysis.json`。因此仅改评审提示词还不足以接通语义失败后的重试路线；这项路线问题在本次分析中确认，尚未修改实现。

30 张已有图片中，准入为 compatible 的 21 张，unresolved 的 9 张。compatible 只是外观相容，并非身份确认。五个弃答中，playwright、liberator 在准入就没有可用图；student、mullah、doubter 则经过 9B 提取后，被语义抽象全部丢弃。尤其 student 存在课堂／展示活动线索，不能笼统说原图都是无信息肖像。

| Case | 来源与提取／选择 | 生成及原始 Stage4 | 复查定位与结果 |
| --- | --- | --- | --- |
| fish · 01140794 | 原文指捕捉或尝试捕捉鱼；本轮 9B 和 API 都保留正确词义，选择定义及 take hold of | 两种子均为手持钓竿活动；均 pass，第一张身份与活动 supported | 本轮修复有效，暂保留第一张；不表示成功捕获。历史鱼肉错误属于此前 batch-r1，不是本轮同一输出 |
| haematopus · 02036982 | 原文 oystercatchers；选择定义、鸟属及分类关系 | 两图为黑白鸟、橙红长喙、岸边环境；均 pass，但检查全部 should_include | 概念相容，发布规则要求至少一个 must_include，故未导出；验收结构问题，不能算作纯生图失败 |
| syringa · 12310153 | 定义包含丁香属、旧大陆、芳香、圆锥花序；选定义和属成员关系 | 两图可见紫色花序；均 pass，identity uncertain，整条定义被标 supported | 外观相容但属身份未证实；把可见花序与不可见香气／产地放同一事实，也使 supported 含义过宽。待审 |
| waterway · 04559275 | 定义为水流通过的通道；选择定义及 canal 关系 | 两图为水渠／水道；均 pass，定义 supported | 概念相容；与 haematopus 一样没有 must_include，被发布规则挡住。待审 |
| rock oil · 14980579 | 正确保留“主要由烃组成的深色油”；选择定义和油类别 | 提示词转为极近距离油面，图像为反光纹理；均 pass，事实 uncertain | 纹理不能区分石油或验证化学组成；可视化方案信息不足，评审 pass 不等于已核验。待审 |
| migrator · 10314952 | 词义和 traveler 关系正确 | 提示词为通行旅客／交通枢纽，两个种子出现室内大船或巨大船状负载；均 pass，其中一个四项评分均较高 | 生图形态失真，评审遗漏明显异常；具体旅客活动的提示词也不够明确。待审，无自动重试 |
| official · 10372373 | 正确保留“获授职务的工作者”，不是 office room 的意思 | 提示词诉诸姿态体现权威，成图为办公人物；均 pass，角色事实 uncertain／unverifiable | 在提示词桥接阶段就把不可见职务变成泛化外观；后续审核一过一拒，仍不稳定。待审 |
| belief · 05941423 | 正确保留“被视为真实的认知内容”及 mental object | 提示词直接要求几何体象征信念；成图为圆形、球体或几何浮雕；原始两图 pass，语义仅 1／2 分 | 错误始于抽象概念到画面的桥接，SDXL 执行了无辨识力方案；后续同图审核均 reject，却无可执行修复项。待审 |
| freedom · 13991823 | 选择不受外界约束地行动、说话、思考三个定义片段 | 提示词以独立人物、没有障碍来表示自由；成图为线框人像；原始两图语义均 1 分仍 pass | 方案过弱且 SDXL 未实现明确动作；后续同图均 reject，反馈仍无修复任务。待审 |
| chromatography · 13448778 | 没有 Stage0 图谱事实；原始文本足以提取分离过程，选择定义 | 提示词要求柱内不同色带；成图为彩色管／装饰丝带；第一张原始四项均 4 分并导出，第二张语义 1 分也 pass | 提示词已有合理过程方向，但渲染及评审均不可靠；后续规则仍放行，助手未确认分离过程，撤出暂保留集 |
| piciform bird · 01838038 | 一张准入图；9B 提议 5 项，抽象留喙、头部和翅膀纹样；像素核验三项均 supported | 9B 忠实写入三项；Stage4 四项均 4 分并导出 | 精细白纹位置和“双翅条”仍不清楚；问题来自上游提取及核验过度确定，不能只归罪于写作。待审 |
| ganoid fish · 02638596 | 四张准入图；9B 提议 6 项，抽象留 5 项；像素核验仅背鳍 supported，体形／骨板／吻部／触须 uncertain | 只选择背鳍，写一段短文；Stage4 pass | 过滤挡住了多数过度断言，但跨图宽泛引用导致信息损失；暂保留为单样本背鳍观察，不认证分类 |
| bludgeon · 02855793 | 四张准入图；提取主要来自木棒图；抽象留木纹、粗糙表面、圆孔，像素核验均 supported；最终选前两项 | 文本描述木质杆身和粗糙纹理，保留身份／用途未核验；Stage4 pass | 这例筛选与写作有效，未枚举制服、人物或背景；暂保留为该木棒实例描述 |
| screen · 04151581 | 四张准入图；9B 取第四张的框架、纱网、邻接玻璃和窗体配置；抽象去玻璃和颜色；像素核验仅纱网 supported | 只写细密半透明网面；Stage4 pass | 简化有效，避免玻璃被当成纱网属性；代价是描述稀疏，不能宣称覆盖全部 screen。暂保留 |
| surface lift · 04362821 | 两张准入图；9B 提议橙色护垫支柱、缆线、悬吊座椅；抽象去护垫，像素核验拒绝确认座椅 | 仅选择缆线和支柱；文本不再写座椅／衣着；Stage4 pass | 过滤成功，未把不清楚的机械结构升级为事实；暂保留为实例结构描述 |
| student · 10557854 | 三图两 compatible、一 unresolved；9B 实际接收两图，却只提站在前方者的姿势、头发、衣裤等 5 项；抽象全丢 | 选择 insufficient_evidence；无文本候选，无候选评审 | 有课堂与展示活动，9B 主体定位／特征遗漏是可改善部分；当前弃答避免错写，但不能断言证据完全无用 |
| mullah · 10338231 | 三图均 compatible，但身份未确认；9B 提出头巾、胡须、衣袍 5 项，抽象全丢 | insufficient_evidence；无文本候选，无候选评审 | 配图是否确属角色仍缺证据；服饰刻板线索不应变成角色定义。保留弃答，compatible 不代表已排除错误配图 |
| playwright · 10030277 | 三图均 unresolved：讨论人物、舞台旁人物和普通肖像都不能确认剧作身份 | 没有进入 9B 特征提取；选择为空，无生成和候选评审 | 来源无法提供职业证据，前置停止合理；舞台背景也不等于作者身份 |
| doubter · 09779124 | 两图 unresolved，一图因疑似怀疑表情 compatible；9B 却只提出帽子、球衣、字样、纹身 4 项，抽象全丢 | insufficient_evidence；无生成和候选评审 | 衣着过滤正确；表情本身也可能是专注等其他状态。既有提取遗漏，也有无法核验心理状态的证据限制 |
| liberator · 10257221 | 三图均 unresolved：肖像、聚餐、车辆合照均未显示明确解放角色或行动 | 没有进入 9B 特征提取；选择为空，无生成和候选评审 | 来源证据不足，合理弃答；不能归为生成模型写作失败 |

下一轮应优先处理三个流程点：发布条件与概念相容／身份认证分开；语义失败明确路由为重写视觉方案、定向重提或弃答；视觉提议按具体图片和目标主体保存，避免跨图属性过度合并。不能只提高一个分数阈值，也不能把所有 uncertain 都强制重试。以上是本次逐例分析提出的后续工作，未声称已经修复或重新跑通。

## 2026-11-07：Qwen3.5-9B 单卡实测（此前六例）

**9B 的提取、选择、写作与 Stage 4 已完成六例，但未通过扩大批量的质量验收。** 使用相同六个诊断实体及其 17 张原图；沿用历史准入和盲描述，其中 13 张准入原图实际进入 9B。不是重新执行全部预处理，也不是代表性随机样本。新提取、选择和写作不提供图谱断言或遮蔽目标文本。

完整 BF16 权重位于 `/medisco/linjiashi/models/Qwen/Qwen3.5-9B`，从官方 ModelScope 仓库下载的 17 个文件均核对 SHA256，总计 19,329,394,438 字节。新环境 `/medisco/linjiashi/venvs/taor-qwen35` 使用 Torch 2.8.0、Transformers 5.3.0；原有 7B 环境保持原状。成功作业 `taor-qwen35-9b-1107-r4` 使用一张 A100 80GB，现已结束。试验服务为 Transformers／SDPA，关闭 thinking、贪心解码，输出上限 4096，应用层验证 JSON，未启用受约束解码器；请求中的 repetition_penalty 按实际参数执行。它与旧 7B 的后端、选择模型均有差异，不能作为只改变模型一个变量的准确率或速度对照。

本轮由 **Qwen3.5-9B 看原图提取、选择特征并写作**，gpt-6-luna 进行独立像素复核和 Stage 4 Judge，沿用本地神经评分配置。初提取 24 项，现有像素／资格策略放行 15 项，9B 选择 13 项，生成 6 条文本；Stage 4 六例全部 pass、13 项检查均 supported，没有触发 Reflect。19 次真实生成调用（含一次纯文本预检）均正常结束，无空响应或 token 截断；其中六次提取实际传入原图，六次选择和六次写作均没有原图输入。未运行新 SDXL 出图，也未启动 100 实体或全量。

| 实体 | 9B 最终文本内容与助手逐项复核 |
| --- | --- |
| piciform bird | 保留尖直喙和粗粒度羽色，不再加入停栖面；明确不能确认细粒度物种。仅作为样本特征使用 |
| bludgeon | 保留长而刚性的圆柱外形、黑色及来源限制；黑色只属于部分来源实例，文字中 taxonomy 措辞仍冗余 |
| screen | 保留框架、网孔、把手，未重现上轮玻璃格枚举；按具体来源视图使用，不泛化到所有 screen |
| surface lift | 提取到缆线和支柱，但结构事实中夹带橙色下部，仍需复核相关性与抽象程度 |
| mullah | 仍选入白胡须、头巾并生成描述；虽保留身份／角色未验证的限制，这些特征仍与角色语义无关，拒绝采用 |
| liberator | 仍选入棕色眼睛和微笑；没有断言人物做过解放行为，但这些肖像细节与实体语义无关，拒绝采用 |

助手逐项复核保留 3 条有使用限制的候选、1 条待复核、2 条因语义无关拒绝；这不是独立人工标注，也不是模型总体正确率。**像素上成立和选中事实被复述，不能证明这些事实与目标实体语义相关。** 9B 仍把角色人物的服饰／外观标成 entity_feature，Stage 4 也把它们作为已选事实放行；增加身份不确定性声明并不能解决相关性问题。

另冻结同一份 9B 已复核特征，只替换 Stage 2 为 gpt-6-luna。API 选择 7 项，mullah、liberator 均返回 insufficient_evidence；棍棒仅保留结构，screen 保留网孔与框架。该对照**只重跑选择，没有重新写作**，不能称为另一个已经完成的混合生成闭环。surface lift 的橙色细节仍与支柱捆绑在同一事实中，API 选择也保留它，说明仅筛选事实 ID 无法去掉复合事实内部的非必要细节。

现阶段可将 9B 作为提取／写作候选，继续保留独立语义筛选；不能把整条链路交给 9B 后直接扩大批量。还需让上游事实更易于抽象，并让评审独立判断实体相关性。

结果根目录为 `outputs/wn18rr-qwen35-9b-20261107-r1/`：`reviewed/entity_descriptions.json` 保留全部真实文本、来源、逐项复核与旧版对照；`reviewed/summary.json` 为计数；`attempt-r3/` 为最终运行；`selection-control/` 为同事实选择对照。初次缺少共享盘挂载、字符串消息兼容失败、缺失 token 计数接口的记录分别保留，不并入最终语义质量计数。始终顺序申请单卡，只停止过本轮自己提交的未完成试验作业。相关 CPU 回归 39 项通过，另验证真实消息预处理、token 计数和实际文本生成接口。

新增 `configs/qwen35-9b-visual-description.yaml`，试验入口可用 `--model-config`、`--selector-model-config` 指定原生提取／写作与筛选模型。冻结新的 root 并启动已关闭 thinking 的对应服务后，原生阶段使用：

```bash
python -m scripts.retry_entity_visual_description --root <新的冻结目录> --phase native \
  --model-config configs/qwen35-9b-visual-description.yaml \
  --selector-model-config configs/qwen35-9b-visual-description.yaml
```

## 2026-11-07：实体相关视觉语义重试（上一轮）

目标改为保留图片中与实体有关的少量可见语义特征，不要求描述覆盖图谱关系，也不将每张图片的背景、衣着、姿态逐项列出。新增显式任务 `entity_visual_description_v1`，默认旧配置不变。

本轮仍使用前轮的六个真实实体及其 17 张原图：piciform bird、bludgeon、screen、surface lift、mullah、liberator。准入、盲描述和原图字节来自已冻结的历史记录；新提取、筛选和写作输入不提供图谱断言。完整图谱与未选观察继续留在审计产物，隐藏目标文本未使用。此为受控重试，不是重新执行全部预处理的质量基准，也没有新 SDXL 出图。

直接让 7B 看图重试的结果保存在 `outputs/wn18rr-entity-visual-description-20261107/attempt-r4/`：它仍会初提取人物衣着、雪景等内容，并在最终写作补入未选的停栖面、物体形状。该结果不作为“7B 已能可靠抽取相关语义”的证据。

目前采用的受控分工为 **gpt-6-luna 从原图提取并筛选实体相关特征，Qwen2-VL-7B 只根据冻结的特征写作**。写作阶段不再重看原图，避免绕过筛选重新带入背景。两种写作模型使用同一份特征清单；7B 写作调用均确认原图输入数为 0。相关性筛选属于模型判断，不是真值概率。

| 实体 | 最终选择的图片语义 | 7B 写作复核 |
| --- | --- | --- |
| piciform bird | 长而尖的喙、身体浅色横斑、头部浅色条纹 | 保留这些观察，没有再加入停栖面或模糊背景；只作为样本特征 |
| bludgeon | 细长外形 | 明确身份未核验，不再补写颜色、木材或尖端；可用信息较少 |
| screen | 矩形玻璃格、成对面板、圆孔表面 | 仍有重复枚举和不同原图属性合并的问题，待复核；API 对照按不同视图分别描述 |
| surface lift | 上方缆线、竖直支撑杆 | 保留结构，没有人物衣着；身份／机制不确定性未充分表述，待复核 |
| mullah | 无足够实体相关的视觉特征 | 不生成外观清单，不凭头巾或胡须推断宗教角色 |
| liberator | 无足够实体相关的视觉特征 | 不凭肖像、衬衫或背景描述该角色 |

最终清单共 9 项所选视觉特征；7B 写作阶段 4 个实体生成文本、2 个保留证据不足。9 项所选检查均获模型评审 supported、未触发反思，但逐项文本复核仅将前两例列为有使用限制的候选，screen 与 surface lift 仍待复核。未扩大到 100 或全量。

原始结果及最终带来源的清单位于 `outputs/wn18rr-entity-visual-description-20261107/attempt-r5/`，优先查看 `reviewed/entity_descriptions.json` 与 `reviewed/summary.json`。其中并列保留 API 与 7B 文本、特征引用、身份／特征不确定性和复核说明。原生 7B 提取失败对照、初次 API 格式失败、任务标记保存／恢复修复前记录均保留，各轮源码指纹或精确源码归档可追溯，未覆盖历史结果。

实现集中在已有阶段：提取不填每图配额；筛选仅能选择有图片观察的事实，不能选入图谱边；所选特征是可简洁组织的内容范围，不是全部必须枚举的检查清单。只有明确的无依据断言才要求修复，缺少未选图谱关系不再是失败。新任务标记贯穿写作、评分、产物保存及恢复，仍严格核对原始检查 ID 和来源。图片描述与像素复核的额外解释不作为新的写作事实或评分依据。

相关回归 **232 passed**。收紧了评审引用范围，但两条真实旧 7B 负例仍未被 Judge 检出额外细节，不能把模型放行当完整质量保证；当前清单继续保留人工逐项复核状态，不自动删除或改写旧文本。这里的逐项复核由助手完成，非独立人工标注。

新配置：`configs/gpt6-luna-visual-description.yaml` 用于提取和筛选，`configs/qwen7b-visual-description.yaml` 用于只接收特征的 7B 写作。完整 CLI 使用 `--missing-mode text --stage1-model-config configs/gpt6-luna-visual-description.yaml --stage2-model-config configs/gpt6-luna-visual-description.yaml --model-config configs/qwen7b-visual-description.yaml --max-selected-facts 4`，并沿用独立像素复核和 Stage 4 配置。`scripts/retry_entity_visual_description.py` 的 `api`／`native`／`writer` 阶段分别保留 API 对照、原生看图对照与冻结语义清单的 7B 写作，不将不同模型的工作混报。

## 2026-11-07：20 实体补全试跑

**工程链路可批次执行，但当前 7B 配置未通过扩大批量的质量验收；未启动 100 实体或全量。**

当前导入数据的 test.tsv 涉及 5,323 个去重实体，其中 4,493 个没有对齐图片，文本全部存在。本轮在生成前冻结 10 个真实缺图实体、10 个遮蔽文本实体，后者共 30 张原图；覆盖角色、器物、动植物、动作、抽象概念及训练邻边为空的实体。清单是诊断选例，文本例限定为 1–4 张原图，不能作为代表性随机样本或模型总体正确率。

实际执行源码为 `ef809f4`，150 个生产源文件与冻结指纹一致。7B 承担 Stage 1 描述／抽取、Stage 2 选择、文本／生图提示词／反思；gpt-6-luna 承担图片准入、像素复核与 Stage 4 Judge。本地 Qwen3-Reranker、SBERT、CLIP、BLIP 与真实 SDXL 均按配置执行，始终最多申请一张 A100 80GB。只加载 train 图谱；测试边仅用于确定实体范围，不进入生成。

| 项目 | 实测 |
| --- | --- |
| 实体任务 | 20 个均尝试，18 个完成生成与 Stage 4，2 个在 Stage 2 失败 |
| 补图 | 10 个实体、20 张真实 1024×1024 PNG，20 个不同像素 SHA |
| 图片控制 | 每实体一个提示词、两个种子 20260709／20260710；20 steps、guidance 7.5；未触发图片反思 |
| 补文本 | 8 个实体生成文本；2 个 Stage 2 HTTP 400，批次继续执行 |
| 文本反思 | 5 次真实反思全部原样返回，均标记 unchanged_text／unresolved |
| 保守复核索引 | 1 份文本与所选事实相符、1 张图符合通道／水流概念；16 个待复核、2 个失败 |

最后一行依据结构化检查与助手逐项来源／像素复核，不是独立人工真值标注，也不证明下游 KGC 指标提升或精确身份正确率。`doubter` 文本仍有冗余和 `soul` 词义不清的问题；`waterway` 图片只作为通道／水流概念补全。

主要质量失败：`fish`、`migrator` 发生词义偏移；`rock oil` 不能明确表达油；抽象概念被无依据具象化；带原图的若干文本遗漏所选关系，且 7B 不能执行事实编辑。`mullah` 虽复述全部所选外观事实，却额外依据头巾、胡须推断宗教角色，Judge 未检出该额外推断。`playwright` 的图谱检查通过，但存在写作输入之外的人名扩写，保守复核保留为待检查。最终写作未传原图的两个案例事实覆盖较好；这不是同事实集的有／无图对照，不能作因果结论。

在完整试跑结束后做了三项小修复，原始运行文件不改写：

- 将抽取协议里整个字段值为 `__TARGET__` 的身份引用绑定到已知实体名。4 个真实来源字段的 CPU 重放通过，不据此纠正实体类型或宣称重跑了抽取。
- Stage 2 本地修复请求复用已有 `decoder_safe_json_schema`，只移除解码器不支持的语法约束；应用层仍拒绝重复事实 ID。两个失败均为修复请求的 `uniqueItems` 兼容问题。另申请短时单卡，重放 ganoid fish 的同一修复消息：5,194 输入 token、3,072 输出预算不变，真实 7B 返回 HTTP 200，通过事实 ID 校验；未重跑该实体的完整生成／评审。
- 导出除检查已保存的 gate=pass，还要求必需检查均 supported、无矛盾和未解决的额外断言；不将无效修复导出。可选断言的不可核验性继续保留。首批另结合逐项复核，拦下仅覆盖所选事实却有来源外推断的候选。

三个问题在修复前复现为 7 个失败用例；修复后的相关回归为 **188 passed**。同一批次内第二候选／修复候选渲染异常也已隔离，成功 PNG 保留，追加出图预算不会因重新启动而重置。该异常路径由针对性回归验证，本次真实 SDXL 出图未发生渲染异常。

原始记录位于 `outputs/wn18rr-completion-20-20261107-r2/`。原始 `exports/` 与 `completion_manifest.jsonl` 记录的是修复前较弱规则下的临时候选，**采用结果请看 `reviewed/completion_manifest.jsonl` 与 `reviewed/accepted_completions.jsonl`**。`verification/measured-results.json` 保留计数与逐例情况；`verification/review-source-executed.py` 是离线复核时的精确实现。671 个原始文件及冻结数据输入在复核后相同，未用隐藏目标补写结果。

新增入口：`scripts/run_wn18rr_completion_batch.py` 和 `scripts/run_wn18rr_completion_cluster.sh`；配置 `configs/qwen7b-text-vision.yaml` 显式开启所选原图参与写作。为新版本重新试跑时，先在新目录冻结，再按上文 ClusterX 方法顺序申请一张卡；每个实体使用独立 attempt、原图／源码指纹、初始检查点、终态校验与失败记录，已有目录拒绝覆盖。

```bash
PYTHONPATH=. /medisco/linjiashi/venvs/taor-control/bin/python \
  -m scripts.run_wn18rr_completion_batch --prepare --root <新的输出目录>
# 在新分配的单卡作业内：
TAOR_COMPLETION_ROOT=<新的输出目录绝对路径> \
  bash scripts/run_wn18rr_completion_cluster.sh
```

同版本恢复使用相同 root；仅重试失败实体时为集群入口附加 `--retry-failed`。源码变化会拒绝继续旧生成任务。`--review` 是单独的离线复核模式，只读取历史输入和产物、核验指纹并使用当前导出规则，不调用模型，且拒绝覆盖已有 reviewed 目录。本轮短时解码验证入口为 `--selection-control <失败记录路径>`；其通过不等于整体语义质量通过。

十实体 API 文本诊断及形态／展示细节选择、词汇去重、关系语义评审修正见 [本轮优化记录](evaluation/2026-11-07-text-cohort-optimization.zh.md)。对照中的措辞调整未显示明显优势；具体改进来自上游事实组织和检查输入。

实体名称、关系与原图共同参与最终文本写作的配置和实测，见 [原图联合写作](evaluation/2026-11-07-entity-context-image-writing.zh.md)。启用 `configs/gpt6-luna-text-vision.yaml`；只发送已准入且与所选事实关联的原图。

用户指定 gpt-6-luna 后的生成／反思测试、显式模型配置和单卡入口见 [gpt-6-luna 实测](evaluation/2026-11-06-wn18rr-gpt6-luna.zh.md)。

2026-11-06 后续混合评审入口与实测见 [独立审核报告](evaluation/2026-11-06-wn18rr-hybrid.zh.md)。该入口保留本地 7B 生成和真实 SDXL，显式使用独立 API 做准入、像素复核与 Stage 4 Judge；以下早期冒烟记录仍按当时的配置解读。

## 数据输入

主 CLI 与 doctor 均支持 `--dataset-type wn18rr`，不需要生成 MKG-W 中间文件。

必需文件：`entities.txt`、`relations.txt`、`entity2text.txt`、`train.tsv`。实体／关系列表为每行一个原始键；文本为 `实体键<TAB>文本`，图谱为 `head<TAB>relation<TAB>tail`。实体键保持八位数字及前导零，内部整数编号只按 entities.txt 的行顺序生成，不改变对外 entity_key。

- 默认只加载 train.tsv；`--graph-split train-valid` 显式加入 dev.tsv，`all` 再加入 test.tsv。错误三元组与未知键明确报错，不静默丢弃。
- 可选 `wordnet-mlj12-definitions.txt` 格式为 `offset<TAB>__lemma_POS_sense<TAB>definition`。仅使用独立的 lemma 作为公开名称，不把该文件的定义作为附加证据。没有该文件时名称回落为原始实体键，不从隐藏目标文本反推名称。
- 可选 `relation2text.txt` 提供关系名称；不改变原始主客体或入／出方向。
- `image_alignment/entity2images.json` 按原始键映射相对路径列表。一旦存在对齐表，以该表为准，空列表／缺少键保持缺图；不从已排除的原始目录重新绑定图片。声明路径逃逸或文件缺失明确失败。
- 没有对齐表的原生数据，可沿用 images/<原始键>/ 下的图片查找。
- 缺 text 时只提供已有图片与合法 train 邻域，源文本清空，真值仅留在 ground_truth；缺 image 时，全部原图从生成输入移除，保留文本。所有对齐的可用图片都遵循现有多图 Stage 1 协议。

## 当前集群路径

数据解压到新目录：`datasets/wn18rr-import-20261106/WN18RR`，原始压缩包 `/medisco/linjiashi/Code/WN18RR.rar` 保留。该数据有 40,943 个实体、11 种关系、86,835 条 train 三元组。

模型下载到共享目录：

- `/medisco/linjiashi/models/Qwen/Qwen2-VL-7B-Instruct`
- `/medisco/linjiashi/models/SDXL/stable-diffusion-xl-base-1.0`
- 原有 `/medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B`

Qwen、SDXL 通过 ModelScope 的 Qwen/Qwen2-VL-7B-Instruct、AI-ModelScope/stable-diffusion-xl-base-1.0 下载。每个下载文件记录 SHA256，核对仓库清单给出的校验值；完整来源及清单位于本地输出目录。SDXL 下载标准 Diffusers 权重，不使用其他模型的替代图或占位图。项目本地 SDXL 根目录链接到上述共享目录；该链接只在本地 Git exclude 中忽略，不提交权重或绝对路径链接。

隔离推理环境 `/medisco/linjiashi/venvs/taor-inference` 继承已有 worldmm_env 的 Torch 2.7.1，并单独安装 Transformers 4.51.3、Diffusers 0.31.0、Accelerate 1.1.1。不改动已有环境的包；继承环境中 worldmm 自身的版本要求与此覆盖不一致，因此该隔离环境仅用于 TAOR，不用于运行 WorldMM。Qwen 服务使用 `/medisco/linjiashi/venvs/taor-qwen7b`：它继承 skillrl 的 Torch 2.8.0，并独立安装 vLLM 0.11.0。vLLM 的 Qwen2-VL 在 A100 的 80 维视觉注意力头上回落到 upstream FlashAttention，但该分支传入 CPU 的 cu_seqlens，实际预热报 `cu_seqlens_q must be on CUDA`。`scripts/prepare_taor_qwen_vllm.py` 仅在这个 TAOR venv 内将该元数据移到 q.device，不修改共享包、模型权重或提示词。原／修复文件哈希记录在 qwen-vllm-patch.json。服务启用项目 RequestDeadlineMiddleware，保留新版 request_id 与超时取消协议。

重建该服务环境：

```bash
/medisco/linjiashi/miniforge3/envs/skillrl/bin/python -m venv --system-site-packages /medisco/linjiashi/venvs/taor-qwen7b
/medisco/linjiashi/venvs/taor-qwen7b/bin/python -m pip install --ignore-installed --no-deps vllm==0.11.0
python scripts/prepare_taor_qwen_vllm.py
```

以上仅对新建／专用环境执行；不要改装他人的已运行环境。

## 可复跑的单卡入口

原有集群操作说明见 `/medisco/linjiashi/WorldMM/ENVIRONMENT.md` 的 ClusterX 部分；本机旧 clusterx 脚本指向不存在的解释器，当前通过现存 Python 3.13 和原 ClusterX site-packages 运行：

```bash
export CLUSTERX_CFG_PATH=/medisco/linjiashi/clusterx/clusterx.yaml
export PYTHONPATH=/medisco/linjiashi/clusterx/.venv/lib/python3.13/site-packages
/medisco/linjiashi/worldmm_miniforge3/bin/python3 -m clusterx.cli --help
```

配置内凭据仅由 SDK 读取，不打印或复制到 Git。任务申请一节点、一 GPU，队列 `queue-p-reserved-medisco`，不继承当前无 GPU 的 CUDA 掩码；不重启／停止其他旧任务。镜像与 AFS 挂载取自用户已有成功作业的非凭据配置。

实际 GPU 任务入口为 `scripts/run_wn18rr_cluster_smoke.sh`，每次新建 attempt 目录，保留失败日志和已生成结果，已提交任务 ID 与资源参数保存在本地 `cluster-job.json`。脚本等待准备文件、启动本地真实 Qwen 服务，执行 doctor，然后分别跑文本和图片单实体任务；退出时仅关闭自己启动的 Qwen 服务，任务进程退出后释放申请资源。

本次冻结单实体 `02436645`（water deer），一张实际已对齐图片，用于工程冒烟，非质量对照采样；原始记录、train 哈希和源策略在 `frozen-smoke-inputs.json`。图片生成参数为 1024×1024、20 steps、guidance 7.5、base seed 20260708；候选使用适配器派生的实际种子并写入元数据。原生 Qwen 默认 token 预算保留。

运行入口的主要参数为：

```bash
python -m mmkgc.run \
  --dataset-type wn18rr \
  --dataset-dir datasets/wn18rr-import-20261106/WN18RR \
  --graph-split train \
  --entity-manifest outputs/wn18rr-onboarding-20261106/one-entity.json \
  --limit 1 --candidate-budget 1 \
  --missing-mode image --skip-stage4 \
  --stage1-reranker-model-dir /medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B \
  --stage1-reranker-device cuda:0 --sdxl-device cuda:0 \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 --sdxl-height 1024 --sdxl-steps 20 \
  --output-root outputs/wn18rr-onboarding-20261106/runs \
  --run-name <新的运行名>
```

文本任务替换为 `--missing-mode text`。默认输出目录拒绝覆盖已有 run。

本次先使用新分支已有的 `--skip-stage4` 验证 Stage 0–3 与真实模型生成；Stage 4／Reflect 评审、质量对照及全量生成不属于本次冒烟的通过结论。没有 judge 凭据时不把缺失评分伪装成验收通过。

## 验证与运行结果

针对性测试入口：`tests/test_wn18rr_data.py`（20 项）；加上现有多图 Stage 1、共享事实选择和三元组建模，共 **103 passed**。另检查旧 doctor 测试时发现 `test_doctor_minimal_run_is_ok` 仍假设默认不探测模型、不启用评分，与本分支当前默认值不同；未改动其过时断言，也没有宣称全库通过。日志与实际运行状态写在 `outputs/wn18rr-onboarding-20261106/`。`attempt-*/gpu-smoke-completed.json` 只有在真实生成的 PNG 可以解码且尺寸为 1024×1024 后才写入；排队、doctor 通过、提示词生成本身均不代表真实出图成功。


启动阶段失败也保留：最初不带 `-ssh` 的历史镜像标签实际拉取失败；随后 skillrl 原环境触发上述视觉预热错误；旧 graphr1 vLLM 0.6.3 能启动，但不接受新分支 Stage 1 的 request_id 字段，首请求返回 400。后续使用独立环境与新版协议，不删除或替换这些失败记录。每次前一作业已终止／失败释放 GPU 后才提交下一次，始终最多申请一张卡。


## 本次真实实测结果（2026-11-06，上海时间）

成功作业：`taor-wn18rr-qwen7b-sdxl-1106-r3`，单张 NVIDIA A100-SXM4-80GB。

| 项目 | 实测结果 |
| --- | --- |
| Native WN18RR 接入与相关回归 | 103 passed |
| GPU doctor | 数据、Qwen／Stage 1／Stage 2 服务、reranker 文件、SDXL、CUDA 均通过；Stage 4 显式跳过 |
| 单实体补文本 | 1 完成、0 失败，1 条真实 Qwen 文本 |
| 单实体补图片 | 1 完成、0 失败，1 张真实 SDXL PNG；已实际打开检查像素 |
| Qwen chat 请求 | 成功尝试中 9 次 HTTP 200，0 次 400／500；此前启动失败另列保留 |
| 图片参数 | 1024×1024，20 steps，guidance 7.5，实际派生 seed 20260709 |
| SDXL 适配器记录的推理耗时 | 2,983 ms，不包含权重与 pipeline 加载耗时 |
| Stage 4／Reflect | 未执行，ranking_status=not_evaluated |

图片产物：`outputs/wn18rr-onboarding-20261106/attempt-r3/runs/image-qwen7b-sdxl/entities/02436645/generated_images/image_001.png`。同目录 JSON 包含实际提示词、种子、模型路径和无 fallback 标记。

文本产物：`outputs/wn18rr-onboarding-20261106/generated-text.txt`；机器汇总为 `smoke-summary.json`，每个 run 的完整阶段、调用和 metrics 均保留。原压缩包的完整 SHA256 为 `2bb6130b0da0e7ff62d09eab7c9b996f22f3f58ed766596011ec567f8027063b`。

**工程链路已跑通，内容质量未通过。** 实际文本同时将 water deer 写成 rodent 与 marsupial；来源是 Stage 1 从图片抽取的 mf:0001／mf:0002，Stage 2 将其与合法图谱 `(water deer, hypernym, mouse deer)` 一起保留，Stage 3 又复述了这些冲突分类。没有用隐藏真值纠正提示词或删掉失败观察。实际生成图是沼泽中的鹿状动物，能证明真实出图；精确物种身份和事实一致性尚未建立，不能把通用鹿图称为身份保真。下一步应先处理来源抽取与分类冲突，再开展质量验收和多实体比较；本次未启动全量。
