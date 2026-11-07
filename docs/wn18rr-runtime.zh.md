# WN18RR 原生接入与 Qwen2-VL-7B／SDXL 冒烟

## 2026-11-07：9B／API 混合方案的 20 实体闭环（最新）

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
