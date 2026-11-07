# TAOR 集群交接：统一事实建模、共享事实选择与多视角生成

交接日期：2026-11-02。本文按当前代码、已保存的测试报告和真实 API 运行记录核对。集群环境尚未接入核验。

## 1. 接下来要完成什么

**把已上传的新流水线在集群上用原定本地模型跑通，先修复已复现的生成容错问题，再用固定实体和相同预算比较旧版、新版，以及多视角是否真的有用。**

交付目标有两部分：

1. 工程上：Qwen2-VL-7B、Qwen3-Reranker-4B、SDXL、Stage 4 评审与一次 Reflect 能完成小规模真实运行；失败原因明确，单个候选失败不丢弃已完成结果。
2. 实验上：先完成 10～20 个真实实体的受控对照，保存原始输入、每阶段输出、实际图片、失败、耗时与复核意见；据此决定是否扩大运行。结果可以是新版没有优势，不能为了证明方法有效而改变评判标准。

目前适合作为研发快照继续，不是已经完成真实模型质量验收的稳定版。

## 2. 代码、分支和传输范围

| 项目 | 当前状态 |
| --- | --- |
| 仓库 | <https://github.com/Roglic-SWQ/TAOR> |
| 已上传分支 | `pipeline-refactor`，用户要求不用 `codex` 命名 |
| 当前实现提交 | `74bcea40f039f605917a6067ba50532e78ecc9c4` |
| 提交说明 | `Unify multimodal facts and share evidence across generated views` |
| 旧版对照提交 | `1f92a9e0cdd4dda2895f4223781bc23805616264` |
| 代码与文档版本关系 | 上述提交是实现基点；本交接随其后的独立文档提交提供 |
| PR | 本次没有创建 PR |
| 最近 API 实验后的生产代码修复 | 尚未实施，下面的待办仍然存在 |

实现已上传 `pipeline-refactor`；本文与交接任务书也随其后的独立文档提交提供。首次 clone 该分支即可读取两份文档。分支 HEAD 可以是后续文档提交，不能因此误认为实现基点已经改变。

### 2.1 Git 中有的内容

本次流水线代码、测试、运行配置示例，以及既有方法／审查文档已在提交中。重点文档：

- `docs/stage0-evidence-selection.zh.md`
- `docs/stage1-stage2-runtime.zh.md`
- `docs/stage2-candidate-plan.zh.md`
- `docs/gpt4omini-prompts-v2.zh.md`
- `docs/fact-observation-reflection.zh.md`
- `docs/evaluation/2026-11-02-pipeline-review.zh.md`
- `docs/evaluation/2026-11-02-media-two-step.zh.md`
- `README_ENV.md`、`docs/mkgw-missing-image-output-layout.zh.md`

这些文档包含不同时间的验证结论。例如上传前报告中的“未调用真实模型”，指上传前那一轮检查；之后另做了本文第 5 节的 API 小实验，不能混为同一轮验证。

### 2.2 Git 中没有的内容

- 模型权重、MKG-W 数据、`.env` 凭据、`outputs/` 运行产物。
- 本机的 `.claude/`、`WN18RR/`。
- 本机未跟踪的 `docs/mmkgc-collaborative-completion-implementation-plan.zh.md`：它是独立研究计划，不能据此声称对应算法已经实现。
- 本次 API 对照脚本与结果也在 `outputs/`，只拉 Git 分支拿不到。

本次提供的交接压缩包另行包含最新 API 对照的证据目录、本文和简短任务书；不含模型、真实 MKG-W 数据或本机凭据。压缩包文件列表和 SHA256 清单可用于传输核对。

完整压缩包仍为单独传输材料，不在 Git 中；没有它也可先做环境盘点、代码检查、两个 P0 问题的最小回归与修复。需要逐条审计历史 API 调用时再补传原始证据，不因缺少压缩包停止所有工程工作。压缩包是生成当时的快照，其中关于文档尚未推送的说明已被当前 Git 文档的发布状态取代。

本机原 API 环境文件位置为 `D:\CodE\ai lab\DSH-Med\.env.standardization.local`。这是历史授权路径，不是集群可用路径。集群凭据由使用者在集群本地配置，勿写入 Git、任务日志或交接文档。

## 3. 已确认的方法设计与当前实现

研究动机：**保留实体的语义、属性和关系，通过已有模态与图谱的共同证据补充缺失模态，最终服务实体识别和 MMKGC。** 当前工程目标不是把所有非视觉关系硬画进图里，也不是用相似度冒充事实正确率。

主流程如下：

```text
train 图谱 + 该实体已有的非目标模态
  → Stage 0：完整一跳关系／事实目录
  → Stage 1：统一、有来源的主谓宾事实；媒体先描述，再回看原媒体抽取
  → relevance rerank：保留语义、关系和目标模态用途的粗筛
  → Stage 2：LLM 选择一份共享事实集合
  → Stage 3：同一份事实生成文本，或协调生成多个图片提示词并调用 SDXL
  → Stage 4：相同事实参照下的观察与融合排序
  → 一次有条件的 Reflect 修复／重新评分
```

### 3.1 Stage 0：先完整保存，不在数据入口任意截断

已实现：

- 新主线保存实体完整、去重的一跳图谱事实目录，保留原 subject、relation、object 和方向。
- MKG-W 邻居加载默认不再按文件顺序截到 32 个。
- 新主线 Stage 0 不先让 Qwen 硬删除关系；历史筛选分支保留用于兼容。
- 修正过旧路径 `selected_relations` 与 `stage0_v3` 决策记录不同步的问题。

边界：完整目录指当前输入图谱切分中的一跳邻域，不是整个知识图谱或无限多跳检索。完整保存也不等于后续输入模型的 top-k 已获得高召回保证。

入口：`mmkgc/data/mkgw.py`、`mmkgc/stages/coarse.py`、`mmkgc/stages/evidence_sampling.py`。

### 3.2 Stage 1：多来源统一为有证据的三元组

默认协议：`grounded_triples_v2`。

- 图谱三元组由程序原样保留，模型无需重新转述整张关系表。
- 文本抽取的是有来源的主谓宾事实，可以是属性，也可以是作者、地点等关系；不是只抽 attribute。
- 相同主谓宾可以合并来源，保留证据引用。描述和基于该描述抽取的事实不是两份独立佐证。
- 模型看到最多 32 条代表性图谱身份示例；完整图谱目录仍被程序保存。这里的 32 是模型输入视图上限，不再是原始事实总数上限。
- 实体类型来自有证据的模型画像，属于推断结果；不会作为 rerank query 中的权威硬条件。
- 目标模态按缺失任务遮蔽，不能把隐藏目标内容拿来抽取事实。

方向示意，以下 ID 只说明格式：

```json
{
  "fact_id": "example:incoming-designer",
  "subject": "Lina Vale",
  "relation": "designed",
  "object": "Harbor Museum",
  "direction": "in",
  "evidence_refs": ["example:train-edge"],
  "source_modalities": ["graph"]
}
```

这里正在处理 Harbor Museum，`in` 表示相对目标实体的入边。原三元组仍然是“Lina Vale 设计 Harbor Museum”，不会为了以博物馆开头而颠倒主客体。后续生成和评分继续使用这个方向。

媒体采用用户确认的两步设计：

| 来源 | 第一步 | 第二步 |
| --- | --- | --- |
| 图片 | 看原图生成自然语言描述 | 同时看同一张原图及描述，抽取三元组 |
| 视频 | 看抽帧及可用音轨，生成带时间信息的描述 | 再读取冻结后的同一份媒体输入和描述，抽取三元组 |
| 音频 | 听原音频，生成转写／声音描述 | 再听同一份音频并参考描述，抽取三元组 |

当前媒体边界：视频约 1 fps 抽帧，最大帧宽 768；提取第一音轨为单声道 16 kHz WAV。单来源／内联解码数据限制 7 MiB，时长限制 120 秒，超限拒绝，不静默截取前半段。依赖 ffmpeg／ffprobe。低帧率抽样仍会遗漏短时事件，不能称为逐帧完整感知。

Qwen2-VL 不支持听音频；涉及音频的 Stage 1 必须接音频能力模型，示例为 `configs/media-stage1-api.yaml`。不要把视频／音频知识建模支持误写成整个系统已经支持视频／音频生成：**当前主 CLI 的缺失模态只有 text 和 image。** MKG-W 默认 loader 也不会凭空提供视频、音频，需要数据适配器／记录中确实包含相应来源。

Stage 1 默认共享 180 秒阶段预算，最多两次业务修复；修复重建原始输入，不把错误回答当新证据。网络重试也消耗阶段总预算。

入口：`mmkgc/stages/triples.py`、`modeling.py`、`sources.py`、`semantic.py`；`mmkgc/adapters/image_knowledge.py`、`media_knowledge.py`、`qwen.py`。

### 3.3 Rerank：任务相关性粗筛

已按讨论改成“哪些事实有助于实体语义、重要关系、身份识别及目标模态表达”，不再对照几乎相同的原证据做事实性匹配。

- Query：动态实体标签＋目标模态；不固定 museum，不强制使用推断 entity type。
- 公共 instruction：保留实体语义、定义性属性和有意义关系，同时考虑模态表达；不能画出来的关系也可能有身份价值。
- Document：候选事实的 subject、relation、object、direction。
- 对全部事实打分，默认每批 8 条，向 Stage 2 暴露前 64 条；完整目录仍保留。
- `relevance_score` 是相关性，不是真实性／置信度，不存在“0.99 就证明正确”的含义。
- 版本：`entity_semantics_modality_relevance_v2`。

完整 catalogue 解决了“证据被入口永久删除”的问题；top-64 是否漏掉关键事实仍需通过共同事实清单测量。暂时不要再叠加 primary、must-include 分组、按实体类型配额等规则。

入口：`mmkgc/adapters/evidence_reranker.py`。类名 `Qwen3EvidenceReranker` 和旧 `score_knowledge_evidence` 仍存在兼容代码，判断实际流程应看调用位置和 `score_semantics`，不要只看历史类名。

### 3.4 Stage 2：只选一份共享事实

默认协议：`shared_fact_selection_v1`。模型只返回：

```json
{"status":"ready","selected_fact_ids":["实际存在的事实 ID"]}
```

已实现：

- 最多选择 `max_selected_facts` 条，默认 64；允许少选，不凑满。
- `ready` 至少一条；证据不足用 `insufficient_evidence` 和空列表。
- 程序校验 ID、证据引用、重复项、数量和结构；从 Stage 1 绑定事实原文，模型不重写事实。
- 不再输出 primary／secondary／context／excluded、focus、候选事实组合、槽位、配额或逐事实评分理由。
- 内部 `fs:shared` 只是程序供所有候选复用的上下文，不是额外的模型决策。
- 默认最多两次业务修正、一次网络重试，共用 300 秒预算；重复无效回答提前结束。

三个参数要分清：

| 参数 | 控制什么 | 默认 |
| --- | --- | ---: |
| `--relevance-fact-budget` | 粗筛后给 Stage 2 看多少条 | 64 |
| `--max-selected-facts` | Stage 2 最多选择多少条 | 64 |
| `--candidate-budget` | Stage 3 初始生成尝试次数 | 文本 1／图片 3 |

历史 v1/v2 候选组合和旧角色协议保留用于重放旧任务；它们不是当前新任务默认。

入口：`mmkgc/stages/candidate_planning.py`、`plan_context.py`、`mmkgc/adapters/relation_planner.py`、`mmkgc/plan_schema.py`。

### 3.5 Stage 3：同一组事实，多种表达

文本：`entity_semantics_text_v4`，默认一条自然英文描述。长度跟随事实内容，不再强制 1～2 句；保留方向、时间、数量、归属、否定等限定，不往证据之外扩写。

图片：`entity_semantics_image_v4`。同一组事实驱动多个提示词，模型按顺序读取本轮此前接受的提示词，协调表达重点和构图；之前的生成内容只用于协调，不能成为事实证据。

- 图片提示词输出 `positive_prompt`、`used_text_cues`、`omitted_risky_details` 三字段。
- 正向提示词最多 80 个英文词、650 个字符，没有最低长度要求。
- 可见属性决定画面，非视觉关系可用于身份背景，不要求全部道具化。
- 未知真实外观采用保守概念表达，不得声称是实体真实照片或官方封面。
- 不再用固定五视角／focus 模板，也不让 Stage 2 为每张图另选一组事实。
- 主 CLI 默认图片 3 次、文本 1 次，可显式指定任意正整数；不是方法硬上限。部分历史低层 helper 默认仍为 5，勿据此判断主线数量。
- 归一大小写与空白后完全相同的提示词跳过 SDXL；不是语义去重，也没有按质量自动停止。
- SDXL 顺序出图、候选分别使用种子；实际候选数可能少于尝试预算。Reflect 还可能产生额外修复版本，应单独计数。

**已知缺陷：单个图片提示词调用异常仍可能中止整轮 Stage 3，导致前面成功生成的候选没有进入返回结果。尚未修复。** `--continue-on-error` 只在实体层继续，解决不了这个问题。

入口：`mmkgc/stages/generation.py`、`mmkgc/adapters/qwen.py`、图片本地生成适配器；提示词说明见 `docs/gpt4omini-prompts-v2.zh.md`。

### 3.6 Stage 4 与 Reflect：共同参照评分，不能据此直接宣布事实正确

- 保留 GPT-4o-mini 事实观察与多信号融合评分。
- 所有共享事实候选都以同一份 Stage 2 事实集合构建参照，不能缩小某个候选的事实列表以获得更容易的分数。
- 图片事实编入内部 `should_include`：未展示会影响覆盖，但不单独因遗漏而强制修复；直接矛盾仍可能触发修复。文本对所选事实的覆盖约束更强。
- 内部 `must_include` 等字段是程序派生的执行约束，不是重新引入 Stage 2 的事实角色规划。
- 逐事实观察和最终 hard gate／融合排序仍是不同机制；不是 GPT 提到任何问题都会自动淘汰。观察置信度、冲突类别和修复条件都影响最终动作。
- 一次有条件的 Reflect 沿用原事实集；修复不能靠删除原事实降低难度，应修正旧问题并保留正确内容。失败／更差的修复不能覆盖更好的原候选。
- `REPAIR_CONFIDENCE=0.8` 尚未在真实基准上校准。

这次 API 小实验没有跑生产 Stage 4 或 Reflect；也没有验证最终自动淘汰策略的真实误报／漏报率。

入口：`mmkgc/ranking/context.py`、`engine.py`、`mmkgc/observation.py`、`mmkgc/pipeline.py`。

## 4. 已完成的测试与未通过范围

以下是已完成记录，不代表当前集群已经复现：

| 检查 | 结果 | 证据边界 |
| --- | --- | --- |
| 共享选择等较早专项选择 | 358 passed，20 deselected | mock／接口测试；排除了 8 个历史提示词快照及 12 个 Bash 路由测试 |
| 上传前最终专项选择 | 134 passed | 覆盖媒体、三元组、Stage 0、相关性、共享选择及修复的两个测试 |
| 同环境旧版扩大扫描 | 799 passed，179 failed，12 deselected | Windows／Python 3.13；另忽略 3 个导入 fcntl 的模块 |
| 同环境新版扩大扫描 | 974 passed，178 failed，12 deselected | 相同扫描条件；逐测试 ID 核对新增失败 0 项 |
| 语法、diff、凭据模式检查 | 通过 | 静态检查，不是效果验证 |

178 项既有失败没有全部修完，包含旧接口／提示词快照、缺少模型依赖、Windows 路径／符号链接问题。不能写“全仓测试通过”。

集群应先跑专项，再检查 Linux 可运行的历史排除项；不得照搬 Windows 的 `fcntl` 排除。建议专项命令：

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_media_knowledge.py \
  tests/test_image_knowledge.py \
  tests/test_shared_fact_selection.py \
  tests/test_stage0_evidence_selection.py \
  tests/test_task_relevance.py \
  tests/test_triple_modeling.py \
  tests/test_model_fusion_engine.py::test_text_graph_extraction_runs_under_judge_concurrency \
  tests/test_qwen_adapter.py::test_image_prompt_generator_uses_fresh_repair_prompt_for_length_truncation \
  --tb=short
```

完整审查见 `docs/evaluation/2026-11-02-pipeline-review.zh.md`。本次交接只新增文档与传输材料，没有为交接重新运行整个测试库。

## 5. 最近真实 API 小实验：结果、失败与限制

### 5.1 到底运行了什么

目录：`outputs/comparisons/2026-11-02-old-new-api-pilot/`。

- 源码固定为旧 `1f92a9e` 与新 `74bcea4`，分别用独立源码快照运行。
- 四个合成压力案例，不是从真实 MKG-W 随机抽样。
- 真实调用 Stage 0～3 的模型请求；图片分支仅生成图片提示词，**没有调用 SDXL 画图**。
- 生成 API 请求模型 `gpt-5.6-luna`，返回标识 `gpt-5.6-luna-2026-07-09`；这只是服务端报告的模型身份，不是独立验证过权重。
- Rerank 调用 API 的 `Qwen/Qwen3-Reranker-4B`，未确认服务端拼接模板与本地实现完全相同。
- 不含本地 Qwen2-VL、生产 Stage 4、Reflect、实际图片质量或下游 KGC 评测。
- 温度 0，文本 1／图片提示词 3 次预算，双方相同设置。隐藏目标文本未进入生成请求，输入仅用合成 train 图谱及合法已有模态。

四个案例：

| 案例 | 检查什么 |
| --- | --- |
| `dense_museum` | 36 个邻居，30 条目录噪声后面才有关键视觉属性和入边设计者；检查早截断损失 |
| `incoming_authorship` | 作者→小说的入边、体裁、语言、改编关系；检查方向与无外观时的保守表达 |
| `nonvisual_person` | 作曲家、出生地、作品、乐器和奖项；没有脸、衣着或演出事件证据 |
| `qualified_visuals` | 蓝色、恰好两个圆窗、平屋顶、东侧入口、2012 年前后用途；检查数量与时间限定 |

### 5.2 文本结果

用两版共用的 23 条事实清单逐句核对，并保存原文中的对应字面片段。该复核由 Codex agent 完成，不是独立人工盲标。

| 案例 | 旧版覆盖 | 新版覆盖 |
| --- | ---: | ---: |
| 博物馆 | 2/6 | 6/6 |
| 小说 | 2/5 | 5/5 |
| 作曲家 | 3/5 | 5/5 |
| 塔楼 | 3/7 | 7/7 |
| 合计 micro coverage | **10/23，43.5%** | **23/23，100%** |

两版文本均 4/4 完成。例：

```text
旧版：Harbor Museum is a museum located in Bayhaven.

新版：Harbor Museum is a museum located in Bayhaven and designed by Lina Vale.
Its entrance is an arch, its facade is made of red brick, and it has a square clock tower.
```

只可得出“这些合成案例中，新版文本保留的共同事实更多”。不能称为真实 MKG 召回率或无幻觉证明。新版作曲家文本出现 `*Winter Signals*`，仍违反 plain-prose 的格式要求。

### 5.3 图片提示词主实验失败了什么

原配置 `image_prompt_max_token=384`：旧版 3/4 条件完成，新版 1/4 条件完成。

最终失败都出现在 Stage 3：HTTP 200，但 `finish_reason="length"`、`content=""`，返回的 completion tokens 和 reasoning tokens 都为 384。推理消耗用完预算，没有留下最终 JSON；程序先解析 JSON，结果变成 `JSONDecodeError`。一次格式修复仍使用相同上限，继续失败。

这个观察说明本次 API 推理模型与 384-token 预算不匹配，不能直接推出本地 Qwen2-VL 的失败率，也不应该把所有模型的默认上限一律改成 2048。

另外，新版作曲家案例先成功完成一个提示词，第二个失败后整轮失败，暴露候选隔离／部分结果保存缺陷。

补充诊断单独冻结方案：双方统一改为 2048 tokens，读取各自原来的 Stage 1／2 产物及其哈希，只重跑图片 Stage 3。双方均 4/4 完成。它与原 384 主实验必须分别报告，不能拿补跑结果替换原失败率。

### 5.4 提示词质量发现

- 旧版把小说体裁／标题扩成“穿时代服装的旅人和木船”；把获奖关系扩成“演出后在音乐厅领奖，还有观众”。这些具体场景无源证据。
- 新版通常更贴近证据，但博物馆候选仍写出 `an attached square clock tower`；原事实只有“有方形钟楼”，没有说明与主体相连，存在未经支持的空间细化。
- 新版小说以稿纸、笔、胶片表达作者／改编关系，属于概念表现；能否帮助实体识别尚未证实。
- 三个新版提示词常重复相同事实和道具，主要改机位／措辞。还不能证明多视角优于同提示词换种子。

这些是实际提示词原文检查，不能等同于最终像素质量。

### 5.5 外部自动评审也有问题

另用 GPT-4o-mini 对共同事实清单做 A/B 交换位置的盲评。文本 8 次中有 3 次、补充图片提示词 8 次中有 4 次未通过严格事实 ID／完整性检查，共 7/16 次无效；部分格式有效的回答仍漏掉明显无依据场景，小说图片提示词的偏好随位置改变。

因此不把自动汇总分数当作新版质量优越的主要证据。它也不是对生产 Stage 4 的实验。保留失败和原始响应，供后续改评估流程。

### 5.6 实验程序本身的修正

最初 API reranker 桥只实现 `.score_batch`，旧版调用 `.score`，造成前 5 个旧版条件在桥接层失败。已修正桥接器并重跑这 5 个无效条件，原错误记录保留于 `old_harness_attempt_0`，不计入方法失败率。没有因此重跑并隐藏有效轨道中的真实生成失败。修正打断了原计划的完整交错执行顺序，勿宣称严格按原交错顺序完成。

### 5.7 证据文件索引

| 文件／目录 | 用途 |
| --- | --- |
| `comparison.html` | 可直接打开的文本／图片提示词并排报告，含失败与限制 |
| `summary.json` | 完整机器可读汇总；含不能直接采信为主结论的自动评审统计 |
| `manifest.json`、`headroom_protocol.json` | 原实验与补充诊断协议、固定案例和预算 |
| `text_fact_audit.json` | 23 条共同事实、逐项覆盖和原文字面证据 |
| `review_findings.json` | 提示词原文审查发现 |
| `harness_correction.json` | 桥接器错误、排除与重跑说明 |
| `preflight.json` | 当时本机与 API 可用情况 |
| `runs/` | 原主实验逐阶段产物、结果、请求／响应及日志 |
| `image_headroom/` | 双方 2048-token 补充轨道，含冻结来源哈希 |
| `judging/`、`judging_headroom/` | 自动盲评原始请求、响应与失败 |
| `fixtures/`、`snapshots/` | 合成输入与当时两份源码快照 |
| `experiment.py`、`judge.py`、`text_audit.py`、`report.py` | 历史实验／审计脚本 |

**便携性限制：** 历史 `experiment.py`、`judge.py` 写死了本机 Windows 环境文件位置，并用特定 API 桥接器；不是可在 Linux 上原封不动启动的本地模型实验入口。可读取记录，不应直接运行旧脚本再产生付费请求。若需复跑，先另建可配置脚本并冻结新协议，保留原证据。

## 6. 集群上优先做的工程工作

### P0.1 空响应／截断要有正确错误分类

修改前先依据第 5 节记录的字段构造最小回归：HTTP 200＋空 content＋`finish_reason=length`；原始证据可用后再核对完整调用。区分网络失败、截断、空输出、无效 JSON、语义合同错误。

验收：

- 在 JSON 解析前识别结束原因，错误日志保留模型身份、预算、finish_reason、usage、候选 ID 和阶段，不打印凭据。
- 截断后的修复仍然有限次、有总时限；支持按模型配置输出预算，不能无限重试。
- 修复只用原事实与证据，不把坏响应转为事实。
- 现有原生 Qwen 配置不因一次替代 API 实验而被无证据整体放大。

主要位置：`mmkgc/adapters/qwen.py`。具体实现先沿实际响应解包与图片格式修复调用链核对，避免在多个层重复重试。

### P0.2 候选级隔离与及时保存

最小回归：候选 1 成功、候选 2 超时或无效、候选 3 成功；以及全部失败。

验收：

- 单候选失败记录为失败项，继续剩余候选；已成功项仍可评分、输出和恢复。
- 图片字节、提示词与实际调用记录及时保存；进程重启不会把已完成结果当作从未生成。
- 后续提示词协调只读已接受的提示词，不读失败输出。
- ID、种子、预算与失败原因可追溯，不为凑满数量无限补生成。
- 全部无可用结果时明确返回失败／无有效候选，不包装成成功。
- 兼容现有旧产物读取，针对真正断点恢复做检查。

主要位置：`mmkgc/stages/generation.py`、`mmkgc/pipeline.py`、`mmkgc/io/writer.py`、恢复逻辑。修复应聚焦，不为了容错再新增一套复杂调度器。

### P1.1 收紧“表达自由”与“增加事实”的边界

相机、光照、画面布局可以变化；钟楼是否相连、人物穿什么、是否发生演出／领奖等需要证据。对已复现的空间细化和捏造事件补短小、通用的约束，避免不断追加几十条例外。

概念表现与实体真实外观要区分。无视觉证据时不能因“好看”就把泛化人物／建筑称为身份保真。新版文字中的 Markdown 格式问题可同时小修。

验收以实际原文、实际图像和共同证据为准，不以模型自评“我遵守了”作为通过条件。

### P1.2 原生模型的小规模实验

先完成第 7～9 节。工程修复后要记录新的提交，不能继续把修复版结果都标成 `74bcea4`。

## 7. 集群环境准备

### 7.1 先确认现场，不直接改正在运行的环境

集群的调度器、分区、账号、队列、GPU 型号、共享存储、网络与现有任务均未知。先拿到分配的计算节点，再记录：

```bash
pwd
git status --short --branch
nvidia-smi
python --version
python -m pip freeze > /tmp/taor-environment-before.txt
python -c 'import os; print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))'
```

`README_ENV.md` 的 Ubuntu 22.04、Python 3.11、双 RTX 4090 是历史部署说明，不是对当前集群的确认。已有可用环境先检查，不在共享环境里直接升级依赖；缺依赖时建立独立环境。

### 7.2 取代码并固定版本

新的工作目录可执行：

```bash
git clone --branch pipeline-refactor https://github.com/Roglic-SWQ/TAOR.git TAOR
cd TAOR
git rev-parse HEAD
git show --no-patch --oneline 74bcea40f039f605917a6067ba50532e78ecc9c4
```

实验必须固定提交。如果分支之后前进，建立独立 detached worktree，不对已有任务目录 `reset --hard`：

首次 clone 后先保留并读取交接文档。`74bcea4` 是实现基点，早于交接文档提交；对照 worktree 可能没有本文，继续从主工作目录读取即可。

```bash
git worktree add --detach ../TAOR-old-1f92a9e 1f92a9e0cdd4dda2895f4223781bc23805616264
git worktree add --detach ../TAOR-new-74bcea4 74bcea40f039f605917a6067ba50532e78ecc9c4
```

代码 worktree 不会自动带模型、数据和 `.env`。每个工作目录都要核对资源路径；共用只读权重与数据即可，不必复制多份。开始修复前再建立用户认可名称的开发分支并记录基点，保留上述冻结版本。

### 7.3 依赖与资源

主依赖入口 `requirements.txt`，历史固定组合包括 Torch 2.5.1+cu121、Transformers 4.51.3、Diffusers 0.31.0、vLLM 0.6.4.post1。Qwen 服务使用 `.venv-qwen7b` 覆盖 Transformers 4.46.1／Tokenizers 0.20.1，主流水线中的 Qwen3 reranker 需要 Transformers 至少 4.51。

全新隔离环境按 `README_ENV.md` 安装，并保存实际成功的版本清单；这套依赖尚未在当前集群从零安装验收。不要按文档中的 sudo 驱动安装步骤修改受管理集群节点。

必须准备的资源：

| 资源 | 当前项目约定路径 |
| --- | --- |
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct/` |
| Qwen3-Reranker-4B | `Qwen3-Reranker-4B/` |
| SDXL Base | `SDXL/stable-diffusion-xl-base-1.0/` |
| CLIP | `CLIP/ViT-B-32.pt` |
| BLIP ITM | `blip-itm-large-coco/` |
| Sentence-BERT | `Sentence-BERT/all-mpnet-base-v2/` |
| BERTScore，辅助可选 | `models/bertscore/roberta-large/` |
| 数据 | `datasets/MKG-W/`，包括映射、train／valid／test、实体／关系信息和可用图片 |
| 媒体工具 | 系统可用的 `ffmpeg`、`ffprobe` |

当前 SDXL 路径校验限制为项目的指定 Base 目录，不能以为随便给一个外部 `--sdxl-model-dir` 就能通过。若共享存储用链接，检查 `mmkgc/defaults.py::validate_sdxl_model_dir` 的 resolve 规则；只链接 Base 子目录到外部可能违反必须位于 SDXL 根目录的检查。可在确认目标后链接整个模型根目录并保持子目录结构，不覆盖原有目录。

Stage 4 的 GPT judge 使用 `OPENAI_BASE_URL`、`OPENAI_API_KEY`。本地 Qwen 默认 `http://127.0.0.1:12345/v1`、占位 key `EMPTY`；外部带密钥 API 使用可验证 HTTPS，不把明文 HTTP 地址直接复用为鉴权传输。

`/models` 返回正常和 doctor 通过只说明服务／部分资源可达，不证明模型实际能看图、听音频，或完整权重已经成功载入。必须有一次真实最小请求与实际出图。

### 7.4 GPU 分配，尤其注意集群掩码

历史部署是 Qwen 单独一张 GPU，reranker＋SDXL＋本地评分使用另一张；这不是强制硬件规格，也不是对显存足够的保证。先实际测峰值和模型驻留。

`scripts/run_qwen7b.sh` 会以 `QWEN_GPU` **覆盖** `CUDA_VISIBLE_DEVICES`；部分批处理脚本也会覆盖掩码。集群调度器可能已分配设备编号、UUID 或 MIG，不能盲目照抄物理 0／1，从而访问未分配资源。

仅当已确认本节点分配恰为物理 0 和 1 时，历史示例是：

```bash
# 终端 A：同一已分配节点，Qwen 服务
QWEN_GPU=0 bash scripts/run_qwen7b.sh

# 终端 B：后续主流水线命令只暴露物理 GPU 1
export CUDA_VISIBLE_DEVICES=1
```

终端 B 中设备应写 `cuda:0`，并用 `configs/stage4-model-fusion-gpu1-visible.yaml`，该文件各本地 scorer 已写为 `cuda:0`。普通 `stage4-model-fusion.yaml` 含 `cuda:1`，在仅暴露一张卡时会出错。

若调度器分配不同，用现场核对后的掩码和启动方式，或直接在保留调度器掩码的独立作业／服务中启动模型。本文不提供猜测的 Slurm 分区／账号，也不代表已提交任何集群作业。

## 8. 当前版本的最小启动步骤

下列参数名已与当前 `mmkgc.run`／`mmkgc.doctor` parser 核对。命令在项目根目录运行，假设资源路径已准备好、Qwen 服务可达、GPU 掩码已由上一节正确设置。本机仅检查了命令结构，没有在集群执行。

### 8.1 建立一个真实实体的 smoke manifest

下面脚本从实际 train 邻域里找一个至少有关系、原描述和可用图片的实体，供人工遮蔽目标模态后做两种 smoke。它只建立一个启动样例，不是有代表性的评测抽样。使用新目录，重复运行会拒绝覆盖。

```bash
python - <<'PY'
import json
from pathlib import Path
from mmkgc.data.mkgw import MKGWDataStore

store = MKGWDataStore("datasets/MKG-W", graph_split="train")
target = Path("outputs/cluster-smoke-inputs")
target.mkdir(parents=True, exist_ok=False)
for entity_id in sorted(store.entity_id_to_uri):
    if (store.neighbors(entity_id)
        and store.entity_metadata(entity_id)["description"].strip()
        and store.image_path(entity_id)):
        rows = [{"entity_key": store.entity_key(entity_id)}]
        (target / "one.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(rows)
        break
else:
    raise SystemExit("没有找到同时具有 train 邻域、原描述和图片的 smoke 实体；检查数据后另选真实 ID。")
PY
```

Manifest 要求对象数组，例如 `[{"entity_key":"真实实体键"}]`；不是纯字符串数组。也支持包含 `samples` 数组的 JSON 和每行一个对象的 JSONL。正式对照必须冻结同一批 entity_key 和顺序。

MKG-W 还支持 `partial_missing` 标注，它会保留部分来源，不等同完全遮蔽。正式全缺失对照先检查实体元数据中的相关状态，保存实际 source／target policy，不能只看 CLI 模式就认定目标全部被遮蔽。

### 8.2 Doctor

```bash
python -m mmkgc.doctor \
  --dataset-dir datasets/MKG-W \
  --graph-split train \
  --missing-mode image \
  --stage1-reranker-device cuda:0 \
  --sdxl-device cuda:0 \
  --stage4-config configs/stage4-model-fusion-gpu1-visible.yaml
```

doctor 不替代实际推理测试。服务探测主要访问 `/models`；reranker 在 doctor 中主要校验本地文件，不等于跑过全部打分。

### 8.3 单实体补文本

```bash
python -m mmkgc.run \
  --dataset-type mkgw \
  --dataset-dir datasets/MKG-W \
  --graph-split train \
  --missing-mode text \
  --entity-manifest outputs/cluster-smoke-inputs/one.json \
  --limit 1 \
  --candidate-budget 1 \
  --relevance-fact-budget 64 \
  --max-selected-facts 64 \
  --stage1-reranker-model-dir Qwen3-Reranker-4B \
  --stage1-reranker-device cuda:0 \
  --stage1-model-config configs/qwen7b-stage1.yaml \
  --stage2-model-config configs/qwen7b-stage2.yaml \
  --model-config configs/qwen7b-mainline.yaml \
  --stage4-config configs/stage4-model-fusion-gpu1-visible.yaml \
  --output-root outputs/cluster-smoke \
  --run-name "new-text-$(date +%Y%m%d_%H%M%S)"
```

### 8.4 同实体补图片，真实调用 SDXL

```bash
python -m mmkgc.run \
  --dataset-type mkgw \
  --dataset-dir datasets/MKG-W \
  --graph-split train \
  --missing-mode image \
  --entity-manifest outputs/cluster-smoke-inputs/one.json \
  --limit 1 \
  --candidate-budget 3 \
  --relevance-fact-budget 64 \
  --max-selected-facts 64 \
  --stage1-reranker-model-dir Qwen3-Reranker-4B \
  --stage1-reranker-device cuda:0 \
  --stage1-model-config configs/qwen7b-stage1.yaml \
  --stage2-model-config configs/qwen7b-stage2.yaml \
  --model-config configs/qwen7b-mainline.yaml \
  --stage4-config configs/stage4-model-fusion-gpu1-visible.yaml \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-device cuda:0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-guidance-scale 7.5 \
  --sdxl-seed 20260708 \
  --output-root outputs/cluster-smoke \
  --run-name "new-image-$(date +%Y%m%d_%H%M%S)"
```

注意：

- 主 CLI 跑完整流水线，包含 Stage 4 和可能的一轮 Reflect；不只是 Stage 3。
- 仅支持 `--missing-mode text` 或 `image`，没有 `both`。
- 旧提交没有 `--max-selected-facts`、`--relevance-fact-budget`，不能把新命令原封不动给旧版。对照 harness 以共同参数加各方法真实支持的接口运行，另存解析后的最终配置。
- `configs/qwen7b-mainline.yaml` 的 `prompt_profile: qwen7b_original_v1` 是兼容字段；当前结构化流程使用的实际模板还由代码协议决定，需看请求日志中的 v4 版本，不能只看这个配置字符串。
- 默认图片 token 上限仍为 384。先用原生 Qwen 检查；若修改，另存配置并注明原因，双方对照保持可比，不覆盖主实验原结果。
- `--continue-on-error` 可用于后续多实体试跑；有实体失败时仍返回非零退出码，并保存失败汇总，不是“忽略错误等于成功”。

### 8.5 验收看哪些输出

每个新 run 独立目录，已有目录会拒绝覆盖。重点查看：

```text
<run>/config.json
<run>/results.jsonl
<run>/metrics_summary.json
<run>/failures.json                         # 有实体失败时
<run>/entities/<实体目录>/
  0_coarse_relation_filtering.json
  1_knowledge_modeling.json
  2_fine_grained_relation_selection.json
  3_relation_alternation_generation.json
  4_kg_aware_candidate_consistency_ranking.json
  final.json
  model_calls.jsonl
  image_model_calls.jsonl                   # 有相应调用时
  ...实际图片／元数据／修复产物
```

至少人工看一个完整实体：完整目录数量、入边方向、事实来源、rerank top-k、Stage 2 选择、三个实际提示词、生成图片、事实观察、排序和修复是否一致。不能只看退出码或最终分数。

## 9. 小规模对照实验的固定方案

### 9.1 先确定实体与共同事实清单

10～20 个真实实体，先覆盖人物／建筑或地点／作品／其他非人物对象，以及高关系度、入边、数量／时间限定和视觉证据稀疏情况。先定清单再运行，不能只挑新版表现好的实体。

MKG-W 图谱输入固定 `train`。同实体按缺失任务遮蔽目标模态；valid／test 三元组不得放进生成提示词。若下游任务另有实体隔离切分，还要冻结对应 entity split，不能把当前常规 train split 直接称为已经满足 inductive 协议。

共同事实 panel 从合法输入证据独立整理，**不由某个方法自己选出的事实集合定义**。这样才能发现旧入口、rerank、Stage 2 各自漏掉了什么。

可遮蔽原本有目标模态的实体做配对检查；隐藏原图／描述只能用于评测，不能回流到建模、提示词或 Reflect。真正原本缺图的实体没有真实参考图，需明确依赖证据一致性与身份判断，不能虚构 GT 图像指标。

### 9.2 图片三组，用完整名称记录 arm

| arm | 事实处理与生成 | 回答的问题 |
| --- | --- | --- |
| `old_pipeline` | 固定旧提交的原流程，显式图片预算 3 | 整体旧版参照 |
| `new_shared_same_prompt` | 新建模／rerank／Stage 2，一份精选事实；同一提示词调用 SDXL 三个种子 | 共享事实方案本身的表现 |
| `new_shared_multiview` | 与上一组完全相同的 Stage 1／2 事实和选择；协调三个提示词、相同三个种子 | 多视角是否比同提示词换种子更有用 |

**`new_shared_same_prompt` 不是当前主 CLI 已有开关，需要编写小型对照 harness。** 直接把主线 candidate-budget 设成 3 不会得到这一组；复制相同提示词给当前共享生成器还可能触发去重。该 ablation 要在冻结提示词后直接对同一提示词执行三次受控 SDXL 调用，保留种子与调用记录，不修改生产去重规则来伪装现成功能。

后两组共用同一份已保存 Stage 1／2 产物并记录哈希，同一 SDXL 参数及种子列表，才可以把差异归因于提示词协调。先生成的那个提示词可同时作为 same-prompt 组基准，选择规则在运行前固定，不能事后挑最好的提示词给某一组。

旧版与新版整体对照固定相同原始数据、entity keys、模型、预算和共同 panel；旧版 32 邻居截断等就是待比较的方法差异，不把新版事实偷偷补给旧版。完整三组结果不能直接当作对每个单独模块的因果证明；如需定位收益，再在独立实验中一次只改一个模块。

文本先比较旧／新各一条；无需为了图片三组结构人为增加文本候选。

### 9.3 预算、模型和修复如何控制

- 冻结 Qwen、reranker、SDXL 权重来源／版本；保存依赖、配置、实际提示词版本、温度、token 上限、种子、图片尺寸、采样步数和 guidance。
- 初始图片每组同为 3 次尝试，不给失败更多无限补跑。记录请求数、成功数、跳过数、真实图片数、重试数、token、耗时、峰值显存；不能只展示最好一张。
- 多视角会增加提示词协调成本，单提示词组三种子不应伪造相同 LLM 调用数；报告真实成本。
- 先保存和比较原始 Stage 3；Stage 4／Reflect 的效果单独报告。若把修复结果并入主结果，需要双方相同额外预算和协议，不能新版本多修复几轮。
- 原 384 与任何增大预算的诊断轨道分开，不覆盖失败；修复版与原提交也分开命名。
- 不混用替代 API 模型与原生 Qwen 的结果，不把 API reranker 视为本地 reranker 已复现。

### 9.4 评什么、怎样避免自评偏差

| 层次 | 主要检查 |
| --- | --- |
| 证据目录 | 输入合法一跳事实是否完整保存，subject／object／direction 是否保真 |
| Rerank | 共同关键事实在 top-k 中的保留率、漏掉的事实、关系／模态偏向和耗时 |
| Stage 2 | 精选后覆盖、冗余、保留的重要语义和方向，不用“选得少”直接代表好 |
| 文本 | 共同事实覆盖、错误、无依据断言、限定保留、自然性与格式 |
| 实际图片 | 可见属性一致性、无依据细节、身份可辨性、画质；非视觉关系不能一律按“没画出来＝错误” |
| 多候选集合 | 有证据的互补性、冗余、是否引入身份漂移；展示全部候选 |
| 系统 | 完成率、失败分布、重试、时间、显存、API／GPU 成本 |

真实图像必须看像素，不能只给提示词打分。若开展候选实体识别／检索，固定干扰实体与协议，不能用“看起来像一座塔”代替“识别对具体实体”。无真实外观证据的概念图单独标识。

自动 judge 只作辅助：候选匿名、位置交换、同一共同 panel，严格校验条目完整性，并复核明显错误。格式不合格／自相矛盾的响应单列，不补成满分。不能直接拿两种方法各自的生产 Stage 4 分数跨方法判断质量，因为参照事实和模型偏好可能不同。

统计按实体汇总；同一实体的三张图和两次位置交换不是更多独立样本。样本小，报告逐实体结果和不确定性，不做夸张显著性结论。

下游 filtered MRR／Hits 目前未跑。若继续做，需要固定 KGC 模型、训练预算、split、候选范围和过滤规则，另列实验；内容质量提升不自动等于补全指标提升。

## 10. 批处理与恢复的现有坑

- `scripts/run_mkgw_image_completion_4580.sh` 面向旧的固定 4580 清单，不是 10～20 个实体的通用试验脚本；默认数据目录还有 `datasets/MKG-W bak`，不要直接照抄。
- 该入口支持准备、前台、恢复与失败重试，使用 Linux 文件锁及独立 job 快照。
- `--resume` 读的是 **job 保存的源码／配置快照**。拉取新 Git 后继续旧 job，不代表旧 job 自动切成新方法。
- 已有大任务不重启、不覆盖，不混写新旧阶段产物。新方法开新 run/job，明确实际 snapshot commit。
- 现有主 CLI 会保留实体失败汇总，但还没有解决第 6 节的候选级失败隔离。不要把实体级 continue 当作候选级容错验收。
- 冒烟前不启动 4580 全量任务；先确认权重、显存、judge、路径和一次完整链路。

## 11. 完成顺序与最终交付

建议顺序：

1. 只读确认现场、已有任务、代码版本、数据与模型；建立隔离目录和环境记录。
2. 运行相关离线测试，复现两个已知故障，再做最小修复并记录新提交。
3. doctor＋一条真实补文本＋一条真实补图片；人工核对完整阶段链。
4. 冻结 10～20 个实体、共同事实 panel、全部参数和三组协议；编写缺少的 same-prompt 对照 harness。
5. 跑真实本地模型小实验，保留失败、图片和成本；逐实体并排审查。
6. 根据结果决定是否扩大。只有运行稳定、来源绑定和评价协议都成立，才考虑批量；不能因为文字结果更长就直接全量。

最终至少提交／交付：

- 修复代码与有意义的回归测试；记录基点和最终 commit，不混入模型／数据／密钥。
- 环境、资源路径、模型与数据指纹、GPU 掩码及峰值显存记录。
- 冻结的实体清单、共同 panel、实验配置／种子、可复跑的命令或脚本。
- 每组全部阶段产物、原始调用、实际图片、失败、重试与耗时；原始结果和 Reflect 结果分开。
- 一份并排报告：先给完成率和事实错误，再给覆盖、身份、质量与集合互补性；明确实测／未测。
- 若没有改善或多视角仅改变措辞，直接给出证据和建议，不将失败案例排除后只报成功样本。

## 12. 最重要的状态结论

**已经完成：** 新流程的设计讨论与实现、专项和同环境回归对照、研发快照上传；四个合成案例的真实 API Stage 0～3 测试和原文审计。

**尚未完成：** 新发现的截断错误分类与候选隔离修复、集群部署验收、本地 Qwen／本地 reranker／SDXL 的真实端到端质量对照、same-prompt ablation、生产 Stage 4／Reflect 效果校准、多视角的实际收益、真实 MKG 大样本与下游 KGC 指标。
