# 千问 7B、GPT-4o-mini 与 Reranker 运行环境

2026-10-28 按用户要求，Stage 2 也切换为本地千问。CLI、Python API 和
4 / 10 / 50 实体启动脚本的 Stage 1 / 2 / 3 默认模型均为
`Qwen/Qwen2-VL-7B-Instruct`；Stage 4 的 GPT 评分使用 `gpt-4o-mini`。
配置中的 `inference.prompt_profile: qwen7b_original_v1` 保留历史模板选择。
新任务的 Stage 1 使用下述三元组协议，Stage 3 对新协议使用结构化事实输入；
历史知识协议仍可使用原始模板。实际模型看调用日志的 `model_name`；
历史模板说明见 `docs/qwen7b-original-prompts.zh.md`。
显式 GPT 对照配置仍采用 `gpt4omini_stage03_v2`，说明见 `docs/gpt4omini-prompts-v2.zh.md`。

| 阶段 | 当前模型与职责 |
| --- | --- |
| Stage 0 | 程序保留完整一跳目录，新任务不调用 LLM 筛掉关系 |
| Stage 1 | 图谱、文本及媒体观察统一成三元组；本地 Qwen3-Reranker-4B 按生成和身份相关性粗筛 |
| Stage 2 | Qwen2-VL-7B-Instruct 选择有上限的共享事实集合 |
| Stage 3 | Qwen2-VL-7B-Instruct 生成文本或图像提示词；SDXL 生成图像 |
| Stage 4 | GPT-4o-mini 与 CLIP、BLIP 等评分器参与候选排序 |

Stage 3 使用 `configs/qwen7b-mainline.yaml`，Stage 1 知识建模使用
`configs/qwen7b-stage1.yaml`，Stage 2 使用 `configs/qwen7b-stage2.yaml`；
三者默认连接 `http://127.0.0.1:12345/v1`。
主配置保留 4096 token 的语言模型输出预算及 384 token 的图像提示词预算。
Stage 4 GPT 评分器使用 `configs/openai-gpt4o-mini-judge.yaml`。
Stage 2 可通过 `--stage2-model-config` 或脚本的 `STAGE2_MODEL_CONFIG` 显式切回 GPT 对照。

`qwen7b-mainline.yaml`、`qwen7b-stage1.yaml` 和环境变量版 `qwen-local.yaml`
均显式配置历史千问 profile。Stage 1 的新默认协议为 `grounded_triples_v2`：
程序保留完整图谱目录，模型从已有文本、图片观察、视频帧／片段、音频转写或声音观察抽取三元组，
并返回简要 `entity_profile` 和 `uncertain_aspects`。不再要求逐条生成 claim，
也不单独输出 `key_attributes` 或 `text_visual_profile`。

每条事实统一表示为 `subject / relation / object / evidence_refs`，`relation` 即谓词。
例如 `(Museum, architect, Jane)` 与 `(Jane, designed, Museum)` 各自保留主谓宾方向；
不会为了让目标实体位于主语而交换两端。只有主谓宾完全相同的记录才合并来源，
不把不同谓词自动视为同义。图谱沿用 `rf:` ID，新增模态事实由程序分配 `mf:` ID。
没有可用观察内容时，模型返回空 `extracted_triples`，图谱事实仍然保留。

已有纯文本可用 `text`，图片可用 `image_summary`，视频可用 `video_summary`／`video_transcript`，
音频可用 `audio_transcript`／`audio_summary`。分段观察使用 `existing_modalities.observations`：

```json
{
  "evidence_id": "existing:video:segment:1",
  "modality": "video",
  "content": "Museum entrance doors open inward.",
  "locator": {"asset": "clip.mp4", "start_seconds": 2.0, "end_seconds": 3.0}
}
```

已有原图由 `QwenLocalSemanticStagePlanner` 自动执行两步视觉建模，均使用 Stage 1 模型配置：

1. 看原图生成自然语言描述。此步不传入 KG、目标描述或 ground truth，单独记录为 `stage1_image_description`。
2. 模型同时读取这份描述和原图，修正描述错误、补充遗漏的可见细节，再返回通常的三元组提取结果。

部件属性和空间关系以目标的组成／布局事实表达，例如 `(Scene, has layout, blue circle left of red square)`，
避免直接输出与目标标签无连接的部件三元组，也避免为了满足目标锚定而丢掉这些细节。

两次调用及格式修正都附带同一份冻结的原始 PNG／JPEG 字节，不能退化成第二步只读摘要。
视觉事实必须引用 `existing:image:original`；描述支持该事实时也引用 `existing:image_summary`，
纠正描述中的错误时只引用原图。引用检查不等于程序验证了视觉判断正确。
自然语言描述、图片 SHA-256、源文件位置和调用引用保存为 `image_modeling`；
原始观察输入另存为 `source_inputs`，生成描述不会反写数据集或伪装成数据集原有摘要。
模型调用日志只保存图片元数据，不写入 base64。描述生成、提取、网络重试和业务修正共享 Stage 1 的 180 秒预算；
描述为空、截断、读取失败或视觉调用失败时停止，不悄悄改用无图抽取。

有原图时会重新生成本次描述；只有现成摘要而没有原图时，仍可按已有观察文本接入，
但不会标记为完成了两步看图。
仅有尚未处理的媒体路径时记录 `unprocessed_modalities`，不把路径当作事实。
图片属于待补目标时，两步视觉调用均跳过。目标模态及显式标明由它派生的观察仍被屏蔽。
CLI 生成目标仍为 `text`／`image`。

### 文本抽取与请求超时

单一原始文本来源使用 `source_bound_text_triples_v1` 传输协议：只发送原文和目标实体，
模型只返回主语、关系、宾语；程序校验格式与精确目标标签后，绑定本次唯一的原始文本来源。
请求中不提供图谱事实和图谱证据 ID，模型也不负责生成证据 ID；完整图谱事实仍由程序保留并合并。
模型若混入证据字段会被拒绝，不会将任意错误引用改为文本来源。不能支持的内容仍应省略或记录为不确定。
本地文本抽取和局部修正不使用旧版 vLLM 的约束解码，避免其生成阻塞；Python 校验保持启用。
长文本按约 600 字符的连续短段依次抽取，后续段附带原文开头的身份上下文。
每个原文字符均被覆盖；所有段共用 Stage 1 总时间预算，结果按三元组去重合并回完整来源目录。
错误条目通过 `stage1_source_patch_v1` 局部重抽，已通过的条目保持原样。
修正结果仍经过完整校验，不能通过直接替换错误证据 ID 来绕过来源约束。

`stage1_source_patch_v2` 兼容省略未请求的空 `fields`／`facts` 容器，
并识别单一完整 JSON 代码块及明确的 `response`／`response_template` 外壳；重复 JSON 键、多份响应、
额外修正路径及有内容的未知字段仍被拒绝。部分修正只更新实际返回的请求项，
随后再次完整校验，已修正项保持原样，下一次只请求仍错误或缺失的条目。
每次最多请求修正 3 个字段，按初始错误数量有界分组，格式错误重试包含最新校验反馈。
格式始终错误时只拆分失败原文段重新抽取，并保留该段中已完整通过校验的三元组。

文本抽取报告 `finish_reason=length`、反复输出同一三元组或返回不可解析 JSON 时，
不连续重发相同完整抽取请求，而是仅将失败原文段拆为约 280 字符或更短的子段。
最多再拆两层；每次保留绝对原文字符范围与原始身份上下文，已成功段不重新生成。
各子段均需输出完整 JSON 并通过原有来源／标签校验，最终合并回完整证据目录。
不会用补括号的方式接受残缺三元组，所有拆段、修正和网络请求仍共用原来 180 秒预算。
此项修改保持 Stage 1 的采样温度和 Stage 3 的图像呈现规则不变。

本地 Stage 1 请求携带独立请求 ID 与服务端时限。千问服务需加载
`mmkgc.adapters.request_deadline.RequestDeadlineMiddleware`（启动脚本已配置），
到期时对对应 vLLM 引擎请求执行 abort，再返回 504；客户端不会立即重复提交超时请求。
服务取消失败或确实失去响应时，批处理的有界服务重启仍作为恢复措施。

### 多张原图

MKG-W 优先读取 `image_alignment/qid2images.json`、`entity2images.json`、`entity_id2images.json`
中的图片列表，按实体 URI、短 ID 或数字 ID 查找。没有索引条目时，扫描 `images_by_id/<数字 ID>/`、
`images_by_id/<实体 ID>/` 及旧版 `images/<实体名称>/`，保留所有发现的原图并按路径去重。
`EntityRecord.image_paths` 保存完整列表，`existing_modalities.image` 兼容性地保留首图。
目标是缺失图片时，加载器屏蔽整份图片列表，Stage 1 不读取或附加这些目标原图。

多图执行“逐图描述 → 带原图联合抽取 → 统一合并”。3 张图片通常为 3 次描述请求加 1 次抽取请求，
不含网络重试和格式修正。每张图片使用独立的 `existing:image:001:original`、
`existing:image:001:description` 等引用。引用某份描述的三元组必须同时引用该描述对应的原图；
可以只引用原图来纠正描述错误，不能用另一张图的 ID 代替。单图继续使用旧 ID 和旧产物结构。

原图在首次模型调用前全部读取并冻结。每次描述只附加对应图片，联合抽取及其格式修正保留同一份原图字节。
`image_modeling.images` 分别保存描述、图片指纹、路径和调用引用，`extraction_batches` 记录批次。
全部事实统一参与后续相关性排序和 Stage 2 共享事实选择，不为图片单独分配生成配额。

`inference.stage1_image_batch_size` 默认 4，可在 `configs/qwen7b-stage1.yaml` 或
`configs/media-stage1-api.yaml` 中调整。超过批大小时分别抽取，再按完全相同的主谓宾合并来源；
各批只能引用本批实际附加的图片，其他批的图片和描述不会进入本批抽取提示词。
联合视觉判断发生在批内，程序不声称完成了跨批语义冲突识别或同义关系合并。
模型应保留不同视角的限定并记录不确定项，程序保留这些结果，不通过多图投票决定事实真假。

启动脚本通过 `QWEN_MAX_IMAGES_PER_PROMPT` 设置 vLLM 的 `--limit-mm-per-prompt image=4`，默认 4。
服务端图片上限不能低于抽取批大小；已有服务需重启后才应用该设置。图片分辨率、文字及输出仍共享模型上下文，
若请求超出服务上下文，应减小批大小或调整服务配置；程序不会偷偷丢弃图片来满足上限。
描述、所有批次、网络重试及格式修正仍共享 Stage 1 的总时限（默认 180 秒）。
任何声明的原图读取失败、描述失败或抽取失败都会报告错误，不把部分图片的结果伪装成完整成功。
本次验证采用模拟模型响应和真实小图片／音频文件，未调用真实千问、GPT 或图像生成模型。

视频／音频也自动执行“原媒体生成描述 → 描述加原媒体证据提取三元组”：

- 视频 `video` 接受本地 MP4／MOV／WebM。ffprobe 检查时长和流，ffmpeg 从完整时间轴约每秒取一帧，
  画面最大宽度 768，并提取完整第一音轨为单声道 16 kHz WAV。两步使用完全相同的冻结画面和音轨。
  每帧附带实际时间位置。采样间隙的短暂事件、其他音轨及空间声学信息可能遗漏，不能视作全帧读取。
- 音频 `audio` 接受本地 WAV／MP3，两步都发送同一份冻结的原文件。第一步同时要求忠实转写和
  非语音声音描述；第二步重新听原音频，区分听到的声音、推测的声源与引述的说话内容。
- 每个已有媒体各生成一份描述，最后一次联合提取事实；同时有图、视频、音频时共 4 次正常调用。
  不同来源仍使用同一套 `subject / relation / object / evidence_refs`，需要时在宾语中保留近似时间段。
- 视频／音频事实分别必须引用 `existing:video:original`／`existing:audio:original`。
  来源文件、SHA-256、实际帧位置、音轨指纹和描述调用保存为 `media_modeling`；原始观察另存 `source_inputs`。
  保存恢复时检查原始媒体文件名与字节指纹，防止同路径换内容后复用旧结果。
- 当前单文件上限为 7 MiB、时长上限为 120 秒，解码后的帧和音轨合计也不得超过 7 MiB。
  超限、缺少解码工具或调用失败时停止，不静默截取前段或降级为只读摘要。尚未实现长媒体分段批处理。
  所有媒体加载、描述、联合提取和修正共享原有 Stage 1 总预算。

模型必须支持实际发送的图片和音频。默认 Qwen2-VL 可看图片／视频采样帧，不能听音轨；
有音频时必须配置支持音频的 Stage 1 模型。示例 `configs/media-stage1-api.yaml` 读取
`TAOR_STAGE1_MODEL`、`OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，通过已有 `--stage1-model-config` 接入。
`lmm.audio_input_type` 默认 `input_audio`，其 `data` 字段传入纯 base64 音频；
`audio_url` 分支使用带 MIME 前缀的 data URL。编码格式依据
[OpenAI 官方音频输入示例](https://developers.openai.com/api/docs/guides/audio-chat-completions)。
SiliconFlow／vLLM-Omni 一类接口可按其协议改为 `audio_url`，
协议参考 [SiliconFlow 多模态输入](https://docs.siliconflow.cn/docs/userguide/capabilities/multimodal-vision)
和 [Qwen-Omni 输入示例](https://www.alibabacloud.com/help/en/model-studio/qwen-omni)。
不能只凭 `/models` 列表或 HTTP 200 判断媒体通道可用：本次探测遇到原生 `video_url` 被忽略的情况，
因此使用明确的时间标记画面和音轨输入。

2026-11-02 的真实视频／音频两步小例、失败记录和音频感知错误，见
[媒体两步验证](evaluation/2026-11-02-media-two-step.zh.md)。链路与契约通过不代表事实准确。

两步视觉 API 小例保存在 `outputs/explanations/image_two_step_case/trace.json`，原图为同目录的 `source.png`。
本次实际响应模型为 `gpt-5.6-luna-2026-07-09`：2 次调用使用同一图片指纹，先生成描述，
再提取 7 条事实，保留对象、左右位置、间隔、相对大小和背景。该例是合成图形的接口验证，
未运行本地 Qwen 权重，也不是真实图片数据集的视觉质量或召回率测量。

文本抽取必须引用真实存在的已有模态证据，不能借图谱引用伪装为文本抽取，
也不会把整段文本自动挂到每条图谱事实上。来源检查验证引用边界，不能保证模型抽取语义正确。
新阶段产物的事实列表保存为 `data.triples`；内部兼容接口仍使用 `relation_facts`，
Stage 2、生成和评分消费同一份主谓宾及来源。历史 `qwen_grounded_knowledge_v1`、`grounded_triples_v1` 产物仍可恢复。

新任务的 reranker query 只包含动态实体标签与目标模态；共享指令要求保留实体语义身份、定义性属性
和重要关系，同时考虑目标模态表达，兼具两者价值的事实优先。Query 不再加入模型归纳的实体类型。
document 保留候选三元组与方向，不包含来源模态或原证据包，也不把候选断言重复写入 query。
来源模态仍保存在事实中用于追溯。指令版本为 `entity_semantics_modality_relevance_v2`。
所有事实都先评分，再选本轮 Stage 2 输入；
每次最多 8 对本地推理，默认本轮视图 64 条，可用 `--relevance-fact-budget` 调整。
关系的相关性由其最高分事实汇总，仅作为概览，不增加一次独立关系事实性检查。
分数与名次记录为 `relevance_score`／`relevance_rank`，完整事实和 `relevance_selection` 一起保存。
不使用真假阈值，粗筛未展示的事实仍留在目录。旧 `score_knowledge_evidence` 仅用于显式历史协议兼容。

格式、空字段或错误来源引用会触发最多两轮完整回答修正，修正输入只有原始源数据和错误列表；
初次调用与修正共享默认 180 秒预算。日志记录 `grounded_triples_v2`、
`stage1_triple_source_repair_v1` 和每轮 `validation_attempts`。

2026-11-01 早期 `grounded_triples_v1` Museum 案例：通过用户提供的 API 配置，实际响应模型为
`gpt-5.6-luna-2026-07-09`。3 条图谱事实保留，已有文本抽出 3 条事实，Stage 2 选择其 ID，
Stage 3 生成 3 个图像提示词。补充候选事实范围说明后，复查未发现提示词混用未选的立面／入口细节。
此例未运行 Stage 0、reranker、SDXL 或 Stage 4，也不能代表本地 Qwen 权重的运行结果。
原始请求与响应保存在本地 `outputs/explanations/unified_triples_case/trace.json`，
初次运行记录另存为 `trace-initial.json`；来源引用检查和提示约束均不构成抽取语义正确性的保证。

随后 `grounded_triples_v2` 的 API 小例保存在 `outputs/explanations/task_relevance_case/trace.json`。
实际 LLM 为 `gpt-5.6-luna-2026-07-09`，reranker 为 API 版 `Qwen/Qwen3-Reranker-4B`；
自定义相关性说明拼接进 API query，本地模板等价性未验证。合成图谱和预设的文本、视频／音频观察
合并为 14 条三元组，全部参与图像／文本两个任务的相关性排序，并完成图像任务的 3 个候选提示词。
图像任务的红砖立面约 0.962、文件名约 0.0014；文本任务的类型和收藏主题排在前面，
网站仍排第 3，说明排序尚需样本评估，不能把单例当成高召回证明。此例没有运行原始媒体感知、SDXL 或 Stage 4。

Stage 2 默认对网络超时、断连及临时 HTTP 故障额外重试 1 次，与模型契约修正分别计数，
阶段总预算为 300 秒。修正使用原始输入与结构化错误，不追加错误模型响应。
本地千问 Stage 3 的重复／截断修正使用独立短请求、lm-format-enforcer
JSON 约束和 1.1 重复惩罚，最多额外请求 2 次。新三元组任务使用结构化生成模板，SDXL 参数保持原样。
实现与验证见 `docs/evaluation/2026-10-28-failed8-recovery.zh.md`。

共享事实图像任务使用 `grounded_image_focus_v4` 与 `entity_semantics_image_v7`：
保留同一组 Stage 2 事实，先生成第一条描述，再让后续描述参考已接受的完整提示词与上一条结果，
自主选择不同的属性、场景细节、视角、构图或渲染重点。没有固定的中景／近景／广景顺序或主体位置。
每条候选分成“重点计划 → 描述 → 压缩”三次模型请求。模型先输出 `visual_focus`、
`supporting_fact_ids`、含五个呈现维度的 `rendering` 对象和 `difference_from_previous`。
程序检查引用事实存在，并对照所有已接受计划，允许“新视觉重点”或“有意义的呈现变化”任一成立，
不要求同时改变重点和拍法，也不强制改变任何固定数量的呈现维度。
例如，从红砖纹理转向拱门结构时，允许景别、视角、主体位置、风格和光照全部保持相同。
前序描述已经顺带提过的特征仍可成为下一张的主要重点，无须新增事实才能获得候选。
提示词优先引导有助于实体识别的可见细节，避免为凑差异强行改变风格、光照或虚构外观；
从“百老汇经历”换成“好莱坞经历”本身不能证明画面发生变化。
只在解释中声称不同、用常见同义词替换原重点，不能作为差异依据。
描述通常 50–100 词，最多 120 词／1000 字符；
随后保留主体与所选视觉重点，压缩成约 20–70 个英文词的 SDXL 提示词，最多 70 词／650 字符。
证据较少时可少于 20 词，不为凑长度添加事实。描述和压缩请求保留相同的 Stage 2 事实；
压缩请求携带候选描述与来源 ID，不重复传入整段传记和二跳路径；完整来源仍保存在描述请求及阶段产物中。
历史协议／直接单次调用继续兼容旧的 80 词上限；长度修正仍使用更短的 55 词／400 字符预算。
前序模型结果只用于避免重复、选择新重点，不作为事实证据；对象关系、数量和属性保持原样，
风格、光照与氛围可作为渲染选择。描述和压缩后的提示词均须表达计划中的至少一项新重点或
呈现变化，避免压缩抹掉差异。计划、描述、压缩请求及各步采样参数均写入模型调用记录。
图像提示词独立使用 `image_prompt_temperature: 0.3`、`image_prompt_top_p: 0.9`，
每个候选及改写记录独立、可复现的采样种子，Stage 1／2 与文本采样参数保持独立。
目标候选数与请求预算分开：3 个目标通常请求 9 次，最多请求 18 次；计划修正、描述修正、
压缩修正和重复改写同样消耗共享预算，每个候选最多额外改写两次。剩余预算不足三次时停止新候选。
计划或差异核验持续失败时记录拒绝原因并跳过该候选，后续候选只参考已经接受的结果。
完全相同、单纯改标点、编号或重排词语，以及呈现特征相同的高度相近提示词会被拦截。
去重与差异落实检查使用规范化词语、常见同义词和呈现维度的启发式比较，
共同处理常见居中同义表述和“不是近景”等否定标签，避免将其误当作新的呈现方式。
这些检查不能保证识别所有语义重复、判断重点是否可见或证明图像实际不同；
事实 ID 合法也不等于证明计划或图像事实一致。可见性和实体用途由生成提示词约束，
实际图片的质量与一致性仍需后续评估和实验检验。
有限改写后仍不足目标图片数时，从已接受的 prompt 中轮流复用，补齐缺少的图片候选。
复用不再调用语言模型，不增加 prompt 请求预算，也不进入后续候选的已接受 prompt 历史。
每个图片候选保留自己的 `image_001` 等编号，SDXL 根据基础种子和编号获得不同、可复现的种子。
原始候选记录 `image_candidate_kind=distinct_prompt`；补齐候选记录 `seed_variant`、
`prompt_source_candidate_id`、`seed_variant_index` 和原因；这些信息保存在阶段产物、图片元数据和调用日志中。
`image_prompt_search` 分别记录 `unique_prompt_count`、`seed_variant_count`、`generated_image_count`、
`prompt_shortfall`（独立 prompt 不足数量）和 `shortfall`（图片候选不足数量）。
仅生成提示词时也保留计划好的种子变体，实际生成图片数仍为零。没有任何已接受 prompt 时不会补齐。
种子变体沿用同一事实和 prompt，不计作新的视觉重点，也不能保证图片视觉差异或事实正确。
新协议的格式修正保留 Python 校验，并避开已观察到阻塞的旧版约束解码路径；历史协议仍兼容原路径。

超长或截断的图像提示词使用 `qwen_image_prompt_source_rebuild_v2` 修正：保留原选中事实、
完整来源和构图要求，改用简短输出指令，要求实际画面描述不超过 55 词／400 字符，
修正请求的两份审计列表为空，避免枚举片名耗尽输出预算。Python 仍执行通用的
80 词／650 字符上限及重复短语校验，拒绝照抄响应占位文字。修正仍消耗每实体共享请求预算。
若服务报告 `finish_reason=length`，且第一个 `positive_prompt` 字符串完整、通过长度和重复校验，
仅尾部审计字符串列表被截断，则原样保留提示词并将两份审计列表置空，记录
`audit_recovery.audit_complete=false` 及原始响应。提示词本身不完整、超长、字段异常或其他 JSON
语法错误仍不能恢复；不通过补写提示词、截断文字或放宽长度限制掩盖失败。这不是事实正确性核验。

Stage 2 新任务使用共享事实选择协议，旧候选组合、角色／槽位定向修正保留用于历史协议兼容。
当前字段、来源绑定与新旧任务恢复方式见 [Stage 2 精简计划](stage2-candidate-plan.zh.md)。

Stage 3 的文本生成和图像提示词请求支持并发。在主模型配置中设置：

```yaml
inference:
  stage3_max_concurrency: 3
```

`qwen7b-mainline.yaml` 并发上限默认为 3，适用于历史独立候选及定向修复。
新共享事实流程的首轮提示词按顺序生成，使后续请求能读取已生成提示词并协调表达重点。
保留的 `gpt4omini-mainline.yaml` 为 5；未指定该字段的配置默认串行（1）。
每个实体按候选预算准备文本或图像提示词，再按原候选顺序汇总，
SDXL 在调用线程逐张生成，避免多个线程同时操作同一 GPU 模型。
初次生成和 Stage 4 反馈后的重新生成都采用这一调度。
调用缓存及元数据按线程隔离，候选 ID、提示词响应和日志保持对应；请求失败时
保留该候选的调用记录并抛出异常，不用其他候选结果代替。CLI 的 `config.json`
记录实际 `stage3_max_concurrency`。这项配置只作用于 Stage 3，实体和其他阶段
仍按原来的依赖顺序执行；SDXL 的图像尺寸、步数、种子和候选预算不变。

Stage 2 千问与 GPT 的新任务均记录 `shared_fact_selection_v1`。模型只返回 `status` 和
`selected_fact_ids`，选择一个最多 `--max-selected-facts` 条事实的集合，允许少于上限。
生成预算与事实数量解耦；文本默认一条完整描述，图片运行预算默认 3，与正式批处理入口一致，可单独配置。
Stage 3 依据共享事实组织构图和叙述，不要求 Stage 2 focus，也不套用旧视角模板。
保存恢复兼容旧 `candidate_fact_plan_v1` / `v2` 及角色／槽位协议，保留原图片字节。
程序检查事实 ID、重复、数量上限和来源证据，并绑定共同事实供生成与评分使用。
新图片流程允许少于预算的候选，完全相同的提示词跳过 SDXL；尚无语义去重或效果驱动早停。

GPT 使用简单 JSON Schema；本地千问首次请求保持普通生成，修正时使用小型 JSON 约束。
仍最多修正两轮，重复无效响应提前停止，网络重试和业务修正共享阶段总预算。
日志的 `validation_attempts` 保留每轮响应与错误，`contract_repair_stopped_reason` 记录停止原因。

视觉评分向 GPT 发送大于 512 KiB 的 RGB PNG 时，使用质量 90 的 JPEG 传输，
保持原始像素分辨率，以减小请求体；小图、已有 JPEG 和含透明度的图片保持原格式。
生成的 PNG 文件及 CLIP、BLIP 的本地图像输入不受传输编码影响。

Stage 4 的独立 GPT 评分请求支持有限并发。在使用的 `stage4` YAML 中设置：

```yaml
gpt4omini_judge:
  max_concurrency: 10
```

主配置和 `stage4-model-fusion-gpu1-visible.yaml` 已设为 10；旧配置未指定该字段时
默认串行（1）。并发仅用于同一实体的独立候选评分，本地模型在主线程执行。
实际并发取配置上限和当前待评分候选数的较小值；每实体 5 张候选图时，最多同时
执行 5 个评分请求。该设置不增加候选图数量，也不启用多个实体同时运行。
模型响应按原候选顺序合并，调用日志记录实际墙钟耗时。线程中的 HTTP 请求通过
异步总超时限制整个请求；超时、429 和临时服务错误按照 `max_retries` 有界重试，
数值型 `Retry-After` 最多等待 60 秒。并发所需 `httpx` 已列入运行环境依赖文件。

Stage 0 → Stage 1 → Stage 2 的数据依赖保持顺序。并发提高独立请求的吞吐，
Stage 2 的候选事实引用和数量仍执行完整校验。上游中转接口的实际并发
能力需要通过小批运行验证；本次修改没有恢复已由用户停止的 50 实体任务。

主流程在 `kb` 环境使用 Transformers 4.51.3 加载 Qwen3-Reranker-4B。

首次安装或重建环境，在项目根目录执行：

```bash
conda activate kb
python -m pip install -r requirements.txt
```

运行入口自动读取项目 `.env`，保留调用者已设置的环境变量。
`.env` 中的 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 用于 GPT 服务，
两个 GPT 配置都明确指定 `gpt-4o-mini`。

运行默认 CLI、Python API 或 `scripts/run_mkgw_image_test_*.sh` 前，需要启动本地
Qwen7B 服务：`bash scripts/run_qwen7b.sh`。脚本默认将千问放在物理 GPU 0，
4 / 10 / 50 实体流水线通过 `CUDA_VISIBLE_DEVICES=1` 使用物理 GPU 1，避免将
常驻千问服务与 Reranker、SDXL 和本地评分模型放在同一张卡。
`MODEL_CONFIG`、`STAGE1_MODEL_CONFIG`、`STAGE2_MODEL_CONFIG` 可以分别覆盖阶段配置。
如服务地址不同，需要修改两个千问配置或分别指定替代配置。

保留 GPT 对照运行方式：

```bash
MODEL_CONFIG="$PWD/configs/gpt4omini-mainline.yaml" \
STAGE1_MODEL_CONFIG="$PWD/configs/gpt4omini-mainline.yaml" \
bash scripts/run_mkgw_image_test_4.sh
```

直接 CLI 的对应覆盖参数为 `--model-config` 和 `--stage1-model-config`。
已启动的 16 实体 GPT 测试使用输出目录中的配置快照，继续按该轮原配置执行；
上述默认值变更作用于后续新进程。

批量测试可向 `python -m mmkgc.run` 追加 `--continue-on-error`：单个实体失败时
写入 `failures.json` 和 `metrics_summary.json`，随后继续处理其余实体；只要有
失败，整个任务最终返回非零退出码。未指定时仍在首次失败后停止。

Qwen7B 服务环境保留在 `.venv-qwen7b`，与流水线的 `kb` 环境分离。
现有 vLLM 0.6.4.post1 的 Qwen2-VL 实现依赖 Transformers 4.46.1，
可通过以下命令重建服务环境：

```bash
conda activate kb
python -m venv --system-site-packages .venv-qwen7b
.venv-qwen7b/bin/python -m pip install transformers==4.46.1 tokenizers==0.20.1
bash scripts/run_qwen7b.sh
```

`QWEN_PYTHON` 可覆盖 Qwen 服务所用的 Python 路径。

当前依赖已统一到项目根目录的 `requirements.txt`；服务环境共享基环境的软件包，
仅覆盖上述两个版本。完整的从零安装步骤见 [环境配置 README](../README_ENV.md)。
