# GPT-4o-mini 提示词 v2

版本标识：`gpt4omini_stage03_v2`。阶段输出的业务字段保持兼容，模型调用日志记录提示词版本。

2026-10-27 按用户要求，默认 Stage 0 / 1 / 3 的模型与提示词均已恢复为千问版本。
原始模板通过 `qwen7b_original_v1` 启用，见 `docs/qwen7b-original-prompts.zh.md`。
本文件记录的 v2 仅供显式 GPT 配置对照使用；Stage 2 槽位方案和并发实现保留。

## Stage 0：高召回关系粗筛

独立说明高召回任务：保留实体身份、属性和目标模态可能需要的关系；不确定时保留，
明确无关或元数据噪声才删除。关系角色和候选数量分配由 Stage 2 决定。

移除“每个列表最多 3 项”的通用限制，输入中提供 `max_selected_relations`，
其值等于本次提供的候选数，沿用最多 24 个候选的输入边界。
程序验证输出关系与方向必须来自提供的候选，且不能重复。
没有候选时跳过模型调用；候选全部无关时允许返回空选择，不补造关系。

## Stage 1：完整知识建模

将外层任务改为完整建模，保留每个已有 `fact_id` 对应一个 claim 的约束。
提示词给出入边示例，明确 `direction=in` 不改变原始事实的主客体。
身份、属性、视觉线索及引用应有输入证据支持；未知信息进入 `uncertain_aspects`。
GPT 不输出可信度分数、`allowed_use` 或生成约束；证据评分继续由 Qwen3-Reranker-4B 完成。

## Stage 3：候选级结构化输入

模型输入分别组织以下字段：

- `task`：目标模态、生成角度和多样性方向。
- `source_facts`：当前候选的 primary 与显式关联 secondary 事实，保留事实 ID、方向及证据引用。
- `source_evidence`：相关证据，区分候选事实、身份消歧和 context-only 用途。
- `must_include` / `should_include` / `context_only`：生成要求和仅供消歧的背景。
- `identity_context`：有证据引用的实体身份。
- `supported_visual_details`：与当前候选证据关联的视觉属性。
- `candidate_visual_cues`：需结合原始证据核对的视觉提示，不直接视作已确认外观。
- `rendering_options`：可选择的构图、机位、光照、景深、表现风格及中性背景。
- `regeneration_feedback`：重生成修正要求，与事实列表分开。

文本提示词现使用 `entity_semantics_text_v4`：长度随所选事实调整，不限制固定句数，
覆盖有证据支持的不同事实，保留主客体、方向与限定条件，以自然的实体描述表达。
内部字段名、事实 ID、任务说明不应出现在最终描述中。新候选事实计划的接口见
`docs/stage2-candidate-plan.zh.md`，上述角色字段仅用于历史协议。

该文本模板参考 [Knowledge Bridger 的知识驱动生成实现](https://github.com/Guanzhou-Ke/Knowledge-Bridger/blob/main/src/knowledge_bridger/knowledge.py#L383)：
将属性与关系综合成连贯描述。当前任务以目标实体为中心，共享事实集合由 Stage 2 决定，
Stage 3 直接生成一条描述，自行组织语言。文本分支不再叠加图片规则与重复的三元组说明；
只有收到修复反馈时才附加修复指令。此调整尚未通过真实模型对照验证质量收益。

图像提示词现使用 `entity_semantics_image_v4`，保留 `positive_prompt`、`used_text_cues`、
`omitted_risky_details` 三个字段。参考 [Knowledge Bridger 的图像提示词生成](https://github.com/Guanzhou-Ke/Knowledge-Bridger/blob/main/src/knowledge_bridger/knowledge.py#L314)，
使用结构化属性与关系指导扩散模型。当前流程以目标实体为中心，Stage 2 选择一个共享事实集，
Stage 3 基于该集合按顺序组织多个图像提示词，后续调用读取已接受提示词以协调表达重点。
先前生成内容不是事实来源；多视角不能补造未知几何结构。按大小写与空白归一后的相同提示词
跳过 SDXL，语义近似尚未自动去重。文本默认一条完整描述，生成尝试预算与事实数量上限分别配置。

模型根据所选三元组保留可见属性和能直接表现的关系，保留参与者、角色、数量与空间关系。
非视觉关系用于实体识别和消歧，无须逐条画出；出生地、设计者等背景不能自动变成地点、人物或风格。
原始证据帮助理解所选事实，不能引入未选属性；构图、视角和光照不能改变事实属性。
缺少实体具体外观时允许保守的类型级概念表现，并记录未知细节；这不等于恢复了真实实体外观。

取消 25～55 词的建议区间，无最小长度；沿用现有程序校验的最多 80 词、650 字符上限。
这只是当前接口的长度限制，不等同于 SDXL 的 tokenizer 长度。正常请求不附加修复指令；
有观察反馈时保留原事实范围修复，结构化输入的格式修复也沿用同一模板、证据和候选事实。
格式修复沿用既有 JSON 约束的 400 字符上限。新图片版本记录在实际请求与调用日志中，
包含格式修复后的日志；旧千问模板仍用于历史协议。真实生成质量收益尚未验证。

新共享事实协议要求适配器通过 `generate_with_context` 接收结构化输入；旧协议仍兼容 `generate` 接口。
新版图像正向提示词不再追加程序抽取的关系片段，避免将非视觉关系在模型输出后重新加入画面。

模板位置：`mmkgc/stages/semantic.py` 和 `mmkgc/generation_prompts.py`。
