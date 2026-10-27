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

文本提示词要求输出 1～2 句英文描述。生成角度只控制表达重点，不能增加事实；
内部字段名、事实 ID、任务说明不应出现在最终描述中。

图像提示词保留 `positive_prompt`、`used_text_cues`、`omitted_risky_details` 三个字段，
正向提示词建议 25～55 词，不为凑词数添加事实。
出生地等背景关系不能自动变成画面地点，导演和类型不能推导出演员脸或官方海报。
背景关系也不能通过“某地氛围”“受某地启发”等措辞转化为画面风格。
缺少外观证据时采用保守的类型级概念表现，并记录省略的具体细节。

新版模型适配器通过 `generate_with_context` 接收结构化输入；现有自定义适配器的 `generate` 接口保持兼容。
新版图像正向提示词不再追加程序抽取的关系片段，避免将非视觉关系在模型输出后重新加入画面。

模板位置：`mmkgc/stages/semantic.py` 和 `mmkgc/generation_prompts.py`。
