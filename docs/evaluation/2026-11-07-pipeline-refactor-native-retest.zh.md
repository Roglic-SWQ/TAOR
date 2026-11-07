# pipeline-refactor 单卡三例复测

本轮验证 Reflection 意见传递修复的真实模型闭环。执行提交为 `b62028719b763abf6c697fcdd118ab9ce451cc65`，其运行代码与 `b5210f7` 相同；合并提交保留远端 `pipeline-refactor` 历史并恢复最初两份交接文档。未创建新分支。

## 实际资源与范围

集群已有可用配置 `/medisco/linjiashi/clusterx/clusterx.yaml`，通过 `CLUSTERX_CFG_PATH` 读取。最初只检查默认 HOME 配置目录而报告缺失，是现场查找不完整；该路径早已记录在 `docs/wn18rr-runtime.zh.md`。不打印或复制配置中的凭据。

作业 `taor-reflect3-1107-r1` 分配一张 NVIDIA A100-SXM4-80GB，最终调度状态 `JobStatus.SUCCEEDED`，作业退出并释放资源。复用此前成功作业的镜像与挂载，未重启或停止旧任务。

- Qwen3.5-9B：真实图像描述、特征提议、文本与图片提示词生成、Reflection 修改。
- Qwen3-Reranker-4B：本地来源相关性排序。
- gpt-6-luna：来源准入／核验、共享事实选择、Stage 4 Judge 与提示词复查。
- SDXL base 1.0：真实 1024×1024 图片，20 steps、guidance 7.5。
- Stage 4：本地 SBERT 文本评分、通过门控后的 CLIP／BLIP 图片评分，与 Judge 融合。被语义门控拒绝的图片不继续计算 CLIP／BLIP。

预先冻结 waterway、belief 两个真实缺图实体及 bludgeon 一个模拟缺文本实体；WN18RR 当前对齐数据没有真实缺文本实体。图片先以同提示词运行种子 20260709、20260710，最多追加一次 Reflection 图片。没有启动全量，也不是旧／新或多视角受控比较。

## 逐例结果

| 实体 | 实际产物 | Stage 4 与最终状态 |
| --- | --- | --- |
| waterway / 04559275 | 两张水道图片，无 Reflection | 两张均通过；选出 image_002 并导出 |
| belief / 05941423 | 两张几何抽象图，Reflection 后再生成一张 | 三张语义分均为 0；一次修复后仍无可采用结果，结束为 no_usable_candidate |
| bludgeon / 02855793 | 一条基于原图特征的文本，无 Reflection | 文本通过并导出；没有把样本外观泛化为所有 bludgeon 的定义 |

waterway 两图的实体语义分分别为 3、4，事实一致性均为 4；选中图确实显示有两岸边界的连续水道。人工查看两张实际图片后，风格仍偏绘画，不能因 Judge 高分声称已达到摄影质量。

bludgeon 从第三张原图的木纹／木质外观和一端收窄的细长形状生成：

> The pictured subject is an elongated cylindrical object with a visible grain pattern, indicating wooden construction. It features a shape that tapers towards one end.

Judge 给实体语义 3、事实一致性 4、语义覆盖 3、可读性 4。来源记录明确为 `image`、`depicted_instance`，没有加入人物衣着、背景及其他图片的全部细节。复核原图可见棒状形态与木质外观，但这是低分辨率样本，材质仍属于外观判断；不能证明所有目标实例均为木质。

belief 初始提示词只把“认作真实的认知内容”写成发光几何物体，SDXL 得到套叠方框／圆球。Judge 明确指出缺少信念相关视觉线索，建议使用认知或命题表达。已核对真实 Qwen 请求：`judge_feedback` 与原始 Judge 的四维分数、原因、证据、建议、未确认新增信息和诊断完全一致。

但 Qwen 的修复提示词仍要求“中央发光方块代表命题，外围套叠方框”，结果继续是几何图案。三张图视觉可用性均为 4，实体语义均为 0，事实一致性为 null；评分均正常接收，没有因 null、缺少前缀或观察不完整而整体失败。第二张的不完整逐项观察仅记录 `unassessed_checks` 诊断。

因此，本轮证实反馈传递和有界终止有效，未证实 belief 的语义生成已修复。问题仍包括 Qwen 将建议转化为可画场景的能力，以及现有“保留有用原内容／构图”提示可能造成的锚定；后者是待验证解释，不能仅凭本例认定因果。API 上的提示词 supported 也未预测真实图像成功。

## 核验与边界

491 项相关软件回归通过，无失败、错误或跳过；这不是全仓测试结论。另有 waterway、belief 两例真实 API 提示词修复完成，但与本次新生成图片分别保存。

本轮 GPU 共生成 5 张不同哈希的 SDXL 图片和 1 条文本。15 次本地 Qwen 响应（含预检）均为 `finish_reason=stop`，无空响应。三个实体都完成 Stage 4，没有 `evaluation_failures`；失败内容明确保留，没有输出悬空的 needs_review，也没有无限重试。未遇到候选级传输失败，容错能力由专项回归覆盖，不能声称本轮自然复现了该故障。

运行后重新核对 161 个执行源文件、冻结清单、配置／辅助脚本、原始数据与四张源图哈希，均未变化。5 张图片可正常解码，元数据均为 `sdxl-local`、`fallback_used=false`；导出文件哈希与选中候选一致。完整原始候选、Judge 调用、一次 Reflection 的前后结果均保留。

全部原始记录位于本地 `outputs/pipeline-refactor-retest-20261107-r1/`：`native-verification.json`、`cluster-final.json`、`native-batch-r1/summary.json` 及各实体 `attempts/`。`analyze_native.py` 可重新核验；`verification/scope.json` 中“提交待配置”是冻结前的历史备注，最终作业状态以 `cluster-final.json` 为准。

当前只支持小规模继续诊断，三例不能证明大批量通过率或多视角收益。尤其不能因作业 SUCCEEDED 将 belief 计为内容成功。
