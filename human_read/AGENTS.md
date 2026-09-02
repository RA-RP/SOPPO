# `SOPPO/human_read/`：研究迭代主入口

本目录是实验与 idea 碰撞、供用户逐轮仔细阅读和逐门决策的地方。每轮严格按 `theory → exp → code → 服务器执行 → result → nextCycle` 推进；代码阶段的主文档位于 `../code/CODE_OVERVIEW.md`。

## 阶段状态机

| 阶段 | 允许开展的核心工作 | 退出条件 | 未退出时锁定的内容 |
| --- | --- | --- | --- |
| `THEORY_DISCUSSION` | 与用户阅读、讨论、修订理论 | 用户明确通过带版本号的当前理论 | 实验定稿、代码实现/上传、服务器执行、结果与下一轮讨论 |
| `EXP_DISCUSSION` | 与用户讨论实验问题、对照、数据、指标、判据和预算 | 用户明确通过当前设计 | 代码实现、上传、服务器任务、结果与下一轮讨论 |
| `CODE_IMPLEMENTATION` | 在 `../code/` 将获批设计转化为源码、配置和服务器脚本，并维护 `CODE_OVERVIEW.md` | 实现映射与静态复核完成，用户明确确认当前代码版本可以提交服务器 | 上传、服务器任务、结果与下一轮讨论 |
| `SERVER_EXECUTION` | 上传已确认代码，并在服务器测试、运行、评价、聚合 | 服务器白名单结果摘要或终止/失败摘要与证据索引齐全，可进入结果解读 | 下一轮讨论；结果只能依据实证填写 |
| `RESULT_HANDOFF` | 与用户核对执行完整性、事实、解释、局限和结论边界 | 结果文档归档完成，且用户明确确认交接完成 | 下一轮讨论 |
| `NEXTCYCLE_DISCUSSION` | 依据本轮已交接证据讨论下一步 | 用户明确通过规划；创建新 cycle 并回到理论讨论 | 新一轮实验设计与执行 |

### 共同门禁规则

- 任一时刻只允许一个活动阶段；当前阶段写在本文件和对应 `current_*` 文件中。
- 用户沉默、未反对、继续提问或要求查看材料都不算通过。批准记录至少包含对象、版本、日期和“用户已明确通过/授权”的状态。
- 下游内容可以保留为锁定草案供将来讨论，但不得被称为“已定稿”“已批准”。实验通过只解锁代码阶段；代码交接未获用户明确确认前不得上传或运行。
- 返回任一上游阶段时，其下游全部重新锁定；若上游版本发生实质变化，原批准不自动沿用。
- 失败、负面、提前终止和无结论运行同样进入 `RESULT_HANDOFF`。

## 当前状态

- Cycle ID：`cycle-20260901-01` / Round4
- 唯一活动阶段：`CODE_IMPLEMENTATION`
- Round3交接：旧五方法formal已完成并行政结项；DPO-reward extension未运行且不再推进。4090-3旧`runs/`和`envs/`已于2026-09-01按用户明确指令行政清空。
- 当前理论：`theory/current_theory.md` `r4-theory-v2.0`，2026-09-02用户明确通过。
- 当前实验：`exp/current_experiment.md` `round4-exp-v2.0`，2026-09-02用户明确通过。
- 当前范围：Qwen3-1.7B、UltraFeedback/UltraChat各0.1、epoch1，对比DPO、SSPO、StaticPE和FrozenPE；DPO effective batch16，其余三法effective batch64；目标评价为A100生成、4090 judge的AlpacaEval兼容WR/LC。
- 已执行代码：`round4-code-v2.0.0`及修复提交`92259df`、`98dc1aa`、`67ebed0`均于2026-09-02获用户明确批准并在A100执行。`67ebed0`已进入Qwen3模型前向，但自定义梯度检查点包装器未兼容`functools.partial`而停止，未完成优化step。
- 当前代码交接：`round4-code-v2.0.4`（本次修复后的仓库`HEAD`）；用户已明确授权自行提交、同步与迭代，直至 full-chain smoke 通过。
- 当前执行边界：因服务器纯实现缺陷返回代码阶段；新exact commit未经确认前，上传、新环境、smoke与formal重新锁定。formal的既有意向不允许绕过新exact commit的代码交接门禁和full-chain smoke门禁。
- 目标硬件与流程：FusionOne 8×A100为用户已实机验证事实；先创建/占用2张，4090-3准备镜像/数据并经SSH传入，全部smoke与formal在同一2卡上顺序执行。具体显存/拓扑/映射仍待preflight。
- `RESULT_HANDOFF`及其后阶段：锁定。

## 阅读顺序

1. 当前Round4理论与实验基线：`theory/current_theory.md` `r4-theory-v1.0`、`exp/current_experiment.md` `round4-exp-v1.0`。
2. Round1/Round2历史：读取`theory/theory_changelog.md`、`exp/experiment_archive.md`和Git历史，不再从两个`current_*`混读。
3. Round2不完整结果交接：读取`result/current_result.md`与`code/ROUND2_LIVE_HANDOFF.md`；后续服务器操作前必须重新获取实时证据。
4. Round4正式代码入口为`../code/CODE_OVERVIEW.md`；既有StaticPE代码仍须按获批实验逐项复核。
5. 当前按已批准的`round4-code-v1.0`执行；结果与下一轮仍按状态机顺序推进。

各子目录的 `AGENTS.md` 规定具体字段和更新动作。

## 三类记录的共同语义

- `current_*`：只保留当前轮最值得阅读的完整版本，可以随当前轮推进而更新。
- `*_archive` / changelog：按 cycle ID 追加所有版本或变化，原则上只追加、不回写历史结论。
- `useful_*`：经过复盘后精选、预计可写入论文的内容；不得自动把所有版本复制进来。

## 每轮完成条件

- 六个阶段使用同一个 cycle ID，所有实际运行都列出 experiment ID，代码阶段记录代码版本或源码校验值。
- 理论写出可证伪预测，实验设计预先写出判断标准。
- 结果区分事实、解释与局限，并链接 `../../exp/<experiment_id>/` 的原始证据。
- 下一步必须由本轮证据驱动，并说明继续、修订或放弃哪一部分理论。
- 当前文件与全量归档已经同步；有论文价值时再同步精选历史文件。
- 理论、实验设计、代码交接和结果交接的显式用户确认均已记录，且下一轮规划已明确通过。

## 证据读取边界

- 本目录只阅读服务器生成的聚合摘要与远程证据索引，不在本地加载数据、模型或逐样本预测重新计算结果。
- 若现有摘要不足以支持解释，应新增服务器评价/聚合任务，而不是在本地安装包或运行分析代码。
- 文档可以引用服务器上的原始证据路径，但不得把黑名单产物复制到本地工作区。
