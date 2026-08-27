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

- Cycle ID：`cycle-20260818-01`
- 唯一活动阶段：Round3 PE reward amendment `CODE_IMPLEMENTATION`
- Round2交接：2026-08-26只读核验确认正式experiment已在step590停止，step580/589/590保留、第二方法未启动、两个pruner未运行；旧环境已删除但runs/checkpoints保留，不得清理或覆盖。
- Round3修订：用户于2026-08-27明确要求恢复早期Theory v0.2的`pi_theta/pi_ref` PE，把旧两个动态臂登记为SimPO-reward，并新增两个DPO-reward动态臂；对应`theory/current_theory.md` `r3-theory-v1.1`与`exp/current_experiment.md` `round3-exp-v1.6`。
- 当前实现范围：Qwen3-1.7B、SSPO双源缩放数据、七方法统一250 steps、共同1K selection、独立997-pair双head test；旧五方法产物不可覆盖，新两个方法在旧controller终态后以独立extension experiment各自三卡串行。
- 本地代码边界：本地只编辑与静态文本复核，不import/运行/测试项目；未经用户审阅不commit/push。
- 服务器事实：旧五方法formal `round3-20260826-04`绑定exact commit `b1beef5550ac47c9c78b98c1729014cc153b1251`，仍须以实时只读证据确认终态；不得热改checkout、停止controller或删除产物。
- 当前授权：只解锁本地`round3-code-candidate-v0.6`实现与静态检查。未经用户审阅本次代码交接，不得commit/push、部署、测试或启动两方法extension；Round4保持锁定。
- Round3 extension `SERVER_EXECUTION`、`RESULT_HANDOFF`、`NEXTCYCLE_DISCUSSION`与Round4：锁定

## 阅读顺序

1. 当前Round3实现：`theory/current_theory.md` v1.1与`exp/current_experiment.md` v1.6是用户明确要求的修订规范；当前入口是`../code/CODE_OVERVIEW.md`，代码交接尚待确认。
2. Round1/Round2历史：读取`theory/theory_changelog.md`、`exp/experiment_archive.md`和Git历史，不再从两个`current_*`混读。
3. Round2不完整结果交接：读取`result/current_result.md`与`code/ROUND2_LIVE_HANDOFF.md`；后续服务器操作前必须重新获取实时证据。
4. Round3当前只实现获批规范；实质偏离时返回`EXP_DISCUSSION`重新批准。
5. 代码交接获用户明确确认后才可执行；结果与下一轮仍按状态机顺序推进。

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
