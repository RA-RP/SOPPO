# Round3激活规划：跳过Round2证据等待

> 历史激活记录，当前不处于`NEXTCYCLE_DISCUSSION`。2026-08-27用户已在同一Round3内把PE reward修订为`r3-theory-v1.1`/`round3-exp-v1.6`：旧五方法run不可改写，新增两个DPO-reward动态方法；当前活动阶段为该修订的`CODE_IMPLEMENTATION`，Round4仍锁定。下文五方法/v0.9/v1.4内容只说明最初如何激活Round3，不再是当前实现合同。

## 0. 状态

- 来源Cycle/Round：`cycle-20260818-01` / Round2
- 目标：同一研究cycle中已批准的Round3
- 规划版本：`round3-activation-plan-v0.1`
- 用户批准：2026-08-26明确指示“Round2先不管了，直接开始Round3”
- Round3理论：`../theory/current_theory.md` `r3-theory-v0.9`，方案B勘误已批准
- Round3实验：`../exp/current_experiment.md` `round3-exp-v1.4`，方案B勘误已批准
- 当前唯一活动阶段：Round3 `SERVER_EXECUTION`（v0.4获批重新验证，通过后直接挂载formal）

## 1. 证据摘要与理论处理

Round2没有可核验的最终白名单证据，因此对Round2保持`NO_CONCLUSION`，不用它支持、反驳或修改理论。Round3不冒充“由Round2结果驱动”；它是用户在独立阅读SSPO源码、数据与评价合同后批准的新研究路线。

## 2. 选定行动

1. 在本地仅编辑`code/`下的Round3源码、配置、服务器脚本与`CODE_OVERVIEW.md`；
2. 实现五方法：DPO-1K、GitHub-loss SSPO、DPO-8K、DPO+PE-SFT+rollout、DPO+PE-rollout-only；
3. 实现SSPO双源数据manifest、共同1K validation selection、独立997-pair双head final test、SSPO state round-trip和双vLLM replica ACK协议；
4. 实现独立Round3 train/rollout环境文本与3×4090服务器入口，但在代码交接获用户确认前不上传、不创建服务器环境、不测试或运行；
5. 完成实现映射和静态文本复核后，向用户交接diff及服务器待验收清单。

## 3. 成功条件与资源边界

- 每条获批实验合同都能追踪到配置、实现符号、输出和服务器验收项；
- Round3实现不修改Round1/Round2旧入口的冻结语义，不覆盖旧配置/产物；
- 本地不安装依赖，不import项目，不运行formatter/linter/test、数据、模型、评价或GPU任务；
- 未经用户审阅不`git commit`、不`git push`；
- 后续服务器阶段必须先只读核验Round2潜在运行、GPU和磁盘；有冲突时fail closed，不停止或删除Round2产物。

## 4. 规划交接

- 规划已归档：是，见`plan_archive.md`
- 用户明确通过：是
- 确认日期：2026-08-26
- 下一入口：`../../code/CODE_OVERVIEW.md`
