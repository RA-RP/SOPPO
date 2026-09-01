# 所有规划归档

本文件按时间顺序追加每一版下一轮规划，包括后来未执行或被推翻的计划。历史条目只追加状态和复盘，不改写当时理由。

## 归档模板

### <来源 cycle ID> → <目标 cycle ID> — <日期> — 规划版本 <N>

- 对应结果：
- 证据摘要：
- 对理论的处理：
- 候选行动与优先级：
- 选定行动及理由：
- 成功条件：
- 资源与风险：
- 最终状态：待执行 / 已执行 / 部分执行 / 放弃 / 被替代
- 后续复盘：

## 归档记录

### 初始化预置规划 — 2026-08-21 — 历史占位稿

- 对应结果：当时尚无结果。
- 候选行动：提炼首版理论、整理`C_\gamma`、基于理论设计首个MVP。
- 状态：已被后续Round1/Round2工作事实替代；从未作为正式`NEXTCYCLE_DISCUSSION`获得批准。
- 归档原因：2026-08-25整理current文档时，用户要求旧候选不再与当前主线混排。完整旧正文可由Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`的`human_read/nextCycle/current_plan.md`恢复。

### `cycle-20260818-01` Round2 → Round3 — 2026-08-26 — 规划版本 `round3-activation-plan-v0.1`

- 对应结果：`../result/current_result.md` `round2-result-administrative-close-v0.1`，`NO_CONCLUSION`
- 证据摘要：Round2未取得最终白名单证据；experiment ID、server commit、step、PID与final metrics均未知
- 对理论的处理：不用Round2支持、反驳或修改理论；Round3 `r3-theory-v0.8`是用户独立审阅并批准的方案
- 候选行动与优先级：用户直接选定Round3，不再等待Round2结果或优化Round2
- 选定行动及理由：进入已批准`round3-exp-v1.3`的`CODE_IMPLEMENTATION`，实现Qwen3-1.7B五方法、SSPO双源数据、共同selection、独立test与双vLLM replica架构
- 成功条件：实验合同到代码/配置/输出/服务器验收的映射完整，旧Round1/Round2入口不被静默改写
- 资源与风险：本地只做静态编辑；服务器阶段必须先核验Round2潜在运行、GPU与磁盘，不停止或删除旧产物
- 最终状态：已批准；Round3 `CODE_IMPLEMENTATION`于2026-08-26激活
- 后续复盘：待Round3代码交接和服务器验收

### `cycle-20260818-01` Round3 → `cycle-20260901-01` Round4 — 2026-09-01 — `round4-activation-plan-v0.1`

- 对应结果：`../result/current_result.md` `round3-result-administrative-close-v0.1`
- 证据摘要：Round3五方法formal完成；DPO-1K/DPO-8K形成paired基线，GitHub-loss SSPO与SimPO-reward PE未超过DPO-1K；DPO-reward amendment未执行。
- 对理论的处理：不把未执行amendment视为负面结果；Round4改为DPO、SSPO、StaticPE三方法直接比较，并增加AlpacaEval 2.0。
- 选定行动：同一Qwen3-1.7B和10%双源数据；DPO只读labeled；DPO effective batch16，SSPO/StaticPE effective batch64；先4090-3 2-step全链smoke，再迁移不可变镜像到FusionOne 8×A100。
- 成功条件：三方法数据/model/init/eval可追溯，loss分项持久化，adapter merge/reload通过，完整805条Alpaca输出与官方LC/win rate齐全。
- 资源与风险：4090 Docker/scratch待实时验证；FusionOne 8×A100由用户确认已实机验证；私有凭据和大体积资产禁止进入Git/镜像。
- 最终状态：用户明确批准目标切换；新cycle从Round4 `THEORY_DISCUSSION`开始，下游仍锁定。

### `cycle-20260901-01` Round4执行顺序更新 — 2026-09-01 — `round4-activation-plan-v0.2`

- 用户决定：为优先占用目标资源，先创建/占用FusionOne的2张A100。
- 更新顺序：4090-3准备镜像和冻结数据 → SSH直传A100仓库外数据盘并校验SHA → 2×A100容器preflight → 三方法顺序2-step smoke → 单独formal授权 → 同卡顺序训练与AlpacaEval。
- 删除项：4090-3训练smoke和三方法6卡并发不再属于Round4计划。
- 风险：4090-3的Docker/BuildKit与scratch仍须实时解决；A100侧模型供给方式仍待冻结。
- 门禁：本更新不越过Round4 `THEORY_DISCUSSION`，不构成任何服务器动作授权。
