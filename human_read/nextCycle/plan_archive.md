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
