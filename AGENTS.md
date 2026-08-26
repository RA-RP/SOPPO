# `SOPPO/`：研究执行区与唯一 Git 仓库

本目录把上游想法与资料转化为可复现代码和逐轮研究证据。

## Git 与服务器目录边界

- 本目录是本地与服务器上的唯一 Git 仓库根；只允许本目录下存在 `.git/`。
- `code/observe/LLM-output-density/` 是由本仓库统一管理的普通目录，不得保留或创建嵌套 `.git/`。
- 本地仓库路径是 `ICLR/SOPPO/`；本地 `ICLR/` 自身不是 Git 仓库。
- 服务器仓库路径是 `<SERVER_BASE>/SOPPO/`；静态资料位于与其平级的 `<SERVER_BASE>/ICLR/`，后者没有 `.git/`。
- 服务器环境、缓存、数据、模型、运行产物和回传摘要分别位于 `<SERVER_BASE>/{envs,cache,data,models,runs,exports}/`，不得写入 Git 仓库。

## 子目录职责

- `code/`：当前方法的正式实现；维护面向人的代码总览、功能模块和超参数说明。
- `exp/`：实际运行的实验；每次实验必须拥有独立文件夹并保留配置、日志和产物索引。
- `human_read/`：理论、实验设计、结果解释与下一轮规划，是用户逐轮阅读和决策的主入口。
- `human_read/code/`：面向用户的代码架构、infra 学习和 pipeline 说明；只保存人类可读文档，不放置正式实现、可执行入口、运行产物或依赖环境。该目录中的文档可以引用根目录 `code/` 的实现，但不替代 `code/CODE_OVERVIEW.md`。

## 迭代闭环

每轮按以下带显式用户门禁的顺序推进：

1. `THEORY_DISCUSSION`：在 `human_read/theory/` 与用户阅读、讨论和修订当前理论；用户明确通过该理论版本后才解锁下一步。
2. `EXP_DISCUSSION`：在 `human_read/exp/` 与用户讨论并设计检验理论的实验；用户明确通过设计后只解锁代码实现。
3. `CODE_IMPLEMENTATION`：在 `code/` 将获批设计转化为源码、配置和服务器脚本，同时维护 `code/CODE_OVERVIEW.md`；用户明确确认代码交接完成、可以提交服务器后才解锁执行。
4. `SERVER_EXECUTION`：经 SFTP 上传已确认代码，并只在服务器完成测试、数据、模型、smoke、训练、评价和聚合；在 `exp/<experiment_id>/` 保存本地允许保留的摘要与远程证据索引。
5. `RESULT_HANDOFF`：在 `human_read/result/` 与用户逐项讨论结果、偏差、解释和结论边界；用户明确确认交接完成后才解锁下一步。
6. `NEXTCYCLE_DISCUSSION`：在 `human_read/nextCycle/` 与用户讨论下一轮。规划明确通过后创建新的 cycle，并返回理论讨论。

任一时刻只能有一个活动阶段。用户沉默、未反对或普通追问不是批准；所有通过都要在对应 `current_*` 或 `CODE_OVERVIEW.md` 记录版本、日期和状态。预先存在的下游文档或代码接口草案不构成实现或执行授权。不得跳过代码交接直接上传，也不得跳过结果解释而直接将一次运行宣称为理论证据；失败、无显著差异和无法判断的实验也要进入完整记录。

## 当前活动阶段

- 当前 cycle：`cycle-20260818-01`
- 唯一活动阶段：Round3 `CODE_IMPLEMENTATION`
- 当前入口：`code/CODE_OVERVIEW.md`、`human_read/theory/current_theory.md` `r3-theory-v0.9`与`human_read/exp/current_experiment.md` `round3-exp-v1.4`
- 已完成门禁：原`r3-theory-v0.8`/`round3-exp-v1.3`于2026-08-25获批；冻结数据审计发现`test_prefs`只有1,997个有效pair后，用户于2026-08-26明确批准方案B及本地修改，形成`r3-theory-v0.9`/`round3-exp-v1.4`并保持`CODE_IMPLEMENTATION`。
- Round2边界：2026-08-26只读核验确认`exp-20260824-05-round2-tp2`已在step590停止，step580/589/590保留，第二方法未启动，两个pruner均未运行。旧环境已删除但runs/checkpoints仍在；不得删除或覆盖。
- 当前本地代码基线：方案B实现已出现在本地HEAD `8eedca51067162562cade72408930c8b2321ffda`；该commit存在本身不等于代码交接或服务器执行批准。当前GLM指南等后续修订仍为未提交worktree，Codex不commit/push。
- 协作分工：用户于2026-08-26指定Codex为Round3设计与代码主责，只有Codex可以创作或修订`SOPPO/`内容；GLM只按`code/scripts/round3/GLM_VALIDATION_GUIDE.md`机械部署用户指定的exact commit、执行分阶段命令并回传证据，不得现场编辑/修复源码、commit或push。
- Round3实现合同：Qwen3-1.7B非量化LoRA；SSPO双源类型缩放数据；DPO-1K、GitHub-loss SSPO、DPO-8K与两个动态PE共五方法；全部1 epoch/250 steps；共同1K validation selection与独立997-pair双head final test；GPU0单卡训练，动态方法在GPU1/2使用双vLLM replica与step/adapter hash/ACK屏障。PE-static与AlpacaEval/MT-Bench不在Round3实现范围。
- 锁定阶段：Round3 `SERVER_EXECUTION`、`RESULT_HANDOFF`与`NEXTCYCLE_DISCUSSION`。
- 下一阶段条件：Round3实现、`CODE_OVERVIEW.md`追踪映射与本地静态复核完成，且用户明确确认当前代码版本可提交服务器。

## 标识与交叉引用

- 研究轮次使用 `cycle-YYYYMMDD-NN`，例如 `cycle-20260818-01`。
- 单次实验使用 `exp-YYYYMMDD-NN-short-name`；同一轮可包含多个实验。
- 每份设计、原始实验、结果和规划都应写出对应标识及相对路径。
- `current_*` 文件只表示当前工作状态；`*_archive` 保存全部版本；`useful_*` 只收录经复盘后预计会用于论文的内容。

## 变更同步

- 代码目标、模块、入口、默认超参数或输出格式变化时，同步更新 `code/CODE_OVERVIEW.md`。
- 实验参数覆盖值记录在实验文件夹，不反向冒充代码默认值。
- 新建的代码模块目录和单次实验目录默认继承本文件与其父目录规则；除非出现独立工作流，否则不复制新的说明文件。
- 若当前活动阶段或批准状态发生变化，同步更新本文件、`human_read/AGENTS.md` 与相应 `current_*` 文件；不得只在聊天中口头推进阶段。

## 执行位置

- 本地只维护源码、配置、实验设计和人类可读记录，不安装项目依赖，也不运行任何涉及依赖、数据、模型、测试、评价或 GPU 的命令。
- 所有可执行研究任务必须依据本地 `../machine/` 的运行手册提交到服务器；CPU 单元测试、数据审计和统计聚合也不例外。
- 服务器保存数据、模型、逐样本输出和原始日志。本地 `exp/` 与 `human_read/` 只保存聚合摘要、配置文本、校验值和远程证据索引。
