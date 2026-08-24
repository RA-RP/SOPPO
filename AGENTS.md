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
- 唯一活动阶段：`CODE_IMPLEMENTATION`
- 当前入口：`human_read/exp/current_experiment.md` v0.6 的第二轮 rollout 扩展；当前实现目标为3×4090上 GPU0–1 native TP-LoRA、GPU2独立vLLM
- 已完成门禁：第一轮既有理论/实验/代码与执行记录保持冻结；用户已于2026-08-24明确确认 Round2 commit `c2c9069a0b1a1187c8e709729b33b15aaec8c454`，服务器 clean checkout 和 train/rollout 两个环境已核验。
- 最近服务器证据：commit `f54f6f4d744d138c80f2309ec5e350f1d5a428b3` 的 `exp-20260824-02-round2-tp2` 已通过24k锚点、config、server tests、GPU稳定等待和vLLM ready；首条strong smoke在optimizer step前被错误的“必须存在sharded DTensor”门禁拒绝。Transformers 5.4实际使用checkpoint-backed本地Tensor slices；控制器失败后还残留vLLM EngineCore。
- 当前代码变更：TP门禁改为用safetensors完整shape逐项验证每个rank的本地shard shape、TP plan和device mesh；vLLM改为独立进程组并按STOP→TERM→KILL边界清理全部子进程；`status_all.sh`新增strong-smoke子状态。该修复未提交、待用户审阅，因此活动阶段保持 `CODE_IMPLEMENTATION`。
- 锁定阶段：`SERVER_EXECUTION`、`RESULT_HANDOFF`、`NEXTCYCLE_DISCUSSION`
- 下一解锁条件：用户审阅当前TP/清理修复 diff，手工形成并明确确认新的 clean commit；保留失败experiment并使用新ID重启server tests与strong smoke。

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
