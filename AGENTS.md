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

- 当前 cycle：`cycle-20260901-01` / Round4
- 唯一活动阶段：`CODE_IMPLEMENTATION`
- 当前入口：`code/CODE_OVERVIEW.md`
- 批准基线：`human_read/theory/current_theory.md` `r4-theory-v1.0`与`human_read/exp/current_experiment.md` `round4-exp-v1.0`均于2026-09-01获用户明确通过；用户明确授权直接进入code阶段。
- Round4范围：同一`Qwen/Qwen3-1.7B`、UltraFeedback/UltraChat默认各0.1、epoch1，对比DPO-label-only、SSPO与StaticPE，并补充统一的AlpacaEval 2.0评价链。
- 已明确决定：StaticPE `lambda=0.1`；每方法2 GPU、GA8；SSPO/StaticPE每设备4、effective batch64；DPO每设备1、effective batch16；三者不重复或截断数据来强凑相同步数。
- 目标资源与流程：用户于2026-09-01确认亲自验证FusionOne存在8张A100，并决定先创建/占用2张；当前`A100-2` SSH目标呈现AutoDL风格入口，不能据此认定为同一FusionOne资源。该目标已核验为2×A100-PCIE-40GB、Python3.12.3/CUDA12.4可用，数据盘和文件存储未挂载；本轮经用户明确选择直接使用系统盘`/root`下的平级仓库外目录。
- Round3边界：formal `round3-20260826-04`的旧五方法已完成并行政结项；拟议DPO-reward extension未运行，取消其继续执行。4090-3旧`runs/`和`envs/`已于2026-09-01按用户明确指令行政清空，不能再引用为现存证据。
- 当前代码：旧服务器基线为`round4-code-v1.0.3` / exact commit `2854c10b6da56f650c91f45d9a685b98cdc02375`；本地已形成`round4-code-v1.1.0`候选 exact code commit `af6dac49044978d76aeca4d5fcb0d11856a1c104`，补齐全链smoke、离线Alpaca与普通DPO指标缺陷修复，尚未上传或运行。
- 当前执行边界：4090-3旧exact commit的137-wheel离线包和Qwen/UltraFeedback/UltraChat三份资产已复核；4090→A100传输、逐文件SHA和`/root/envs/round4-py312`环境门禁均已通过。服务器审计发现缺口后已返回代码阶段；`v1.1.0`的exact commit、wheelhouse、新增Alpaca资产和新A100环境均待交接批准。formal训练仍按实验合同单独授权。
- 锁定阶段：`SERVER_EXECUTION`、`RESULT_HANDOFF`与`NEXTCYCLE_DISCUSSION`锁定；当前修复版完成静态交接后才重新解锁smoke执行。
- 下一阶段条件：`round4-code-v1.1.0`形成exact本地commit并完成静态交接，用户明确确认该版本可以提交服务器，进入`SERVER_EXECUTION`；服务器白名单执行完成或形成完整失败摘要后才进入`RESULT_HANDOFF`。

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
