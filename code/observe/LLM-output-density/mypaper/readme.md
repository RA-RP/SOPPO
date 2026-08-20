# mypaper Workspace Guide

> **Clean-repository note.** This copy keeps the original research workflow, but large
> `local_experiment_results/` trees are intentionally excluded. The compact evidence
> needed by the current paper is under `../paper_artifacts/`. Absolute `/root/...` paths
> below describe the original execution environment and should be interpreted through
> `docs/PORTABILITY.md` after cloning elsewhere.

本文档说明 `/root/LLM-output-density/mypaper` 的目标结构、各目录职责，以及每个 research cycle 的迭代执行流程。

## 目录定位

`mypaper/` 是论文研究材料的工作区，不是单纯的历史归档。之后应区分两类内容：

1. **当前最新工作面**：放在 `mypaper/` 根目录下的固定主题文件夹中，始终代表当前论文和下一轮实验应优先读取的最新版本。
2. **跨 cycle 演进记录**：放在 `research_cycles/` 下，只保留按板块的跨 cycle 演进文件（`*_evolution.md`），不再保留按 cycle 的完整快照文件夹。详细 artifact（eval 表、FINDING 文档、原始结果）存入 `local_experiment_results/<cycle_name>/`。

## 目标根目录结构

后续整理时，`mypaper/` 根目录应维护以下最新视图：

```text
mypaper/
  readme.md
  related_work/
  theory/
  exp/
    experimental_methodology/
  code/
  result/
  lecture/
  next_cycle/
  research_cycles/
  local_experiment_results/
  literature_library/
  paper_drafts/
```

### `related_work/`

保存当前最新的相关工作综述、source matrix、文献定位、未解决问题和引用策略。

这里应回答：

- 当前研究问题放在什么文献脉络中？
- 哪些已有工作支持或挑战我们的假设？
- 哪些 related-work 结论已经被最新实验结果更新？

### `theory/`

保存当前最新的理论设定、机制假设、指标解释、变量定义和 claim 边界。

这里应回答：

- 我们到底想证明或证伪什么？
- OPD/SFT/ID/OOD/geometry 等核心概念如何定义？
- 哪些机制解释是允许的，哪些需要更多证据？

### `exp/`

保存当前最新的实验设计、acceptance criteria、运行计划、数据/模型选择、评价协议和实验检查清单。`exp/` 内部分两层，职责不同：

- `exp/experimental_methodology/`：**稳定层**，跨 cycle 维护，回答"怎么做、为什么这么做"——模型身份与 chat/think 格式（`model_and_data.md`）、训练流程（`training_protocol.md`）、评测协议（`evaluation_protocol.md`）、几何探针协议（`geometry_protocol.md`）。这一层存在的原因：早期 `exp/` 只记录"接下来做什么+gate"，完全没有实现细节，直接导致 MATH500/GSM8K 的测量 bug 拖了很久才被发现（详见 `theory/`、`result/` 中的相关记录）。
- `exp/current_experiment_design.md` 等：**本轮层**，每个 cycle 只更新发生变化的部分。

这里应回答：

- 准备阶段：使用的是什么数据集、使用了哪些模型？（稳定层）
- 训练阶段：有哪些训练方式？每种训练方式的配置如何（student 是否开启长思考、cutoff 是多少、是否是 chat 格式）？（稳定层）
- 测试阶段：测试哪个下游任务，哪些 ID/OOD，测试的格式如何、是否和训练时匹配？（稳定层）
- geometry 阶段：要测试哪些 geometry 指标，现在认为哪个为主哪些为辅，最好和 theory 结合？（稳定层）
- 分析阶段：要分析哪些指标之间的关系，这些关系和 theory 有什么关系？（本轮层）
- 下一次实验要跑什么？成功、失败、无法判断分别由什么 gate 决定？（本轮层）
- 实验输出必须包含哪些表、图、日志和 artifact？（本轮层）

### `code/`

代码编写对话的当前工作面，包含四类文件：

| 文件 | 类型 | 说明 |
|---|---|---|
| `current_code_structure.md` | 持续维护 | 当前代码库完整说明（脚本结构、运行方式、协议）。每 cycle 追加变更节，不全文重写。 |
| `code_evolution.md` | append-only | 跨 cycle 演进：每轮新增/删除/修改了什么，以及实际执行反馈。 |
| `current_code_brief.md` | 覆盖维护 | 当前 cycle 的具体任务说明（要跑什么、输入输出路径、copyback 规则）。 |
| `QA_<cycle>.md` | 每 cycle 一份 | 代码对话开始前的 QA 讨论。确认后才开始执行；保留原文，不删除。 |

**每 cycle 流程**：Experiment Design 对话写 `current_code_brief.md` → 代码对话新建 `QA_<cycle>.md` 提问 → 用户回答后内化结论 → 开始执行 → 执行完成后将实际情况追加到 `code_evolution.md` 并更新 `current_code_structure.md`。

代码对话启动时必读顺序：

```
readme.md
↓ code/current_code_brief.md（当前任务）
↓ code/current_code_structure.md（代码库全貌）
↓ exp/experimental_methodology/（稳定协议层，4 文件）
↓ exp/current_experiment_design.md（本轮设计规格）
↓ local_experiment_results/<上一 cycle>/（先验结果，如需要）
```

**仓库级代码描述文件 `../Readme-experiments.md`**（位于仓库根目录，mypaper 之外）：`experiments/opd_sft_h1` 的**代码流程与配置权威说明**——按流水线阶段（数据母池 / 训练 / 评测 / 探针几何）逐节描述函数、参数、产物路径与常见问题。`code/current_code_structure.md` 的 frontmatter 即以它为 `source`：前者是仓库级源头，后者是 mypaper 工作面内按 cycle 追加变更的镜像/延伸。需要查「某个阶段的代码到底怎么实现、配置项含义、为什么这么设」时读 `../Readme-experiments.md`；需要查「本 cycle 改了哪些脚本」时读 `code/`。训练机制的实现细节（如 Cycle 08 verl OPD 的 loss 形式与 seqlen 归一化口径）也记录在 `../Readme-experiments.md` 的对应章节。

### `result/`

最高层、最可读的结果叙事，使用增量形式记录，包含三个核心文件：

- `incremental_log.md`：每个 cycle 加了什么、修正了什么（增量追加，历史不改写）。
- `current_picture.md`：当前整个项目所有**可用**实验结论的聚合视图。有效 = 有修正版用修正版、未被覆盖的用原始结果、已知协议有缺陷的不算可用。
- `claims_allowed.md`：当前可声称和不可声称的结论边界。

这里应回答：

- 这一轮新实验得到了什么？（`incremental_log.md`）
- 修正之前实验后得到了什么新结果？（`incremental_log.md`）
- 现在整个项目所有可用的实验说明了什么？（`current_picture.md`）

详细数字和 artifact 在 `local_experiment_results/<cycle_name>/` 中，`result/` 通过路径引用它们，不直接存大表。

### `lecture/`

结果到论文的中间层。`result/` 回答"实验说明了什么"，`lecture/` 回答"论文应该怎么讲"——把可用的实验结论转化为论文的论证结构、叙事主线和证据映射，供 `paper_drafts/` 直接取用。

包含三类文件：

- `current_paper_argument.md`：**持续维护**。当前论文的论证骨架——核心 claim、每个 claim 的支撑证据（指向 `result/` 或 `local_experiment_results/`）、叙事顺序、图表选用建议、尚未被实验覆盖的论证空缺。每个 cycle 更新发生变化的部分，不全文重写。
- `lecture_evolution.md`：**append-only**。每个 cycle 追加一节，记录本轮实验给论文论证带来了什么变化：新增了哪个支撑点、推翻或削弱了哪个论点、哪个空缺被填上或被确认为持续空缺。
- `QA_<cycle>_lecture.md`：**每 cycle 一份**。进入 lecture 阶段前提出需要与用户确认的论证问题（这个 claim 现在成立吗？这张图够不够说明问题？），用户回答后内化结论，再更新 `current_paper_argument.md`；保留原文，不删除。

这里应回答：

- 本轮实验结果支持论文的哪个核心论点？（`lecture_evolution.md`）
- 当前论文的论证结构是什么，每个 claim 的证据链是否完整？（`current_paper_argument.md`）
- 哪些论证空缺还需要后续实验填补，哪些可以在讨论章节中承认局限？（`current_paper_argument.md`）

与其他目录的边界：

- `result/` 是实验结论层，不讨论论文结构。`lecture/` 不重复记录数字，通过路径引用 `result/` 和 `local_experiment_results/`。
- `paper_drafts/` 是最终落笔层。`lecture/current_paper_argument.md` 是 `paper_drafts/` 写作时的论证蓝图，而非草稿本身。

### `next_cycle/`

保存下一轮 research cycle 的种子，使用三个文件增量维护：

- `open_questions.md`：**动态增量维护**。记录所有当前开放问题（包含上一轮未解决的问题和当前轮新增问题）。每次只保留 open 状态的问题，resolved 的移入 `history_questions.md`。
- `history_questions.md`：**append-only**。每轮 cycle 结束时追加一节，记录本轮解决了哪些问题和新提出了哪些问题。
- `QA.md`：**覆盖维护**。每次进入 Next Cycle Seed 阶段时重写，提出关键问题 → 用户回答 → 讨论 → 结论内化到 `open_questions.md` 和 `history_questions.md`。历史 QA 内容不保留在本文件（已移入 `history_questions.md`）。

这里应回答：

- 从 result 看来这一轮的问题是什么？
- 下一轮为什么要做？
- 下一轮从当前结果中继承什么？
- 下一轮必须避免重复哪些错误？

## 历史与辅助目录

### `research_cycles/`

跨 cycle 演进记录目录，**只包含按板块的演进文件**（2026-06-16 重构，原按 cycle 的完整快照文件夹已删除）：

```text
research_cycles/
  related_work_evolution.md   + -ch.md
  theory_evolution.md         + -ch.md
  exp_design_evolution.md     + -ch.md
  result_interpretation_evolution.md + -ch.md
  next_cycle_seed_evolution.md + -ch.md
```

每个文件按时间顺序追加每轮的**增量**（这轮相对上一轮变了什么、为什么变），不是逐轮全文复述。详细的每轮 artifact（eval 表、FINDING 文档、原始结果、完整设计文档）存入 `local_experiment_results/<cycle_name>/`；当前工作面文档在 `exp/`、`theory/`、`result/`、`next_cycle/` 中维护。

每个 cycle 收尾时，在对应的 evolution 文件末尾追加新的一节，描述该轮的增量变化。

### `local_experiment_results/`

按 cycle 保存详细实验产物：eval 数字表、FINDING 文档、geometry 表、训练配置、运行环境信息、figures。这里是"想看详细数字就来这里"的第二层存储，也是被 `result/` 和 `exp/` 引用时的实际路径。

规则：

- 结果叙事（解释、增量摘要、可用结论）写入 `result/`。
- 大表、FINDING 文档、原始输出放在 `local_experiment_results/<cycle_name>/`。
- `result/` 中通过相对路径链接到这里的 artifact。

### `literature_library/`

保存文献材料、PDF、读书摘要和可复用的文献背景。

### `paper_drafts/`

保存论文草稿、LaTeX、中文草稿、revision checklist 和投稿版本。

> **变更记录（2026-06-16）**：`legacy_archive/` 已删除。其内容是 cycle 编号体系之前的旧草稿（引用 `nextStepCodex/`、`OOD-ID relatework/` 等已不存在的旧路径），已被 `literature_library/` 完整取代，不再需要单独保留。注意：`source_notes/ood_id_positioning_with_ppl.md`（用几何/PPL 指标筛选 SFT 数据以缓解 OOD 退化的早期设计）已迁移至 `related_work/source_notes/`——它是仍然有效的相关工作，只是尚未并入当前 OPD-vs-SFT 主线，见 `research_cycles/related_work_evolution.md`。

## 对话分工与读写隔离

每个阶段的工作在**独立对话**中完成。核心规则：

> **每类对话只写自己的主目录，读取其他目录但不修改。**

这样每次对话只需加载自己阶段的上下文，大幅减少 token 开销，不同阶段的问题可以在不同对话中独立处理。

### 对话角色与读写范围

| 对话角色 | ✏️ 写入范围（仅此） | 📖 启动时必读 |
|---|---|---|
| **Theory** | `theory/` | `readme.md` → `result/current_picture.md` → `result/claims_allowed.md` → `next_cycle/open_questions.md` |
| **Related Work** | `related_work/` | `readme.md` → `next_cycle/open_questions.md` → `result/current_picture.md` → `result/claims_allowed.md` → `theory/current_theory_update.md` → `related_work/related_work.md` → `related_work/source_matrix.md` → `related_work/incremental_related_work.md` |
| **Experiment Design** | `exp/`、`code/current_code_brief.md` | `readme.md` → `result/current_picture.md` → `result/claims_allowed.md` → `next_cycle/open_questions.md` → `theory/current_theory_update.md` → `related_work/source_matrix.md` |
| **Code Execution** | `local_experiment_results/<cycle>/`、`code/code_evolution.md`（追加 handin 节）、`code/current_code_structure.md`（追加变更节） | `readme.md` → `code/current_code_brief.md` → `code/current_code_structure.md` → `exp/experimental_methodology/`（4 文件全读）→ `exp/current_experiment_design.md` |
| **Result Analysis** | `result/` | `readme.md` → `local_experiment_results/<cycle>/RESULTS_*.md` → `code/code_evolution.md`（handin 节）→ `exp/experimental_methodology/evaluation_protocol.md` → `result/current_picture.md` |
| **Lecture** | `lecture/` | `readme.md` → `result/current_picture.md` → `result/claims_allowed.md` → `lecture/current_paper_argument.md` → `lecture/lecture_evolution.md` |
| **Next Cycle Seed** | `next_cycle/` | `readme.md` → `result/current_picture.md` → `result/claims_allowed.md` → `exp/current_acceptance_criteria.md` → `next_cycle/open_questions.md` → `next_cycle/history_questions.md` |
| **Paper Writing** | `paper_drafts/` | `readme.md` → `lecture/current_paper_argument.md` → `result/current_picture.md` → `result/claims_allowed.md` → `theory/current_theory_update.md`（之后只读，不写其他目录） |

### 特殊读写规则

1. **`exp/experimental_methodology/` 的写入权归 Experiment Design 对话**。Result Analysis 对话发现协议问题时，在 `result/` 中标注 `⚠️ 协议问题待同步`，然后发起新的 Experiment Design 对话来更新稳定层，不在 Result 对话中直接改 `exp/`。

2. **`research_cycles/` evolution files 由关闭 cycle 时的对话追加**，其他对话只读。通常由 Result Analysis 或 Experiment Design 对话负责追加本轮 evolution 条目。

3. **`related_work/current_source_matrix.md` 的 Eval Protocol Sources 区块**：Experiment Design 对话或 Result Analysis 对话在采纳协议来源时可直接追加（协议来源属于 related work 的一部分，这是项目规则）。

4. **`code/code_evolution.md` 的 handin 节**由 Code Execution 对话追加，**Result Analysis 对话只读**——实际执行情况（真实命令、路径偏差、新发现问题）是研究侧了解执行实情的唯一通道，不应由研究侧覆盖。

5. **协议研究规则（适用于所有对话）**：设计任何评测协议或训练格式之前，必须按顺序查阅：(1) 官方技术报告/文档，(2) 同系列/类似模型，(3) 社区实践（lm-eval Issues 等）。被采纳的来源必须添加到 `related_work/current_source_matrix.md`。详见 `exp/README.md`。

### 对话启动检查单

任何新对话进入本项目，按顺序执行：

```
1. 读 mypaper/readme.md（本文件）
2. 确认当前 cycle 名称（查 exp/current_experiment_design.md 的 yaml header）
3. 确认自己的对话角色（Theory / Related Work / Experiment Design / Code / Result Analysis / Next Cycle Seed / Paper Writing）
4. 按上表"启动时必读"顺序读取对应文件
5. 确认自己的写入范围，不写其他目录
```

## Cycle 迭代流程

每一轮 cycle 应按以下顺序推进。**每个阶段开始前必须经过 QA 审查**，流程如下：

1. 在对应阶段目录下新建 `QA_<cycle>_<phase>.md`，列出进入该阶段前需要回答的关键问题
2. 用户回答问题
3. 双方讨论，达成共识后，在 QA 文件中记录讨论结论，再进入该阶段的实际写作
4. Cycle 收尾时（步骤 8），将 QA 文件中的关键结论内化到该阶段目录对应的增量更新文件中（见"维护规则"）；QA 文件本身保留在原目录，不删除

**例外：Next Cycle Seed 阶段不使用 `QA_<cycle>_<phase>.md` 模式，而是覆盖维护 `next_cycle/QA.md`**（单文件，每次进入该阶段时重写）。讨论结论内化到 `next_cycle/open_questions.md` 和 `next_cycle/history_questions.md`；历史 QA 内容通过 `history_questions.md` 保存，不保留旧 QA 文件。

### 1. 更新当前问题

从 `next_cycle/` 读取下一轮种子（或查阅 `research_cycles/next_cycle_seed_evolution.md` 最新条目），确认本轮要回答的核心问题。

输出：

- 当前 cycle 的问题陈述
- 本轮 claim 边界
- 需要保留或修复的上轮经验

### 2. 更新相关工作

在 `related_work/` 中更新最新文献脉络和 source matrix。

输出：

- related-work update
- source matrix
- 新增问题列表

### 3. 更新理论与指标

在 `theory/` 中更新机制假设、变量定义、指标定义和允许结论。几何指标、ID/OOD 轴、scorer 规则等应在这里有自包含说明。

输出：

- theory update
- metric definitions
- claim constraints

### 4. 设计实验

在 `exp/` 中写实验设计和 acceptance criteria。实验必须预先指定 gate，避免结果出来后临时换主指标。

输出：

- experiment design
- acceptance criteria
- required artifacts
- `code/current_code_brief.md`（代码任务说明，代码对话必读）

### 5. 执行与 copyback

**代码对话启动前**：先在 `code/` 下新建 `QA_<cycle>.md`，列出需要与用户确认的问题（协议参数、路径选择、不确定的设计决定）。用户回答后将结论内化回 `code/current_code_brief.md`，再开始执行。

代码对话按顺序读取：`readme.md` → `code/current_code_brief.md` → `code/current_code_structure.md` → `exp/experimental_methodology/`（4 文件）→ `exp/current_experiment_design.md`。

代码执行完成后：
1. 将必要 artifact copyback 到 `local_experiment_results/<cycle_name>/`，把运行路径、表格、图、原始输出位置写清楚。
2. 在 `code/code_evolution.md` 的本 cycle 节追加 **handin 子节**，记录实际命令、实际路径、与 brief 规格的偏差、执行中发现的问题。
3. 在 `code/current_code_structure.md` 末尾追加本 cycle 新增/修改的代码说明。

输出：

- eval tables
- geometry tables
- figures
- summary JSON
- run provenance
- `code/code_evolution.md`（追加 handin 节）
- `code/current_code_structure.md`（追加代码变更节）

### 6. 导入与解释结果

**在写 result/ 之前**，先读取 `code/code_evolution.md` 的本 cycle handin 节，将关键实现细节（实际命令参数、发现的 protocol 偏差、新问题）合并回 `exp/experimental_methodology/evaluation_protocol.md` 或其他稳定层文件，再开始 result 写作。

在 `result/` 中写最新结果报告、结果解释、失败/成功分析和 allowed claims。报告应区分：

- gate 判定指标
- 辅助解释指标
- 噪声级发现
- artifact 或 scorer 问题
- 不能声称的结论

输出：

- latest result report
- failure or success analysis
- claims allowed/disallowed
- artifact index

### 7. Lecture（结果→论文论证）

**进入前**：在 `lecture/` 下新建 `QA_<cycle>_lecture.md`，就本轮结果对论文论证的影响提出问题（例如：这个 claim 现在成立吗？需要哪张图来说明？哪个论点被削弱了？）。用户回答后内化结论，再更新文件。

在 `lecture/` 中：
1. 更新 `current_paper_argument.md`：根据本轮可用结论调整论证骨架——新增支撑点、标注被削弱的论点、更新证据链（指向 `result/` 或 `local_experiment_results/`）、标记仍待填补的论证空缺。
2. 在 `lecture_evolution.md` 末尾追加本 cycle 节：说明本轮实验给论文论证带来了什么变化。

输出：

- `lecture/current_paper_argument.md`（更新）
- `lecture/lecture_evolution.md`（追加本 cycle 节）
- `lecture/QA_<cycle>_lecture.md`（保留）

### 8. 生成下一轮种子

在 `next_cycle/` 中：
1. 覆盖重写 `QA.md`，提出关键问题 → 用户回答 → 讨论
2. 将解决的问题从 `open_questions.md` 移出，追加到 `history_questions.md`（该 cycle 节）
3. 将新问题追加到 `open_questions.md`
4. 写或更新 cycle intent 文件（`current_cycle<N>_intent.md`）

输出：

- `next_cycle/QA.md`（覆盖）
- `next_cycle/open_questions.md`（更新，仅保留 open 项）
- `next_cycle/history_questions.md`（追加本 cycle 节）
- `next_cycle/current_cycle<N>_intent.md`（新建或更新）

### 9. 关闭 cycle 记录

当一个 cycle 完成后：

1. 把本轮详细 artifact（RESULTS、FINDING、eval 表、figures）copyback 到 `local_experiment_results/<cycle_name>/`。
2. 在 `research_cycles/*_evolution.md` 各文件末尾追加该 cycle 的增量条目。
3. 把 `related_work/`、`theory/`、`exp/`、`result/`、`next_cycle/` 更新为下一轮的最新入口。
4. 将各阶段 QA 文件（`QA_<cycle>_<phase>.md`）中的关键结论内化到对应目录的增量更新文件：

   | 阶段目录 | 增量更新目标文件 |
   |---|---|
   | `result/` | `result/incremental_log.md` |
   | `lecture/` | `lecture/current_paper_argument.md`（更新）+ `lecture/lecture_evolution.md`（追加）|
   | `next_cycle/` | `next_cycle/open_questions.md`（更新）+ `next_cycle/history_questions.md`（追加）|
   | `exp/` | `exp/current_experiment_design.md` |
   | `theory/` | `theory/` 中对应的定义/假设文件 |
   | `related_work/` | `related_work/source_matrix.md` |
   | `code/` | `code/code_evolution.md`（追加 cycle 节）+ `code/current_code_structure.md`（追加变更节）|

   内化内容包括：协议决定、修正记录、gate 定义、经验教训。QA 文件本身保留原目录，不删除。

## 维护规则

- 根目录的 `related_work/`、`theory/`、`exp/`、`result/`、`lecture/`、`next_cycle/` 永远代表最新状态。
- `research_cycles/` 只包含 evolution files，代表跨 cycle 演进的叙事记录，不含完整快照。
- 结果报告必须链接到原始 artifact，不要只写结论。
- 指标定义必须自包含，不能只依赖上一轮报告。
- 预注册 gate 和辅助指标要分开写，避免把探索性指标误写成 confirmatory evidence。
- QA 文件（`QA_<cycle>_<phase>.md`）：每个阶段开始前在对应目录新建，讨论确认后推进。Cycle 收尾时将 QA 中确认的协议决定、修正记录、gate 定义等关键结论内化到该阶段目录对应的增量更新文件（见步骤 8 的内化表格）；QA 文件本身保留原目录，不删除。**例外：Next Cycle Seed 阶段使用 `next_cycle/QA.md`（覆盖维护）而非 `QA_<cycle>_<phase>.md`；内化目标是 `open_questions.md` + `history_questions.md`。**
- 修改历史 cycle 时，应说明是补充 provenance、修正错误，还是复制最新内容回填。
- 重要结论必须同时保留英文/中文版本时，应让两者内容一致；若只维护一个版本，应在 README 或文件头说明。
- 当前工作面中的 Markdown 文件应同步维护 `-ch.md` 中文阅读版，例如 `current_experiment_design.md` 对应 `current_experiment_design-ch.md`，`README.md` 对应 `README-ch.md`。中文版用于阅读，英文/原始版和 source path 用于 provenance。


## 维护与覆盖的边界

本工作区默认采用**持续维护**，不是每轮一次性覆盖。维护的含义是：保留仍然成立的内容，修改已经改变的内容，新增缺失的内容，删除或移入历史归档已经失效的内容。任何时候都应让当前工作面保持完整、自洽、可直接读取。

### 持续维护的内容

以下目录和文件应被持续维护，而不是整目录覆盖：

- `related_work/`：维护当前最新文献脉络。已有仍成立的综述和 source matrix 应保留；新增文献、删改过时判断、标注被实验结果推翻的论点。
- `theory/`：维护当前理论框架和指标定义。已稳定的定义保留；被新实验修正的机制假设要改写；新增指标需要补充计算方式和解释边界。
- `exp/`：维护当前实验设计，分两层维护节奏。`experimental_methodology/`（模型/数据/训练/评测/几何协议）是稳定层，只在实现细节真的变化时才编辑，不随每个 cycle 重写；`current_experiment_design.md` 等本轮文件，新 cycle 只修改发生变化的模型、数据、配对规则、acceptance criteria 和 required artifacts。
- `code/`：四文件结构维护。`current_code_structure.md` 持续追加；`code_evolution.md` append-only；`current_code_brief.md` 每 cycle 覆盖；`QA_<cycle>.md` 每 cycle 新建后保留。
- `result/`：维护当前最新结果解释。仍成立的结论保留；新结果改变了判断时要改写；新增表格和 artifact 链接应补入；被推翻的 claim 应移到历史说明或 disallowed claim。
- `lecture/`：两文件持续维护。`current_paper_argument.md` 每 cycle 更新论证骨架（新增支撑点、调整论证空缺、更新证据链）；`lecture_evolution.md` append-only，追加每 cycle 的论证变化节；`QA_<cycle>_lecture.md` 每 cycle 新建后保留。
- `next_cycle/`：三文件结构维护。`open_questions.md` 始终只保留 open 问题（resolved 的移入 `history_questions.md`）；`history_questions.md` append-only，每轮追加一节；`QA.md` 覆盖维护，每次进入 Next Cycle Seed 阶段时重写。
- `readme.md`：维护工作区规则。规则改变时增量修改，不用重写整份说明。
- `paper_drafts/`：维护论文草稿。章节、论点和引用按最新证据增量修改；旧投稿版本或大改前版本应另存归档。

### 历史保留的内容

以下目录原则上是历史记录或原始产物，不作为当前最新工作面直接覆盖：

- `research_cycles/`：只含 evolution files，append-only。每个 cycle 收尾时追加新的一节，已有的历史小节不改写。
- `local_experiment_results/`：实验 artifact 和 copyback 结果。原始 CSV/JSON/figure 应保留；若重新生成，应保留来源说明，必要时用版本号区分。
- `code/code_evolution.md`：append-only，不改写历史条目。`QA_<cycle>.md` 文件讨论完成后保留原文，不删除。
- `literature_library/`：文献库。新增文献和摘要，不删除仍可能有引用价值的材料。

### 可以覆盖或重建的内容

覆盖只适用于**派生视图**，并且必须能从保留的源文件重新生成：

- 自动生成的索引文件、目录 README、artifact index。
- 从 CSV/JSON 生成的摘要表，前提是原始 CSV/JSON 已保留。
- 当前 latest 视图中明确由某个历史 cycle 汇总而来的拷贝件，前提是更新时保留 provenance，并且不是手工批注的唯一来源。
- 临时中间文件、缓存文件、明显过时且已有归档副本的草稿副本。

覆盖前应检查三点：

1. 源文件或历史版本是否已经保留。
2. 新内容是否完整覆盖旧内容承担的职责。
3. 是否需要在 README、文件头或 changelog 中说明替换原因。

## 整理记录（Changelog）

### 2026-06-16：架构重构

之前的整理任务（从最新 cycle 抽取最新五大目录、补入 geometry 指标定义、为每个目录加 README）已完成。本次重构解决的是更根本的问题：当时的工作面"讲了最新结论，但不是完整内容"——尤其 `exp/` 只记录"接下来做什么+gate"，完全没有模型/数据/训练/评测/几何协议的实现细节，这是 MATH500/GSM8K 测量 bug 拖了很久才被发现的直接原因。本次变更：

1. 新增 `exp/experimental_methodology/`（`model_and_data.md`、`training_protocol.md`、`evaluation_protocol.md`、`geometry_protocol.md`），把稳定的实现细节从"本轮要做什么"中分离出来，事实素材直接从 `experiments/opd_sft_h1/` 的代码和配置中核实抽取，未确认的细节明确标注待确认，不编造。
2. `theory/geometry_metric_definitions.md` 和 `related_work/current_source_matrix.md` 各增补一段设计动机/综述导言，回答"为什么是这些指标/这组论文"。
3. 新增 `research_cycles/*_evolution.md`（5 份 + 中文版），按板块记录 Cycle 03→05 的增量演进，不要求逐轮全文复述；原有按 cycle 的完整快照文件夹后续已删除（见下方 2026-06-16 第二次重构记录）。
4. 删除 `legacy_archive/`（内容已被 `literature_library/` 取代）。
5. 同步本文件与 `readme-ch.md`，消除此前的双语内容分叉。

### 2026-06-23：code/ 目录重构

`code_handoffs/` 和 `code_handins/` 合并入 `code/`。`code/` 由四类文件构成：`current_code_structure.md`（当前代码库全貌，持续追加）、`code_evolution.md`（跨 cycle 演进 + 执行 handin，append-only）、`current_code_brief.md`（当前任务说明，每 cycle 覆盖）、`QA_<cycle>.md`（每 cycle 代码对话启动前的 QA，保留不删）。原 `code_handoffs/` 和 `code_handins/` 目录作为历史存档保留，不再作为活跃工作面引用。

### 2026-06-16：result 结构重构 + research_cycles 快照删除

`result/` 重构为三文件结构（`incremental_log.md`、`current_picture.md`、`claims_allowed.md`），取代之前的"单 cycle 视角文档 + 修订横幅"模式。`research_cycles/` 的按 cycle 完整快照文件夹（`cycle_03/`、`cycle_04/`、`cycle_05/`）全部删除，只保留 5 份 `*_evolution.md` 文件。关键 artifact（FINDING 文档、RESULTS 表、source_notes）迁移到 `local_experiment_results/<cycle_name>/` 和 `related_work/source_notes/`。
