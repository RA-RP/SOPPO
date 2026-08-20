# 当前下一轮入口

```yaml
view_type: current_workspace_section_ch
section: next_cycle
status: active
current_cycle: cycle_06_sft_scale_and_forgetting
source_file: README.md
last_organized: 2026-06-18
```

本目录是当前 next-cycle 入口，维护下一步要做什么。旧的 next-cycle seed 保留在 `../research_cycles/`。

## 当前文件

| 文件 | 作用 | 维护方式 |
|---|---|---|
| `open_questions.md` / `-ch.md` | 所有当前开放问题（上轮未解决的 + 本轮新增的）。已解决的问题移出到 `history_questions.md`。 | 动态增量 |
| `history_questions.md` / `-ch.md` | 逐轮记录：每轮 cycle 收尾追加一节，记录本轮解决的问题和新提出的问题。 | 按 cycle append-only |
| `QA.md` / `-ch.md` | Next Cycle Seed 对话的入口。每次进入该阶段时覆盖重写：提问 → 用户回答 → 讨论 → 内化到 `open_questions.md` + `history_questions.md`。 | 每次 cycle 覆盖 |
| `current_cycle<N>_intent.md` / `-ch.md` | 当前 cycle 种子（由 Next Cycle Seed 对话写入，由 Experiment Design 对话读取）。Experiment Design 将其内容正式写入 `exp/current_experiment_design.md` 后可删除。 | 每个活跃 cycle 一份；Exp Design 写完后删除 |

## Next Cycle Seed 对话启动顺序

1. `../readme.md`（整体规则和读写边界）
2. `../result/current_picture.md`
3. `../result/claims_allowed.md`
4. `../exp/current_acceptance_criteria.md`（如存在）
5. `open_questions.md`（当前所有开放问题）
6. `history_questions.md`（问题演进上下文）
7. 覆盖重写 `QA.md`，提出本轮问题 → 讨论 → 内化

## 当前状态

Cycle 05 于 2026-06-18 关闭。Gate A05 / B05 / C05 全部通过。Gate C04/D04 CONFIRMED FAIL（未翻转）。Cycle 06 种子已写入。
QA 之后，Cycle 06 已收窄为 **SFT-only 的可学习性与退化诊断 cycle**。

**Cycle 06 核心问题：** 在再次做 OPD-vs-SFT 比较之前，先确认普通 SFT
在明确的模型、数据、任务和 train/eval 格式下是否能带来真实的下游 / ID 增益，
以及 SFT 在什么数据规模开始导致 ID/OOD 退化。

**当前实验方向：**
- SFT-first 诊断：从未微调 student/base 出发训练 SFT，在预注册的数据规模和评测协议下确认当前模型 / 数据 / 任务 / 格式是否支持后续 OPD-vs-SFT 比较。OPD 遗忘预防训练暂时后置。

**下一步：** Cycle 06 Experiment Design 阶段。读取 `current_cycle06_intent.md`、
`exp/experimental_methodology/model_and_data.md`、
`exp/experimental_methodology/training_protocol.md`、
`exp/experimental_methodology/evaluation_protocol.md`、
`exp/experimental_methodology/geometry_protocol.md`，然后写
`exp/current_experiment_design.md` 和 `exp/current_acceptance_criteria.md`（cycle06 版）。
