# 当前实验设计目录

```yaml
view_type: current_workspace_section_ch
section: exp
status: active
current_cycle: cycle_05_matched_control_id_ood
source_file: README.md
last_organized: 2026-06-16
```

本目录是当前实验设计入口。它维护最新 implementation-facing 设计；历史 per-cycle 计划仍保存在 `../research_cycles/` 和 `../code_handoffs/` 中。分两层：`experimental_methodology/`（稳定、跨 cycle 的实现细节）和 `current_*.md`（只记录本轮设计和 gate）。

## 当前文件

| 文件 | 作用 | 主要来源 |
|---|---|---|
| `experimental_methodology/model_and_data-ch.md` | 稳定层：模型身份、chat/think 格式、训练/探针数据。 | 代码核实（`experiments/opd_sft_h1/`） |
| `experimental_methodology/training_protocol-ch.md` | 稳定层：OPD/SFT 训练机制、LoRA、vLLM colocate、进程隔离。 | 代码核实 |
| `experimental_methodology/evaluation_protocol-ch.md` | 稳定层：ID/OOD 任务映射、chat-template/thinking 规则、匹配规则。 | 代码核实 + FINDING_05 文档 |
| `experimental_methodology/geometry_protocol-ch.md` | 稳定层：GetSlice 数据采集协议（层/模块/探针）。 | 代码核实 |
| `current_experiment_design.md` / `current_experiment_design-ch.md` | 当前 Cycle 05 实验设计（仅本轮层）。 | Cycle 05 experiment design |
| `current_acceptance_criteria.md` / `current_acceptance_criteria-ch.md` | 当前 Cycle 05 Gates A-F。 | Cycle 05 acceptance criteria |
| `current_code_handoff.md` / `current_code_handoff-ch.md` | 当前交给代码侧执行的 handoff 视图。 | Cycle 05 code handoff archive |

## 维护规则

- `experimental_methodology/` 只在实现细节真的变化时才编辑，不随每轮重写。
- `current_*.md` 是按轮持续维护的，不是每轮整目录覆盖。
- `../code_handoffs/` 中的 per-cycle handoff 是历史快照，不应覆盖。
- 新 cycle 开始时，先更新本目录的 current 文件，再把当时版本归档到对应 cycle 目录。
