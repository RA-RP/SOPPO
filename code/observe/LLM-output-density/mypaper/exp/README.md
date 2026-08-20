# Current Experiment Design

```yaml
view_type: current_workspace_section
section: exp
status: active
current_cycle: cycle_05_matched_control_id_ood
last_organized: 2026-06-16
```

---

## ⚠️ 协议设计强制规则：先查官方，再查社区，被采纳则关联 related_work

**在设计任何评测协议或训练格式之前，必须按以下顺序查阅：**

1. **官方技术报告 / 官方文档**：模型官方 arXiv 论文、HuggingFace model card、官方评测脚本。
   例：MMLU 最终发现 Qwen3 官方协议是 base model mode（arXiv 2505.09388），而不是 chat template。

2. **同系列 / 类似大小的其他模型**：如果目标模型没有明确说明，看同系列上一代（Qwen2.5）或类似规模模型是怎么做的。

3. **社区实践**：lm-eval-harness 的 GitHub Issues、Hugging Face Discussions、leaderboard 的评测说明。
   例：lm-eval-harness Issues #3405/#3576/#3322 均确认 MMLU no-chat-template 是正确协议。

**不要先写代码、先试参数，再靠结果反推。** 查阅过程中记录来源（论文、issue 链接、model card 链接），写入对应的协议文档。

### 被采纳的来源必须关联到 related_work

`../related_work/` 是整个项目的文献库。**如果某个查询来源最终被采纳进协议或设计，必须将该来源添加到 `../related_work/current_source_matrix.md`。** 具体要求：

- 写明来源标识（arXiv ID、GitHub issue 链接、blog URL 等）
- 写明该来源支持了哪个具体协议决策（例如："Qwen3 arXiv 2505.09388 → MMLU base model mode"）
- 写明该来源与论文论点的关系（是直接证据、方法基础、还是评测对齐依据）

这样做的目的：协议设计和论文论点之间有可追溯的 citation 链。来源只是"查过"但没有采纳进设计的，不需要强制加入 related_work，但若对理解模型行为有帮助可选择性记录。

这条规则来源于 Cycle 05 的教训：MMLU 尝试了 4 种配置（generative+think on、generative+think off、loglikelihood+chat+think on、loglikelihood+chat+think off），得分从 0.00 到 0.47 都出现过，最后才发现官方早有明确说法。提前查阅可以节省大量摸索时间。

---

This directory is the current experiment-design entry point. It is maintained as the latest implementation-facing design, while historical per-cycle plans remain in `../research_cycles/` and `../code_handoffs/`. It has two layers: `experimental_methodology/` (stable, cross-cycle implementation details) and the `current_*.md` files (this cycle's design and gates only).

## Current Files

| File | Role | Main Source |
|---|---|---|
| `experimental_methodology/model_and_data.md` | Stable: model identity, chat/think format, training/probe data. | Code-grounded (`experiments/opd_sft_h1/`) |
| `experimental_methodology/training_protocol.md` | Stable: OPD/SFT training mechanics, LoRA, vLLM colocate, process isolation. | Code-grounded |
| `experimental_methodology/evaluation_protocol.md` | Stable: ID/OOD task mapping, chat-template/thinking rules, matching regimes. | Code-grounded + FINDING_05 docs |
| `experimental_methodology/geometry_protocol.md` | Stable: GetSlice data-collection protocol (layers/modules/probes). | Code-grounded |
| `current_experiment_design.md` | Active Cycle 05 experiment design (this-cycle layer only). | Cycle 05 experiment design |
| `current_acceptance_criteria.md` | Active Cycle 05 gates A-F. | Cycle 05 acceptance criteria |
| `current_code_handoff.md` | Current implementation-facing handoff view. | Cycle 05 code handoff archive |

## Maintenance Notes

- `experimental_methodology/` is edited in place only when the underlying implementation actually changes — it is not rewritten per cycle.
- Current design files (`current_*.md`) are maintained incrementally per cycle.
- Per-cycle handoff snapshots in `../code_handoffs/` are historical and should not be overwritten.
- When a new cycle starts, update these current files and then archive a snapshot under the relevant cycle folder.
