# QA: Cycle 05 — Exp Phase

```yaml
phase: exp
cycle: cycle_05_matched_control_id_ood
status: discussion_complete
discussion_closed: 2026-06-17
created: 2026-06-17
internalize_to: exp/current_experiment_design.md
```

本 QA 文件是 exp 阶段开始前的审查清单。
目标：在正式覆盖重写 `current_experiment_design.md`、`current_acceptance_criteria.md` 以及更新 `experimental_methodology/evaluation_protocol.md` 之前，确认三轴协议的具体参数、模型路径和 code handoff 格式。

---

## Q1：exp 文件的修改范围

**问题：** `current_experiment_design.md` 目前仍是 base-ceiling 审查框架，有大量预测试决策日志。`current_acceptance_criteria.md` 仍是旧的 Gate A-F。

以下哪个策略合适？

- a) 全部覆盖重写：`current_experiment_design.md` 改为三轴 eval 协议规格，`current_acceptance_criteria.md` 改为 A05/B05/C05 + cycle04 gate 回顾标准
- b) 只在顶部加 banner 重定位，正文保留作为历史背景
- c) 其他（请说明）

**你的回答：**
a — 全部覆盖重写。

**📌 讨论结论（2026-06-17）：** ✅ 已确认。`current_experiment_design.md` 覆盖为三轴 eval 协议规格；`current_acceptance_criteria.md` 覆盖为 Gate A05/B05/C05 + cycle04 gate 回顾标准。

---

## Q2：各任务的三轴参数确认

**问题：** 请对以下每个任务，确认三轴设置（think / chat / token）是否正确，或者需要修改：

**生成任务（已有 pre-result 验证）：**

| Task | think 轴 | chat 格式 | token 截断 | 提取方式 | 状态 |
|---|---|---|---|---|---|
| GSM8K | `enable_thinking=False` | `--apply_chat_template` | 3072 | `\boxed{}` aware | 已验证（4模型）|
| MATH500 | `enable_thinking=False` | `--apply_chat_template` | 3072 | `last_boxed` + `math_verify` | 已验证（4模型）|
| NuminaMath-test | `enable_thinking=False` | `--apply_chat_template` | — | hardened scorer | cycle04 已验证 |

**MMLU（待修复，优先尝试方案）：**

| Task | think 轴 | chat 格式 | token | 提取方式 | 状态 |
|---|---|---|---|---|---|
| MMLU | `mmlu_generative` + `enable_thinking=True` | `--apply_chat_template` | — | `think_end_token` + 字母提取 | 待验证 |

**其余 OOD-lite（未确认）：**

| Task | think 轴 | chat 格式 | 状态 |
|---|---|---|---|
| TruthfulQA-MC1 | ? | ? | 未确认 |
| WinoGrande | ? | ? | 未确认 |

**请逐项确认或修改。TruthfulQA 和 WinoGrande 是否也需要切换到 generative 模式，还是用 `enable_thinking=False` + loglikelihood 就够了？**

**你的回答：**
也优先切换到 generative 模式，立刻联网调研。（已查）

**📌 讨论结论（2026-06-17）：** ✅ 已确认，三轴参数最终定如下：

| Task | Axis | think 轴 | chat 格式 | token | 提取 | 依据 |
|---|---|---|---|---|---|---|
| NuminaMath-test | ID | `False` | ✅ | 3072 | last_boxed + math_verify | cycle04 已验证 |
| GSM8K | OOD | `False` | ✅ | 3072 | extract_pred + last_boxed | pre-result 4模型验证 |
| MATH500 | OOD | `False` | ✅ | 3072 | last_boxed + math_verify | pre-result 4模型验证 |
| MMLU | OOD-lite | `True` | ✅ | generative | think_end_token + 字母提取 | **待 phase-1 验证** |
| TruthfulQA-MC1 | OOD-lite | `False` | ✅ | loglikelihood | lm_eval 默认 MC | 社区先例 |
| WinoGrande | OOD-lite（宽） | `False` | ❌ 不加 | loglikelihood | lm_eval 默认 MC | 社区标准：chat template 对所有模型降分；用户确认接受，认为这是更宽泛的 OOD 信号（不同输出模式） |

WinoGrande 不加 chat template 是有意为之的格式不一致，必须在 eval 报告和 evaluation_protocol.md 中明确注明。

---

## Q3：剩余 4 个模型的 checkpoint 路径

**问题：** 以下 4 个模型还没有在修正协议下跑过，code handoff 需要确认路径。

- `opd_lmbda1`
- `sft_n256`
- `sft_n512`
- `sft_n1024`

已有的 4 个模型路径是否也需要在 handoff 里列出（供代码侧 cross-check）？

请告知：
1. 这 4 个模型的 checkpoint 路径（或者说明"代码对话里再确认"）
2. 已有 4 个模型的路径是否需要列入 handoff

**你的回答：**
1. 在代码对话中再确认
2. 是的，但可能由代码对话完成

**📌 讨论结论（2026-06-17）：** ✅ 已确认。所有 8 个模型的 checkpoint 路径在 code_handoff 中以 `[TBD — 代码对话填写]` 占位；代码对话执行后在 code_handin 中写入实际路径。

---

## Q4：`experimental_methodology/evaluation_protocol.md` 是否需要更新

**问题：** 三轴协议的发现（chat template、thinking、token 截断）是**稳定层**级别的知识，应当写入 `experimental_methodology/evaluation_protocol.md`，而不是只在本轮 exp design 里有。

目前 evaluation_protocol.md 是否已经包含 chat template 和 thinking 模式的设置说明？还是只有 cycle04 之前的内容？

- a) 需要更新 evaluation_protocol.md，把三轴框架和任务分类（生成 vs loglikelihood）写进去
- b) 不需要，evaluation_protocol.md 已经够用，只更新本轮 exp design 就行
- c) 其他（请说明）

**你的回答：**
a — 需要更新，但区分已完成和未完成；result 阶段前先通过 code_handin 合并实际实现；在 readme 中新增 code_handins/ 概念，让代码对话能读 readme 并反馈实现细节。

**📌 讨论结论（2026-06-17）：** ✅ 已确认，包含新概念：

**新增：`code_handins/` 目录**
- `code_handoffs/`（研究→代码）与 `code_handins/`（代码→研究）形成双向通道
- 代码对话执行后，在 `code_handins/<cycle_name>/` 写入：实际命令、实际路径、与 handoff 规格的偏差、发现的问题
- 流程：`code_handoff` → 代码执行 → `code_handin` → 更新 `exp/experimental_methodology/` → 写 result

**evaluation_protocol.md 更新策略：**
- pre-result 已验证的内容：直接写入
- 未经完整 8 模型验证的内容：标注 `[待 cycle05 code_handin 确认]`
- 最终确认在 result 阶段前完成

**readme 更新：** 新增 `code_handins/` 目录描述 + 代码对话应读取文件的说明 + step5/step6 的流程更新。

---

## Q5：code handoff 的格式和内容

**问题：** cycle05 没有新训练，code handoff 只需要描述"re-evaluate 8 models"。你希望 `current_code_handoff.md` 包含哪些内容？

请确认以下清单，并说明是否还有补充或删减：

1. 模型列表（8 个 checkpoint 路径）
2. 每个任务的完整 `lm_eval` 调用命令（含所有 flags）
3. 结果 copyback 目标路径：`local_experiment_results/cycle_05_matched_control_id_ood/`
4. 必须输出的 artifact 格式（JSON、stdout log、summary CSV）
5. Gate A05/B05/C05 的判定标准（代码侧需要知道跑完后应该产出什么）

**你的回答：**
是的，5 项无问题。

**📌 讨论结论（2026-06-17）：** ✅ 已确认，code_handoff 包含：模型列表（路径 TBD）、每任务完整 lm_eval 命令、copyback 目标路径、artifact 格式要求、Gate A05/B05/C05 判定标准。

---

## Q6：MMLU generative 验证策略

**问题：** `mmlu_generative + enable_thinking=True + think_end_token` 是优先尝试方案，但还没有验证。

你希望在正式 8 模型全量跑之前，先用一个小规模验证来确认这个方案可行吗？还是直接跑全量？

- a) 先用 1-2 个模型（比如 `base` + `theta0`）跑 MMLU generative，确认分数合理后再全量跑
- b) 直接跑全量（8 模型），一次性出结果
- c) 其他（请说明）

**你的回答：**
a — 先用 base + theta0 测试 MMLU generative，确认后自动进入全量跑，全程连续不需要用户介入。

**📌 讨论结论（2026-06-17）：** ✅ 已确认。code_handoff 结构化为两阶段连续脚本：
- Phase 1：base + theta0 跑 MMLU generative，检查分数 > 0.25（高于随机）
- 判断逻辑写入脚本：通过 → 自动进入 Phase 2；失败 → 停止并记录 B05 失败，不触发全量跑
- Phase 2：8 模型全量跑 GSM8K / MATH500 / MMLU generative / TruthfulQA / WinoGrande

---

*讨论完毕（2026-06-17）。结论内化至 `exp/current_experiment_design.md`。*
