# QA: Cycle 05 — Result Phase

```yaml
phase: result
cycle: cycle_05_matched_control_id_ood
status: discussion_complete
created: 2026-06-17
discussion_closed: 2026-06-17
internalize_to: result/incremental_log.md
```

本 QA 文件是 result 阶段开始前的审查清单。每个问题必须回答后，才能进行结果文件的写入或修改。
问题由 Claude 提出，用户回答，双方讨论确认合理后，进入实际结果整理工作。

> **状态（2026-06-17）：** 所有问题已回答并讨论完毕。Cycle 收尾时将关键结论内化至 `result/incremental_log.md`。

---

## Q1：cycle05 的范围边界

**问题：** cycle05 的实验范围是什么？目前的理解是"仅 eval 协议审查（无新训练）"，还是包含了新的训练运行？

- a) 仅 eval 协议审查：对 Cycle 04 已有 checkpoint 重新评测，修正三个 eval bug（GSM8K chat template、MATH500 dollar-span、MMLU collapse）。
- b) eval 审查 + 至少一次新训练运行。
- c) 其他（请说明）。

**你的回答：**
a。Cycle05 实际上是 cycle04 的实验修补版本，只负责 eval 协议审查。需要确认三轴：① model 是否开启 think、② 当前测试是否都使用 chat 格式、③ token 截断使用多少比较合适。

**📌 讨论结论（2026-06-17）：** ✅ 已确认。Cycle05 = Cycle04 的 eval 协议修复版本，不包含新训练。三轴定义：
1. **think 轴**：`enable_thinking=True / False`
2. **chat 格式轴**：`--apply_chat_template` 是否开启
3. **token 截断轴**：`max_gen_tokens` 设置

注意：三轴对生成任务（GSM8K/MATH500）与 loglikelihood 任务（MMLU）的作用机制不同，exp 阶段需分类设计协议。

---

## Q2：原始结果文件的位置

**问题：** 修正后的 GSM8K 和 MATH500 评测（4 个模型已有修正数字）的原始输出文件存在哪里？

目前 `local_experiment_results/cycle_05_matched_control_id_ood/` 里只有 3 个 FINDING 文件和一个 README 占位符，没有 RESULTS_05.md 或原始 JSON/CSV。

- a) 原始输出在某个本地实验路径，请告知路径。
- b) 原始输出还不存在——修正数字是手工整理进去的，尚无完整实验日志。
- c) 原始输出在其他位置（请说明）。

**你的回答：**
b。现在代码可能还不完整，现在只负责设计；代码对话会让代码侧将结果路径反馈到 `code_handoffs/` 中的文件夹。

**📌 讨论结论（2026-06-17）：** ✅ 已确认，路径规则已厘清：
- `code_handoffs/`：研究侧→代码侧的**交接 brief**（实验计划、接口约定），遵照 readme 设计
- `local_experiment_results/cycle_05_matched_control_id_ood/`：代码跑完后的**结果 copyback**（eval 表、JSON、figures）

两者不混用。Code handoff 文件中需明确说明结果 copyback 的目标路径。

---

## Q3：剩余 4 个模型的状态

**问题：** `opd_lmbda1`、`sft_n256`、`sft_n512`、`sft_n1024` 四个模型在修正协议下是否已经重新评测？

- a) 已经重新评测，结果在 ___（路径）。
- b) 尚未重新评测；cycle05 result 阶段应记录"待完成"并列出 blocking 原因。
- c) 决定不在 cycle05 内完成，留到 cycle06。

**你的回答：**
b。Cycle05 还没有真正运行完整。现在的状态是 cycle04 完成 result 刚开始 cycle05，但 cycle05 没有按正常流程走，而是先进行了预测试得到了一个 pre-result。现在还在 cycle05 的开始阶段，下一步是走 next_cycle。

**📌 讨论结论（2026-06-17）：** ✅ 已确认，解读 B 正确：
- Cycle05 实际上未正式运行，当前 result/ 中的 cycle05 内容属于"预测试/协议审查预记录"，不是正式结果
- 现有 3 个 FINDING 文件和 4 模型修正数字原地保留，不修改，不删除
- result/ 的正式写入在完整 cycle05 实验跑完后再执行
- 当前阶段推进顺序：result QA 确认 → next_cycle 阶段 → exp 设计 → 执行 → 正式 result 写入

---

## Q4：MMLU collapse 的解决状态

**问题：** Finding 05c（MMLU 在 chat template 下全部塌陷到 0.2295）目前是什么状态？

- a) 已找到原因和修复方案，有新的有效 MMLU 数字。
- b) 已知原因（thinking mode 干扰 loglikelihood），尚无有效修复数字，OOD-lite 仍不可用。
- c) 原因仍未确认，需要进一步调查。

**你的回答：**
c。还不能完全确认，需要按 Q1 中提出的三轴进行设计检查。优先尝试 `mmlu_generative` + `enable_thinking=True` + `think_end_token`。

**📌 讨论结论（2026-06-17）：** ⚠️ 已确定修复方向，待实验验证：
- MMLU collapse（0.2295 随机水平）是 Qwen3 thinking 模型在 loglikelihood 模式下的已知问题（lm-eval-harness issue #3576 有明确记录）
- **修复方向（优先尝试）：** 换用 `mmlu_generative` task + `enable_thinking=True` + `think_end_token`（模型走 CoT，从 think block 结束后提取字母答案）
- `\boxed{}` loglikelihood 不适用于 MMLU——MMLU 答案是单字母 A/B/C/D，非数学表达式
- 此修复方案需在 exp 设计阶段正式写入评测协议，结果纳入 cycle05 正式结果

---

## Q5：result 阶段的产出清单

**问题：** 本轮 result 阶段应产出哪些文件/更新？请确认下列清单是否完整，或者需要添加/删减：

1. `result/incremental_log.md`：追加 cycle05 增量条目（当前标注"in progress"，需要封口）。
2. `result/current_picture.md`：更新当前可用数字（补入修正后完整表格，标清哪些仍是 broken）。
3. `result/claims_allowed.md`：按修正结果更新允许/不允许的 claim。
4. `local_experiment_results/cycle_05_matched_control_id_ood/RESULTS_05.md`：新增正式的结果汇总表（目前缺失）。
5. 其他（请说明）。

**你的回答：**
1/2/3/4 都需要。现在只是 pre-result，实际上还没有完整的 cycle，在真实的 cycle 之后正常填入。

**📌 讨论结论（2026-06-17）：** ✅ 已确认，四项产出均需要，但均在正式 cycle05 实验完成后填入：
1. `result/incremental_log.md`：追加 cycle05 正式条目（当前 "in progress" 条目保留，收尾时封口）
2. `result/current_picture.md`：更新完整修正数字表格（所有 8 模型，标清 ✅/❌ 状态）
3. `result/claims_allowed.md`：按修正结果更新 claim 边界，加入 cycle04 gate 回顾结论
4. `local_experiment_results/cycle_05_matched_control_id_ood/RESULTS_05.md`：新增正式结果汇总表

---

## Q6：当前 result 文件的有效性

**问题：** 现有 `incremental_log.md`、`current_picture.md`、`claims_allowed.md` 里 cycle05 相关的内容——你认为有哪些是正确可保留的，哪些是需要修正或删除的？

**你的回答：**
这些还是待考量，现在需要完整执行 cycle05 才能判断，暂时保留。

**📌 讨论结论（2026-06-17）：** ✅ 已确认，保守策略合理：
- 现有 cycle05 相关内容（3 个 FINDING、4 模型修正数字、incremental_log 中的"in progress"条目）作为"预记录"保留，不动
- 完整 cycle05 实验完成后再对照验证并按需更新

---

## Q7：gate 判定

**问题：** cycle05 是否有预注册的 gate 判定标准？如果有，请说明 cycle05 的成功/失败/无法判断条件各是什么；如果没有预注册，这次 result 阶段要不要补一个事后 gate？

**你的回答：**
cycle05 是 cycle04 的工程修改版，重点是对齐测试和输出。gate 保留 cycle04 的 gate，并在完成工程修改后重新判断 cycle04 gate 是否真的被拒绝。如果合理就不需要在 cycle06 推进；不合理则需要设计 cycle06 检测 gate04 的问题。

**📌 讨论结论（2026-06-17）：** ✅ 已确认（融合方案）：

**Cycle05 工程 gate（新增）：**
- Gate A05：所有 8 个模型在修正协议下完成重新评测（三轴对齐：think/chat/token）
- Gate B05：MMLU collapse 问题有明确结论（已修复 / 已解释并记录为 limitation）
- Gate C05：修正后数字自洽，无新 scoring artifact

**Cycle04 Gate 回顾（纳入 cycle05 result 解释，不单开 cycle）：**
用修正后数字重新判断 Cycle04 Gate B/C/D 是否改变：
- 若改变 → 直接在 cycle05 result 里更新结论
- 若不变 → 记录确认，cycle06 设计新实验应对

---

*讨论完毕（2026-06-17）。Cycle 收尾时将 Q1/Q4/Q7 的关键结论内化至 `result/incremental_log.md`。*
