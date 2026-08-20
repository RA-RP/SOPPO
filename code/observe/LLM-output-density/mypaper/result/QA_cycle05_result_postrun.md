# QA: Cycle 05 — Result 写入阶段（跑完后）

```yaml
phase: result_postrun
cycle: cycle_05_matched_control_id_ood
status: awaiting_answers
created: 2026-06-17
internalize_to: result/incremental_log.md
prerequisite_qa: result/QA_cycle05_result.md
```

本 QA 是代码跑完、拿到全部 8 模型数据之后的 result 写入审查。
前置 QA（`QA_cycle05_result.md`）在跑实验前完成，已确认范围和 gate 框架。
本文件聚焦：数据怎么解读、哪些写进文件、哪些留 provisional。

---

## Q1：MMLU 处理方式

代码对话发现 `mmlu_generative + enable_thinking=True` 方案在此硬件上得分归零（4 种配置均失败）。
实际改用 `standard mmlu loglikelihood + enable_thinking=False + 5-shot + 8科目子集`，当前得分范围 0.5696–0.5850，Gate B05 通过。

**你在代码对话中正在并行讨论此问题。**
写 result 文件时 MMLU 部分如何处理？

- a) 完全留空，等代码对话给出最终结论再填。
- b) 用当前 8 科目子集数字先占位，标注 `[provisional]`，代码对话结束后更新。
- c) 直接把当前协议（loglikelihood + no-thinking + 5-shot）作为 cycle05 确认协议，写入文件，不再等待。
- d) 其他（请说明）。

**你的回答：**
c
---

## Q2：Cycle04 Gate B 回顾 — matched control anchor

Gate B 的核心问题：修正后的 GSM8K 数字下，匹配 SFT anchor 还是 `sft_n128` 吗？

修正后 GSM8K 数字：

| Model | GSM8K | vs base (+/−) |
|---|---|---|
| base | 0.7005 | — |
| theta0 | 0.7180 | +0.018 |
| opd_lmbda05 | 0.7180 | +0.018 |
| opd_lmbda1 | **0.7346** | **+0.034** |
| sft_n128 | 0.7096 | +0.009 |
| sft_n256 | 0.6839 | −0.017 |
| sft_n512 | 0.6732 | −0.027 |
| sft_n1024 | 0.6141 | −0.087 |

关键观察：`sft_n128`（+0.009）远低于 `opd_lmbda1`（+0.034）和 `opd_lmbda05/theta0`（+0.018）。
`sft_n128` 是否仍是 OPD vs SFT 的公平 matched control anchor？请说明你的判断。

**你的回答：**
现在早就不是通过GSM8K进行匹配了，GSM8K只是一个泛化的ID而已，使用NUM-CoT的测试结果进行匹配。
---

## Q3：`sft_n1024` 的退化

`sft_n1024` 出现明显反向模式：

| 轴 | sft_n1024 | base | 差值 |
|---|---|---|---|
| NuminaMath ID | 0.3890 | 0.5516 | **−0.163** |
| GSM8K | 0.6141 | 0.7005 | **−0.086** |
| MATH500 | 0.5400 | 0.7140 | **−0.174** |
| MMLU | **0.5850** | 0.5696 | +0.015（8模型最高） |
| TruthfulQA | 0.3317 | 0.3341 | −0.002 |
| WinoGrande | **0.6322** | 0.6172 | +0.015 |

模式：数学能力全面低于 base，但 MMLU/WinoGrande 略高。

这个结果你怎么看？
- a) 这是 SFT 过拟合到 NuminaMath 分布的证据（训练越多，数学 OOD 迁移越差），应作为独立发现记录。
- b) 符合预期（n=1024 样本对 1.7B 模型已经过拟合），不需要特殊标注，正常写入结果表。
- c) 此数据点太异常，需要先 re-run 确认再写入 result。
- d) 其他（请说明）。

**你的回答：**
d 我认为这可能是训练方式的问题，我想知道现在的训练是chat格式的吗，是要求开启think格式的吗？也就是问，推训是否统一。 如果不是，这是不是很可能是模型经常不开think导致格式错误？我希望你能检查一下它现在开think的可靠程度。
---

## Q4：`opd_lmbda1` GSM8K 最高的解读

`opd_lmbda1` 是全部 8 个模型中 GSM8K 最高（0.7346），但 NuminaMath ID 低于 base（0.5348 vs 0.5516）。

| Model | NuminaMath ID | GSM8K | MATH500 |
|---|---|---|---|
| base | 0.5516 | 0.7005 | 0.7140 |
| theta0 | 0.5527 | 0.7180 | 0.7260 |
| opd_lmbda05 | 0.5392 | 0.7180 | 0.7080 |
| **opd_lmbda1** | **0.5348** | **0.7346** | 0.7140 |

在 Cycle04 的 broken 协议下，`opd_lmbda1` 的 GSM8K 是否看起来有 headroom？
如果之前是因为协议问题而显示"无余量"，那现在 opd_lmbda1 的+0.034 GSM8K gain 应该如何纳入 result 的解读框架？

- a) 这是 cycle05 的核心发现之一：OPD（lambda=1）在 GSM8K 上有 statistically meaningful gain，逆转了 cycle04 结论，应当明确标注。
- b) +0.034 的单轴差异还不够，需要结合 MATH500 和 OOD-lite 多轴才能判断，本 cycle 只记录数字，不下结论。
- c) 其他（请说明）。

**你的回答：**
b 我现在非常想知道训练是在什么格式下进行的？是chat+think吗？
---

## Q5：result 文件写入范围

综合 Q1–Q4 的回答，确认本次写入的文件和留空规则：

1. `result/incremental_log.md`：追加 cycle05 正式条目。
   MMLU 相关格子用 `[provisional — pending MMLU resolution]` 占位？
2. `result/current_picture.md`：更新 8 模型完整数字表。
   WinoGrande 需要特别注明"无 chat template，不可与其他任务直接横向比较"？
3. `result/claims_allowed.md`：更新 claim 边界，加入 Cycle04 Gate B/C/D 回顾结论。
   Gate C（OPD vs SFT OOD-lite 稳定性）依赖 MMLU，是否也标为 provisional？
4. `local_experiment_results/cycle_05_matched_control_id_ood/run_01/RESULTS_05.md`：已由代码对话写入，是否需要研究侧补充任何内容？

**你的回答（确认）：**
按当前 6 个指标写入，不等 ARC-challenge（ARC 出来后做增量更新）。
1. `incremental_log.md`：MMLU 直接写入（Q1=c），不加 provisional。sft_n1024 数字照实记录，注明 math catastrophic forgetting。
2. `current_picture.md`：8 模型完整数字表，含 sft_n1024 退化分析。
3. `claims_allowed.md`：更新 claim 边界，Gate C04 confirmed fail 但标注为 non-signal。
4. `RESULTS_05.md`：研究侧不补充（代码对话已写入）。

---

**📌 讨论结论（2026-06-17）：** ✅ 写入完成。三文件已同步更新至 Cycle 05 最终状态。ARC-challenge 结果到来时做增量追加。

*本 QA 文件已关闭。*
