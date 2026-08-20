# 下一轮种子演进（Cycle 03 → 05，中文阅读版）

```yaml
view_type: cross_cycle_evolution_log_ch
source_file: next_cycle_seed_evolution.md
last_organized: 2026-06-16
```

> 本文件是 `next_cycle_seed_evolution.md` 的中文阅读版，内容与源文件保持一致。当前稳定的 next-cycle 入口在 `../next_cycle/`。

## Cycle 03 → Cycle 04 种子

全文来源：`cycle_03_trl_opd_like/07_next_cycle_seed/next_experiment_intent.md`。

要求：提高/取消 `eval_limit`、加更多 OPD 设置/seed、用更不饱和的 `theta0`、把几何扩展到 14 层之外。

**采纳情况：** 四项全部、完整地体现在 Cycle 04 的设计里（见 `exp_design_evolution-ch.md`）。

## Cycle 04 → Cycle 05 种子

全文来源：`cycle_04_opd_stability_gain/07_next_cycle_seed/next_experiment_intent.md`。

要求：先做 base-ceiling audit；只有在那之后才测试 OPD 稳定性信号能否在 equal-budget、equal-ID、`theta0` 饱和度扫描下存活，matched-GSM8K 只作诊断；每个 ID/OOD 表必须有 base row；要有 confidence/bootstrap interval；保存 UV/sketch artifact 以备未来 principal-angle 分析；显式标记 extraction-floored 任务。

**采纳情况：**

- Base-ceiling audit：已做（Gate B），它最初的判读后来被 chat-template 修复纠正了（见 `result_interpretation_evolution-ch.md`）。
- 通过 NuminaMath-test 做 ID/OOD 轴分离：已做（Gate C 设计）。
- Equal-ID / equal-budget matching：**还没执行**——按 `../exp/current_experiment_design-ch.md` 的决策日志，在（后来被推翻的）Phase 1"设置已 headroom-limited"判断之后被降低了优先级。现在 headroom 已经被重新确立，这些 matching 规则重新进入范围，但仍待执行。
- UV/sketch artifact：**仍然没保存**——开放项，在 `../exp/experimental_methodology/geometry_protocol-ch.md` 里继续跟踪。
- Extraction-floored 任务标记：做到了，而且不止——这个习惯直接帮助抓住了 GSM8K chat-template bug。
- Confidence/bootstrap interval：部分做到（FINDING_05 文档里给了 4 个重测模型的 binomial stderr；还没扩展到剩下 4 个）。

## Cycle 05 → Cycle 06 种子

**还没写。** Cycle 05 还没收尾——`cycle_05_matched_control_id_ood/` 下的 `05_result_interpretation/`、`06_next_related_work/`、`07_next_cycle_seed/` 仍是占位 stub。现在真正紧急的下一步（用修正后的评测协议重测 `opd_lmbda1`、`sft_n256/512/1024`，再用全部 8 个模型重做 Gate B–F）本身还属于 Cycle 05 内部，不是 Cycle 06 的种子。写出真正的 Cycle 06 种子属于延后处理的"Cycle 05 进度同步"工作，本次架构整理有意不做这件事。
