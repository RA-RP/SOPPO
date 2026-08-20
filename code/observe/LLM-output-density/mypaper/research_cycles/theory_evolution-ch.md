# 理论演进（Cycle 03 → 05，中文阅读版）

```yaml
view_type: cross_cycle_evolution_log_ch
source_file: theory_evolution.md
last_organized: 2026-06-16
```

> 本文件是 `theory_evolution.md` 的中文阅读版，内容与源文件保持一致。当前稳定的理论和指标定义在 `../theory/current_theory_update-ch.md` 和 `../theory/geometry_metric_definitions-ch.md`。

## Cycle 03 —— 起点

全文来源：`cycle_03_trl_opd_like/02_theory_design/theory_protocol_03.md`、`my_existing_method.md`。

Cycle 03 定义了本项目的核心理论对象：activation-conditioned operator `A_t = W_t L_t`（权重矩阵通过激活诱导的白化变换 `L_t` 来观察，而不是看裸权重空间）。`my_existing_method.md` 是用户自己对同一想法最原始的非正式推导——也就是现在 `geometry_metric_definitions-ch.md` 的直接前身。主指标是**spectral gap stability**；辅助指标是 spectral drift、effective rank、X-S geometry gap，可选 CKA。公平性锚点是**matched MATH500 ID gain**，并要求显式的 `SFT-small-LR` 对照来排除"OPD 只是更新更少"的解释。核心假设（K1）：

> 在 matched MATH500 gain 下，OPD 比 continued SFT 更好地保留 General OOD，因为它用更小的 activation-conditioned principal geometry 扰动达到了同样的 ID 提升。

**Cycle 03 就已经存在的设计-执行差距**：实际跑的最小化 TRL 实验（`RESULTS_03_v2.md`）匹配的是 **GSM8K** gain，不是 MATH500，几何测量也只在**第 14 层**，不是本理论文档定义的完整多探针设计。这个差距与其说是错误，更像是有意为之的 feasibility-first pilot——但它预示了后面才被发现的测量问题（见 `result_interpretation_evolution-ch.md` 的 Cycle 04/05 条目）。

## Cycle 03 → Cycle 04：理论没改写，但结果迫使重新解读

全文来源：`cycle_04_opd_stability_gain/02_theory_design/theory_update_04.md`（= 当前 `../theory/current_theory_update-ch.md`）。

理论实质内容在 Cycle 04 没有变——K1 和 matched-gain-anchor 框架都保留了。变的是*结果*，它迫使重新解读：matched-gain anchor 被证明会被 `theta0` 的饱和度混淆（一个不那么饱和的 `theta0` 会把 matched SFT 对照逼到几乎没训练过的 checkpoint，这种 checkpoint 自然"稳定"）。这是实证发现的，不是 Cycle 03 理论文档预测出来的。Cycle 04 还产出了 `MATH500=0` 抽取假象的发现——这是评测有效性发现，不是理论修订，但它直接促成了 Cycle 05 引入 NuminaMath-test 作为更干净的 ID 轴。

## Cycle 04 → Cycle 05：正式修订为"K1-base-aware"

全文来源：`cycle_05_matched_control_id_ood/02_theory_design/theory_update_05.md`（= 当前 `../theory/current_theory_update-ch.md`）。

这是本项目第一次**显式、预注册**地针对证据修订理论（Cycle 04 那次是解读层面的重新框定，这次是直接改写假设）。新增的显式变量：`base_performance`（headroom 门）、`theta0_saturation`、`id_performance` 与 `ood_retention`（之前都混在"GSM8K"里）、`training_budget`、`geometry_movement`（现在要结合 ID learning 一起读，不是看 raw drift）。修订后的假设：

> K1-base-aware：只有当所选 base/task/data 设置在未训练 base 之上展现出可测量的 headroom 后，OPD-like 训练才能和 SFT 进行有意义的比较。在有 headroom 的前提下，OPD-like 训练只有在可比的 ID learning 或可比的训练预算下更好地保留 OOD 表现，才算更稳定——不能只看可比的 GSM8K gain。

为什么：Cycle 04 加入 base model 后发现，未训练 base 与 `theta0` 持平，在 ID 上还高于 OPD——暗示可能根本没有可供测试提升的 headroom。Falsifiers 也第一次被正式列出（见源文件），不像 Cycle 03/04 那样隐含。**这次修订的实证前提（无 headroom）后来被 Cycle 05 内部的一次测量协议修复部分推翻**——见 `result_interpretation_evolution-ch.md` 的 Cycle 05 条目；**理论文档本身还没有被改写**以反映这次推翻，这是留给下一个更新 `theory/` 的人的开放项。
