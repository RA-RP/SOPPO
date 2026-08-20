# 当前 Cycle 08 意图

```yaml
artifact_type: next_cycle_intent_ch
cycle: cycle_08_h_opd_vs_sft_comparison
status: ready_for_experiment_design
created: 2026-06-27
source_qa: next_cycle/QA.md
source_file: current_cycle08_intent.md
write_owner_next: Experiment Design
```

本文件是 Next Cycle Seed → Experiment Design 的 Cycle 08 交接文件。

---

## Seed 决策

Cycle 08 是 **H-OPD vs SFT 对比 cycle**。

Cycle 07 在 Qwen3-4B-Base + Math-CoT-20k（LoRA，think-format）上建立了 SFT 参考轨迹：
A07 PASS、B07 FULL PASS、step_20 三重共定位几何信号、OverlapLift < 1 单调上升。
Cycle 08 在同样设置上训练 OPD arm，进行首次直接对比。

> **一个 OPD arm，一个 lambda，与 Cycle 07 SFT 相同的模型 + 数据 + LoRA 配置。**

---

## OPD Arm 设计

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-4B-Base（与 Cycle 07 相同的 checkpoint） |
| 训练数据 | Math-CoT-20k（与 Cycle 07 相同） |
| 训练格式 | Think-format LoRA（与 Cycle 07 相同） |
| LoRA 配置 | 与 Cycle 07 相同的 rank/alpha/目标模块 |
| Checkpoint grid | 与 Cycle 07 相同的 10 个点：step_000…624 |
| OPD 正则化 | 单个 arm，一个 best-guess lambda——**不做 sweep** |
| Lambda 选择 | Experiment Design 根据先前 cycle（lmbda0.5 或 lmbda1.0）和理论层选择，必须明确说明理由 |

**为什么只跑单个 arm：** OPD 训练开销大。在 4B+Math-CoT 制度下确认可行性之前，
不做 lambda sweep。

---

## 评测协议

**生成式任务与 Cycle 07 完全相同：**
- MATH500：无 chat template，think-format 生成，max_tokens=4096，`\boxed{}`-aware scorer
- as-run acc 为主要指标（cap-robust 已确认：0.620@4096 ≈ 0.617@24576）
- **Cycle 08 不更改 max_tokens**——截断已被证明在计算上是合理的

**每个 checkpoint 额外报告：**
- 非终止率（对标 Cycle 07 轨迹表格）
- Finisher acc（correct/finisher count）——用于事后分解分析

**OOD-lite——对 OPD arm 做完整轨迹跟踪（全部 10 个 checkpoint）：**
- GPQA-Diamond（无 chat template，loglikelihood）
- MMLU-Pro（无 chat template，loglikelihood）
- 理由：C07 发现 SFT arm MMLU-Pro 轻微退化（final −2.2 SE）；Gate C08 测试 OPD arm
  是否能更好地保留 MMLU-Pro；需要完整轨迹才能排除 timing 效应

**Geometry——与 Cycle 07 相同的协议：**
- effective_rank、xs_log_spectrum_gap，全部 10 个 checkpoint
- OverlapLift（down_proj、attention 模块），全部 10 个 checkpoint
- UV/sketch artifact 必须保存（与 Cycle 07 相同；OverlapLift 计算所需）
- Drift from base checkpoint（step_000）——与 Cycle 07 相同的负对照参考

---

## Gate 定义（seed 级；Experiment Design 必须预注册确切阈值）

Gates 按重要性排序：B08（阻塞性）> C08 > D08 > A08。

### Gate B08 — 能力保留（主 gate，阻塞性）

**问题：** OPD arm 的 final MATH500 是否不低于 SFT（非劣效性），并且是否避免了
step_20 级别的能力下陷？

子条件：
1. **Final 非劣效性：** OPD arm final MATH500 ≥ SFT arm final − 1SE。
   （参考：0.680 − 0.020 = 0.660；Experiment Design 使用实测 SFT step_624 SE 设定确切值。）
2. **下陷严重程度：** OPD arm 最低 finisher acc > SFT step_20 finisher acc（0.594）。
   或：若 OPD 存在下陷，其幅度更小或发生在不同 step。

**PASS：** OPD 能力匹配 SFT，且下陷动态不同。支持"OPD 在主要任务上不劣于 SFT"的核心主张。
**FAIL：** OPD 显著低于 SFT——在推进前先排查 lambda、学习率或训练步数问题。

---

### Gate C08 — MMLU-Pro 保留（第二）

**问题：** OPD arm 在 final checkpoint 的 MMLU-Pro 退化是否小于 SFT arm？

**通过条件：** OPD arm MMLU-Pro(step_624) − MMLU-Pro(step_000) >
SFT arm MMLU-Pro(step_624) − SFT arm MMLU-Pro(step_000)（即退化更小）。

**PASS：** 支持"OPD 正则化在等效训练条件下比 SFT 更好地保留通用知识"的主张。
**FAIL：** OPD 对 MMLU-Pro 的退化程度与 SFT 相当或更严重——该保留轴不成立。

---

### Gate D08 — Geometry 模式（第三）

**问题：** OPD arm 在其能力低点的 geometry 是否不同于 SFT step_20 三重共定位？

比较内容：
- OPD arm 是否出现三重共定位（argmin acc = argmax rank = argmin xs_gap）？若有，在哪个 step？
- Effective rank 轨迹：OPD 是否与 SFT 一样在下陷时出现峰值？
- xs_gap 轨迹：OPD 是否与 SFT 一样在下陷时出现最小值？

**此 gate 为诊断性，不阻塞。** 记录 OPD geometry 轨迹并与 Cycle 07 SFT 轨迹表格对比。

---

### Gate A08 — OverlapLift 比较（第四，解读最复杂）

**问题：** OPD OverlapLift 轨迹是否与 SFT 系统性地不同？

SFT arm：OverlapLift < 1 全程，单调上升（down_proj 0.48→0.74；attention 0.04–0.28）。
On/off-principal 解读从 Cycle 07 延后到 Cycle 08。

Cycle 08 需要：
1. 在全部 10 个 checkpoint 测量 OPD arm OverlapLift（与 Cycle 07 相同的模块）。
2. 建立或近似 LoRA-aware 参考基准用于归一化。Experiment Design 必须在运行前确定归一化方案。
3. 比较 OPD vs SFT OverlapLift 轨迹；注意 OPD 是否上升更快、更慢或到达不同平台。

**此 gate 为诊断性，不阻塞。** 目标：在 Cycle 08 中通过解决 LoRA-aware 参考问题，
对两个 arm 都给出 on/off-principal 结论。

---

## SFT 参考轨迹（来自 Cycle 07）

所有 Cycle 08 比较以 Cycle 07 SFT 轨迹为参考对照。**Cycle 08 不重新运行 SFT arm。**

关键参考点：
- **下陷 checkpoint：** step_20（finisher acc 0.594，非终止率 6.8%，argmax rank，argmin xs_gap）
- **恢复 checkpoint：** step_160（首次超过 base 的步骤）
- **峰值 checkpoint：** step_320（MATH500 0.698）
- **Final checkpoint：** step_624（MATH500 0.680，非终止率 ~50%，finisher acc 0.98）
- **SFT MMLU-Pro @ final：** 相对 step_000 约 −2.2 SE

---

## 遗留项——已关闭

| 项目 | 决策 |
|---|---|
| C06 PARTIAL（geometry copyback） | **已关闭。** 不再追踪。 |
| A06b UNDETERMINED（base stop-fix） | **已关闭。** 不再追踪。 |

---

## Experiment Design 必须明确的决策

1. **Lambda 选择：** OPD arm 使用哪个 lambda？从先前 cycle（lmbda0.5 或 lmbda1.0）
   或理论层给出理由，写入 `exp/current_experiment_design.md`。
2. **LoRA-aware OverlapLift 参考：** 如何归一化以得出 on/off-principal 结论？
   选项：random-LoRA baseline 运行、解析界、或 OverlapLift 文献中的替代归一化。
   在运行之前决定。
3. **B08 确切阈值：** 使用 Cycle 07 实测数字预注册非劣效性阈值和下陷严重程度条件。
4. **C08 确切阈值：** 预注册 MMLU-Pro 保留条件。

---

## Cycle 08 后的路由

- **若 B08 + C08 均通过：** 强力证据——OPD 能力匹配 SFT 且 MMLU-Pro 保留更好。
  核心论文主张可行。Cycle 09 聚焦 geometry 机制和论文写作。
- **若 B08 通过、C08 失败：** OPD 能力匹配但 OOD 保留不成立；论文须收窄范围。
- **若 B08 失败：** OPD 劣于 SFT，在进入 Cycle 09 前先排查 lambda。
- **D08 和 A08：** 无论 B08/C08 结果如何，记录 geometry 和 OverlapLift 发现，
  推进 LoRA-aware OverlapLift 解读。

---

## 下一步阅读顺序

Experiment Design 应按以下顺序阅读：

1. `next_cycle/current_cycle08_intent.md`（英文版，主要参考）
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `theory/current_theory_update.md`（OPD lambda 理由）
5. `exp/experimental_methodology/training_protocol.md`
6. `exp/experimental_methodology/evaluation_protocol.md`
7. `exp/experimental_methodology/geometry_protocol.md`
8. `local_experiment_results/cycle_07_base_sft_trajectory/run_01/RESULTS_07.md`

然后撰写 `exp/current_experiment_design.md`、`exp/current_acceptance_criteria.md`
以及 cycle08 代码交接文件。
