# 当前验收标准（中文阅读版）

```yaml
view_type: current_latest_view_chinese
cycle: cycle_07_base_model_sft_feasibility
status: active_cycle07_gates
maintenance_rule: 中文版用于阅读，英文原始版 current_acceptance_criteria.md 用于 provenance
last_organized: 2026-06-24
```

---

# 验收标准 07：Base 模型 SFT 轨迹研究

## Gate A07 — 可行性（主门控，阻塞性）

**问题：** Qwen3-4B-Base + Math-CoT-20k SFT 能否在 MATH500 上产生可测量的 gain？

**通过条件：** 最终 checkpoint（step_624）的 MATH500 精度严格高于 base 至少 1 SE。

```
MATH500(step_624) > MATH500(step_000) + 1·SE
SE = sqrt(p * (1 - p) / N)，其中 p = MATH500(step_000)，N = 500
```

外部参考锚（Rethink SFT Table 9）：base MATH500 = 70.8 → SE ≈ 0.020。
预注册阈值在运行时用**实测** step_000 分数计算，不用外部锚点。

**通过：** 可行性确认，进入 Cycle 08（H-OPD）。
**失败：** 检查超参数（LR、epochs、LoRA rank）后再启动 Cycle 08。

**重要：** 不得用早期 checkpoint 判断 A07。Rethink SFT Table 9 显示 step 20 时 MATH500 下降到 62.9，之后才恢复到 86.2。Gate A07 以 **step_624** 为判断节点，辅以轨迹最高点。

---

## Gate B07 — 轨迹特征（诊断性）

**问题：** dip-and-recovery 模式是否可见？

**Dip 定义：** ∃ 步数 s ∈ {5, 10, 20, 40}，使得 `MATH500(step_s) < MATH500(step_000)`。
**Recovery 定义：** ∃ 步数 s' > s，使得 `MATH500(step_s') > MATH500(step_000)`。

**完全通过：** dip + recovery 均可见。直接验证 Rethink SFT 动态在 LoRA variant 上成立。
**部分通过：** recovery 可见但无明显 dip（LoRA 可能跳过暂态）。可解释为 LoRA 特有的收敛模式。
**失败：** 无 recovery。升级排查——可能是 LR、数据或格式问题。

Gate B07 结果决定 Cycle 08 三方对照的参照点定义（dip checkpoint 和 recover checkpoint）。

---

## Gate C07 — OOD-lite Gain（非阻塞）

**问题：** think-format 数学训练是否迁移到 OOD-lite 任务？

**通过条件（参考）：**
```
GPQA-D(step_624) > GPQA-D(step_000)  OR  MMLU-Pro(step_624) > MMLU-Pro(step_000)
```

**通过：** OOD 推理能力提升，与 Rethink SFT 全参数结果（GPQA-D +23.3，MMLU-Pro +29.3）方向一致（LoRA variant 预计幅度更小）。
**失败：** 可接受——math SFT 在 LoRA 规模下可能不迁移到知识类任务。Gate C07 **不阻塞** Cycle 07 完成或 Cycle 08 启动。

---

## Cycle 07 完成条件

以下全部满足时，本 cycle 宣告完成：

1. 全部 10 个 checkpoint（step_000 至 step_624）完成 5 个任务的评测。
2. `trajectory_scores.csv` 已填写（10 × 5 分数 + ±SE）。
3. `response_length_trajectory.csv` 已填写（10 个 checkpoint，MATH500 mean response tokens）。
4. 全部 10 个 checkpoint 的几何文件已保存（含 UV/sketch，含 OverlapLift）。
5. Gate A07 判断已写入 `RESULTS_07.md`。
6. `run_provenance.json` 完整。

**延后任务（非阻塞）：** Cycle 06 geometry copyback —— 将 Cycle 06 run 目录中的
`geometry_metrics.csv` + `principal_evidence.csv` 复制到
`local_experiment_results/cycle_06_sft_feasibility_and_degradation/run_01/`，
关闭 C06 PARTIAL。

---

## 运行作废条件

以下任一情况出现，本次运行无效（不报告结果，需重新运行）：

- 模型路径不是 `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base`。
- 训练样本未使用 `seed=42`。
- Checkpoint 网格 {5,10,20,40,80,160,320,480,624} 中有任何一个缺失或损坏。
- GPQA-D 或 MMLU-Pro 评测时使用了 chat template（必须是无 chat 的 loglikelihood）。
- MATH500/NuminaMath/AIME24 使用非零温度但未记录生成 seed。
- 几何 UV/sketch 文件未保存（无法计算 OverlapLift）。

---

## 预注册允许结论

| 结果模式 | 允许表述 |
|---|---|
| A07 通过，B07 完全通过 | "Qwen3-4B-Base + Math-CoT-20k SFT 呈现 dip-and-recovery 轨迹；最终 MATH500 超 base +X.X。Dip 出现于 step S，在 step S' 恢复高于 base。" |
| A07 通过，B07 部分通过 | "在 step S' 确认 recovery，无明显 dip（LoRA variant 可能跳过暂态）。最终 MATH500 超 base +X.X。" |
| A07 失败 | "最终 checkpoint 未通过 A07 阈值。建议检查 LR/epochs/数据后再启动 Cycle 08。" |
| C07 通过 | "OOD-lite gain（GPQA-D +X.X，MMLU-Pro +X.X）确认 think-format 数学训练的跨领域泛化。" |
| C07 失败 | "最终 checkpoint 未观察到 OOD-lite gain。LoRA variant 的增益局限于数学领域。" |

## 不允许（Cycle 07 范围）

- 不得声称 H-OPD 假设成立——H-OPD 是 Cycle 08 假设。
- 不得将 Cycle 04/05/06 结果直接对比——模型、格式、数据均不同。
- 不得将 Rethink SFT 的 4B 数字标注为"已复现"——外部参考来自全参数训练，LoRA variant 动态不同。
- 不得用 instruct 系列结果作为主线——instruct 协议在 Cycle 07 不再是主要评测轴。
- 不得做出任何 OPD 相关主张——本 cycle 无 OPD arm。
