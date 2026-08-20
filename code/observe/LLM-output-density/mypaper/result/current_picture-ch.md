# 当前图景：所有有效实验结果

```yaml
view_type: current_aggregate
status: active
last_updated: 2026-06-18
maintenance_rule: update_when_new_results_land_or_supersede_prior_results
```

本文件回答：**当前全套有效实验结果说明了什么？**

有效 = 有修正版用修正版；尚未被覆盖的用原始结果；基于已知缺陷协议的结果即使尚未替换也排除在外。

每个 cycle 怎么到达现在这个状态，见 `incremental_log-ch.md`。详细表格和证据，见 `local_experiment_results/`。

---

## 当前可以读出的结论

### ID 轴 — NuminaMath-test（有效，全 8 个模型，来自 Cycle 05）

协议：chat template 已加，`enable_thinking=False`，`\boxed{}` 感知评分，open-answer 子集，N=1024（干净留出集，与训练 prompt 零重叠）。

| 模型 | NuminaMath ID（open-answer）|
|---|---|
| base（未训练 Qwen3-1.7B instruct）| 0.5516 |
| theta0 | 0.5527 |
| opd_lmbda05 | 0.5392 |
| opd_lmbda1 | 0.5348 |
| sft_n128 | 0.5370 |
| sft_n256 | 0.5516 |
| sft_n512 | 0.5179 |
| sft_n1024 | 0.3890 |

**解读：** base 和 theta0 在 ID 轴上近似并列领先。两个 OPD arm 的 ID 都低于 base。sft_n128/256 接近 base；sft_n512 略低。sft_n1024 严重退化（−0.163 vs base——数学灾难性遗忘）。没有任何训练方式相对未训练 base 产生 ID 净增益。

**Matched control 对照锚点：** OPD vs SFT 对比的匹配基准是 NuminaMath-test ID（不是 GSM8K）。在此基准下，sft_n128（0.5370）是 opd_lmbda05（0.5392，Δ=+0.0022）和 opd_lmbda1（0.5348，Δ=−0.0022）的最近 SFT 锚点。

---

### GSM8K 轴 — 修正后，全 8 个模型有效（来自 Cycle 05）

协议：chat template + `enable_thinking=False` + 3072 tokens + `\boxed{}` 感知抽取。N=1319。作为 OOD/迁移轴（不是训练域分数）。

| 模型 | GSM8K | ±SE |
|---|---|---|
| base | 0.7005 | 0.0126 |
| theta0 | 0.7180 | 0.0124 |
| opd_lmbda05 | 0.7180 | 0.0124 |
| **opd_lmbda1** | **0.7346** | 0.0122 |
| sft_n128 | 0.7096 | 0.0125 |
| sft_n256 | 0.6839 | 0.0128 |
| sft_n512 | 0.6732 | 0.0129 |
| sft_n1024 | 0.6141 | 0.0134 |

**解读：** opd_lmbda1 是全 8 个模型中最高的（+0.0341 over base，+0.0166 over theta0）。两个 OPD arm 都达到或超过 theta0。SFT 随数据量 n 显示退化趋势：sft_n128 接近 theta0，sft_n256/512 低于 base，sft_n1024 严重退化（−0.086）。

---

### MATH500 轴 — 修正后，全 8 个模型有效（来自 Cycle 05）

协议：chat template + `enable_thinking=False` + 3072 tokens + `math_verify` 符号评分 + `last_boxed` 抽取。N=500。

| 模型 | MATH500 | ±SE |
|---|---|---|
| base | 0.7140 | 0.0202 |
| theta0 | 0.7260 | 0.0199 |
| opd_lmbda05 | 0.7080 | 0.0203 |
| opd_lmbda1 | 0.7140 | 0.0202 |
| sft_n128 | 0.7260 | 0.0199 |
| sft_n256 | 0.7080 | 0.0203 |
| sft_n512 | 0.7140 | 0.0202 |
| sft_n1024 | 0.5400 | 0.0223 |

**解读：** theta0 和 sft_n128 并列最高（+0.0120 over base）。opd_lmbda05 和 sft_n256 略低于 base（−0.006）。opd_lmbda1 和 sft_n512 与 base 持平。与 GSM8K 不同，OPD 在此轴上没有展现出一致优势。sft_n1024 严重退化（−0.174）。

---

### OOD-lite（MMLU / TruthfulQA / WinoGrande / ARC-challenge）— 有效，全 8 个模型（来自 Cycle 05）

**协议说明：**
- MMLU：base model 模式（无 chat template，5-shot，标准 loglikelihood）——Qwen3 官方协议（arXiv 2505.09388，`mmlu_protocol_version=2`）
- TruthfulQA-MC1：chat template + loglikelihood，0-shot
- WinoGrande：无 chat template，0-shot，loglikelihood
- ARC-challenge：无 chat template，25-shot（lm-eval 任务默认值），`acc_norm`，N=1172——与 MMLU/WinoGrande 相同的无 chat template loglikelihood 协议

| 模型 | MMLU | TruthfulQA MC1 | WinoGrande | ARC acc_norm |
|---|---|---|---|---|
| base | 0.6061 | 0.3341 | 0.6172 | 0.4565 |
| theta0 | 0.6019 | 0.3415 | 0.6235 | 0.4590 |
| opd_lmbda05 | 0.6004 | 0.3390 | 0.6251 | 0.4650 |
| opd_lmbda1 | 0.6061 | 0.3427 | 0.6235 | 0.4616 |
| sft_n128 | 0.6049 | 0.3439 | 0.6290 | 0.4599 |
| sft_n256 | 0.6033 | 0.3415 | 0.6235 | 0.4667 |
| sft_n512 | 0.6086 | 0.3378 | 0.6322 | 0.4710 |
| sft_n1024 | 0.6065 | 0.3317 | 0.6322 | 0.4667 |

**解读：** 全 8 个模型的 spread：MMLU ±0.0041（range 0.0082），TruthfulQA ±0.0061（range 0.0122），WinoGrande ±0.0075（range 0.0150），ARC-challenge ±0.0073（range 0.0145）。所有模型——包括表现出严重数学退化的 sft_n1024——在全部 4 项 OOD-lite 指标上都聚集在 <1.5% 内。这个轴作为能力保持检验有效，但不是 OPD vs SFT 之间的判别信号。

参考背景：TruthfulQA-MC1 0.33-0.34 对 1.7B 模型是正常的（Qwen2.5-1.5B-Instruct ≈ 0.30，Llama-2-7B ≈ 0.306）。无异常。

---

### sft_n1024 退化汇总

sft_n1024 表现出与其他所有模型定性不同的格局：

| 轴 | sft_n1024 | base | Δ |
|---|---|---|---|
| NuminaMath ID | 0.3890 | 0.5516 | **−0.163** |
| GSM8K | 0.6141 | 0.7005 | **−0.086** |
| MATH500 | 0.5400 | 0.7140 | **−0.174** |
| MMLU | 0.6065 | 0.6061 | +0.000 |
| TruthfulQA | 0.3317 | 0.3341 | −0.002 |
| WinoGrande | 0.6322 | 0.6172 | +0.015 |
| ARC-challenge | 0.4667 | 0.4565 | +0.010 |

数学轴全部退化；OOD-lite 稳定或略高。训练/评测格式对齐已验证（Qwen3 tokenizer 在训练和评测路径中注入完全相同的空 think block）。这种格局与 1.7B 模型在 n=1024 SFT 样本下的数学推理灾难性遗忘一致，不是格式假象。sft_n1024 被排除在 OPD vs SFT 对比之外；sft_n128/256/512 是相关的 SFT 对照。

---

### Gate 分析 — Cycle 04 Gates 用 Cycle 05 修正数字重分析

| Gate | Cycle 04 结论 | Cycle 05 重分析 | 变化 |
|---|---|---|---|
| A04：可行性 | PASS | 确认 PASS | 无变化 |
| B04：matched control | PASS | 确认 PASS | 匹配基准明确为 NuminaMath-test ID；sft_n128 仍是两个 OPD arm 的最近锚点 |
| C04：稳定性 | FAIL | 确认 FAIL | OPD 未在任何 OOD-lite 指标上击败 sft_n128；但 spread <1.5%——非信号，不是负向发现 |
| D04：增益 | FAIL | 确认 FAIL | 依赖 Gate C04 |

**Gate C04 说明：** 原始 Gate C04 失败基于无效的 MMLU 数字。用正确的 OOD-lite 数字，OPD 在形式上仍在全部 4 个指标上低于 sft_n128（形式上失败成立）。然而，全部 8 个模型包括 base 在每项 OOD-lite 指标上都聚集在 <1.5% 内——失败反映的是当前设置下判别力不足，不是 OPD 损害了泛化。

---

### 几何 — 来自 Cycle 04（仅方向性信号）

GetSlice（多层，7 模块 × 3 层 × 2 prompt 类型）：
- OPD 相对 `theta0` 降低 effective rank（谱更集中）。
- SFT 提高 effective rank（谱更分散）。
- OPD 未能一致降低相对 `theta0` 的漂移；`opd_lmbda1` 第 14 层漂移高于所有 SFT 对照。
- UV/principal-angle 产物未保存 → 不能做子空间重叠 claim。

---

## 尚未完成的

| 待办项 | 状态 |
|---|---|
| 几何在修正后 matched-control 框架下的重解读 | 尚未开始——Cycle 04 GetSlice 数据有效（checkpoint 未变）；需用 NuminaMath ID 匹配而不是 GSM8K 匹配重新解读 |
| 任何 Cycle 06 新训练 | 尚未开始；当前评测审计分析已完成 |
