# 当前可声称的结论

```yaml
view_type: current_claim_boundary
status: active
last_updated: 2026-06-18
maintenance_rule: update_when_results_change_or_new_cycle_lands
```

本文件定义当前有效实验结果支持哪些结论。具体数字见 `current_picture-ch.md`；每条发现的来源见 `incremental_log-ch.md`。

---

## 允许声称的结论

**关于实验流程：**
- Cycle 04 完成了比 Cycle 03 更严格的 OPD-like vs SFT 对比。
- 工程循环现在更完善：大样本 GSM8K、OOD-lite 评测、多层 GetSlice 几何、matched-pair 选取、ID/OOD 分轴评测，均已整合。
- NuminaMath-test 是可用的 ID 评测轴（holdout 切分与训练 prompt 零重叠）。

**关于 Cycle 03 → 04 复现：**
- Cycle 03 的 OPD-like 稳定性信号没有在 Cycle 04 更严格条件下复现。
- matched-gain 对比对 `theta0` 饱和度敏感：不饱和的 `theta0` 会让 matched SFT 对照退化到几乎静止的控制。
- Cycle 04 中，matched SFT 对照锚点退化为 `sft_n128`（几乎未离开 `theta0`）。
- 这是一个 control-anchor 失败模式，不是 OPD 整体比 SFT 差的证据。

**关于 ID 轴（NuminaMath-test，有效）：**
- 未训练的 `Qwen3-1.7B` instruct checkpoint 在 NuminaMath ID 上与 theta0 近似并列（open-answer 0.5516 vs 0.5527），且 ID 高于两个 OPD arm。
- 在当前 1.7B instruct + ≤1024 NuminaMath 训练数据设置下，没有任何训练方式相对未训练基座产生了 ID 净增益。
- GSM8K 应被视为 OOD/迁移轴，而不是主要性能轴。
- OPD vs SFT 对比的匹配基准是 NuminaMath-test ID（不是 GSM8K）。在此基准下，sft_n128（0.5370）是 opd_lmbda05（0.5392，Δ=+0.0022）和 opd_lmbda1（0.5348，Δ=−0.0022）的最近 SFT 锚点。

**关于评测协议修正（Cycle 05 发现，全 8 个模型完成）：**
- Cycle 04 原始 GSM8K（~0.41 区间）和 MATH500（0.000）几乎全是测量地板：缺少 chat template、生成预算不足、dollar-span 抽取 bug（MATH500）、训练/评测格式不匹配。
- 修正协议让全部 8 个模型的 GSM8K 提升到 ~0.61-0.73，MATH500 提升到 ~0.54-0.73。
- `theta0` 在 GSM8K（+0.0175）和 MATH500（+0.0120）两个轴上都领先未训练 `base`；单轴各不到 ~1σ，但方向一致。
- 修正协议下确实存在高于未训练基座的 headroom，推翻了 Cycle 04 的 base-ceiling 读数。
- `opd_lmbda1` 是全 8 个模型中 GSM8K 最高的（+0.0341 over base，+0.0166 over theta0），但 NuminaMath-test ID 低于 base（0.5348 vs 0.5516）。
- 任何接近 0 的数学评测数字，必须先审计抽取是否失败，再解释为模型无能力（项目级规则）。

**关于 OOD-lite（MMLU / TruthfulQA / WinoGrande / ARC-challenge，Cycle 05 有效，全 8 模型）：**
- 有效的 MMLU 数字通过 base model 模式获得（无 chat template，5-shot，标准 loglikelihood）——Qwen3 官方协议（arXiv 2505.09388，`mmlu_protocol_version=2`）。
- ARC-challenge 使用与 MMLU 和 WinoGrande 相同的无 chat template loglikelihood 协议（25-shot lm-eval 任务默认值，acc_norm，N=1172）。
- OOD-lite 在全部 8 个模型上的 spread 极小：MMLU ±0.0041，TruthfulQA ±0.0061，WinoGrande ±0.0075，ARC-challenge ±0.0073。任何模型之间在任何 OOD-lite 指标上都没有可区分的差异。
- OOD-lite 仅作为能力保持检验：fine-tuning（所有 arm）在当前设置下不显著改变通用语言能力。
- TruthfulQA-MC1 0.33-0.34 对 1.7B 模型是正常的（参考：Qwen2.5-1.5B-Instruct ≈ 0.30，Llama-2-7B ≈ 0.306）。无异常。
- Gate C04 形式上确认 FAIL（OPD 未在 OOD-lite 上击败 matched SFT），但这是非信号：全部 8 个模型在全部 4 项 OOD-lite 指标上均聚集在 <1.5% 内。失败反映的是判别力不足，不是 OPD 损害了泛化。

**关于 sft_n1024 退化：**
- `sft_n1024` 表现出严重的数学轴退化（NuminaMath ID −0.163，GSM8K −0.086，MATH500 −0.174 vs base），OOD-lite 表现稳定。
- 训练/评测格式对齐已验证（Qwen3 tokenizer 在训练和评测路径中注入完全相同的空 think block）；退化不是格式假象。
- 这种格局与 1.7B 模型在 n=1024 SFT 样本下的灾难性遗忘一致。
- `sft_n1024` 是异常值，不用作 OPD 分析的对照锚点；sft_n128/256/512 是相关的 SFT 对照。

**关于几何：**
- OPD 相对 `theta0` 降低 effective rank（激活谱更集中）；SFT 提高 effective rank。
- 这些仅是方向性信号；UV / principal-angle 产物未保存，不能做子空间重叠 claim。

---

## 不允许声称的结论

- 不能声称完整复现 OPD，或 OPD 提升了推理准确率。
- 不能声称 OPD 总体上比 SFT 更稳定。
- 不能声称 Cycle 04 有广泛的 OOD 泛化。
- 不能用任何 MATH500 = 0.0（Cycle 04 原始）数字作为模型无能力的证据。
- 不能将本地未训练的 `Qwen3-1.7B` instruct checkpoint 写成官方 `Qwen3-1.7B-Base` 预训练 checkpoint。
- 不能引用来自 Cycle 04 原始跑或 Cycle 05 chat-template 重跑（MMLU 崩溃到 0.2295）的 OOD-lite 数字；只用 Cycle 05 base-model-mode 数字。
- 不能将 OOD-lite 用作 OPD vs SFT 之间的判别轴——全 8 个模型的 spread <1.5%，不足以支持任何可靠对比。
- 不能引用任何模型未修正的 GSM8K 数字（0.33-0.43 区间）。
- 不能视 Cycle 04 的 base-ceiling / 无 headroom 读数在 GSM8K 或 MATH500 轴上为已确认。
- 不能声称 principal-angle 或 subspace-overlap 证据（UV 产物未保存）。
- 不能将 Cycle 03 正向信号和 Cycle 04 负向结果视为简单矛盾；两者都受 matched-control 对照锚点的混淆。
- 不能写"Gate D 差一点就通过了"——它在预注册标准下是失败的。
- 不能在不保留未训练基座为一等参照的情况下进行进一步的 OPD vs SFT 对比。
- 不能用 `sft_n1024` 作为 OPD 分析的对照锚点；它是异常值。
- 不能将 Gate C04 失败解读为 OPD 损害了泛化——该轴在当前设置下判别力不足。

---

## 推荐表述

**Cycle 04 主结论：**
```
Cycle 04 对 Cycle 03 的初步稳定性信号进行了更严格的否定性复现。结果表明，
matched-gain 下的 OPD vs SFT 比较对 theta0 饱和度和 control-anchor 选取敏感。
```

**关于 GSM8K 上的窄幅 OPD 信号：**
```
OPD-like 训练相对 theta0 维持或微升了 GSM8K，但效果低于预注册的改善阈值，
且没有带来稳健的 OOD-lite 或几何稳定性优势。
```

**关于修正后的 base-ceiling：**
```
修正 GSM8K 和 MATH500 评测协议后，theta0 在两个轴上都领先未训练基座
（+0.0175 和 +0.0120），尽管单轴各不显著。headroom 存在；
原来的 base-ceiling 读数是测量假象。
```

**关于 OOD-lite：**
```
所有训练方式——OPD 和 SFT——在 MMLU（5-shot，base model 模式）、TruthfulQA-MC1、
WinoGrande 和 ARC-challenge（25-shot，无 chat template）上均保持通用语言能力
在 ±1.5% 以内。这个轴确认了能力保持，但在当前实验设置中对 OPD 和 SFT 的
判别力不足。
```

**关于 sft_n1024：**
```
sft_n1024 表现出严重的数学轴退化（NuminaMath ID −0.163，MATH500 −0.174 vs base）
同时 OOD-lite 表现稳定，与 1.7B 模型在该 SFT 数据规模下的灾难性遗忘一致。
排除在 OPD vs SFT 主要对比之外。
```

**避免：**
```
OPD 失败了。
SFT 比 OPD 更好。
OPD 提升了 GSM8K。
Cycle 04 证明了机制是错的。
MATH500 说明模型不会做竞赛数学。
base 和 theta0 在 GSM8K 上持平。（仅在损坏协议下成立，现已过时）
OPD 损害了泛化。（Gate C 失败是非信号，不是负发现）
```
