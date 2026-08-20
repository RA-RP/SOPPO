# 当前理论更新：从 Matched-Gain Stability 到 Base-Aware Stability

```yaml
view_type: current_latest_view_ch
status: active_cycle08_theory
source: (snapshot removed 2026-06-16; this file is primary)
last_organized: 2026-06-29
```

## 为什么需要更新理论

Cycle 03 支持一个较窄的 K1-style story：

```text
在 matched GSM8K gain 下，OPD-like training 可能比 continued SFT 有更低 OOD-lite degradation 和更低 spectral drift。
```

Cycle 04 说明这个故事 under-specified。matched-gain control 会随 `theta0` saturation 选择完全不同的 SFT anchor。当 matched SFT 几乎没有离开 `theta0` 时，与任何真正移动的训练方法相比都会不公平。

新增 base-model test 进一步说明：untrained Qwen3-1.7B base 已经在 ID/GSM8K 上近似等于 `theta0`，且 ID 高于 OPD arms。这意味着当前设置可能几乎没有 headroom，无法让任何训练方法显示 meaningful gain。

## 更新后的假设

```text
K1-base-aware: 只有当所选 base/task/data 设置在 untrained base 之上显示 measurable headroom 后，OPD-like training 才能与 SFT 进行有意义比较。
在这个前提下，OPD-like training 只有在 comparable ID learning 或 comparable training budget 下更好保留 OOD behavior，才能被称为更稳定；不能只看 comparable GSM8K gain。
```

该版本分离四个 Cycle 03 混在一起的量：

- ID learning：NuminaMath-test 表现。
- OOD transfer：GSM8K 和 OOD-lite 表现。
- base headroom：trained model 相对 untrained base 的提升。
- representation movement：spectral drift、effective rank、subspace change。

## 关键变量

| 变量 | 作用 |
|---|---|
| `base_performance` | 判断任务是否在 untrained model 之上有 measurable headroom。 |
| `theta0_saturation` | 决定剩余 ID/OOD gain 空间，以及 matched SFT anchor。 |
| `id_performance` | 用 NuminaMath-test 测 same-domain capability。 |
| `ood_retention` | 用 GSM8K 和 OOD-lite 测 transfer/generalization。 |
| `training_budget` | 防止把 trained OPD 与几乎未训练 SFT 比较。 |
| `geometry_movement` | 测 representation change，但应相对 learning amount 解释。 |
| `extraction_validity` | 防止 scoring artifact 导致错误结论。 |

## Predictions

### P0: Base Headroom

如果 base、`theta0`、OPD、SFT 在 ID/OOD 上都在不确定性内持平，则当前设置不适合测试 improvement 或 stability。下一步应重设 task/base/data，而不是继续 OPD-vs-SFT。

### P1: Equal-ID Matching

如果 OPD 有真实 stability advantage，那么在相近 NuminaMath-test performance 下，它应至少和 SFT 一样好地保留 GSM8K/OOD-lite。

### P2: Equal-Budget Matching

如果 Cycle 04 negative result 主要是 control-anchor artifact，那么 equal-budget comparison 应不再像 matched `sft_n128` 那样对 OPD 不利。

### P3: Theta0 Saturation

`theta0` 越饱和，matched-gain selection 越可能转向更大或更退化的 SFT controls；`theta0` 越不饱和，matched-gain selection 可能坍缩到极小 SFT controls。

### P4: Geometry Normalization

Raw drift from `theta0` 可能惩罚学得更多的方法。geometry metric 应与 ID learning 和 OOD retention 一起解释。

## Falsifiers

以下情况会削弱 base-aware K1：

- 没有 trained variant 在不确定性之外超过 untrained base；
- OPD 在 equal-ID matching 下不能比 SFT 更好保留 OOD；
- OPD 在 equal-budget matching 下不能比 SFT 更好保留 OOD；
- OPD geometry movement 更大但没有对应 OOD retention；
- 结论仍只是因 matching anchor 改变而翻转。

## 当前论文层 claim

Cycle 05 完成前，最安全 claim 是：

```text
早期 OPD-like 实验显示，stability conclusion 强烈依赖 base headroom 与 control-anchor construction；在声称 OPD stability advantage 前，需要 base-aware ID/OOD matching。
```

---

# Cycle 06 理论更新：SFT 可行性作为 K1 前置条件

```yaml
artifact_type: theory_update
cycle: cycle_06_sft_feasibility_and_degradation
date: 2026-06-23
status: active
qa_source: theory/QA_cycle06_theory.md
```

## 什么改变了，什么没变

K1-base-aware 仍是主假设，不变。Cycle 06 对理论的贡献非常小——其目的是工程验证：在干净 SFT baseline 下找到合适的模型、训练格式和测评协议。SFT catastrophic forgetting 是已知现象，不需要新的理论变量。

Cycle 06 的 theory 层面只增加：
1. 说明 K1 何时才能被检验的前置条件子节。
2. Claim boundary 中的模型身份显式定义。
3. Gate 解释规则（防止结果出来后临时换标准）。
4. 更新 OOD-lite 和禁止 claim 边界。

不新增机制变量。SFT 退化曲线（规模 → 退化）是 motivating observation 和工程发现，由 Experiment Design 处理。

## Cycle 06 K1 有效性的前置条件

K1-base-aware 只有同时满足以下两条才能被有意义地检验：

1. **SFT 可学性确认**（Gate A06 pass）：SFT-from-base 在主要 ID/downstream 轴上比未训练 base 产生可测量的提升。
2. **协议有效性确认**（Gate C06 pass）：训练和测评格式及 scorer 协议在训练运行前已通过验证。

如果两个条件都不满足，当前设置不适合进行 OPD-vs-SFT 比较。下一步应是重设模型、任务、数据或格式，而非继续跑 OPD 训练。

## 模型身份 Claim Boundary

```text
local_base_row ≠ official_pretraining_base_checkpoint

本地 Qwen3-1.7B（instruct stack，已 post-trained）：
  path: /root/autodl-tmp/model/Qwen/Qwen3-1.7B
  已经历 instruction tuning，不能等同于 pretraining-only LM 的起点。
  其 0-shot chat 表现不代表 pretraining-only 的能力下限。

官方 Qwen3-1.7B-Base（pretraining-only）：
  path: Qwen/Qwen3-1.7B-Base（HuggingFace hub 官方）
  无 instruction tuning；chat/think 格式假设不自动成立。
  需要单独的测评协议。
```

如果 Cycle 06 切换到官方 Base checkpoint，理论在下一轮需要区分两者的适用假设（例如，Base 的初始能力更低是否意味着更大的 SFT headroom，目前尚未实验）。

## Gate A06 / B06 理论解释边界

以下是解释规则，不是数字 threshold。数字 threshold 由 Experiment Design 预注册。

```text
A06 pass：
  当前 model/task/data/protocol 设置具备足够 SFT 可学性。
  可以考虑在后续 cycle 进行 OPD 比较。
  不代表 OPD 比较已就绪——Experiment Design 仍需决定 cost、control 和 matching。

A06 fail：
  当前设置 headroom 不足或存在 protocol/task/model 不匹配。
  不能解读为 OPD 本身失败。
  应先调整 model/task/data 或格式再跑 OPD。

B06 degradation observed：
  SFT scale 在当前设置下可诱导能力 tradeoff。
  OPD forgetting-prevention 测试在后续 cycle 中变得有意义。

B06 no degradation：
  当前设置在测试规模内对 SFT 表现稳定。
  OPD forgetting-prevention 在此设置中暂不可测。
```

## OOD-lite Claim Boundary（从 Cycle 05 正式写入）

OOD-lite（MMLU / TruthfulQA / WinoGrande / ARC-challenge）在当前 1.7B + NuminaMath 设置下只是**capability-preservation check**，不是 OPD-vs-SFT 或 SFT scale 的判别轴。

实验依据：Cycle 05 显示全部 8 个模型在四个任务上 spread <1.5%，包括出现严重数学退化的 `sft_n1024`。当前设置下该轴判别能力不足。

此边界适用，直到未来实验在一个或多个 OOD-lite 任务上出现更大的、协议有效的 spread。

## Cycle 06 禁止 Claim 列表

1. 不声称 OPD 已防止 catastrophic forgetting。Cycle 06 不训练 OPD arm。
2. 不把 Cycle 06 定性为 OPD-vs-SFT comparison。
3. 不把 `sft_n1024` 退化写成已证明的机制（过拟合、遗忘等）。只写成 motivating observation / consistency signal。
4. 不在当前设置下把 OOD-lite 用作判别轴。
5. 不把本地 Qwen3-1.7B instruct checkpoint 等同于官方 Qwen3-1.7B-Base（pretraining-only）。
6. 不把 geometry diagnostic 写成机制证明。
7. 不把 protocol-invalid 数字当作理论证据：
   - 旧 Cycle 04 GSM8K（~0.41 band）：协议有缺陷，不代表能力。
   - 旧 Cycle 04 MATH500（0.000）：extraction failure，不代表能力。
   - Cycle 05 带 chat template 的 MMLU（0.2295）：协议错误。

---

# Cycle 07 理论更新：从 base-aware 边界到轨迹机制（OPD vs SFT_dip vs SFT_recover）

```yaml
artifact_type: theory_update
cycle: cycle_07_base_model_sft_feasibility
date: 2026-06-24
status: active
qa_source: theory/QA_cycle07_theory.md
scaffolding: related_work/source_notes/rethink_sft_experiment_tables.md §7
external_anchor: "Ren et al., Rethinking Generalization in Reasoning SFT, arXiv 2604.06628 (2026)"
```

## 什么变了：项目第一个正向机制

Cycle 04–06 是**防御性**的：它们确立的是*什么时候比较无效*（base 饱和、无 headroom、能力地板、协议 artifact）。Cycle 06 theory 尤其是最小的工程级更新。

Cycle 07 性质不同。它引入项目**第一个正向、可证伪的机制假设**。触发点是 Rethink SFT（Ren et al. 2026）的外部证据：reasoning SFT 下 cross-domain 性能遵循 **dip-and-recovery** 轨迹，而非单调曲线。

这不替换 base-aware 框架，而是**嵌套其中**（Q1=c）。三层构成一条逻辑链：

```
第 1 层（入场条件）   K1-base-aware：只有在 untrained base 之上有可测量 headroom 时，
                    比较才有意义。
        │  （Rethink SFT 从外部强化了这一层：官方 Qwen3-1.7B-Base 复现能力地板；
        │    换 4B + long-CoT 正是为越过这道门槛。）
        ▼
第 2 层（机制）       Dip-and-recovery：OOD-drop 是 checkpoint 所处优化阶段的属性，
                    不是方法的固有属性。
        ▼
第 3 层（预注册）     H-OPD：OPD 是不经过 dip、直达 recover 状态的捷径。（Cycle 08 检验。）
```

## 核心重构：OOD-drop 是 checkpoint 属性，不是 method 属性（Q2）

前几个 cycle 隐含把 OOD-drop / 稳定性当作**方法的固有属性**（"SFT 掉 OOD，OPD 更稳"）。Rethink SFT 的 dip-and-recovery 表明这一表述是错的：**同一个** SFT，在早 checkpoint 掉 OOD（dip），在晚 checkpoint 回升、常常反超 base（recover）。

**重构后的中心 claim：**

```text
OOD-drop 是优化轨迹上的暂态，由 checkpoint 所处优化阶段（dip vs recover）决定。
"SFT 必然掉 OOD" 是把 checkpoint 属性误读为 method 属性。
```

**Caveat（强制 claim 边界）：** 这一重构目前由**外部**证据支撑（Rethink SFT，Qwen3-4B/8B/14B）。本项目**尚未在自己 pipeline 内复现 dip-and-recovery**。Cycle 07 在 Qwen3-4B-Base 上的多 checkpoint 采样正是内部验证。在此之前，把该重构当作有外部证据支持的强假设，而非已定论的结果。

## 三方框架：OPD vs SFT_dip vs SFT_recover

SFT 不是一个点，而是 SFT_dip → SFT_recover 的轨迹。两个状态有不同的可测量签名（Rethink SFT，Qwen3-4B-Base + Math-CoT-20k，Table 9；行为定性来自 App C）：

| 维度 | SFT_dip（~step 20–40） | SFT_recover（~step 320+） | Rethink SFT 4B 锚点 |
|---|---|---|---|
| ID（训练域） | 未巩固（MATH500 62.9 < base 70.8） | 巩固 / 反超（MATH500 86.2） | Table 9 |
| OOD reasoning | 掉（GPQA 17.3, LCB 14.5） | 回升 / 反超（GPQA 37.9, LCB 24.4） | Table 9 |
| response_length | 暴涨（~28–30k） | 收缩（~7–23k） | Sec 3.2; Fig 13 |
| 行为 | shallow imitation，长而不收敛 | internalized procedural pattern（会 backtrack、自校验） | App C.3, C.6 |

4B run 也有可见 dip（MATH500 step 20 = 62.9，低于 base 70.8，再升到 86.2），所以该框架的两个状态在此模型规模上预期可观测。

## H-OPD：Cycle 08 的预注册假设（Q3）

```text
H-OPD：OPD ≈ 不经过 dip 暂态、直达 recover 状态的捷径。on-policy 训练减少 exposure
bias，模型更少绕道经过 shallow-imitation 状态。
```

**可证伪预测（Cycle 08，需 OPD arm —— Cycle 07 不检验）：**

- 若 H-OPD 成立：OPD 的（ood_drop、response_length surge、几何暂态峰）**显著小于 SFT_dip**，而 ID/OOD 终点 **≈ SFT_recover**。
- 若 H-OPD 失败：OPD 经历与 SFT 相当的 dip，或终点不匹配 SFT_recover。

**文献支撑（OPD 腿，已在 Related Work 登记）：** Shenfeld et al. 2026（online 训练忘得少、更贴近 base 分布 → 直接预测 OPD 的 drift-from-base 更平）；Chen et al. 2025（on-policy 缓解 forgetting）；Tajwar et al. 2024（on-policy 采样的价值）。

**边界：** Cycle 07 不训 OPD arm，不产生 OPD 证据。H-OPD 写在这里只作为预注册目标，防止 Cycle 08 结果出来后临时编故事。

## Cycle 07 轨迹变量（Q4）

不同于 Cycle 06（刻意不建变量表，因为退化是工程已知量），Cycle 07 是真机制，需要变量：

| 变量 | 含义 | 角色 |
|---|---|---|
| `optimization_stage` | checkpoint 在 SFT 轨迹上所处阶段（dip / 过渡 / recover） | OOD-drop 归因到的核心自变量 |
| `response_length` | 生成长度；dip 暴涨、recover 收缩 | dip↔recover 的**可观测代理**（Rethink SFT Sec 3.2）；**必收 artifact** —— Experiment Design 必须沿轨迹采集 |
| `ood_drop` | OOD reasoning 相对 base 的变化 | 因变量；从 method 属性重构为 checkpoint 属性 |
| `id_consolidation` | ID（训练域）是否巩固 / 反超 base | 区分 shallow imitation 与 internalized |
| 几何（轨迹） | effective rank / eigengap γ_r / drift-from-base **沿轨迹**读 | 机制层证据；见 `geometry_metric_definitions.md`（Cycle 07 节） |

**对 Experiment Design 的硬要求：** 几何指标与 `response_length` 必须在 SFT 轨迹的**多个 checkpoint** 采样，不能只取终点 checkpoint —— 否则 dip 暂态不可见，Cycle 08 三方对照将缺基线。

## 旧负结果的重新定性（Q6）

Rethink SFT 为项目 Cycle 04–06 在 Qwen3-1.7B + NuminaMath 上的负结果提供了外部解释。它们很可能是**两个独立缺陷的叠加**，而非 SFT 方法的终态失败：

- **能力地板：** Qwen3-1.7B 训不出泛化。论文在*官方* Qwen3-1.7B-Base 上复现（Table 6：640 步后 MATH500 净负 58.9 → 56.2），所以不是 instruct/no-think artifact。
- **数据质量地板：** NuminaMath-1.5 短解是论文表现最差的一档数据（Table 2：把 OOD reasoning 打到地板）。

```text
重新定性：Cycle 04–06 的 instruct 负结果与"弱模型 + 烂数据双坑"一致，而非 SFT 方法
本身的终态失败。
```

**Caveat（强制）：** Rethink SFT 是外部证据（4B/8B/14B）。本项目尚未在自己 pipeline 内复现 4B + long-CoT 的正向结果。在 Cycle 07 Gate A07 出结果前，这一重新定性是**有外部证据支持、待内部验证的假设**，不是已确认的结论。它把项目前半段从"一串看不出名堂的负结果"转成逐步隔离混淆因素的方法学链条（saturation → headroom → 能力地板 + 数据质量）—— 但链条的最后一环是待验证的，不是假定的。

## Cycle 07 claim 边界与禁止项（Q7）

1. 不把 OOD-drop 写成 SFT 方法的固有属性 —— 它是 checkpoint / 优化阶段属性（除非未来证据推翻）。
2. 不把 Rethink SFT 的 4B/8B/14B 结论当作"本项目已验证" —— 它是外部证据，尚未在本 pipeline 复现。
3. 不把 H-OPD 写成结论或已观测 —— Cycle 07 不训 OPD；H-OPD 是 Cycle 08 的预注册假设。
4. 不在看完整 trajectory 前判 SFT "fail" —— dip-and-recovery + under-optimization 护栏：短 epoch checkpoint 会系统性低估泛化（Rethink SFT Sec 3）。
5. 不把旧 1.7B + NuminaMath 负结果写成"SFT 方法失败" —— 现归因弱模型 + 烂数据双坑（待内部验证）。
6. `response_length` / effective-rank / γ_r 暂态峰是**诊断代理 / 待验证假设**，不是机制证明；需多 checkpoint 采样。
7. 不把 dip-and-recovery 写成本项目已复现的现象 —— 它来自外部工作；Cycle 07 是首次内部检验。

## 与既有理论的关系

K1-base-aware（Theory 05）与 Cycle 06 SFT-feasibility 前置条件**均仍成立**，作为上述链条的第 1 层（入场条件）。模型身份边界（`local_base_row ≠ official_pretraining_base_checkpoint`）在 Cycle 07 直接成为承重项：本轮专门换到**官方 Qwen3-4B-Base** pretraining checkpoint + long-CoT think-format，这是第一次有意越过 K1 headroom 门槛的尝试。


---

# Cycle 08 理论更新：H-OPD 主动检验与 matched-LoRA 轨迹比较

```yaml
artifact_type: theory_update
cycle: cycle_08_h_opd_vs_sft_comparison
date: 2026-06-29
status: active
qa_source: theory/QA_cycle08_theory.md
depends_on:
  - cycle_07_base_model_sft_feasibility
```

## 从 Cycle 07 到 Cycle 08 改变了什么

Cycle 07 已把原本依赖外部文献的 SFT 轨迹假设推进为本项目内部观测：Qwen3-4B-Base + Math-CoT-20k + think-format LoRA 具备可学习性，SFT 轨迹有真实 dip-and-recovery，且早期 dip 伴随 geometry / output-control 暂态。Cycle 08 因此是 H-OPD 的第一次直接检验，不再只是准备阶段。

```text
当前 H-OPD 检验：
OPD 应当压平、推迟或避开 SFT dip 暂态，同时匹配 SFT recovered endpoint。
如果 OPD 复现同样的 dip，或不能到达 recovered endpoint，则 H-OPD 在当前
teacher/lambda/cold-start 设置下被削弱。
```

这不意味着 Cycle 07 已产生 OPD 证据。Cycle 07 是 SFT 轨迹 baseline；Cycle 08 才加入用于检验机制的 OPD arm。

## Primary comparison unit

主比较单位是 **matched checkpoint-grid trajectory**，不是单个 checkpoint。理论层对象是：

```text
SFT trajectory:  base -> SFT_dip -> transition -> SFT_recover
OPD trajectory:  base/cold-start -> 同 checkpoint grid 上的 OPD checkpoints
```

单个 checkpoint 只作为 landmark，而不是因果比较本身。关键 landmark 包括 dip 深度、dip 时间、recovery 时间、endpoint capability、response-length transient、geometry transient。Iso-ID / iso-capability cross-section 可以作为二级摘要帮助解释，但不能替代轨迹比较。

B08/C08 仍是有用 gate，但它们是 trajectory landmarks 的判定接口；不能把设计重新退回单点 matched-control experiment。

## Cold-start 可解释性

Cold-start 选择会改变 H-OPD 检验的含义：

- **Level 0：no cold-start / step_000 OPD。** 这是最严格的 H-OPD 检验。如果成功，说明 OPD 可从原始 base 避开 SFT dip，证据最强。如果在 teacher-overlap 极低时失败，失败可能是 conditioning / teacher-alignment 问题，而不是 H-OPD 的直接证伪。
- **Level 1：由 overlap probe 选择的 minimal pre-recovery cold-start。** 这是条件式 H-OPD 检验。只有当 cold-start 发生在 SFT recovery 之前，且目的只是进入可行 teacher-overlap 区间、不是继承 recovered SFT 状态时，才保持可解释。

Recover-point cold-start 不应作为有意义的 H-OPD 设计。它既不是“避开 dip”的有效消融，也不是当前设置下常见训练 baseline；它只能测试从已 recovery 的 SFT 状态继续 OPD。

## Lambda 原则

Theory 不给 Cycle 08 指定具体 lambda 数值。决策规则是：

```text
在只有一个 OPD arm 时，选择仍预期能保持 endpoint capability 的最强 OPD contrast。
```

如果 `lambda=1.0` 能满足 B08，它是最干净的 OPD 机制对比，也最能说明 OPD 性质。更小 lambda 只是当能力风险过高时的保守选择。最终数值由 Experiment Design 决定。

## Gate 解释

**B08：数学能力与 dip avoidance。**

- Pass：OPD 与 SFT_recover 能力兼容，且不复现同等严重的 early SFT dip；H-OPD 仍可行。
- Fail：当前 OPD 配置在所选 lambda、teacher、cold-start、overlap、实现下还不能替代 SFT。若 B08 严重失败，不得用 geometry 或 OverlapLift positive 来声称 H-OPD。

**C08：非目标能力 preservation。**

- Pass：OPD 在所选非目标轴上相对 SFT 有 preservation evidence。
- Fail / mixed：OPD 可能匹配数学能力，但没有显示更广泛 preservation advantage。

MMLU-Pro 是当前 Cycle 08 轴，但不能把它当作 OOD 的全部。Related Work 已找到更多 capability-degradation benchmarks；具体 benchmark suite 可在 Experiment Design / Code 阶段经过协议检查后选择或扩展。这些补充 benchmark 可以加强或削弱 preservation story，但不能事后变成主要 H-OPD gate。

**D08：geometry transient。**

Geometry 只有在与 B/C 行为结果一致时才支持机制解释。Geometry 不能拯救 capability failure。

**A08：OverlapLift trajectory。**

OPD-vs-SFT 的 OverlapLift trajectory 是有效的相对比较，因为两者使用同一 model/data/checkpoint-grid，并且 LoRA regime matched。相对 OPD-vs-SFT 比较不需要额外 LoRA-aware random reference。但如果要作“on-principal / off-principal”这类相对 universal/random/full-rank reference 的绝对判断，则必须显式定义 reference，否则只能保持限定。

Principal-angle 或实测 subspace-rotation claim 不属于 Cycle 08 theory；注册到 Cycle 09 再讨论。

## Cycle 08 证据层级

```text
Strong claim:
  B08 pass + C08 pass + D08/A08 mechanism signals aligned。
  OPD 匹配 recovered math capability，同时降低 SFT dip / preservation cost，
  且轨迹级机制信号一致。

Narrow claim:
  B08 pass，但 C08 fail 或 mixed。
  OPD 能匹配数学能力 / 避开严重数学 dip，但更广泛 preservation advantage 不成立。

Negative or redesign:
  B08 fail。
  先调查 lambda、teacher、cold-start、overlap、实现，再谈 OPD stability claim。
```

## Cycle 08 禁止 claim

1. 不从 Cycle 07 单独声称 OPD 或 H-OPD 成立；Cycle 07 没有 OPD arm。
2. 不把 Cycle 08 单 OPD arm 失败直接解释为“OPD 理论失败”；需先区分 lambda、teacher、cold-start、overlap、实现问题。
3. 不用 recover-point cold-start 结果声称“OPD 避开了 dip”。
4. 不把 single-checkpoint comparison 写成 primary causal comparison；matched trajectory 是 primary。
5. 不把 OPD-vs-SFT OverlapLift 写成相对未定义 reference 的绝对 on/off-principal verdict。matched-LoRA 下的 OPD-vs-SFT 相对比较本身是允许的。
6. 不让 geometry / OverlapLift positive 拯救 B08 capability failure。
7. 不把 official Qwen3 benchmark scores 和本项目协议分数混为同一量。
8. 不把 non-termination 仅写成 eval artifact；Cycle 07 已确认它是真实 output-control behavior，同时 as-run accuracy 是有效 end-to-end task-success measure。
9. 不把 supplementary capability-degradation benchmarks 事后改写成 primary gates。

---

# Cycle 09 理论更新：被抢发后的贡献重定位（Tier-0）

```yaml
artifact_type: theory_update
cycle: cycle_09_aaai_competitiveness_completion
date: 2026-07-08
status: active
qa_source: theory/QA_cycle09_theory.md（Q1–Q8 = a；讨论决议 E1–E5）
scoop_audit: related_work/source_notes/cycle09_scoop_audit_and_gap_map.md
unblocks: Experiment Design（Tier-0 framing 已在下方拍板）
```

## Tier-0 决定：什么死了，什么活着

**已死（scoop 审计定论）：** (1) "OPD 权重方向 off-principal" 作贡献——已被全参发表（arXiv 2606.07082、2606.13657）。(2) "引入激活 ER 作观测量"独立卖点——2605.30524 已 headline。(3) "激活压缩↔OOD"独立卖点——被部分占据且方向相反。

**活着的护城河（论文核心 claim 领地）：**

```text
在 OPD-vs-SFT 匹配轨迹对照下（同 student/数据/checkpoint 网格/LoRA），
以白化的输出相关激活谱（SVD-LLM lens）为观测量：两种监督通道产生相反的
谱-暂态签名。SFT 出现中层（L18）暂态秩扩张 + 权重方向早期 off-principal
重定向（ρ: step_5–10 off → step_20 翻 on），伴随 OOD（MMLU-Pro）侵蚀；
OPD 更新方向全程稳定、谱单调收缩，伴随 OOD 保持与每步能力优势
（MATH500 +0.10~0.24；final 0.848 vs 0.752）。与塌缩文献"压缩→有害"
相反：预测 OOD 结局的不是压缩量，而是暂态破坏的存在与幅度。
```

任何单篇 competitor 都不具备的证据形态：**三个独立观测空间的时序咬合**——ρ（权重方向，step_5–10）→ ER bump（激活谱，step_20）→ 能力 dip / OOD 侵蚀（行为）。

**领域发展线路（related-work 章骨架）：** 阶段1 权重空间静态（"更新落在哪"，2606.×2 完成）→ 阶段2 表示空间静态/预后（"训完表示长什么样"，2605.30524）→ 阶段3 训练动态（"过程中发生什么"，Rethink SFT / 2509.12235）。**汇合点 = 本项目**：过程视角 × 输出相关谱观测 × 监督通道平行对照 × OOD 结局绑定。

## 贡献分层结构（Q1 = 分层）

| 层 | 内容 | 证据依赖 |
|---|---|---|
| **主贡献** | 锐化 (b)：OPD-vs-SFT × 白化输出相关谱 × 轨迹 × 方向反转 | 仅 Tier A 分析（已完成或在手） |
| **机制层** | (c) 几何→过程→结局链，作显式假设框架；off-KD control + 双 ER 及时且干净 → **升级共同主贡献** | Tier B |
| **条件副贡献** | (a′) LoRA-regime 与全参文献的差异 | A09 结局（两分支都可发表） |
| **方法贡献** | 测量学批判：OverlapLift 撤回 + bf16/LoRA 使权重方向指标失效 | 已完成（Cycle 08）+ A09 双轨差值 |

写作规则：主 claim 永远不依赖 Tier B 实验；条件贡献在正文显式携带其条件。

## (a′) LoRA-vs-全参：决策树与"过程/终态"双轨

**框架（E1，用户提出并采纳）：** bf16 vs fp32 = **终态 vs 过程**。bf16 部署件是终态（交付的模型真实携带的更新——TPNT 的生态效度对象）；fp32 更新是过程（训练在几何上真正做了什么——归因干净的对象）。LoRA-vs-全参归因问题住在**过程侧**；双轨都报告，两轨之差量化 bf16 对方向指标的扭曲（方法贡献的数据）。

**数值秩证据（E2，记录在案）：** 测得 ΔW numerical rank ≈ 2121，而 LoRA r=32 更新 BA 秩 ≤ 32。测量对象 `bf16(W0+BA) − bf16(W0)` 的秩结构被满秩量化阶梯噪声主导；ρ² ≈ 0.02 的"弥散"是 **bf16 噪声地板的签名而非 LoRA 的**——干净 BA 应高度集中于 32 个方向。（"弥散来自 LoRA 结构"的初始猜想在 QA 讨论中被此证据反转。）

**A09 执行序（E2）：** 第一步——找到 Cycle 08 两臂的 LoRA adapter 文件（两臂均 LoRA r=32），直接以 **fp32(B@A)** 计算 ρ：零训练成本，消掉测量级混淆（adapter 相对舍入 ~1e-3 量级 vs merge−subtract 相对 W0 量级）。仅当 adapter 不可得、或需排除训练级 bf16 效应时才重训。

**决策树（预注册）：**
- fp32/干净 BA 仍 on-principal（与全参文献相反）→ "更新几何是参数化依赖的，非方法内禀"——对 2606.×2 普适性的边界修正（其 off-principal 是全参结论，不能外推到 LoRA 部署实践）。(a′) 成立。两个必写澄清：(i) 与 LoRA intruder-dimensions 的调和——"大部分弥散 + 轻微 on 偏置"与少量 intruder 方向可共存；(ii) LoRA 差异点**只对 2606.×2 成立**——2605.30524 自己就是 LoRA。
- fp32 翻转 off-principal（与全参文献一致）→ "bf16 量化足以翻转方向判定"——方法警告强化；(a′) 消亡，诚实报告。
- **A09 出结果前：不做任何 on/off-principal 声明。**

## off-policy KD control（Q3 = 变体 i）

OPD 与 SFT 之间有三个混淆：输入分布（on-policy rollout vs 静态语料）、监督信号（稠密 teacher logits vs one-hot 文本）、teacher 身份（8B logits vs 32B 生成文本）。control 臂设计：同 8B teacher、同 forward_kl_topk loss、同 LoRA r=32、同 checkpoint 网格——唯一改变是输入从 student rollout 换成 **teacher 离线生成的静态回复**（经典 off-policy 序列级 KD）。off-KD vs OPD 干净隔离 on-policy 采样；off-KD vs SFT（密度+teacher 身份）仍混淆，留 Cycle 10。工程注：无 rollout 环路，teacher top-k 可离线预计算——比 OPD 臂便宜。

**解释树（预注册）：**

| off-KD 签名 | 结论 |
|---|---|
| SFT-like（L18 bump + 早期 off-principal + OOD 侵蚀） | on-policy 采样是 OPD 签名的驱动（H-onpolicy 支持） |
| OPD-like（无 bump + OOD 保持） | 稠密 logit 监督已足够，on-policy 非驱动——与 2606.07082 权重侧 control 汇合；强化 Cycle 10 数据筛选猜想 |
| 混合（如无 bump 但 OOD 侵蚀） | bump 与 OOD 保持的驱动因素解耦——信息量最大，写作最谨慎 |

三种结局都有论文写法。

## 双 ER 实验（Q6）：与 2605.30524 的正面对话

策略 = "构念差异"打底 + "潜在反例"进攻。同批 checkpoint（三臂）同时计算：我们的白化 GetSlice ER（p ∝ σ，SVD-LLM lens）与他们的 raw 中心化协方差 ER（λ = σ²，d-归一化，多域探针——与 OOD 域探针需求合并）。构念规范见 `geometry_metric_definitions.md`（Cycle 09 节）。

**三结局树（预注册）：**
1. raw ER 下 OPD 也压更多且 OOD 更好 → **对 2605.30524 headline 的直接反例/精修**（"压缩量不是 OOD 损伤的度量，暂态破坏才是"）——最强结局。
2. 只有白化 ER 有判别信号 → 信号特定于输出相关谱；措辞降为"观测量选择敏感"。
3. raw ER 复现"压缩→坏"、白化 ER 反转 → 两种压缩构念分工；仍新颖，须讲清分工机制。

## 旋转 vs 值裁决（Q5）：回应 2509.12235

辩护前提：白化算子依赖输入协方差，权重方向旋转会在白化激活谱上表现为谱形变化——我们的 ER 不是纯"值"统计量。但必须实证而非断言：θ_r（激活主夹角，UV 已开）与 ER 比赛谁更好地 (i) 分离两臂 (ii) 跟踪 MMLU-Pro Δ。完整规格（r-网格、bootstrap、探针电池）见 `geometry_metric_definitions.md`。

**裁决树：** θ_r 更好 → 采纳"旋转+值分布"双轴观测框架，引 2509.12235 为跨空间收敛（他们是权重侧）；ER 仍是激活侧判别量 → 实证回应（"白化激活空间中判别监督通道的是值分布暂态而非子空间旋转"）；互补分工（如 θ_r 管 OOD、ER 管 dip）→ 观测量分工框架。

## Cycle 09 claim 边界（Q7，10 条）

1. A09 之前：不做任何 on/off-principal 权重方向声明（含 LoRA 侧）；不引用已撤回的 OverlapLift。
2. off-KD control 出结果前：不把 OPD 签名归因于 on-policy；机制语言保持非因果（co-locate / consistent with）。
3. 双 ER 之前：与 2605.30524 的方向反转只能写"白化构念下观察到相反方向，构念可比性待判定"。
4. θ_r 之前：ER 的地位是"当前唯一干净分离两臂的已测观测量"，不是"正确的观测量"。
5. 任何 "compression" 声明必须限定构念（白化输出相关 vs raw 表示性）；禁用裸的"压缩好/坏"。
6. L18：选层标准（判别效应最大）预注册且必须报告全层剖面；禁止只报 L18。
7. n 的诚实：单模型/单数据集/单 seed/臂数 ≤3——所有主张限定 "in this setting"；CI 落地前，n=10 轨迹相关只作 diagnostic。
8. 能力 margin 必须伴随 C09 输出控制分解与推理成本说明（OPD step_624 回复 ~4k tokens；16384 cap 下 trunc 0.92）。
9. 论文正文不得出现 "Cycle N" 叙事。
10. 不比较 1.7B 旧系列与 4B 系列的数字。

## 执行纪律（Q8 + E5）

| Tier | 内容 | 性质 | 论文角色 |
|---|---|---|---|
| **A**（第 1–2 周，阻塞主线） | B09 CI；双 ER；L18 层级重推导 + dip 共位裁决；UV 重探针 + θ_r；C09 输出控制分解 | 纯分析 | 主贡献内核 |
| **B**（并行，尽力赶） | A09（adapter-first，重训为 fallback）；off-KD control 臂 | ≤2 次训练 | (a′) + (c) 升级通道；迟到 → 自动降级，不 block 提交 |
| **C**（Tier A 落地后） | theory 定稿 + 按分层 framing 重写论文 | 写作 | — |

**应急预案（预注册）：** Tier B 任一未及 → 对应贡献自动降级（(a′) → future work；(c) 保持 hypothesis）；Tier A 分析与提交决策不等 Tier B。

**mini-cycle 协议（E5）：** Tier A 分析走 theory → coder → theory 快速环（"mini-cycle09"）；coder 产物落 `local_experiment_results/cycle_09.../mini/` 保 provenance；theory 可持 provisional reading。**护栏：** 任何读数进入 claims_allowed / current_picture 前必须过 Result 整合（可在 Tier A 收尾一次性过）——快速环加速理论迭代，不豁免 claim 门禁。Tier B 训练走完整 theory → exp → coder → result 链。

## Mini-Round 1 裁决（2026-07-08，E5 快速环——provisional，待 Result 整合）

输入：`local_experiment_results/cycle_09.../run_01/mini/`（T1–T4，F1–F5 flag）。按预注册裁决树执行；本轮探测层仅 {9, 18, 27}。

### R1（H1/F3 → 按 E1 规则）："SFT 早期 off-principal 重定向"叙事在过程轨上**退役**

三条 adapter-product（BA）轨——`sft_fp32_BA`、`sft_bf16_BA`、`opd_top32_approx`——在**所有步、两臂全部弱 on-principal**（ρ²_U ≈ 0.02–0.026，z +5..+9，k=32）。bf16-merged 轨的 SFT step_5 off（z=−6.7）在任何 BA 轨都不复现。由 `sft_bf16_BA` ≈ `sft_fp32_BA`（更新本身的 bf16 取整无害）可知：污染源是 **merge−相减-vs-base 流水线**，且集中在更新最小的早期步。

按 E1（归因住过程侧）：护城河的"ρ: step_5–10 off → 翻 on"这条腿作为 claim **退役**，改写为**部署轨测量发现**并入方法贡献（"bf16 部署件上的 merge−subtract 会翻转早期步方向判定"）。三空间链重构为：

```text
证据腿：   激活暂态（层级 ER uptick，两臂皆有）
          + 行为（dip 时序/恢复；能力 CI；OOD 结局）
零假设/批判腿：干净轨权重方向不判别两通道（k=32 下两臂皆弱 on-principal）
          ——反而强化 framing (b) 的论点：判别观测量在激活谱，不在权重方向。
```

OPD 干净轨只是近似（adapter 已 prune；merged−base 的 top-32 SVD）；OPD 过程轨的最终裁定等 A09 重训——**A09 优先级上调**。

### R2（H2/F2 → Q4 分支触发）：dip 共位真实存在且层限定；"OPD 无 bump"是 mean 伪迹

L18 与 L9 上**两臂都有** dip 共位的 ER uptick（L18：OPD +1.3 @step_5；SFT +3.6 @step_20）；L27 都没有。D08 的"OPD 单调收缩无 bump"是 L27 单调下降在均值里冲掉了中层小峰。Q4 预注册的**"共位保留、限定中层"**分支触发；"双通道 dip 几何异质"分支不触发。机制表述**统一化**：

```text
两种监督通道的能力 dip 都与中层 ER uptick 共位。
通道差异在于：暂态幅度（L18 上 SFT ≈ 3× OPD）、时序（step_20 vs step_5）、
恢复跨度（step_160 vs step_20）、终态压缩深度（L27 分歧，OPD 更低）。
跟踪 OOD 结局的是暂态幅度——不是压缩量
（SFT 大破坏 → MMLU-Pro 侵蚀；OPD 小破坏 → 保持）。
```

→ Result 同步：`current_picture` 的 "both dips co-locate" 在限定中层后**正确**；D08 "no bump" 措辞须改为层限定。

### R2b（F1 → 层规则已由用户拍板，2026-07-08）

"判别效应最大"判据被**否决**（用户裁定）：它被终态 ER 分歧主导（选出 L27），错过了论文真正关心的暂态结构。预注册判据改为**"dip 暂态最清晰"**（ER-uptick ↔ 能力 dip 共位 + 可见恢复），据此选出 **L18 为唯一 headline 层**（L9 confirmatory）。L27 的终态分歧（40.5）仍作为压缩→OOD 叙事内的支持证据报告——不再是第二个 headline。护栏：(i) 全 36 层剖面（round 2）仍强制——既是反 cherry-pick 展品，也检查未探测层是否有更清晰的暂态（若有则并报；预注册的是判据而非层号）；(ii) L18 uptick 幅度很小（+1.3 / +3.6 ER）——round 2 必须给 ER 附上探针 bootstrap 误差带，之后才能把 uptick 写成已确立。

### R3（H3 → E2 修正；(a′) 重塑）：弥散是 LoRA 的真实属性，不是 bf16 噪声

干净 fp32 秩-32 的 SFT BA 有 ρ²_U(k=32) ≈ 0.024——与 bf16-merged 同水平。E2 的预期（"纯 BA 应集中 → 高 ρ²"）被**证伪**：LoRA 的 32 个方向大部分在 base top-32 子空间之外，只带小而稳健（z>5）的 on 偏置。E2 对 merge−subtract 噪声的诊断（秩 2121 vs 真 32）仍成立。**(a′) 状态**：SFT 干净轨 = 弱 on-principal，与外部全参 SFT-on 结论**一致**——SFT 侧无矛盾。LoRA-vs-全参矛盾现在完全悬于 OPD 的干净 BA → A09 裁决。

### R4（H4/F4 → 按 B09 护栏）："更深"从 dip 叙事中退役

Paired CI（B=4096；draws 256/1024/4096 稳定）：final +0.096 [+0.058,+0.134] ✓、peak +0.114 ✓、AUC +84.1 ✓、非终止峰 +0.180 ✓——全部排除 0。**dip 深度 +0.036 [−0.018,+0.090] 含 0。** 有支持的 dip 叙事是：OPD 的暂态**更早**（step_5 vs step_20）且**更短**（step_20 已恢复 vs step_160）——不是"更深/更锐"。能力优势 claim 现已有 CI 背书。

### 稳定新增（进 provisional picture）

- C09 分解：OPD 后期精度主要由**截断但已 boxed** 样本承载（非终止 0.9+，截断样本 85–92% 在截断前已 boxed，均长 ~15k @16384；finisher acc 0.93–1.00）。论文必须带 Q7#8 推理成本说明。
- CI 实践：draws 256/1024/4096 几乎无差 → 小 draw 数即可，附录报稳定性（E4 精神）。

### Claim 边界增量（追加到 Q7 清单）

11. 不声称"OPD dip 比 SFT 更深/更锐"——CI 含 0；只可写更早/更短。
12. 不在过程轨上声称"SFT 早期 off-principal 重定向"——那是 merge−subtract 测量伪迹；只能作为部署轨方法学发现使用。
13. 任何 ER bump / no-bump 表述必须层限定；禁止 mean-over-layers 的 bump 声明（均值掩盖了两臂中层小峰）。
14. A09 干净 BA 落地前，不做任何 OPD 过程轨方向声明（当前 OPD 轨是 top-32 近似）。

### Round-2 优先级（按裁决重排）

1. **A09 OPD 干净 BA 重训**（H1/H3 悬于此）——Tier B，现为最高。
2. UV 重探针 + θ_r，聚焦 L18/L9 dip 步：旋转能否在干净轨 ρ 不判别处判别？
3. 全 36 层剖面（验证双镜头规则）。
4. 双 ER（不变）；off-KD control（不变）。

## 补遗（2026-07-10）：2509.12235 精读——威胁 #2 降级；干预实验推迟

对 arXiv 2509.12235 v3 全文精读（PDF 在工作区根目录）得到三层证据评估，此后按此执行 engage：

1. **观测层（扎实）**：SFT/RL 期间 Σ 变化 ≤0.005（其 App F.2）；权重侧主夹角 top 秩（k≤8）≈0°、k=32–384 <1°、到 k≈512 才 ~13.9°（其 Table 7）。旋转真实存在但住在中尾秩段，顶部权重子空间几乎不动。
2. **Protected-SFT 层（preliminary 但是真干预）**：训练时对 top-k 奇异向量子空间正交化可保 late-SFT 的 OOD（其 Table 6、Fig 11）——前向干预，无回退混淆。作者自标 preliminary（App A）。
3. **恢复层（§4.3，Figs 5–8——不支持因果主张）**：在其自测 Σ 稳定性下，换向量代数上 ≈ 把被分析矩阵（仅 W_QKV+MLP；o_proj/embedding/LN 留在 End）回滚到 MaxOOD checkpoint；换值 ≈ 恒等。其自身数字印证：恢复后 LLaMA 剖面落在 MaxOOD 剖面上（ID→~20 vs MaxOOD 时代 ~25–30；OOD→~15–17 vs MaxOOD 17.5；Qwen 锚点见其 Table 3：MaxOOD ID≈20–25/OOD≈17.5–19.7，End ID 46.6/OOD 14.1）。无回退对照、无幅度匹配对照；Qwen 上部分回滚甚至不恢复 OOD。**"OOD 跟旋转不跟值"的强因果版本只由这一层支撑——该层不成立。**

**Engage 措辞（论文用）**：承认第 1、2 层；批评第 3 层"在其自测谱稳定性（≤0.005）下与部分 checkpoint 回退不可区分，缺回退对照与幅度匹配对照"。

**后果**：
- 对我们 ER 观测量的"旋转 vs 值"压力降级：回应 = 上述分析批评 + 我们自己的双空间数据（激活侧 θ_r 以 17–20° 判别两臂；权重侧 θ_r^W 在 LoRA 下按 Davis–Kahan 预期 ≈0——跨空间对照本身支持观测空间论点，并与其 Table 7 顶秩 ≈0° 相呼应）。
- **因子化权重空间干预（交换实验）推迟（用户裁定，2026-07-10）**：它干预的是零假设腿而非理论核心观测量；其合理位置是理论主体完成后的"白化空间 vs 参数空间"对照，且届时规模可小得多。与 protected-SFT 式前向干预（训练中压制中层暂态——Tier B，属训练）一并记为 post-theory 候选。分析批评本身无需实验。

## Mini-Round 3 临时裁决（2026-07-11——待 coder handin + Result 整合）

输入：`mini/R3_*` 产物。按预注册树执行；Q1–Q5 讨论产生的新标准已记入 `geometry_metric_definitions.md` Round-3 增补节。

**P-R1 —— L18 故事定稿（样本级）。** 真探针样本 bootstrap（n=32 窗口 × 256 draws）：OPD step_5 uptick +0.83 [0.65, 1.08]，SFT step_20 +4.76 [3.70, 5.70]；**14 个模块格全部排除 0**。幅度比 ≈5.7×（修正 r2 的 ~10× 估计）。模块定位：MLP（gate/up/down）+ o_proj 承载暂态，k/v 最小。R2 的共位主张现已达到预注册证据标准。

**P-R2 —— 同域 eviction 假说证伪；break-not-movement 在 OOD 域内成立。** X 条件化白化 ER（L18 终点）：OPD 在 OOD 条件化下收缩最多（X_ood −21.3 > X_math −17.3）却保持 OOD 行为；SFT 几乎不收缩 X_ood（−4.7）却侵蚀——且 SFT 的 X_ood 谱在 step_20–40 有自己的小峰。解释裁定：**ER 是组织度统计量，不是容量计量表**；OPD 的全域平滑收缩是函数级重组的签名（与向 teacher 计算收拢的蒸馏机制一致），不是域级删除。方向分辨的封口（保留子空间是否覆盖 OOD 所需方向）待 R3-8（tail/r_ε + 输出漂移 + 输出空间分解）。新记录的最大臂分歧：L27 X_math_hard——OPD −86.7 vs SFT **+17.4（膨胀）**。

**P-R3 —— θ_w 近地板证实；双空间主张按 Q4 标准拆分。** 权重侧旋转：SFT 至 step_40 处于 1.23° identity 地板，624 时 2.35°；OPD 近似 624 时 4.60°（top-32 caveat）。与激活侧（17–23°）的**量级对照**以描述性成立；任何"观测空间优越性"主张等 R3-9 配对判别力检验。见下 claim 边界 #15。

**P-R4 —— IFEval 成为第二根判别性 OOD 轴（与 MMLU-Pro 同向）。** prompt_strict：OPD 0.272→0.316（+4.4 pts，峰 0.355@160）；SFT 0.272→0.214（−5.7 pts）。TruthfulQA 平（preservation check，与 1.7B 时代一致）。IFEval 维持 preservation 轴预注册身份——佐证性证据，不升主 gate（Cycle 08 禁项 #9）。

**P-R5 —— ID 轴补全（统一 cap）。** numina @12288（配对试点证明 cap-robust；N=200）：OPD 0.64/0.67/0.61 vs SFT 0.54/0.535/0.56（step 40/160/624）——OPD 每个测量步领先（+0.05~+0.135）。**AIME24 avg@10**（N=30，10 种子，cap 24576，secondary）：OPD 0.270@320 / 0.240@624 vs SFT 0.187@624（+0.053）。强制 caveat：AIME 被截断主导（OPD trunc 0.97、SFT 0.80），margin 部分由输出控制承载（boxed-before-trunc：OPD 0.52 vs SFT 0.025）——引用必须携带 C09 分解，不得作为干净能力数。

**P-R6 —— dip 步附近的行为微暂态（exploratory）。** IFEval 局部最低：OPD 0.251@step_10（几何 dip 后一格）、SFT 0.253@step_20（正在其几何 dip 步）；SFT MMLU-Pro dip @40；SFT 的 S/X/H H/ood ER 暂态 −45.7 @step_20。跨空间对齐 suggestive，但单个 delta ~1σ（IFEval SE≈2 pts，单种子）——待 R3-10（prompt 级配对 bootstrap + 暂态共位矩阵）；exploratory，永不作 gate。

### Claim 边界增量（追加到 Q7 清单）

15. "观测空间优越性"（whitened vs raw；激活 θ vs 权重 θ）在配对同尺度判别力检验通过前不得声称（规格见 `geometry_metric_definitions.md`）；此前此类表述限于描述性（"构念特定信号"、"量级对照"）。
16. 方向分辨指标（R3-8）落地前，不得作任一方向的 eviction 假说声明：仅凭 ER 收缩无法说明哪些方向幸存。
17. 行为暂态共位（P-R6）为 exploratory；R3-10 统计出来前不得写成已确认的跨空间链。
18. AIME24 数字凡引用必须内联截断/输出控制 caveat（两臂 trunc ≥0.8；margin 部分由 boxed-before-trunc 承载）。

### Round-4 修订（2026-07-12，窗口 v2 设计讨论）

- **P-R1 口径修订**：L18 uptick 误差带的测量基底是首窗（题面+解答开头）探针——内部有效，但按证据分层裁定降为**先导层**；论文级地位需 v2 随机窗重推导（预期稳健，但预期须验证）。
- **P-R2 修订**：跨域收缩排序（"OPD 在 OOD 条件化下收缩最深，−21.3 > −17.3"）被探针文本性质混淆（生成 CoT vs 题面文本）——排序主张撤回；承重事实是**探针内**的：E_ood 上的臂对照（OPD −21.3 vs SFT −4.7）和各探针的轨迹形状。同域 eviction 证伪由探针内证据支撑，仍然成立。
- Claim 边界 19：论文中的探针类数字必须来自 v2 窗口标准（规格见 `geometry_metric_definitions.md` Round-4 节）；v1 几何读数只能以先导层身份出现并注明窗口口径。
- Claim 边界 20：不同文本性质探针（S 型 vs E 型）之间禁止跨域收缩幅度比较。
