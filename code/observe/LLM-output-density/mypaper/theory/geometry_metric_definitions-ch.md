# Geometry 指标定义

```yaml
view_type: current_metric_definition_ch
status: active_cycle08_metric_definition
primary_code: /root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/geometry_metrics.py
main_historical_definition: (snapshot removed 2026-06-16)
last_organized: 2026-06-29
```

## 范围

这些指标用于总结 GetSlice 产生的 activation-conditioned whitened module matrices 的奇异谱。它们是 descriptive mechanism/stability signals。当前 claim gates 中，drift 是 confirmatory stability metric；effective rank、spectral gap 和 X-S spectrum level gap 是辅助但关键的机制证据。

设 `sigma = (sigma_1, ..., sigma_n)` 是去除非有限值后的非负奇异谱，`sigma0` 是同一 layer/module 下 theta0 的 diagonal S-probe baseline spectrum。log-space 指标使用 `eps = 1e-12`。

## 设计动机：为什么用谱视角，为什么是这四个指标

本项目需要一个"训练让模型内部表征移动了多少"的机制级信号，而不只是输出准确率，因为 Cycle 03/04 的核心问题是关于**稳定性**（同样学到这么多东西的前提下，OPD 是否比 SFT 移动得更少），单看输出准确率无法区分"没有移动"和"移动了但落到了同样好的地方"。选择对 activation-conditioned 矩阵做奇异谱分析，而不是用别的替代方案（比如原始权重范数距离、梯度范数、或输出 token KL 散度），是因为相关的微调/遗忘研究（`LoRA vs Full Fine-tuning: An Illusion of Equivalence`，见 `../related_work/current_source_matrix-ch.md`）显示奇异向量结构能暴露出标量距离指标看不出的、性质不同的更新行为（例如"intruder dimensions"）。GetSlice 是本项目工具链里现成的具体实现（怎么跑见 `../exp/experimental_methodology/geometry_protocol-ch.md`）。

在这个选择之下，四个指标承担的是有意设计成互不重叠的不同角色：

- `spectral_drift_from_theta0_diag` 是**confirmatory**指标，因为它直接量化"模型离起点移动了多远"，这正是稳定性 gate 需要设阈值的对象。
- `effective_rank` 和 `spectral_gap` 是**辅助**指标，因为它们描述的是谱的*形状*（集中 vs 分散、主方向强弱），不是移动量本身——能区分"移动很多但谱依然 well-conditioned"和"移动很少但谱坍缩到更少方向"这两种 raw drift 本身分不清的情况。
- `X_S_spectrum_level_gap` 是辅助指标，因为它回答的是和前三者不同的问题：不是"这个模型变了多少"，而是"这个模型在 training-target 条件下的谱，离一个共享参照探针有多远"——这个指标原则上可以暴露一个模型是不是在过拟合自己的 S 来源（为什么不同模型角色的 S 不同，见 `../exp/experimental_methodology/geometry_protocol-ch.md`），而不是学到能泛化的东西。
- `principal_angle` 是计划中但目前算不出来的指标（默认不保存 UV/sketch artifacts）；它能提供其余四个指标都给不了的子空间重叠证据，这也是为什么它还留在表里而不是直接删掉。

## 指标

| Metric | 含义 | 公式 / 实现 | 解释边界 |
|---|---|---|---|
| `effective_rank` | 基于谱熵的维度展开程度。 | `p_i = sigma_i / sum_j sigma_j`; `exp(-sum_i p_i log p_i)`。 | 越高表示谱分散到更多方向；越低表示更集中。低并不自动更好。 |
| `spectral_gap` | 顶部奇异方向的主导程度。 | k=1 时为 `sigma_1 - sigma_2`。 | 可表示更强 leading direction，但只看谱顶部，需与 effective rank 一起读。 |
| `spectral_drift_from_theta0_diag` | 相对 theta0 的整体 log-spectrum movement。 | `sqrt(mean((log(sigma_t+eps)-log(sigma0+eps))**2))`。 | Cycle 04 Gate C 使用的稳定性指标。Raw drift 可能惩罚学得更多的方法，因此 Cycle 05 要结合 ID learning 解释。 |
| `X_S_spectrum_level_gap` | shared X probe 与 model S probe 的有符号 log-level gap。 | `mean(log(sigma_X+eps)-log(sigma_S+eps))`。 | 衡量 reference/general probe 与 training-target probe 的谱强度差；接近 0 可能表示更对齐，但它是辅助指标。 |
| `spectral_gap_abs_delta_from_start` | spectral gap 相对 theta0 的绝对变化。 | `abs(gap_1(sigma_t)-gap_1(sigma0))`。 | 早期设计指标；v2 后不总是单独落盘，可由 `spectral_gap` 和 theta0 row 重算。 |
| `principal_angle` | 奇异向量子空间 overlap。 | 需要保存 UV/sketch artifacts。 | Cycle 04 未保存，因此不能声称 principal-angle evidence。 |

## 实现与 artifact

- 指标函数：`/root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/geometry_metrics.py`
- Cycle 04 metric rows：`/root/LLM-output-density/local_experiment_results/cycle_04_opd_stability_gain/geometry_metrics.csv`
- Cycle 04 full singular spectra：`/root/autodl-tmp/cycle04_opd_stability_gain/tables/geometry_long.csv`

## 当前解释规则

预注册 geometry stability gate 使用 drift，但报告中必须同时呈现 effective rank、spectral gap 和 X-S gap 作为辅助 geometry evidence。若辅助指标与 drift verdict 不一致，应写成 mechanism signal mixed，而不是丢弃辅助指标。

---

## Cycle 06 解释说明（SFT-Only runs）

```yaml
cycle: cycle_06_sft_feasibility_and_degradation
date: 2026-06-23
qa_source: theory/QA_cycle06_theory.md (Q6=a)
```

### Diagnostic 地位

Cycle 06 SFT-only runs 中，所有 geometry 指标均为**诊断信号（diagnostic only）**。由于本轮不训练 OPD arm，这些指标不能支持 OPD 机制 claim。不得把 Cycle 06 geometry 结果用作"OPD 比 SFT 更稳定或更不稳定"的证据。

允许的用途：可用 geometry 检验 representation 变化是否与 SFT learning（Gate A06 pass 场景）或退化（Gate B06 场景）同向。这些是相关性信号，不是机制证明。

### lower effective rank：解释边界

SFT 后 effective rank 降低**不自动等同于稳定性信号**。它也可能意味着：
- 有害的表征集中或 collapse
- 任务特化但泛化能力下降

任何时候都应把 effective rank 和实际获得的 ID learning 量一起解释。如果 effective rank 下降但主轴 ID gain 未出现，应视为警告信号（可能是有害 collapse），而不是稳定的证据。

### Cycle 06 drift 参照点

Cycle 04 和 05 使用 `theta0` 作为 drift 参照点（`spectral_drift_from_theta0_diag`）。对于 Cycle 06：

- 如果训练从本地 base checkpoint 或官方 `Qwen3-1.7B-Base` 开始，而非从 `theta0` 开始，则 `spectral_drift_from_theta0_diag` 不再是合适的主 drift 指标。
- Experiment Design 在运行 geometry 之前必须明确指定参照 checkpoint。
- 如果同时计算 base-relative 和 theta0-relative drift，必须清晰标注（`drift_from_base`、`drift_from_theta0`），不得混淆。
- Raw drift 仍会惩罚学得更多的方法（K1-base-aware P4）；报告 drift 时必须同时报告 ID learning 量。

---

## Cycle 07 更新：轨迹读法与子空间稳定性指标

```yaml
cycle: cycle_07_base_model_sft_feasibility
date: 2026-06-24
qa_source: theory/QA_cycle07_theory.md (Q5)
governs: 仅理论定义与解释；实现（改 k 参数、算 stable rank / energy ratio、多 checkpoint
         采样）归 exp/experimental_methodology/geometry_protocol.md
```

### 从静态终点到轨迹

Cycle 04/05 把几何读作**静态终点对照**（"OPD 降 effective rank，SFT 升"）。Cycle 07 三方框架（见 `current_theory_update.md` Cycle 07 节）要求把每个几何指标**沿 SFT 轨迹多 checkpoint 读**。单看终点 checkpoint 会丢失 dip 暂态，而这正是三方框架要研究的结构。这适用于**全部四个指标**，不只 effective rank 和 drift —— `spectral_gap`（现升级为 γ_r，见下）和 `X_S_spectrum_level_gap` 同样纳入轨迹读法。

### 子空间稳定性：从 top-1 gap 到 eigengap γ_r（Davis–Kahan）

Cycle 04 用的 `spectral_gap(sigma, 1) = σ_1 − σ_2` 只衡量**单个**主方向相对第二的领先，判断不了**子空间**的稳定性。主子空间的稳定性是 top-r 个奇异方向的**联合**属性，不是 top-1 向量的属性。

**Davis–Kahan 界（主导数学）：** 对两个矩阵 M 与 M+E，它们 top-r 奇异子空间之间的最大主夹角 θ_r 满足

```
sin θ_r  ≤  ‖E‖ / (σ_r − σ_{r+1})
                   └──── 第 r 处的 eigengap，记作 γ_r ────┘
```

所以 eigengap **γ_r = σ_r − σ_{r+1}** 是 top-r 子空间稳定性的标准代理：γ_r 越大，top-r 子空间在训练扰动下越难旋转。这是 top-1 gap 的正确推广，也**正是项目此前记忆中的 "gamma"**（2026-06-24 确认；此前并未实现任何 `gamma` 指标 —— `gamma_s.jsonl` 是 OPD 的 S-probe 存储文件，无关）。操作上，γ_r 就是把 `spectral_gap` 的 `k=1` 改成 `k=r`。

**r 怎么定：** r 是主子空间维度，按 (layer, module) 自适应 —— 例如捕获 ~90% 谱能量的维度（energy-ratio@r），或 `round(effective_rank)` —— 而非全局拍一个 k。

### 新辅助指标（用现有谱即可算，无需 UV）

| 指标 | 公式 | 角色 |
|---|---|---|
| `eigengap_gamma_r` | σ_r − σ_{r+1}（即 `spectral_gap(sigma, r)`） | top-r 子空间稳定性的**容量代理**（Davis–Kahan 分母） |
| `stable_rank` | ‖A‖_F² / σ_1² = Σ σ_i² / σ_1² | 连续"有效维度"；抗噪；补充熵版 `effective_rank`（无需选 k） |
| `energy_ratio_at_r` | Σ_{i≤r} σ_i² / Σ σ_i² | top-r 子空间捕获的谱能量占比；也用于定 r |

这些只用已采集的奇异谱，无需重跑 GetSlice 或保存 UV。

### Principal angle：Cycle 07 后继续延后（用户决定 2026-06-24；2026-06-29 更新）

**直接、实测**的子空间稳定性量是 principal angle θ_r 本身（Davis–Kahan 左边），它比较两个 checkpoint 实际的 top-r 子空间。它需要保存 UV/sketch artifact。

**Cycle 07 决定：** Cycle 07 **不开 UV 保存**；principal angle 以及是否开 UV 延后到 Cycle 07 之后再讨论。**Cycle 08 更新决定：** OPD-vs-SFT OverlapLift 比较按 matched-LoRA 的相对轨迹比较推进；principal-angle / subspace-rotation 解释注册到 Cycle 09，而不在 Cycle 08 使用。后果与边界：

- γ_r 只是**容量代理**（gap 大 ⇒ 子空间*能*抵抗旋转），由单个模型的谱算出。它**不是**子空间在 checkpoint 间实际旋转了多少的测量。
- 因此 Cycle 07 可以用 γ_r 描述子空间稳定性*容量*，但不能做实测子空间旋转 claim。不要把 γ_r 当成 θ_r 写。

### 暂态峰假设（待 Cycle 07 多 checkpoint 数据检验）

三方框架下，几何签名预期是**沿轨迹的暂态结构**，不是终点水平：

- **SFT_dip：** effective rank / drift-from-base 可能暂态升高，γ_r 塌缩（主子空间变得 ill-defined / 易旋转）—— shallow imitation 的几何签名。
- **SFT_recover：** effective rank / drift 稳定，γ_r 恢复 —— internalized。
- **（Cycle 08，H-OPD）：** OPD 是否压平这个暂态峰，即全程让 γ_r 不塌、effective rank 不冲高？

这是对长期 open question "lower effective rank 是稳定信号还是有害 collapse" 的机制化回答 —— 取决于所处优化阶段：dip 阶段 effective rank 暂态冲高 + γ_r 塌缩是有害不稳定，不是稳定。**这是待 Cycle 07 多 checkpoint 数据检验的假设，不是已确立的读法。**

### 提醒 Experiment Design

`geometry_metric_definitions.md` 是理论（定义 + 解释）。以下实现项归 `geometry_protocol.md`，须由 Experiment Design 决定 / 预注册：

1. 用 `k = r`（按 layer/module 自适应）调用 `spectral_gap`，并在 `geometry_metrics.py` 实现 `stable_rank` 和 `energy_ratio_at_r`。
2. 沿每条 SFT 轨迹的多个 checkpoint 采样几何指标**和 `response_length`**（不只终点）。
3. drift 从 **base** checkpoint（Qwen3-4B-Base）量，不从 `theta0`。
4. Cycle 07 UV 保存保持**关闭**；Cycle 08 将 principal-angle / measured subspace-rotation 解释注册到 Cycle 09。


---

## Cycle 08 更新：matched-LoRA 轨迹下的 OverlapLift

```yaml
cycle: cycle_08_h_opd_vs_sft_comparison
date: 2026-06-29
qa_source: theory/QA_cycle08_theory.md (Q6, Q9, Q10)
governs: 只管理论解释；实现与 artifact 保存决策归 exp/code
```

### 允许 OPD-vs-SFT 相对比较

Cycle 08 在同一 model、data family、checkpoint grid、LoRA regime 下比较 OPD 与 SFT。因此，在 matched-LoRA 设计下，OverlapLift trajectories 可以直接作为 **OPD-vs-SFT 相对 geometry signal** 比较。核心问题是：

```text
OPD 的 teacher-subspace overlap trajectory 是否系统性不同于 matched SFT trajectory，
尤其是在 SFT dip 与 recovery landmarks 附近？
```

这不需要先有单独的 LoRA-aware random baseline 才能比较。因为 OPD 与 SFT 的 adaptation regime 已经 matched，LoRA 因素在两者之间被控制。

### Random-LoRA baseline caveat

Random-LoRA baseline 可能归一化 update count、adapter footprint 或某些 training-budget 特征，但不一定归一化真实训练目标诱导的低秩结构。因此，Cycle 07 的 SFT-only OverlapLift < 1 且单调上升，不能单独升级为绝对 on/off-principal verdict。

Cycle 08 通过同 LoRA regime 下的 OPD-vs-SFT 对比来回答相对问题。但如果要声称“这个更新相对正确 random/full-rank reference 是 on-principal”，仍必须显式定义 reference。

### Principal angle 留到 Cycle 09

Principal-angle 或 measured subspace-rotation claim 需要保存 UV/sketch artifacts，并显式定义 top-r。Cycle 08 theory 不做这些 claim，也不把它们作为 OPD-vs-SFT OverlapLift 比较的前置条件。

该问题注册到 Cycle 09：届时再决定是否保存所需 artifact，以及如何定义 top-r subspaces，再作直接 subspace-rotation claim。

### Gate 地位

OverlapLift 只有在与行为结果一致时才支持机制解释：

- 若 B08/C08 支持 OPD，且 OverlapLift 相对 SFT 更有利，则 A08 加强 H-OPD story。
- 若 B08 严重失败，有利的 OverlapLift 不能拯救 capability failure。
- 若 B08 pass 但 OverlapLift mixed，则报告 mechanism signal mixed，而不是强行贴 on/off-principal 标签。

> **Cycle 08 事后注（2026-07-05，为连续性记录）：** OverlapLift 作为 on/off-principal 信号已**撤回**（与 scale-matched random-LoRA null 不可区分；magnitude/bf16 驱动），由尺度不变的 ρ 替代（`result/cycle08_rho_metric_spec.md`）。测量修正见下方 Cycle 09 节。

---

## Cycle 09 更新：层级化读法、实测旋转（θ_r）、双 ER 构念、探针电池 v2.1

```yaml
cycle: cycle_09_aaai_competitiveness_completion
date: 2026-07-08
qa_source: theory/QA_cycle09_theory.md（Q4/Q5/Q6 + E1–E4）
governs: 仅定义与解释；实现归 exp/code
supersedes: "principal angle 推迟" —— UV 保存现已开启（用户决定 D2，2026-07-08）
```

### 层级化读法规则（D3 / Q4）

几何主视角从 mean-over-layers 改为**层级化**：

1. **Headline 层** = OPD-vs-SFT 判别效应最大的层（当前 L18）。选层标准预注册为"判别效应最大"，且**必须同时报告全层剖面**（反 cherry-pick）。外部锚：2605.30524 同样发现塌缩集中于中后层——中层 headline 有跨论文一致性。
2. mean-over-layers 表降为附录汇总。（Cycle 08 D08 的判别信号——SFT L18 step_20 rank bump vs OPD 无 bump——是层级发现，mean 表会稀释它。）

### dip 共位裁决程序（Q4 —— 已记录的矛盾）

**矛盾（2026-07-08 发现）：** `result/current_picture.md` 写 "Both dips co-locate with the ER uptick"，但 mean-over-layers 表（`opd_vs_sft_geometry.md`）显示 OPD effective_rank 从 step_0 单调下降（argmax = step_0，无 uptick）；OPD 侧离其 step_5 dip 最近的量是 **xs_log_spectrum_gap argmin（step_10）**，不是 ER 上凸。

**程序（预注册）：** 在 headline 层用四个量（ER、xs_gap、drift、θ_r）同时重推导两臂 dip 的共位关系：

- OPD dip 在 L18 层级有 ER uptick → 保留"共位"表述（限定 headline 层）；
- 仍没有 → 机制故事改写为**双通道 dip 几何异质**（SFT：ER-bump 型；OPD：xs-gap 型——两种几何上不同的暂态），并标记 Result 修正 `current_picture.md`。

两个分支都可发表；第二个可能更有趣。裁决前禁止把两臂 dip 签名平均成一个故事。

### θ_r 激活：实测子空间旋转（D2 / Q5 + E4）

Cycle 09 所有探针运行**开启 UV/sketch 保存**（三臂 × 全 checkpoint 网格）——纯探针成本，**无需重训**。

- **定义：** θ_r = top-r 激活奇异子空间之间的主夹角，(i) 相邻 checkpoint 间、(ii) 各 checkpoint vs base，逐层 × 逐模块。
- **r-网格（E4）：** r ∈ {8, 16, 32, 64, 128} ∪ {energy@80%, 90%, 95%}。主 r = energy@90%（headline）；θ_r(r) 完整稳健性曲线进附录。与 ρ 报 k=32/128 双档同一逻辑，堵死"你挑了 r"。
- **不确定性（E4）：** 对探针样本 bootstrap → θ_r 置信带；外加一次性探针数敏感扫描（如 n_samples ∈ {8,16,32,64}）——用户先前消融显示不敏感；复核一次后固定 n，作为附录稳健性证据。
- **预测（预注册）：** SFT 在 headline 层 step_10–20 附近出现 θ_r 旋转尖峰（与 ER bump 共位）；OPD 平滑。
- **裁决树（回应 2509.12235）：** 比较 ER 与 θ_r 谁更好地 (i) 分离两臂 (ii) 跟踪 MMLU-Pro Δ。θ_r 更好 → 采纳旋转+值分布双轴框架，引 2509.12235 为跨空间收敛（他们是权重侧）；ER 更好 → 实证回应："白化激活空间中判别监督通道的是值分布暂态而非子空间旋转"；分工 → 观测量分工框架。
- **γ_r 与 θ_r：** γ_r（eigengap）仍是*容量*代理；θ_r 是*实测*旋转。两者都报告；禁止把 γ_r 写成 θ_r。

### 双 ER 构念规范（Q6）

两种 ER 构念在同批 checkpoint、同批探针上并列计算。任何"压缩"声明必须注明所用构念。

| | 本项目（白化） | 2605.30524（raw） |
|---|---|---|
| 输入 | GetSlice 白化激活条件模块矩阵（Cholesky-jitter 协方差白化） | raw 残差流 hidden state，仅中心化 |
| 谱权重 | p_i ∝ σ_i | p_i ∝ λ_i = σ_i² |
| 归一化 | 无 | × 1/d（范围 [0,1]） |
| 粒度 | 逐模块（q/k/v/o/gate/up/down）× 层 | 整层残差流 |
| 探针 | 探针电池（见下），含各臂 S | 固定多域探针，teacher-forced |
| 含义 | **输出相关谱**——白化使 σ_i 对应该层输出误差最优截断（SVD-LLM，arXiv 2403.07378——正式方法学引用） | 表示铺开度 |

对比的三结局树（预注册）见 `current_theory_update.md`（Cycle 09 节）。

### 探针电池 v2.1（E3）

全部 X 探针冻结、版本化、跨臂跨 checkpoint 共用；纯探针成本。

| 探针 | 域 | 状态 | 服务 |
|---|---|---|---|
| `X_math`（MATH500 式） | ID | 沿用 | 主线轨迹 |
| `X_math_hard`（AIME 式；建议用历史 AIME 避免与 eval 集耦合） | ID-难 | **新增** | 难度分层 ID；**generation-free，绕过 AIME24 非终止失效**，给难数学 regime 一条可观测通道 |
| `X_bos` | 无条件 | 沿用 | 参照 |
| `X_ood_knowledge`（MMLU-Pro 式） | OOD 知识 | **新增，最高优先** | 把压缩→OOD 从跨域推断变成同域观测 |
| `X_teacher`（8B teacher 固定回复集） | 监督源 | **新增** | 三臂共用一把监督源尺子；off-KD control 的 teacher 对齐动力学 |
| `X_general`（开放域文本） | 通用 | **新增** | 与 2605.30524 多域探针的构念可比性 |
| `S`（各臂沿用；off-KD 臂 S = 其 teacher 静态训练回复） | 训练目标 | 沿用 | xs_gap |

预算紧时优先级：X_ood_knowledge > X_teacher > X_general ≈ X_math_hard。

### 权重侧测量修正（E1 / E2 —— 指针）

ρ（尺度不变权重方向对齐）规格见 `result/cycle08_rho_metric_spec.md`。Cycle 09 修正（因改变指标计算方式而记录于此）：

- **双轨（E1）：** bf16 = 终态（部署件真实携带的更新，TPNT-conform）；fp32 = 过程（训练在几何上真正做的事，归因干净）。双轨都报告；两轨之差量化 bf16 对方向指标的扭曲。
- **测量修正（E2）：** ΔW 改为**直接从保存的 LoRA adapter 计算 B@A**（fp32 运算；bf16 存储的 adapter 可接受——舍入相对 adapter 元素量级 ~1e-3），替代 `bf16(W0+BA) − bf16(W0)`（舍入相对 W0，约为更新的百倍）。
- **数值秩证据：** 测得 ΔW 秩 ≈ 2121 vs LoRA 真实秩 ≤ 32——merge−subtract 对象的秩结构被量化噪声主导；ρ² ≈ 0.02 的"弥散"是噪声地板签名，非 LoRA 的。
- **A09 顺序：** adapter 优先（零训练成本）；仅当 adapter 不可得或需排除训练级 bf16 效应时才重训。

### 提醒 Experiment Design（Cycle 09）

1. 所有探针运行开启 UV/sketch 保存；预算存储。
2. 建探针电池 v2.1（三臂 × 网格）；版本化并冻结。
3. 实现 θ_r：r-网格 + 探针 bootstrap + 一次探针数扫描。
4. 在同批探针输出上计算两种 ER 构念。
5. A09：先找 Cycle 08 adapter 文件；fp32(B@A) 先于任何重训。
6. Required artifacts：全层剖面（不只 L18）、θ_r(r) 曲线、双 ER 表、headline 层的各臂 dip 共位表。

### Mini-Round 1 测量裁定（2026-07-08——provisional，待 Result 整合）

来源：`local_experiment_results/cycle_09.../run_01/mini/`（T1/T4）。以下改变两个指标的计算与读法：

1. **ρ 估计量规则：** LoRA 更新方向的**必用估计量是 adapter 乘积 BA**。merge−subtract 对象 `bf16(W0+BA) − bf16(W0)` 会翻转早期步方向判定（SFT step_5：merged 轨 off z=−6.7 vs 干净 BA 轨 on z=+7.9）并抬高数值秩（≈2121 vs 真 ≤32）。**adapter 本身的 bf16 取整无害**（`sft_bf16_BA` ≈ `sft_fp32_BA`）——污染源是小更新量级下的 merge−subtract 流水线。双轨报告不变：BA = 过程/claim 轨；merged = 部署/生态轨。
2. **E2 修正（记录）：** 干净秩-32 BA 上 ρ²_U(k=32) ≈ 0.022–0.026——相对 base top-k 子空间的弥散是 LoRA-regime 的真实属性（带小而稳健的 on 偏置，z>5），不是 bf16 噪声地板。
3. **OPD 干净 BA 缺口：** OPD adapter 已被 prune；`opd_top32_approx`（merged−base 的 top-32 SVD）只是**近似**——A09 重训给出真 adapter 前，不做任何 OPD 过程轨方向声明。
4. **层限定规则（用户 2026-07-08 拍板；取代上文单一 headline 表述）：** 禁止 mean-over-layers 的 ER bump/no-bump 声明——均值掩盖了 L9/L18 上**两臂都有**的 dip 共位小峰（L18：OPD +1.3、SFT +3.6 ER），而 L27 单调下降。Headline 层 = **L18**，选层判据预注册为**"dip 暂态最清晰（uptick↔dip 共位 + 可见恢复）"**——不用"终态判别效应最大"（它会选 L27 并错过暂态）。L9 = confirmatory；L27 终态分歧 = 压缩→OOD 故事内的支持证据，不是第二 headline。全 36 层剖面（round 2）仍强制（反 cherry-pick + 检查是否有更清晰的暂态层；预注册的是判据而非层号）；ER uptick 须有探针 bootstrap 误差带后才可写成已确立。
5. **CI 实践：** paired-bootstrap CI 在 draws {256, 1024, 4096} 间稳定；小 draw 数即可——附录报稳定性检查。

### S/X/H 探针语义（Round 3——新轴，用户 2026-07-09 提出，2026-07-10 正式化）

把**监督通道编码进探针语义**的新分类法。它与旧 S/X 设计**并存**（不取代）：旧探针仍是全部既有裁决（L18 uptick、θ_r、双 ER）的基础；S/X/H 开的是新观测轴。**两套分类法的 ER 值禁止混用——条件化不同，分表报告。**

| 角色 | 定义 | OPD 臂 | SFT 臂 | 域 |
|---|---|---|---|---|
| **S**（source/初始） | base（step_0）的生成——初始输出分布，跨臂共享 | base 生成 | 同（共享） | 各探针域 |
| **X**（训练信号） | 各臂*真实*的监督文本 | 该 checkpoint **自己的 on-policy rollout**（逐 checkpoint 重新生成） | **数据集 CoT**（固定外部文本；= SFT 的旧 S） | 仅训练域（math） |
| **H**（half：产生但非训练） | checkpoint 自生成、但**未用于训练**的文本 | 非训练域（MMLU-Pro 式、通用…） | math 自生成 + 非训练域（SFT 训的是数据集，不是自己的输出） | 各域 |

设计收益：(i) on-policy vs off-policy 的监督差异**直接编码进探针**——OPD 的 X 是自生成、SFT 的 X 是外部固定、两臂的 H 都是自生成非训练；(ii) 修复旧不对称 caveat（旧 S 对 OPD 是 off-target，OPD 的真训练信号是 rollout）。

**预注册解释规则：**
1. dip / OOD-drop 的读数必须**在 S、X、H 三种条件化谱下联合报告**（用户要求）：如 L18 暂态出现在初始分布（S）下、只在训练信号（X）下、还是在生成非训练文本（H）下？
2. 本轮该轴**仅诊断**——不得单独由 S/X/H 得出机制 claim。它可服务的两个预注册假设：(a) OPD 的 X 条件化谱是否跟踪其 rollout 分布随训练的漂移；(b) 非训练域的 H 条件化谱使"压缩挤占未训练方向"假设可同域观测。
3. 生成参数（采样、长度上限、各域 n）必须冻结、版本化并记入 manifest；X_OPD 与 H 需要**逐 checkpoint 生成**——provenance 强制。
4. 可比性护栏：S/X/H 自带基线列（step_0 按构造即 S）；轨迹增量只在分类法内部计算。

### 输出精度指标 + 判别力标准（Round-3 增补，2026-07-11）

动机：Q1 缺口（ER 是形状摘要——分不清"剪掉了任务所需方向"和"把同样功能整合进更少方向"）+ 用户的 SVD-LLM 原生提议。三个新指标：

1. **域条件化尾部能量 / 功能秩。**
   `tail_energy(r) = Σ_{i>r} σ_i² / Σ σ_i²` = 1 − `energy_ratio_at_r`（Cycle 07 定义，改为域条件化解读）；在 SVD-LLM 白化下这**就是**该域上 rank-r 压缩的输出精度损失。
   `功能秩 r_ε = 使 tail_energy(r) ≤ ε 的最小 r`——该域的功能在输出误差容忍度 ε 下真正需要的方向数。
2. **模块输出漂移**（含方向信息，弥补 log-谱 drift 的方向盲）：`‖ΔW·X‖_F / ‖W₀·X‖_F` 逐（模块, 层, 探针域）——模块在该域上的实际输出变化量。
   **零重探针成本：** `‖ΔW·X‖_F² = tr(ΔWᵀ ΔW Σ_X)`，Σ_X 由 R3-4 战役已存的 per-probe 输入因子构造，ΔW = BA 在手。
3. **输出空间分解**（"激活侧 ρ"，可选）：`‖ΔW·X‖²` 能量落在 base top-r 输出方向内 vs 尾部的比例——"收缩有没有碰到该域所需方向"的方向分辨检验。

**预注册预测（R3-8 运行前）：**
- P-tail-1：SFT 在 X_ood 上 step_20 出现 tail_energy / r_ε 暂态尖峰（与其 ER uptick 共位）；OPD 的 r_ε(X_ood) 平滑下降。
- P-tail-2（eviction 检验）：若 OPD 的 OOD 条件化收缩在删除 OOD 所需方向，其限制在 base top-r_ε(X_ood) 子空间内的输出漂移应不成比例地大；良性整合读法预测——尽管总漂移更大，OPD 的 top 子空间限制漂移与 SFT 相当。

**判别力标准（用户裁定 2026-07-11，Q4）：**
"A 空间比 B 空间更能判别两臂"**不得**由"有/无"模式或跨构念的原始量级（如 whitened −15.8 vs raw −0.11，尺度不可比）声称。要求的检验：

```text
在同一批探针样本上（whitened 与 raw 来自同一次前向），
每个空间计算无量纲判别力统计量
   d_space = |mean(OPD) − mean(SFT)| / 合并 bootstrap SD，
再对配对差 d_whitened − d_raw 做探针样本 bootstrap；
优越性声明要求该差的 95% CI 排除 0。
```

通过之前：(i) whitened-vs-raw 的表述停留在"构念特定信号"（T8 措辞）；(ii) 权重-vs-激活 θ 对照拆为**量级对照**（17–23° vs ≤2–5° 含 1.2° identity 地板——描述性，成立）与**判别力对照**（需两侧 θ 的误差带——待做）。适用于论文中所有"观测空间优越性"语言。

---

## Round-4 规格（2026-07-12）：探针命名 S/E/X/H、窗口 v2、指标套件定稿

2026-07-11/12 设计讨论定案（用户裁定）。**取代 v1 窗口机制与 Round-3 的 S/X/H 实例化，作为一切论文级测量的标准。**

### 探针命名（定稿）：四型

| 型 | 定义 | 旧名映射 |
|---|---|---|
| **S_d** | base 生成的域 d 文本——冻结、跨臂共享 | S_math = 旧 `X_math`/X_base（base rollout！）；S_bos = 旧 `X_bos`；S_ood / S_general = base 在 E_d prompt 池上的生成 |
| **E_d** | 外部固定语料（无生成段） | E_ood = MMLU-Pro 题面+选项（14 子科，seed42，128 条）；E_general = wikitext 切片（128 条）；E_math_hard = AIME 题面；E_math（教材式数学文本）= 可选、缓做 |
| **X_arm** | 真实训练信号 | X_OPD = 逐 ckpt 自身 math rollout；X_SFT = 数据集 CoT（v2 中题面 mask） |
| **H_d** | 逐 ckpt 自生成，**仅非训练域，两臂对称** | H_ood、H_general、**H_bos（新增）**；SFT 的 math 自生成从 H 移除（用户裁定） |

**跨域幅度 caveat**：跨域收缩排序（如 E_ood −21.3 vs S_math −17.3）被文本性质混淆（生成 CoT vs 题面文本）。承重比较是**探针内**的：同一探针上的臂对照与轨迹。

### 窗口 v2（长度解耦设计）

1. 窗长 512，**整体落在生成区内**（prompt 留在前向上下文、不进统计）。E 型无生成段——全文可用。
2. **每样本固定 k=3 个均匀随机偏移窗**（生成不足 3×512 时允许重叠；记录实际 k）。窗种子与生成种子分开。每窗记录相对位置 → 前/中/后三分箱诊断免费获得。
3. 生成类探针（S / X_OPD / H）：**≥3 个生成种子批**，报批间 mean±SD。
4. **层级归一化（权重与长度解耦的规则）**：窗级 token 均值 gram `(1/n_w)XᵀX` → 样本级对其各窗取均值 → 语料级**样本等权均值**。估计对象 = 样本等权处理剖面，与 `response_length`（自有预注册变量）刻意正交。k 只影响估计*精度*，永不影响*权重*。
5. 白化矩阵由层级化协方差构造（Σ 的常数缩放不改变 ER/tail/EC 等比值量）。
6. **Bootstrap 单元 = 样本**（窗嵌套其内）——升级 Round-3 的 `probe_window` 单元。

### 证据分层规则（用户裁定）

v2 = 论文级测量标准。v1 生成类探针（Round-3 S/X/H）：superseded。固定文本结果——含 L18 uptick 样本级误差带——降为**先导层（pilot tier）**：内部有效（文本跨臂跨 ckpt 恒同），口径改写为"题面+解答开头窗"；论文级地位需 **v2 随机窗重推导**。行为评测、ρ、θ_w、能力 CI 与窗口无关，维持原级。v2 战役前必做 **窗口构成审计**（v1 各探针格首窗的 prompt-token 占比，CPU）。

### 指标套件——定稿

**M1：tail_energy / r_ε + 有效压缩（EC）判据。**
逐（模块, 层, 臂, 步, 探针）：`tail(r) = Σ_{i>r}σ²/Σσ²`；r_ε 取 ε∈{0.05, 0.01}。白化双轨：**逐 ckpt**（主——当前函数的压缩预算，SVD-LLM 忠实语义）与**冻结 base**（副——纯权重侧视角；两轨之差量化输入协方差漂移）。用户提出的 EC 判据：

```text
EC（相对 base，同一探针）:=  r_ε(t) < r_ε(0)              （冗余收缩）
                          且 drift_core 小，               （核心稳定）
     drift_core = RMS[ log σ̂_i(t) − log σ̂_i(0) ]，i ≤ r_ε，σ̂ = σ/Σσ
```

仅为奇异值代理——具体破坏了什么由 M3 检验。预注册：OPD 终点在各域满足 EC；SFT dip 步暂态违反核心稳定（与 uptick 共位）；SFT 终点近中性。

**M2：模块输出漂移。** `‖ΔW·X‖_F / ‖W₀·X‖_F`，**raw**（不白化）输入；X₀（base 因子）为主参照、X_t 为副；由 `tr(ΔWᵀΔW·Σ_X)` 从已存因子 + BA 计算（OPD = top-32 近似，caveat 内联）。模块局部量；端到端补充 = **M2b 表示漂移** `‖h_t − h₀‖/‖h₀‖`（同文本）。

**M3：保留与去向——U/V 双侧。** 由 base 白化矩阵 M₀ = W₀S₀ = U₀Σ₀V₀ᵀ（逐探针域）；r 与 M1 的 r_ε 衔接。

| 侧 | 保留率 | 去向 | 回答什么 |
|---|---|---|---|
| U（输出） | e_keep_U(r) = ‖U₀ᵣᵀW_tX₀‖²/‖U₀ᵣᵀW₀X₀‖² | φ_U = ‖U₀ᵣᵀΔWX₀‖²/‖ΔWX₀‖² | 还在*写*下游需要的输出吗？ |
| V（输入） | e_keep_V(r) = ‖W_tS₀V₀ᵣ‖²/‖W₀S₀V₀ᵣ‖² | φ_V = ‖ΔWS₀V₀ᵣ‖²/‖ΔWS₀‖² | 还在*读*该域的主输入方向吗？ |

联读：φ 高 + e_keep≈1 → 所需方向内精化；φ 高 + e_keep↓ → 覆写；e_keep↓ → eviction（V 侧失聪是更可能的遗忘机制；接 2509.12235 的 V 旋转与我们的 θ_V）。整族活在**冻结 base 基底**——冻结副轨存在的结构性理由。预注册（P-tail-2 定稿版）：良性整合 → OPD 的 e_keep_U/V(E_ood) ≈ 1 而 ER 收缩；eviction → e_keep 下降。
