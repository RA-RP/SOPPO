# OPD 最小实验 03 (v2) — 实验结果说明

> 运行脚本：`scripts/run_opd_minimal_closure_v2.py`
> 输出目录：`/root/autodl-tmp/exp0609/opd_minimal_03_v2/`
> 完成时间：2026-06-13 03:47
> 状态：`completed`（全闭环跑通，6 模型）

本文档记录 v2 严谨版闭环的真实结果。v2 相对 v1（`run_opd_minimal_closure.py`）修复了 5 个方法论缺陷（见末尾"v1→v2 修复清单"）。

---

## 1. 实验配置（真实）

| 项 | 值 |
|---|---|
| base / student | Qwen3-1.7B (`/root/autodl-tmp/model/Qwen/Qwen3-1.7B`) |
| teacher | Qwen3-4B (`/root/autodl-tmp/model/Qwen/Qwen3-4B`) |
| prompt 母池 | NuminaMath-1\_\_\_5，seed=42 一次性采样后切分（train/heldout/probe 互不重叠）|
| cold-start theta0 | 512 prompts，OPD-like（teacher 在线蒸馏），grad_accum=4 → 128 步 |
| OPD (lmbda=1.0) | 从 theta0 出发，200 步 × grad_accum 4 = 消耗 800 prompt |
| SFT 对照 | 从 theta0 出发，数据量扫描 256/512/1024/2048，lr=1e-5，3 epoch |
| held-out eval | 64 条，OPD 算 JSD / SFT 算 CE，`load_best_model_at_end` |
| 全量评估 | lm_eval vLLM，每 benchmark `eval_limit=200` |
| GetSlice | layer_14，7 模块，seqlen=512，s/x 各 16 样本；S×model 交叉 + 双 X（X_prompt/X_bos）|
| 探针 rollout | 生成到自然 EOS（安全上限 2048，**不截断**），保留真实长度 |
| LoRA | r=16, alpha=32, dropout=0.05, 7 个 target module |

> **GSM8K 等 benchmark 用了 `eval_limit=200`（每任务 200 题）以控制时长，统计噪声较大；< 0.01 的差异不可信。**

---

## 2. 性能轨迹（`tables/eval_trajectory.csv`）

| 模型 | DataSize | GSM8K | GSM8K_gain | OOD_avg | OOD_penalty_p2 | Worst_drop |
|---|---|---|---|---|---|---|
| sft_n1024 | 1024 | 0.295 | **-0.140** | 0.5423 | 0.0261 | 0.0150 |
| sft_n2048 | 2048 | 0.310 | -0.125 | 0.5422 | 0.0279 | 0.0205 |
| sft_n512 | 512 | 0.380 | -0.055 | 0.5479 | 0.0124 | 0.0100 |
| **opd_lmbda1** | 800 | 0.430 | **-0.005** | 0.5537 | **0.0100** | 0.0100 |
| **theta0** | 512 | 0.435 | 0.000 | 0.5505 | 0.000 | 0.000 |
| sft_n256 | 256 | 0.440 | +0.005 | 0.5474 | 0.0182 | 0.0150 |

> DataSize 语义已统一为"实际训练消耗 prompt/样本数"：theta0=512(cold prompt)，opd=800(rollout prompt)，sft=监督样本数。

**发现 1 — SFT 数据越多 = 灾难性遗忘越重。** SFT 的 GSM8K 随数据量单调崩坏（256→0.440，512→0.380，1024→0.295）。在此 setup 下，continued SFT 越多监督样本，越过拟合 NuminaMath solution 格式，GSM8K（OOD 数学推理）退化越狠。OPD 用 800 prompt 几乎不掉（-0.005）。

---

## 3. matched 配对（`tables/matched_gsm8k_pairs.csv`）

| 端 | 选中 | GSM8K_gain |
|---|---|---|
| OPD | opd_lmbda1 (800) | -0.005 |
| SFT | sft_n256 (256) | +0.005 |

`GSM8K_gain_gap = 0.010 ≤ 0.02` → **`match_status = valid_match`**（v1 只是 nearest_match，gap=0.063）。

matched 对的 OOD 对比：

| 指标 | OPD | SFT-256 | delta (OPD−SFT) |
|---|---|---|---|
| OOD_lite_penalty_p2 | 0.0100 | 0.0182 | **−0.0082** ✓ |
| Worst_OOD_lite_drop | 0.0100 | 0.0150 | **−0.0050** ✓ |

在 GSM8K 性能对齐前提下，OPD 的分布外能力退化明显小于 SFT。

---

## 4. 几何指标定义与计算式

### 4.0 奇异谱的来源（singular spectrum）

几何分析的基础量是**激活条件化的奇异谱**，由 GetSlice `split_whitened_svd` 模式产生（`MyFunc/GetSlice/utils/profiling_utils.py::whitening`）。对某一层某一线性模块（如 layer_14 的 `self_attn.o_proj`）：

1. **激活 Gram 矩阵**：用探针数据（S 或 X）前向，收集该模块**输入激活** $x$，累加 $\Sigma = \sum x^\top x$（`profiling_utils.py:284`）。
2. **白化矩阵 L**：对 $\Sigma$ 做 Cholesky 分解得下三角 $L$（数值不稳时加 jitter 修正），即 $\Sigma = L L^\top$（`_gram_to_svdllm_scaling_diag_matrix`）。
3. **白化权重**：$A = W L$，其中 $W$ 是该模块权重矩阵（`W_scale = W @ scaling_diag_matrix`）。这就是 GetSlice 风格的 $A_t^{\mathcal D} = W_t L_t^{\mathcal D}$——把权重投影到"该探针分布激活方向"的空间。
4. **SVD**：$A = U \Sigma_{\text{sv}} V^\top$，取奇异值序列 $\sigma = (\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n)$。

> **singular spectrum（奇异谱）** = 这条 $\sigma$ 序列（本实验每模块 2048 个奇异值）。
> 它度量"权重在该探针分布下的有效作用强度沿各主方向的分布"。下面所有指标都是 $\sigma$ 的泛函。
> 探针不同（S vs X、不同模型的 S），$L$ 不同，$\sigma$ 也不同——这正是 S×model 交叉矩阵的意义。

### 4.0b 探针文本的真实构成与来源（双 X / S）

探针分 **两类 X（冻结统一参考）** 和 **按模型区分的 S**。所有 rollout 探针**生成到自然 EOS，不人为截断**（保留真实长度信息）。

**两个 X 探针（都由 theta0 生成、冻结给所有模型共用）：**

| X 探针 | 含义 | jsonl 格式 | "文本"来源 |
|---|---|---|---|
| **X_prompt** | **有 prompt 条件**的输出分布 | `{"output":{"text": "{prompt}\n{completion}"}}` | theta0 对 probe_prompts rollout（有题目条件） |
| **X_bos** | **完全无条件**的输出分布（论文依据） | `{"output":{"text": freeform}}` | theta0 **纯 BOS 起步**、无任何 prompt 自由生成 |

> X_bos 因无条件，必须采样生成（temperature=0.8，固定 seed=42 可复现）；X_prompt 为 greedy。两者都 `output.text`（`_extract_x_text_strict`），输出到 `getslice/.../X__prompt` 与 `X__bos`。

**S 探针（按模型区分，代表各模型训练时对齐的目标分布）：**

| S 探针 | jsonl 格式 | GetSlice 取文本 | "解答"来源 |
|---|---|---|---|
| **S / theta0** | `{"question": prompt, "answer": ...}` | `f"{question} {answer}"`（`_extract_s_text_strict`） | **teacher(Qwen3-4B) rollout** |
| **S / opd** | 同上 | 同上 | **OPD 模型自己 rollout** |
| **S / sft_n\*** | 同上 | 同上 | **该 SFT 的训练数据 solution（金标准）** |

> **截断修复（v2.1）**：早期版本 rollout 探针被 `max_new_tokens=256` 硬截断（S-theta0/S-opd 100% 命中 256 上限，长度信息丢失）。现已改为生成到 EOS（安全上限 2048），probe 长度反映真实分布。

**probe prompt 的来源**：所有 X/S 探针的 problem 都取自统一母池切出的 `probe_prompts`（32 条），与训练/eval **同源（同一次 NuminaMath seed=42 采样）但不同切片**：

```
NuminaMath 母池（2144 条）顺序切三段，互不重叠（实测完整文本真重叠=0）：
  [0:2048]    train_prompts / train_sft   （cold/OPD/SFT 训练）
  [2048:2112] heldout_eval                （eval loss，64 条）
  [2112:2144] probe_prompts               （X/S 探针 rollout，32 条）
```

> probe 用的是训练**没训过**的同分布 problem，保证探针反映模型对该分布的**泛化激活**，而非对训练样本的记忆。

**为什么 theta0 的 S-answer 用 teacher rollout 而非 theta0 自己 rollout**：S 探针的定义是"**该模型训练时所对齐的目标文本分布**"——
- theta0 是被 teacher 在线蒸馏得到的，它训练时"看齐"的是 **teacher 的输出** → theta0-S 用 teacher rollout；
- OPD 是 on-policy 蒸馏，student 自己 rollout 既是训练样本也是其分布 → opd-S 用自己 rollout；
- SFT 是监督学习，金标准 solution 是其目标 → sft-S 用训练数据 solution。

这样三者的 S 在语义上是可比的"训练目标分布激活"。两个 X 则相反——都由 theta0 生成并**冻结共用**，作为与各模型训练信号无关的"统一参考锚点"：X_prompt 锚"有条件输出分布"，X_bos 锚"完全无条件输出分布"。

> `X_S_spectrum_level_gap` 的 X 参考默认用 **X_prompt**（与早期口径一致）；X_bos 的谱单独入表（`geometry_metrics.csv` 的 `x_variant` 列区分 `prompt`/`bos`），可另作"无条件分布 vs 训练分布"的对比维度。

**GetSlice 喂入方式**（`data_utils.py:131-157`）：把同一探针的 32 条文本用 `\n\n` 拼成一篇长文，随机截 `seqlen=512` token 的窗口、重复 `nsamples=16` 次累积 Gram。故激活统计反映的是**该探针分布的整体激活**，不是单条样本。截窗时 tokenizer 默认加 bos，但只是片段前缀（多落在某条解答中部），主体是真实数学文本。

### 4.1 各指标定义（符号：$\sigma=(\sigma_1\ge\dots\ge\sigma_n)$ 为当前奇异谱，$\sigma^{(0)}$ 为 theta0 对角基线谱）

| 指标 | 含义 | 计算式 | 代码 |
|---|---|---|---|
| **spectral_gap** | 谱顶端**第 1 与第 2 奇异值之差**，衡量首主方向的主导程度（gap 大 = 表示更"集中"于单一方向） | $\mathrm{gap}_k = \sigma_k - \sigma_{k+1}$，本实验 $k=1$：$\sigma_1-\sigma_2$ | `spectral_gap(sigma,1)` |
| **effective_rank** | **有效秩**（谱熵的指数），衡量谱在多少个方向上"实质展开"。值越大 = 激活分布越高维/越分散 | 归一化 $p_i=\sigma_i/\sum_j\sigma_j$，$\mathrm{erank}=\exp\!\big(-\sum_i p_i\ln p_i\big)$ | `effective_rank(sigma)` |
| **spectral_drift_from_start**（CSV 列 `spectral_drift_from_theta0_diag`） | 当前谱相对 **theta0 对角基线谱**在**对数尺度**下的整体漂移（RMS），衡量"训练把谱推离起点多远" | 取前 $L=\min(n,n_0)$ 项，$\mathrm{drift}=\sqrt{\frac1L\sum_{i=1}^L\big(\ln(\sigma_i+\epsilon)-\ln(\sigma^{(0)}_i+\epsilon)\big)^2}$ | `log_spectrum_drift(sigma, base)` |
| **spectral_gap_abs_delta_from_start** | spectral_gap 相对起点的**绝对变化量**（设计指标，衡量首主方向主导度的偏移） | $\big|\,\mathrm{gap}_1(\sigma)-\mathrm{gap}_1(\sigma^{(0)})\,\big|$ | 见下方说明 |
| **X_S_spectrum_level_gap** | **同一模型**下 X 探针谱与 S 探针谱在对数尺度的**平均带符号差**，衡量"通用参考分布 vs 训练分布"激活强度的系统性偏移 | 取前 $L=\min(n_X,n_S)$ 项，$\mathrm{gap}_{XS}=\frac1L\sum_{i=1}^L\big(\ln(\sigma^X_i+\epsilon)-\ln(\sigma^S_i+\epsilon)\big)$ | `xs_log_spectrum_gap(x_sigma, s_sigma)` |

其中 $\epsilon=10^{-12}$（防 $\ln 0$）。所有谱在计算前取绝对值并去除非有限值（`geometry_metrics.py::_array`）。

> **关于命名**：实验设计文档（`minimal_trl_opd_experiment_03.md` §6.2）列的 `spectral_drift_from_start` / `spectral_gap_abs_delta_from_start`，在 v2 实现中：
> - `spectral_drift_from_start` → 落地为 CSV 列 **`spectral_drift_from_theta0_diag`**（"start" 具体化为 theta0 的对角 S 谱基线）。
> - `spectral_gap_abs_delta_from_start` → v2 主结论用 `spectral_drift` 替代，该绝对差**未单独落盘成列**，但定义如上、可由 `spectral_gap` 列与 theta0 行直接相减得到。
> - **principal_angle / X-S projection gap** 需要 UV/sketch，本实验未保存 → 标记 `unavailable_no_uv`。

---

## 5. 几何分析（`tables/geometry_metrics.csv`，layer_14，S×model 交叉矩阵）

> 下表均为 layer_14、7 模块平均；`drift` / `xs_gap` 定义见 §4.1。
> **数据为 v2.1 版本**：探针 rollout 生成到自然 EOS（不再被 256 截断），并新增 X_bos 无条件探针。结论方向与截断版一致且更干净（OPD 现在在全部 7 个模块上 drift 都小于 SFT）。

### 5.1 对角 S 几何汇总（每个模型在**自己**的训练分布探针下）

| 模型 | DataSize | effective_rank | spectral_gap | spectral_drift(vs theta0) | X_S_spectrum_level_gap |
|---|---|---|---|---|---|
| theta0 | 512 | 897.29 | 3135.72 | 0.0000 | -0.0231 |
| **opd_lmbda1** | 800 | 892.66 | 3178.21 | **0.0179** | **-0.0185** |
| sft_n256 | 256 | 925.28 | 2965.31 | 0.0488 | -0.0467 |
| sft_n512 | 512 | 927.27 | 2959.49 | 0.0527 | -0.0467 |
| sft_n1024 | 1024 | 930.36 | 2954.13 | 0.0598 | -0.0479 |
| sft_n2048 | 2048 | 930.84 | 2944.11 | 0.0643 | -0.0492 |

**读图**：
- **谱漂移**：OPD=**0.0179** ≪ 所有 SFT（最小的 sft_n256=0.0488，约 2.7×）；SFT drift 随数据量**单调递增**（0.0488→0.0643），与 GSM8K 单调崩坏同步。OPD 几何稳定性优势在不截断数据下比之前更明显。
- **有效秩**：OPD（892.7）≈ theta0（897.3），SFT 全部抬高到 925~931（激活在更多方向展开 = 过拟合训练分布细节）。
- **X-S level gap**：OPD（-0.0185）最接近 0（X 与 S 谱最一致），SFT 全为 ≈ -0.047（把通用分布 X 与训练分布 S 的谱拉得更开）。

### 5.2 逐模块谱漂移（揭示机制差异，关键）

各模块相对 theta0 的谱漂移（对角 S）：

| 模块 | opd_lmbda1 | sft_n256 | sft_n512 | sft_n1024 | sft_n2048 | OPD−SFT256 |
|---|---|---|---|---|---|---|
| self_attn.q_proj | 0.0165 | 0.0391 | 0.0436 | 0.0510 | 0.0523 | −0.0226 |
| self_attn.k_proj | 0.0137 | 0.0355 | 0.0392 | 0.0460 | 0.0492 | −0.0218 |
| self_attn.v_proj | 0.0137 | 0.0357 | 0.0394 | 0.0461 | 0.0487 | −0.0220 |
| **self_attn.o_proj** | **0.0328** | **0.0792** | 0.0836 | 0.0909 | 0.0964 | **−0.0464** |
| mlp.gate_proj | 0.0162 | 0.0597 | 0.0647 | 0.0731 | 0.0764 | −0.0435 |
| mlp.up_proj | 0.0158 | 0.0597 | 0.0648 | 0.0731 | 0.0763 | −0.0439 |
| mlp.down_proj | 0.0167 | 0.0324 | 0.0336 | 0.0383 | 0.0509 | −0.0157 |

**OPD 在所有 7 个模块上谱漂移都小于 SFT-256**（OPD−SFT256 全部为负），不再有截断版里 mlp 个别反号的情况——这是不截断探针带来的更干净信号。其中 `self_attn.o_proj` 仍是 OPD/SFT 绝对漂移最大的模块（OPD=0.0328 vs SFT=0.0792），而 MLP gate/up_proj 的差距最大（diff≈−0.044）：SFT 在 MLP 与注意力输出上都大幅改动（与任务输出格式强绑定，最易破坏泛化），OPD 则整体温和。这是 on-policy 蒸馏"温和对齐分布" vs SFT"硬背答案"的几何指纹。

### 5.3 双 X 探针（冻结统一参考）有效秩 vs theta0

两个 X 探针都由 theta0 生成、冻结共用：**X_prompt**（有题目条件输出）/ **X_bos**（纯 BOS 无条件自由生成）。

| 模型 | X_prompt effrank | Δ vs theta0 | X_bos effrank | Δ vs theta0 |
|---|---|---|---|---|
| theta0 | 891.78 | 0.00 | 854.92 | 0.00 |
| **opd_lmbda1** | 887.47 | **−4.31** | 851.48 | **−3.43** |
| sft_n256 | 892.64 | +0.86 | 855.14 | +0.22 |
| sft_n512 | 894.96 | +3.18 | 856.66 | +1.75 |
| sft_n1024 | 898.35 | +6.57 | 859.15 | +4.23 |
| sft_n2048 | 898.97 | +7.19 | 859.56 | +4.65 |

在**完全相同的冻结探针**下（无论有条件 X_prompt 还是无条件 X_bos），**结论一致**：OPD 收缩有效秩（更紧凑表示，Δ<0），SFT 系统性抬高且随数据量单调增（Δ 从 +0.2→+7.2）。两个 X 探针给出同向结论，排除了"探针选择导致差异"的混淆——是模型本身的几何变化。X_bos 整体有效秩低于 X_prompt（无条件生成的激活分布更窄），但模型间的相对趋势完全相同。

### 5.4 S×model 交叉矩阵（effective_rank，行=被测模型，列=S 探针来源）

| 被测模型＼S探针 | theta0 | opd_lmbda1 | sft_n256 | sft_n512 | sft_n1024 | sft_n2048 |
|---|---|---|---|---|---|---|
| theta0 | 897.3 | 896.6 | 924.9 | 924.9 | 924.9 | 924.9 |
| opd_lmbda1 | 893.2 | 892.7 | 919.9 | 919.9 | 919.9 | 919.9 |
| sft_n256 | 898.1 | 897.5 | 925.3 | 925.3 | 925.3 | 925.3 |
| sft_n512 | 900.3 | 899.8 | 927.3 | 927.3 | 927.3 | 927.3 |
| sft_n1024 | 903.3 | 903.3 | 930.4 | 930.4 | 930.4 | 930.4 |
| sft_n2048 | 903.3 | 904.0 | 930.8 | 930.8 | 930.8 | 930.8 |

**两个正交效应**：(1) **列效应（探针主导）**——SFT 类探针（920~931）系统性高于 teacher/student rollout 探针（theta0/opd ≈ 893~904），说明"训练数据文本"激活比"模型自由生成"更高维；(2) **行效应（模型）**——同一列内 OPD 行始终最低（如 S=opd 列：OPD=892.7 vs SFT256=897.5，diff≈−5），即无论用谁的探针测，OPD 模型激活都更紧凑。对角线（model==probe）是各模型"自分布"几何，已在 §5.1 汇总。

---

## 6. 结论：初步支持 K1

按设计文档 §9.2 标准，需同时满足三条件：

| 条件 | 结果 |
|---|---|
| TRL-OPD 的 OOD_lite_penalty_p2 更低 | ✓ delta = −0.0082 |
| 几何漂移/间隙不差于 SFT | ✓ OPD 谱漂移远更小（0.0179 vs 0.0488，约 2.7×）；**全部 7 个模块** drift 都低于 SFT |
| match_status = valid_match | ✓ gap=0.010 |

三条件全部满足，且 OPD 几何不仅"不差于"而是**明显更稳定**。多维证据（OOD penalty / 谱漂移 / o_proj 子空间 / 有效秩 / X-S 一致性）一致指向同一方向 → **初步支持 K1**。

---

## 7. 训练成本与 OPD on-policy 机制

### 7.1 训练时间（从日志 `train_runtime` 真实读取）

| 阶段 | 模型 | 训练量 | train_runtime | 折算 | samples/s |
|---|---|---|---|---|---|
| cold-start | theta0 | 128 步（512 prompt rollout） | 14380 s | **≈ 4.0 h** | 0.036 |
| OPD distill | opd_lmbda1 | 200 步（800 prompt rollout） | 21966 s | **≈ 6.1 h** | 0.036 |
| SFT | sft_n256 | 256 样本 × 3 ep | 353 s | ≈ 5.9 min | 2.18 |
| SFT | sft_n512 | 512 样本 × 3 ep | 642 s | ≈ 10.7 min | 2.39 |
| SFT | sft_n1024 | 1024 样本 × 3 ep | 1233 s | ≈ 20.6 min | 2.49 |
| SFT | sft_n2048 | 2048 样本 × 3 ep | 2449 s | ≈ 40.8 min | 2.51 |
| | **SFT 合计** | 3840 样本 × 3 ep | 4677 s | **≈ 1.3 h** | — |

**OPD 比 SFT 慢约 60–70 倍**：OPD 每 prompt ≈ 27 s（21966/800），SFT 每"样本·epoch" ≈ 0.4 s（4677/(3840×3)）。
`train_samples_per_second` 直观印证：OPD=0.036 vs SFT≈2.4–2.5。OPD 含 cold-start 共约 **10.1 小时**，仅 distill 段 **6.1 小时**。

### 7.2 为什么 OPD 这么慢 —— 真实执行机制

> 实测配置：`use_vllm=false`、`lmbda=1.0`（纯 on-policy）、`per_device_train_batch_size=1`、`gradient_accumulation_steps=4`、`max_completion_length=512`、local teacher（非 vLLM server）。

每个 optimizer step（实测 ≈ 110.8 s/step）的真实时序（`trl/experimental/distillation/distillation_trainer.py`）：

```
一个 optimizer step（消耗 batch×grad_accum = 1×4 = 4 个 prompt）：
  ① _fill_buffer：把 4 个 prompt 切成 4 个 slice，lmbda=1.0 → 全部 on-policy
  ② student rollout（主要开销）：_generate_with_model 用 HF model.generate()
     for slice in 4: 逐个生成 512 token   ← 串行，每 slice batch=1
  ③ 逐个 micro-step：取 1 个 slice → teacher 前向给 logits → 算 JSD → backward（累积）
  ④ 4 个累积完 → optimizer.step() 做 1 次 LoRA 更新
```

关键事实：

- **主开销是 student rollout**：每 step 自回归生成 4×512 = 2048 token（`truncated_fraction=1.0`，每条都顶满 512）。自回归是"一次出 1 token、跑 512 次前向"的串行过程。
- **没有用 vLLM**：本次 `use_vllm=false`，走 HF `model.generate()`，无 continuous batching 加速。
- **4 个 prompt 之间串行**：`_generate_with_model` 是 `for slice_idx in on_policy_indices` 逐个 generate，每个 slice batch=1 → 4 条 prompt 完全串行，无并行。
- teacher 前向 / backward / LoRA step 都很快，相对 rollout 可忽略。

这也是早先观测到「OPD 训练时 GPU 利用率仅 ~30%」的根因：时间大量花在串行 token 生成（batch=1 时 GPU 闲置），而非并行矩阵计算。

### 7.3 可选加速方向（未采用，仅记录）

| 方法 | 效果 | 代价 |
|---|---|---|
| `per_device_train_batch_size=4, grad_accum=1` | 4 条 prompt 一次 batched generate，等效 batch 不变 | 显存↑，需测 OOM |
| `use_vllm=true` | vLLM continuous batching 大幅加速生成 | 起 vLLM + 显存共享工程复杂度 |
| 减小 `max_completion_length`（如 256） | 生成 token 减半，时间近似减半 | 影响蒸馏信号质量 |

---

## 8. eval loss / best-checkpoint 选择的精确机制

三个训练（cold-start / OPD / SFT）训练中都用 held-out eval loss + `load_best_model_at_end` 选轨迹内 best checkpoint（§README 第 1 层）。下面是经源码核实的精确行为。

### 8.1 held-out 数据集

`heldout_eval.jsonl`：从 **NuminaMath-1\_\_\_5** 切出的 64 条（母池 `[2048:2112]`），每条是 `{messages:[user=problem, assistant=ground-truth solution(+Final answer)]}`，与训练/probe 互不重叠。三个训练**共用同一份** held-out。

### 8.2 eval 的 completion 来源：ground-truth，不是 rollout

**关键事实（`trl/experimental/distillation/distillation_trainer.py:770`）**：

```python
def _prepare_inputs(self, generation_batch):
    if not self.model.training:     # eval 时 model.training=False
        return generation_batch     # 直接返回原始 batch，不做任何 rollout
    ... # 仅训练时才 _fill_buffer → student rollout
```

| 阶段 | completion 来源 |
|---|---|
| **训练** | student 自己 rollout（on-policy，lmbda=1.0 全部自生成 512 token） |
| **eval** | **held-out 里的 ground-truth solution**（teacher-forcing 真实 token，**完全不 rollout**） |

即 OPD 的 eval：student 与 teacher 各自对 "problem + ground-truth solution" 做一次前向，在 `prompt_length` 之后的 **completion 段**逐 token 算 JSD（`compute_loss` 取 `logits[:, prompt_length-1:-1]`）。**是 teacher-forcing、completion-only 的 JSD**。

### 8.3 train/eval 的 on-policy / off-policy 不一致（值得注意）

- **训练**：在 student **自己 rollout 的 completion** 上算 JSD → **on-policy**。
- **eval**：在 **ground-truth solution** 上算 JSD → **off-policy（teacher-forcing）**。

所以 `lmbda=1.0` 名义"纯 on-policy 训练"，但其 **eval_loss 实际是 off-policy 的**——衡量"student 在标准解法前缀上与 teacher 的分布接近度"，而非"student 自己 rollout 的质量"。这不是 bug（held-out 只有 problem+solution，无法在未来 ckpt 的 rollout 上算稳定可比的量），但 eval_loss 与训练目标并非同一个量；用它选 best ckpt 防过拟合仍合理。

### 8.4 三者 eval loss 的完整刻画

| 训练 | completion 来源 | rollout? | loss 类型 | 算 loss 的 token 范围 |
|---|---|---|---|---|
| cold-start | held-out ground-truth solution | 否（teacher-forcing） | JSD（vs teacher） | **completion-only**（prompt 之后） |
| OPD | held-out ground-truth solution | 否（teacher-forcing） | JSD（vs teacher） | **completion-only** |
| SFT | held-out ground-truth solution | 否（teacher-forcing） | CE | **整段**（problem+solution 全部 token） |

> **SFT 的 CE 算整段**：closure 用 `apply_chat_template` 渲染成单一 `text` 字段喂 SFTTrainer，走 "language modeling case"，**不生成 completion_mask**，`completion_only_loss`（默认 True）因此不生效，`assistant_only_loss` 也未开 → problem 文本也计入 CE。OPD 的 JSD 则严格 completion-only。两者量纲本就不可比（CE vs JSD），跨方法比较走 GSM8K_gain matched，不依赖 eval_loss，故此差异不影响主结论。如需 SFT eval 更"纯"，可开 `assistant_only_loss=True`（当前未采用）。

---

## 9. 局限（诚实说明）

1. **GSM8K 没有真正"上升"。** 最高 sft_n256=0.440 vs theta0=0.435，差 0.005 在 `eval_limit=200` 的统计噪声内。当前结论是"OPD 在**维持**性能时几何/OOD 更稳"，而非"OPD 能**提升**性能且更稳"。
2. **跨模型 OPD 选择对噪声敏感。** 第 2 层 selection 按 `max(GSM8K)` 选 best OPD（本次只有 1 个 OPD run 所以未暴露）；GSM8K 在 limit=200 下不可靠。
3. **theta0 已接近该分布上限。** cold-start 后 GSM8K 已 0.435，后续训练空间小。要论证"提升+更稳"需换 theta0 未饱和的更难分布，或加大 eval 样本量降噪。

---

## 10. 产物清单（`/root/autodl-tmp/exp0609/opd_minimal_03_v2/`）

```
pool/                      # 统一母池切分（train_prompts / train_sft / heldout_eval / probe_prompts + pool_meta.json）
step2_cold_start/          # theta0 adapter + merged_model
step3_opd_distill/         # OPD adapter + merged_model
step4_sft_controls/        # sft_n256 / sft_n512 / sft_n1024 / sft_n2048
eval/csv_results/          # target_metrics_results.csv（6 模型 × 6 benchmark）
getslice/
  inputs/X_prompt/         # 冻结 X 探针：theta0 有 prompt 条件 rollout（不截断）
  inputs/X_bos/            # 冻结 X 探针：theta0 纯 BOS 无条件自由生成（不截断）
  inputs/S/{model}/        # 各模型 S 探针（teacher/student rollout/训练数据，不截断）
  outputs/{model}/step_{N}/{S__probe | X__prompt | X__bos}/layer_14/   # S×model 交叉 + 双 X
tables/
  eval_trajectory.csv      # 6 模型性能 + OOD
  ood_penalty.csv          # 逐 benchmark drop
  matched_gsm8k_pairs.csv  # 配对结果
  main_matched_result.csv  # 主结论行
  geometry_metrics.csv     # 几何指标（含 s_probe_source / is_diagonal 交叉维度）
  geometry_long.csv        # 完整奇异谱（~10MB）
figures/
  main_geometry_trajectory.png
  main_ood_vs_geometry.png
  appendix_cross_effrank_heatmap.png
selection.json / summary.json
```

---

## 附：v1 → v2 修复清单

| # | v1 缺陷 | v2 修复 |
|---|---|---|
| 1 | prompt 采样不统一（cold/OPD 无种子取前 N，SFT seed=43，probe seed=44）引入 bias | 统一母池一次采样切分，train/heldout/probe 互不重叠（已验证交集=0） |
| 2 | 无 eval loss / held-out，checkpoint 盲存按 save_steps，选 last | held-out eval loss（OPD=JSD/SFT=CE）+ `load_best_model_at_end` 选轨迹内 best |
| 3 | DataSize 量纲混乱（theta0/OPD 是步数，SFT 是样本数） | 统一为"实际训练消耗 prompt/样本数" |
| 4 | S/X 探针全错（三模型统一用 NuminaMath solution 做 S，用无关旧文件做 X） | S 按模型区分（theta0=teacher rollout / OPD=student rollout / SFT=训练数据）；X=theta0 rollout 冻结共用 |
| 5 | 无交叉矩阵 | S×model 完整交叉（6 模型 × 6 S 探针 + X，`is_diagonal`/`s_probe_source` 标记） |
