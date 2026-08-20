# Source Note: Rethink SFT — 实验表格梳理（model / dataset / datasize × ID/OOD）

```yaml
view_type: extended_source_note
source_paper: "Ren et al., Rethinking Generalization in Reasoning SFT, arXiv 2604.06628 (2026)"
local_pdf: ../../Rethink SFT.pdf
purpose: 把论文实验表格整理成 model/dataset/datasize × ID/OOD 关系，供 Cycle 07 related-work 决策与用户自审
status: reference
created: 2026-06-24
note: 数字均从 PDF 抽取；个别被排版截断的单元格已标注 (?)。以 PDF 原表为准。
```

本笔记不是论文综述，而是**把论文实验拆成"哪条轴在动、ID/OOD 怎么变"的对照表**，方便判断 Cycle 07 该选什么 model/data/format。

---

## 0. Benchmark 分类（论文 App B.3）

论文把评测分成四组。理解 ID/OOD 关系前先记住这个分类：

| 组别 | Benchmark | 度量 | 相对训练数据(math long-CoT)的位置 |
|---|---|---|---|
| **ID reasoning** | MATH500 (avg@3), AIME24 (avg@10) | accuracy | **同域**：数学推理，与训练数据同分布 |
| **OOD reasoning** | LiveCodeBench v2 (avg@3), GPQA-Diamond (avg@3), MMLU-Pro (pass@1) | accuracy | **跨域推理**：代码 / 研究生科学 / 知识密集，训练中无此领域 |
| **General capability** | IFEval, AlpacaEval 2.0 (RM score), HaluEval, TruthfulQA(helpful) | 各自 | **通用能力**：指令遵循 / 开放质量 / 真实性 |
| **Safety** | HEx-PHI (ASR, 越低越安全) | attack success rate | **安全**：long-CoT SFT 会让它变差（不对称泛化） |

解码默认：zero-shot、temperature 0.6、max 32768 tokens、无 system prompt、统一 `\boxed{}` 模板、math 用 math-verify、评测前删 last `</think>` 之前内容。

---

## 1. 轴一：MODEL CAPABILITY（同一数据 Math-CoT-20k，不同模型/规模）

### 1a. 未训练 base 分数 —— "起跑线"（Table 5，节选）

| Base model | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | HaluEval | TruthfulQA(t/h) |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-1.7B-Base** | 58.9 | 5.7 | 9.1 | 14.3 | 22.0 | 35.0 | 41.8 | 69.9 / 68.7 |
| **Qwen3-4B-Base** | 70.8 | 11.0 | 20.6 | 14.6 | 35.5 | 49.2 | 50.8 | 81.6 / 48.2 |
| **Qwen3-8B-Base** | 76.8 | 12.0 | 29.1 | 37.5 | 55.2 | 54.8 | 50.5 | 84.7 / 83.6 |
| **Qwen3-14B-Base** | 77.8 | 14.7 | 37.5 | 41.8 | 61.8 | 64.2 | 54.7 | 80.1 / 94.4 |
| Qwen2.5-1.5B-Base | 58.7 | 4.3 | 9.4 | 21.0 | 15.0 | 37.0 | 45.7 | 78.5 / 24.2 |
| Qwen2.5-7B-Base | 66.7 | 7.7 | 15.7 | 26.9 | 34.8 | 41.1 | 45.3 | 80.7 / 96.5 |
| Qwen2.5-14B-Base | 65.7 | 8.0 | 17.6 | 28.6 | 40.9 | 49.0 | 56.1 | 85.6 / 93.8 |
| InternLM2.5-20B-Base | 53.7 | 1.7 | 17.6 | 20.9 | 34.3 | 51.1 | 60.6 | 70.3 / 72.6 |

> 读法：规模越大，OOD reasoning（LCB/GPQA/MMLU-Pro）起点越高。1.7B 在 GPQA(14.3)、LCB(9.1)、MMLU-Pro(22.0) 上几乎贴地——它**还没有可被 SFT 激活的跨域推理底子**。

### 1b. 训练后（Math-CoT-20k, 8 epochs）四个规模的泛化（Sec 5 / Fig 12 文字结论 + Table 6）

论文 Sec 5 定性结论（同一数据、同一协议，只变模型规模）：

| 规模 | 训练后 cross-domain 行为 |
|---|---|
| Qwen3-14B-Base | 明显 dip-and-recovery，**最终广泛提升**（ID+OOD 都涨） |
| Qwen3-8B-Base | 有 recovery，提升**较小** |
| Qwen3-4B-Base | 有 recovery，提升**更小**（"smaller improvements"） |
| **Qwen3-1.7B-Base** | 在所有 benchmark 上、即使 late checkpoint 也只 **marginal / negative gain** |

**Qwen3-1.7B-Base + Math-CoT-20k 完整训练动态（Table 6）：**

| Step | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval |
|---|---|---|---|---|---|---|
| Base | 58.9 | 5.7 | 9.1 | 14.3 | 22.0 | 35.0 |
| 10 | 55.7 | 4.3 | 9.9 | 15.0 | 20.8 | 35.9 |
| 20 | 53.5 | 3.3 | 11.5 | 15.2 | 16.3 | 30.7 |
| 40 | **42.9** (dip 谷底) | 1.7 | 4.1 | 9.8 | 18.3 | 29.5 |
| 80 | 49.6 | 2.7 | 6.3 | 6.2 | 26.2 | 30.1 |
| 160 | 51.6 | 4.0 | 4.9 | 8.1 | 30.5 | 30.5 |
| 320 | 56.5 | 5.0 | 7.4 | 6.6 | 31.7 | 31.7 |
| 480 | 59.1 | 3.7 | 6.5 | 7.9 | 33.2 | 33.5 |
| 640 | **56.2** | 5.0 | 7.0 | 9.4 | 33.7 | 33.7 |

> 读法：1.7B 跑完 640 步，**MATH500 净负（58.9→56.2）**、AIME/LCB/GPQA 基本回不到起点，只有 MMLU-Pro（22→33.7）涨。这就是"能力地板"——**和模型是不是 instruct 无关，官方 1.7B-Base 同样训不出泛化**。这正是本项目 Cycle 04-06 在 1.7B 上反复撞到的天花板的外部独立佐证。

**对照 Countdown-CoT-20k（Table 7，1.7B-Base）step640：** MATH500 57.1 / AIME24 5.3 / LCB 6.3 / GPQA 18.7 / MMLU-Pro 34.2 / IFEval 29.9 —— 同样 marginal。

---

## 2. 轴二：TRAINING DATA（同一模型，不同数据质量/结构）—— 论文最关键的表

### Table 2：四种数据配置 × 三个模型（last checkpoint, 8 epochs, 20k 量级）

四种数据（同 20k 量级，差别在质量与结构）：

| 数据 | 构成 | 关键差异 |
|---|---|---|
| **Math-CoT-20k** | OpenR1-Math-220k query + Qwen3-32B(thinking) long-CoT + math-verify 过滤 | 高质量 long-CoT（含 backtracking/verification） |
| **Math-NoCoT-20k** | Math-CoT-20k 去掉 `<think>...</think>` | 同 query+同最终解，**无探索过程** |
| **NuminaMath-20k** | 同 query，解来自 **NuminaMath-1.5 人写短解** | 短、质量参差（本项目当前用的就是这类） |
| **Countdown-CoT-20k** | Countdown 算术游戏 query + Qwen3-32B long-CoT | 纯 procedural，无数学领域知识 |

**Qwen3-14B-Base：**

| Data | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | AlpacaEval | HaluEval | TruthfulQA |
|---|---|---|---|---|---|---|---|---|---|
| Base | 77.8 | 14.7 | 37.5 | 44.1 | 61.8 | 64.2 | 0.53 | 54.7 | 94.4 |
| **Math-CoT** | **95.1** | **66.0** | **55.1** | **63.3** | 74.4 | 68.9 | 1.42 | 72.8 | 95.6 |
| Math-NoCoT | 82.4 | 17.0 | 40.3 | 48.3 | 69.1 | **71.7** | **2.11** | 70.9 | **100** |
| **NuminaMath** | 74.8 | 14.0 | **20.4** | 38.4 | 59.0 | 52.8 | **−0.45** | 62.7 | 88.6 |
| Countdown-CoT | 91.5 | 41.7 | 43.8 | 53.0 | 65.4 | 61.3 | 1.36 | 72.3 | 92.4 |

**Qwen3-8B-Base：**

| Data | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | AlpacaEval | HaluEval | TruthfulQA |
|---|---|---|---|---|---|---|---|---|---|
| Base | 76.8 | 12.0 | 29.1 | 39.7 | 55.2 | 54.8 | −0.13 | 50.5 | 84.6 |
| **Math-CoT** | **90.1** | **47.7** | 31.0 | 48.5 | 69.5 | 56.6 | 0.92 | 71.3 | 91.3 |
| Math-NoCoT | 77.4 | 13.3 | 31.8 | 45.1 | 60.9 | **63.7** | **1.85** | 56.2 | **100** |
| **NuminaMath** | 72.5 | 7.0 | **7.9** | 37.2 | 48.2 | 37.8 | **−0.70** | 64.2 | 89.8 |
| Countdown-CoT | 87.6 | 26.0 | 32.8 | 41.6 | 56.6 | 52.3 | 0.13 | 68.9 | 75.5 |

**InternLM2.5-20B-Base：**

| Data | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | AlpacaEval | HaluEval | TruthfulQA |
|---|---|---|---|---|---|---|---|---|---|
| Base | 53.7 | 1.7 | 17.6 | 20.9 | 34.3 | 51.1 | −0.84 | 60.6 | 72.6 |
| **Math-CoT** | **80.3** | **16.7** | 26.0 | 41.4 | 48.3 | 52.6 | **1.23** | 69.1 | 99.8 |
| Math-NoCoT | 57.9 | 5.0 | 17.9 | 35.7 | 48.3 | 52.3 | (?) | (?) | (?) |
| **NuminaMath** | 45.7 | 1.0 | **3.9** | 29.3 | 38.9 | 39.1 | **−1.55** | 58.3 | 99.0 |
| Countdown-CoT | 54.5 | 2.3 | 22.1 | 25.3 | 47.5 | 39.6 | 0.40 | 70.2 | 75.5 |

> **三条读法（这是本项目最该吸收的）：**
> 1. **NuminaMath 短解是最差的一档**：它不仅 ID(MATH500/AIME) 不涨，连 base 都不如（14B 77.8→74.8），且把 OOD reasoning **打到地板**（14B LCB 37.5→20.4，8B LCB 29.1→7.9，20B LCB 17.6→3.9），AlpacaEval 转负。→ **本项目当前用的 NuminaMath-1.5 正是这类数据。**
> 2. **long-CoT vs no-CoT（Math-CoT vs Math-NoCoT，同 query 同答案，只差有没有探索过程）**：long-CoT 在 ID 和 OOD reasoning 上全面碾压 no-CoT（14B AIME 66.0 vs 17.0，GPQA 63.3 vs 48.3）。**→ procedural pattern（不是数学内容本身）是迁移来源。** 注意 no-CoT 在 IFEval/AlpacaEval/TruthfulQA 这些"通用能力/指令"上反而略好——所以"no-think 看起来不差"只在通用轴，推理迁移轴上它输。
> 3. **Countdown（纯算术游戏、零数学知识）也能提升数学/科学**：14B Countdown 把 MATH500 91.5、GPQA 53.0 拉起来，甚至在 reasoning 上压过 NuminaMath。→ 再次说明是**结构/过程**在迁移。但对弱模型(InternLM)Countdown 只有 marginal——又回到能力依赖。

---

## 3. 轴三：DATA SIZE / OPTIMIZATION（同模型同数据，变数据量/曝光/训练量）

### 3a. Table 1：固定 640-step budget，三种 schedule（Qwen3-14B-Base, Math-CoT-20k）

| Setting | 数据量 / bsz / epoch | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval |
|---|---|---|---|---|---|---|---|
| Setting 1 | 20k / 256 / 8ep | **95.1** | **66.0** | **55.1** | **63.3** | **74.4** | **68.9** |
| Setting 2 | 2.5k / 32 / 8ep | 94.9 | 61.7 | 51.7 | 59.9 | 73.6 | 63.4 |
| Setting 3 | 20k / 32 / 1ep | 92.9 | 48.0 | 45.4 | 46.8 | 59.8 | 59.8 |

> 读法：**Setting 2 vs 3 同 budget、同步数，差别只在"8 遍小数据 vs 1 遍大数据"。Setting 2 全面更好 → 重复曝光(multi-epoch) 比一遍覆盖更有效。** Setting 1 再加数据多样性，又比 2 好 → 数据量在固定步数外仍有边际价值。对长 CoT，**under-optimization（训练不足）比 over-optimization 风险更大**。

### 3b. dip-and-recovery（Sec 3 / Table 6 已示范）

cross-domain 性能**先降后升**：短 epoch checkpoint 会系统性**低估** SFT 的泛化潜力。1.7B 的 Table 6 里 MATH500 在 step40 跌到 42.9 才慢慢回到 56。→ **判 "SFT 不泛化" 前必须确认训练充分、看完整 trajectory，否则会把 under-optimization 误读成不泛化。** 这是本项目读 A07/A06b 结果时必须套上的护栏。

### 3c. 过拟合 stress test（Sec 3.4，Qwen3-14B-Base, Math-CoT-20k）

只有在**激进组合**（高 LR + 无 LR decay + 长 epoch，如 LR 1e-4/16ep/constant）下才出现明显过拟合：OOD 广泛下降、连 ID math 也降、response length 回升。默认 schedule（LR 5e-5/8ep/cosine）下主要是 under-fitting 信号。

---

## 4. 默认训练超参（Table 3，主实验）

| 超参 | 值 |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 256 |
| Weight decay | 0.01 |
| Warmup | 10% of total steps |
| LR schedule | Cosine decay |
| Grad steps / epoch | 80 |
| Training epochs | 8 |
| Max prompt length | 3072 |
| Max response length | 16384 |
| Trainer | Verl SFT trainer |
| 硬件 | 训练 8×H200，评测 2×H200 |

**训练 prompt 模板（base 也套 Qwen chat template）：**
```
<|im_start|>user
{Question}
Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
{Response} <|endoftext|>
```
`{Response}` = teacher 的 thinking process + step-by-step final summary + answer（即含 `<think>...</think>` 的 long-CoT）。

**数据生成参数（Math-CoT-20k）：** Qwen3-32B thinking enabled，temp 0.6 / top-p 0.95 / top-k 20 / min-p 0，max 16384 tokens，每 query 多采样、math-verify 留正确解、多解随机取一。

---

## 5. 一句话总结：三条轴各自怎么影响 ID/OOD

| 轴 | 对 ID reasoning | 对 OOD reasoning | 本项目含义 |
|---|---|---|---|
| **模型能力↑**(1.7B→14B, 同数据) | 都涨，但小模型涨得少 | **关键分水岭**：≥8B 才稳健跨域，1.7B 几乎不跨域，4B 临界 | 必须离开 1.7B；4B 是最小可行、偏临界 |
| **数据质量/结构**(NuminaMath短解 ↔ long-CoT, 同模型) | long-CoT≫no-CoT≫NuminaMath | **NuminaMath 把 OOD 打到地板**；long-CoT/Countdown 提升 OOD | 必须离开 NuminaMath 短解，换 verified long-CoT |
| **数据量/曝光**(1ep大数据 ↔ 多ep小数据, 同budget) | 多 epoch 更好 | 多 epoch 更好；短 epoch 低估泛化(dip-recovery) | 训练要足够、看完整 trajectory 再判 fail |

> 三者**联合条件**（论文 App A）：强模型 + 烂数据仍不泛化(Sec4)；好数据 + 训练不足显得无效(Sec3)；弱模型 + 好数据充分训练也只 marginal(Sec5)。没有单因素万能。**本项目 Cycle 07 同时踩了"1.7B 弱模型"+"NuminaMath 烂数据"两个坑**——换 4B base + long-CoT 数据是同时拆这两个坑。

---

## 6. 专题：Qwen3-4B-Base 能拿到什么（重要 caveat）

**数据可得性警告：** 论文对 4B **只给了未训练 base 分数（Table 5），没有给训练后逐步数值表。** 4B 训练后结果只以曲线形式出现在 Fig 12（performance）/ Fig 13（response length），无数字。下面区分"论文实测"与"插值推断"。

### 6a. Qwen3-4B-Base 未训练 base —— 论文实测（Table 5 完整行）

| 指标 | 组别 | Qwen3-4B-Base | 对比 1.7B | 对比 8B |
|---|---|---|---|---|
| MATH500 | ID | 70.8 | 58.9 | 76.8 |
| AIME24 | ID | 11.0 | 5.7 | 12.0 |
| LCB v2 | OOD | 20.6 | 9.1 | 29.1 |
| **GPQA-D** | OOD | **14.6** | 14.3 | 37.5 |
| MMLU-Pro | OOD | 35.5 | 22.0 | 55.2 |
| IFEval | General | 49.2 | 35.0 | 54.8 |
| AlpacaEval (RB2/Sky/Mis) | General | −2.21 / 2.10 / 1.69 | −4.99/−10.26/−2.28 | −0.13/1.68/4.47 |
| HaluEval | General | 50.8 | 41.8 | 50.5 |
| TruthfulQA (truth/help) | General | 81.6 / 48.2 | 69.9 / 68.7 | 84.7 / 83.6 |

> **结构性问题：4B 的 GPQA-Diamond base 只有 14.6，几乎等于 1.7B 的 14.3，远低于 8B 的 37.5。** 科学 OOD 这一轴上 4B 的底子和 1.7B 一样薄——这是 4B 偏临界的核心证据。其它轴(MATH/LCB/MMLU-Pro)4B 明显高于 1.7B、接近 8B 的一半到 2/3。

### 6b. 训练后 4B —— 论文实测（App D Table 9/10/11，⚠️更正先前插值）

**更正记录（2026-06-24）：** 先前本节用 1.7B/8B 插值推断 4B，把 GPQA 猜成"可能持平甚至负"——**实测完全相反，4B GPQA 大幅正向（14.6→37.9）**。论文 App D 对 4B 给了完整训练表，以下为实测数字，作废之前的插值。

**Table 9 — Qwen3-4B-Base + Math-CoT-20k（默认 schedule，= 本项目本轮拟用配方）：**

| Step | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | HaluEval | TruthfulQA(t/h) |
|---|---|---|---|---|---|---|---|---|
| Base | 70.8 | 11.0 | 20.6 | 14.6 | 35.5 | 49.2 | 50.8 | 81.6/48.2 |
| 10 | 64.9 | 9.0 | 21.5 | 19.9 | 31.5 | 46.3 | 48.7 | 78.2/54.5 |
| 20 | 62.9 (dip) | 13.0 | 17.0 | 17.3 | 30.8 | 33.0 | 63.1 | 83.6/37.7 |
| 40 | 71.5 | 15.3 | 14.5 | 19.4 | 50.1 | 37.2 | 67.7 | 96.7/47.9 |
| 80 | 77.1 | 19.0 | 18.1 | 23.9 | 55.6 | 40.4 | 71.1 | 95.1/53.0 |
| 160 | 78.8 | 22.3 | 17.6 | 27.9 | 59.1 | 41.6 | 71.6 | 92.7/64.4 |
| 320 | 86.3 | 29.0 | 23.5 | 33.3 | 63.1 | 45.4 | 72.6 | 89.5/80.2 |
| 480 | 87.1 | 27.3 | 23.5 | 36.7 | 63.2 | 50.0 | 69.3 | 89.2/84.6 |
| **640** | **86.2** | **30.7** | **24.4** | **37.9** | **64.8** | 49.4 | 69.4 | 90.8/84.7 |

**最终 gain（base→640）：** MATH500 +15.4 / AIME24 +19.7 / LCB +3.8 / **GPQA +23.3** / **MMLU-Pro +29.3** / IFEval ≈0 / HaluEval +18.6 / TruthfulQA +9.2/+36.5。

**Table 11 — Qwen3-4B-Base + Math-NoCoT-20k（去 think；用户看到的那张）：**

| Step | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval | HaluEval | TruthfulQA(t/h) |
|---|---|---|---|---|---|---|---|---|
| Base | 70.8 | 11.0 | 20.6 | 14.6 | 35.5 | 49.2 | 50.8 | 81.6/48.2 |
| **640** | **74.3** | **11.7** | **22.0** | **38.2** | **59.0** | 55.9 | 56.4 | 94.5/100.0 |

**最终 gain（base→640）：** MATH500 +3.5 / AIME24 +0.7 / LCB +1.4 / GPQA +23.6 / MMLU-Pro +23.5 / IFEval +6.7 / TruthfulQA +12.9/+51.8。

**Table 10 — Qwen3-4B-Base + Countdown-CoT-20k（纯算术游戏）step640：** MATH500 83.8 / AIME 21.0 / LCB 25.0 / GPQA 38.6 / MMLU-Pro 60.3 / IFEval 41.4。

> **CoT vs NoCoT 在 4B 上的关键对比（同模型同 query 同答案，只差有没有探索过程）：**
> - **ID 数学推理靠 CoT**：MATH500 86.2 vs 74.3（+11.9），**AIME24 30.7 vs 11.7（+19.0，几乎全部 AIME gain 来自 CoT）**。
> - **OOD 科学/知识两者都涨且接近**：GPQA 37.9 vs 38.2、MMLU-Pro 64.8 vs 59.0——**即使 NoCoT 也把 GPQA/MMLU-Pro 拉起 +23 左右**。
> - **General/指令/真实性 NoCoT 略好**：IFEval 55.9 vs 49.4、TruthfulQA 100 vs 84.7。
> - 结论：long-CoT 的增量主要在**难数学推理（AIME/MATH500）和代码**；GPQA/MMLU-Pro 这种 MC 知识题，去不去 think 都涨。这与论文主表（Table 2）在 8B/14B 上的模式一致。

### 6c. 对 Cycle 07 Gate A07 的含义（按实测更新）

- **4B 实测是"会泛化"的一侧，不是临界失败**。用 Math-CoT，4B 在 ID(MATH500/AIME) 和 OOD(GPQA/MMLU-Pro) 上**都有大幅、稳健的正向 gain**，GPQA/MMLU-Pro 甚至 +23~29。先前担心的"GPQA 可能转负"被实测推翻。
- 因此 **A07 即使把 gate 绑到 OOD reasoning（GPQA/MMLU-Pro），4B 也有较大把握过**——不必为了 OOD gain 被迫升 8B。
- "4B 提升小于 8B/14B" 指的是**绝对天花板**（AIME 30.7 vs 8B 47.7 vs 14B 66.0；MATH500 86.2 vs 90.1 vs 95.1），不是"4B 训不出泛化"。4B 完全够撑 feasibility（A07）和后续 OPD-vs-SFT 对比的信号。
- **唯一仍需注意**：4B 有可见 dip（MATH500 step20 跌到 62.9 才回升并反超），判 fail 前必须训够步数、看完整 trajectory（dip-and-recovery 护栏照旧适用）。LCB(代码) 在 4B 上 gain 最小(+3.8)，是 4B 最弱的 OOD 轴。

---

## 7. 理论深化脚手架：OPD vs SFT_dip vs SFT_recover（Related Work 备料，正式改写归 Theory 对话）

```yaml
scope_note: 本节是 Related Work 为理论深化准备的文献脚手架，不是理论定稿。
            正式机制假设/变量/claim 边界由 Theory 对话写入 theory/。
status: scaffolding_for_theory_conversation
user_confirmed: 2026-06-24 (方向认可 + on-policy 文献现在就加)
```

### 7.1 核心升级：OOD-drop 从 method 属性 → checkpoint 属性

原命题（method 二元）："OPD vs SFT，谁的 OOD-drop 小、继续学习好、几何指标如何。"
升级命题（trajectory）："**OOD-drop 是优化轨迹上的暂态，由 checkpoint 所处优化阶段决定**（Rethink SFT dip-and-recovery, Sec 3）。SFT 不是一个点，而是 SFT_dip → SFT_recover 的轨迹。"

### 7.2 SFT_dip vs SFT_recover 特征对照（Rethink SFT 实测，4B Math-CoT）

| 维度 | SFT_dip（~step20–40） | SFT_recover（~step320+） | 文献锚点 |
|---|---|---|---|
| 继续学习(ID) | 未巩固（MATH500 62.9<base） | 巩固/反超（MATH500 86.2） | Table 9 |
| OOD-drop | 掉（GPQA 17.3, LCB 14.5） | 回升反超（GPQA 37.9, LCB 24.4） | Table 9 |
| response length | 暴涨（~28–30k） | 收缩（~7–23k） | Sec 3.2；Table 9 底部 |
| 行为定性 | shallow imitation，长而不收敛 | internalized procedural pattern，会 backtrack/自校验 | C.3, C.6 |

### 7.3 三方框架与可证伪假设

**中心假设 H-OPD：** OPD ≈ "不经过 dip 暂态、直达 recover 状态"的捷径（on-policy 减少 exposure bias，少绕路）。

预注册对照（Cycle 08，需 OPD arm）：
- 若 H-OPD 成立：OPD 的 (OOD-drop, response-length surge, effective-rank/drift 暂态峰) **显著小于 SFT_dip**，而 ID/OOD 终点 ≈ SFT_recover。
- 你的几何指标在三方框架中的新角色：**effective rank / drift-from-base 是否在 SFT_dip 出现暂态峰、在 recover 回落？OPD 是否压平该峰？** —— 把"OPD 降 rank / SFT 升 rank"的静态对照升级为沿轨迹的暂态结构对照。

文献支撑（OPD 腿）：Chen et al. 2025（on-policy 缓解 forgetting）、Shenfeld et al. 2026（online 忘得少、贴近 base 分布 → 直接预测 drift-from-base 更平）、Tajwar et al. 2024（on-policy 采样本身的价值）。

### 7.4 分层：哪些 Cycle 07 能测，哪些 Cycle 08

| 层 | 内容 | 何时 | 对 Experiment Design 的硬要求 |
|---|---|---|---|
| **Cycle 07** | SFT_dip vs SFT_recover：沿 4B-base SFT 轨迹，OOD-drop / 继续学习 / 几何指标 + **response length** 的关系 | 本轮（无需 OPD） | **沿轨迹多 checkpoint 采样**几何指标 + response length（不能只存终点 checkpoint），否则 Cycle 08 三方对照缺基线 |
| **Cycle 08** | 加 OPD 腿：验证 H-OPD（OPD = recover-without-dip） | 需 OPD arm | 保存 UV/sketch 几何 artifact；OPD 与 SFT 同 budget/同 checkpoint 网格采样 |

### 7.5 给各对话的输出

- **Theory 对话**：把 7.1–7.3 落成机制假设（H-OPD）、变量（新增 `optimization_stage` / `response_length` 作为 dip↔recover 的可观测代理；几何指标改为"沿轨迹暂态结构"而非静态终点）、claim 边界（OOD-drop 是 checkpoint 属性，不可写成"SFT 必然 OOD-drop"）。
- **Experiment Design 对话**：把 7.4 的"多 checkpoint 几何 + response-length 采样"写进 Cycle 07 required artifacts；response length 列为必收分析指标。
- **Related Work（本对话）已完成**：登记 Rethink SFT（核心）+ on-policy-forgetting 文献簇（OPD 腿备料），见 `related_work.md` / `source_matrix.md` / `incremental_related_work.md`。
