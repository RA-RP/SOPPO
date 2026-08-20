# Source Note: Capability-Degradation Benchmark 目录（Cycle 08 eval 拓展）

```yaml
view_type: extended_source_note
cycle: cycle_08_h_opd_vs_sft_comparison
purpose: 为"能力退化/遗忘"评测拓展 benchmark 覆盖（超出 Rethink SFT 偏难推理的套件），给出每个 benchmark 测什么/规模/Qwen3-4B-Base·8B-Base 表现
status: reference
created: 2026-06-27
sources:
  - "Qwen3 Technical Report, arXiv 2505.09388 (Table 6 = 8B-Base, Table 7 = 4B-Base 官方 base 分数)"
  - "各 benchmark 原始论文 + lm-eval/OpenCompass 常用 eval split 规模"
  - "外部 commonsense 分数（indicative，协议不明）"
caveat: 分数高度协议依赖——Qwen 官方 ≠ Rethink SFT ≠ 本项目 lm-eval。官方数字是参照点，本项目必须在统一协议下自测。
```

## 背景与诉求

Cycle 07/08 当前 eval 套件来自 Rethink SFT，**偏难推理**：MATH500/AIME24（ID 数学）、LCB v2/GPQA-D/MMLU-Pro（OOD 推理）、IFEval/AlpacaEval/HaluEval/TruthfulQA（通用）、HEx-PHI（安全）。本项目训练在**数学 long-CoT** 上，核心关切是**能力退化（遗忘）**——即训练数学是否损害**非数学**能力。退化轴应强调**base 已强、非目标**的能力（base 高 → 掉得见），而非偏难/base 弱的轴。

> **关键区分**：数学（GSM8K/MATH500/AIME）是**目标/ID 轴**（应升）；退化轴 = **非目标能力**（commonsense / 广知识 / 阅读 / 基础代码 / 指令 / 真实性 / 安全）。

---

## Benchmark 目录（按能力维度；规模=常用 eval split；分数=Qwen3 官方 base）

### A. 广知识（knowledge breadth）— 退化=知识遗忘

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **MMLU** | 57 学科多选知识 | 14,042 | **72.99** | **76.89** | base 强、cheap loglikelihood；经典退化探针。**当前未在套件**（套件用更难的 MMLU-Pro） |
| MMLU-Redux | MMLU 清洗子集 | ~3,000 | 72.79 | 76.17 | MMLU 的低噪声版 |
| MMLU-Pro | 10 选、更难知识推理 | 12,032 | 50.58 | 56.73 | **已在 Cycle 07 套件**（OOD 推理） |
| SuperGPQA | 研究生 285 学科 | 26,529 | 28.43 | 31.64 | 偏难、base 弱；退化探针价值低 |

### B. Commonsense 推理 — 经典遗忘探针（cheap、base 强）

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **HellaSwag** | 情境续写常识 | 10,042 (val) | — | ~78.5* | Qwen3 官方不报；cheap MC |
| **WinoGrande** | 代词消解常识 | 1,267 | — | ~59* | 本项目 OOD-lite 已有(1.7B 0.62)；*外部值存疑 |
| **ARC-Challenge** | 小学科学多选 | 1,172 | — | ~91.1* | 本项目 OOD-lite 已有(1.7B 0.46) |
| **PIQA** | 物理常识 | 1,838 (val) | — | ~88.7* | Qwen3 官方不报 |

\* = 外部单一评测，协议不明，仅 indicative。**Qwen3 官方不报 commonsense**（对强模型偏饱和）→ 本项目需自测（cheap loglikelihood，lm-eval 直接支持）。

### C. 广/多步推理（非数学）

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **BBH** | 23 项困难多步推理 | 6,511 | **72.59** | **78.40** | base 强、官方有数；**当前未在套件**——优质退化/推理保持探针（生成式 CoT，较贵） |
| GPQA-Diamond | 研究生科学多选 | 198 | 36.87(main) | 44.44(main) | **已在套件**；Diamond 子集 198 高方差 |
| AGIEval | 人类考试题 | ~8,062 | — | — | 官方不报；广推理 |
| DROP | 段落离散推理 | 9,536 (dev) | — | — | 阅读+推理；官方不报 |

### D. 阅读理解

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| DROP | 段落数值/离散推理 | 9,536 | — | — | 官方不报 |
| RACE | 中学英语阅读多选 | ~4,887 | — | — | 官方不报；cheap MC |

### E. 代码

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **EvalPlus** (HE+/MBPP+) | 增强测试的函数级代码 | HE+164 / MBPP+378 | **63.53** | **67.65** | 官方有数；基础代码退化探针 |
| MBPP | 入门 Python | 500 (test) | 67.00 | 69.80 | 官方有数 |
| MultiPL-E | 多语言代码 | 多语言 | 53.13 | 58.75 | 官方有数 |
| CRUXEval (CRUX-O) | 代码推理(输出预测) | 800 | 55.00 | 62.00 | 官方有数 |
| HumanEval | 函数级代码 | 164 | （含于 EvalPlus） | | 经典但易被 EvalPlus 取代 |
| LiveCodeBench v2 | 防污染竞赛代码 | ~511 | — | — | **已在套件**（OOD 代码，偏难、生成贵） |

### F. 数学（= 目标/ID 轴，非退化轴）

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **GSM8K** | 小学数学应用题 | 1,319 | **87.79** | **89.84** | base 强；**当前未在套件**——可作"易数学"目标轴（vs 难的 MATH500/AIME），也能反向看是否过拟合难题而损易题 |
| MATH | 竞赛数学 | 5,000 / 500(MATH500) | 54.10(full) | 60.80(full) | **MATH500 已在套件** |
| AIME24 | 竞赛数学(极难) | 30 | — | — | **已在套件**；N=30 高方差，需 avg@k |

### G. 指令遵循

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **IFEval** | 可验证指令遵循 | 541 prompts | ~35(Rethink) | — | **已在套件**；⚠️ base 分低、SFT 后升——测**获得**非纯退化 |

### H. 真实性/事实性/幻觉

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base | 备注 |
|---|---|---|---|---|---|
| **TruthfulQA** | 抗常见错误信念 | 817 | — | — | **已在套件 + OOD-lite**；MC1/MC2 cheap |
| HaluEval | 幻觉判别 | 35k(项目用3k) | — | — | **已在套件** |
| SimpleQA | 短事实问答 | 4,326 | — | — | 官方不报；事实召回 |
| TriviaQA | 事实召回 | ~17,944 | — | — | 官方不报；base 强、cheap |

### I. 多语言（若关心）

| Benchmark | 测什么 | 规模(eval) | 4B-Base | 8B-Base |
|---|---|---|---|---|
| MGSM | 多语言小学数学 | 250/语言 | 67.74 | 76.02 |
| MMMLU | 多语言 MMLU | 多语言 | 71.42 | 75.72 |
| INCLUDE | 多语言知识 | — | 56.29 | 59.40 |

### J. 安全 / K. 开放质量

| Benchmark | 测什么 | 规模 | 备注 |
|---|---|---|---|
| HEx-PHI | 有害指令拒绝(ASR) | 300×10 | **已在套件**(安全) |
| AlpacaEval 2.0 | 开放式有用性(RM/judge) | 805 | **已在套件** |
| Arena-Hard | 困难开放式(judge) | 500 | 未在套件 |

---

## 对本项目的建议：能力退化套件如何补

**核心思路**：当前套件偏难推理 + 偏"获得"轴。退化套件应补**base 强、非目标、覆盖广**的轴，且**按成本分层**（Cycle 08 要在 10-checkpoint × OPD/SFT 双臂上 track，成本敏感）。

### 分层建议

**Tier 1 — cheap loglikelihood-MC，全 checkpoint track（广覆盖退化"呼吸表"）**
- **MMLU**（广知识，base 72.99/76.89，官方有数）
- **Commonsense 簇：HellaSwag / WinoGrande / ARC-Challenge / PIQA**（经典遗忘探针；WinoGrande/ARC 本项目 OOD-lite 已有，复用协议）
- **TruthfulQA-MC1**（真实性，已有）
- → 这些便宜、可在每个 checkpoint 跑，构成"能力保持广度"轴。⚠️ 但本项目 1.7B 时 OOD-lite 非判别(<1.5%)；4B base 更强、headroom 更大，值得重测，但需先验证判别力。

**Tier 2 — 生成式，关键 checkpoint 抽测（深度退化）**
- **BBH**（广多步推理，base 72.59/78.40，官方有数，当前缺——强烈建议补）
- **GSM8K**（易数学，base 87.79/89.84；看难题训练是否损易题）
- **EvalPlus / MBPP**（基础代码，官方有数）
- → 生成式（尤其 think-format 长 CoT）贵，按预算在 dip/recover/final 等 landmark 抽测即可。

**可选扩展**：DROP/RACE（阅读）、AGIEval（广推理）、SimpleQA/TriviaQA（事实召回）。

### ⚠️ 三条方法学警告（必须遵守）

1. **协议依赖、不可混用**：Qwen 官方 ≠ Rethink SFT ≠ 本项目 lm-eval。例：4B-Base MMLU-Pro 官方 50.58 vs Rethink SFT 35.5（差 15 点！）。**官方数字只作"量级 sanity-check 参照"，本项目所有 benchmark 必须在一套固定协议下自测**（chat/think、few-shot、scorer），否则重蹈 Cycle 05 MMLU 协议坑。
2. **退化轴要 base 强**：MMLU/GSM8K/HellaSwag/ARC/BBH（base 高 → 掉得见）。指令类（IFEval/AlpacaEval/Arena-Hard）base 低、SFT 后升，测的是**获得**不是退化，别当退化轴。
3. **commonsense 无官方 Qwen3 参照**（官方视其饱和）→ 本项目自测，cheap，但要确认在 4B 上有判别力（不是又一个 <1.5% 非信号）。

---

## 给下游

- **Experiment Design**：从 Tier 1/Tier 2 选退化套件并**预注册**；区分目标轴(数学)与退化轴(非数学)；为每个 benchmark 固定协议（防协议坑）。成本上 Tier 1 全 track、Tier 2 抽测。
- **Related Work（本对话）**：把 benchmark 来源登记进 source_matrix 的 Eval Protocol Sources（Cycle 08 capability-degradation 候选轴）；本笔记是目录。
- **注意**：这是 eval **广度**拓展，不改 Cycle 08 既定的 H-OPD primary gate（B08/C08 仍以 MATH500 + MMLU-Pro 为主）；退化套件是**补充的能力保持证据**。
