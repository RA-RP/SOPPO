# 相关工作增量记录

```yaml
view_type: incremental_log_ch
status: active
maintenance_rule: append_only_per_cycle
last_updated: 2026-06-27
sync_status: 与 incremental_related_work.md 同步至 Cycle 08（2026-06-27）
```

每轮 cycle 新增/删除的论文及原因。**删除应谨慎**：只有在确认某篇论文完全不再被论文论点使用时才删除，并注明原因。

---

## Cycle 04 — 初始文献集（2026-06-15）

### 新增

| 论文 | 新增原因 |
|---|---|
| Agarwal et al., GKD / OPD, ICLR 2024 | OPD 机制核心依据：student-visited states 训练与标准 KD 的区别 |
| verl OPD 文档 | 定义本项目所用 OPD 实现的实际训练循环 |
| Rethinking On-Policy Distillation, 2026 | OPD overlap dynamics 机制；motivates less-saturated theta0 设计 |
| A Survey of On-Policy Distillation, 2026 | OPD 分类框架；定位本项目为白盒、token-level、output-space OPD |
| On the Impact of Fine-Tuning on CoT Reasoning, 2024 | 建立 SFT 退化 baseline；支持将 continued SFT 视为可能的退化路径 |
| LoRA vs Full Fine-tuning, 2024 | 为谱分析作为机制信号提供依据；连接 GetSlice 指标与文献 |
| OPRD: On-Policy Representation Distillation, 2026 | 前向指针：representation-aware OPD 方向，本项目处于其上游 |

### 删除

无

---

## Cycle 05 — 评测协议来源（2026-06-17）

### 新增

| 论文 / 来源 | 新增原因 |
|---|---|
| Qwen3 Technical Report, arXiv 2505.09388 | 被采纳为 MMLU 评测协议依据（mmlu_protocol_version=2）。经过 4 种配置失败后通过查阅官方报告确认 base model mode |
| lm-eval-harness Issues #3405 / #3576 / #3322 | 社区独立复现，与官方报告二次确认 MMLU no-chat-template 协议。性质：methodology grounding，非论文论点 |

### 删除

无

---

## Cycle 06 — SFT 可学习性、遗忘与 Qwen3 协议来源（2026-06-18）

### 新增

| 论文 / 来源 | 新增原因 |
|---|---|
| Luo et al., *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning* | Cycle 06 改为 SFT-only feasibility/degradation cycle；该文献提供 1B-7B LLM continual fine-tuning 中 catastrophic forgetting 的经验背景 |
| Kalajdzievski, *Scaling Laws for Forgetting When Fine-Tuning Large Language Models* | 支持把 SFT 数据量、update steps、tuned parameter scale 作为实验变量，而不是实现细节 |
| Dong et al., *How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition*（ACL 2024） | 与 Cycle 06 的 SFT data-scale / ability tradeoff 问题最接近；支持 ID/OOD 分轴和数据规模曲线 |
| Ding and Wang, *Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting* | 作为未来 mitigation baseline；若 Cycle 06 发现 SFT 退化，后续可比较 OPD-like regularization 与 mixed/rehearsal-style mitigation |
| Qwen3 官方博客 / 使用说明 | Q4 QA 决定：Experiment Design 前必须查清 Qwen3 官方训练/评测格式；该来源说明 post-training pipeline 与 `enable_thinking=True/False` 用法 |
| Qwen/Qwen3-1.7B model card | 记录 post-trained Qwen3-1.7B 的 chat-template、thinking mode、`</think>` parsing 行为；用于避免推训格式不一致 |
| Qwen/Qwen3-1.7B-Base model card | 记录 Qwen3-1.7B-Base 是 pretraining-only checkpoint；用于区分 base arm 与 post-trained arm |
| HuggingFaceTB/qwen3-1.7b-gsm8k-sft model card | 社区工程参考：Qwen3-1.7B GSM8K SFT 使用 conversational `<think>` 格式，并报告 target gain / adjacent MATH-500 tradeoff；非官方 benchmark |
| Unsloth Qwen3 run/fine-tune guide | 社区工程参考：补充 Qwen3 thinking/non-thinking practical settings；低于官方来源，只作格式检查辅助 |
| `source_notes/cycle06_qwen3_sft_protocol_brief.md`（及 `-ch.md`） | 本地综合来源笔记，供 Experiment Design 前读取；记录 Qwen3-1.7B / Qwen3-1.7B-Base 模型身份、协议风险和 baseline sanity-check 问题 |

### 更新

| 既有来源 | 更新原因 |
|---|---|
| Lobo et al., *On the Impact of Fine-Tuning on Chain-of-Thought Reasoning* | 补齐作者、NAACL 2025 会议版信息，并从一般 SFT 退化来源升级为 Cycle 06 SFT feasibility/degradation 的核心来源之一 |
| Qwen3 Technical Report | 从 Cycle 05 MMLU 协议依据扩展为 Cycle 06 Qwen3 model identity、base-score sanity check、post-training pipeline 的官方来源 |
| Source matrix positioning | 将本轮定位改为 SFT feasibility/degradation first；OPD 保留为后续比较背景；geometry 降级为 diagnostic-only；control-anchor 文献先留位不搜索 |
| `source_notes/cycle06_qwen3_sft_protocol_brief.md` | 用户审查指出训练/测试格式报告不完整；补入当前项目训练格式、post-trained vs base 模型差异、GSM8K/MATH500/MMLU/TruthfulQA/WinoGrande/ARC 的 chat/think/token/scorer 表 |

### 删除

无

---

## Cycle 07 — Base 模型 SFT 可行性 + 理论深化文献（2026-06-24）

QA 见 `QA_cycle07_related_work.md`；实验表格抽取见 `source_notes/rethink_sft_experiment_tables.md`。本轮把主模型转 Qwen3-4B-Base、训练数据转 verified long-CoT、格式转 long-CoT think-format、ID 轴转 MATH500+AIME24（OPD 推迟 Cycle 08）。

### 新增

| 论文 / 来源 | 新增原因 |
|---|---|
| **Rethink SFT — Ren et al., *Rethinking Generalization in Reasoning SFT*, arXiv 2604.06628（2026）** | **本轮核心来源。** 用 base 模型 + long-CoT 数学 SFT 证明 cross-domain generalization 是 optimization×data×capability 的条件性质。一次性支撑本轮四线决策（model 4B-base / data long-CoT / format think / eval MATH500+AIME24+LCB/GPQA/MMLU-Pro），并提供 SFT_dip/recover 轨迹的实证骨架。注意：≠ 已有的 "Rethinking On-Policy Distillation"(2604.13016)。 |
| Chen et al., *Retaining by Doing: On-Policy Data Mitigates Forgetting*, arXiv 2510.18874（2025） | 三方框架（OPD vs SFT_dip vs SFT_recover）OPD 腿备料：on-policy 数据缓解 forgetting，支撑"OPD ≈ recover-without-dip"。用户 2026-06-24 确认现在就加（不按 OPD 推迟）。 |
| Shenfeld et al., *RL's Razor: Why Online RL Forgets Less*, ICLR 2026 | OPD 腿备料：online/on-policy 忘得少，mode-seeking 贴近 base 分布 → 预测 OPD 的 drift-from-base 平于 SFT_dip 暂态。 |
| Tajwar et al., *Preference Fine-Tuning Should Leverage On-Policy Data*, ICML 2024 | OPD 腿次级支撑：on-policy 采样本身（非仅数据质量）驱动优势。 |
| 评测轴来源（候选，待 Experiment Design 采纳）：MATH500 (Hendrycks 2021)、AIME24、LiveCodeBench v2 (Jain 2025)、GPQA-Diamond (Rein 2024)、MMLU-Pro (Wang 2024) | 决策 Q4：ID 改 MATH500+AIME24、OOD 采纳论文套件。登记来源 + Rethink SFT Table 5 可核对 base 分数，供 baseline sanity-check。正式采纳/预注册由 Experiment Design 完成。 |
| `source_notes/rethink_sft_experiment_tables.md` | Rethink SFT App D 全表抽取（base 分数 Table 5、四种数据×多模型 Table 2、1.7B/4B/8B/14B 训练动态 Table 6–21 节选）+ 4B 专题 + 三轴（model/data/datasize）×ID/OOD 关系 + 理论深化脚手架。 |

### 更新

| 既有来源 | 更新原因 |
|---|---|
| OPD 文献簇（GKD / verl OPD / Rethinking OPD / OPD survey / OPRD） | 不删除、不降级。Cycle 07 仍为背景；新增的 on-policy-forgetting 文献簇与之衔接，共同构成 Cycle 08 三方框架 OPD 腿的文献基础。 |
| Source matrix positioning | Cycle 06"SFT feasibility first"推进为 Cycle 07"base-model SFT feasibility, generalization-conditional + 理论深化（SFT 轨迹 dip/recover、OOD-drop 为 checkpoint 属性）"。 |
| 本地实验证据表（Cross-Cycle Anchors） | 新增四条：1.7B 天花板被 Rethink SFT 官方 1.7B-Base 独立复现；NuminaMath 被点名低质量数据；no-think 发现与 long-CoT 互证；dip-and-recovery 把 OOD-drop 重定义为 checkpoint 属性。 |

### 删除

无。

### 留位提醒（仍有效）

control-anchor / matched-comparison validity 文献：Cycle 07 仍不搜索；Cycle 08 进入 OPD-vs-SFT 对齐比较时，与 on-policy-forgetting 文献簇一并激活，Related Work 须提醒用户是否补搜。

---

## Cycle 08 — OPD 配方 / 工具 / 对比有效性（2026-06-27）

QA 见 `QA_cycle08_related_work.md`；研究笔记见 `source_notes/cycle08_opd_teacher_coldstart_tooling.md`。Cycle 08 = H-OPD vs SFT 对比（4B-Base + Math-CoT-20k + think-LoRA，单 OPD arm 单 lambda，Cycle 07 SFT 轨迹作 matched control）。本轮补 OPD arm 设计（teacher/cold-start/框架）与对齐比较有效性的文献。

### 新增

| 论文 / 来源 | 原因 |
|---|---|
| **Law of Capacity Gap in Distilling LMs, arXiv 2311.07052** | Cycle 08 teacher 选择定律：最优 teacher 随 student 线性、非越大越好；gap 过大 student 更差 → 4B-Base 选 Qwen3-8B 同家族（用户 Q1 已定），32B 仅离线数据。 |
| **The Efficiency Misnomer, ICLR 2022 (arXiv 2110.12894)** | Cycle 08 matched-comparison validity 方法学锚点（Q4=a，control-anchor 留位激活）：单一/单点指标误导，应多维 + best-effort 多 alternative；支撑轨迹级对比为主、单点作 landmark、iso-ID 作二级。 |
| **TRL GKD Trainer + colocated vLLM**（docs + blog, PR #3394 / v0.18.0） | Q3 工具来源：TRL 能跑 OPD 但非高速（rollout 走 HF generate，3–8× 开销，GKD 无一等 vLLM）。 |
| **verl Async On-Policy KD recipe**（verl docs: algo/opd + advance/async-on-policy-distill） | Q3 工具来源：高速 OPD 专用（vLLM rollout + async teacher + top-k forward KL；工程约束：推理服务器只返 top-k logprob → 必须 top-k KL；cold-start 不内置）。 |
| `source_notes/cycle08_opd_teacher_coldstart_tooling.md` | 本地综合：teacher 选择 / cold-start 可跳过性 + overlap 判据 / TRL vs verl 工具对比；给 Exp/Code 的输出。 |
| **能力退化 eval 拓展（用户追加）：Qwen3 Technical Report base 表 + benchmark 原始论文（MMLU/HellaSwag/WinoGrande/ARC/PIQA/BBH/GSM8K/EvalPlus/TruthfulQA…）** | Rethink SFT 套件偏难推理，对"能力退化/遗忘"覆盖不足。补非数学、base 强的退化轴（Tier 1 cheap-MC + Tier 2 生成式），登记 Qwen3-4B/8B-Base 官方分数作 sanity-check。候选见 related_work-ch.md "Cycle 08 能力退化评测轴"；正式选轴归 Experiment Design。 |
| `source_notes/cycle08_capability_degradation_benchmarks.md` | benchmark 目录：能力维度 × 测什么/规模/Qwen3-4B·8B-Base 分数 + 分层建议（Tier1/2）+ 三条协议警告。 |

### 更新

| 既有来源 | 更新原因 |
|---|---|
| Rethinking On-Policy Distillation (Li et al., 2604.13016) | 从 Cycle 04/06 的"theta0 cold-start 动机"扩充为 Cycle 08 OPD 配方实测来源：补 teacher-student 实测配对（同家族 ~2–5×、常 domain-RL）、cold-start 两阶段、overlap ratio 判据（72%→91%）、cold-start 可跳过性与 ~3000 步 plateau（scale-dependent，不照搬）。 |
| verl OPD documentation | 从"定义 OPD 实现"扩充为 Cycle 08 高速训练工具来源（async recipe + top-k KL 实现约束 + 同家族 teacher 要求）。 |
| control-anchor 留位 | **状态：激活。** Cycle 06 留位 → Cycle 08 OPD-vs-SFT 对齐比较已触发，按 Q4=a 本轮补搜（The Efficiency Misnomer 等），并把本项目 Cycle 03/04 锚点退化记为内部证据。 |

### 删除

无。

### 给下游的提醒（写入 source matrix / source note，非外部文献）

- Exp/Code：(1) teacher=Qwen3-8B-Instruct；(2) cold-start 用 Cycle 07 轨迹 + **overlap probe** 选点（不盲猜步数），⚠️base→instruct+think 格式 gap 使纯 from-base 风险高；(3) 实现用 **top-k KL**（推理服务器约束）；(4) 框架 TRL(慢)/verl(快) 先做 rollout 吞吐 probe。
- Theory/Exp Design：OPD-vs-SFT 比较设计采**轨迹级对比为 primary、单点作 landmark、iso-ID 作二级**（用户 Q4 追问的方法学结论）。
