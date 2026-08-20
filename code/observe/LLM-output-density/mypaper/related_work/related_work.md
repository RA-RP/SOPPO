# Related Work — Detailed Paper Entries (详述)

```yaml
view_type: detailed_bibliography
status: active
maintenance_rule: add_new_entries_per_cycle; remove_only_if_confirmed_unused_and_note_reason
last_updated: 2026-07-07
```

每篇引用论文的具体信息：标题、作者、年份、来源标识、URL、内容概括、与本项目的关联。
综述视角（分类介绍、论文间关系）见 `source_matrix.md`。
每轮新增/删除记录见 `incremental_related_work.md`。

---

## OPD Mechanism Papers

### Agarwal et al., GKD / On-Policy Distillation (ICLR 2024)

- **Authors:** Rishabh Agarwal, Nino Vieillot, Mohammad Ghavamzadeh, et al.
- **Year:** 2024
- **Venue:** ICLR 2024
- **arXiv / URL:** https://arxiv.org/abs/2306.13649
- **Summary:** Proposes GKD (Generalized Knowledge Distillation), training on student self-generated sequences with teacher feedback. Demonstrates that on-policy generation reduces the train-test mismatch inherent in standard KD (which trains on fixed teacher outputs). Evaluated on arithmetic reasoning and instruction-following tasks.
- **Relevance:** Core theoretical basis for why OPD-like training (student-visited states) should differ from SFT. Without this, there is no principled reason to expect OPD and SFT to diverge in behavior.

---

### verl On-Policy Distillation (OPD) Documentation

- **Authors:** verl team
- **Year:** 2024–2025
- **Venue:** Software documentation / technical reference
- **URL:** https://verl.readthedocs.io/en/latest/algo/opd.html
- **Summary:** Describes verl's OPD implementation: OPD samples states from the student policy during training, reducing exposure bias relative to SFT and standard KD, and provides dense token-level supervision rather than sparse RLVR rewards. Defines the practical training loop used in this project.
- **Relevance:** Defines the exact OPD contrast used in this project's experiments. The operational distinction (student-policy sampling vs. fixed SFT sequences) is the mechanism under investigation.

---

### Rethinking On-Policy Distillation of Large Language Models (2026)

- **Authors:** Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding
- **Year:** 2026
- **arXiv / URL:** https://arxiv.org/html/2604.13016v1
- **Summary:** Analyzes OPD phenomenology and identifies a mechanism: OPD succeeds when there is growing overlap between the student's visited states and the teacher's high-probability tokens. When the initial student-teacher gap is too large, cold-start initialization or teacher-aligned prompts are necessary to bring the student into the overlap region.
- **Relevance:** Motivates the less-saturated `theta0` design in Cycle 04/06. Explains why starting from a near-saturated `theta0` may prevent OPD from functioning. Sets expectations for cold-start sensitivity.
- **Cycle 08 recipe specifics (teacher / cold-start / overlap — 从 arXiv 2604.13016 抽取):**
  - **Teacher-student 配对（实测）：** Qwen3-1.7B ← Qwen3-4B（或 4B-Non-Thinking-RL-Math），~2.4×；Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO；DeepSeek-R1-Distill-1.5B ← 7B teachers（~4.7×）。共性：**同家族、teacher ~2–5×、常为 domain-RL 变体**。
  - **Cold-start 机制：** 标准两阶段 = off-policy SFT cold-start（teacher 生成 200K responses，SFT student；OpenThoughts3-1.2M math 子集）→ 再 OPD。cold-start 的作用是抬 **overlap ratio**（student top-k ∩ teacher top-k），成功 OPD 中 **72%→91%**，shared top-k 持 **97–99%** 质量；"只优化 overlap 区即可恢复几乎全部 OPD 收益"。
  - **Cold-start 可跳过性（回答本项目 Q2）：** 纯 OPD-from-base **可能但有 tradeoff**——cold-start 时 OPD 条件差（dense implicit reward 被 outlier 主导）。救援策略：(i) off-policy cold-start SFT，(ii) teacher-aligned prompt selection。报告称 cold-start 约 **3000 SFT 步后 plateau**，**step 500 仍比 base 差**（"minimal teacher-trace exposure 不足以分布对齐"）；注意这是 200K/1.2M 大数据尺度，**步数 scale-dependent，本项目不能照搬**。判据用 **overlap ratio / entropy gap / token-level advantage 的上升趋势**。
  - **对本项目的关联（Cycle 08）：** 给 OPD arm 的 teacher 选择（→ Qwen3-8B 同家族）与 cold-start 选点（→ 用 Cycle 07 轨迹 + overlap probe，而非固定步数）提供直接依据。⚠️ 本项目 student=4B-Base（无 think）+ teacher=8B-Instruct（有 think）+ think-format 训练，初始 overlap 很可能低 → 纯 from-base 风险高，需 overlap probe 定夺最小 cold-start。

---

### A Survey of On-Policy Distillation for Large Language Models (2026)

- **Authors:** Mingyang Song, Mao Zheng
- **Year:** 2026
- **arXiv / URL:** https://huggingface.co/papers/2604.00626
- **Summary:** Surveys OPD methods, organizing them along three axes: feedback signal type, teacher access level (white-box vs. black-box), and loss granularity (token-level vs. sequence-level). Provides a taxonomy of the current OPD landscape.
- **Relevance:** Positions this project within the OPD taxonomy — white-box, token-level, output-space OPD-like. Framing paper for situating contributions.

---

### OPRD: On-Policy Representation Distillation (2026)

- **Authors:** Shenzhi Yang, Guangcheng Zhu, Bowen Song, Haobo Wang, Mingxuan Xia, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen
- **Year:** 2026
- **arXiv / URL:** https://arxiv.org/html/2606.06021v1
- **Summary:** Extends OPD supervision from output token distributions to intermediate hidden representations. Demonstrates that aligning representation-level distributions during on-policy generation improves transfer and stability beyond output-only OPD.
- **Relevance:** Forward pointer — shows the field moving toward representation-aware OPD. This project's output-space-OPD-plus-activation-geometry design is positioned just upstream of this direction, not yet implementing it. Future Cycle may need to engage with this paper more directly.
- **Current use level:** Future direction only.

---

## Reasoning-SFT Generalization & On-Policy Forgetting (Cycle 07)

### Rethinking Generalization in Reasoning SFT (Ren et al., 2026) — **本轮核心来源**

- **Authors:** Qihan Ren, Peng Wang, Ruikun Cai, Shuai Shao, Dadi Guo, Yuejin Xie, Yafu Li, Quanshi Zhang, Xia Hu, Jing Shao, Dongrui Liu (Shanghai AI Lab / SJTU / USTC)
- **Year:** 2026
- **arXiv / URL:** arXiv 2604.06628 ; code github.com/Nebularaid2000/rethink_sft_generalization ; checkpoints on HuggingFace/ModelScope
- **Title:** *Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability*
- **Summary:** Revisits the "SFT memorizes, RL generalizes" narrative for **reasoning SFT on pretrained base models** (Qwen3-14B/8B/4B/1.7B-Base, Qwen2.5-Base, InternLM2.5-20B-Base; default data Math-CoT-20k = OpenR1-Math-220k queries + Qwen3-32B long-CoT responses, math-verify filtered, max 16384 tok; AdamW lr 5e-5, bsz 256, cosine, 8 epochs). Finds cross-domain generalization is **not absent but conditional**, jointly shaped by three factors: (1) **optimization dynamics** — apparent non-generalization is often an under-optimization artifact; cross-domain performance shows a **dip-and-recovery** pattern, so short-epoch checkpoints underestimate generalization; repeated exposure (multi-epoch on small data) beats one-pass; (2) **training data** — low-quality short solutions (NuminaMath-1.5) broadly hurt generalization, while verified long-CoT traces yield consistent cross-domain gains; procedural patterns (backtracking/verification), not domain content, drive transfer (even a Countdown arithmetic game transfers); (3) **model capability** — stronger models internalize transferable procedural patterns while weaker ones imitate surface verbosity; 1.7B-Base shows marginal/negative gains even at late checkpoints, 4B/8B/14B recover and broadly improve. Generalization is **asymmetric**: reasoning improves while safety (HEx-PHI ASR) degrades, more so with CoT. Full per-model/per-data/per-step tables in App D (Tables 5–36); detailed extraction in `source_notes/rethink_sft_experiment_tables.md`.
- **Relevance (本项目多重核心)：**
  1. **模型决策**：直接给出 Qwen3-1.7B-Base 训练失败的实测（Table 6, MATH500 净负），证明本项目 1.7B 天花板是能力地板问题、与 instruct 无关；4B-Base（Table 9）用 long-CoT 在 ID+OOD 全面正向（GPQA 14.6→37.9、MMLU-Pro 35.5→64.8），支持 Cycle 07 转 4B-base。
  2. **数据决策**：点名 NuminaMath-1.5 短解是"广泛损害泛化"的低质量数据，其高质量配方（强 teacher long-CoT + math-verify 过滤）支持本项目离开 NuminaMath。
  3. **格式决策**：long-CoT（含 `<think>`、procedural pattern）是迁移来源，与本项目 Cycle 06"no-think SFT 破坏 thinking 功能"的发现互证；支持 Cycle 07 翻转到 long-CoT think-format。
  4. **评测协议**：ID=MATH500+AIME24、OOD reasoning=LCB v2/GPQA-D/MMLU-Pro，并给出可核对的 base 分数（Table 5），用于本项目 baseline sanity-check（防止重蹈 MATH500 测量 bug）。
  5. **理论深化骨架**：dip-and-recovery + response-length-as-diagnostic（Sec 3.2）+ shallow-imitation vs internalized-pattern 案例（C.3/C.6），是本项目把 SFT 从单点拆成 **SFT_dip / SFT_recover** 轨迹、把 OOD-drop 重定义为 checkpoint 属性的实证基础。
- **Qwen3-4B 训练规格 + baseline（供本项目对齐与 sanity-check；4B 在 Sec 5 能力实验中以默认协议训练，Table 3 / Table 9）：**
  - **超参（default schedule, Table 3）**：AdamW；LR 5e-5；batch size 256；weight decay 0.01；warmup 10% of total steps；cosine decay；grad steps/epoch 80；**epochs 8 → 总 640 步**；max prompt 3072；max response 16384；Verl SFT trainer；8×H200。
  - **训练数据量**：Math-CoT-20k = **20,480 条**（OpenR1-Math-220k query + Qwen3-32B thinking long-CoT + math-verify 过滤）。
  - **训练格式带 `</think>`：是。** target `{Response}` 为 Qwen3-32B（thinking enabled）输出，含 `<think>...</think>` long-CoT，外套 Qwen chat template（`...Please reason step by step, and put your final answer within \boxed{}.`）。佐证：Math-NoCoT-20k 即"把 Math-CoT-20k 的 `<think>...</think>` 删除"。
  - **训练后 baseline（Table 9, step 640）**：MATH500 86.2 / AIME24 30.7 / LCB v2 24.4 / GPQA-D 37.9 / MMLU-Pro 64.8 / IFEval 49.4 / HaluEval 69.4 / TruthfulQA 90.8/84.7（未训 base：70.8 / 11.0 / 20.6 / 14.6 / 35.5 / 49.2）。
  - 完整逐步表与 4B 专题见 `source_notes/rethink_sft_experiment_tables.md` §4/§6。
- **Use level:** Cycle 07 primary reference（model/data/format/eval 四线决策 + 理论深化骨架 + 4B 训练对齐基线）。

### Chen et al., Retaining by Doing: On-Policy Data Mitigates Forgetting (2025)

- **Authors:** Howard Chen, Noam Razin, Karthik Narasimhan, Danqi Chen
- **Year:** 2025
- **arXiv / URL:** https://arxiv.org/abs/2510.18874
- **Summary:** Shows that on-policy data (training on the model's own generations) plays a central role in mitigating catastrophic forgetting during continual post-training, relative to off-policy/fixed-target SFT.
- **Relevance:** **为 Cycle 08 三方框架（OPD vs SFT_dip vs SFT_recover）的 OPD 腿备料。** 直接支撑"OPD ≈ 不经过 dip 暂态、on-policy 保留能力"的机制假设。本轮仅登记，不作 OPD claim。
- **Use level:** Forward-looking source for Cycle 08 OPD leg.

### Shenfeld et al., RL's Razor: Why Online RL Forgets Less (ICLR 2026)

- **Authors:** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal
- **Year:** 2026
- **Venue / URL:** ICLR 2026 ; https://openreview.net/forum?id=7HNRYT4V44
- **Summary:** Argues that online/on-policy reinforcement learning forgets less than off-policy SFT, attributing the difference to mode-seeking updates toward KL-minimal policies that stay close to the base distribution.
- **Relevance:** **Cycle 08 OPD 腿备料。** Provides a mechanistic account ("stay near base distribution") that maps onto the geometry-drift leg of the project's analysis: it predicts OPD should show smaller/flatter drift-from-base than the SFT_dip transient. 本轮仅登记。
- **Use level:** Forward-looking source for Cycle 08 OPD leg.

### Tajwar et al., Preference Fine-Tuning Should Leverage On-Policy Data (ICML 2024)

- **Authors:** Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, Aviral Kumar
- **Year:** 2024
- **Venue / URL:** ICML 2024 ; https://openreview.net/forum?id=bWNPx6t0sF
- **Summary:** Demonstrates that fine-tuning benefits from suboptimal, on-policy data, isolating on-policy sampling (not just data quality) as the source of the advantage.
- **Relevance:** Secondary support for the on-policy-forgetting cluster; grounds the "on-policy sampling itself matters" premise behind OPD. 本轮仅登记，Cycle 08 OPD 腿引用。
- **Use level:** Forward-looking support source.

---

## OPD Recipe, Tooling & Comparison Validity (Cycle 08)

研究笔记：`source_notes/cycle08_opd_teacher_coldstart_tooling.md`。本节服务 Cycle 08 OPD arm 设计（teacher / cold-start / 训练框架）与 OPD-vs-SFT 对齐比较的有效性。

### Law of Capacity Gap in Distilling Language Models (2023)

- **Year:** 2023（v4 更新）
- **arXiv / URL:** https://arxiv.org/abs/2311.07052
- **Summary:** 提出蒸馏的"capacity gap 定律"：**最优 teacher 规模随 student 规模线性变化**，并非越大越好。teacher 太大相对 student → "curse of capacity gap"，student 反而更差（"a superior student often arises from a teacher of a relatively small scale"）。在 <3B 与 7B 上验证。
- **Relevance:** Cycle 08 teacher 选择的核心定律依据。支持本项目为 4B-Base student 选 **Qwen3-8B（~2×）同家族 teacher** 而非 14B/32B；32B 仅作离线数据生成者，不作 live teacher。
- **Use level:** Cycle 08 teacher-choice grounding。

### TRL GKD Trainer + Co-located vLLM (engineering / tooling)

- **Source:** HuggingFace TRL docs（GKD trainer）+ blog "No GPU left behind: Co-located vLLM in TRL"（PR #3394, v0.18.0）
- **URL:** https://github.com/huggingface/trl/blob/main/docs/source/gkd_trainer.md ; https://huggingface.co/blog/vllm-colocate
- **Summary:** TRL GKDTrainer 做 on-policy 蒸馏：`lmbda`（on-policy 比例，0=SFT，1=全 on-policy）、`beta`（JSD 在 forward/reverse KL 间插值）、`teacher_model`。on-policy 生成开销 3–8×；GKD 文档**未提 vLLM**（rollout 走 HF generate）。colocated vLLM（v0.18.0）文档化的是 **GRPO**，GKD/online-DPO 未列；提速 1.4–1.7×。
- **Relevance:** 回答本项目 Q3——TRL **能跑但非高速** OPD（rollout 是 HF-generate 瓶颈）。Code/Exp Design 框架决策依据。
- **Use level:** Engineering / tooling grounding（Cycle 08）。

### verl Async On-Policy KD Recipe (engineering / tooling)

- **Source:** verl 文档（OPD algo + async on-policy distill recipe）
- **URL:** https://verl.readthedocs.io/en/latest/algo/opd.html ; https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html
- **Summary:** verl 高速 OPD 专用方案。`distillation.*` 命名空间，单/多 teacher（`teacher_key` 路由），**teacher 须同 tokenizer（同家族）**（文档例 Qwen3-32B→8B）。loss：**top-k forward KL**（`loss_mode=forward_kl_topk`，topk 默认 32）或 PG-OPD single-sample KL（k1/k3/mse）。**关键工程点：推理服务器只返 sampled token + teacher top-k 的 logprob，不支持任意 token id → 实践用 top-k KL。** rollout 用 vLLM，async teacher 独立 GPU pool（`AsyncTeacherLLMServerManager`，每条 rollout 完成即触发，无 batch barrier）；权重同步优化（~3× + ~4×）。**cold-start 不内置**（上游自备 student）。
- **Relevance:** 回答本项目 Q3——**高速 OPD 的专用方案**；并给实现细节（top-k KL、async teacher、同家族 teacher 约束）。若 Cycle 08 OPD rollout 速度成瓶颈则评估迁移。
- **Use level:** Engineering / tooling grounding（Cycle 08）。

### The Efficiency Misnomer (ICLR 2022) — Matched-Comparison Validity

- **Authors:** Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, Yi Tay
- **Year:** 2021/2022
- **arXiv / URL:** https://arxiv.org/abs/2110.12894
- **Summary:** 论证单一成本指标（参数量 / FLOPs / 吞吐）会**误导**模型对比；公平对比应报多个成本维度。compute-matched / iso-FLOP 对比本身也有陷阱（如削层凑 compute 会人为弱化 baseline）；"无简单公平对比时应 best-effort + 展示多个 alternative"。
- **Relevance:** Cycle 08 control-anchor / matched-comparison validity 的方法学锚点（Q4=a）。支持本项目"OPD vs SFT 对齐比较"的**轨迹级对比（matched checkpoint grid）为主、单点作 landmark、iso-ID 作二级**的设计，并防御 reviewer 的混杂质疑。直接呼应本项目 Cycle 03/04 的"matched-GSM8K-gain 锚点退化"教训。
- **Use level:** Cycle 08 comparison-validity grounding（control-anchor 留位 → 本轮激活）。

---

## OPD/SFT Geometry Competitors & Scoop Audit (Cycle 09)

Cycle 09 = AAAI 竞争力收尾 + scoop 审计。详细 gap map + framing 判断见 `source_notes/cycle09_scoop_audit_and_gap_map.md`（Related Work → Theory 转交文件）。

### On the Geometry of On-Policy Distillation (2026)

- **arXiv / URL:** 2606.07082 — Shen, Li, Yin, Leong, Wang, Chen, Han, Lee, Fung
- **Summary:** OPD vs SFT vs RLVR 的**参数空间**更新几何；OPD 处于 "relaxed off-principal" 区（更少权重、更避开 principal 方向），且早期 "subspace locking"（更新迅速锁进低维通道）。control：**rollout 转 off-policy 仍保持 rank 动态 → on-policy 不是 weight-rank 的驱动**。
- **Relevance (Cycle 09 scoop):** 抢发了"OPD off-principal weight geometry"，使 framing (a) 地基失效。但**纯 weight-space，不碰 activation ER / dip-recover / OOD**。其 off-policy control 支持"数据筛选或可复现 OPD"的猜想。
- **Use level:** Cycle 09 scoop competitor（weight-space）。

### Dense Supervision, Sparse Updates: Sparsity and Geometry of On-Policy Distillation (2026)

- **arXiv / URL:** 2606.13657 — Guo Yu, Wenlin Liu, Yulan Hu, Hao-Xuan Ma, Jun-Peng Jiang, Han-Jia Ye
- **Summary:** OPD 的**权重**更新稀疏、谱集中、落在近零权重坐标；full-param 分析（含子网络恢复）。
- **Relevance (Cycle 09 scoop):** 与 2606.07082 共同坐实 OPD off-principal weight geometry 已发表。**不碰 activation space / OPD-vs-SFT 能力轨迹 / OOD**。
- **Use level:** Cycle 09 scoop competitor（weight-space）。

### Representation Collapse in Sequential Post-Training of LLMs (2026) — **头号威胁**

- **arXiv / URL:** 2605.30524 — Yichen Liu 等
- **Summary:** 顺序 post-training（SFT/DPO/safety，**基本全 LoRA** r16/α32）使隐状态表示塌缩成 low-rank/各向异性/同质化；用指标套件（headline: **effective rank + anisotropy + LoRA overlap + future-task score**）**预测**未来可塑性/OOD/校准退化（混合效应回归，in-sample、无 CV/R²、明确非因果），并用干预（replay/feature-refresh/diversity 正则/LoRA 去相关，三向 Pareto、控目标表现）做因果。**ER = raw 中心化残差流协方差、$\lambda=\sigma^2$、$d$-归一化、固定 12k 多域探针。方向：压缩→OOD 差。**
- **Relevance (Cycle 09 scoop):** **占了"activation 压缩↔OOD"**（但 SFT 族、LoRA、静态预测、方向相反、raw 非白化 ER）。**不做 OPD、不比 OPD-vs-SFT、不做单run dip-recover、非白化谱**——正是我们护城河交集所在。ER 构念对照 + engage 策略见转交文件 §2。
- **Use level:** Cycle 09 primary competitor（representation-space）；须直接 engage。

### SVD-LLM: Truncation-aware SVD for LLM Compression (2024)

- **arXiv / URL:** 2403.07378 — Xin Wang 等
- **Summary:** 白化输入激活使**奇异值与该层输出误差截断最优**对应。
- **Relevance:** **本项目 GetSlice 白化 ER 的方法学来源**——我们的谱是"输出相关白化谱"，区别于 2605.30524 的 raw 表示秩。应作为我们 ER metric 的正式引用。
- **Use level:** Methodology grounding（本项目 ER 定义）。

### RL Fine-Tuning Heals OOD Forgetting in SFT (2025) — **头号威胁 #2**

- **arXiv / URL:** 2509.12235
- **Summary:** RL/on-policy fine-tuning 修复 SFT 造成的 OOD 遗忘（Qwen2.5-7B 恢复至 99%、Llama-3.2-11B 85%）；SVD 分析发现 **OOD 行为跟 singular-vector 旋转、而非 singular-value 变化**。
- **Relevance (Cycle 09):** 与本项目 OPD-vs-SFT 几何主线高度平行（on-policy 修复 SFT OOD + SVD）。其 **"旋转非秩"** 结论**直接质疑本项目基于 effective rank（奇异值）的 core observable**——须 Theory 回应（SVD-LLM 白化 + principal_angle 能否捕捉旋转 / 是否扩展 observable）。与 2605.30524 并列的必答威胁。
- **Use level:** Cycle 09 primary competitor（须直接 engage）。

### Supervised Fine-Tuning on Curated Data is Reinforcement Learning (iw-SFT, 2025)

- **arXiv / URL:** 2507.12856
- **Summary:** 证明 SFT on 筛选/curated 数据 = 最大化 RL 目标下界；iw-SFT（importance-weighted SFT）收紧该界并超越 curated-data SFT。
- **Relevance (Cycle 09):** **"数据筛选让 SFT 逼近 RL/on-policy" 的前提已被发表**——本项目"ER 数据筛选让 SFT 逼近 OPD"创新点须以"ER 几何 criterion 复现 OPD 几何/OOD 签名"差异化，不能声称该前提为新。
- **Use level:** Cycle 09/10 data-selection competitor（须区分）。

### Data-Selection-for-Forgetting Baseline Cluster (2024–2026)

- **Sources:** GrADS (2511.08620, gradient-aware)、FisherSFT (Fisher information)、Gradient-Orthogonality data selection (2602.06359)、Layer-wise-gradient data quality (2504.10766)；机制近邻 Subspace Geometry Governs Forgetting in LoRA (2603.02224)、Low-rank Example Associations (2406.14026)。
- **Relevance (Cycle 10):** 数据筛选创新点的**必打 baseline**（criterion 均为梯度/信息，非 ER/表示几何——本项目差异点）+ geometry↔forgetting 机制近邻。补搜结论：ER/几何 selection criterion 未被直接占，但邻域拥挤。
- **Use level:** Cycle 10 data-selection baselines（待该轴启动时精读）。

### OPD/RL/SFT OOD-Degradation Evidence Cluster (Cycle 09, Theory 供料)

详细逐篇实验设置+结果+问题所在见 `source_notes/cycle09_opd_ood_degradation_evidence.md`。结论：衰退 **SFT ≈ off-policy KD ＞ OPD ＞ RL**；破坏**主要是格式/行为/分布，知识大体保住**。⚠️ "OPD 比 SFT 保 OOD 好"**已被广泛确立** → 本项目不能当新颖 claim。

- **Thinking Machines,《On-Policy Distillation》(blog, 2025)** — Qwen3-8B 个性化：SFT 100%文档 IF-eval 85%→45%，OPD 恢复到 83% 且保知识（内部 QA 41%）；数学 off-policy 60%→RL 68%→OPD 74.4%(1/10 成本)。**明确区分"损坏行为/格式、保住知识"**。
- **nrehiew,《SFT, RL, and OPD Through a Distributional Lens》(blog, 2025)** — LiveCodeBench 遗忘排序 **SFT 0.286 ＜ OPD 0.297–0.314 ＜ RL 0.320**；**"数据来源(on-policy)>teacher"**；机制 **SFT=Forward-KL(mode-covering) vs OPD/RL=Reverse-KL(mode-seeking)**，衰退=分布收窄。
- **Self-Distillation Enables Continual Learning (2601.19897, Shenfeld & Damani)** — SDFT(on-policy 自蒸馏) vs SFT：知识获取 OOD **98% vs 80%**；技能保留 64.5 vs 53.4；推理 answer-only 监督下 SFT 坍缩(23.5%,3273tok) 而 SDFT 保住(43.7%,4180tok)；机制 reverse-KL、距 base **0.68 vs 1.26 nats**。**must-cite（机制近亲）**。
- **Revisiting Catastrophic Forgetting (2406.04836, 2024)** — ⚠️实为 **loss-landscape flatness/SAM** 论文（非知识-vs-格式）：sharper landscape→更多遗忘，SAM 缓解(+7.01,随规模增)。作"遗忘↔参数损失面几何"另一支 + SAM baseline。
- **When Does Online Imitation Help (2606.30445)** — realizability 理论：**非可实现**时 OPD 比 SFT 指数级优势、随 horizon 增长；**可实现**时优势消失。→ OPD 优势 setting-dependent；本项目（long-CoT 长 horizon、32B teacher≫4B student=非可实现）落在"OPD 该有优势"区间（理论依据 + caveat）。
- **⚠️ KL-方向更正（用户 2026-07-07 质疑正确）**：`OPD=reverse KL` 是 conflation；KL 方向 ⊥ on-policy。Self-Distillation=reverse（设计选择）；本项目 **verl OPD 默认=top-k FORWARD KL**（另有 PG-OPD k1/k3=reverse-ish）。**须核实 Cycle 08 `loss_mode`**：若 forward → 本项目 OPD 与 SFT 同向、差别在 on-policy+top-k、reverse-KL 机制不可借用（反利好几何护城河）。

### Knowledge-vs-Format：Forgetting 破坏格式/访问而非知识 (Cycle 09)

回答 Theory Q5。**共识：微调"遗忘"主要是格式/指令-访问退化，参数知识大体保住。** 详见 `source_notes/cycle09_opd_ood_degradation_evidence.md` §C。⚠️ 重磅含义：本项目 C08 的 MMLU-Pro"OOD-drop"很可能是格式/访问中介（支持 C09 output-control gate），几何链精修为"压缩→格式收窄→指令-访问丢失→表观 OOD-drop"。

- **Instruction Vector (2406.12227, LLaMA2-7B-Chat/LoRA)** — 拆知识概率 vs 指令概率；微调后**指令 −10.24、知识 +1.93**；IV 稳定(cos~0.95)、重注入恢复 0.03→0.47 → 遗忘=新 head 抑制原计算图、非知识丢失。**最干净的知识-格式分解证据。**
- **Superficial Alignment / LIMA** — 知识来自预训练、后训练塑格式/风格（经典假设）。
- **Linearly Decoding Refused Knowledge (2507.00239)** — 指令微调压制**生成访问**，但知识在表示里仍可线性解码。
- **Limitations of Instruction Tuning (2402.05119)** — SFT 学输出空间/格式的表层模式，非知识逻辑。
- **Use level:** Cycle 09 Theory 供料（知识 vs 格式；OOD 轴须区分格式 IFEval / 知识闭卷探针）。

---

## SFT Degradation Papers

### On the Impact of Fine-Tuning on Chain-of-Thought Reasoning (NAACL 2025; arXiv 2024)

- **Authors:** Elita Lobo, Chirag Agarwal, Himabindu Lakkaraju
- **Year:** 2025 (conference version); 2024 (arXiv)
- **Venue:** NAACL 2025
- **arXiv / URL:** https://arxiv.org/abs/2411.15382 ; https://aclanthology.org/2025.naacl-long.584/
- **Summary:** Studies how fine-tuning affects CoT reasoning and faithfulness. Reports that task-specific fine-tuning can shift reasoning behavior and reduce CoT faithfulness, even when task-specific performance improves.
- **Relevance:** Establishes that the SFT baseline is not a guaranteed-improvement baseline. For Cycle 06, this source motivates measuring not only whether SFT improves the selected target, but also whether reasoning behavior or adjacent math axes degrade.

---

### Luo et al., Catastrophic Forgetting During Continual Fine-Tuning (2023 / 2025)

- **Authors:** Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang
- **Year:** 2023 arXiv; later journal version in IEEE/ACM TASLP
- **arXiv / URL:** https://arxiv.org/abs/2308.08747
- **Summary:** Empirically evaluates catastrophic forgetting in LLM continual instruction tuning across domain knowledge, reasoning, and reading comprehension. Reports that forgetting is observed for LLMs in the 1B-7B range, directly overlapping the scale of this project.
- **Relevance:** Provides the broad catastrophic-forgetting baseline for Cycle 06. It supports treating SFT-from-base as a live degradation curve rather than assuming monotonic improvement from more SFT samples.

---

### Kalajdzievski, Scaling Laws for Forgetting When Fine-Tuning LLMs (2024)

- **Authors:** Damjan Kalajdzievski
- **Year:** 2024
- **arXiv / DOI:** https://arxiv.org/abs/2401.05605 ; https://doi.org/10.48550/arXiv.2401.05605
- **Summary:** Quantifies forgetting during LLM fine-tuning and reports an inverse relationship between downstream fine-tuning performance and forgetting. Finds that forgetting increases with the number of update steps and that LoRA/PEFT does not eliminate the problem.
- **Relevance:** Directly motivates Cycle 06's scale-curve framing: SFT data size and update count must be treated as experimental variables, not as benign implementation details.

---

### Dong et al., SFT Data Composition and Ability Tradeoffs (ACL 2024)

- **Authors:** Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, Jingren Zhou
- **Year:** 2024
- **Venue / DOI:** ACL 2024, https://aclanthology.org/2024.acl-long.12/ ; https://doi.org/10.18653/v1/2024.acl-long.12
- **Summary:** Studies how math reasoning, code generation, and general alignment abilities scale with SFT data amount, data composition, model size, and training strategy. Finds distinct scaling patterns across abilities, performance conflicts under some data compositions, and catastrophic forgetting under sequential training.
- **Relevance:** Provides the closest methodological analogue for Cycle 06's SFT data-scale question. It supports pre-registering data sizes and reporting ID/OOD axes separately, because different abilities can respond differently to SFT.

---

### Ding and Wang, Improved SFT to Mitigate Catastrophic Forgetting (2025)

- **Authors:** Fei Ding, Baiqiao Wang
- **Year:** 2025
- **arXiv / DOI:** https://arxiv.org/abs/2506.09428 ; https://doi.org/10.48550/arXiv.2506.09428
- **Summary:** Frames SFT as a common source of general-capability degradation when adapting open-source LLMs to specialized domains, especially when the original SFT data is unavailable. Proposes mixing reconstructed general-purpose instruction data with new domain-specific data to reduce forgetting.
- **Relevance:** Not a current method input for Cycle 06, but useful as a mitigation reference. If Cycle 06 finds SFT-from-base improves the target but degrades adjacent math/OOD axes, later cycles can compare OPD-like regularization against rehearsal/mixed-data mitigation baselines.

---

## Spectral / Geometry Mechanism Papers

### LoRA vs Full Fine-tuning: An Illusion of Equivalence (2024)

- **Authors:** Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma
- **Year:** 2024
- **arXiv / URL:** https://arxiv.org/html/2410.21228v3
- **Summary:** Analyzes weight-matrix singular structure in LoRA and full fine-tuning. Identifies "intruder dimensions" — directions that emerge in fine-tuning and correlate with forgetting. Shows that spectral properties of weight matrices can expose qualitatively different update behavior between methods.
- **Relevance:** Justifies treating singular-spectrum movement (via GetSlice activation-conditioned spectra) as a meaningful mechanism proxy rather than an arbitrary diagnostic. See `theory/geometry_metric_definitions.md` for how this connects to the specific metrics used.

---

## Eval Protocol Sources

### Qwen3 Technical Report (2025)

- **Authors:** Qwen Team
- **Year:** 2025
- **arXiv:** 2505.09388
- **URL:** https://arxiv.org/abs/2505.09388
- **Summary:** Official Qwen3 technical report describing model architecture, training, and evaluation. Section on evaluation methodology specifies that MMLU is evaluated in base model mode (no chat template, 5-shot, standard loglikelihood). Report scores for Qwen3-4B on MMLU (~72.99%) align with no-chat-template lm-eval runs, not chat-template runs.
- **Relevance:** Primary authority for MMLU evaluation protocol. Adopted as `mmlu_protocol_version=2` in Cycle 05 after four failing configurations. See `exp/experimental_methodology/evaluation_protocol.md` for full decision record.
- **Use level:** Methodology grounding (eval protocol, not paper argument).

---

### Qwen3 Official Blog / Post-Training and Usage Guide (2025)

- **Authors:** Qwen Team
- **Year:** 2025
- **URL:** https://qwenlm.github.io/blog/qwen3/
- **Summary:** Official Qwen3 release and usage note. Describes the post-training pipeline for hybrid thinking/non-thinking models: long-CoT cold start, reasoning RL, thinking-mode fusion, and general RL. It also shows how `enable_thinking=True/False` is used with `tokenizer.apply_chat_template`.
- **Relevance:** Cycle 06 protocol grounding for Qwen3-1.7B. It makes clear that the post-trained Qwen3 model is not the same experimental object as Qwen3-1.7B-Base: the former has thinking/non-thinking behavior and chat-template control, while the latter is pretraining-only.
- **Use level:** Engineering / protocol grounding.

---

### Qwen/Qwen3-1.7B Model Card

- **Authors:** Qwen Team
- **Year:** 2025
- **URL:** https://huggingface.co/Qwen/Qwen3-1.7B
- **Summary:** Official Hugging Face model card for the post-trained Qwen3-1.7B model. Documents the hybrid thinking mode interface, including `enable_thinking=True` as default in `apply_chat_template`, `enable_thinking=False` for non-thinking mode, and parsing of generated content around the `</think>` token.
- **Relevance:** Directly informs Cycle 06 training/eval format decisions if using the post-trained Qwen3-1.7B checkpoint. It also explains why train/eval mismatch around chat template and thinking blocks can create measurement artifacts.
- **Use level:** Engineering / protocol grounding.

---

### Qwen/Qwen3-1.7B-Base Model Card

- **Authors:** Qwen Team
- **Year:** 2025
- **URL:** https://huggingface.co/Qwen/Qwen3-1.7B-Base
- **Summary:** Official Hugging Face model card for the pretraining-only Qwen3-1.7B-Base checkpoint. Records the model as causal language model, training stage `Pretraining`, 1.7B parameters, 28 layers, 32k context, and points users to Qwen's official benchmark tables.
- **Relevance:** Cycle 06 protocol grounding for the user's proposed base-model arm. This source marks Qwen3-1.7B-Base as a distinct object from the post-trained Qwen3-1.7B; it should not be evaluated with chat/think assumptions unless the experiment explicitly instruction-tunes it into that format.
- **Use level:** Engineering / protocol grounding.

---

### HuggingFaceTB qwen3-1.7b-gsm8k-sft Model Card

- **Authors:** Hugging Face TB / community model card
- **Year:** 2026
- **URL:** https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft
- **Summary:** Community fine-tuned Qwen3-1.7B model optimized for GSM8K. The card reports 247,467 training examples from GSM8K train and MetaMathQA GSM-related examples, conversational messages with `<think>...</think>` reasoning, and a two-stage SFT configuration. It reports high GSM8K accuracy but lower MATH-500 than its stated base comparison.
- **Relevance:** Not an official Qwen baseline, but useful engineering evidence for Cycle 06: a Qwen3-1.7B math SFT setup can improve a targeted GSM8K-style axis while degrading an adjacent MATH-500-style axis, and it uses explicit conversational/think formatting.
- **Use level:** Community engineering reference; do not treat as official benchmark.

---

### Unsloth Qwen3 Run and Fine-Tune Guide

- **Authors:** Unsloth documentation team
- **Year:** 2026
- **URL:** https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
- **Summary:** Community engineering guide for running and fine-tuning Qwen3. Summarizes practical thinking/non-thinking mode settings and notes that non-thinking mode uses empty `<think></think>` framing while thinking mode should avoid greedy decoding.
- **Relevance:** Secondary protocol reference for Cycle 06 implementation checks. Useful for spotting format mismatches in local training/evaluation, but lower authority than Qwen official docs and model cards.
- **Use level:** Community engineering reference.

---

### lm-eval-harness Issues \#3405 / \#3576 / \#3322

- **Authors:** lm-eval-harness community contributors
- **Year:** 2024–2025
- **URL:** https://github.com/EleutherAI/lm-evaluation-harness/issues/3405 (and #3576, #3322)
- **Summary:** Community issue threads confirming that Qwen3 thinking models collapse to random baseline (~0.2295) under MMLU with `--apply_chat_template`, and that no-chat-template base model mode is the correct protocol for these models. Multiple independent users reproduced the behavior.
- **Relevance:** Community-level independent confirmation of the Qwen3 technical report's protocol. Secondary evidence supporting `mmlu_protocol_version=2` adoption.
- **Use level:** Methodology grounding (eval protocol confirmation).

---

### Cycle 07 Candidate Eval Axes (per Rethink SFT protocol — pending Experiment Design adoption)

> 以下为 Cycle 07 拟采纳的评测轴来源，对齐 Rethink SFT 与社区基准（决策 Q4：ID 改 MATH500+AIME24、OOD 采纳论文套件）。按项目规则，**正式采纳与预注册由 Experiment Design 完成**；此处先登记来源与论文可核对 base 分数，供 baseline sanity-check。

- **MATH500** — Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*, NeurIPS 2021 D&B. 500-题子集。本项目已跑通（Cycle 05 修正后协议）。Rethink SFT 用 avg@3。ID 轴。
- **AIME24** — AIME 2024（30 题竞赛级），HuggingFace `math-ai/aime24`. Rethink SFT 用 avg@10。ID 轴（高方差，需多采样）。**本项目新增轴。**
- **LiveCodeBench v2** — Jain et al., *LiveCodeBench: Holistic and Contamination-Free Evaluation of LLMs for Code*, ICLR 2025. 511 题代码推理。Rethink SFT 用 avg@3。OOD reasoning 轴（代码）。**本项目新增轴，工程依赖较高。**
- **GPQA-Diamond** — Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, COLM 2024. 198 题研究生科学。Rethink SFT 用 avg@3、随机 shuffle 选项降位置偏差。OOD reasoning 轴（科学）。**本项目新增轴。**
- **MMLU-Pro** — Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*, arXiv 2406.01574 (2024). Rethink SFT 用 pass@1、均匀抽 1000 题/类。OOD reasoning 轴（知识密集）。与本项目现有 OOD-lite MMLU 不同（MMLU-Pro 更难）。
- **可核对 base 分数（Rethink SFT Table 5，Qwen3-4B-Base，用于 sanity-check）：** MATH500 70.8 / AIME24 11.0 / LCB v2 20.6 / GPQA-D 14.6 / MMLU-Pro 35.5。本项目若自测 4B-base 偏离这些值过大，应先查协议（chat template / think / stop / scorer / few-shot）。
- **Use level:** Cycle 07 candidate eval-protocol grounding（pending Experiment Design adoption & pre-registration）。

---

### Cycle 08 Capability-Degradation Eval Axes (candidate — pending Experiment Design adoption)

> Cycle 08 在 Rethink SFT 套件（偏难推理）之外，补**能力退化/遗忘**覆盖。目标轴=数学（应升）；退化轴=非数学、base 已强的能力。详见 `source_notes/cycle08_capability_degradation_benchmarks.md`（含每个 benchmark 测什么/规模/Qwen3-4B-Base·8B-Base 官方分数）。Relatework 拍板的候选集如下，正式预注册归 Experiment Design。

- **Tier 1（cheap loglikelihood-MC，建议全 checkpoint track）：** MMLU（14,042；知识广度）、HellaSwag（10,042；情境常识）、WinoGrande（1,267；代词消解，OOD-lite 已有）、ARC-Challenge（1,172；科学常识，OOD-lite 已有）、PIQA（1,838；物理常识）、TruthfulQA-MC1（817；真实性，已有）。
- **Tier 2（生成式，关键 checkpoint 抽测）：** BBH（6,511；23 项多步推理，base 强、官方有数，当前缺，强烈建议补）、GSM8K（1,319；易数学，看难题训练是否损易题）、EvalPlus（HumanEval+164/MBPP+378；基础代码）。
- **官方 base 参照（Qwen3 Technical Report arXiv 2505.09388, Table 6/7；仅作量级 sanity-check）：** MMLU 4B-Base 72.99 / 8B-Base 76.89；BBH 72.59 / 78.40；GSM8K 87.79 / 89.84；EvalPlus 63.53 / 67.65；MMLU-Pro 50.58 / 56.73；GPQA 36.87 / 44.44；MATH 54.10 / 60.80。Qwen3 官方**不报 commonsense**（HellaSwag/WinoGrande/ARC/PIQA），需本项目自测。
- **来源（benchmark 原始论文）：** MMLU (Hendrycks 2021)、HellaSwag (Zellers 2019)、WinoGrande (Sakaguchi 2020)、ARC (Clark 2018)、PIQA (Bisk 2020)、BBH (Suzgun 2022)、GSM8K (Cobbe 2021)、EvalPlus (Liu 2023)、TruthfulQA (Lin 2022)。
- **⚠️ 三条警告（见 source note）：** (1) 分数协议依赖、不可跨源混用（官方 ≠ Rethink SFT ≠ 本项目 lm-eval，例 4B-Base MMLU-Pro 官方 50.58 vs Rethink 35.5）→ 本项目统一协议自测；(2) 退化轴须 base 强（IFEval/AlpacaEval base 低、测"获得"非退化）；(3) commonsense 在 4B 须先验判别力（防 1.7B 时 <1.5% 非信号重演）。
- **Use level:** Cycle 08 candidate capability-degradation eval-protocol grounding（pending Experiment Design adoption & pre-registration）。
