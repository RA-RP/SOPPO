# 相关工作 — 论文详述

```yaml
view_type: detailed_bibliography_ch
status: active
maintenance_rule: add_new_entries_per_cycle; remove_only_if_confirmed_unused_and_note_reason
last_updated: 2026-06-27
sync_status: 与 related_work.md 同步至 Cycle 08（2026-06-27；含 Rethink SFT/on-policy 簇/OPD 工具/capacity-gap/Efficiency Misnomer）
```

每篇引用论文的具体信息：标题、作者、年份、来源标识、URL、内容概括、与本项目的关联。
综述视角（分类介绍、论文间关系）见 `source_matrix-ch.md`。
每轮新增/删除记录见 `incremental_related_work-ch.md`。

---

## OPD 机制论文

### Agarwal et al., GKD / On-Policy Distillation（ICLR 2024）

- **作者：** Rishabh Agarwal, Nino Vieillot, Mohammad Ghavamzadeh 等
- **年份：** 2024
- **发表：** ICLR 2024
- **arXiv / URL：** https://arxiv.org/abs/2306.13649
- **内容概括：** 提出 GKD（广义知识蒸馏），在 student 自生成序列上使用 teacher feedback 进行训练。证明 on-policy 生成减少了标准 KD（在固定 teacher 输出上训练）固有的训练-测试分布偏差。在算术推理和指令遵循任务上进行了评测。
- **与本项目的关联：** OPD-like 训练（student 访问的 states）与 SFT 为何应有差异的核心理论依据。没有这一层，就没有理由期待 OPD 和 SFT 在行为上有任何区别。

---

### verl On-Policy Distillation（OPD）文档

- **作者：** verl 团队
- **年份：** 2024–2025
- **类型：** 软件文档 / 技术参考
- **URL：** https://verl.readthedocs.io/en/latest/algo/opd.html
- **内容概括：** 描述 verl 的 OPD 实现：OPD 在训练过程中从 student policy 采样 states，相比 SFT 和标准 KD 减少 exposure bias，提供 dense token-level 监督而非稀疏的 RLVR 奖励。定义了本项目所使用的实际训练循环。
- **与本项目的关联：** 定义了本项目实验中使用的 OPD 实现方式。训练 state distribution 的差异（student policy 采样 vs 固定 SFT 序列）正是本项目研究的机制。

---

### Rethinking On-Policy Distillation of Large Language Models（2026）

- **作者：** Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding
- **年份：** 2026
- **arXiv / URL：** https://arxiv.org/html/2604.13016v1
- **内容概括：** 分析 OPD 的现象学并识别其机制：当 student 访问的 states 与 teacher 高概率 token 的重叠逐渐增加时，OPD 才能成功。当初始 student-teacher gap 过大时，cold-start 初始化或 teacher-aligned prompts 是必要条件。
- **与本项目的关联：** 为 Cycle 04/06 中使用饱和度较低的 `theta0` 提供了依据。解释了为什么从近饱和的 `theta0` 出发可能导致 OPD 失效。设定了对 cold-start 敏感性的预期。
- **Cycle 08 配方实测（teacher / cold-start / overlap，从 2604.13016 抽取）：**
  - **Teacher-student 配对：** Qwen3-1.7B ← Qwen3-4B（或 4B-Non-Thinking-RL-Math），~2.4×；Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO；DeepSeek-R1-Distill-1.5B ← 7B（~4.7×）。共性：**同家族、teacher ~2–5×、常 domain-RL 变体**。
  - **Cold-start：** 标准两阶段 = off-policy SFT cold-start（teacher 生成 200K responses，SFT student）→ 再 OPD；作用是抬 **overlap ratio**（成功 72%→91%，top-k 持 97–99% 质量）。
  - **可跳过性（本项目 Q2）：** 纯 OPD-from-base 可能但条件差；判据用 overlap ratio / entropy gap 上升趋势；~3000 步 plateau 是 200K/1.2M 大数据尺度，**不照搬**。⚠️ 本项目 4B-Base(无think)→8B-Instruct(有think)+think-format，初始 overlap 很可能低 → 纯 from-base 风险高，需 overlap probe。
  - **关联：** 给 Cycle 08 OPD teacher（→8B 同家族）与 cold-start 选点（→Cycle 07 轨迹 + overlap probe）提供直接依据。

---

### A Survey of On-Policy Distillation for Large Language Models（2026）

- **作者：** Mingyang Song, Mao Zheng
- **年份：** 2026
- **arXiv / URL：** https://huggingface.co/papers/2604.00626
- **内容概括：** 对 OPD 方法进行综述，沿三个维度组织分类：反馈信号类型、teacher 访问级别（白盒 vs 黑盒）、loss 粒度（token-level vs sequence-level）。提供了当前 OPD 领域的分类框架。
- **与本项目的关联：** 将本项目定位于 OPD 分类框架中——白盒、token-level、output-space OPD-like。用于论文定位的框架性文献。

---

### OPRD: On-Policy Representation Distillation（2026）

- **作者：** Shenzhi Yang, Guangcheng Zhu, Bowen Song, Haobo Wang, Mingxuan Xia, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen
- **年份：** 2026
- **arXiv / URL：** https://arxiv.org/html/2606.06021v1
- **内容概括：** 将 OPD 监督信号从输出 token 分布扩展到中间隐状态。证明在 on-policy 生成过程中对齐表示层分布可以在 transfer 和稳定性上超越仅 output-space 的 OPD。
- **与本项目的关联：** 前向指针——展示领域向 representation-aware OPD 迁移的趋势。本项目的 output-space OPD + activation geometry 设计恰好处于这一方向的上游，尚未实现表示层方法。未来 Cycle 可能需要更深入地参考此论文。
- **当前使用级别：** 仅作为未来方向。

---

## Reasoning-SFT 泛化 & On-Policy 遗忘（Cycle 07）

### Rethinking Generalization in Reasoning SFT（Ren et al., 2026）— **本轮核心来源**

- **作者：** Qihan Ren, Peng Wang, Ruikun Cai, Shuai Shao, Dadi Guo, Yuejin Xie, Yafu Li, Quanshi Zhang, Xia Hu, Jing Shao, Dongrui Liu（Shanghai AI Lab / 上海交大 / 中科大）
- **年份：** 2026
- **arXiv / URL：** arXiv 2604.06628 ; 代码 github.com/Nebularaid2000/rethink_sft_generalization ; checkpoints 在 HuggingFace/ModelScope
- **标题：** *Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability*
- **内容概括：** 在 **base 模型上的 reasoning SFT** 视角下重审"SFT memorizes, RL generalizes"（用 Qwen3-14B/8B/4B/1.7B-Base、Qwen2.5-Base、InternLM2.5-20B-Base；默认数据 Math-CoT-20k = OpenR1-Math-220k query + Qwen3-32B long-CoT、math-verify 过滤、max 16384 tok；AdamW lr 5e-5、bsz 256、cosine、8 epochs）。发现 cross-domain generalization **不是不存在，而是条件性**的，由三因素联合决定：(1) **optimization dynamics**——表观不泛化常是 under-optimization artifact；cross-domain 表现呈 **dip-and-recovery**，短 epoch checkpoint 低估泛化；重复曝光（小数据多 epoch）优于一遍覆盖；(2) **training data**——低质量短解（NuminaMath-1.5）广泛损害泛化，verified long-CoT 给出一致 cross-domain gain；procedural pattern（backtracking/verification）而非领域内容驱动迁移（连 Countdown 算术游戏都能迁移）；(3) **model capability**——强模型内化可迁移 procedural pattern，弱模型只模仿表面 verbosity；1.7B-Base 即使 late checkpoint 也 marginal/negative，4B/8B/14B 会 recover 并广泛提升。泛化是**不对称**的：reasoning 提升而 safety（HEx-PHI ASR）退化，CoT 更明显。全部 per-model/per-data/per-step 表在 App D（Table 5–36）；抽取见 `source_notes/rethink_sft_experiment_tables.md`。
- **与本项目的关联（多重核心）：**
  1. **模型决策**：直接给出 Qwen3-1.7B-Base 训练失败的实测（Table 6，MATH500 净负），证明本项目 1.7B 天花板是能力地板问题、与 instruct 无关；4B-Base（Table 9）用 long-CoT 在 ID+OOD 全面正向（GPQA 14.6→37.9、MMLU-Pro 35.5→64.8），支持 Cycle 07 转 4B-base。
  2. **数据决策**：点名 NuminaMath-1.5 短解是"广泛损害泛化"的低质量数据，其高质量配方（强 teacher long-CoT + math-verify 过滤）支持本项目离开 NuminaMath。
  3. **格式决策**：long-CoT（含 `<think>`、procedural pattern）是迁移来源，与本项目 Cycle 06"no-think SFT 破坏 thinking 功能"的发现互证；支持 Cycle 07 翻转到 long-CoT think-format。
  4. **评测协议**：ID=MATH500+AIME24、OOD reasoning=LCB v2/GPQA-D/MMLU-Pro，并给出可核对 base 分数（Table 5），用于本项目 baseline sanity-check（防止重蹈 MATH500 测量 bug）。
  5. **理论深化骨架**：dip-and-recovery + response-length-as-diagnostic（Sec 3.2）+ shallow-imitation vs internalized-pattern 案例（C.3/C.6），是本项目把 SFT 从单点拆成 **SFT_dip / SFT_recover** 轨迹、把 OOD-drop 重定义为 checkpoint 属性的实证基础。
- **Qwen3-4B 训练规格 + baseline（供本项目对齐与 sanity-check；4B 在 Sec 5 能力实验中以默认协议训练，Table 3 / Table 9）：**
  - **超参（default schedule, Table 3）**：AdamW；LR 5e-5；batch size 256；weight decay 0.01；warmup 10% of total steps；cosine decay；grad steps/epoch 80；**epochs 8 → 总 640 步**；max prompt 3072；max response 16384；Verl SFT trainer；8×H200。
  - **训练数据量**：Math-CoT-20k = **20,480 条**（OpenR1-Math-220k query + Qwen3-32B thinking long-CoT + math-verify 过滤）。
  - **训练格式带 `</think>`：是。** target `{Response}` 为 Qwen3-32B（thinking enabled）输出，含 `<think>...</think>` long-CoT，外套 Qwen chat template（`...Please reason step by step, and put your final answer within \boxed{}.`）。佐证：Math-NoCoT-20k 即"把 Math-CoT-20k 的 `<think>...</think>` 删除"。
  - **训练后 baseline（Table 9, step 640）**：MATH500 86.2 / AIME24 30.7 / LCB v2 24.4 / GPQA-D 37.9 / MMLU-Pro 64.8 / IFEval 49.4 / HaluEval 69.4 / TruthfulQA 90.8/84.7（未训 base：70.8 / 11.0 / 20.6 / 14.6 / 35.5 / 49.2）。
  - 完整逐步表与 4B 专题见 `source_notes/rethink_sft_experiment_tables.md` §4/§6。
- **使用级别：** Cycle 07 primary reference（model/data/format/eval 四线决策 + 理论深化骨架 + 4B 训练对齐基线）。

### Chen et al., Retaining by Doing: On-Policy Data Mitigates Forgetting（2025）

- **作者：** Howard Chen, Noam Razin, Karthik Narasimhan, Danqi Chen
- **年份：** 2025
- **arXiv / URL：** https://arxiv.org/abs/2510.18874
- **内容概括：** 表明 on-policy 数据（在模型自身生成上训练）在 continual post-training 中对缓解 catastrophic forgetting 起核心作用，相比 off-policy/固定 target 的 SFT。
- **与本项目的关联：** **为 Cycle 08 三方框架（OPD vs SFT_dip vs SFT_recover）的 OPD 腿备料。** 直接支撑"OPD ≈ 不经过 dip 暂态、on-policy 保留能力"的机制假设。本轮仅登记，不作 OPD claim。
- **使用级别：** Forward-looking source（Cycle 08 OPD 腿）。

### Shenfeld et al., RL's Razor: Why Online RL Forgets Less（ICLR 2026）

- **作者：** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal
- **年份：** 2026
- **发表 / URL：** ICLR 2026 ; https://openreview.net/forum?id=7HNRYT4V44
- **内容概括：** 论证 online/on-policy 强化学习比 off-policy SFT 忘得更少，归因于 mode-seeking 更新趋向 KL-minimal policy、贴近 base 分布。
- **与本项目的关联：** **Cycle 08 OPD 腿备料。** 提供机制解释（"贴近 base 分布"），映射到本项目几何分析的 drift 腿：它预测 OPD 的 drift-from-base 应小于/平于 SFT_dip 暂态。本轮仅登记。
- **使用级别：** Forward-looking source（Cycle 08 OPD 腿）。

### Tajwar et al., Preference Fine-Tuning Should Leverage On-Policy Data（ICML 2024）

- **作者：** Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, Aviral Kumar
- **年份：** 2024
- **发表 / URL：** ICML 2024 ; https://openreview.net/forum?id=bWNPx6t0sF
- **内容概括：** 证明 fine-tuning 受益于 suboptimal、on-policy 数据，把 on-policy 采样（而非仅数据质量）单独识别为优势来源。
- **与本项目的关联：** on-policy-forgetting 文献簇的次级支撑；为 OPD 背后的"on-policy 采样本身重要"前提提供依据。本轮仅登记，Cycle 08 OPD 腿引用。
- **使用级别：** Forward-looking support source。

---

## OPD 配方 / 工具 / 对比有效性（Cycle 08）

研究笔记：`source_notes/cycle08_opd_teacher_coldstart_tooling.md`。服务 Cycle 08 OPD arm 设计（teacher / cold-start / 训练框架）与 OPD-vs-SFT 对齐比较有效性。

### Law of Capacity Gap in Distilling Language Models（2023）

- **年份：** 2023（v4）
- **arXiv / URL：** https://arxiv.org/abs/2311.07052
- **内容概括：** 蒸馏 "capacity gap 定律"：**最优 teacher 随 student 规模线性变化**，非越大越好；teacher 太大 → "curse of capacity gap"，student 反而更差。在 <3B 与 7B 上验证。
- **与本项目的关联：** Cycle 08 teacher 选择核心依据。支持 4B-Base student 选 **Qwen3-8B（~2×）同家族 teacher**，非 14B/32B；32B 仅离线数据生成。
- **使用级别：** Cycle 08 teacher 选择依据。

### TRL GKD Trainer + Co-located vLLM（工程 / 工具）

- **来源：** HuggingFace TRL docs + 博客 "Co-located vLLM in TRL"（PR #3394, v0.18.0）
- **URL：** https://github.com/huggingface/trl/blob/main/docs/source/gkd_trainer.md ; https://huggingface.co/blog/vllm-colocate
- **内容概括：** GKDTrainer 做 on-policy 蒸馏（`lmbda`/`beta`/`teacher_model`）；on-policy 生成开销 3–8×；GKD 文档未提 vLLM（rollout 走 HF generate）；colocate 文档化的是 GRPO，GKD 未列，提速 1.4–1.7×。
- **与本项目的关联：** Q3——TRL **能跑但非高速** OPD。框架决策依据。
- **使用级别：** 工程 / 工具依据（Cycle 08）。

### verl Async On-Policy KD Recipe（工程 / 工具）

- **来源：** verl 文档（OPD algo + async on-policy distill recipe）
- **URL：** https://verl.readthedocs.io/en/latest/algo/opd.html ; https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html
- **内容概括：** 高速 OPD 专用。`distillation.*`，单/多 teacher（`teacher_key` 路由），teacher 须同 tokenizer（同家族，文档例 Qwen3-32B→8B）。loss：**top-k forward KL**（`forward_kl_topk`，topk 默认 32）或 PG-OPD（k1/k3/mse）。**关键：推理服务器只返 sampled token + teacher top-k 的 logprob，不支持任意 token id → 实践用 top-k KL。** vLLM rollout + async teacher 独立 pool（每条 rollout 完成即触发，无 batch barrier）；权重同步优化（~3×+~4×）。cold-start 不内置。
- **与本项目的关联：** Q3——**高速 OPD 专用方案** + 实现约束（top-k KL、async teacher、同家族 teacher）。OPD rollout 成瓶颈则评估迁移。
- **使用级别：** 工程 / 工具依据（Cycle 08）。

### The Efficiency Misnomer（ICLR 2022）— Matched-Comparison Validity

- **作者：** Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, Yi Tay
- **年份：** 2021/2022
- **arXiv / URL：** https://arxiv.org/abs/2110.12894
- **内容概括：** 单一成本/单点指标会误导模型对比；公平对比应报多维 + best-effort 多 alternative；compute-matched/iso-FLOP 本身也有陷阱（削层凑 compute 会弱化 baseline）。
- **与本项目的关联：** Cycle 08 control-anchor / matched-comparison validity 方法学锚点（Q4=a）。支持"OPD vs SFT 对齐比较"采**轨迹级对比为主、单点作 landmark、iso-ID 作二级**；呼应本项目 Cycle 03/04 "matched-GSM8K-gain 锚点退化"教训。
- **使用级别：** Cycle 08 对比有效性依据（control-anchor 留位 → 激活）。

---

## SFT 退化论文

### On the Impact of Fine-Tuning on Chain-of-Thought Reasoning（NAACL 2025；arXiv 2024）

- **作者：** Elita Lobo, Chirag Agarwal, Himabindu Lakkaraju
- **年份：** 2025（会议版）；2024（arXiv）
- **发表：** NAACL 2025
- **arXiv / URL：** https://arxiv.org/abs/2411.15382 ; https://aclanthology.org/2025.naacl-long.584/
- **内容概括：** 研究 fine-tuning 对 CoT 推理和 faithfulness 的影响。结果显示，任务特定 fine-tuning 可能改变推理行为并降低 CoT faithfulness，即使任务特定表现有所提升。
- **与本项目的关联：** 建立 SFT baseline 并非"保证提升"的基准。对 Cycle 06 来说，它支持同时测量 SFT 是否提升目标轴，以及是否损害推理行为或相邻数学轴。

---

### Luo et al., Catastrophic Forgetting During Continual Fine-Tuning（2023 / 2025）

- **作者：** Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang
- **年份：** 2023 arXiv；后续 IEEE/ACM TASLP 期刊版
- **arXiv / URL：** https://arxiv.org/abs/2308.08747
- **内容概括：** 从领域知识、推理、阅读理解等角度经验评估 LLM continual instruction tuning 中的 catastrophic forgetting。报告称 1B-7B 规模模型普遍会出现 forgetting，正好覆盖本项目 1.7B 模型规模。
- **与本项目的关联：** 为 Cycle 06 提供宽口径 catastrophic forgetting baseline。它支持将 SFT-from-base 视为需要实际测量的退化曲线，而不是假设更多 SFT 样本必然单调提升。

---

### Kalajdzievski, Scaling Laws for Forgetting When Fine-Tuning LLMs（2024）

- **作者：** Damjan Kalajdzievski
- **年份：** 2024
- **arXiv / DOI：** https://arxiv.org/abs/2401.05605 ; https://doi.org/10.48550/arXiv.2401.05605
- **内容概括：** 量化 LLM fine-tuning 中的 forgetting，并报告下游 fine-tuning 表现和 forgetting 之间的反向关系。研究还指出 forgetting 会随 update steps 增加，LoRA/PEFT 并不能消除该问题。
- **与本项目的关联：** 直接支持 Cycle 06 的 scale-curve 设计：SFT 数据量和更新步数必须作为实验变量，而不是无害的实现细节。

---

### Dong et al., SFT Data Composition and Ability Tradeoffs（ACL 2024）

- **作者：** Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, Jingren Zhou
- **年份：** 2024
- **发表 / DOI：** ACL 2024, https://aclanthology.org/2024.acl-long.12/ ; https://doi.org/10.18653/v1/2024.acl-long.12
- **内容概括：** 研究数学推理、代码生成、通用 alignment 能力如何随 SFT 数据量、数据组成、模型规模和训练策略变化。发现不同能力有不同 scaling pattern，某些数据组合会带来 performance conflict，sequential training 会导致 catastrophic forgetting。
- **与本项目的关联：** 与 Cycle 06 的 SFT data-scale 问题最接近。它支持预注册数据规模，并分别报告 ID/OOD 轴，因为不同能力可能对 SFT 作出不同响应。

---

### Ding and Wang, Improved SFT to Mitigate Catastrophic Forgetting（2025）

- **作者：** Fei Ding, Baiqiao Wang
- **年份：** 2025
- **arXiv / DOI：** https://arxiv.org/abs/2506.09428 ; https://doi.org/10.48550/arXiv.2506.09428
- **内容概括：** 将 SFT 描述为 open-source LLM 适配专门领域时常见的 general capability degradation 来源，尤其当原始 SFT 数据不可获得时更严重。论文提出将重构的通用 instruction 数据与新领域数据混合，以缓解 forgetting。
- **与本项目的关联：** 不是 Cycle 06 的当前方法输入，但可作为 mitigation 参考。如果 Cycle 06 发现 SFT-from-base 提升目标轴但损害相邻数学/OOD 轴，后续 cycle 可将 OPD-like regularization 与 rehearsal/mixed-data mitigation baseline 比较。

---

## 谱 / 几何机制论文

### LoRA vs Full Fine-tuning: An Illusion of Equivalence（2024）

- **作者：** Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma
- **年份：** 2024
- **arXiv / URL：** https://arxiv.org/html/2410.21228v3
- **内容概括：** 分析 LoRA 和 full fine-tuning 的 weight-matrix 奇异结构。识别出"intruder dimensions"——fine-tuning 中出现的与遗忘相关的新方向。证明 weight matrix 的谱属性可以揭示两种方法之间质性不同的更新行为。
- **与本项目的关联：** 为将奇异谱移动（通过 GetSlice activation-conditioned spectra）作为有意义的机制代理（而非任意指标）提供了依据。详见 `theory/geometry_metric_definitions.md`（"Design Rationale"）。

---

## 评测协议来源

### Qwen3 Technical Report（2025）

- **作者：** Qwen 团队
- **年份：** 2025
- **arXiv：** 2505.09388
- **URL：** https://arxiv.org/abs/2505.09388
- **内容概括：** Qwen3 官方技术报告，描述模型架构、训练和评测。评测方法部分明确说明 MMLU 使用 base model 模式（no chat template，5-shot，standard loglikelihood）。Qwen3-4B 的报告 MMLU 分数（~72.99%）与 no-chat-template lm-eval 运行对齐，与 chat-template 运行不对齐。
- **与本项目的关联：** MMLU 评测协议的主要权威来源。在 4 种配置均失败后，通过查阅官方报告确认采用 base model mode（`mmlu_protocol_version=2`）。详见 `exp/experimental_methodology/evaluation_protocol.md`。
- **使用级别：** Methodology grounding（评测协议依据，非论文论点）。

---

### Qwen3 官方博客 / Post-training 与使用说明（2025）

- **作者：** Qwen 团队
- **年份：** 2025
- **URL：** https://qwenlm.github.io/blog/qwen3/
- **内容概括：** Qwen3 官方发布与使用说明。描述 hybrid thinking/non-thinking 模型的 post-training pipeline：long-CoT cold start、reasoning RL、thinking-mode fusion、general RL。也展示了如何在 `tokenizer.apply_chat_template` 中使用 `enable_thinking=True/False`。
- **与本项目的关联：** Cycle 06 中 Qwen3-1.7B 协议设定的依据。它说明 post-trained Qwen3 与 Qwen3-1.7B-Base 不是同一个实验对象：前者有 thinking/non-thinking 和 chat-template 控制，后者是 pretraining-only。
- **使用级别：** 工程 / 协议依据。

---

### Qwen/Qwen3-1.7B Model Card

- **作者：** Qwen 团队
- **年份：** 2025
- **URL：** https://huggingface.co/Qwen/Qwen3-1.7B
- **内容概括：** post-trained Qwen3-1.7B 的官方 Hugging Face model card。记录 hybrid thinking mode 接口，包括 `apply_chat_template` 中默认 `enable_thinking=True`、使用 `enable_thinking=False` 进入 non-thinking mode，以及围绕 `</think>` token 解析生成内容。
- **与本项目的关联：** 如果 Cycle 06 使用 post-trained Qwen3-1.7B checkpoint，它直接决定 train/eval 格式。也解释了为什么 chat template 和 thinking block 的推训不一致会造成测量 artifact。
- **使用级别：** 工程 / 协议依据。

---

### Qwen/Qwen3-1.7B-Base Model Card

- **作者：** Qwen 团队
- **年份：** 2025
- **URL：** https://huggingface.co/Qwen/Qwen3-1.7B-Base
- **内容概括：** pretraining-only Qwen3-1.7B-Base 的官方 Hugging Face model card。将模型记录为 causal language model，训练阶段为 `Pretraining`，参数量 1.7B，28 层，32k context，并指向 Qwen 官方 benchmark 表。
- **与本项目的关联：** Cycle 06 中 base-model arm 的协议依据。该来源明确 Qwen3-1.7B-Base 与 post-trained Qwen3-1.7B 是不同对象；除非实验明确把它 instruction-tune 到 chat/think 格式，否则不应套用 chat/think 假设。
- **使用级别：** 工程 / 协议依据。

---

### HuggingFaceTB qwen3-1.7b-gsm8k-sft Model Card

- **作者：** Hugging Face TB / 社区 model card
- **年份：** 2026
- **URL：** https://huggingface.co/HuggingFaceTB/qwen3-1.7b-gsm8k-sft
- **内容概括：** 一个针对 GSM8K 数学推理优化的社区 Qwen3-1.7B fine-tuned model。model card 报告训练使用 247,467 个样本，来源为 GSM8K train 和 MetaMathQA GSM-related examples，格式为带 `<think>...</think>` 推理的 conversational messages，并给出两阶段 SFT 配置。其结果表现为 GSM8K 较高，但 MATH-500 低于其声明的 base comparison。
- **与本项目的关联：** 不是 Qwen 官方 baseline，但对 Cycle 06 很有工程参考价值：Qwen3-1.7B math SFT 可能提升目标 GSM8K-style 轴，同时损害相邻 MATH-500-style 轴，而且训练格式显式使用 conversational/think。
- **使用级别：** 社区工程参考；不得当作官方 benchmark。

---

### Unsloth Qwen3 Run and Fine-Tune Guide

- **作者：** Unsloth 文档团队
- **年份：** 2026
- **URL：** https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
- **内容概括：** Qwen3 运行与 fine-tune 的社区工程指南。总结 thinking/non-thinking mode 的实际设置，并指出 non-thinking mode 使用空 `<think></think>` framing，thinking mode 不应使用 greedy decoding。
- **与本项目的关联：** Cycle 06 实现检查的二级协议参考。可用于发现本地训练/评测格式不一致，但权威性低于 Qwen 官方文档和 model card。
- **使用级别：** 社区工程参考。

---

### lm-eval-harness Issues \#3405 / \#3576 / \#3322

- **作者：** lm-eval-harness 社区贡献者
- **年份：** 2024–2025
- **URL：** https://github.com/EleutherAI/lm-evaluation-harness/issues/3405 （及 #3576、#3322）
- **内容概括：** 社区 issue 讨论帖，确认 Qwen3 思维模型在 MMLU 使用 `--apply_chat_template` 时会崩溃到随机基线（~0.2295），并确认 no-chat-template base model 模式是这类模型的正确协议。多位独立用户复现了该行为。
- **与本项目的关联：** Qwen3 技术报告协议的社区级独立确认。支持 `mmlu_protocol_version=2` 采纳的二次证据。
- **使用级别：** Methodology grounding（评测协议确认）。

---

### Cycle 07 候选评测轴（按 Rethink SFT 协议 — 待 Experiment Design 采纳）

> 以下为 Cycle 07 拟采纳的评测轴来源，对齐 Rethink SFT 与社区基准（决策 Q4：ID 改 MATH500+AIME24、OOD 采纳论文套件）。按项目规则，**正式采纳与预注册由 Experiment Design 完成**；此处先登记来源与论文可核对 base 分数，供 baseline sanity-check。

- **MATH500** — Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*, NeurIPS 2021 D&B。500-题子集。本项目已跑通（Cycle 05 修正后协议）。Rethink SFT 用 avg@3。ID 轴。
- **AIME24** — AIME 2024（30 题竞赛级），HuggingFace `math-ai/aime24`。Rethink SFT 用 avg@10。ID 轴（高方差，需多采样）。**本项目新增轴。**
- **LiveCodeBench v2** — Jain et al., *LiveCodeBench: Holistic and Contamination-Free Evaluation of LLMs for Code*, ICLR 2025。511 题代码推理。Rethink SFT 用 avg@3。OOD reasoning 轴（代码）。**本项目新增轴，工程依赖较高。**
- **GPQA-Diamond** — Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, COLM 2024。198 题研究生科学。Rethink SFT 用 avg@3、随机 shuffle 选项降位置偏差。OOD reasoning 轴（科学）。**本项目新增轴。**
- **MMLU-Pro** — Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*, arXiv 2406.01574 (2024)。Rethink SFT 用 pass@1、均匀抽 1000 题/类。OOD reasoning 轴（知识密集）。与本项目现有 OOD-lite MMLU 不同（MMLU-Pro 更难）。
- **可核对 base 分数（Rethink SFT Table 5，Qwen3-4B-Base，用于 sanity-check）：** MATH500 70.8 / AIME24 11.0 / LCB v2 20.6 / GPQA-D 14.6 / MMLU-Pro 35.5。本项目若自测 4B-base 偏离这些值过大，应先查协议（chat template / think / stop / scorer / few-shot）。
- **使用级别：** Cycle 07 候选评测协议依据（pending Experiment Design adoption & pre-registration）。

---

### Cycle 08 能力退化评测轴（候选 — 待 Experiment Design 采纳）

> Cycle 08 在 Rethink SFT 套件（偏难推理）之外补**能力退化/遗忘**覆盖。目标轴=数学（应升）；退化轴=非数学、base 已强的能力。详见 `source_notes/cycle08_capability_degradation_benchmarks.md`（每个 benchmark 测什么/规模/Qwen3-4B·8B-Base 官方分数）。

- **Tier 1（cheap loglikelihood-MC，建议全 checkpoint track）：** MMLU（14,042）、HellaSwag（10,042）、WinoGrande（1,267，OOD-lite 已有）、ARC-Challenge（1,172，OOD-lite 已有）、PIQA（1,838）、TruthfulQA-MC1（817，已有）。
- **Tier 2（生成式，关键 checkpoint 抽测）：** BBH（6,511，多步推理，base 强、官方有数，当前缺，强烈建议补）、GSM8K（1,319，易数学）、EvalPlus（HE+164/MBPP+378，基础代码）。
- **官方 base 参照（Qwen3 Technical Report 2505.09388, Table 6/7）：** MMLU 4B 72.99 / 8B 76.89；BBH 72.59 / 78.40；GSM8K 87.79 / 89.84；EvalPlus 63.53 / 67.65；MMLU-Pro 50.58 / 56.73；GPQA 36.87 / 44.44；MATH 54.10 / 60.80。Qwen3 官方**不报 commonsense**，需自测。
- **来源：** MMLU (Hendrycks 2021)、HellaSwag (Zellers 2019)、WinoGrande (Sakaguchi 2020)、ARC (Clark 2018)、PIQA (Bisk 2020)、BBH (Suzgun 2022)、GSM8K (Cobbe 2021)、EvalPlus (Liu 2023)、TruthfulQA (Lin 2022)。
- **⚠️ 三条警告：** (1) 分数协议依赖、不可跨源混用（官方 ≠ Rethink SFT ≠ 本项目 lm-eval，例 4B-Base MMLU-Pro 官方 50.58 vs Rethink 35.5）→ 本项目统一协议自测；(2) 退化轴须 base 强（IFEval/AlpacaEval 测"获得"非退化）；(3) commonsense 在 4B 须先验判别力（防 1.7B 时 <1.5% 非信号重演）。
- **使用级别：** Cycle 08 候选能力退化评测协议依据（pending Experiment Design adoption & pre-registration）。
