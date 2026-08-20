# Source Note: Cycle 08 OPD — Teacher 选择 / Cold-start 步数 / TRL 高速训练可行性

```yaml
view_type: extended_source_note
cycle: cycle_08_h_opd_vs_sft_comparison
purpose: 回答 Cycle 08 OPD 设计的三个空缺（teacher 模型 / SFT cold-start 步数 / 高速训练工具），并给本项目 4B-Base 场景的文献依据与建议
status: reference
created: 2026-06-27
sources:
  - "Song & Zheng, A Survey of On-Policy Distillation, arXiv 2604.00626 (本地笔记 literature_library/summaries/OPD相关工作.pdf)"
  - "Li et al., Rethinking On-Policy Distillation, arXiv 2604.13016 (cold-start/overlap recipe)"
  - "Law of Capacity Gap in Distilling LMs, arXiv 2311.07052 (teacher 选择定律)"
  - "Distillation Scaling Laws, arXiv 2502.08606"
  - "TRL GKD Trainer docs + vLLM colocate (huggingface.co/blog/vllm-colocate, PR #3394, v0.18.0)"
  - "verl Async On-Policy KD recipe (verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)"
note: 本笔记是 Related Work 的文献脚手架；teacher/cold-start 的最终预注册归 Experiment Design，框架/benchmark 归 Code。
```

Cycle 08 intent 的 OPD arm 设计表**没有指定 teacher 模型，也没有指定 cold-start 步数**——这两项是 OPD 的必需输入。本笔记按文献回答，并落到本项目 4B-Base 场景。

---

## Q1：OPD 用什么模型作为 teacher？

### 定律：teacher 不是越强越好（capacity gap）

- **Law of Capacity Gap（arXiv 2311.07052）**：**最优 teacher 规模随 student 规模线性变化**；teacher 太大 → "curse of capacity gap" → student 反而更差。"A superior student often arises from a teacher of a relatively small scale instead of a larger one." 在 <3B 和 7B 上都验证过。
- OPD survey（本地笔记）一致："teacher 不是越强越好；teacher 太强而 student 太弱时出现 capacity gap，student 吸收能力饱和。"
- 蒸馏 scaling law（2502.08606 / survey）：compute 要在 teacher inference / student rollout / student update 间分配，teacher 越大 inference 成本越高，边际收益递减。

### 实际 OPD 论文的 teacher-student 配对（Rethinking OPD, Li et al. 2604.13016）

| Student | Teacher | 比例 | 备注 |
|---|---|---|---|
| Qwen3-1.7B (non-thinking) | **Qwen3-4B** / Qwen3-4B-Non-Thinking-RL-Math | ~2.4× | **同家族**；teacher 常为 domain-RL'd 变体 |
| **Qwen3-1.7B-Base** | **Qwen3-4B-Base-GRPO** | ~2.4× | 同家族 base-to-base，teacher 经 GRPO |
| DeepSeek-R1-Distill-Qwen-1.5B | R1-Distill-Qwen-7B / Skywork-OR1-Math-7B | ~4.7× | 同 tokenizer 家族 |

**共性模式：** (1) **同家族**（同 tokenizer → top-k KL 直接可算，无需跨词表对齐 hack）；(2) teacher ≈ **2–5× student**；(3) teacher 常是**领域 RL/specialized 变体**（RL-Math、GRPO），不是裸 base。

### 落到本项目（student = Qwen3-4B-Base）

| 候选 teacher | 比例 | 评价 |
|---|---|---|
| **Qwen3-8B（instruct 或 math-RL 变体）** | 2× | **建议甜点**：镜像论文 1.7B←4B(2.4×)，同家族，gap 适中，成本可控 |
| Qwen3-14B | 3.5× | 上界；更强但 capacity-gap 风险与成本上升 |
| Qwen3-32B（= Math-CoT-20k 数据生成者） | 8× | **不建议作 live teacher**：gap 过大 + rollout 打分成本极高；它适合离线生成 SFT 数据，不适合在线 OPD |
| Qwen3-4B-Instruct（同尺寸） | 1× | 便宜的 capacity-matched baseline，白盒同 tokenizer；但"teacher 优势"弱（同尺寸），是本项目旧 setup（4B-instruct teacher → 1.7B student）的逻辑延伸 |

> **结论：Qwen3-8B 同家族 teacher 是文献最支持的选择。** 若存在 Qwen3-8B 的 math-RL/GRPO 变体则更佳（贴合论文"domain-RL teacher"模式）；否则 8B-instruct。32B 仅作离线数据生成者，不作 live teacher。

---

## Q2：SFT cold-start 多少步？

### 为什么需要 cold-start

- OPD 在 cold-start 时**条件极差**：dense implicit reward 被少数 outlier 主导，student 分布与 teacher 高概率区重叠太低，直接 OPD 会不稳定/无效。
- 标准配方 = **off-policy SFT cold-start → 再 OPD**。cold-start 的作用是把 student **推进 teacher 的 overlap 区**。

### 机制与判据（Rethinking OPD, Li et al.）

- **overlap ratio**（student top-k ∩ teacher top-k）是核心量：成功 OPD 中从 **72% 升到 91%**；shared top-k 持有 **97–99%** 概率质量。"只优化 overlap 区就能恢复几乎全部 OPD 收益。"
- 他们的 cold-start：teacher 生成 **200K** 条 responses，用来 SFT student（OpenThoughts3-1.2M math 子集）。
- 评估不同 SFT 步数的 cold-start checkpoint：早期 checkpoint 比 base 还差、distill 无收益；随 SFT 步数增加而改善（student 分布逐渐收敛向 teacher）；**报告称约 3000 SFT 步后性能 plateau**（注意：他们数据量远大于本项目，此步数 scale-dependent，不能照搬）。
- 切换判据（survey + 本项目笔记）：**teacher-student KL / pass rate / overlap ratio**。

### 落到本项目（关键洞察：cold-start 轨迹已经有了）

**本项目不需要盲猜 cold-start 步数——Cycle 07 已经在 4B-Base + Math-CoT-20k 上跑出 10 点 SFT 轨迹（step_000…624）。** 因此：

1. **用 overlap ratio 选 cold-start 点**：对选定 teacher，在每个 Cycle 07 checkpoint 上测 student-top-k ∩ teacher-top-k overlap，选 overlap 越过条件化阈值（文献经验 ~70%+ 且在上升）的最早 checkpoint。把 overlap 数值写进设计作为依据（防止重蹈"没测就拍脑袋"）。
2. **⚠️ cold-start 选择与 H-OPD 假设强耦合（核心设计张力）：**
   - cold-start **太晚**（step_320 peak / recover）→ OPD 从已 recover 状态起步 → **无法测"OPD 是否避开 dip"**（Gate B08 子条件 2 变空洞）。
   - cold-start **太早/极小**（base / pre-dip）→ 能测 H-OPD，但 OPD 可能因 overlap 太低而条件化差 → 可能**因错误原因失败**。
   - **解法**：选 overlap 刚越过条件化阈值、但在 SFT recovery 巩固之前的 checkpoint；用 overlap 数值证明其"足够条件化但未 recover"。这是一个真正的 Experiment Design 决策。
3. Cycle 07 轨迹锚点（来自 intent）：dip=step_20（finisher acc 0.594）、recovery=step_160（首超 base）、peak=step_320（MATH500 0.698）、final=step_624（0.680）。候选 cold-start 集中在 **step_20–160** 之间需测 overlap 定夺。
4. 历史对照：本项目旧 theta0 = 轻量 SFT（Cycle 04 用 256 prompts）。Cycle 07 的 10 点 grid 是它的现代化、可测版本。

### cold-start 能否跳过（用户 Q2 追问，文献回答）

- **能跳过吗？** 纯 OPD-from-base **可能但有 tradeoff**：cold-start 时 OPD 条件差（dense implicit reward 被少数 outlier 主导）。救援策略：(i) off-policy cold-start SFT、(ii) teacher-aligned prompt selection。
- **能小规模测吗？** 能，文献给了现成诊断：跑几十步 OPD，log **overlap ratio / entropy gap / token-level advantage** 轨迹。overlap 起点够高且**上升** → 可跳过；低且不升 → 需 cold-start。这是便宜的 probe。
- **要多少步？** 不能照搬 Rethinking OPD 的 ~3000 步 plateau（step 500 仍比 base 差）——那是 200K teacher responses / OpenThoughts-1.2M 大数据尺度。本项目 20k+LoRA+640 步 grid 完全不同，**按 overlap 选点，不按固定步数**。
- **⚠️ 本项目特殊风险（决定性）**：student=Qwen3-4B-**Base**（无 chat/think）+ teacher=Qwen3-8B-**Instruct**（有 chat/think）+ **think-format 训练**。从 raw base(step_000) 起，student 根本不产 think-format → 与 instruct teacher 的初始 overlap 很可能**很低** → **纯 from-base（用户倾向的 b）风险高**。但 Cycle 07 已把 4B-Base SFT 成 think-format。
- **建议（a/b 混合）**：Exp/Code 先测 step_000 的 overlap——若已够高且 probe 中上升，用 b（无 cold-start）；否则取"刚够获得 think-format + 过 overlap 阈值"的**最小** Cycle 07 checkpoint 作 cold-start。最终归 Exp/Code 实测预注册。

---

## Q3：现在的 TRL 能不能满足高速训练 OPD？

### TRL 现状

- **TRL GKDTrainer 功能上能做 OPD**：`lmbda`（on-policy 比例，0=纯 SFT，1=全 on-policy）、`beta`（JSD 在 forward/reverse KL 间插值）、`teacher_model`。
- **但高速是短板**：on-policy 生成开销是 off-policy 的 **3–8×**；GKD trainer 文档**未提 vLLM**，即其 rollout 仍走 HF `generate`（慢）。
- TRL v0.18.0（PR #3394）加了 **colocated vLLM**（训练/推理同 GPU 轮流，消除 idle），但博客明确只讲 **GRPO**；**GKD / online-DPO 未列为已支持**。且 colocate 提速本身有限（1.5B 1.43× / 7B 1.73× / 72B 1.26×）。

### verl 现状（高速 OPD 的专用方案）

- verl 有 **Async On-Policy KD recipe**，专为高速 OPD 设计：
  - **vLLM 做 rollout**（pure inference mode，Ray futures 异步生成）。
  - **异步重叠** generation / teacher-query / update（One-Step-Off 需 2 warmup，Two-Step-Off 需 3 warmup）。
  - 权重同步优化：批量 tensor load（~3×）+ Megatron 用 gather-to-root 替 allgather（~4×）。
  - teacher 信号 = 每 token top-k log-prob + indices；student = token-wise sparse KL；TP-aware KL kernel；ZeroMQ 异步 teacher 查询。
  - （文档未给绝对吞吐对比，但明确"以放弃严格 on-policy 换取端到端吞吐与硬件利用率的大幅提升"。）

### 落到本项目

| | TRL | verl async OPD |
|---|---|---|
| 能否跑 OPD | ✅ GKDTrainer | ✅ 专用 recipe |
| rollout 加速 | HF generate（慢）；colocate vLLM 只对 GRPO 文档化 | ✅ vLLM + 异步重叠 |
| 适配规模 | 单 arm / 小实验勉强可接受但慢 | 高速/可扩展、工程更重 |
| 本项目现状 | 当前 OPD 走 TRL DistillationTrainer/Config（model_and_data.md） | 需迁移成本 |

> **诚实结论：当前 TRL 能"跑"OPD，但不是"高速"OPD**——rollout 是 HF-generate 瓶颈（3–8× 开销），GKD 没有一等公民的 vLLM 集成。要"高速训练 OPD"，**verl 的 async OPD recipe 是专用方案**（vLLM rollout + 异步重叠 + 权重同步优化）。对本项目 Cycle 08 的单 arm / 4B-LoRA / 20k / 640 步规模，TRL 可能可忍但慢；若 OPD 迭代速度成为瓶颈（结合 Cycle 07 显存/rollout 分析：rollout 本就是最重的部分），应评估 verl async OPD 或给 rollout 接 vLLM。
>
> **边界**：框架选择 + 实测 benchmark 归 Code / Experiment Design；本笔记只做工具综述。

---

## 给下游的输出

- **Experiment Design**：(1) teacher 选 Qwen3-8B 同家族（math-RL 变体优先）；(2) cold-start 用 Cycle 07 轨迹 + overlap ratio 选点，注意与 H-OPD 的耦合（别从 recover 起步）；(3) 把 overlap ratio 列为 cold-start 选择依据 artifact；(4) 在 lambda 之外，把 teacher 与 cold-start checkpoint 写进预注册。
- **Code**：评估 TRL（现状）vs verl async OPD（高速）；若用 TRL，确认 rollout 是否能接 vLLM；做一次 rollout 吞吐 probe。
- **Related Work（本对话）**：新增 capacity-gap 定律、TRL/verl OPD 工具来源；更新 Rethinking OPD 条目（补 teacher/cold-start/overlap 实测）。

## Q4：control-anchor / matched-comparison validity（留位激活，用户选 a）

### 概念（用本项目自身教训）

- **control anchor** = 拿 OPD 去比的那个具体 SFT 配置。选哪个 SFT checkpoint/配置当锚点，决定对比意味着什么。
- **matched comparison** = 让 OPD/SFT 在混杂变量上对齐，使唯一差别是方法本身。匹配基准可以是 ID gain / training budget / checkpoint grid / OOD level——**不同基准给出不同甚至相反结论**。
- **validity** = 这个匹配对比是否真支撑因果 claim，还是匹配/锚点方式造成的假象。
- **本项目教训（内部锚点证据）**：Cycle 03/04 用"匹配 GSM8K gain"，匹配出的 SFT 锚点塌缩成近未训练的 sft_n128，对任何真学了的方法都不公平。theory falsifier 原话："若结论只因匹配锚点改变而翻转"则站不住。

### 用户 Q4 追问的方法学结论：为什么单点而非整条趋势

**用户直觉正确，且比单点匹配更对。**
- 单点匹配一个**结果变量**（ID gain）= Cycle 03/04 那类错误（结果轴选锚点 → 退化锚点 + 选择偏差）。
- theory 已把 OOD-drop 重定义为**轨迹属性**（dip-recover）。单 checkpoint 比与轨迹框架**自相矛盾**。
- **正解**：在**输入轴**（training budget / checkpoint grid，Cycle 08 已采用同 grid）匹配 → 比**整条 dip-recover 轨迹**为 **primary**；单点（dip 深度/时点、recover step、终点）作 **landmark / 摘要统计量**，不是独立点匹配；**iso-ID 匹配作二级 cross-section**（等学习量下谁留更多 OOD = 本项目 P1 预测）。
- 文献：The Efficiency Misnomer（2110.12894，单一/单点指标误导，应多维 + best-effort 多 alternative）；compute-matched / iso-FLOP 对比方法学（equal compute from shared checkpoint；削层凑 compute 会人为弱化 baseline 的陷阱）。

### 边界

正式比较设计（哪条作 primary gate、iso-ID 怎么取）归 **Theory / Experiment Design**；Related Work 只提供文献 + 上述方法学输入。Cycle 08 intent 现有的单点 gate（B08 终点非劣、step_20 dip 严重度）应理解为**轨迹上的 landmark**，不是 primary 的点匹配。

---

## control-anchor 留位状态：已激活（Cycle 06 QA Q3 承诺已兑现）

Cycle 06 QA Q3 约定"进入 OPD-vs-SFT 对齐比较时 Related Work 必须提醒是否补搜 control-anchor 文献"。Cycle 08 是该时点，用户 Q4 选 **a（现在补搜）**，已补 The Efficiency Misnomer 等方法学锚点并记入 source_matrix / related_work。留位关闭。
