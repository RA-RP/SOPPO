# QA: Cycle 08 — Related Work Phase

```yaml
phase: related_work
cycle: cycle_08_h_opd_vs_sft_comparison
status: discussion_complete
created: 2026-06-27
discussion_closed: 2026-06-27
internalize_to:
  - related_work/related_work.md
  - related_work/source_matrix.md
  - related_work/incremental_related_work.md
read_scope_confirmed:
  - readme.md
  - next_cycle/current_cycle08_intent.md
  - next_cycle/open_questions.md
  - result/current_picture.md
  - theory/current_theory_update.md (H-OPD framework)
  - literature_library/summaries/OPD相关工作.pdf (本地 OPD survey 笔记)
  - "web: Rethinking OPD 2604.13016 / capacity-gap 2311.07052 / TRL GKD+vLLM / verl async OPD"
write_scope: related_work_only
research_note: source_notes/cycle08_opd_teacher_coldstart_tooling.md
```

Cycle 08 = H-OPD vs SFT 对比（4B-Base + Math-CoT-20k + think-LoRA，单 OPD arm 单 lambda，Cycle 07 SFT 轨迹作参照）。intent 未指定 **teacher 模型**与 **cold-start 步数**——本轮 related work 补这两项的文献依据，并回答 TRL 高速 OPD 可行性。研究结论详见 `source_notes/cycle08_opd_teacher_coldstart_tooling.md`。

---

## 研究结论速览（详见 source note）

- **Q1 teacher**：capacity-gap 定律（最优 teacher 随 student 线性、非越大越好）+ OPD 论文实际配对（Qwen3-1.7B←4B 2.4×、同家族、teacher 常 domain-RL'd）→ 本项目 4B-Base 的甜点是 **Qwen3-8B 同家族**（math-RL 变体优先）；32B 仅作离线数据生成者，不作 live teacher。
- **Q2 cold-start**：标准配方 = off-policy SFT cold-start → OPD；判据 = overlap ratio（72%→91% 成功）/ KL / passRate。**本项目已有 Cycle 07 的 10 点 SFT 轨迹**，可用 overlap ratio 选 cold-start 点，无需盲猜步数。**⚠️ 与 H-OPD 强耦合**：cold-start 太晚（recover）则测不了"OPD 避开 dip"。
- **Q3 TRL**：TRL GKDTrainer 能跑 OPD 但**非高速**（rollout 走 HF generate，3–8× 开销，GKD 无一等 vLLM）；**verl async OPD recipe 是高速专用**（vLLM rollout + 异步重叠 + 权重同步优化）。

---

## Q1：本项目 OPD teacher 选哪个？

- a.（建议）**Qwen3-8B 同家族**（若有 math-RL/GRPO 变体优先，否则 8B-instruct）：~2× student，gap 适中，白盒同 tokenizer，镜像论文 1.7B←4B 配置。
- b. **Qwen3-4B-Instruct（同尺寸）**：便宜的 capacity-matched baseline，延续本项目旧 setup 逻辑；但 teacher 优势弱。
- c. **Qwen3-14B**：更强，capacity-gap 风险与成本上升。
- d. **Qwen3-32B**（数据生成者）：不建议作 live teacher（8× gap + 成本），仅离线数据。

**Codex 初步建议：** a。文献最支持，且与 capacity-gap 定律一致。

**你的回答：**
我认为说的没问题，就使用qwen3-8B的instruct

---

## Q2：cold-start checkpoint 怎么选？

- a.（建议）**用 Cycle 07 SFT 轨迹 + overlap ratio 选点**：测各 checkpoint 对选定 teacher 的 top-k overlap，选越过条件化阈值（~70%+ 上升）但在 recovery 巩固前的最早点（候选 step_20–160），用 overlap 数值证明"足够条件化但未 recover"——既满足 OPD 条件化，又保留 H-OPD 可测性。
- b. 从 **base（step_000）直接 OPD**，无 cold-start：最纯粹测 H-OPD"OPD 是否独立避开 dip"，但 OPD 可能因 overlap 过低条件化差、为错误原因失败。
- c. 从 **recover 点（step_320）cold-start**：OPD 条件最好，但**测不了"OPD 避开 dip"**（B08 子条件 2 空洞）。
- d. 固定步数照搬文献（如 ~3000 步）：不适用——本项目数据/规模远小，且我们已有自己的轨迹。

**Codex 初步建议：** a。这是唯一同时满足"OPD 可条件化"与"H-OPD 可测"的方案；并把 overlap ratio 作为新增 artifact。最终步号由 Experiment Design 测 overlap 后预注册。

**你的回答：**
我认为b是最纯的，如果能直接使用b训练肯定是最好的，我们能够进行小规模的测试知道能不能跳过cold-start吗？如果不能需要进行多少步cold-start呢？有没有相关工作进行过说明？

---

## Q3：训练框架 — TRL 还是 verl？

- a. **先用现有 TRL**：单 arm 小实验，可忍受慢 rollout；省迁移成本。
- b.（建议评估）**迁移/评估 verl async OPD**：若 OPD rollout 速度成瓶颈（Cycle 07 已知 rollout 最重），verl 是高速专用方案。
- c. **TRL + 给 rollout 接 vLLM**：折中，工程量中等，但 GKD 的 vLLM 集成非一等公民，需自验。

**Codex 初步建议：** 这是 Code / Experiment Design 的决定，Related Work 只提供工具综述。建议 Code 先做一次 rollout 吞吐 probe 再定；若只跑一个 arm，a 可接受，但要预期慢。

**你的回答：**
是的，这需要exp和coder阶段决定，但是你作为relatework，需要为此获得足够的相关工作说明，搜索一些verl的相关说明等。

---

## Q4（承诺触发）：control-anchor / matched-comparison 文献现在补搜吗？

Cycle 06 QA Q3 约定：进入 OPD-vs-SFT 对齐比较时，Related Work 必须提醒是否补做 control-anchor / matched-comparison validity 文献搜索。**Cycle 08 正是该时点。**

现状：Cycle 08 用 Cycle 07 SFT 轨迹作 matched control（同 model/data/LoRA/checkpoint grid），matching 已高度受控。

- a. **现在补搜**：为 paper 的"matched comparison validity"提供外部文献支撑（若 paper 想把受控对比作为方法贡献之一）。
- b. **暂不补搜**：当前 matching 已足够受控，等 paper writing 阶段若 reviewer 要求再补。
- c. 只留位，继续推迟。

**Codex 初步建议：** 取决于 paper 定位。若 OPD-vs-SFT 受控对比是核心卖点 → a；若本轮重心是 H-OPD 机制验证、matching 已自洽 → b。请用户定夺。

**你的回答：**
a，和你的对话中，我认为使用ID gain作为匹配不错，同时我想知道为什么不能使用多个点进行对比，为什么使用一个checkpoint。甚至，为什么是看具体的点，不是看整个趋势，就像之前讲的dip-recover是一个趋势。

---

## Q5：本轮 source matrix 更新强度

- a.（建议）中更新：新增 capacity-gap 定律 + TRL/verl OPD 工具来源；更新 Rethinking OPD 条目（补 teacher/cold-start/overlap 实测）；新增"OPD 工程/工具"分类。
- b. 小更新：只更新 Rethinking OPD 条目 + 加 capacity-gap 一篇。

**Codex 初步建议：** a。

**你的回答：**
a

---

## 📌 讨论结论（2026-06-27，已写入正式文件）

1. **Q1 teacher**：Qwen3-8B-Instruct（同家族 ~2×，capacity-gap 定律支持）。32B 仅离线数据生成者。
2. **Q2 cold-start**：用户倾向 b（无 cold-start）。文献结论：纯 from-base 可能但有 tradeoff，判据是 **overlap ratio**（成功 72%→91%）；步数 scale-dependent（~3000 步 plateau 不照搬）。⚠️ 本项目 base(无think)→instruct(有think)+think-format → 初始 overlap 很可能低、纯 b 风险高。**结论：Exp/Code 先做 overlap probe，step_000 行用 b，不行取最小 Cycle 07 checkpoint 作 cold-start**（a/b 混合）。
3. **Q3 框架**：TRL 能跑但非高速（HF-generate rollout，3–8×）；verl async OPD 是高速专用（vLLM + async + top-k KL）。实现注意：推理服务器只返 top-k logprob → 必须 top-k KL。框架决策 + rollout probe 归 Code/Exp Design。
4. **Q4 control-anchor**：a（现在补搜，留位激活）。补 The Efficiency Misnomer 等。用户追问"单点 vs 整条趋势"→ 方法学结论：**轨迹级对比为 primary、单点作 landmark、iso-ID 作二级**；单点匹配结果变量(ID gain)= Cycle 03/04 锚点退化那类错误。正式比较设计归 Theory/Exp Design。
5. **Q5**：a（中更新）。
6. **Q6（用户追加）：能力退化 eval 广度拓展。** Rethink SFT 套件偏难推理，对"能力退化/遗忘"覆盖不足 → 广搜 benchmark（测什么/规模/Qwen3-4B·8B-Base 分数），Relatework 拍板退化轴候选：Tier 1 cheap-MC 全 track（MMLU/HellaSwag/WinoGrande/ARC-C/PIQA/TruthfulQA-MC1）+ Tier 2 生成式抽测（BBH/GSM8K/EvalPlus）。已登记 Qwen3 官方 base 分数作 sanity-check。三条警告：协议不可混用、退化轴须 base 强、commonsense 须先验 4B 判别力。不改 H-OPD primary gate；正式选轴归 Experiment Design。

## 已执行写入（related_work/ 范围内）

1. `related_work.md`：更新 Rethinking OPD 条目（补 teacher 配对/cold-start/overlap/可跳过性实测）；新增"OPD Recipe, Tooling & Comparison Validity (Cycle 08)"章节（capacity-gap 定律 2311.07052、TRL GKD+vLLM-colocate、verl async OPD、The Efficiency Misnomer 2110.12894）。
2. `source_matrix.md`：新增 Cycle 08 文献定位（teacher/cold-start/工具/control-anchor）；矩阵加 capacity-gap + Efficiency Misnomer 行 + 新增"OPD 工程/工具来源"分类；本地证据表加 Cycle 03/04 锚点退化（内部证据）+ Cycle 07 轨迹作 cold-start 池/对照臂。
3. `incremental_related_work.md`：追加 Cycle 08 小节（新增/更新/删除无/下游提醒）。
4. `source_notes/cycle08_opd_teacher_coldstart_tooling.md`：补 cold-start 可跳过性（overlap probe + base→instruct gap）+ Q4 control-anchor 方法学（轨迹 vs 单点）；control-anchor 留位标记关闭。
5. （Q6）`related_work.md` 新增 "Cycle 08 Capability-Degradation Eval Axes" 候选块；`source_matrix.md` Cycle 08 定位加 (D) + 方法学表加 Qwen3 base 表/source-note 行；`incremental_related_work.md` Cycle 08 新增 eval 拓展条目；新建 `source_notes/cycle08_capability_degradation_benchmarks.md`（benchmark 目录）。
6. `-ch` 版（related_work-ch / source_matrix-ch / incremental_related_work-ch）已同步 Cycle 08 全部内容（含 eval 拓展）。

## 给下游对话的提醒（已写入文件）

- **Theory / Experiment Design**：OPD-vs-SFT 比较采轨迹级对比为 primary、单点作 landmark、iso-ID 作二级；teacher=8B-Instruct；cold-start 用 overlap probe 选点。
- **Code / Experiment Design**：实现用 top-k KL（推理服务器约束）；框架 TRL(慢)/verl(快) 先做 rollout 吞吐 probe；cold-start overlap probe 是个便宜的前置小实验。
