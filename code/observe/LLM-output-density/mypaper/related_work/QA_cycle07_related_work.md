# QA: Cycle 07 — Related Work Phase

```yaml
phase: related_work
cycle: cycle_07_base_model_sft_feasibility
status: discussion_complete
created: 2026-06-24
discussion_closed: 2026-06-24
note: "2026-06-24 用户自审 Rethink SFT 实验表格后确认 Q1-Q4；并提出理论深化（OPD vs SFT_dip vs SFT_recover），确认现在就加 on-policy-forgetting 文献簇。表格梳理 + 理论脚手架见 source_notes/rethink_sft_experiment_tables.md。正式 related-work 文件已更新。"
internalized: true
internalize_to:
  - related_work/related_work.md
  - related_work/source_matrix.md
  - related_work/incremental_related_work.md
read_scope_confirmed:
  - readme.md
  - next_cycle/open_questions.md
  - next_cycle/current_cycle07_intent.md
  - result/current_picture.md
  - result/claims_allowed.md
  - theory/current_theory_update.md
  - exp/experimental_methodology/model_and_data.md
  - related_work/related_work.md
  - related_work/source_matrix.md
  - related_work/incremental_related_work.md
  - "Rethink SFT.pdf (primary reference, full read)"
write_scope: related_work_only
hard_dependency: Experiment Design is BLOCKED until this phase closes (per current_cycle07_intent.md)
```

本 QA 文件是 Cycle 07 related-work 阶段开始前的审查清单。按照工作区规则，本阶段只写 `related_work/`，读取其他目录但不修改。用户回答并讨论确认后，再更新正式 related-work 文件。

Cycle 07 是 **base model SFT feasibility cycle**：主实验模型从 Qwen3-1.7B instruct 转向 base（pretraining checkpoint），OPD arm 推迟到 Cycle 08。`current_cycle07_intent.md` 把 Related Work 设为硬依赖，要求本阶段确认五个决策点（模型、训练格式、训练数据、主评测轴、stop sequence）。

---

## 主参考文献已读：Rethink SFT（Ren et al., 2026）

**Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability** — Qihan Ren, Peng Wang, ... Dongrui Liu (Shanghai AI Lab / SJTU / USTC). arXiv 2604.06628, 8 Apr 2026. 代码/模型已开源（github.com/Nebularaid2000/rethink_sft_generalization）。

> 注意：这篇 ≠ source matrix 现有的 "Rethinking On-Policy Distillation"（2604.13016，是 OPD 文献）。两者是不同论文，本轮新增的是 reasoning-SFT generalization 这篇。

### 核心论点
"SFT memorizes, RL generalizes" 被重新审视：reasoning SFT 的 cross-domain generalization **不是不存在，而是有条件的**，由三个因素共同决定——**optimization dynamics、training data、model capability**。

### 与本项目五个决策点直接相关的证据

**(1) 模型能力 — Table 5 / Sec 5 / Fig 12（关键）**
- 论文直接给出 Qwen3-{1.7B,4B,8B,14B}-Base 的 base 分数（Table 5，未训练）：

  | Base model | MATH500 | AIME24 | LCB v2 | GPQA-D | MMLU-Pro | IFEval |
  |---|---|---|---|---|---|---|
  | Qwen3-1.7B-Base | 58.9 | 5.7 | 9.1 | 14.3 | 22.0 | 35.0 |
  | Qwen3-4B-Base | 70.8 | 11.0 | 20.6 | 14.6 | 35.5 | 49.2 |
  | Qwen3-8B-Base | 76.8 | 12.0 | 29.1 | 37.5 | 55.2 | 54.8 |
  | Qwen3-14B-Base | 77.8 | 14.7 | 37.5 | 41.8 | 61.8 | 64.2 |

- **Sec 5 结论：higher-capability models generalize better。** 14B 有明显 dip-and-recovery 并最终广泛提升；8B/4B 也有 recovery 但提升较小；**1.7B 在所有 benchmark 上、即使到 late checkpoint 也只有 marginal 或 negative gain。**
- **Table 6（Qwen3-1.7B-Base + Math-CoT-20k 训练动态）：** MATH500 58.9 → step640 56.2（净负），AIME24 5.7→5.0，LCB 9.1→7.0，GPQA 14.3→9.4，只有 MMLU-Pro 22.0→33.7 上升。
- **对本项目的含义：** 1.7B 训不出泛化**不是 instruct checkpoint 的 artifact，而是能力地板问题**——论文用官方 1.7B-Base 复现了同样的失败。这强力支持"离开 1.7B"。但 4B 在论文里只是"smaller improvements"，稳健的 cross-domain generalization 出现在 8B/14B。**4B 是最小可行、且偏临界。**

**(2) 训练数据 — Sec 4 / Table 2（挑战"保留 NuminaMath"）**
- 论文对比四种数据：Math-CoT-20k（默认，long-CoT）、Math-NoCoT-20k（去掉 think 块）、**NuminaMath-20k（NuminaMath-1.5 人写短解，质量参差）**、Countdown-CoT-20k。
- **结论：data quality matters。NuminaMath-20k 这种 low-quality 短解在 Table 2 中导致 broad OOD degradation 且 in-domain math 几乎无 gain，是表现最差的一类；而 verified long-CoT 给出一致的 cross-domain gain。** 论文还指出 NuminaMath 在 Sec 3 的 dip-and-recovery 中几乎不 recovery。
- 默认配方 **Math-CoT-20k = OpenR1-Math-220k 的 query + Qwen3-32B（thinking enabled）生成 long-CoT response + math-verify 过滤只留正确解，max 16384 tokens。**
- **对本项目的含义：** 本项目正在用的 NuminaMath-1.5 短解，恰好是论文点名"会广泛损害泛化"的低质量数据。**"保留 NuminaMath" 这一假设受到直接挑战。**

**(3) 训练/评测格式 — Sec 4 / App B.2-B.3（挑战"no-think concise"方向）**
- 论文主线：**long-CoT（含 `<think>...</think>`、含 backtracking/verification 等 procedural pattern）才是迁移的来源**；procedural pattern 而非 domain content 驱动 generalization。
- **训练即使对 base 模型也套 Qwen chat template**（App B.2 Table 3 prompt：`<|im_start|>user {Question} Please reason step by step, and put your final answer within \boxed{}. <|im_end|> <|im_start|>assistant {Response} <|endoftext|>`）。即把 base 模型训练进 chat+think 格式。
- 评测：zero-shot、temp 0.6、max 32768 tokens、只用 user prompt 无 system prompt（与训练一致）、统一 `\boxed{}` final-answer 模板、math benchmark 用 math-verify、评测前 **remove everything before last `</think>`**。
- **对本项目的含义：** 我们 Cycle 06 走的是 *no-think concise SFT*，并发现它"destroys the function of thinking"。这与论文完全吻合——论文说 no-think 数据虽不那么伤 safety，但 long-CoT 才是泛化来源。**本项目的格式方向可能需要从 no-think 翻转到 long-CoT think-format。**

**(4) 主评测轴 — App B.3**
- ID reasoning：**MATH500（avg@3）+ AIME24（avg@10）**。
- OOD reasoning：LCB v2（代码，avg@3）、GPQA-Diamond（科学，avg@3）、MMLU-Pro（pass@1）。
- General：IFEval、AlpacaEval 2.0、HaluEval、TruthfulQA。Safety：HEx-PHI（ASR）。
- **对本项目的含义：** 论文不把 NuminaMath-test 作 ID 轴，而用 MATH500+AIME24。本项目当前 ID 轴是 NuminaMath-test。是否切换/补充是一个决策点。

**(5) stop sequence — 重新框定 A06b 问题**
- 论文没有用 few-shot plain-text base 评测，因此**不存在我们 Cycle 06 那种 runaway `Problem:` 续写**问题；它把 base 训练进 chat/think 格式后 zero-shot 评测，靠学到的 EOS/`</think>` 收尾，用 max-token 截断而非 few-shot stop 序列。
- **对本项目的含义：** 如果 Cycle 07 跟随论文走 chat+think+zero-shot 路线，则 Cycle 06 的 base stop-fix 问题（few-shot plain-text）大部分被绕开；但**预实验 Task 1（给现有 base runner 加 stop seq 重测以关闭 A06b）仍应做**，因为它是对"旧 plain-text base 协议"的清算，且便宜。

**(6) optimization dynamics — Sec 3（方法学警告）**
- **dip-and-recovery：** cross-domain 性能先降后升，short-epoch checkpoint 会系统性低估 SFT 的泛化潜力；under-optimization 比 over-optimization 更常见。默认训 8 epochs。
- **repeated exposure > one-pass：** 同等 compute 下，小数据多 epoch 比大数据一遍更有效（concurrent Kopiczko et al. 2026 一致）。
- **对本项目的含义：** 判 A07/A06b "fail" 之前必须确认训练足够（多 epoch、看完整 trajectory），否则可能把 under-optimization 误读成 SFT 不泛化。这是给 Experiment Design 的协议输入。

### 论文里可顺带新增的二级来源（出现在其 reference / 与本项目相关）
- OpenR1-Math-220k（Lozhkov et al. 2025）— 候选训练 query 源。
- Huan et al. 2025（"Does math reasoning improve general LLM capabilities?"）— 本项目"math SFT 是否迁移"的最直接先验，论文复现并修正了它。
- Chu et al. 2025（"SFT memorizes, RL generalizes"）— 论文要 reframe 的源命题。
- Kopiczko et al. 2026（data repetition beats data scaling in long-CoT SFT）— 支持多 epoch。
- math-verify（HF 2024）— 已在本项目 MATH500 scorer 中使用，可正式登记为 scorer 来源。

---

## 待讨论问题

### Q1：模型规模 — 4B 够吗，还是要 8B / 保留 1.7B-base 作对照？

**背景：** intent 选了 Qwen3-4B-Base。但 Rethink SFT 直接显示：1.7B（含 base）训不出泛化是能力地板问题；4B 只有"smaller improvements"；稳健 cross-domain generalization 在 8B/14B 才出现。4B 偏临界。

- a.（建议）**Qwen3-4B-Base 作主模型 + 保留 Qwen3-1.7B-Base 作"已知失败"负对照**（论文 Table 6 给了可对照的 1.7B-Base 复现基准）。4B 是论文支持的最小可行点，1.7B 复现能验证我们 pipeline 与论文一致。
- b. 直接上 **Qwen3-8B-Base** 作主模型，确保落在"稳健泛化"区，代价是训练/显存成本上升（论文用 8×H200）。
- c. 只用 4B，不做 1.7B 对照，省成本。
- d. 4B + 8B 双主模型，做 capability sweep（最贵，最接近论文 Sec 5 设计）。

**Codex 初步建议：** a。先用 4B 站稳、用 1.7B-Base 复现论文负结果当 sanity；若 A07 在 4B 仍只有 marginal gain，再按论文证据升到 8B（routing 已写在 intent）。

**你的回答：** c。只用 Qwen3-4B-Base 作主模型，本轮不做 1.7B-base 对照。（Codex 注：论文 Table 5/6 的 1.7B-Base 基准仍记录在 source matrix，供未来需要时直接对照，不必本轮重训。若 A07 在 4B 只有 marginal gain，按 intent routing 升 8B。）

---

### Q2：训练数据 — 保留 NuminaMath 短解，还是换成 verified long-CoT？

**背景：** intent 暂定"保留 NuminaMath"。但 Rethink SFT 点名 NuminaMath-1.5 短解是"会广泛损害泛化"的低质量数据，其默认配方是 OpenR1-Math-220k query + 强 teacher（Qwen3-32B）long-CoT + math-verify 过滤。

- a.（建议）**换成 long-CoT 配方**（OpenR1/类似 query + 强 teacher long-CoT + math-verify 过滤），并**保留一个 NuminaMath 短解 arm 作对照**，直接在本项目复现"低质量数据 vs long-CoT"的数据质量对比。
- b. 完全保留 NuminaMath，本轮只换模型不换数据（与 intent 一致，但与论文证据相悖，A07 很可能复现"无 gain"）。
- c. 换成 long-CoT，但不保留 NuminaMath 对照（省一条 arm）。
- d. 保留 NuminaMath query，但把 target 从短解换成强 teacher 生成的 long-CoT（query 不变、解法升级，最接近论文 Math-CoT vs NuminaMath 的受控对比）。

**Codex 初步建议：** a 或 d。这是本轮最关键决策——若不动数据，A07 很可能只是再次确认"无 gain"，但归因会含混（是模型还是数据？）。论文已经把数据因素单独拎出来，本项目应避免把数据缺陷误读成 SFT/能力缺陷。

**你的回答：** 换成 long-CoT，不保留 NuminaMath 对照（接近选项 c）。本轮训练数据切到 verified long-CoT 配方（OpenR1 类 query + 强 teacher long-CoT + math-verify 过滤），不再用 NuminaMath-1.5 短解、也不单设 NuminaMath 对照 arm。（Codex 注：论文 Table 2 已提供 NuminaMath vs long-CoT 的外部对照证据，本项目本轮不复现该对照，直接采用论文判定的高质量配方。具体 query 源/样本量/split 由 Experiment Design 预注册。）

---

### Q3：训练/评测格式 — 翻转到 long-CoT think-format + chat template？

**背景：** 本项目 Cycle 06 用 no-think concise SFT 并发现它破坏 thinking 功能；Rethink SFT 表明 long-CoT（含 `<think>`、procedural pattern）才是泛化来源，且对 base 模型也套 chat template 训练、zero-shot 评测、评测前删 `</think>` 之前的内容。

- a.（建议）**跟随论文：base 模型用 Qwen chat template + long-CoT think-format 训练，zero-shot（temp 0.6）评测，统一 `\boxed{}` 模板，math-verify 评分，评测前删 last `</think>` 之前内容。**
- b. 保留 no-think concise 格式，只换模型（与 Cycle 06 一致，但论文证据指向这会再次失败）。
- c. 同时做 think vs no-think 两个格式 arm，把格式作为受控变量（更贵，但能在本项目内复现论文 Math-CoT vs Math-NoCoT 对比）。

**Codex 初步建议：** a 为主线；若预算允许，c 的 no-think 对照很有价值（直接对应论文 Table 2 的 CoT vs NoCoT，且能复用我们 Cycle 06 的 no-think 发现）。

**你的回答：** a。翻转到 long-CoT think-format（base 套 chat template + long-CoT 训练，zero-shot 评测，删 last `</think>` 之前内容）。本轮不单设 no-think 对照 arm。

---

### Q4：主评测轴 — 是否从 NuminaMath-test 切到 MATH500+AIME24，并采纳论文 OOD 套件？

**背景：** 论文 ID 轴是 MATH500（avg@3）+AIME24（avg@10），OOD reasoning 是 LCB v2 / GPQA-D / MMLU-Pro。本项目当前 ID 轴是 NuminaMath-test，OOD-lite 是 MMLU/TruthfulQA/WinoGrande/ARC（已被判定为 <1.5% 非判别轴）。

- a.（建议）**ID 轴改用 MATH500+AIME24（与论文和社区对齐，便于 baseline sanity-check），保留 NuminaMath-test 作 in-distribution 训练域指标；OOD reasoning 采纳 LCB v2 + GPQA-D + MMLU-Pro（比现有 OOD-lite 更有判别力）。**
- b. 维持 NuminaMath-test 作主 ID 轴，OOD 维持现有 OOD-lite，不引入论文套件（连续性最好，但 OOD-lite 已知无判别力，且不便与论文对照 baseline）。
- c. 折中：ID 加上 MATH500+AIME24，OOD 暂不扩到 LCB/GPQA（代码/科学评测成本与依赖较高），下一轮再加。

**Codex 初步建议：** a 或 c。至少要把 MATH500（已在本项目跑通）+AIME24 纳入，因为论文 Table 5 给了可直接核对的 base 分数，能立刻发现训练/测试格式有没有重蹈 MATH500 覆辙。LCB/GPQA 是否本轮就上取决于 Experiment Design 的工程预算。

**你的回答：** a。ID 轴改用 MATH500（avg@3）+AIME24（avg@10），NuminaMath-test 降为训练域指标；OOD reasoning 采纳论文套件 LCB v2 / GPQA-Diamond / MMLU-Pro。（具体 avg@k、temp、token budget、scorer 由 Experiment Design 预注册。）

---

### Q5：本轮 source matrix 更新强度 + control-anchor 留位提醒

**背景：** Cycle 06 约定：重新进入 OPD-vs-SFT aligned comparison 时，Related Work 必须提醒是否补做 control-anchor / matched-comparison 文献搜索。Cycle 07 仍是 SFT-only（OPD 推迟到 Cycle 08），所以**本轮只提醒、不搜索**。

- 更新强度：
  - a.（建议）中更新：新增 Rethink SFT 为本轮**核心来源**，顺带登记 2-4 篇二级来源（OpenR1-Math-220k、Huan et al. 2025、Chu et al. 2025、Kopiczko et al. 2026、math-verify 视相关性取舍），重写 source matrix 的 Cycle 07 定位为"base-model SFT feasibility, capability/data/optimization-conditional"，把 NuminaMath 标注为"论文点名的低质量数据，待本项目对照验证"。
  - b. 小更新：只加 Rethink SFT 一篇 + 改定位文字。
  - c. 大更新：系统检索 base-model SFT、data-quality、capability-scaling 三方向。

- control-anchor 留位：本轮**确认仅提醒不搜索**，等 Cycle 08 进入 OPD-vs-SFT 对齐比较时再触发。✅/✏️？

**Codex 初步建议：** a + 确认 control-anchor 本轮只提醒不搜索。

**你的回答：** a（中更新）+ control-anchor 本轮只提醒不搜索，确认。Rethink SFT 登记为核心来源 + on-policy-forgetting 文献簇（见 Q7 增补）已加入。

---

### Q6：写作禁止项确认

请确认以下边界：
1. 不把 Rethink SFT 的 4B/8B/14B 结论当作"本项目已验证"——它是外部证据，本项目尚未在自己的 pipeline 上复现。
2. 不把"换 long-CoT 数据/格式"写成已决定的实验设计——Related Work 只给出文献依据和建议，决策与预注册归 Experiment Design。
3. 不把新论文与现有 "Rethinking On-Policy Distillation"（2604.13016）混为一谈。
4. 不删除现有 OPD 文献簇——OPD 仍是 Cycle 08 的比较背景，只是本轮不扩展。
5. 不在本轮搜索 control-anchor / matched-comparison 文献（仅留位提醒）。

- a. 全部正确。
- b. 需要修改其中某几条。

**Codex 初步建议：** a。

**你的回答：** a。全部正确。

---

## Q7（增补，2026-06-24）：理论深化 + on-policy-forgetting 文献现在加

用户基于 Rethink SFT 提出**理论深化**：把原"OPD vs SFT 的 OOD-drop/继续学习/几何指标"二元对照，升级为 **OPD vs SFT_dip vs SFT_recover** 三方对照；核心是把 OOD-drop 从 method 属性重定义为 **checkpoint 属性**（dip-and-recovery）。

**讨论结论：**
1. 方向认可，文献撑得住。正式理论改写归 **Theory 对话**写 `theory/`；Related Work 只提供文献脚手架（见 `source_notes/rethink_sft_experiment_tables.md` §7）。
2. 三方框架的 **OPD 腿需要"on-policy/online 忘得少"文献簇** —— **用户确认现在就加**（不按 OPD 推迟）：Chen et al. 2025、Shenfeld et al. 2026、Tajwar et al. 2024，登记为 Cycle 08 OPD 腿备料。
3. **分层**：SFT_dip vs SFT_recover + 几何指标 + response-length 关系 **Cycle 07 即可测**（无需 OPD），但要求沿轨迹**多 checkpoint 采样**；OPD 腿留 Cycle 08。
4. **新增分析指标建议**：response length（Rethink SFT Sec 3.2 验证的 dip↔recover 诊断量）——提醒 Experiment Design 列为必收 artifact。

---

## 📌 讨论结论（2026-06-24，已写入正式文件）

1. **Q1 模型**：只用 Qwen3-4B-Base，不做 1.7B-base 对照（表格自审后确认；4B Table 9 实测全面正向）。
2. **Q2 数据**：换 verified long-CoT，不留 NuminaMath 对照。
3. **Q3 格式**：翻转到 long-CoT think-format（base 套 chat template，zero-shot 评测，删 last `</think>` 前内容）。
4. **Q4 评测**：ID 改 MATH500+AIME24，OOD 采纳 LCB v2/GPQA-D/MMLU-Pro；NuminaMath-test 降为训练域指标。具体预注册归 Experiment Design。
5. **Q5 更新强度**：中更新；control-anchor 留位仅提醒不搜索（Cycle 08 与 on-policy 文献簇一并激活）。
6. **Q6 禁止项**：全部生效。
7. **Q7 理论深化**：三方框架（OPD vs SFT_dip vs SFT_recover）+ on-policy-forgetting 文献现在加；正式理论改写交 Theory 对话；Cycle 07 需多 checkpoint 采样几何 + response length。

## 已执行写入（related_work/ 范围内）

1. `related_work.md`：新增 Rethink SFT（核心条目）+ on-policy-forgetting 文献簇（Chen/Shenfeld/Tajwar）+ Cycle 07 候选评测轴来源（MATH500/AIME24/LCB/GPQA/MMLU-Pro，待 Exp Design 采纳）。OPD 文献不删不降级。
2. `source_matrix.md`：新增 Cycle 07 文献定位（generalization-conditional + 理论深化）；核心论文矩阵 + 本地证据表更新；control-anchor 留位提醒续存。
3. `incremental_related_work.md`：追加 Cycle 07 小节（新增/更新/删除无/留位提醒）。
4. `source_notes/rethink_sft_experiment_tables.md`：App D 全表抽取 + 4B 专题 + 三轴×ID/OOD + 理论深化脚手架 §7。

## 通过 QA 后的预期写入（已执行，保留原计划记录）

1. `related_work/related_work.md`：新增 Rethink SFT（Ren et al. 2026）核心条目 + 经确认的二级来源条目；对现有 OPD 条目不删除、不降级（OPD 仍是 Cycle 08 背景）。
2. `related_work/source_matrix.md`：重写文献定位为 Cycle 07 "base-model SFT feasibility（capability/data/optimization-conditional）"；新增"reasoning-SFT generalization 的条件性"分类；把 NuminaMath 在本地证据表中标为"论文点名低质量数据，待本项目对照"；记录 control-anchor 留位提醒（Cycle 08 触发）。
3. `related_work/incremental_related_work.md`：追加 Cycle 07 小节，列新增来源与理由；删除：无。
4.（视 Q4 决定）若采纳论文 OOD 套件/评测协议，按项目规则把对应协议来源追加到 source matrix 的 Eval Protocol Sources 区块，供 Experiment Design 引用。
