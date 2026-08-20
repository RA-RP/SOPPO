# QA: Cycle 07 — Theory Phase

```yaml
phase: theory
cycle: cycle_07_base_model_sft_feasibility
status: discussion_complete
created: 2026-06-24
discussion_closed: 2026-06-24
internalize_to:
  - theory/current_theory_update.md
  - theory/current_theory_update-ch.md
  - theory/geometry_metric_definitions.md
  - theory/geometry_metric_definitions-ch.md
read_scope_confirmed:
  - readme.md
  - result/current_picture.md
  - result/claims_allowed.md
  - next_cycle/open_questions.md
  - next_cycle/current_cycle07_intent.md
  - related_work/QA_cycle07_related_work.md
  - related_work/source_notes/rethink_sft_experiment_tables.md   # 理论脚手架 §7
  - theory/current_theory_update.md
  - theory/geometry_metric_definitions.md
write_scope: theory_only
hard_dependency_note: Related Work 已完成（discussion_complete 2026-06-24）；本阶段是 Q7 理论深化的正式落地。
```

本 QA 文件是 Cycle 07 theory 阶段开始前的审查清单。按工作区规则，本阶段只写 `theory/`，读取其他目录但不修改。用户回答并讨论确认后，再更新正式 theory 文件。

---

## 角色与工程边界

当前对话角色：**Theory**。

允许写入：`mypaper/theory/`

不允许写入：`result/`、`related_work/`、`exp/`、`next_cycle/`、`research_cycles/`、`local_experiment_results/`、`code_handoffs/`、`code_handins/`

Theory 阶段的职责：更新机制假设、变量定义、指标解释和 claim 边界。可为 Experiment Design 提供概念框架和解释规则，但不预注册具体数字阈值、不改 eval/training protocol、不写代码 handoff。

---

## 本轮定位：与 Cycle 06 截然不同——这是项目第一个正向机制理论

Cycle 06 theory 是**最小贡献**：能力退化是工程常识，theory 只更新 claim 边界，不建新变量。

Cycle 07 theory 是**实质深化**。Related Work 阶段（QA Q7，用户已确认方向）要求把项目核心命题从

> 二元对照："OPD vs SFT，谁的 OOD-drop 小、继续学习好、几何指标如何。"

升级为

> 轨迹框架："**OOD-drop 是优化轨迹上的暂态，由 checkpoint 所处优化阶段决定**（dip-and-recovery）。SFT 不是一个点，而是 SFT_dip → SFT_recover 的轨迹。OPD 可能是不经 dip、直达 recover 的捷径。"

这是项目从"base-aware 防御性边界"（前几个 cycle 都在说'什么时候不能比较'）转向**第一个正向、可证伪的机制假设**。Theory 这次要建变量、立假设、重构几何指标读法。

正式理论改写由本对话（Theory）负责；Related Work 只提供了文献脚手架，见 `related_work/source_notes/rethink_sft_experiment_tables.md §7`。

---

## 核心背景摘要

### Rethink SFT（Ren et al. 2026）的关键实测（外部证据）

Related Work 引入的主参考文献，直接为三方框架提供机制锚点。

**(1) dip-and-recovery 是实测现象（Qwen3-4B-Base + Math-CoT-20k, Table 9）：**

| Step | MATH500 | GPQA-D | response length 行为 | 阶段 |
|---|---|---|---|---|
| Base | 70.8 | 14.6 | — | 起点 |
| 20 | 62.9（谷底，<base） | 17.3 | 暴涨(~28–30k) | **SFT_dip** |
| 320 | 86.3 | 33.3 | 收缩 | 过渡 |
| 640 | 86.2 | 37.9（反超 base +23.3） | ~7–23k | **SFT_recover** |

cross-domain 性能**先降后升**。短 epoch checkpoint 会系统性**低估** SFT 泛化潜力。

**(2) 旧负结果的外部解释——1.7B + NuminaMath 是"双坑"：**
- 论文用**官方 Qwen3-1.7B-Base** 复现了同样的失败（Table 6：跑完 640 步 MATH500 净负 58.9→56.2）→ **1.7B 训不出泛化是能力地板，与 instruct/no-think 无关**。
- 论文点名 **NuminaMath-1.5 短解是最差的一档数据**（Table 2：把 OOD reasoning 打到地板，14B LCB 37.5→20.4）。
- **本项目 Cycle 04-06 同时踩了"1.7B 弱模型"+"NuminaMath 烂数据"两个坑。**

**(3) 4B 实测是"会泛化"的一侧（§6c）：** Math-CoT 下 4B 在 ID(MATH500 +15.4, AIME +19.7) 和 OOD(GPQA +23.3, MMLU-Pro +29.3) 都有大幅稳健正向；但有可见 dip（MATH500 step20→62.9），LCB(代码)gain 最小(+3.8)。

### Cycle 07 实验决策（Related Work 已确认，Experiment Design 待预注册）

- 模型：**Qwen3-4B-Base**（pretraining checkpoint），本轮不做 1.7B-base 对照。
- 数据：**verified long-CoT**（OpenR1 类 query + 强 teacher long-CoT + math-verify 过滤），弃用 NuminaMath 短解。
- 格式：**翻转到 long-CoT think-format**（base 套 chat template，zero-shot 评测，删 last `</think>` 之前内容）。
- 评测：ID 改 **MATH500+AIME24**，OOD 采纳 **LCB v2 / GPQA-D / MMLU-Pro**；NuminaMath-test 降为训练域指标。
- OPD：**推迟到 Cycle 08**。Cycle 07 是 SFT-only。

### 当前结果边界（claims_allowed.md，必须遵守）

- Cycle 06 instruct no-think SFT：A06 FAIL（confirmatory，无 gain + math 退化）。
- Cycle 06 base series：A06b UNDETERMINED（截断 artifact，待 stop-fix 重测）。
- Cycle 06 geometry：C06 PARTIAL（CSV 未 copyback，未分析）。
- 旧 1.7B 8-model 结果（Cycle 04/05）仍有效，但与 Cycle 06/07 不同 checkpoint，不可跨 series 比较。

### Theory 分层（source_notes §7.4）

| 层 | 内容 | 何时 |
|---|---|---|
| **Cycle 07** | SFT_dip vs SFT_recover：沿 4B-base SFT 轨迹的 OOD-drop / 继续学习 / 几何 + response length 关系 | 本轮（无需 OPD），需**多 checkpoint 采样** |
| **Cycle 08** | 加 OPD 腿：验证 H-OPD（OPD = recover-without-dip） | 需 OPD arm |

---

## Q1：三方框架与现有 K1-base-aware 的结构关系？

**背景：** `current_theory_update.md` 现有两层——Theory 05（K1-base-aware：有 headroom 才能比较）+ Cycle 06（SFT feasibility 前置条件）。Cycle 07 的三方轨迹框架与它们是什么关系？

- a. **替换**：三方框架成为新主框架，K1-base-aware 降为历史。
- b. **并列**：三方框架作为独立的 Cycle 07 大节，与 K1-base-aware 平行。
- c.（建议）**递进嵌套**：保留 K1-base-aware 作为"何时能比较"的**前提层**，三方框架作为"比较时如何理解 OOD-drop"的**机制层**。即：K1（headroom 前提）→ dip-recovery（OOD-drop 是 checkpoint 属性）→ H-OPD（OPD = recover-without-dip 捷径）。

**Theory 判断与推荐：c。** 理由：K1-base-aware 的核心断言（没有 base headroom 就无法判断稳定性）现在被 Rethink SFT **从外部强化**了——1.7B 能力地板正是"无 headroom"的实例，论文用官方 1.7B-Base 复现。所以 K1 不该废，它是三方框架的入场券（4B + long-CoT 正是为了越过 K1 门槛）。三方框架是在"已越过 headroom 门槛"之后才有意义的机制细化。三者构成清晰的递进逻辑链，不是互相取代。

**你的回答：**
c，是的我认为c是一个很好的角度，结合了前人认为的OOD-drop和rethink sft的发现

---

## Q2：核心 claim——OOD-drop 从 method 属性重定义为 checkpoint 属性？

**背景：** 这是 Cycle 07 theory 最重要的单点（source_notes §7.1）。前几个 cycle 隐含把 "OOD-drop / 稳定性" 当作**方法**的固有属性（"SFT 会掉 OOD，OPD 更稳"）。Rethink SFT 的 dip-and-recovery 表明：同一个 SFT，早 checkpoint 掉 OOD（dip）、晚 checkpoint 反超（recover）——**OOD-drop 是 checkpoint / 优化阶段的属性，不是方法的固有标签。**

- a.（建议）**接受并写为核心机制 claim**：OOD-drop 是优化轨迹上的暂态，由 checkpoint 所处优化阶段（dip/recover）决定；"SFT 必然 OOD-drop" 是把 checkpoint 属性误读为 method 属性。
- b. 接受方向，但弱化：只写成"OOD-drop 可能受 checkpoint 阶段影响"，不作为中心命题。
- c. 不接受，维持 method 二元对照。

**Theory 判断与推荐：a。** 理由：这是整个 Cycle 07 深化的支点。如果不把 OOD-drop 重定义为 checkpoint 属性，三方框架、H-OPD、几何暂态读法全都失去地基。这个重定义也直接改写了项目对 Cycle 04-06 负结果的解读（见 Q6）——之前"SFT 掉 OOD"的观察很可能是 dip checkpoint + under-optimization，而非方法终态。**Caveat（必须写入边界）：** 这一重定义目前由外部论文（4B/8B/14B）支撑，本项目尚未在自己 pipeline 复现 dip-and-recovery；Cycle 07 多 checkpoint 采样正是为了内部验证。

**你的回答：**
我也倾向于a，但是这还需要更多的测试进行验证，暂时先认为是a

---

## Q3：中心假设 H-OPD 现在就写入吗（Cycle 07 不训 OPD）？

**背景：** source_notes §7.3 的中心假设 **H-OPD：OPD ≈ "不经过 dip 暂态、直达 recover 状态"的捷径**（on-policy 减少 exposure bias，少绕路）。但 Cycle 07 不训 OPD（推迟 Cycle 08）。on-policy-forgetting 文献簇（Chen 2025 / Shenfeld 2026 / Tajwar 2024）已在 Related Work 登记为 OPD 腿备料。

- a.（建议）**现在就写入 H-OPD 作为预注册待检验假设**，明确标注"Cycle 08 检验，Cycle 07 不训 OPD"，并写出可证伪的对照预测（OPD 的 OOD-drop / response-length surge / 几何暂态峰显著小于 SFT_dip，而终点 ≈ SFT_recover）。
- b. 只写 SFT_dip / SFT_recover 二态机制，H-OPD 留到 Cycle 08 theory 再写。
- c. 写入 H-OPD 但只作为开放问题，不写预注册预测。

**Theory 判断与推荐：a。** 理由：预注册待检验假设正是 theory 的本职——在数据出来前写清"如果 H-OPD 成立会看到什么"，恰恰防止 Cycle 08 结果出来后临时编故事（这正是项目反复强调的护栏）。on-policy 文献簇已备好，H-OPD 有文献支撑（Shenfeld 2026 online 忘得少、贴近 base 分布 → 直接预测 OPD 的 drift-from-base 更平）。严格标注"Cycle 07 不产生 OPD 证据、H-OPD 是 Cycle 08 检验对象"即可避免越界。

**你的回答：**
a，是的你说的没错。


---

## Q4：新变量定义——optimization_stage / response_length / 几何轨迹结构？

**背景：** source_notes §7.5 要求 Theory 新增变量。Cycle 06 我们明确**不建变量表**（工程问题）；Cycle 07 是真机制，需要变量。候选：

| 候选变量 | 含义 | 角色 |
|---|---|---|
| `optimization_stage` | checkpoint 所处阶段（dip / 过渡 / recover） | 把 OOD-drop 归因到的核心自变量 |
| `response_length` | 生成长度（dip 暴涨 ~28–30k，recover 收缩 ~7–23k） | dip↔recover 的**可观测代理**（Rethink SFT Sec 3.2 验证） |
| `ood_drop` | OOD reasoning 相对 base 的暂态下降 | 从 method 属性改为 checkpoint 属性的因变量 |
| `id_consolidation` | ID（训练域）是否巩固/反超 base | 区分 shallow imitation vs internalized |
| 几何指标（effective rank / drift-from-base） | 改为"沿轨迹暂态结构"读法（见 Q5） | 机制层证据 |

- a.（建议）**全部写入，建立 Cycle 07 轨迹变量表**，明确 `response_length` 是必收的 dip↔recover 诊断代理（提醒 Experiment Design 列为 required artifact）。
- b. 只写 `optimization_stage` 和 `response_length`，几何变量留给 Q5 在 geometry 文件处理。
- c. 只写文字描述，不建变量表。

**Theory 判断与推荐：a（变量表）+ 几何细则在 Q5 落到 geometry 文件。** 理由：三方框架要可操作，必须把"优化阶段"和"response length 代理"显式建模。response length 尤其关键——它是 Rethink SFT 已验证的 dip/recover 单变量诊断量，本项目可低成本沿轨迹采集。变量表写在 `current_theory_update.md`，几何指标的"暂态结构"操作定义放 `geometry_metric_definitions.md`（Q5）。

**你的回答：**
你的说法我十分同意，按照你说的

---

## Q5：几何指标角色升级——从静态终点对照到沿轨迹暂态结构？

**背景：** 现有 geometry 读法（Cycle 04/05）是**静态终点对照**："OPD 降 effective rank，SFT 升 rank"。三方框架要求升级为**沿轨迹暂态结构**（source_notes §7.3）：effective rank / drift-from-base 是否在 SFT_dip 出现**暂态峰**、在 recover **回落**？（Cycle 08：OPD 是否**压平**该峰？）

是否更新 `geometry_metric_definitions.md`，补充三点：

1. **轨迹读法 > 终点读法**：geometry 指标应沿 SFT 轨迹多 checkpoint 解读，单看终点 checkpoint 会丢失 dip 暂态结构。
2. **effective rank 暂态峰假设**：SFT_dip 阶段可能伴随 effective rank / drift 暂态升高（shallow imitation、表征发散），recover 阶段回落；这是待 Cycle 07 多 checkpoint 数据验证的假设。
3. **drift reference 已为 base**（Cycle 06 已改）：4B-base 训练的 drift 从 base checkpoint 量，并要求沿轨迹采样以暴露暂态峰。

- a.（建议）三点全写入 `geometry_metric_definitions.md` + `-ch.md` 的新 Cycle 07 小节。
- b. 只写"轨迹读法 > 终点读法"（第 1 点），暂态峰假设留作开放问题。
- c. 暂不改 geometry 文件，只在 current_theory_update.md 加注。

**Theory 判断与推荐：a。** 理由：Cycle 06 我们已把 drift reference 改为 base 并写了"lower effective rank 不自动等于稳定"。Cycle 07 顺势把这条升级为完整的轨迹读法——这正是 open_questions「lower effective rank 是稳定还是有害 collapse」的机制化回答：它取决于所处 dip/recover 阶段。同时这是给 Experiment Design 的硬要求（多 checkpoint 几何采样），写进 geometry 文件最自然。

**你的回答：**
是的，同时你不要忘记分析gap等其它指标，而且我认为现在取top1-gap实际上是不太好的，这个gap用于判断主向量张成空间的稳定性，应该是多个向量组成的张成空间，而不是top1，你有什么想法吗？在我的印象里我还有一个判断指标gamma。

---

## Q6：旧负结果重新定性——从"SFT 在此设置失败"到"弱模型 + 烂数据的联合失败"？

**背景：** 这关系到项目整体叙事（next_cycle paper positioning open #3 正在问"Cycle 03-06 instruct 结果如何在 paper 中定性"）。Rethink SFT 现在提供了外部解释：Cycle 04-06 的 1.7B + NuminaMath 负结果，很可能是**两个独立缺陷的叠加**——
- 模型坑：1.7B 是能力地板（论文官方 1.7B-Base 复现同样失败，Table 6）。
- 数据坑：NuminaMath-1.5 短解是论文点名"会广泛损害 OOD"的最差数据（Table 2）。

Theory 是否把旧负结果从"SFT 在此设置不行 / base-ceiling"重新定性为"弱模型 + 烂数据双坑，而非 SFT 方法本身的终态失败"？

- a.（建议）**写入重新定性，但严格标注证据来源**：现在有**外部证据**表明旧负结果很可能是 capability-floor + data-quality 双坑（非方法失败）；Cycle 07（4B + long-CoT）将在本项目 pipeline 内验证这一重定性。
- b. 写入更强版本：直接判定旧负结果是双坑导致（不强调"待本项目验证"）。
- c. 不重新定性，维持旧 base-ceiling 叙事，只在 result 层记录新证据。

**Theory 判断与推荐：a。** 理由：这个重新定性对 paper 叙事极重要——它把项目前半段从"一串看不出名堂的负结果"转成"逐步定位混淆因素（saturation → headroom → 能力地板 + 数据质量）的方法学链条"，与 K1-base-aware 的演进完全自洽。但必须用 a 而非 b：Rethink SFT 是外部论文（4B/8B/14B），本项目尚未自己复现 4B+long-CoT 的正向结果；在 Cycle 07 A07 出结果前，重定性只能是"外部证据支持的强假设 + 待内部验证"，不能写成已确认。这也守住了 claim 边界规则"不把外部结论当本项目已验证"。

**你的回答：**
a

---

## Q7：Cycle 07 theory 的 claim 边界与禁止项

**背景：** 综合三方框架、H-OPD、模型/数据迁移，请确认禁止项。

1. 不把 OOD-drop 写成 SFT 方法的固有属性——它是 checkpoint / 优化阶段属性（除非未来证据推翻）。
2. 不把 Rethink SFT 的 4B/8B/14B 结论当作"本项目已验证"——它是外部证据，本项目尚未在自己 pipeline 复现。
3. 不把 H-OPD 写成结论或已观测——Cycle 07 不训 OPD，H-OPD 是 Cycle 08 的预注册待检验假设。
4. 不在看完整 trajectory 前判 SFT "fail"——dip-and-recovery + under-optimization 护栏：短 epoch checkpoint 会系统性低估泛化（Rethink SFT Sec 3）。
5. 不把旧 1.7B + NuminaMath 负结果写成"SFT 方法本身失败"——现在归因弱模型 + 烂数据双坑（待 Cycle 07 内部验证）。
6. `response_length` / effective-rank 暂态峰是**诊断代理 / 待验证假设**，不是机制证明；需多 checkpoint 采样。
7. 不把 dip-and-recovery 写成本项目已复现的现象——它来自外部论文，Cycle 07 才首次在本项目内测试。

- a. 全部正确，写入 claim 边界。
- b. 需要修改某几条（请说明）。

**Theory 判断与推荐：a。**

**你的回答：**
a

---

## Q8：正式写入方式与文件组织

**背景：** Cycle 07 theory 内容明显比 Cycle 06 重（建变量、立假设、重构几何读法）。`theory/` 当前是 `current_theory_update.md` + `geometry_metric_definitions.md` 两个主文件（各含 `-ch.md`）。

- a.（建议）**中-大更新，保持两文件结构**：在 `current_theory_update.md` 新增 Cycle 07 大节（三方框架 + H-OPD + 轨迹变量表 + 重定性 + claim 边界），并更新 `geometry_metric_definitions.md` 的轨迹暂态读法。四个文件（含 `-ch.md`）同步。
- b. **新建专门文件**：把三方轨迹框架单独拆成 `theory/sft_trajectory_theory.md`（+`-ch.md`），`current_theory_update.md` 指向它。
- c. **小更新**：只在 current_theory_update.md 追加简短 Cycle 07 子节，几何不动。

**Theory 判断与推荐：a。** 理由：readme 维护规则倾向"增量维护、不轻易重构目录"。三方框架虽是重要深化，但它与 K1-base-aware 是递进关系（Q1=c），写在同一文件里能呈现完整逻辑链（headroom 前提 → 轨迹机制 → H-OPD），比拆文件更利于阅读。geometry 更新是 Q5 的自然延伸。若未来 Cycle 08 OPD 腿落地后内容继续膨胀，再考虑拆文件（届时 trajectory 框架已成熟，值得独立）。

**你的回答：**
a

---

## 通过 QA 后的预期写入（草案，待 Q1–Q8 确认）

假设按推荐通过（Q1=c, Q2=a, Q3=a, Q4=a, Q5=a, Q6=a, Q7=a, Q8=a）：

1. **更新 `theory/current_theory_update.md`**：新增 "Cycle 07 Theory Update: From Base-Aware Boundary to a Trajectory Mechanism (OPD vs SFT_dip vs SFT_recover)" 大节，含：
   - 三方框架与 K1-base-aware 的递进关系（Q1）。
   - 核心 claim：OOD-drop 是 checkpoint 属性（Q2）。
   - SFT_dip vs SFT_recover 特征对照（引 Rethink SFT 4B 实测）。
   - H-OPD 预注册待检验假设 + 可证伪预测（Q3，标注 Cycle 08 检验）。
   - Cycle 07 轨迹变量表（`optimization_stage` / `response_length` / `ood_drop` / `id_consolidation`）（Q4）。
   - 旧负结果重新定性（弱模型 + 烂数据双坑，待内部验证）（Q6）。
   - Cycle 07 claim 边界与 7 条禁止项（Q7）。
2. **更新 `theory/current_theory_update-ch.md`**：中文版同步。
3. **更新 `theory/geometry_metric_definitions.md`**：新增 Cycle 07 小节——沿轨迹暂态结构读法、effective rank 暂态峰假设、drift-from-base 多 checkpoint 采样要求（Q5）。
4. **更新 `theory/geometry_metric_definitions-ch.md`**：中文版同步。

---

## 📌 讨论结论与已执行写入（2026-06-24）

✅ 全部 8 题确认：**Q1=c, Q2=a, Q3=a, Q4=a, Q5=a（+技术深化）, Q6=a, Q7=a, Q8=a**。

### Q5 技术深化（用户在 Q5 提出，已讨论闭环）

1. **gamma 定位**：代码中无 gamma 判别指标 —— `gamma_s.jsonl` 是 OPD 的 S-probe 存储文件，无关。用户确认记忆中的 gamma = **Davis–Kahan eigengap γ_r = σ_r − σ_{r+1}**（top-1 gap 推广到 top-r 子空间）。
2. **子空间稳定性指标**：top-1 gap 升级为 eigengap γ_r；新增 `stable_rank`、`energy_ratio_at_r`（三者用现有谱即可算，无需 UV）。
3. **principal angle**（实测子空间旋转）需保存 UV —— **用户决定本轮不开，留 Cycle 08 统一讨论**。Cycle 07 子空间稳定性只用 γ_r 容量代理，不做实测旋转 claim。
4. **gap 等指标纳入轨迹读法**：`spectral_gap`（→γ_r）和 `X_S_spectrum_level_gap` 一并沿轨迹解读，不只 effective rank / drift。

### 已执行写入（theory/ 范围，4 文件）

- `current_theory_update.md` / `-ch.md`：新增 Cycle 07 大节（三层递进链、OOD-drop 重定义为 checkpoint 属性、三方框架 SFT_dip vs SFT_recover、H-OPD 预注册假设、轨迹变量表、旧负结果重新定性、7 条禁止项）。
- `geometry_metric_definitions.md` / `-ch.md`：新增 Cycle 07 节（轨迹读法、eigengap γ_r + stable rank + energy-ratio、principal angle 留 Cycle 08、暂态峰假设、提醒 Experiment Design）。

### 提醒 Experiment Design（不在 theory 写入范围，须在 exp/ 落地）

1. 实现 `spectral_gap(sigma, r)`（k=r 自适应）、`stable_rank`、`energy_ratio_at_r`。
2. 沿轨迹**多 checkpoint** 采样几何指标 **+ `response_length`**（必收 artifact）。
3. drift 从 **base**（Qwen3-4B-Base）量，不从 `theta0`。
4. 本轮 **UV 保存关闭**（principal angle 留 Cycle 08）。
