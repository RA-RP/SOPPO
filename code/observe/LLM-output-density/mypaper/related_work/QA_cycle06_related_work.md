# QA: Cycle 06 — Related Work Phase

```yaml
phase: related_work
cycle: cycle_06_sft_feasibility_and_degradation
status: discussion_complete
created: 2026-06-18
discussion_closed: 2026-06-18
internalize_to:
  - related_work/related_work.md
  - related_work/source_matrix.md
  - related_work/incremental_related_work.md
read_scope_confirmed:
  - readme.md
  - next_cycle/open_questions.md
  - next_cycle/current_cycle06_intent.md
  - result/current_picture.md
  - result/claims_allowed.md
  - theory/current_theory_update.md
  - related_work/related_work.md
  - related_work/source_matrix.md
  - related_work/incremental_related_work.md
write_scope: related_work_only
```

本 QA 文件是 Cycle 06 related-work 阶段开始前的审查清单。按照工作区规则，本阶段只写 `related_work/`，读取其他目录但不修改。用户回答并讨论确认后，再更新正式 related-work 文件。

---

## 已读上下文摘要

Cycle 06 的当前定位已经从“继续 OPD-vs-SFT 对比”收窄为 **SFT-only feasibility and degradation cycle**：

- 不训练新 OPD arms。
- 先确认 SFT-from-base 是否能在干净协议下产生可解释的 ID/downstream gain。
- 再测 SFT 数据规模增加时是否出现 ID/OOD degradation。
- OPD、matched control、geometry 暂时作为后续设计的背景，不作为本轮主比较。

因此，Cycle 06 related work 的主要任务不是扩写 OPD 背景，而是补强三类文献支撑：

1. SFT 是否可能损害 reasoning / induce forgetting。
2. 为什么必须先建立 base-aware headroom 和 SFT feasibility。
3. 为什么 train/eval format 与 scorer protocol 必须作为方法学来源记录。

---

## Q1：Cycle 06 related-work 的主轴是否切到 SFT feasibility？

**问题：** 本轮 related work 是否应明确把主轴从“OPD 是否比 SFT 稳定”改成“先验证 SFT-from-base 是否可学习、是否随数据规模退化”？

- a. 是。Cycle 06 只服务 SFT baseline、degradation curve、headroom 建设；OPD 文献保留为背景，不新增 OPD 主张。
- b. 否。仍应继续围绕 OPD-vs-SFT stability comparison 扩文献。
- c. 折中。SFT-only 是实验主轴，但 related work 同时预备下一轮 OPD forgetting-prevention 论述。

**Codex 初步建议：** a，兼顾 c 的轻量铺垫。正式文件中应把 OPD 章节降为“later comparison background”，新增或加强 SFT degradation / headroom / protocol-validity 章节。

**你的回答：**
你理解的到位，我认为是a。现在不是推翻之前的工作，而是对最开始的内容进行补充，可能更偏工程类。

---

## Q2：是否需要新增 SFT degradation / catastrophic forgetting 文献簇？

**问题：** 现有 source matrix 只有一篇 “Fine-tuning on CoT reasoning” 支撑 SFT 可能退化。Cycle 06 要不要新增一个更明确的文献簇，覆盖 continued SFT、catastrophic forgetting、task specialization、small-model reasoning degradation？

- a. 需要。Cycle 06 的核心问题就是 SFT scale curve，必须至少补 2-4 个相关来源。
- b. 暂不需要。先用现有 CoT fine-tuning 文献，等实验确认退化后再扩。
- c. 只补一两个最直接来源，避免 related work 过早膨胀。

**Codex 初步建议：** a 或 c。最小可行更新是新增 “SFT degradation / forgetting baseline” 小节，并把 `sft_n1024` 的本地发现定位为待解释现象，而不是把它直接当成已证明机制。

**你的回答：**
你回答的依旧很好，我认为是a

---

## Q3：是否需要新增 base/headroom/control-anchor 方法论文述？

**问题：** 当前理论已经转向 base-aware stability：没有 base headroom，就不能判断 OPD 或 SFT 的稳定性优势。related work 是否需要新增“baseline construction / control anchor / matched comparison validity”这一类方法学文献或概念来源？

- a. 需要。Cycle 04-05 的主要教训就是 control-anchor failure，应在 source matrix 中有独立方法学位置。
- b. 暂不需要。这个教训来自本项目实验，不必用外部文献支撑。
- c. 只在 source matrix 的本地实验证据区扩写，不新增外部来源。

**Codex 初步建议：** c 作为本轮保守做法；如果后续 paper 以“matched controls warning”为贡献之一，再执行外部文献搜索。

**你的回答：**
我认为都不是，这个工作现在不需要但是之后必定会需要，为了不影响现在的工作，你预先给这个related work留位，现在先不进行搜索，但是后续进行OPD和SFT对齐时，你需要提醒我要不要进行这个related work的搜素。

---

## Q4：训练与评测协议来源是否由 Related Work 先补？

**问题：** Cycle 06 Experiment Design 必须决定 SFT training format、chat/think settings、token budget、extractor/scorer。Related Work 是否应在本阶段提前补充协议来源，例如 Qwen3 官方文档、chat template / thinking mode 文档、lm-eval task protocol 来源？

- a. 是。Related Work 先补足协议来源，Experiment Design 直接引用。
- b. 否。协议来源应由 Experiment Design 在采纳具体 protocol 时补入 source matrix。
- c. Related Work 只列出待查来源清单，不把未采纳来源写入正式 source matrix。

**Codex 初步建议：** c。因为具体 protocol 还没由 Experiment Design 预注册，本阶段可以写一个“Cycle06 protocol sources to verify”清单，正式采纳来源等 Experiment Design 决定后再追加到 Eval Protocol Sources。

**你的回答：**
a 我认为需要尽可能的了解现在官方、别人是怎么做训练、测试的，尽可能了解它们训练、测试的格式和数据量，以及知道baseline、相近模型的测试结果大致是多少，这样我才能知道我测试、训练有没有出现显著的问题（例如之前math500出现的测试格式问题）。而且我现在就能在exp设计之前告诉你我大致要训练的模型，测试方案：1.模型现在尝试现有的qwen3-1.7B以及qwen3-1.7B-base，我猜测base模型变化会更显著一些，同时base在测试上比较简单，没有think和chat。2.qwen3-1.7B需要去查找对应训练时使用什么格式，测试时官方、别人使用的是什么格式，通常的得分是多少。3.qwen3-1.7B-base也需要搜索官方别人是怎么训练的，测试是什么格式，通常的得分是多少。

---

## Q5：geometry 文献在 Cycle 06 中如何处理？

**问题：** Cycle 06 将 geometry 作为 diagnostic，不作为主 gate 或机制 claim。related work 是否需要继续扩展 geometry/representation 文献？

- a. 暂不扩展。保留现有 LoRA-vs-full-finetuning 与 OPRD 作为机制背景。
- b. 需要补 representation drift / subspace overlap / forgetting geometry 来源，为保存 UV/sketch artifact 提供文献动机。
- c. 只在 source matrix 中标注 geometry 本轮降级为 diagnostic，未来 OPD cycle 再扩。

**Codex 初步建议：** c。当前最重要的是防止 geometry 被写成机制证据；source matrix 应明确“diagnostic only”。

**你的回答：**
你说的没问题，就是c

---

## Q6：本轮 source matrix 的更新形式

**问题：** 本轮正式更新应采用哪种强度？

- a. 小更新：只改综述定位和本地实验证据，不新增外部来源。
- b. 中更新：新增 2-4 篇 SFT degradation / forgetting 相关来源，并同步 `related_work.md`、`source_matrix.md`、`incremental_related_work.md`。
- c. 大更新：系统检索 SFT degradation、control construction、eval protocol、geometry 四个方向，重组 source matrix。

**Codex 初步建议：** b。Cycle 06 已经把研究问题切到 SFT feasibility/degradation；没有新增来源会显得文献基础滞后，但大更新会超过当前阶段的必要范围。

**你的回答：**
b 但是我在Q4说的要检查的这些工程问题是必须要讨论的，同时不计入这里说的2-4篇

---

## Q7：哪些内容不能写进本轮 related work？

**问题：** 请确认以下禁止项是否正确：

1. 不把 Cycle 06 写成 OPD-vs-SFT comparison。
2. 不声称 OPD 已经防止 forgetting。
3. 不把 OOD-lite 当成 OPD/SFT 的 discriminating axis。
4. 不把 `sft_n1024` 直接写成“已证明的过拟合机制”，只写成“consistent with / motivates testing”。
5. 不引用未核实或未采纳的协议来源作为正式依据。

- a. 全部正确。
- b. 需要修改其中某几条。

**Codex 初步建议：** a。

**你的回答：**
a

---

## 📌 讨论结论（2026-06-18）

✅ 已确认本阶段可以进入正式 related-work 更新。结论如下：

1. **Cycle 06 related work 主轴改为 SFT feasibility / degradation first。** 这不是推翻 OPD 主线，而是先补齐工程和 baseline ladder：在干净协议下确认 SFT-from-base 是否能学、是否随数据规模退化。
2. **新增 SFT degradation / catastrophic forgetting 文献簇。** 本轮新增 2-4 篇相关来源，不把 `sft_n1024` 写成已证明机制，只把它写成 motivating observation / consistency signal。
3. **Base/headroom/control-anchor 方法论文述先留位。** 本轮不搜索、不扩展该方向；但在后续重新进入 OPD-vs-SFT aligned comparison 时，Related Work 必须提醒用户是否补做 baseline construction / control anchor / matched comparison validity 文献搜索。
4. **训练与评测协议来源现在就补。** 在 Experiment Design 前，Related Work 需要尽量查清 Qwen3-1.7B 与 Qwen3-1.7B-Base 的官方/社区训练格式、测试格式、常见 baseline 分数和协议陷阱；这些工程来源不计入 Q6 的 2-4 篇 SFT degradation 文献。
5. **Geometry 本轮降级为 diagnostic-only。** 只在 source matrix 中标注，不新增 representation / subspace 文献。
6. **写作禁止项确认。** 不把 Cycle 06 写成 OPD-vs-SFT comparison；不声称 OPD 已防止 forgetting；不把 OOD-lite 当判别轴；不把 `sft_n1024` 直接写成已证明过拟合机制；未核实或未采纳协议来源不得写成正式依据。

## 通过 QA 后的预期写入

若以上问题按默认建议通过，本阶段将执行：

1. 更新 `related_work/source_matrix.md`：
   - 将 Cycle 06 定位改为 SFT feasibility / degradation first。
   - 新增或加强 “SFT degradation / forgetting baseline” 分类。
   - 将 OOD-lite 明确降级为 capability-preservation check。
   - 将 geometry 标注为 diagnostic-only。
2. 更新 `related_work/related_work.md`：
   - 只新增经过核实的外部来源条目。
   - 对现有 OPD 条目的 relevance 作轻量降级说明，不删除。
3. 更新 `related_work/incremental_related_work.md`：
   - 在 Cycle 06 小节追加新增来源与理由。
   - 若不删除来源，则“删除：无”。
