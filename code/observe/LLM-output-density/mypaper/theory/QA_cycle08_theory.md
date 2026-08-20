# QA: Cycle 08 — Theory Phase

```yaml
phase: theory
cycle: cycle_08_h_opd_vs_sft_comparison
status: discussion_complete
created: 2026-06-29
discussion_closed: 2026-06-29
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
  - next_cycle/current_cycle08_intent.md
  - theory/README.md
  - theory/current_theory_update.md
  - theory/geometry_metric_definitions.md
  - related_work/QA_cycle08_related_work.md
  - related_work/source_matrix.md
  - related_work/source_notes/cycle08_opd_teacher_coldstart_tooling.md
  - related_work/source_notes/cycle08_capability_degradation_benchmarks.md
write_scope: theory_only
```

本 QA 文件是 Cycle 08 theory 阶段开始前的审查清单。按照工作区规则，本阶段只写 `theory/`，读取其他目录但不修改。用户回答并讨论确认后，再更新正式 theory 文件。

---

## 角色与工程边界

当前对话角色：**Theory**。

允许写入：

- `mypaper/theory/`

不允许写入：

- `related_work/`
- `exp/`
- `code/`
- `result/`
- `next_cycle/`
- `research_cycles/`
- `local_experiment_results/`
- `paper_drafts/`

Theory 阶段职责：

- 更新机制假设、变量定义、指标解释和 claim 边界。
- 把 Cycle 08 的 OPD-vs-SFT 轨迹比较写成可证伪理论框架。
- 为 Experiment Design 提供理论层面的比较单位、gate 解释和禁止项。

Theory 阶段不做：

- 不预注册具体运行命令、路径、batch size、框架实现。
- 不改训练/评测协议稳定层。
- 不写 `exp/current_experiment_design.md` 或 `code/current_code_brief.md`。
- 不把尚未执行的 Cycle 08 OPD arm 写成结果。

---

## 当前 cycle 判定

`exp/current_experiment_design.md` 仍是 Cycle 07 设计，因为 Cycle 08 Experiment Design 尚未开始；但 `next_cycle/current_cycle08_intent.md`、`next_cycle/open_questions.md`、`related_work/QA_cycle08_related_work.md` 已确认当前研究进入：

```text
cycle_08_h_opd_vs_sft_comparison
```

因此本 Theory 阶段按 Cycle 08 处理。

---

## 已读上下文摘要

Cycle 07 已经把 Cycle 06/07 theory 的关键外部假设变成内部结果：

- Qwen3-4B-Base + Math-CoT-20k + think-format LoRA 的 SFT 可学习性成立：A07 PASS。
- SFT 轨迹有真实 dip-and-recovery：MATH500 step_20 dip，step_160 起 recover，step_320 peak。
- step_20 dip 是 genuine capability dip，不是 output-control artifact；后续 step_40–160 的 non-termination explosion 是第二个时间错位的 output-control transient。
- Geometry 上 step_20 出现 triple co-location：argmin acc = argmax effective_rank = argmin xs_log_spectrum_gap。
- OverlapLift < 1 且单调上升，但 LoRA r=32 下随机 baseline 不足以给 on/off-principal verdict；解释延后到 Cycle 08。

Cycle 08 的任务是训练 OPD arm，在同 model/data/LoRA/checkpoint grid 下，与 Cycle 07 SFT 轨迹比较：

- B08：OPD 是否匹配 SFT final MATH500，并避免或减轻 step_20-level capability dip。
- C08：OPD 是否比 SFT 更好地保存 MMLU-Pro。
- D08：OPD 的 geometry transient 是否不同于 SFT step_20 triple co-location。
- A08：OPD 的 OverlapLift trajectory 是否系统性不同；但 on/off-principal verdict 需要 LoRA-aware reference。

Related Work 已给 Theory/Exp 的新增输入：

- teacher 采用 Qwen3-8B-Instruct，同家族、约 2x capacity gap。
- cold-start 应先做 overlap probe；纯 from-base 是最纯 H-OPD 测试，但 base→instruct+think-format 初始 overlap 可能低。
- comparison validity：trajectory-level comparison 为 primary；single checkpoint 只作 landmark；iso-ID 作为二级 cross-section。
- 能力退化 eval 拓展属于补充 preservation evidence，不改变 B08/C08 primary gate。

---

## Q1：Cycle 08 是否把 H-OPD 从“预注册假设”升级为“当前待检验理论框架”？

**问题：** Cycle 07 已内部验证 SFT dip-and-recovery 与 step_20 geometry triple co-location。Cycle 08 现在训练 OPD arm。Theory 是否应把 `current_theory_update.md` 新增 Cycle 08 小节，明确：

```text
H-OPD is no longer only a future placeholder. It is the active Cycle 08 test:
OPD should either flatten / shift / avoid the SFT dip transient while matching
the SFT recovered endpoint, or H-OPD is weakened.
```

- a. 是。新增 Cycle 08 theory 小节，把 H-OPD 写成当前 active test。
- b. 否。仍只保留 Cycle 07 的预注册 H-OPD，不新增 Cycle 08 theory。
- c. 折中。只写一个短 bridge，等 Experiment Design 后再正式更新。

**Codex 初步建议：** a。Cycle 08 正是 H-OPD 的第一次直接测试，theory 需要在实验前写清可证伪读法。

**你的回答：**
a

---

## Q2：OPD-vs-SFT 的 primary comparison unit 是整条轨迹还是单个 checkpoint？

**问题：** Related Work Q4 已得出方法学输入：单点匹配结果变量会重演 Cycle 03/04 的 control-anchor failure；而 Cycle 07 theory 已把 OOD-drop 定义为 trajectory / checkpoint-stage 属性。Theory 是否应正式规定：

```text
Primary comparison = matched checkpoint-grid trajectory.
Landmarks = dip depth/timing, recovery step, endpoint.
Secondary cross-section = iso-ID or iso-capability comparisons.
```

- a. 是。轨迹级对比为 primary，single checkpoint 只作 landmark，iso-ID 作二级。
- b. 否。B08/C08 单点 gate 就是 primary comparison。
- c. 折中。B08/C08 仍是 gate，但理论解释必须写成 trajectory-first。

**Codex 初步建议：** c 或 a。最干净的理论是 a；为了兼容 intent 中已有 B08/C08 gate，可写成：gate 是 trajectory landmarks 的判定接口，不是单点匹配设计。

**你的回答：**
我认为是a

---

## Q3：cold-start 如何影响 H-OPD 的可解释性？

**问题：** 用户倾向从 base 直接 OPD，因为这是最纯的 H-OPD 测试；Related Work 同时指出 base→8B-instruct+think-format 的初始 overlap 可能太低。Theory 是否应定义一个 cold-start purity hierarchy：

```text
Level 0: no cold-start / step_000 OPD
  Strictest H-OPD test. If it works, strongest evidence. If it fails under low
  overlap, failure may be conditioning failure rather than H-OPD falsification.

Level 1: minimal pre-recovery cold-start selected by overlap probe
  Conditional H-OPD test. Still valid if cold-start is before SFT recovery and
  chosen only to enter the teacher-overlap region.

Level 2: recover-point cold-start
  Not a valid test of "avoid dip"; only tests continued OPD from an already
  recovered SFT state.
```

- a. 写入。Theory 明确区分 strict H-OPD、conditional H-OPD、non-test。
- b. 不写入。Cold-start 是 Experiment Design / Code 的实现问题。
- c. 只写一个 warning，不建立 hierarchy。

**Codex 初步建议：** a。这是理论可解释性的核心，否则不同 cold-start 会对应不同 claim。

**你的回答：**
我认为是a，同时我不认为level2是需要的，这个即算不上消融，也不是常见的训练方法。

---

## Q4：Theory 是否应给 lambda 选择提供默认原则？

**问题：** Cycle 08 intent 要求单个 OPD arm、单个 lambda，并要求 Experiment Design 从 prior cycles 或 theory layer 解释选择。Theory 是否应给出如下原则：

```text
If the goal is the cleanest H-OPD mechanism test, choose the strongest OPD
contrast that is still expected to preserve capability. With only one arm:
- lambda=1.0 gives the clearest on-policy contrast but higher risk of instability;
- lambda=0.5 is more conservative and may better satisfy B08 non-inferiority;
- final choice should be justified as mechanism-purity vs capability-risk tradeoff.
```

- a. Theory 推荐 `lambda=1.0`，因为单 arm 应最大化 OPD contrast。
- b. Theory 推荐 `lambda=0.5`，因为 B08 是 blocking gate，先保 non-inferiority。
- c. Theory 不指定数值，只给出 purity/risk decision rule，Experiment Design 选择。

**Codex 初步建议：** c。Theory 可给原则和 tradeoff，但具体数值最好由 Experiment Design 结合实现/资源风险决定。若你希望 theory 更强势，也可以选 a 或 b。

**你的回答：**
我认为暂时显示c，真实的等到cycle08实验运行结束再考虑是不是写a。最理想的情况肯定是a能够满足我们的要求，这样能最好的说明OPD的性质。

---

## Q5：Gate B08 / C08 / D08 / A08 的理论解释边界如何写？

**问题：** 是否接受下面的 gate interpretation：

```text
B08 pass:
  OPD is capability-compatible with SFT and does not reproduce the same severe
  early SFT dip. H-OPD remains viable.
B08 fail:
  OPD is not yet a viable replacement under this lambda/teacher/cold-start.
  Do not use C/D/A positives to claim H-OPD if B08 fails badly.

C08 pass:
  OPD has preservation evidence on MMLU-Pro relative to SFT.
C08 fail:
  OPD may match math capability but does not show general-knowledge preservation.

D08:
  Geometry is mechanism-supporting only if it aligns with B/C behavior.
  Geometry cannot rescue a capability failure.

A08:
  OverlapLift is diagnostic until LoRA-aware reference is solved; trajectory
  differences can be reported, but on/off-principal verdict requires normalization.
```

- a. 接受，写入 theory。
- b. 接受但需要弱化 B08 fail 的说法。
- c. 不接受，需要重写 gate interpretation。

**Codex 初步建议：** a。尤其是“geometry cannot rescue capability failure”应明确，避免机制指标压过行为结果。

**你的回答：**
a，我认可。同时我想MMLU-pro未必能够代表OOD。我在relatework中查询了一些benchmark，我认为可以将其综合起来看。但是具体的测试是什么，没有测试过我还不确定，我现在希望你在theory中先暂时使用MMLU-pro但是需要标注出之后可能会在coder环节进行具体的判断。

---

## Q6：OverlapLift / principal-angle 的 Cycle 08 理论边界是什么？

**问题：** Cycle 07 的 OverlapLift 已测，但 on/off-principal 被延后；Cycle 08 intent 要求保存 UV/sketch 并解决 LoRA-aware reference。Theory 是否应同步更新 `geometry_metric_definitions.md`：

1. OverlapLift trajectory can be compared OPD vs SFT as a descriptive trajectory metric.
2. On/off-principal verdict is forbidden until a LoRA-aware reference is defined.
3. Random-LoRA baseline may normalize update count but not necessarily low-rank structure.
4. Principal-angle / subspace-rotation claims require saved UV/sketch artifacts and explicit top-r definition.
5. If LoRA-aware reference remains unresolved, A08 stays diagnostic.

- a. 全部写入。
- b. 只写 1/2/5，不写 principal-angle。
- c. 暂不改 geometry 文件，只改 current theory。

**Codex 初步建议：** a。Cycle 08 的一个核心 pending item 就是 OverlapLift / subspace verdict，geometry 文件必须自包含。

**你的回答：**
我不认为这些条目都是对的。1没有问题；2我认为是能够比较的，OPD也是使用LoRA的，两者在这方面上消融了，能够进行比较。3这个肯定的，所以说我之前说要等到cycle08将OPD和SFT进行对比才能知道啊；4现在先暂时不做这个，请你注册到cycle09中，我希望在那里在考虑这个问题；5不是这样的，只需要OPD和SFT进行比较我认为就是可以的，它们之间消融了LoRA。

---

## Q7：能力退化 eval 拓展在 theory 中如何定位？

**问题：** Related Work 新增了 Tier 1/Tier 2 capability-degradation benchmarks（MMLU、HellaSwag、WinoGrande、ARC-C、PIQA、TruthfulQA、BBH、GSM8K、EvalPlus 等）。Theory 是否应把它们定位为：

```text
Supplementary preservation axes, not primary H-OPD gates.
They can strengthen or weaken the "OPD preserves non-target capability" story
only after B08/C08 are interpreted.
They must be separated from target math axes and from protocol-mismatched official scores.
```

- a. 是。写成 supplementary preservation axes，不改 B08/C08 primary status。
- b. 把它们提升为 Cycle 08 theory 的核心 gate 候选。
- c. 不写入 theory，完全留给 Experiment Design。

**Codex 初步建议：** a。它们对 paper 很有用，但不能让 Cycle 08 gate 失焦。

**你的回答：**
a

---

## Q8：Cycle 08 后，什么证据足够支持 paper-level claim？

**问题：** Theory 是否应预先写清证据层级：

```text
Strong claim:
  B08 pass + C08 pass + D08/A08 mechanism signals aligned.

Narrow claim:
  B08 pass, C08 fail or mixed; OPD matches math capability but preservation
  advantage unsupported.

Negative / redesign:
  B08 fail; lambda/teacher/cold-start/training setup must be investigated before
  any OPD stability claim.
```

- a. 写入。Cycle 08 结果解释必须有预先层级。
- b. 暂不写，等结果出来后 Result Analysis 定。
- c. 只写禁止项，不写 claim ladder。

**Codex 初步建议：** a。预先写清 claim ladder 能防止结果出来后临时换主张。

**你的回答：**
a

---

## Q9：哪些 claim 必须禁止？

**问题：** 请确认 Cycle 08 theory 的禁止项是否正确：

1. 不从 Cycle 07 单独声称 OPD 或 H-OPD 成立；Cycle 07 没有 OPD arm。
2. 不把 Cycle 08 的 OPD 单 arm 失败直接解释成“OPD 理论失败”；需先区分 lambda、teacher、cold-start、overlap、实现问题。
3. 不把 recover-point cold-start 的结果解释成“OPD 避开 dip”。
4. 不把 single checkpoint comparison 写成 primary causal comparison；轨迹是 primary。
5. 不在没有 LoRA-aware reference 时给 OverlapLift on/off-principal verdict。
6. 不让 geometry / OverlapLift positive rescue a B08 capability failure。
7. 不把 official Qwen3 benchmark scores 和本项目协议分数混用为同一量。
8. 不把 non-termination 仅写成 eval artifact；Cycle 07 已确认它是真实 output-control behavior，同时 as-run acc 是有效 end-to-end task-success measure。
9. 不把 supplementary degradation benchmarks 改写成事后主 gate。

- a. 全部正确。
- b. 需要修改其中某几条。

**Codex 初步建议：** a。

**你的回答：**
a。但是，其中的3我认为不需要，没有必要从recover-point开始；5我不认可，我之前就说了OPD和SFT之间是能比较的，它们消融了LoRA；

---

## Q10：正式写入强度

**问题：** QA 通过后，本阶段正式更新采用哪种强度？

- a. 小更新：只在 `current_theory_update.md` / `-ch.md` 追加 Cycle 08 小节，不改 geometry 文件。
- b. 中更新：更新 `current_theory_update.md` / `-ch.md`，并同步更新 `geometry_metric_definitions.md` / `-ch.md` 的 Cycle 08 OverlapLift / principal-angle / LoRA-aware reference 边界。
- c. 大更新：重构 theory 目录，拆出独立 `opd_trajectory_theory.md`。

**Codex 初步建议：** b。当前目录仍够用，但 Cycle 08 的 geometry/OverlapLift 边界必须进入 metric definitions。

**你的回答：**
b

---

## 通过 QA 后的预期写入

若以上问题讨论通过，本阶段将执行：

1. 更新 `theory/current_theory_update.md`：
   - 新增 Cycle 08 active H-OPD test 小节。
   - 把 trajectory-level comparison 写成 primary theoretical comparison unit。
   - 写入 Level 0/Level 1 cold-start 可解释性，并排除 recover-point cold-start 作为有效 H-OPD 设计。
   - 写入 B08/C08/D08/A08 的理论解释边界。
   - 写入 Cycle 08 claim ladder 和禁止项。
2. 更新 `theory/current_theory_update-ch.md`：
   - 同步中文阅读版，不制造中英文内容分叉。
3. 更新 `theory/geometry_metric_definitions.md`：
   - 新增 Cycle 08 OverlapLift matched-LoRA 相对比较边界。
   - 明确 principal-angle / measured subspace-rotation 注册到 Cycle 09。
   - 明确 geometry cannot rescue capability failure。
4. 更新 `theory/geometry_metric_definitions-ch.md`：
   - 同步中文阅读版。



---

## 讨论结论（2026-06-29）

1. Cycle 08 将 H-OPD 从 Cycle 07 的预注册假设升级为当前 active test。
2. OPD-vs-SFT 的 primary comparison unit 是 matched checkpoint-grid trajectory；single checkpoints 只是 landmarks，iso-ID / iso-capability 是二级 cross-section。
3. Cold-start 只保留两个可解释层级：Level 0 no cold-start / step_000 OPD；Level 1 minimal pre-recovery cold-start selected by overlap probe。Recover-point cold-start 不作为有效 H-OPD 设计。
4. Theory 暂不指定 lambda 数值，只给 purity/risk decision rule；若 `lambda=1.0` 能通过 B08，它是最理想、最能说明 OPD 性质的设置。
5. B08 是 capability-compatible + dip-avoidance gate；C08 目前使用 MMLU-Pro，但 MMLU-Pro 不代表 OOD 全部，后续 benchmark suite 可由 Experiment Design / Code 根据 Related Work 和协议检查选择或扩展。
6. D08 geometry 与 A08 OverlapLift 只能支持与行为结果一致的机制解释，不能拯救 B08 capability failure。
7. OPD-vs-SFT OverlapLift trajectory 在 matched-LoRA 条件下可以直接比较；不需要先解决 LoRA-aware random reference。限制只保留给绝对 on/off-principal verdict 或相对 universal/random/full-rank reference 的强解释。
8. Random-LoRA baseline 可能归一化 update count / adapter footprint，但不一定归一化真实训练目标诱导的 low-rank structure。
9. Principal-angle / measured subspace-rotation 暂不在 Cycle 08 做，注册到 Cycle 09 再考虑。
10. Supplementary capability-degradation benchmarks 是 preservation axes，不是事后 primary H-OPD gates。
11. Non-termination 不写成纯 eval artifact；Cycle 07 已确认它是真实 output-control behavior，同时 as-run accuracy 是有效 end-to-end task-success measure。

## 已执行写入（2026-06-29）

- `current_theory_update.md` / `current_theory_update-ch.md`：新增 Cycle 08 active H-OPD test、trajectory-first comparison、cold-start 可解释性、gate interpretation、claim ladder 与禁止项。
- `geometry_metric_definitions.md` / `geometry_metric_definitions-ch.md`：新增 Cycle 08 matched-LoRA OverlapLift 解释边界，修正 principal angle 从 Cycle 08 延后到 Cycle 09。
- `README.md` / `README-ch.md`：同步 theory 目录入口说明。
