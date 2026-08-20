# QA: Cycle 06 — Theory Phase

```yaml
phase: theory
cycle: cycle_06_sft_feasibility_and_degradation
status: discussion_complete
created: 2026-06-18
last_updated: 2026-06-23
questions_answered: Q1, Q2, Q3_inferred, Q4, Q5, Q6, Q7, Q8
questions_pending: none
internalize_to:
  - theory/current_theory_update.md
  - theory/current_theory_update-ch.md
  - theory/geometry_metric_definitions.md      # 仅在 Q6 确认后
  - theory/geometry_metric_definitions-ch.md   # 仅在 Q6 确认后
read_scope_confirmed:
  - readme.md
  - result/current_picture.md
  - result/claims_allowed.md
  - next_cycle/open_questions.md
  - next_cycle/current_cycle06_intent.md
  - related_work/QA_cycle06_related_work.md
  - related_work/related_work.md
  - related_work/source_matrix.md
  - related_work/incremental_related_work.md
  - theory/README.md
  - theory/current_theory_update.md
  - theory/geometry_metric_definitions.md
write_scope: theory_only
```

本 QA 文件是 Cycle 06 theory 阶段的审查清单。按照工作区规则，本阶段只写 `theory/`，读取其他目录但不修改。Q1、Q2 已有用户答案；Q3–Q8 待用户回答后推进正式写入。

---

## 角色与工程边界

当前对话角色：**Theory**。

允许写入：`mypaper/theory/`

不允许写入：`result/`、`related_work/`、`exp/`、`next_cycle/`、`research_cycles/`、`local_experiment_results/`、`code_handoffs/`、`code_handins/`

Theory 阶段的职责是更新机制假设、变量定义、指标解释和 claim 边界。Theory 可以为 Experiment Design 提供概念框架和解释规则，但不预注册具体实验阈值、不改 eval/training protocol、不写代码 handoff。

---

## 当前 cycle 判定

```text
cycle_06_sft_feasibility_and_degradation
```

---

## 核心背景摘要

从已读文件提炼出 Theory 阶段必须处理的核心事实。

### 结果层面（Cycle 05 已确认）

- 所有 8 个模型在 NuminaMath ID 上均未超过未训练 base（base / theta0 ≈ 0.55）。
- `sft_n1024` 出现严重数学退化（NuminaMath ID −0.163，MATH500 −0.174），OOD-lite 稳定。
- OOD-lite 四个任务在全部 8 模型间 spread <1.5%，无判别力。
- Gate C04/D04 均 FAIL，但失败原因是判别能力不足，非 OPD 损害泛化。
- Geometry（Cycle 04）只有方向性信号，UV/sketch 未保存，不能做机制 claim。

### Cycle 06 定位（Next Cycle Seed 已确认）

- 纯 SFT-only 可行性和退化检验 cycle，不训练新 OPD arm。
- 先确认 SFT-from-base 在干净协议下是否可学。
- 再测 SFT 规模增加时是否出现退化。
- Geometry 本轮只作 diagnostic，不做主 gate 或机制 claim。

### Theory 阶段的核心认识（Q1 用户答案确立）

**Cycle 06 对理论的贡献非常小。** SFT 能力退化是普遍已知的工程现象，不需要新理论解释。当前 cycle 更多是工程验证：找到合适的模型、训练格式和测评协议。

因此 Theory 在本轮的核心工作是：

> **明确什么条件下才允许进入 OPD 比较（claim boundary 更新）**，而不是为 SFT 退化新建一套理论变量。

SFT-feasibility-first 是 K1 的前置条件，不是新的机制主张。

### 模型身份风险（Q2 用户答案确立）

- 本地 `/root/autodl-tmp/model/Qwen/Qwen3-1.7B`：已经历 post-training 的 instruct-style checkpoint。
- 官方 `Qwen/Qwen3-1.7B-Base`：pretraining-only causal LM。

两者不等同，必须在 claim boundary 里显式区分，防止后续 claim 混淆模型身份。

---

## Q1：Cycle 06 theory 是否应正式改写为 SFT-first ladder？

**背景：** `current_theory_update.md` 仍是 Cycle 05 的 "base-aware K1 / OPD-vs-SFT stability" 框架。是否应把主轴切换到 SFT-first？

**选项：**
- a. 主轴改为 SFT-first ladder，OPD stability 作为后续条件命题保留。
- b. 继续把 OPD-vs-SFT stability 作为主理论，只在备注里提 SFT feasibility。
- c. 保留 K1-base-aware 为主标题，但新增 Cycle 06 子节 SFT-feasibility-first（作为前置条件，而非新机制主张）。

**你的回答（已确认）：c**

> 我认为是 c。之前的推导还是比较有用的，但是当前做的检测也是必要的。但是我认为现在做的不应该放到理论中，因为这是工程上的问题，能力衰退是一个普遍常见的情况，不需要我们对其进行说明，也就是当前 cycle 实际上对理论的贡献非常小，更多的是找到合适的 model 和训练、测评。

**Theory 解读：**

选 c 意味着：

1. K1-base-aware 主框架不变，不重写。
2. 新增一个小节说明：在 SFT-from-base 能产生可测量 gain 的设置中，K1 主张才有意义——这是 **claim boundary 和前置条件**，不是机制解释。
3. SFT 退化曲线（scale → degradation）是 motivating observation，属于工程观察，不在 theory 里建新变量表。

---

## Q2：本轮应如何定义 "base"？

**背景：** Cycle 06 可能涉及两个不同对象。不区分会导致后续 claim 混淆。

**选项：**
- a. 需要。在 claim boundary 章节新增显式定义，区分 `local_base_row` 与 `official_pretraining_base_checkpoint`。
- b. 暂不需要，留给 Experiment Design 写。
- c. 需要，但只写成 claim-boundary warning，不展开成理论变量。

**你的回答（已确认）：a**

> 我建议是写成 a，但是不排除后续将其弱化的可能。

**Theory 解读：**

在 claim boundary 章节新增：

```text
local_base_row ≠ official_pretraining_base_checkpoint

本地 Qwen3-1.7B（instruct stack，已 post-trained）：
  path: /root/autodl-tmp/model/Qwen/Qwen3-1.7B
  在 0-shot chat 下的表现不能等同于 pretraining-only LM 的起点。

官方 Qwen3-1.7B-Base（pretraining-only）：
  path: Qwen/Qwen3-1.7B-Base（HuggingFace hub 官方）
  无 instruction-tuning；chat/think 假设不自动成立。

如果 Cycle 06 改用官方 Base，theory 在下一轮需要区分两者的适用假设
（base 的初始能力差异是否带来更大 headroom，尚未实验）。
```

---

## 📌 Q1–Q2 部分讨论结论（2026-06-23）

✅ 已确认：

1. **Cycle 06 theory 贡献最小化**：K1-base-aware 主框架保留，不改写。SFT-feasibility-first 以小节形式加入，定性为进入 OPD 比较的**前置条件**，不是新机制主张。SFT 退化（catastrophic forgetting）属于工程已知现象，不在 theory 文件里立新变量表。

2. **模型身份 claim boundary 显式化**：在 claim boundary 新增 `local_base_row ≠ official_pretraining_base_checkpoint` 的显式定义与警告，防止后续混淆。

---

## Q3：Theory 的最小写入内容确认

**背景：** 根据 Q1 用户答案，Cycle 06 对 theory 贡献极小，不应把 SFT 退化的工程变量写成新的核心理论变量。但 claim boundary 需要更新以反映前置条件。

**请确认以下清单哪些内容应写入 theory（勾选或标注）：**

| 候选写入内容 | 归属分析 | 应写入？ |
|---|---|---|
| K1 前置条件小节（SFT feasibility as precondition for K1 validity） | Claim boundary / 框架扩展 | ❓ |
| `local_base_row ≠ official_base_checkpoint` 显式定义 | Claim boundary | ✅ Q2 已确认 |
| Gate A06 / B06 的 theory 解释边界（gate 通过/失败如何解释） | Claim boundary（非阈值） | ❓ |
| OOD-lite 降级声明（preservation check only） | Claim boundary 更新 | ❓ |
| `extraction_validity` 已有变量：是否更新为覆盖本轮协议场景 | 现有变量维护 | ❓ |
| `sft_target_learning`、`sft_degradation`、`scale_exposure`、`degradation_onset` 作为新核心变量 | **不推荐**：工程变量，属于 Experiment Design | ❌ 不写 |
| Geometry diagnostic-only 声明（见 Q6） | geometry_metric_definitions.md | 待 Q6 确认 |

**你的回答（Q3 推断，基于 Q4–Q8 答案）：**

Q3 没有直接填写，但从 Q4=a、Q5=a、Q6=a、Q7=a、Q8=b 可完整推断：

| 候选写入内容 | 应写入？ | 依据 |
|---|---|---|
| K1 前置条件小节（SFT feasibility as precondition） | ✅ | Q4=a 接受 Gate A06/B06 boundary，前置条件是其逻辑基础 |
| `local_base_row ≠ official_base_checkpoint` 显式定义 | ✅ | Q2 已确认 |
| Gate A06 / B06 的 theory 解释边界 | ✅ | Q4=a |
| OOD-lite 降级声明（preservation check only） | ✅ | Q5=a |
| `extraction_validity` 变量更新 | ❌ | 现有变量已覆盖概念；Cycle 06 minimal contribution 原则 |
| `sft_target_learning`、`sft_degradation`、`scale_exposure`、`degradation_onset` | ❌ | Q1 确认：工程变量，不写入 theory |
| Geometry diagnostic-only 声明（含 effective rank、drift 解释边界） | ✅ → geometry 文件 | Q6=a 三点全写入 `geometry_metric_definitions.md` |

---

## Q4：Gate A06 / B06 的理论解释边界

**背景：** Experiment Design 会写具体 threshold；Theory 应定义 gate 通过/失败的解释意义，防止结果出来后临时换标准。注意：Theory 只写解释边界，不写数字 threshold。

**建议的解释边界：**

```text
A06 pass：当前 model/task/data/protocol 设置具备足够 SFT 可学性，
           可以考虑在后续 cycle 进行 OPD 比较（但仍需 Experiment Design 决定 cost、control 和 matching）。
A06 fail：当前设置存在 headroom 不足或 protocol/task/model 不匹配；
           不能解读为 OPD 本身失败，应先调整 model/task/data 或格式。

B06 degradation observed：SFT scale 可诱导能力 tradeoff；
                           后续 OPD forgetting-prevention 测试变得有意义。
B06 no degradation：当前设置可能在现有 scale 下对 SFT 稳定；
                     OPD forgetting-prevention 尚不可在此设置中测试。
```

- a. 接受，写入 theory。
- b. 基本接受，但需弱化"A06 pass → 可考虑后续 OPD 比较"的措辞（已由 Experiment Design 管辖）。
- c. 不接受，重新定义。

**你的回答：**
a

---

## Q5：OOD-lite 是否只保留为 preservation check？

**背景：** Cycle 05 已证实 OOD-lite spread <1.5%，无判别力。`claims_allowed.md` 已写入非正式结论，但 theory 文件里尚无对应 claim boundary。

**建议写入 claim boundary：**

```text
OOD-lite（MMLU / TruthfulQA / WinoGrande / ARC-challenge）在当前 1.7B + NuminaMath 设置下
是 capability-preservation check，不是 OPD-vs-SFT 或 SFT scale 的判别轴，
除非未来实验出现更大的、协议有效的 spread。
```

- a. 写入 claim boundary，措辞如上。
- b. 写入 claim boundary，但保留探索性分析空间（若 Cycle 06 出现意外大幅变化，可记录为新发现，不视为已有结论被推翻）。
- c. 不写入 theory，只在 `claims_allowed.md` 维护即可。

**你的回答：**
a

---

## Q6：Geometry 在 Cycle 06 中的理论地位

**背景：** related-work QA 已确认 geometry 本轮降级为 diagnostic-only，不新增 representation 文献。Theory 是否需要同步更新 `geometry_metric_definitions.md`？

**建议写入 `geometry_metric_definitions.md` 的三点：**

1. Cycle 06 SFT-only runs 中，geometry metrics 可用于诊断 representation 变化是否与 SFT learning/degradation 同向，但不能支持 OPD mechanism claims（本轮无 OPD arm）。
2. lower effective rank 不能自动解释为 stability；它也可能是 harmful concentration / collapse，需与 ID learning 量共同解读。
3. drift 必须与 learning amount 共同解释；若 Cycle 06 从 base 或官方 base checkpoint 训练，reference point 应重新定义（不自动沿用 `theta0` 基准）。

**选项：**
- a. 三点全写入 `geometry_metric_definitions.md` 和 `-ch.md`。
- b. 只写 diagnostic-only 声明（第 1 点），不改 metric interpretation 细则（第 2、3 点留给 Experiment Design 后处理）。
- c. 暂不改 geometry 文件，只在 `current_theory_update.md` 中加注。

**你的回答：**
a

---

## Q7：哪些 claim 必须禁止？

**建议的禁止项（基于已有 `claims_allowed.md` + Cycle 06 新增边界）：**

1. 不声称 OPD 已防止 catastrophic forgetting（Cycle 06 无 OPD arm）。
2. 不声称 Cycle 06 是 OPD-vs-SFT comparison。
3. 不把 `sft_n1024` 写成已证明的过拟合或 forgetting 机制，只写成 motivating observation / consistency signal。
4. 不把 OOD-lite 写成当前设置下的判别轴。
5. 不把本地 Qwen3-1.7B instruct checkpoint 写成 official Qwen3-1.7B-Base（pretraining-only）。
6. 不把 geometry diagnostic 写成 mechanism proof。
7. 不把 protocol-invalid 的旧数字（旧 Cycle 04 GSM8K ~0.41、MATH500=0.000 等）当作理论证据。

**选项：**
- a. 全部正确，写入 claim boundary。
- b. 需要修改某几条（请说明）。

**你的回答：**
a

---

## Q8：正式写入方式

**背景：** Q1 用户已明确 Cycle 06 theory 贡献极小，主要任务是 claim boundary 更新和模型身份定义。

- a. **小更新**：只在 `current_theory_update.md` 和 `-ch.md` 追加 Cycle 06 子节（K1 前置条件 + claim boundaries），不改 geometry 文件。适合 Q6 选 b 或 c。
- b. **中更新**：更新 `current_theory_update.md` / `-ch.md`，并同步更新 `geometry_metric_definitions.md` / `-ch.md` 的 Cycle 06 diagnostic 声明和解释细则。适合 Q6 选 a。
- c. **大更新**：重写 theory 目录结构，拆分 SFT feasibility、OPD K1、geometry 三个文件。（不推荐：与 Q1 "贡献极小" 原则矛盾，当前 K1 框架仍成立，无需重构。）

**你的回答：**
b

---

## 📌 全部 QA 讨论结论（2026-06-23）

✅ 所有问题已确认，可进入正式写入。

### 核心结论

1. **Cycle 06 theory 贡献最小化**：K1-base-aware 主框架不变，不重写。SFT-feasibility-first 作为前置条件子节加入，不是新机制主张。SFT 退化是工程已知现象，不建新变量表。

2. **模型身份 claim boundary 显式化**：`local_base_row ≠ official_pretraining_base_checkpoint`，写入 claim boundary 定义。

3. **Gate A06/B06 解释边界写入 theory**（Q4=a）：Theory 只写"如何解释 gate 结果"，不写数字 threshold。

4. **OOD-lite 降级写入 claim boundary**（Q5=a）：在当前 1.7B + NuminaMath 设置下，OOD-lite 只是 capability-preservation check，不是判别轴。

5. **Geometry 三点全写入 `geometry_metric_definitions.md`**（Q6=a）：diagnostic-only 声明 + lower effective rank 解释边界 + drift reference point 更新。中文版同步。

6. **禁止 claim 列表全部写入**（Q7=a）：7 条全部写入 claim boundary 章节。

7. **中等强度写入**（Q8=b）：更新 `current_theory_update.md` / `-ch.md` + `geometry_metric_definitions.md` / `-ch.md`，不重构 theory 目录。

### 实际写入清单

| 目标文件 | 写入内容 |
|---|---|
| `theory/current_theory_update.md` | 新增 Cycle 06 子节：K1 前置条件、模型身份定义、Gate A06/B06 boundary、OOD-lite 降级、禁止 claim 列表 |
| `theory/current_theory_update-ch.md` | 同步中文版 |
| `theory/geometry_metric_definitions.md` | 新增 Cycle 06 diagnostic notes：diagnostic-only 声明、lower effective rank 解读警告、drift reference point 更新 |
| `theory/geometry_metric_definitions-ch.md` | 同步中文版 |
