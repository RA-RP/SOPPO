# 当前 Cycle 07 意图

```yaml
artifact_type: next_cycle_intent_ch
cycle: cycle_07_base_model_sft_feasibility
status: ready_for_related_work_then_experiment_design
created: 2026-06-24
source_qa: next_cycle/QA.md
source_file: current_cycle07_intent.md
hard_dependency: Related Work 阶段必须在 Experiment Design 之前完成
write_owner_next: Related Work → Experiment Design
```

本文件是 Next Cycle Seed → Related Work → Experiment Design 的交接文件，
记录 Cycle 07 QA 讨论后的 seed 决策。

---

## Seed 决策

Cycle 07 是**base 模型 SFT 可行性 cycle**，核心变化如下：

> 将主要实验模型从 **Qwen3-1.7B instruct** 迁移至
> **Qwen3-4B base**（预训练 checkpoint，无 instruct tuning，无 chat/think 格式）。

此决策基于 Cycle 06 两项发现：
- No-think concise SFT **破坏了**思考功能在 instruct 模型中的作用（Cycle 06 think probe，
  indicative，n=100）。Instruct 训练模式不是干净的 SFT 可行性实验起点。
- 相关工作（Rethink SFT 等）在 SFT 对比实验中使用**无 instruct/think 格式的 base 模型**。

**Instruct 模型实验后置到补充实验。** 核心 SFT vs OPD 对比将从干净的 base 预训练 checkpoint 出发构建。

---

## 硬性依赖：Related Work

**在 Related Work 完成之前，Experiment Design 处于阻塞状态。**

Related Work 需要确认或更新以下内容：

| 决策点 | 需要确认的内容 |
|---|---|
| 模型选择 | Qwen3-4B base 是否是正确 checkpoint？size、checkpoint 路径、可用性。 |
| 训练格式 | 文献中 base 模型 SFT 可行性实验使用什么数据格式（纯 Q+A、无 chat template、无 think block）？ |
| 训练数据 | 是否继续使用 NuminaMath？文献是否建议更换？ |
| 主要评测轴 | 对于 base（非 instruct）模型，正确的 ID / 下游轴是什么？NuminaMath-test 是否仍然适用？ |
| Stop sequence | 确认 base 模型的生成式数学评测需要显式 stop sequence。 |

---

## 即时前置任务（pre-experiment）

以下任务开销较小，不需等待 Related Work 完成。应在 Cycle 07 开始时、Experiment Design 之前执行。

### 任务 1：Base series stop-fix（解决 A06b UNDETERMINED）

对所有 base-series 的 numina 和 GSM8K 评测重新运行，为每个 base 模型 runner 加上
`stop=["\n\nProblem:"]`。Cycle 06 已确认此修复可消除约 66% 的 runaway-continuation
截断伪影。

- 产出：第一批**有效的** base-series NuminaMath 和 GSM8K 数字。
- Gate 结论：关闭 A06b，给出明确 PASS 或 FAIL。
- 前置条件：无，可在 Related Work 之前运行。

### 任务 2：Cycle 06 geometry copyback（关闭 C06 PARTIAL）

将 Cycle 06 运行中的 `geometry_metrics.csv` 和 `principal_evidence.csv` 复制回
`local_experiment_results/cycle_06_sft_feasibility_and_degradation/run_01/`，
运行 OverlapLift 分析，将结果记录至 `result/current_picture.md` 的 Cycle 06 geometry 节，
并更新 `result/claims_allowed.md`。

- Gate 结论：关闭 C06 PARTIAL → DONE 或 FAIL。
- 前置条件：需找到 Cycle 06 训练运行中的 CSV 文件。

**注：** Think-probe 扩展（no-think SFT 破坏 thinking 功能，indicative n=100）**后置**。
模型迁移到 base（非 instruct）后，该发现变为次要，现在无需做确认实验。

---

## Cycle 07 主要目标

1. **A07 — Base SFT 可行性：** 从 Qwen3-4B base（预训练 checkpoint）出发的 SFT，
   是否能在主要下游 / ID 轴上产生可测量的增益？具体轴在 Related Work 之后由
   Experiment Design 预注册。

2. **D07 — A06b 重确认：** stop-fix 后，base 模型的 numina 和 GSM8K 可行性如何？
   （来自前置任务 1。）

3. **退化曲线：** 与 instruct no-think SFT 相比，base 模型 SFT 随数据规模增加是否
   同样退化 ID 表现？还是预训练 checkpoint 有不同的行为模式？

4. **Base 模型评测协议：** 建立 base（非 instruct）模型的正确评测协议：chat template、
   think mode、few-shot 格式、stop sequence。必须在 Experiment Design 中最终确定。

---

## 明确后置的项目（来自 Cycle 07 QA）

| 项目 | 后置原因 |
|---|---|
| Think-format SFT | 模型迁移到 base 后不再需要；相关工作可能确认 base-format SFT 是正确方向 |
| Think-probe 确认性扩展 | 若主实验使用 base 模型（无 instruct tuning、无 think mode），该 probe 变为次要 |
| OPD 训练 arm | 先建立干净的 base 模型 SFT baseline；OPD 对比需要有效 SFT 增益才能进行 |
| B07 gate（格式对比：think vs no-think） | 模型迁移到 base 预训练 checkpoint 后，不适用 |
| C07 gate（think-SFT forgetting） | 不做 think-format SFT 则不适用 |

---

## 建议 Gate 结构

以下是 seed 级别的 gate，Related Work 评审后由 Experiment Design 正式确认阈值。

| Gate | 用途 | 前置条件 | PASS → | FAIL → |
|---|---|---|---|---|
| D07（A06b 重确认） | 用 stop sequence 验证 base-series eval | 任务 1 完成 | 可做 base 可行性判断 | 检查 stop sequence 实现 |
| A07：SFT 可学习性 | Base SFT 在 ID/下游轴上超过 untrained base checkpoint | Related Work + Experiment Design | 进入 Cycle 08 退化刻画和 OPD 设计 | 重新设计：更换 scale、数据或评测轴 |
| （可选）Geometry | 收集 base 模型 geometry（OverlapLift、effective rank、从 base checkpoint 出发的 drift） | 训练完成 | 为机制层提供诊断信号 | 无 geometry 结论；从能力结果推进 |

---

## 活跃范围

**Cycle 07 中要做的：**
- 即时前置任务（stop-fix + geometry copyback）
- Related Work 阶段（Rethink SFT + 相关文献）：确认模型、格式、数据、评测轴
- Base 模型 SFT 训练（Qwen3-4B base，格式待定）
- 在预注册的 ID / 下游轴上进行主评测
- 同步收集 geometry（OverlapLift、effective rank、drift）

**后置到 Cycle 08 或以后：**
- OPD 训练 arm 和 OPD-vs-SFT 比较
- Think-format SFT
- Think-probe 确认性扩展
- 论文级别的机制主张

---

## Experiment Design 必须明确的决策

Related Work 完成后，Experiment Design 必须明确预注册以下内容：

1. **模型 checkpoint：** Qwen3-4B base（预训练 checkpoint）的确切路径。
   确认：是 HuggingFace 上的 `Qwen3-4B-Base`，还是其他 checkpoint？
2. **训练格式：** 数据格式（纯 Q+A、无 chat template、无 think block）、
   目标文本结构、tokenization。必须与 Related Work 协议匹配。
3. **训练数据：** 保留 NuminaMath（可能），确认样本量、split 规则。
4. **主要评测轴：** 预注册 Gate A07 的 ID 轴。若因 base 模型评测格式变化而不再
   使用 NuminaMath-test，需说明理由。
5. **Base 模型评测协议：** 关闭 chat template，关闭 think mode，few-shot 格式，
   所有生成式任务加 stop sequence。
6. **Geometry 协议：** drift 的参考 checkpoint（必须是 base，不是 theta0），
   目标层，以及需要保存的压缩 UV/sketch artifact。

---

## Cycle 07 后的路由

- **若 A07 通过（SFT 增益 > 预注册阈值）：** 进入 Cycle 08，刻画退化曲线并设计
  OPD 比较。Base 模型确认为实验平台。
- **若 A07 失败：** 在 OPD 之前重新设计。选项：更换 base 大小、更换任务、
  增加数据、或按 Related Work 调整训练格式。
- **D07：** 无论哪种结果，将 base-series stop-fix 后的数字记录为新的基准真相，
  更新 `result/current_picture.md` 和 `result/claims_allowed.md`。

---

## 下一步阅读顺序

Related Work 应按以下顺序阅读：

1. `next_cycle/current_cycle07_intent.md`（英文版，主要参考）
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `exp/experimental_methodology/model_and_data.md`

然后定位并阅读：
- "Rethink SFT"（首要参考文献）
- 其他 base 模型 SFT 可行性相关论文

Related Work 完成后，Experiment Design 应按以下顺序阅读：

1. `next_cycle/current_cycle07_intent.md`
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `exp/experimental_methodology/model_and_data.md`
5. `exp/experimental_methodology/training_protocol.md`
6. `exp/experimental_methodology/evaluation_protocol.md`
7. `exp/experimental_methodology/geometry_protocol.md`
8. Related Work 输出

然后撰写 `exp/current_experiment_design.md`、`exp/current_acceptance_criteria.md`
以及 cycle07 代码交接文件。
