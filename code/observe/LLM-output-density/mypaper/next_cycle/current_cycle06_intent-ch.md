# 当前 Cycle 06 意图

```yaml
artifact_type: next_cycle_intent_ch
cycle: cycle_06_sft_feasibility_and_degradation
status: ready_for_experiment_design
source_file: current_cycle06_intent.md
created: 2026-06-18
source_qa: next_cycle/QA-ch.md
write_owner_next: Experiment Design
```

本文件是 Next Cycle Seed 交给 Experiment Design 的入口，记录 QA 讨论后的 Cycle 06 种子决策。

---

## 种子决策

Cycle 06 是 **SFT-only 的可行性与退化诊断 cycle**。

本轮暂不做完整 OPD-vs-SFT 比较。除非用户在 Experiment Design 阶段明确重新打开范围，否则不训练新的 OPD arm。

本轮目标是把实验阶梯重新一步一步搭起来：

1. 确认普通 SFT 是否能提高选定的下游 / ID 目标。
2. 测量普通 SFT 是否随数据规模增加导致 ID/OOD 退化。
3. 用这些结果决定后续 OPD 实验是否值得做。
4. Geometry 保留为诊断层，不作为主 gate。

---

## Cycle 06 主问题

1. **SFT 可学习性：** 从未微调 student/base 出发的 SFT，是否能在选定的 primary downstream / ID 轴上产生真实增益？
2. **SFT 退化：** SFT 在什么数据规模（如果存在）开始导致 ID 或 OOD 性能退化？
3. **协议选择：** 哪个模型、数据规模、任务组合和输入输出格式适合建立干净的 SFT baseline？
4. **Geometry 诊断：** 低成本 S/X 与 PE 类 geometry probes 是否显示出与 SFT 学习或退化相关的变化？PE 的精确定义需在 Experiment Design 阶段写清楚。

---

## 当前范围

**训练 / 评测：**

- 只做 SFT-from-base arms。
- 候选初始数据规模：n=256、n=512、n=1024。Experiment Design 可以在有明确资源理由时加入 n=128（定位退化 onset）或 n=2048（压力测试）。
- 所有 eval 表必须包含 unfinetuned student/base reference row。

**后置：**

- 新 OPD arms。
- matched budget 下的 OPD 遗忘预防测试。
- 完整 OPD-vs-SFT stability comparison。
- 基于 geometry 的论文级机制主张。

---

## Experiment Design 必须明确的决策

Experiment Design 必须显式决定并预注册：

1. **模型身份：** 是否继续使用 `/root/autodl-tmp/model/Qwen/Qwen3-1.7B`
   作为 student/base，还是因为 headroom 问题更换模型或任务。
2. **训练数据：** SFT 数据来源、样本规模、split 规则，以及是否直接从 base 出发、不做 `theta0` warmup。
3. **训练格式：** chat/plain 格式、thinking mode、target text structure，以及是否与
   `exp/experimental_methodology/model_and_data.md` 保持一致。
4. **评测轴：** primary ID/downstream axis 和 OOD axes。现有默认候选为
   NuminaMath-test、GSM8K、MATH500 和 OOD-lite preservation checks，但设计必须说明
   NuminaMath 是否存在过强 ceiling。
5. **Scoring audit：** 每个任务的 extraction 与 chat/think/token 设置，遵守
   `exp/experimental_methodology/evaluation_protocol.md`。
6. **Geometry 输出：** 是否运行 S/X 与 PE 类 probes；PE 在本工作区中的操作性定义是什么；
   要测哪些层、模块、artifact，以及是否保存 post-hoc audit 所需的压缩 UV/sketch 输出。

---

## 建议 Gate 形状

以下是 seed-level gates；Experiment Design 需要正式写出精确阈值。

| Gate | 目的 | Pass 含义 | Fail 含义 |
|---|---|---|---|
| A06: SFT 可学习性 | 测试 SFT 是否超过 base | 当前设置可支持后续 OPD 比较 | 先更换模型、任务、数据或格式，不进入 OPD |
| B06: SFT 退化 | 追踪 SFT scale 上的 ID/OOD delta | 退化曲线可测且可解释 | 当前评测轴不能干净诊断遗忘 |
| C06: 协议有效性 | 确认每个任务都有 think/chat/token/extractor 设置 | 分数可解释 | 停止并修复协议，不能写结果 |
| D06: Geometry 诊断 | 低成本收集 S/X 与 PE 类信号 | 可指导后续机制实验 | 不做机制 claim，只用能力结果推进 |

---

## Cycle 06 后的路由

- 若 SFT 不能提高目标：暂不跑 OPD。先重新设计模型、任务、数据规模或输入输出格式。
- 若 SFT 有增益且不退化：该设置可能适合后续 OPD-vs-SFT 比较。
- 若 SFT 有增益但大规模退化：下一轮 OPD cycle 可在已识别 scale 上测试遗忘预防。
- 若收集了 geometry 信号：在有效的能力/退化设置下比较 OPD 与 SFT 之前，只把它们作为诊断信号。

---

## 下一步读取

Experiment Design 应按顺序读取：

1. `next_cycle/current_cycle06_intent.md`
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `exp/experimental_methodology/model_and_data.md`
5. `exp/experimental_methodology/training_protocol.md`
6. `exp/experimental_methodology/evaluation_protocol.md`
7. `exp/experimental_methodology/geometry_protocol.md`

然后写 `exp/current_experiment_design.md`、`exp/current_acceptance_criteria.md`
和 cycle06 code handoff。
