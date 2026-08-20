# 当前实验设计（中文阅读版）

```yaml
view_type: current_latest_view_chinese
cycle: cycle_07_base_model_sft_feasibility
status: active_cycle07_design
maintenance_rule: 中文版用于阅读，英文原始版 current_experiment_design.md 用于 provenance
last_organized: 2026-06-24
```

---

# 实验设计 07：Base 模型 SFT 轨迹研究

## 核心问题

1. Qwen3-4B-Base（预训练 checkpoint）用 think-format long-CoT SFT 后，能否在 MATH500 上取得可测量的 gain？（**Gate A07——主要可行性门控**）
2. SFT 轨迹是否呈现 dip-and-recovery 模式（Rethink SFT, Ren et al. 2026）？——早期 checkpoint MATH500 低于 base，后续回升。
3. response_length 是否跟随预期的轨迹特征（dip 期暴涨，recover 期收缩）？
4. 几何指标（effective rank、drift-from-base、OverlapLift）沿轨迹如何变化？

**本 cycle 不训练 OPD arm**。只跑单条 SFT 轨迹，建立 Cycle 08 三方对照的基线。
H-OPD（OPD ≈ 直接到达 recover 状态跳过 dip 暂态）是 Cycle 08 的预注册假设。

---

## 模型与数据

| 项目 | 值 |
|---|---|
| 学生/Base checkpoint | `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base` |
| 训练数据 | `/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet` |
| 训练样本量 | 5000 行，seed=42 |
| 数据来源 | Rethink SFT `jasonrqh/Math-CoT-20k` —— OpenR1-Math queries + Qwen3-32B think-format long-CoT + math-verify 过滤 |

---

## 训练格式

遵循 Rethink SFT Table 3 模板，通过 Qwen chat template 包装：

```
<|im_start|>user
{problem}
Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

其中 `{response}` = 数据集中的完整 think-format 回答：`<think>...推理过程...</think>分步解答\n\boxed{答案}`。

- 无系统提示（no system prompt）
- 用 Qwen3-4B-Base tokenizer 的 `apply_chat_template`
- Loss 只计算 assistant turn（标准 SFT masking）

---

## 训练超参数

| 参数 | 值 | 备注 |
|---|---|---|
| Base 模型 | Qwen3-4B-Base | 预训练 checkpoint |
| LoRA rank | r=32 | 针对 4B 模型提升（之前 Cycle 04–06 用 r=16） |
| LoRA alpha | 64 | alpha/rank = 2（比例不变） |
| LoRA dropout | 0.05 | 不变 |
| LoRA 目标模块 | q/k/v/o/gate/up/down_proj | 不变 |
| 学习率 | 5e-5 | 与 Rethink SFT Table 3 对齐 |
| 调度器 | cosine decay | |
| Warmup | 10%（约 62 步） | |
| Weight decay | 0.01 | |
| Epochs | 8 | 多 epoch 确保 recovery phase 可见 |
| 有效 batch size | 64 | per_device=1，gradient_accumulation=64 |
| 总步数 | ~624 | = 5000/64 × 8 ≈ 624 步 |
| 最大 prompt 长度 | 3072 | |
| 最大 response 长度 | 16384 | think-format 所需 |
| 训练框架 | TRL SFTTrainer | 无 vLLM colocate |
| dtype | bf16 | |

---

## Checkpoint 网格（Theory 硬要求：多 checkpoint 采样）

在以下步数保存完整 LoRA adapter checkpoint：

```
steps_to_save = [5, 10, 20, 40, 80, 160, 320, 480, 624]
```

加上未训练的 base（step_000），共 **10 个评测目标**。

早期密集采样（步数 5–40）是为了捕捉 dip 暂态（Rethink SFT Table 9：MATH500 在 step 20 出现 dip）。

---

## 评测协议

### 数学任务（生成式，think-format）

三个数学任务均用相同 prompt 格式（chat template + boxed 指令）。
已训练的 checkpoint 会自然产生 `<think>...</think>` 输出；未训练的 step_000 会直接输出，从任意位置提取 `\boxed{}`。

| 任务 | N | 格式 | 最大生成 | 提取器 | 轴 |
|---|---|---|---|---|---|
| MATH500 | 500 | 0-shot，chat template，temp=0.6 | 32768 | last `\boxed{}` + math_verify | ID（主门控轴） |
| NuminaMath-test | 1024 | 0-shot，chat template，temp=0.6 | 32768 | last `\boxed{}` + math_verify | ID（退化检查） |
| AIME24 | 30 | 0-shot，chat template，temp=0.6 | 32768 | last `\boxed{}` + math_verify | OOD（难题，pass@1） |

**Response length 测量（必须）：**
MATH500 每个 checkpoint 额外记录 mean response token length，
产出 `response_length_trajectory.csv`——这是观察 dip/recover 状态的必要诊断数据。

### OOD-lite 任务（log-likelihood，无 chat template）

与项目已建立的 loglikelihood 协议一致。

| 任务 | N | 协议 | 轴 |
|---|---|---|---|
| GPQA-Diamond | 198 | 0-shot，无 chat template，loglikelihood | OOD 推理 |
| MMLU-Pro | 500（limit） | lm-eval，`--limit 500 --seed 42` | OOD 知识 |

**MMLU-Pro 子集说明：** 固定 `--limit 500 --seed 42`，分数只在实验内部可比，不与官方 MMLU-Pro 榜单数字比较。

---

## 几何协议

几何是**诊断性**的（非门控），在全部 10 个 checkpoint 上采集。

- **参照 checkpoint：** step_000（未训练 Qwen3-4B-Base）。所有 drift 指标均相对 base 计算，本 cycle 无 theta0。
- **层：** {9, 18, 27} —— 36 层 Qwen3-4B 的约 25%/50%/75% 位置。
- **模块：** q/k/v/o/gate/up/down_proj（与 LoRA 目标模块相同）。
- **UV/sketch 文件必须保存**（这是 Cycle 04/05 的失败点，导致无法计算 subspace overlap）。
- **principalEvidence.py**：每个 SFT checkpoint 与 step_000 对比，输出 OverlapLift per checkpoint。

---

## 预期轨迹特征（来自 Rethink SFT Table 9，4B + Math-CoT 全参数训练）

| 步数范围 | MATH500 预期行为 | response_length | OOD 预期 |
|---|---|---|---|
| step_000（base） | 70.8（外部锚点） | 中等 | GPQA 14.6 |
| step_005–020 | **dip 低于 base**（≈62–65） | 急剧上升 | 下降 |
| step_040–080 | recovery 开始 | 开始收缩 | 反弹 |
| step_160–320 | 高于 base，继续上升 | 收缩 | 恢复中 |
| step_480–624 | **远高于 base**（目标 ≈86） | 稳定/较短 | 超过 base |

以上是全参数训练的参考预期。我们的 LoRA variant 收敛可能更快，dip 幅度可能更小。实测为准，不预判。

---

## 预期输出结构

```
local_experiment_results/cycle_07_base_sft_trajectory/run_01/
  RESULTS_07.md                    ← gate 判断 + 轨迹摘要
  trajectory_scores.csv            ← 10 checkpoint × 5 任务，附 ±SE
  response_length_trajectory.csv   ← 10 checkpoint，MATH500 mean response tokens
  geometry/
    geometry_metrics_step_{NNN}.csv     (×10)
    principal_evidence_step_{NNN}.csv   (×10)
  figures/
    trajectory_math500.png
    trajectory_ood.png
    response_length.png
    overlap_lift_trajectory.png
  run_provenance.json
```

**Cycle 06 geometry copyback（延后）：** 所有 Cycle 07 评测完成后，将 Cycle 06 run 目录中的
`geometry_metrics.csv` + `principal_evidence.csv` 复制到
`local_experiment_results/cycle_06_sft_feasibility_and_degradation/run_01/`。
这关闭 C06 PARTIAL，但不在 Cycle 07 关键路径上。

---

## 非目标

- 本 cycle 不训练 OPD arm
- 不与 Cycle 04/05/06 SFT 数字直接比较（模型、格式、数据均已改变）
- 不跑 instruct 系列评测（instruct 协议在 Cycle 07 不再是主线）
- 不在 Cycle 07 结论中声称 H-OPD——H-OPD 是 Cycle 08 假设
- Gate A07 通过前，不将 Rethink SFT 的 4B 数字视为已复现
