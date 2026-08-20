# 最小化实验 03：TRL-first OPD-like 闭环

## 0. Material Passport

```yaml
artifact_type: code_experiment_plan
status: implemented_and_run        # v2 中等规模已跑通，见 RESULTS_03_v2.md
date: 2026-06-07
last_revised: 2026-06-13           # 按真实代码 run_opd_minimal_closure_v2.py 修订
implementation_entry: experiments/opd_sft_h1/scripts/run_opd_minimal_closure_v2.py
results_doc: experiments/opd_sft_h1/RESULTS_03_v2.md
linked_main_task: nextStepCodex/00_main_task_03.md
linked_experiment_design: nextStepCodex/03_experiment_design_03.md
linked_code_plan: nextStepCodex/04_code_plan_03.md
implementation_note: "Use TRL first with LoRA; verl sidecar deferred."
```

> **本文档为设计计划，以下标注「实测」处为真实实现/运行偏离原计划的修订。**
> 完整真实结果见 [`RESULTS_03_v2.md`](RESULTS_03_v2.md)；主入口为 `scripts/run_opd_minimal_closure_v2.py`。

## 1. 实验目的

本实验是第三版 K1 主线的最小化工程闭环，不是最终完整实验。

要回答的最小问题：

```text
在同一个 LoRA SFT(Teacher-Rollout) cold-start theta_0 出发时，
TRL on-policy distillation-like LoRA 训练能否生成可评估 checkpoint，
并进入 GSM8K selector / selected full eval / GetSlice geometry / matched-gain 管线？
```

如果能跑通，再观察：

```text
在近似 matched GSM8K gain 下，
TRL-OPD-like 是否比 continued SFT 有更小的 OOD-lite drop，
以及是否有更稳定的 GetSlice activation-conditioned spectrum。
```

## 2. 与完整实验设计的关系

保留约束：

- OPD-like 与 continued SFT 必须从同一个 `theta_0` 出发。
- 主比较使用 matched GSM8K gain；MATH500 降级为 hard-math diagnostic。
- 保留 `SFT-small-LR` 控制，也就是 `SFT-LR-low`。
- OPD-like 必须记录 `lmbda/pi_mix`、rollout prompt source、teacher identity。
- OOD-lite 指标至少记录 MMLU / WinoGrande / TruthfulQA 的 score 与 drop。
- 几何仍围绕 GetSlice 风格 `A_t^D = W_t L_t^D`。

本最小实验的降级：

- Teacher-Rollout cold-start 与 OPD-like 第一阶段都使用 TRL `DistillationTrainer`；cold-start 在 registry 中记为 `cold_start`，OPD-like 记为 `TRL-OPD-like`，不声称等同于 verl 标准 OPD。
- 全部训练优先使用 LoRA/PEFT，不做 full AdamW student 训练。
- 不做 rollout/train/teacher 三方分离。
- 不做 NPD / Lightning-OPD replication。
- 不要求第一阶段保存 principal angle 所需的 UV/sketch；没有 UV 时明确标记 unavailable。

## 3. 实验条件

### 3.1 共享起点

当前没有 `theta_0`，因此先训练一个 LoRA SFT(Teacher-Rollout) cold-start checkpoint：

```text
theta_0 prompt size: 512 prompts
cutoff_len: 4096
training signal: teacher online rollout / distillation over prompts
```

这里的 `SFT(Teacher-Rollout)` 指从已有数据中只取 prompt，训练时由 teacher 在线提供 rollout/distillation signal；它不是使用已有 answer/CoT completion 的普通 SFT，也不是预先落盘的 teacher rollout 数据集。

必须记录：

- base model。
- cold-start prompt source。
- teacher identity and teacher mode。
- checkpoint path。
- full baseline eval：GSM8K、MATH500、MMLU、WinoGrande、TruthfulQA-MC1、TruthfulQA-MC2。
- X/S GetSlice geometry。

### 3.2 Continued SFT controls

**实测（v2）**：原计划用 `SFT-LR-mid / SFT-LR-low` 扫学习率，实现时改为**固定 lr=1e-5、扫数据量** `[256, 512, 1024, 2048]` 来扫出不同 GSM8K_gain 点（公平性由 matched-gain 配对本身保证，不要求训练配置对齐）。每条 SFT trajectory 从同一个 `theta_0` 出发，用 NuminaMath 母池的 problem+solution 做监督，LoRA 训练，4096 长度，3 epoch。

```text
sft_n256 / sft_n512 / sft_n1024 / sft_n2048
learning_rate: 1e-5（固定）
num_train_epochs: 3
监督标签: 母池 train_sft（problem + solution[+answer]）
```

SFT checkpoint selection 目标是找到最接近 OPD selected checkpoint GSM8K_gain 的数据量点。
> 实测发现：SFT 的 GSM8K 随数据量**单调崩坏**（256→0.440, 512→0.380, 1024→0.295），数据越多越过拟合训练分布。

### 3.3 TRL OPD-like runs

最低运行：

```text
TRL-OPD-lmbda-1.0
```

可选：

```text
TRL-OPD-lmbda-0.5
```

训练设置：

```text
student: theta_0
teacher: /root/autodl-tmp/model/Qwen/Qwen3-4B
training type: LoRA / PEFT
lmbda: 1.0（纯 on-policy）, beta: 0.5, loss_top_k: 1
max_length: 4096
max_completion_length: 512
```

**实测（v2）**：OPD 跑 200 步 × grad_accum 4 = 消耗 **800 prompt**（DataSize 语义=实际消耗 prompt 数），
而非计划中的固定 1024。teacher mode = local teacher（非 vLLM server）。

**实测执行机制与成本**（详见 [`RESULTS_03_v2.md`](RESULTS_03_v2.md) §7）：

```text
use_vllm: false（走 HF model.generate，无 vLLM continuous batching）
每 optimizer step（消耗 4 prompt）时序:
  ① 切 4 slice，lmbda=1.0 → 全部 on-policy
  ② student rollout（主开销）: for slice in 4 逐个生成 512 token —— 串行，batch=1
  ③ 逐 micro-step: teacher 前向给 logits → JSD → backward 累积
  ④ 4 个累积完做 1 次 LoRA 更新
```

- **主开销 = student 在线 rollout**（每 step 自回归生成 4×512=2048 token，串行）。
- 4 个 prompt 之间**串行**（batch=1 + for 循环），无并行；GPU 利用率仅 ~30%。
- **训练时间**：OPD distill 段 ≈ 6.1 h（800 prompt），cold-start ≈ 4.0 h；
  对比 4 个 SFT 合计仅 ≈ 1.3 h（3840 样本×3ep）。**OPD 比 SFT 慢约 60–70 倍**，根因即串行 rollout。
- 可选加速：batch=4/grad_accum=1（一次 batched generate）、use_vllm=true、或减小 max_completion_length。

记录字段：

- `lmbda` / `pi_mix_lambda`。
- student rollout prompt source。
- teacher model identity。
- teacher mode：local teacher 或 external vLLM teacher server。
- loss type / `beta` / `loss_top_k`。
- rollout count。
- teacher call count，如果 TRL logs 可解析；否则记录为 missing。
- generated checkpoint paths 与删除状态。

## 4. 最小数据与 checkpoint 设计

### 4.1 Prompt data

第一版用小规模数学 CoT prompt subset：

```text
SFT(Teacher-Rollout) cold-start: 512 prompts
OPD follow-up: 800 prompt（200 步 × grad_accum 4）
SFT follow-up: 256 / 512 / 1024 / 2048（数据量扫描）
source: NuminaMath-1___5
```

**实测（v2）统一母池**：从 NuminaMath-1\_\_\_5 用 seed=42 一次性采样后切分为互不重叠的 4 个 slice，
消除 v1 中"cold/OPD/SFT/probe 各自独立采样"引入的 bias：

```text
train_prompts.jsonl   # prompt-only，cold/OPD/SFT 都从这里取前 N
train_sft.jsonl       # problem + solution[+answer]，SFT 监督标签（与 train_prompts 行对齐）
heldout_eval.jsonl    # held-out 验证集（OPD 算 JSD / SFT 算 CE）
probe_prompts.jsonl   # X/S rollout 用（与训练/eval 不重叠）
```

要求：

- SFT 与 TRL 使用同源 prompt distribution（统一母池保证）。
- prompt source / 切分元信息写入 `pool/pool_meta.json`。
- 由于数据是 CoT，长度按 4096 设计。

### 4.2 Checkpoint selection policy

**实测（v2）两层选择**（替代原计划的"latest + best_by_gsm8k"存储策略）：

```text
第 1 层（训练轨迹内，防过拟合）:
  eval_strategy=steps + load_best_model_at_end=True + metric_for_best_model=eval_loss
  OPD/cold-start: held-out JSD ；SFT: held-out CE
  save_total_limit=2，取轨迹内 eval_loss 最低的 checkpoint

第 2 层（跨模型，构建主结论）:
  OPD 端: max(GSM8K) 选 best run
  SFT 端: GSM8K_gain 最接近 OPD 的数据量点做配对
```

> 注：第 2 层 OPD 选择按 GSM8K 绝对值，在 `eval_limit=200` 下噪声较大；本次单 OPD run 未暴露。
> 实测未做"eval+GetSlice 后删除权重"——所有 6 模型的 adapter + merged_model 都保留在输出目录。

## 5. Evaluation

### 5.1 两级 eval

训练中 selector eval：

```text
实测（v2）: held-out eval loss（OPD/cold-start=JSD, SFT=CE），用于轨迹内 best ckpt
            （原计划是 GSM8K-only selector，改为 eval loss 以防过拟合）
```

`theta_0` 和**全部 6 个模型**的 full eval（实测对所有模型评估，非仅 selected）：

```text
GSM8K
MATH500
MMLU
WinoGrande
TruthfulQA-MC1
TruthfulQA-MC2
（eval_limit=200/任务，控制时长；统计噪声较大）
```

暂不纳入主表：

```text
AIME24: disabled_hard_math_too_sparse
IFEval: deferred
HumanEval: disabled_requires_unsafe_code
```

### 5.2 指标

每个 full-eval checkpoint 输出：

```text
GSM8K
GSM8K_gain
MATH500
MMLU
WinoGrande
TruthfulQA-MC1
TruthfulQA-MC2
OOD_lite_avg
OOD_lite_penalty_p2
OOD_lite_penalty_p3
Worst_OOD_lite_drop
per_benchmark_OOD_lite_scores
per_benchmark_OOD_lite_drops
```

`GSM8K_gain`、MMLU/WinoGrande/TruthfulQA drops 的 baseline 都是同一个 `theta_0`。

## 6. Geometry

### 6.1 Probe

**实测（v2）具体探针来源**：

```text
X: 冻结统一参考探针 = theta0 对 probe_prompts 前 32 条 rollout，所有模型共用同一份 X
S: 按模型区分的训练分布探针
   - theta0  : teacher(Qwen3-4B) 对 cold-start prompt 显式离线 rollout
   - opd     : OPD 模型自己 rollout
   - sft_n*  : 该 SFT 的训练数据采样（problem→question, solution→answer）
```

并实测做了 **S×model 完整交叉矩阵**（每个模型的激活都在所有模型的 S 探针下各跑一遍），
几何表用 `s_probe_source` / `is_diagonal` 标记对角与非对角项。
R probe（TRL student rollout）按计划未作硬性要求；OPD 的 S 即等价于 student rollout probe。

### 6.2 Metric

最低计算：

```text
singular spectrum
spectral gap
spectral_gap_abs_delta_from_start
spectral_drift_from_start
effective_rank
X-S spectrum-level gap
```

如果没有 `sUV/xUV`：

```text
principal angle = unavailable_no_uv
projection X-S gap = unavailable_no_uv
```

### 6.3 观察层

主图先看：

```text
layer_14
self_attn.o_proj
mlp.down_proj
```

**实测（v2）**：仅取 `layer_14` 单层 × 7 模块（q/k/v/o_proj + gate/up/down_proj）。
附录 heatmap 改为 **model × s_probe 交叉的 effective_rank 热图**（`appendix_cross_effrank_heatmap.png`），
非原计划的"28 层 × 7 模块"全层热图。
> 实测关键发现：`self_attn.o_proj` 是 OPD/SFT 机制差异最大的模块——
> OPD drift=0.037（最小），SFT-256 drift=0.111（最大），diff=−0.074。

## 7. Matching

规则：

```text
先用 OPD trajectory 的 GSM8K selector eval 选出 best_by_gsm8k。
然后对每条 SFT trajectory，找最接近 OPD selected GSM8K_gain 的 checkpoint。

valid if |GSM8K_gain_OPD - GSM8K_gain_SFT| <= 2 points.
如果分数是 0-1 标度，则 threshold = 0.02。
otherwise mark nearest_match or unmatched_nearest.
```

主比较：

```text
OOD_lite_penalty_p2
Worst_OOD_lite_drop
MATH500 diagnostic
spectral_gap_abs_delta_from_start
spectral_drift_from_start
effective_rank change
X-S spectrum-level gap
compute cost
```

兜底策略：

```text
valid_match: 可作为主结论。
nearest_match: 只作为 exploratory。
out_of_range: 改做 endpoint-vs-endpoint 或 trajectory diagnostic。
insufficient_gain: 说明训练信号不足，需要调 LR/steps/data。
```

## 8. 预期输出

实验输出目录（实测）：

```text
/root/autodl-tmp/exp0609/opd_minimal_03_v2/
```

实测产物：

```text
pool/{train_prompts,train_sft,heldout_eval,probe_prompts}.jsonl + pool_meta.json
step2_cold_start/ step3_opd_distill/ step4_sft_controls/   # 6 模型 adapter + merged_model
eval/csv_results/target_metrics_results.csv
getslice/inputs/{X, S/{model}}/  getslice/outputs/{model}/step_{N}/{S__probe|X}/layer_14/
tables/eval_trajectory.csv
tables/ood_penalty.csv
tables/matched_gsm8k_pairs.csv
tables/geometry_metrics.csv          # 含 s_probe_source / is_diagonal 交叉维度
tables/geometry_long.csv             # 完整奇异谱
tables/main_matched_result.csv
figures/main_geometry_trajectory.png
figures/main_ood_vs_geometry.png
figures/appendix_cross_effrank_heatmap.png   # model × s_probe 交叉热图
selection.json / summary.json
```

如果某些产物尚未生成，必须在 registry 中标记：

```text
missing
implemented_plumbing_only
deferred
deleted_after_eval_and_getslice
```

## 9. 判断标准

### 9.1 工程成功

满足以下条件即可算第一阶段工程成功：

1. `theta_0` SFT(Teacher-Rollout) cold-start 成功输出 checkpoint。
2. `theta_0` 成功完成 full baseline eval 和 GetSlice。
3. TRL OPD-like run 成功输出 checkpoint，并能用 GSM8K selector 选出 `best_by_gsm8k`。
4. continued SFT control 成功输出 checkpoint，并能选出 `closest_to_opd_gsm8k`。
5. selected OPD/SFT checkpoint 都能进入同一 eval parser。
6. selected OPD/SFT checkpoint 删除前都完成 GetSlice。
7. matched selector 能输出 pair table。

### 9.2 初步支持 K1

仅当同时满足：

```text
TRL-OPD-like lower OOD_lite_penalty_p2
+ geometry drift/gap delta not worse than SFT
+ match_status = valid_match
```

才写成“初步支持”。

**实测（v2）结论：三条件全部满足 → 初步支持 K1。**

```text
match_status = valid_match (gap=0.010 ≤ 0.02)
OOD_lite_penalty_p2: OPD 0.0100 < SFT-256 0.0182 (delta=-0.0082)
谱漂移: OPD 0.0497 < SFT-256 0.0544；o_proj 子空间扰动 OPD 小一个量级
```

但需注意局限（见 RESULTS_03_v2.md §6）：GSM8K 未真正"上升"，差异在 eval_limit=200 噪声内，
当前结论为"OPD 维持性能时几何/OOD 更稳"，而非"提升性能且更稳"。

### 9.3 不支持或需要调整

以下情况不支持直接推进 K1：

- TRL-OPD-like 无法达到可观察的 GSM8K gain。
- SFT controls 无法接近 OPD 的 GSM8K gain，`unmatched_nearest` 太多。
- SFT 完全匹配或优于 TRL-OPD-like。
- OOD-lite 更好但 geometry 无差异，只能写成 weak mechanism。
- geometry 更好但 OOD-lite 不好，不能支持 K1。

## 10. 下一阶段决策

| 结果 | 下一步 |
|---|---|
| TRL 闭环跑通且结果有信号（**当前状态**） | 扩大 eval 样本量降噪 / 换 theta0 未饱和的更难分布，论证"提升+更稳" |
| TRL 闭环跑通但信号弱 | 调 `lmbda`、teacher、prompt source、LR，再重复小实验 |
| TRL 工程受限 | 切换到 verl 标准 OPD sidecar |
| 需要高并行 | 等前两步稳定后再做 verl async / fully async |

### 10.1 实测待办（v2 暴露的改进点）

- 第 2 层跨模型 OPD 选择改用更稳健策略（eval_loss 选 best 或全量 GSM8K 降噪），替代当前 `max(GSM8K@limit200)`。
- 加大 eval 样本量（去掉 eval_limit 或调高）以让 GSM8K 差异可信。
- 换 theta0 未饱和的更难分布，争取真实 GSM8K 提升。
