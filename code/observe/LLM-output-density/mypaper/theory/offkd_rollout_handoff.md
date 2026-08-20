# Theory → Coder 转交：off-KD control 臂 —— 阶段 1（Teacher Rollout）

```yaml
view_type: handoff_note
cycle: cycle_09 → Tier B（训练臂）
stage: 1 of 2（本文件只管 rollout；训练规格另发）
from: Theory
to: Code Execution
created: 2026-07-14
gpu: 96G × 2（两卡切分 prompt 并行 rollout）
urgency: 高 —— 论文截稿在即；rollout 是训练的前置依赖
guard: 本阶段不训练；产物为静态语料 + teacher logprob
```

## 这个实验是干什么的（一句话）

**off-KD = OPD 的 off-policy 版本**：同 teacher、同 loss、同 LoRA、同网格，
**唯一变量是训练输入的来源** —— OPD 用 student 自己的 rollout（on-policy），
off-KD 用 **teacher 预先生成的静态回复**（off-policy）。

它是本项目**唯一**能拆开"on-policy 采样"这个变量的实验。三个混淆变量
（on-policy 采样 / 稠密软目标 / teacher 邻近性）中，它单独隔离第一个。

⚠️ **因此"唯一变量"这个前提必须严格守住** —— 任何采样参数、长度、prompt 集的
偏离都会毁掉这个实验的归因价值。

---

## 阶段 1 任务：Teacher Rollout

### 0. 开工前的阻塞检查（必须先做，5 分钟）

**查 verl 的 `distillation/losses.py` 中 `forward_kl_topk` 的实现**（我们 patch 过这个文件，
见 `code_evolution.md:339`），确认它消费的 teacher logprob 是哪种口径：

- **raw logprob**（未经 temperature scaling）
- 还是 **processed logprob**（经采样参数处理后 —— verl 的 rollout config 里
  `logprobs_mode: 'processed_logprobs'`）

**这决定 rollout 时存什么。存错口径 → KL 数值错误 → 整个实验作废、必须重跑。**

查明后在 `code/QA_cycle09.md` 记录结论，并据此设定 vLLM 的 logprob 采集方式：
- 若需 raw → 用 temperature=1.0 单独取 logprob（或按 verl 的实际消费方式对齐）
- 若需 processed → 按下方采样参数直接采集

**这一条没确认之前不要发车。**

### 1. Rollout 规格（严格对齐 Cycle 08）

```
引擎:         vLLM（与 Cycle 08 一致：rollout.name='vllm'）
teacher:      Qwen3-8B（与 Cycle 08 OPD 的 teacher 同一个）
prompts:      Math-CoT-20k 的 query —— 必须是 OPD 训练用的同一批、同顺序
max_tokens:   10240          ← 对齐 Cycle 08 训练的 max_response（code_evolution.md:339）
temperature:  0.6            ← 对齐 Cycle 08 的 rollout 采样参数
top_p:        0.9
top_k:        -1
n:            1
do_sample:    true
seed:         42
dtype:        bfloat16
logprobs:     32             ← top-32，口径按第 0 步的结论
```

**采样参数的依据**：Cycle 08 的 hydra config（`experiments/opd_sft_h1/scripts/outputs/2026-06-30/16-32-23/.hydra/config.yaml`）
里 `rollout: temperature 0.6 / top_p 0.9 / top_k -1 / n 1 / seed 42`。

⚠️ **注意**：Cycle 08 里 **teacher 从不生成文本**（它只做前向、返回 top-k logprob）。
所以"Cycle 08 的 teacher 采样参数"并不存在。上述参数是 **student 的 rollout 参数**，
我们让 teacher 用同一组参数生成 —— 这样"唯一变量是谁在生成"才成立。

### 2. 两卡并行

按 prompt 切分（前一半 / 后一半），两卡各跑一半，结果合并。**不要用 tensor parallel 切模型**
（8B 单卡放得下，数据并行吞吐更高）。

### 3. 必须记录的字段（每条样本）

| 字段 | 用途 |
|---|---|
| `prompt` / `prompt_token_ids` | 训练输入 |
| `generation` / `generation_token_ids` | 训练目标序列 |
| `teacher_top32_logprob`（逐 token） | **off-KD 的 loss 依赖它；缺了就要重跑 teacher** |
| `finish_reason`（stop / length） | 判断是否截断 |
| `n_tokens` | 长度分布 |
| `has_boxed` | 数据质量诊断 |
| 全部采样参数 + seed | provenance |

### 4. 截断样本：**保留**（用户裁定 2026-07-14）

理由：off-KD 用 **token-level forward_kl_topk**，它在每个 token 位置匹配 teacher 分布，
**不需要"完整答案"**；截断序列前面的 token 照样提供有效监督。这与 OPD 的处理一致
（OPD 的 student rollout 同样有截断，未被丢弃）。

**但必须报告截断率** —— 它是要写进论文的实验事实，且需与 OPD 的 student rollout 截断率对照。

### 5. 产物

```
{EXP_ROOT}/offkd/rollout/
  ├── teacher_rollout.jsonl          （文本 + token ids + finish_reason + 长度 + boxed）
  ├── teacher_top32_logprob.{npy|safetensors}   （逐 token top-32，与 jsonl 行对齐）
  └── rollout_manifest.json          （全部采样参数、seed、模型路径、切分方式、截断率统计）
```

copyback 到 `local_experiment_results/cycle_09.../offkd/`；handin 追加 `code/code_evolution.md`。

---

## 完成后回报（Theory 需要这三个数）

1. **截断率**（finish_reason='length' 的比例）—— 与 OPD student rollout 的截断率对照
2. **长度分布**（mean / median / p90）—— teacher 的 CoT 比 student 长还是短？
3. **logprob 口径的最终结论**（第 0 步的检查结果）

## 阶段 2（训练）—— 另发规格，本文件不含

训练配置将严格复制 Cycle 08：LoRA r32/α64/all-linear、lr 5e-5、loss `forward_kl_topk` topk=32、
token-mean、ppo_epochs=1、max_response 10240、624 步、**同一 checkpoint 网格**
[0,5,10,20,40,160,624]（几何/评测要与 OPD/SFT 逐点对照）。

---

# Coder → Theory 完成回交（2026-07-16）

```yaml
return_status: COMPLETE
stage_1_rollout: complete
stage_1_logprob_pass2: complete_and_validated
stage_2_training: complete_624_steps
checkpoint_backfill: complete_and_validated
evaluation: complete
three_arm_aggregate: complete
formal_eval_finished_cst: 2026-07-16 05:43
aggregate_finished_cst: 2026-07-16 05:44
interpretation_policy: raw readings and provenance only; Theory adjudicates
```

## 1. Stage 1 回报：原文件要求的三个数

| 项目 | 正式读数 |
|---|---:|
| prompts | 5,000 |
| response tokens | 34,486,621 |
| length 截断 | **1,072 / 5,000 = 21.44%** |
| mean length | **6,897.3242** |
| median length | **6,952** |
| p90 / max length | **10,240 / 10,240** |
| has-boxed rate | **84.44%** |
| pass-1 生成时间 | 352.0 min |
| pass-2 logprob 时间 | 112.12 min |

**logprob 最终口径：RAW。** pass 1 只负责按
`temperature=.6 / top_p=.9 / seed=42 / max_tokens=10240` 生成；pass 2 对完全相同的
`prompt + generation` token 序列，以 `temperature=1.0, prompt_logprobs=32`
取得未经采样变换的 teacher top-32 logprob。不能使用 pass-1 sampling-time processed
logprob。

pass-2 正式验证为 **PASS**：5,000 行；top-k shape
`[34,486,621, 32]`；pass-1 末 token exact；missing token id、非有限 logprob、
正 logprob、top-k 排序错误均为 0。5,000 条中 4,999 条满足
`max_prompt_tokens=1024`，1 条 overlong；这与正式训练的 4,999 样本口径一致。

**执行资源偏差（只影响墙钟时间）**：原转交写 96G×2 prompt 切分，实际在用户切换后的
**96G×1** 上串行执行；teacher、prompt 顺序、采样参数、seed 与输出字段未改变。

## 2. Stage 2 训练完成态

正式训练完成 **624/624 steps**：batch 16，2 epochs，312 steps/epoch，
shuffle=false，drop-last=true；Qwen3-4B-Base，LoRA r32/alpha64/all-linear，
AdamW lr `5e-5`，纯 `forward_kl_topk(k=32)`、response-token mean。
本次正式训练耗时 **12,626.61 s = 3.51 h**，峰值训练显存约 18.65 GiB。

| step | loss | student top-32 mass | teacher top-32 mass | grad norm |
|---:|---:|---:|---:|---:|
| 1 | 0.363188 | 0.994181 | 0.999954 | 0.578125 |
| 5 | 0.323489 | 0.985140 | 0.999976 | 0.324219 |
| 10 | 0.263329 | 0.989815 | 0.999971 | 0.144531 |
| 20 | 0.241160 | 0.994237 | 0.999978 | 0.089355 |
| 40 | 0.242604 | 0.995461 | 0.999966 | 0.095703 |
| 160 | 0.167230 | 0.997728 | 0.999970 | 0.093262 |
| 624 | 0.149999 | 0.998289 | 0.999975 | 0.105469 |

**必须保留的实现 caveat**：仓库中没有原文承诺的独立 Stage-2 handoff。本次使用
`standalone HF+PEFT offline forward-KL loop`，不是 Cycle 08 的 verl/FSDP2 在线 runner；
逐项对齐了模型、LoRA、bf16、样本顺序、batch/epoch、AdamW、loss、clamp、token-mean
与 checkpoint，但不能把“runner 实现完全相同”写成已验证事实。Theory 若强调
“唯一变量只有 response source”，须把它表述为**配置与 loss 路径对齐的 off-policy
control**，或另做 runner-equivalence 审计。

## 3. 十点轨迹的 checkpoint provenance

原生正式训练保存点为 `{0,5,10,20,40,160,624}`。为完成用户批准的十点轨迹，
`80/320/480` 从最近 landmark 独立回放生成，不覆盖正式训练目录。

| 回放分支 | 首个恢复步 | 比较步数 | max Δloss | max Δstudent mass | Δteacher mass | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 40→80 | step 41 bitwise exact | 40 | 3.5285e-4 | 1.9378e-5 | 0 | PASS |
| 160→480 | step 161 bitwise exact | 320 | 5.9252e-4 | 2.8620e-5 | 0 | PASS |

两支 formal/replay grad norm 均未触发 clip=1。validator 口径为“首步 bitwise 完整性 +
后续有界数值等价”；因此这三点必须标记为
`numerical_backfill_from_landmark`，**不是 uninterrupted bitwise replica**。

## 4. 正式 eval 协议与完整性

三臂统一 checkpoint grid：
`[0,5,10,20,40,80,160,320,480,624]`。

- checkpoint-wide：MATH500、GPQA-Diamond、MMLU-Pro、IFEval、TruthfulQA-MC1。
- MATH500：N=500；step 0–20 cap=4096，step 40–624 cap=16384。
- Numina：只跑 `{40,160,624}`，N=200，cap=12288。
- AIME24：各臂 MATH500 peak + final，N=30，cap=24576，
  seeds 42–51；secondary。
- 三臂 aggregate：**30/30 行完整**，mandatory task 缺格为 0。

formal off-KD eval 从 2026-07-15 22:08 CST 跑至 2026-07-16 05:43 CST；
三臂汇总于 05:44 完成。

## 5. 原样读数：MATH500 三臂完整轨迹

| step | OPD acc | SFT acc | off-KD acc | off-KD trunc |
|---:|---:|---:|---:|---:|
| 0 | 0.652 | 0.636 | 0.652 | 0.046 |
| 5 | 0.552 | 0.656 | 0.574 | 0.064 |
| 10 | 0.614 | 0.642 | 0.580 | 0.092 |
| 20 | 0.744 | 0.572 | 0.672 | 0.150 |
| 40 | 0.830 | 0.590 | 0.736 | 0.288 |
| 80* | 0.836 | 0.648 | 0.764 | 0.202 |
| 160 | 0.832 | 0.692 | 0.778 | 0.244 |
| 320* | 0.866 | 0.728 | 0.794 | 0.088 |
| 480* | 0.856 | 0.738 | **0.800** | 0.046 |
| 624 | 0.848 | 0.752 | 0.794 | 0.048 |

`*` = off-KD 数值回放 checkpoint。MATH500 peak：OPD
`step320=.866`，SFT `step624=.752`，off-KD `step480=.800`。

## 6. 原样读数：稀疏任务

### NuminaMath

| step | OPD | SFT | off-KD |
|---:|---:|---:|---:|
| 40 | 0.640 | 0.540 | 0.520 |
| 160 | 0.670 | 0.535 | 0.565 |
| 624 | 0.610 | 0.560 | **0.620** |

### AIME24（10-seed mean，secondary）

| arm | peak step / acc | peak-step AIME acc / trunc | final AIME acc / trunc |
|---|---|---|---|
| OPD | 320 / 0.866 | 0.2700 / 0.9700 | 0.2400 / 0.9700 |
| SFT | 624 / 0.752 | 0.1867 / 0.7967 | 同 peak |
| off-KD | 480 / 0.800 | 0.1767 / 0.4300 | 0.1533 / 0.3933 |

## 7. 原样读数：final checkpoint 横向表

| metric @624 | OPD | SFT | off-KD |
|---|---:|---:|---:|
| MATH500 acc | **0.8480** | 0.7520 | 0.7940 |
| Numina acc | 0.6100 | 0.5600 | **0.6200** |
| AIME24 10-seed mean | **0.2400** | 0.1867 | 0.1533 |
| GPQA-Diamond acc | **0.4192** | 0.3990 | 0.3889 |
| MMLU-Pro exact match | **0.4921** | 0.4621 | 0.3536 |
| IFEval prompt-strict | **0.3161** | 0.2144 | 0.2311 |
| IFEval instruction-strict | **0.4556** | 0.3561 | 0.3645 |
| TruthfulQA-MC1 acc | 0.3476 | **0.3660** | 0.3427 |

粗体只标该行最大点估计，**不代表显著性裁决**。完整十点五任务读数以 CSV 为准。

## 8. Theory 必须看到的 flags

1. **F-offKD-1（runner caveat）**：Stage 2 是数值路径对齐的 HF+PEFT loop，
   不是与 Cycle 08 字节相同的 verl runner。
2. **F-offKD-2（回放 caveat）**：off-KD `80/320/480` 数值等价但非逐位等价；
   aggregate 已逐行标出来源。
3. **F-offKD-3（混合 provenance）**：OPD/SFT 复用既有正式结果，off-KD 是本轮新跑；
   例如 base MATH500 点估计 OPD/off-KD=.652、SFT=.636，不能把独立随机评测的
   step-0 差当训练效应。MATH 还按预注册规则混用 early/late 两档 cap。
4. **F-offKD-4（当前未含几何）**：本轮交付是 rollout、训练与行为 eval；
   **尚未对 off-KD 跑 GetSlice/UV/geometry**。十个 merged models 已保留在
   `/root/autodl-tmp/cycle09_offkd/_merged_models/`，可供 Theory 决定是否追加。
5. 本节不执行“off-KD 更像 OPD 还是 SFT”的理论裁决，也不把点估计差写成显著差。

## 9. 交付物索引

论文 copyback 根：
`mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/offkd/`

- `rollout_manifest.json`
- `pass2_validation.json`
- `training_manifest.json`
- `train_metrics.jsonl`
- `backfill_validation.json`
- `evaluation_manifest.json`
- `offkd_eval_trajectory.csv`
- `offkd_aime24_seeds.csv`
- `three_arm_full_trajectory.csv`
- `three_arm_full_trajectory_manifest.json`

大体积 rollout、top-32 memmap、optimizer checkpoint、merged models 与逐题 eval
样本保留在 `/root/autodl-tmp/cycle09_offkd/`。
