# 当前代码结构说明

```yaml
view_type: current_code_structure
section: code
status: active
as_of_cycle: cycle_07_base_sft_trajectory
source: /root/LLM-output-density/Readme-experiments.md
maintenance_rule: update_incrementally_each_cycle_do_not_rewrite_in_full
last_updated: 2026-06-27
```

本文件描述 `experiments/opd_sft_h1` 实验包的当前代码结构和运行方式。每个 cycle 结束时追加 `## 变更（Cycle XX）` 节，说明本轮新增、删除、修改的内容；已稳定的历史描述不改写。

---

# experiments/opd_sft_h1 代码流程及配置

## 1. 总览

`experiments/opd_sft_h1` 是当前活跃的实验包，研究假设为：OPD（Output Probability Distillation）蒸馏相较于等规模 SFT 能在保持 OOD 稳定性的同时获得更大的 in-domain gain。

整个流程由单一入口脚本串行编排，分为五个阶段：

```text
build_unified_pool          (数据母池，一次性，CPU)
    │
    ├── train_opd_arm × (1 + |arms|)   (cold-start theta0 + OPD arms，GPU)
    ├── train_sft_control × |sft_sizes| (SFT controls，GPU)
    │
    ├── run_full_eval_v2                (全量评测，GPU)
    ├── build_eval_selection_gates      (Gate A/B/C/D 初判，CPU)
    │
    └── build_probes_multi              (X/S rollout 生成，GPU)
        └── run_getslice_cross × layers (GetSlice SVD，GPU)
            └── build_geometry_tables_cross + build_figures_cross (几何分析，CPU)
```

### 1.1 术语

`theta0`：冷启动基础模型，由 Qwen3-1.7B base 经 OPD cold-start（lmbda=1.0，teacher rollout 全监督）训练得到；作为所有后续比较的基线。

`OPD arm`：以 theta0 为起点，用 TRL DistillationTrainer 继续蒸馏的模型变体，通过不同的 `lmbda`（蒸馏强度）和 `seed` 区分。当前 cycle04 有两个 arm：`opd_lmbda1`（lmbda=1.0）和 `opd_lmbda05`（lmbda=0.5）。

`SFT control`：以 theta0 为起点，用 TRL SFTTrainer 对 NuminaMath 监督标签微调的对照组，通过不同的训练样本数 `N` 区分，label 格式为 `sft_nN`。

`DataSize`：训练实际消耗的 prompt/样本数，等于 `max_steps × grad_accum_steps`；SFT control 的 DataSize 等于 `num_samples`。

`X 探针`：用 theta0 生成的冻结参考分布，有两种变体：`X_prompt`（prompt-conditioned，greedy EOS 生成）和 `X_bos`（BOS-start unconditional，temperature=0.8 采样）。

`S 探针`：按模型区分的训练分布样本，theta0 的 S 来自 teacher rollout，每个 OPD arm 的 S 来自自己 rollout，每个 SFT control 的 S 来自其训练数据切片。

`OOD_lite_penalty_p2`：各 OOD benchmark 相对 theta0 下降幅度的 L2 范数，用作 OOD 退化门控。

### 1.2 文件结构

```text
experiments/opd_sft_h1/
├── scripts/
│   ├── run_opd_stability_gain_cycle04.py   # 当前主入口（Cycle 04）
│   ├── run_opd_minimal_closure_v2.py       # Cycle 03 v2（工具函数库）
│   ├── run_opd_minimal_closure.py          # Cycle 03 v1（ModelSpec / merge_lora 等基础工具）
│   ├── run_trl_distill_smoke.py            # OPD 训练子进程入口
│   └── regen_probes_dualx.py              # 单独重跑探针的工具脚本
├── opd_sft_h1/                            # Python 包（训练后端 + 几何分析）
│   ├── trl_runner.py                       # run_from_config() 核心训练逻辑
│   ├── registry.py                         # JSONL 注册表 I/O + schema 校验
│   ├── geometry_metrics.py                 # effective_rank / spectral_gap / drift / xs_gap
│   ├── geometry_reader.py                  # GetSlice JSON → geometry_rows
│   ├── eval_ingest.py                      # target_metrics CSV → eval_trajectory / ood_penalty
│   ├── match_math500.py                    # OPD/SFT matched pair 工具
│   ├── ood_metrics.py                      # OOD 聚合指标
│   ├── paths.py                            # resolve_repo_path / ensure_dir
│   ├── run_builder.py                      # build_run_record / load_yaml_config
│   └── table_builder.py                    # CSV 汇总工具
├── configs/                               # 历史实验 YAML（trl_first_minimal.yaml 等）
├── registry/                              # JSON Schema（run / checkpoint）
├── runs/                                  # 历史小规模 run 产物
└── tests/                                 # 单元测试 + fixtures
```

**Cycle 05 新增独立评测脚本**（在 `Eval/` 目录下，不在主流程内）：

```text
Eval/
├── eval_gsm8k_full.py      # GSM8K 修正版：chat template + enable_thinking=False + 3072 tokens
├── eval_math500_full.py    # MATH500 修正版：math_verify + boxed 提取，替代 dollar-span
└── run_cycle05_eval.py     # Cycle 05 全量重评脚本（8 模型 × 修正协议）
```

### 1.3 运行入口

完整运行（48G 单卡，vLLM colocate）：

```bash
python experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py
```

Smoke 模式（小数量快速验证流程通畅，无 GPU 密集计算）：

```bash
python experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py --smoke
```

禁用 vLLM colocate（回退 HF generate，显存不足时使用）：

```bash
python experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py --no-vllm
```

跳过 GetSlice 几何分析（仅跑训练+eval）：

```bash
python experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py --skip-getslice
```

产物默认写到 `/root/autodl-tmp/cycle04_opd_stability_gain/`，可通过 `--exp-root` 覆盖。

---

## 2. 数据母池阶段

### 2.1 功能

`build_unified_pool()` 从 NuminaMath-1.5 parquet 一次性采样并切分为四个互不重叠的 JSONL 切片，落盘后所有后续阶段直接复用，不重复采样。

```bash
# 数据源
/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet
```

### 2.2 切分逻辑

```text
总采样数 = n_sft_max + n_heldout + n_probe
sample = df.sample(n=总采样数, random_state=seed)

[0           : n_sft_max           ]  →  train_sft（SFT 监督标签）/ train_prompts（prompt-only）
[n_sft_max   : n_sft_max+n_heldout ]  →  heldout_eval（held-out 验证集）
[n_sft_max+n_heldout : 全部        ]  →  probe_prompts（X/S rollout 探针）
```

`train_prompts` 和 `train_sft` 行号完全对齐，前者只保留 `problem` 字段供 OPD/cold-start 使用，后者保留 `messages`（user+assistant）字段供 SFT 使用。cold-start 和 OPD arm 实际消耗时从 `train_prompts` 取前 `n` 条，不重新采样。

### 2.3 Cycle 04 参数

| 切片 | full 参数 | smoke 参数 |
|---|---|---|
| n_sft_max（SFT 控制组最大样本数） | 1024 | 16 |
| n_heldout（验证集） | 64 | 8 |
| n_probe（X/S rollout） | 32 | 4 |
| seed | 42 | 42 |

### 2.4 产物

```text
{exp_root}/pool/
├── pool_meta.json       # 切分参数 + 各切片路径（已存在则跳过整个 pool 阶段）
├── train_prompts.jsonl  # {problem: ...} × n_sft_max
├── train_sft.jsonl      # {messages: [{user},{assistant}]} × n_sft_max
├── heldout_eval.jsonl   # {messages: [{user},{assistant}]} × n_heldout
└── probe_prompts.jsonl  # {problem: ..., solution: ...} × n_probe
```

---

## 3. 训练阶段

所有模型都通过 `train_opd_arm()` 或 `train_sft_control()` 训练，产物为 `ModelSpec` 对象（冻结 dataclass，字段：`source`, `size`, `role`, `model_dir`, `checkpoint_path`）。

### 3.1 OPD arm 训练（含 cold-start theta0）

**核心机制**：训练运行在独立子进程（`subprocess` 调 `run_trl_distill_smoke.py`），父进程调用 `merge_lora_adapter()` 合并 LoRA。

子进程隔离的原因：vLLM colocate 模式会在子进程内初始化 `torch.distributed`（NCCL 进程组）。如果多个 OPD arm 训练在同一进程内串行执行，第二个 colocate 实例会撞上已销毁/残留的进程组，导致 NCCL 初始化失败。子进程退出后进程组自动清理，父进程在无 distributed 环境下做 LoRA merge，避免 `peft 0.19.1 + transformers 4.57.6` 的 `EmbeddingParallel` 兼容性 bug。

**训练配置（`_write_opd_arm_config()`）**：

```yaml
model:
  base_model: /root/autodl-tmp/model/Qwen/Qwen3-1.7B
  teacher_model: /root/autodl-tmp/model/Qwen/Qwen3-4B
  student_start_checkpoint: <上一阶段 merged_model 路径，cold-start 时为 None>

trl:
  lmbda: <1.0 或 0.5>          # 蒸馏强度：1.0=纯 teacher 监督，0.5=混合
  beta: 0.5
  loss_top_k: 1
  max_length: 4096
  max_completion_length: 512
  per_device_train_batch_size: 1
  gradient_accumulation_steps: <grad_accum>
  learning_rate: 3.0e-5
  max_steps: <max_steps>
  bf16: true
  optim: adamw_torch
  peft:
    enabled: true
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  use_vllm: true                 # vLLM colocate student rollout
  vllm_mode: colocate
  vllm_gpu_memory_utilization: 0.30
  vllm_max_model_length: 4096
  vllm_enable_sleep_mode: true   # 训练步让 vLLM 休眠，释放显存给 teacher + 反向
  vllm_tensor_parallel_size: 1
  vllm_sync_frequency: 1         # 每个 optimizer step 同步一次 LoRA 权重到 vLLM 引擎
```

`load_best_model_at_end` 在 vLLM colocate 下禁用（与 EmbeddingParallel 不兼容），改为保存末态 checkpoint 并在日志中监控 eval_loss。

**Cycle 04 参数**：

| 阶段 | max_steps（full） | grad_accum | DataSize | lmbda |
|---|---|---|---|---|
| cold-start theta0 | 64 | 4 | 256 | 1.0 |
| opd_lmbda1 | 200 | 4 | 800 | 1.0 |
| opd_lmbda05 | 200 | 4 | 800 | 0.5 |

**子进程入口**：`run_trl_distill_smoke.py` 解析 `--config` YAML，调用 `trl_runner.run_from_config()`，打印结果 JSON 后退出。

**`run_from_config()` 内部流程**（`opd_sft_h1/trl_runner.py`）：

1. 解析 `student_start_checkpoint`；若是 LoRA adapter（有 `adapter_config.json` 但无 `config.json`）则自动合并到 `merged_model/`，返回合并路径。
2. 加载 `DistillationConfig`（`trl.experimental.distillation`），构造 `LoraConfig`（PEFT）。
3. 加载 `train_dataset`（`messages` 格式 HuggingFace dataset）和可选 `eval_dataset`。
4. 实例化 `DistillationTrainer`，调用 `trainer.train()` + `trainer.save_model()`（保存 LoRA adapter）。
5. 写 `run_registry.jsonl` 和 `checkpoints.jsonl`。

**产物**：

```text
{exp_root}/{label}/
├── config_{label}.yaml         # 训练配置（父进程写，子进程读）
├── checkpoint_output/          # LoRA adapter（子进程 save_model 写）
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── merged_model/           # 合并后完整模型（父进程 merge 写）
│       ├── config.json
│       └── model.safetensors.index.json ...
└── registry/
    ├── run_registry.jsonl
    └── checkpoints.jsonl
```

`merged_model/config.json` 存在即视为该 arm 已完成，下次运行直接跳过训练和 merge。

### 3.2 SFT control 训练（`train_sft_control()`）

使用 TRL `SFTTrainer`，从 `train_sft.jsonl` 取前 `num_samples` 条，以 theta0 merged model 为起点做 LoRA 微调。

配置：

```python
lr=1.0e-5, max_seq_length=4096, peft_config=LoraConfig(r=16, lora_alpha=32)
load_best_model_at_end=True   # SFT 训练内无 distributed，可以启用
```

每个 SFT control 训练后立即在进程内合并 LoRA（无 distributed），不需要子进程隔离。

**Cycle 04 SFT 扫参**：`sft_sizes = [128, 256, 512, 1024]`，label 格式 `sft_n{N}`。

**产物**：与 OPD arm 相同目录结构，`model_dir` 指向 `merged_model/`。

---

## 4. 评测阶段

### 4.1 功能

`run_full_eval_v2()` 对所有 `ModelSpec`（theta0 + OPD arms + SFT controls）依次用 lm-eval + vLLM backend 跑 benchmark，输出汇总 CSV。

### 4.2 配置

```yaml
eval:
  enable_eval: true
  enable_result: true
  model_output_root: {exp_root}/eval/models/
  output_origin_root: {exp_root}/eval/origin/
  output_root: {exp_root}/eval/fix/
  csv_path: {exp_root}/eval/csv_results/
  batch_size: 6
  gpu_memory_utilization: 0.65
  vllm_model_len: 4096
  eval_limit: null
  include_path: Eval/tasks
  hf_hub_offline: true
  lm_tasks:
    - gsm8k
    - hendrycks_math500
    - mmlu
    - truthfulqa_mc1
    - truthfulqa_mc2
    - winogrande
```

### 4.3 评测协议（Cycle 05 修正后）

**GSM8K / MATH500**（修正协议，Cycle 05 起适用）：

- `--apply_chat_template`（Instruct 模型必须）
- `enable_thinking=False`（与 non-thinking SFT/OPD 训练数据对齐）
- `max_gen_toks=3072`
- MATH500 使用 `eval_math500_full.py`（`math_verify` + `\boxed{}` 提取），不用 lm-eval 默认 dollar-span

**MMLU / OOD-lite**（修正协议，Cycle 05 起适用）：

- 使用标准 loglikelihood MC 任务，**不加 chat template**，5-shot
- 与 Qwen3 官方技术报告（arXiv 2505.09388）一致（base model mode）

**注意**：原 lm-eval 默认 `hendrycks_math500.yaml` 的 dollar-span scorer 与 thinking-style 输出不兼容，产生全零，已由 `eval_math500_full.py` 替代。原 MMLU 加 chat template + enable_thinking=True 协议崩溃（~0.23，随机水平），已废弃。

### 4.4 Gate 评估（`build_eval_selection_gates()`）

读取 `target_metrics_results.csv`，以 theta0 为基线计算：

- `GSM8K_gain = score - theta0_GSM8K`
- `OOD_lite_penalty_p2 = ||max(0, baseline-score)||₂`
- `Worst_OOD_lite_drop`

产物：

```text
{exp_root}/tables/
├── eval_trajectory.csv
├── ood_penalty.csv
└── matched_gsm8k_pairs.csv
```

---

## 5. 探针与几何分析阶段

### 5.1 X/S 探针生成

**X 探针**（theta0 生成，冻结，所有模型共用）：

| 变体 | 生成方式 |
|---|---|
| `X_prompt` | theta0 greedy EOS，prompt-conditioned |
| `X_bos` | theta0 BOS-start 采样，temperature=0.8，seed=42 |

**S 探针**（按模型区分）：

| 模型 | S 数据来源 |
|---|---|
| theta0 | Qwen3-4B teacher rollout |
| opd_lmbda1 / opd_lmbda05 | 各自 student rollout |
| sft_n128…n1024 | 对应大小 `train_sft.jsonl` 切片 |

### 5.2 GetSlice 交叉矩阵

Cycle 04 在三个层运行：`GEOMETRY_LAYERS = [6, 14, 22]`（Qwen3-1.7B 共 28 层）。

```bash
python GetSlice/slice.py --config {cfg_path}
# cwd: /root/LLM-output-density/GetSlice
```

### 5.3 几何指标

| 指标 | 含义 |
|---|---|
| `effective_rank` | exp(H(p)) where p=softmax(σ²/Σσ²) |
| `spectral_gap` | (σ₁ - σ₂) / σ₁ |
| `log_spectrum_drift` | mean(|log(σᵢˢ) - log(σᵢˣ)|) |
| `xs_log_spectrum_gap` | mean(log(σᵢˢ)) - mean(log(σᵢˣ)) |

---

## 6. 产物目录总览

```text
/root/autodl-tmp/cycle04_opd_stability_gain/
├── pool/
├── theta0_cold_start/
├── opd_lmbda1/ opd_lmbda05/
├── sft_n128/ sft_n256/ sft_n512/ sft_n1024/
├── eval/
│   ├── models/ origin/ fix/
│   └── csv_results/target_metrics_results.csv
├── tables/
│   ├── eval_trajectory.csv  ood_penalty.csv  matched_gsm8k_pairs.csv
│   └── geometry_metrics.csv
├── getslice/
│   ├── inputs/X_prompt/ X_bos/ S/{source}/
│   └── layer{6,14,22}/{model}/{probe}/sMat.json xMat.json
├── figures/
└── summary.json

/root/autodl-tmp/floor_probe/cycle05_full_eval/   # Cycle 05 修正评测产物
```

---

## 7. 常见问题

**Q：为什么 OPD 训练要用子进程，SFT 不用？**

vLLM colocate 会在训练进程内初始化 NCCL 进程组，无法在同一进程内串行启动多个 colocate 实例。SFT 训练不使用 vLLM colocate，可以在进程内串行调用。

**Q：merged_model 已存在时是否会重新训练？**

不会。`train_opd_arm()` 检测到 `merged_model/config.json` 存在即直接返回。pool、GetSlice 输出、lm-eval 结果均有类似 skip-if-exists 逻辑。

**Q：load_best_model_at_end 为什么 OPD 里禁用了？**

`peft 0.19.1 + transformers 4.57.6` 下，vLLM colocate 初始化 distributed 后，`_load_best_model` 会走入 `EmbeddingParallel` 分支导致不兼容。SFT 无 distributed，可以正常使用。

**Q：GetSlice 配置中 cwd 为什么要设为 GetSlice/？**

`GetSlice/slice.py` 内部用相对路径导入 `utils/`，必须以 `GetSlice/` 为工作目录。

---

## 变更（Cycle 06）

**新增脚本（2026-06-23）**

### `experiments/opd_sft_h1/scripts/run_cycle06.py`

Cycle 06 主编排脚本，4 阶段串行（Phase 1 数据母池 → Phase 2 SFT 训练 → Phase 3 评测 → Phase 4 Geometry），支持：
- `--exp-root`：指定产物根目录（默认 `/root/autodl-tmp/cycle06_sft_feasibility_and_degradation/`）
- `--smoke`：小规模验证（n_sft_max=16，每阶段跑最小子集）
- `--start-from-phase N`：从第 N 阶段恢复（不重跑已完成阶段）

**Phase 1**：构建 pool_instruct.jsonl（messages 格式）+ pool_base.jsonl（plain text），以及 heldout_eval / heldout_eval_base / probe_prompts。

**Phase 2**：`train_sft_arm()` 函数，支持 `is_base_series` 标志区分两种训练格式；统一使用 `SFTTrainer`（无 subprocess 隔离）；`per_device_train_batch_size=2, gradient_accumulation_steps=4`，`num_train_epochs=3`，LoRA r=16。

**Phase 3**：`eval_instruct_series()` 调用现有 runner（chat template + enable_thinking=False）；`eval_base_series()` 调用新 runner_base（4-shot CoT，无 chat template）；lm_eval OOD-lite 按系列分别调用（instruct 系列 TruthfulQA 加 chat template，base 系列全部不加）。

**Phase 4**：`_build_s_probe()` 为每个模型生成 GetSlice S 探针（SFT arm 用训练切片，untrained base 用模型自身 rollout，base 系列用 plain text 不调 chat template）；`_run_getslice_layer()` 对每个 (模型, 层) 组合分别运行 GetSlice；`_run_principal_evidence()` 运行 3 次（layer 6/14/22）并调 `AnalyseMat/principalEvidence.py`。

### `experiments/opd_sft_h1/scripts/export_weights.py`

权重导出工具：加载 HuggingFace merged model，通过 `model.model.layers[N].{submodule}.weight` 提取权重，保存为 `.npy`。

- `--mode flat`：直接写入 `--output_dir`（用于 base model 参照，`base_model_npy_dir`）
- `--mode nested`：写入 `--output_dir/{data_size}/`（用于 finetuned 模型，`npy_output_root` 的两层结构）
- 输出文件名格式：`model_layers_{layer}_{module}_weight.npy`（如 `model_layers_6_self_attn.q_proj_weight.npy`）

### `Eval/component/numina/runner_base.py`

NuminaMath base 系列 eval：4-shot CoT，plain text format，无 chat template。前 4 行 test.jsonl 作固定 shots（`N_SHOTS=4`），其余行评分。与 instruct 系列 `runner.py` 接口相同（`run(model, label, outdir, ...)`）。

### `Eval/component/gsm8k/runner_base.py`

GSM8K base 系列 eval：4-shot CoT，plain text。4 条 shots 硬编码为 Wei et al. 2022 CoT paper 标准示例（constants `GSM8K_SHOTS`），写入 summary JSON 的 `shot_source` 字段。

### `Eval/component/math500/runner_base.py`

MATH500 base 系列 eval：4-shot CoT，plain text。4 条 shots 为 Hendrycks MATH 标准示例（constants `MATH500_SHOTS`），写入 summary JSON 的 `shot_source` 字段。

**修改文件**

`Eval/component/numina/runner.py`：默认参数更新为 `temperature=0.7, top_p=0.8, top_k=20, max_tokens=3072`，对齐 Cycle 05 验证协议（原默认 greedy + 1024 tokens）。

---

## 变更（Cycle 07）

**新增脚本（2026-06-25）**

### `experiments/opd_sft_h1/scripts/run_cycle07.py`

Cycle 07 主编排脚本，5 阶段串行（Phase 1 数据 → Phase 2 训练 → Phase 3 评测 → Phase 4 几何 → Phase 5 聚合回传），支持 `--run-root` / `--copyback-root` / `--smoke` / `--start-from-phase N`（1–5）。与 Cycle 06 的根本区别：**单条 SFT 轨迹**（无 arm/对照），按 step 网格存 11 个 checkpoint，逐 checkpoint 评测 + 几何。

- **GPU 隔离铁律**：编排父进程持零 GPU；训练、X-rollout 下放到 `cycle07_gpu_worker.py` 子进程，评测/GetSlice 也各走子进程，退出即由 OS 回收显存。
- 常量：`CHECKPOINT_STEPS=[5,10,20,40,80,160,320,480,624]`（+step_000 base +step_632 final=11），`GEOMETRY_LAYERS=[9,18,27]`，`EFF_BATCH=16`，`FIXED_STEPS=632`，`N_TRAIN=5000`，`N_PROBE=32`，`SEED=42`。
- per-task：`TASK_MAXTOK={math500:4096, numina:4096, aime24:16384}`，`LM_TASKS=[(gpqa, gpqa_diamond_zeroshot,0,None),(mmlu_pro, mmlu_pro,0,100)]`（100/class）。
- `_merge_checkpoint()` 复用 `run_opd_minimal_closure.merge_lora_adapter`（adapter→临时 merged 目录），用后即删。
- `_read_lm_acc()`（**Cycle 07 内修复**）：优先读 lm-eval `groups` 聚合块，metric 键扩展含 `exact_match,custom-extract`——否则 mmlu_pro（指标在 `groups.mmlu_pro` 的 custom-extract）解析为空。
- UV 产物（sUV/xUV .pt）本轮关闭（`save_s_uv_path/save_x_uv_path=None`），defer cycle08。
- Phase 5 `aggregate()` 写 `trajectory_scores.csv` / `response_length_trajectory.csv` / `RESULTS_07.md` / figures，并把 `geometry/*.csv` + `principal_evidence` 镜像进 copyback。**注**：phase5 用进程启动时的源码，若 run 中途改源码不生效（本轮 mmlu_pro 修复即因此未进 phase5 输出，靠独立脚本覆盖修正）。

### `experiments/opd_sft_h1/scripts/cycle07_gpu_worker.py`

GPU 隔离子进程，JSON config 驱动，两子命令：
- `train`：单条 `SFTTrainer`，`GridSaveCallback` 在网格步 `on_step_end` + `on_train_end` 存 LoRA adapter；`max_steps>0` 时覆盖 epochs（固定 632 步、eff_batch 16、max_len 10240、`gradient_checkpointing`）。
- `rollout-x`：base 冻结模型对 probe prompts 做 chat-template greedy 生成，产出 GetSlice X 探针 jsonl。

### `Eval/component/think_math/runner_think.py`

think-format 生成式数学 runner（math500/numina/aime24）。与 Cycle 05/06 的 no-think instruct runner（硬编码 `enable_thinking=False`+`max_model_len=4096`）不同：套 chat template 但**不**强制关 thinking，让模型自然吐 `<think>...</think>`，EOS 终止；记录 per-sample response **token 长度**与 `finish_reason`（dip-surge 诊断 + 事后非终止分析的关键字段）；采样 temp=0.6/top_p=0.9；`max_model_len=32768, max_tokens=30720`（贴 Qwen3-4B-Base 上限）。INSTR 用单换行对齐 Math-CoT-20k 训练格式。复用 `scorer_v2.score`+`scorer.extract_pred`（extract_pred 先剥 `</think>` 再抽末位 boxed）。CLI 子进程隔离调用。数据源：math500 本地 jsonl、numina `/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl`（JUNK 过滤）、aime24 `Maxwell-Jia/aime_2024`。

### `experiments/opd_sft_h1/scripts/cycle07_cap_pilot.py`（事后诊断）

截断/非终止诊断 pilot（非主流程）。merge step_080 → vLLM 大 cap 生成解 censor 长度分布，**分块流式落盘**（每块写 `*_samples.jsonl` + 更新 `pilot_summary.json`，中途可读/可断/crash-safe，修正 v1 一次性 generate 不落盘的缺陷）。per-task cap（math500 24576 / aime24 31744），输出各候选 cap{4k…32k}下截断率。结论：as-run acc 对 cap 稳健、截断本质是非终止（详见 code_evolution.md Cycle 07 handin）。

**修改文件**

`experiments/opd_sft_h1/scripts/run_cycle07.py` `_read_lm_acc()`：见上（mmlu_pro groups/custom-extract 解析修复）。
