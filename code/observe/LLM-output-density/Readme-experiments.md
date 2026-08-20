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

评测配置由 Python dict 内联生成并写到 `{exp_root}/eval/eval_config.yaml`：

```yaml
eval:
  enable_eval: true
  enable_result: true
  model_output_root: {exp_root}/eval/models/       # symlink 目录
  output_origin_root: {exp_root}/eval/origin/      # lm-eval 原始 JSON
  output_root: {exp_root}/eval/fix/                # 结果整理目录
  csv_path: {exp_root}/eval/csv_results/
  batch_size: 6
  gpu_memory_utilization: 0.65
  vllm_model_len: 4096
  eval_limit: null                                 # full 模式：null；smoke：8
  include_path: Eval/tasks                         # 自定义 task YAML 搜索路径
  hf_hub_offline: true
  lm_tasks:
    - gsm8k
    - hendrycks_math500
    - mmlu
    - truthfulqa_mc1
    - truthfulqa_mc2
    - winogrande
```

`include_path: Eval/tasks` 使 lm-eval 加载 `Eval/tasks/hendrycks_math500.yaml`、`Eval/tasks/ifeval.yaml` 等自定义 task，覆盖内置定义（主要是切换到本地数据集路径）。

### 4.3 执行链

```text
run_full_eval_v2()
  │
  ├── link_eval_models()       # 在 eval/models/ 下为每个 ModelSpec 创建 symlink，
  │                            # 以 spec.eval_task（即 source 名）为目录名
  ├── run_eval_vllm(config)    # Eval/allRunEval_vLLM.py
  │   └── subprocess: lm_eval --model vllm --model_args pretrained=... --tasks ...
  │                            # 每个模型独立评测，结果写到 output_origin_root/
  └── eval2res(config)         # Eval/component/Eval2Res.py
      └── 聚合所有 per-task JSON → target_metrics_results.csv
```

### 4.4 注意：MATH500 scoring bug

`Eval/tasks/hendrycks_math500.yaml` 当前使用 lm-eval 默认的 `process_results`（按 `$...$` dollar-span 提取答案），与 thinking-style 输出不兼容，会稳定产生 0.0。已在 `Eval/eval_math500_full.py` 中实现基于 `math_verify` + `\boxed{}` 提取的修复版本，但尚未接入主流程。MATH500 列的结果在接入修复前不可信，应以 GSM8K 为主要 in-domain 指标。

### 4.5 产物

```text
{exp_root}/eval/
├── eval_config.yaml
├── models/                       # ModelSpec symlinks
│   ├── theta0 -> {merged_model}
│   ├── opd_lmbda1 -> ...
│   └── ...
├── origin/                       # lm-eval 原始 per-task JSON
│   └── {model}/{task}/results_*.json
├── fix/                          # 结果整理
└── csv_results/
    └── target_metrics_results.csv  # Source × Benchmark 汇总，主结果表
```

### 4.6 Gate 评估（`build_eval_selection_gates()`）

读取 `target_metrics_results.csv`，以 theta0 为基线计算：

- `GSM8K_gain = score - theta0_GSM8K`（每个模型相对 theta0 的 in-domain 增益）
- `OOD_lite_penalty_p2 = ||max(0, baseline-score)||₂`（OOD benchmark 下降的 L2 范数）
- `Worst_OOD_lite_drop`（单 benchmark 最大下降）

OPD arm 与最接近 `GSM8K_gain` 的 SFT control 配对（`matched_gsm8k_pairs.csv`），供验收门 Gate A/B 对比。

产物：

```text
{exp_root}/tables/
├── eval_trajectory.csv      # 每个模型的完整指标行（含 gain / penalty）
├── ood_penalty.csv          # 每个模型 × benchmark 的逐条 drop 明细
└── matched_gsm8k_pairs.csv  # OPD arm × matched SFT 配对结果
```

---

## 5. 探针与几何分析阶段

### 5.1 X/S 探针生成（`build_probes_multi()`）

从 `probe_prompts.jsonl` 取前 `n_probe` 条生成探针，写到 `{exp_root}/getslice/inputs/`：

**X 探针**（theta0 生成，冻结，所有模型共用）：

| 变体 | 生成方式 | 用途 |
|---|---|---|
| `X_prompt` | theta0 greedy EOS 生成（temperature=0，conditioned on prompt） | prompt-conditioned 参考分布 |
| `X_bos` | theta0 BOS-start 采样（temperature=0.8，seed=42，unconditional） | 无条件参考分布 |

**S 探针**（按模型区分）：

| 模型 | S 数据来源 |
|---|---|
| theta0 | Qwen3-4B（teacher）rollout |
| opd_lmbda1 / opd_lmbda05 | 各自 student rollout |
| sft_n128 / n256 / n512 / n1024 | 对应大小的 `train_sft.jsonl` 切片 |

所有 rollout 均生成到自然 EOS，`max_new_tokens` 仅作安全上限（full 模式：2048）。

**产物格式**：

```text
{exp_root}/getslice/inputs/
├── X_prompt/x_probe.jsonl    # {output: {text: prompt+completion}}
├── X_bos/x_probe.jsonl       # {output: {text: completion}}
└── S/{source}/numina_math_probe/gamma_s.jsonl
    # OPD/theta0：{question: ..., answer: ...}
    # SFT：{question: ..., answer: ...}（从 messages 提取）
```

### 5.2 GetSlice 交叉矩阵（`run_getslice_cross()`）

对每个 `(model_i, S_probe_j)` 组合和每个 `(model_i, X_variant)` 组合，以 subprocess 调用 `GetSlice/slice.py` 做 SVD 分析。

Cycle 04 在三个层运行：`GEOMETRY_LAYERS = [6, 14, 22]`（Qwen3-1.7B 共 28 层）。

```bash
python GetSlice/slice.py --config {cfg_path}
# cwd: /root/LLM-output-density/GetSlice
```

GetSlice 配置（每次调用独立生成）：

```json
{
  "model": "<merged_model 路径>",
  "save_path": "{exp_root}/getslice/layer{L}/{model_i}/{S_or_X_tag}/",
  "DEV": "cuda",
  "model_seq_len": 512,
  "seed": 42,
  "mode": "s_only_svd",
  "target_layer": <L>,
  "s_nsamples": 16,
  "x_nsamples": 16
}
```

结果文件（`sMat.json` / `xMat.json`）已存在则跳过该组合，支持断点续跑。

### 5.3 几何指标汇总（`build_geometry_tables_cross()` + `build_figures_cross()`）

读取所有层的 GetSlice 输出，计算四个几何指标：

| 指标 | 计算方式 | 含义 |
|---|---|---|
| `effective_rank` | exp(H(p)) where p=softmax(σ²/Σσ²) | 奇异值谱的有效维度数 |
| `spectral_gap` | (σ₁ - σ₂) / σ₁ | 主方向优势度 |
| `log_spectrum_drift` | mean(|log(σᵢˢ) - log(σᵢˣ)|) | S 与 X 奇异值谱的对数漂移距离 |
| `xs_log_spectrum_gap` | mean(log(σᵢˢ)) - mean(log(σᵢˣ)) | S 相对 X 的谱均值偏移方向 |

产物：

```text
{exp_root}/tables/geometry_metrics.csv   # model × layer × metric 展平表
{exp_root}/figures/
├── spectrum_diagonal.png    # 各模型对角线（自己 model × 自己 S）的奇异值谱
├── ood_vs_drift.png         # OOD_lite_penalty_p2 × log_spectrum_drift 散点
└── effective_rank_heatmap.png  # model × layer 的 effective_rank 热图
```

---

## 6. 注册表

每次 `run_from_config()` 完成后，写两条 JSONL 记录：

`run_registry.jsonl`（Schema: `experiments/opd_sft_h1/registry/run.schema.json`）：

| 字段 | 含义 |
|---|---|
| `run_id` | UUID，唯一标识一次训练 |
| `method` | 训练方法，取值见 `ALLOWED_METHODS`（trl_opd_like / sft / cold_start 等） |
| `role_label` | theta0 / opd_lmbda1 / sft_n128 等 |
| `status` | running → completed / failed |
| `pi_mix_lambda` | OPD 的 lmbda 值 |

`checkpoints.jsonl`（Schema: `experiments/opd_sft_h1/registry/checkpoint.schema.json`）：

| 字段 | 含义 |
|---|---|
| `checkpoint_id` | `{run_id}__final` |
| `checkpoint_path` | LoRA adapter 目录（`checkpoint_output/`） |

注：每个 arm 的注册表写在自己的 `{exp_root}/{label}/registry/` 下，不共用。

---

## 7. 产物目录总览

```text
/root/autodl-tmp/cycle04_opd_stability_gain/       # {exp_root}
├── pool/                          # 数据母池（一次性，共享）
├── theta0_cold_start/             # cold-start 训练产物
│   ├── config_theta0_cold_start.yaml
│   ├── checkpoint_output/merged_model/
│   └── registry/
├── opd_lmbda1/                    # OPD arm（lmbda=1.0）
├── opd_lmbda05/                   # OPD arm（lmbda=0.5）
├── sft_n128/ sft_n256/ sft_n512/ sft_n1024/    # SFT controls
├── eval/
│   ├── eval_config.yaml
│   ├── models/                    # symlinks
│   ├── origin/                    # lm-eval 原始输出
│   └── csv_results/target_metrics_results.csv
├── tables/
│   ├── eval_trajectory.csv
│   ├── ood_penalty.csv
│   ├── matched_gsm8k_pairs.csv
│   └── geometry_metrics.csv
├── getslice/
│   ├── inputs/X_prompt/ X_bos/ S/{source}/
│   └── layer{6,14,22}/{model}/{probe}/sMat.json  xMat.json
├── figures/
│   ├── spectrum_diagonal.png
│   ├── ood_vs_drift.png
│   └── effective_rank_heatmap.png
├── selection.json                 # OPD/SFT matched pairs + theta0 基线
└── summary.json                   # 实验元信息 + 产物路径索引
```

---

## 8. 常见问题

**Q：为什么 OPD 训练要用子进程，SFT 不用？**

vLLM colocate 会在训练进程内初始化 NCCL 进程组，无法在同一进程内串行启动多个 colocate 实例。SFT 训练不使用 vLLM colocate，也不需要 teacher 模型常驻，可以在进程内串行调用。

**Q：merged_model 已存在时是否会重新训练？**

不会。`train_opd_arm()` 检测到 `merged_model/config.json` 存在即直接返回，跳过训练和 merge。pool、GetSlice 输出、lm-eval 结果均有类似的 skip-if-exists 逻辑。

**Q：MATH500 为什么全是 0.0？**

lm-eval 的默认 hendrycks_math500 scorer 按 `$...$` 截取答案，对 thinking-style（`<think>...</think>\boxed{}`）输出失效，产生全零。MATH500 当前结果不可信，相关修复在 `Eval/eval_math500_full.py`，尚未接入主流程。

**Q：load_best_model_at_end 为什么 OPD 里禁用了？**

`peft 0.19.1 + transformers 4.57.6` 组合下，vLLM colocate 初始化 distributed 后，peft 的 `_load_best_model` 会走入 `_maybe_shard_state_dict_for_tp` 分支并导入 `EmbeddingParallel`，与当前版本不兼容。SFT 训练无 distributed，可以正常使用。

**Q：eval_limit=None 和 eval_limit=8 有什么区别？**

`eval_limit` 透传给 lm-eval 的 `--limit` 参数，控制每个 benchmark 使用的样本数。full 模式设为 `None`（全量）；smoke 模式设为 8，仅用于验证流程通畅，结果无统计意义。

**Q：为什么 theta0 的 DataSize 标记为 256 而不是 cold_steps × grad_accum？**

cold_steps = n_cold // cold_steps_div = 256 // 4 = 64，grad_accum = 4，DataSize = 64 × 4 = 256，与 n_cold 相等，语义一致（消耗了 256 条 prompt 的梯度信号）。

**Q：GetSlice 配置中的 cwd 为什么要设为 GetSlice/？**

`GetSlice/slice.py` 内部用相对路径导入 `utils/`，必须以 `GetSlice/` 为工作目录运行，否则 `from utils.xxx import ...` 会失败。

---

## 9. Cycle 08：verl OPD 训练实现（loss 形式与 seqlen 归一化）

> 本节描述 Cycle 08 的训练机制变化：OPD 不再用自研的 TRL `DistillationTrainer`（见 §3.1），改用 **verl 的 on-policy distillation 路径**（`verl.trainer.main_ppo` + `distillation.enabled=True`），student=Qwen3-4B-Base、teacher=Qwen3-8B。训练在独立的 `verl` conda 环境（`/root/autodl-tmp/envs/verl`，torch2.9/vllm0.12）；评测/几何仍在 `density` 环境复用 Cycle 07 组件。verl 源码：`/root/autodl-tmp/verl`。

### 9.1 OPD loss 的代码位置与两种形式

实现在 `verl/trainer/distillation/losses.py`（聚合/分发）+ `verl/trainer/distillation/fsdp/losses.py`（FSDP top-k 核）。`distillation.distillation_loss.loss_mode` 选择两类：

1. **单样本 KL 估计量**（`loss_mode ∈ {k1, k2, k3, kl, abs, mse, low_var_kl}`，`use_estimator=True`）
   - 只用 teacher 在**实际采样 token** 上的 logprob，与 student 同 token 的 logprob 做单样本散度估计（走 `core_algos.kl_penalty`）。
   - `k1 = student_logprob − teacher_logprob`（**有符号、可为负**，故下游对 loss 做 clamp）；k2、k3 为 Schulman 低方差估计量。
   - **verl 官方 8B OPD 示例用的就是 `k1`**，配 `topk=64`（topk 在此只决定 teacher 传回多少 logprob 用于带宽/诊断，loss 仍是单样本）。

2. **top-k 前向 KL**（`loss_mode=forward_kl_topk`，`use_topk=True`）
   - 在 **logits processor** 内对 teacher 的 **top-k logprobs** 算前向 KL，**不物化 `[B, L, V]` logits**（V=151936 极大，这是长 CoT 下能跑起来的关键）。
   - 因 top-k 截断使两分布不归一，散度可为负 → 代码 `clamp_min(0.0)`。
   - 额外记录诊断：`overlap_ratio`（teacher/student top-k 重叠占比）、`student_mass`/`teacher_mass`（top-k 内的概率质量）。

### 9.2 两种更新方式：policy-gradient OPD vs 监督 OPD（`use_policy_gradient`）

- **`use_policy_gradient=True`（policy-gradient OPD，示例默认）**：把**逐 token 蒸馏 loss 取负当作优势** `advantages = −distillation_losses.detach()`，喂给 PPO/GRPO 的 policy-gradient loss（`get_policy_loss_fn(policy_loss_mode)`）。即「蒸馏散度越小→优势越大」。`adv_estimator=grpo`。
- **`use_policy_gradient=False`（监督 OPD，arXiv:2306.13649 GKD 式）**：蒸馏 loss 直接经 `agg_loss` 聚合并反传。

两条路最终都经由 `loss_agg_mode` 做 seqlen 归一化（见 §9.3）。

### 9.3 seqlen 归一化（核心：`actor.loss_agg_mode`）

**这是平衡不同 rollout 长度的关键开关**，与 verl PPO 同一机制（`core_algos.agg_loss`）。前向并行算出所有位置的 per-token 散度矩阵 `loss_mat: (B, L)`，对单标量做一次反传；归一化由 `loss_agg_mode` 决定：

| `loss_agg_mode` | 公式 | 谁被等权 | 长序列影响 |
|---|---|---|---|
| **`token-mean`（verl 默认）** | `Σ_所有token loss / 总token数` | 每个 **token** | 长序列按 token 数成比例更主导 |
| `seq-mean-token-mean` | 每条序列先 `sum/len`，再对序列平均 | 每条**序列** | 长短序列同权（对应 OPD 公式里的 `1/|y|`） |
| `seq-mean-token-sum` | 每条序列 `sum`（不除长度），再对序列平均 | 序列平均，但 | 长序列在序列内 loss 更大→更主导 |
| `seq-mean-token-sum-norm` | 同上但除固定 `loss_scale_factor`（默认 max len） | — | 用定长归一 |

- 跨 DP / 梯度累加的正确归一靠 `global_batch_info.batch_num_tokens`（`token-mean` 模式下 dp_size>1 时必需），**避免「各 micro-batch 先 mean 再平均」的累加归一 bug**。
- **注意 opd.md 数学式里的 `1/|y|` 对应 `seq-mean-token-mean`，但 verl 实现默认是 `token-mean`** —— 两者梯度不同，必须**显式设置**，不能默认推断。

### 9.4 与 Cycle 07 SFT 的可比性（必须钉死的设计点）

Cycle 07 SFT 用 TRL `SFTTrainer`，交叉熵默认 **token-mean**。为使 OPD 轨迹与 SFT 轨迹的 dip-and-recovery **公平对比**，verl 侧应：

- 设 `actor.loss_agg_mode=token-mean` 对齐 SFT 的归一化口径；**若选 `seq-mean-*` 必须在 result/ 中显式标注口径差异**，因为它会改变长 CoT 样本的相对权重、进而影响 dip 幅度。
- 另需对齐「step 语义」：SFT 的 step = eff_batch 16 的一次梯度更新；OPD 的 step = 对 `train_batch_size` 个 prompt 的一次 rollout+更新迭代。横轴对齐口径（按梯度步 / 按 samples-seen）需在 `exp/current_experiment_design.md` 固定。

### 9.5 关键配置项小结（Hydra override）

```
distillation.enabled=True
distillation.distillation_loss.loss_mode=forward_kl_topk # ★ 监督 OPD 的正确 loss（见下）
distillation.distillation_loss.topk=32                   # dense target，小 batch 稳
distillation.distillation_loss.use_policy_gradient=False # 监督 OPD（直接反传 KL）
distillation.distillation_loss.use_task_rewards=False    # 纯蒸馏、不混任务 reward
actor_rollout_ref.actor.loss_agg_mode=token-mean         # ★ seqlen 归一化，对齐 SFT
actor_rollout_ref.model.lora_rank=32  lora_alpha=64  target_modules=all-linear
```

> **cycle08 定档（2026-06-30，verl 校验实测）：监督 OPD ⟺ `use_policy_gradient=False` + `loss_mode=forward_kl_topk`。**
> verl 硬性拒绝 `use_policy_gradient=False + loss_mode=k1`（`ValueError: Directly backpropagating k1 loss is
> incorrect since gradient of k1 loss wrt model weights does not depend on teacher log probabilities`）——k1 直接反传时
> teacher 项是常数、梯度只剩 student 项,不是蒸馏。单样本估计量 {k1,k2,k3,…} 只属于 policy-gradient 路径
> (`use_policy_gradient=True`,advantages=−loss.detach())。监督路径要用梯度依赖 teacher 概率的 `forward_kl_topk`
> (top-k 前向 KL,GKD 式,arXiv:2306.13649),正是「逐 token 前向 KL 到 teacher」、SFT 的最干净类比。
> 其它 smoke 已验证项:config 全过;缺失依赖 `TransferQueue==0.1.8`(V1/distillation runner 需要)已补;
> **verl OPD 需 ≥2 GPU**(teacher = 独立 resource pool,绝不与 student 共卡 → 单卡跑不了,2×96G 是硬下限)。
> 待正式跑验证:LoRA 在 distillation 路径生效 + verl LoRA checkpoint 保存布局/体积。
