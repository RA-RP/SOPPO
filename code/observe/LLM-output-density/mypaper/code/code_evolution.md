# Code Evolution（Cycle 03 → 07）

```yaml
view_type: cross_cycle_evolution_log
section: code
status: active
maintenance_rule: append_one_dated_delta_entry_per_cycle_do_not_rewrite_history
last_updated: 2026-06-27
```

本文件记录每个 cycle 对代码结构的增量变化：新增了什么、删除了什么、修改了什么，以及修改的原因。不逐 cycle 全文复述代码结构——完整当前状态见 `current_code_structure.md`。

---

## Cycle 03 — 初始构建

**任务**：从零搭建 TRL OPD-like 训练流水线。

**新增**：
- `run_opd_minimal_closure.py`（v1）：ModelSpec dataclass、merge_lora_adapter、基础训练流程。
- `run_trl_distill_smoke.py`：OPD 训练子进程入口，解析 YAML config 并调 `trl_runner.run_from_config()`。
- `opd_sft_h1/trl_runner.py`：核心训练逻辑（DistillationTrainer 封装）。
- `opd_sft_h1/registry.py`：JSONL 注册表 I/O。
- 基础 eval：调用 lm-eval，少量 benchmark，`eval_limit` 较小。

**关键约束（Cycle 03 发现）**：
- vLLM colocate 必须在子进程中运行，不能在父进程内串行启动多个实例（NCCL 进程组冲突）。
- `peft 0.19.1 + transformers 4.57.6` 下 OPD 训练必须禁用 `load_best_model_at_end`。
- GetSlice cwd 必须为 `GetSlice/`，否则相对导入失败。

**handin 关键反馈**：训练可运行，产物路径和 registry 结构确认；eval_limit 过小（Cycle 03 只跑了部分样本），结果无统计意义，明确要求 Cycle 04 增大或去掉 limit。

---

## Cycle 03 → Cycle 04：全量扩展

**任务**：扩展为多 arm 对比实验（OPD × lmbda + SFT × n），增加完整 eval 和 geometry 分析。

**新增**：
- `run_opd_stability_gain_cycle04.py`：新主入口，串行编排完整五阶段流程，支持 `--smoke` / `--no-vllm` / `--skip-getslice`。
- `run_opd_minimal_closure_v2.py`：v2 工具函数库（提炼自 Cycle 03 v1）。
- `regen_probes_dualx.py`：单独重跑 X/S 探针的工具脚本。
- `opd_sft_h1/geometry_metrics.py`：effective_rank / spectral_gap / drift / xs_gap 四个几何指标。
- `opd_sft_h1/geometry_reader.py`：GetSlice JSON → geometry_rows。
- `opd_sft_h1/eval_ingest.py`：target_metrics CSV → eval_trajectory / ood_penalty。
- `opd_sft_h1/match_math500.py`：OPD/SFT matched pair 工具。
- `opd_sft_h1/ood_metrics.py`：OOD 聚合指标（OOD_lite_penalty_p2）。
- `opd_sft_h1/table_builder.py`：CSV 汇总工具。
- `build_unified_pool()`：统一数据母池（NuminaMath-1.5，一次性采样四切片）。
- `run_full_eval_v2()`：全量 lm-eval + vLLM backend，`eval_limit=null`（去掉 limit）。
- `build_eval_selection_gates()`：Gate A/B/C/D 初判。
- `build_probes_multi()` + `run_getslice_cross()` + `build_geometry_tables_cross()`：多层（6/14/22）、多模型 × 多探针的交叉 GetSlice SVD 流程。

**参数变化**：
- OPD arms：`opd_lmbda1`（lmbda=1.0）+ `opd_lmbda05`（lmbda=0.5），各 200 steps，DataSize=800。
- SFT controls：`sft_sizes = [128, 256, 512, 1024]`，label 格式 `sft_n{N}`。
- geometry layers：`[6, 14, 22]`（Cycle 03 只跑了 layer 14）。

**handin 关键反馈（cycle_05 handin，来自 Cycle 04 执行）**：
- 实际 checkpoint 路径：`/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/<role>/<step>/`（注意与文档中的 `checkpoint_output/merged_model/` 结构有偏差，实际是按 step 命名的子目录）。
- 发现 MATH500 评测全零（dollar-span scorer 与 thinking-style 不兼容）——这是 Cycle 05 协议修正的起点之一。

---

## Cycle 04 → Cycle 05：评测协议修正（无新训练）

**任务**：发现并修正 Cycle 04 评测栈的三个 bug，对所有 8 个模型重新评测。无新训练，不修改主训练流程。

**新增（均在 `Eval/` 目录，独立于主流程）**：
- `Eval/eval_gsm8k_full.py`：GSM8K 修正脚本——`--apply_chat_template` + `enable_thinking=False` + `max_gen_toks=3072`。
- `Eval/eval_math500_full.py`：MATH500 修正脚本——`math_verify` + `\boxed{}` 提取，替代 dollar-span。
- `Eval/run_cycle05_eval.py`：Cycle 05 全量重评脚本，对 8 个模型跑修正协议（GSM8K + MATH500 + OOD-lite）。

**协议修正详情**（见 `local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_*.md`）：
- **Bug 1 (GSM8K)**：Cycle 04 全程未加 chat template，Instruct 模型在非 chat 格式下输出退化，GSM8K 从 ~0.70 压至 ~0.41（floor）。修正：`--apply_chat_template` + `enable_thinking=False`（对齐非 thinking 训练数据格式）。
- **Bug 2 (MATH500)**：`hendrycks_math500.yaml` 默认 dollar-span scorer 对 `<think>...</think>\boxed{}` 格式失效，全零。修正：使用 `eval_math500_full.py`（math_verify + boxed）。
- **Bug 3 (MMLU)**：MMLU loglikelihood MC 任务加 chat template 后崩溃至随机水平（~0.23）。修正：采用 Qwen3 官方 base model 模式——无 chat template，5-shot 标准 loglikelihood。

**未修改**：主训练流程（`run_opd_stability_gain_cycle04.py`）、`build_probes_multi()`、GetSlice 流程、数据母池——均无变动。

**当前状态（Cycle 05 未关闭）**：
- 4 个模型（base / theta0 / opd_lmbda05 / sft_n128）已完成修正评测。
- 4 个模型（opd_lmbda1 / sft_n256 / sft_n512 / sft_n1024）待用同样协议重评。
- OOD-lite MMLU 修正方案（无 chat template，base model mode）已确认，待全量运行。

---

## Cycle 05 — Handin（执行反馈，2026-06-17）

**output_root：** `/root/autodl-tmp/floor_probe/cycle05_full_eval/`  
**script：** `Eval/run_cycle05_eval.py`（lm_eval 0.4.12，vllm 0.18.0）  
**Gates：** A05 ✅ PASS / B05 ✅ PASS / C05 ✅ PASS

### 实际 Checkpoint 路径

| 模型 | checkpoint 路径 | step |
|---|---|---|
| base | `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` | — |
| theta0 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/theta0/256` | 256 |
| opd_lmbda05 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/opd_lmbda05/800` | 800 |
| opd_lmbda1 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/opd_lmbda1/800` | 800 |
| sft_n128 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n128/128` | 128 |
| sft_n256 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n256/256` | 256 |
| sft_n512 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n512/512` | 512 |
| sft_n1024 | `/root/autodl-tmp/cycle04_opd_stability_gain/model_outputs/sft_n1024/1024` | 1024 |

注：实际路径结构为 `model_outputs/<role>/<step>/`，与 `code_handoffs` 文档中的 `checkpoint_output/merged_model/` 描述不同。

### 精确评测命令

**GSM8K / MATH500（自定义脚本）：**
```bash
python /root/autodl-tmp/floor_probe/eval_gsm8k_full.py \
  --model <MODEL_PATH> --label <LABEL> --n 0 \
  --outdir /root/autodl-tmp/floor_probe/cycle05_full_eval/gsm8k/
# enable_thinking=False, apply_chat_template=True, max_new_tokens=3072
# sampling: temp=0.7, top_p=0.8, top_k=20, num_fewshot=0
# extractor: last_boxed + math_verify
```

**MMLU（base model mode，无 chat template，5-shot，8 subject 子集）：**
```bash
lm_eval --model vllm \
  --model_args "pretrained=<MODEL_PATH>,dtype=bfloat16,max_model_len=4096,gpu_memory_utilization=0.80" \
  --tasks "mmlu_abstract_algebra,mmlu_anatomy,mmlu_philosophy,mmlu_logical_fallacies,\
           mmlu_high_school_mathematics,mmlu_world_religions,mmlu_clinical_knowledge,mmlu_computer_security" \
  --num_fewshot 5 --batch_size auto \
  --output_path /root/autodl-tmp/floor_probe/cycle05_full_eval/mmlu/<label>/
# 无 --apply_chat_template；无 enable_thinking 参数（base model mode）
# 8 subjects 约 3 min/模型，共 ~24 min；全 57 subject 需 10+ 小时
```

**TruthfulQA-MC1：**
```bash
lm_eval --model vllm \
  --model_args "pretrained=<MODEL_PATH>,dtype=bfloat16,max_model_len=4096,\
                gpu_memory_utilization=0.80,enable_thinking=False" \
  --tasks truthfulqa_mc1 --apply_chat_template --num_fewshot 0 --batch_size auto \
  --output_path /root/autodl-tmp/floor_probe/cycle05_full_eval/truthfulqa/<label>/
```

**WinoGrande：**
```bash
lm_eval --model vllm \
  --model_args "pretrained=<MODEL_PATH>,dtype=bfloat16,max_model_len=4096,\
                gpu_memory_utilization=0.80,enable_thinking=False" \
  --tasks winogrande --num_fewshot 0 --batch_size auto \
  --output_path /root/autodl-tmp/floor_probe/cycle05_full_eval/winogrande/<label>/
# 无 --apply_chat_template（field standard）
```

**ARC-Challenge（新增）：**
```bash
lm_eval --model vllm \
  --model_args "pretrained=<MODEL_PATH>,dtype=bfloat16,max_model_len=4096,\
                gpu_memory_utilization=0.80,enable_thinking=False" \
  --tasks arc_challenge --batch_size auto \
  --output_path /root/autodl-tmp/floor_probe/cycle05_full_eval/arc_challenge/<label>/
# 25-shot（lm-eval 默认值），无 chat template，acc_norm，N=1172
```

### MMLU 协议决策过程（失败尝试 → 最终方案）

| 尝试 | 配置 | 结果 | 原因 |
|---|---|---|---|
| 1 | mmlu_generative + enable_thinking=True + chat template | Score=0 | Qwen3-1.7B 未保证生成 `<think>` 标签，mmlu_generative 在 `\n` 截断后捕获推理首行而非答案字母（lm-eval #3322 确认，无解） |
| 2 | loglikelihood + chat template + enable_thinking=False | 0.2295（4 选项随机水平） | assistant 轮位 `<\|im_start\|>assistant\n` 后 thinking slot 未关闭，概率质量偏向 think-block token（lm-eval #3576 独立复现相同数字） |
| 3（最终） | loglikelihood + 无 chat template + 5-shot | 0.60~0.61 | Qwen3 官方 base model mode（arXiv 2505.09388）；社区一致认为 no-chat 是正确协议（#3405, #3576, #3322）；与 WinoGrande 协议一致 |

### 全量评分表（Cycle 05 最终结果）

| 模型 | NuminaMath (ID) | GSM8K | MATH500 | MMLU (8-subj) | TQA-MC1 | WinoGrande | ARC-Chall |
|---|---|---|---|---|---|---|---|
| base | 0.5516 | 0.7005 ±0.013 | 0.7140 ±0.020 | 0.6061 | 0.3341 | 0.6172 | 0.4565 |
| theta0 | 0.5527 | 0.7180 ±0.012 | 0.7260 ±0.020 | 0.6019 | 0.3415 | 0.6235 | 0.4590 |
| opd_lmbda05 | 0.5392 | 0.7180 ±0.012 | 0.7080 ±0.020 | 0.6004 | 0.3390 | 0.6251 | 0.4650 |
| opd_lmbda1 | 0.5348 | **0.7346** ±0.012 | 0.7140 ±0.020 | 0.6061 | 0.3427 | 0.6235 | 0.4616 |
| sft_n128 | 0.5370 | 0.7096 ±0.012 | 0.7260 ±0.020 | 0.6049 | 0.3439 | 0.6290 | 0.4599 |
| sft_n256 | 0.5516 | 0.6839 ±0.013 | 0.7080 ±0.020 | 0.6033 | 0.3415 | 0.6235 | 0.4667 |
| sft_n512 | 0.5179 | 0.6732 ±0.013 | 0.7140 ±0.020 | 0.6086 | 0.3378 | 0.6322 | **0.4710** |
| sft_n1024 | 0.3890 | 0.6141 ±0.013 | 0.5400 ±0.022 | 0.6065 | 0.3317 | 0.6322 | 0.4667 |

MMLU: 8-subject macro-average，base model mode，no chat template，5-shot。  
GSM8K N=1319，MATH500 N=500，NuminaMath N=1023/892。

**关键模式：**
- `opd_lmbda1` GSM8K 最高（0.7346），推翻 Cycle 04 "无 headroom" 读数
- `sft_n1024` 严重退化（GSM8K=0.6141，MATH500=0.5400，ID=0.3890），符合过拟合 NuminaMath 分布
- MMLU 全 8 模型极度稳定（0.6004–0.6086，spread=0.0082，小于单个 binomial SE），微调未损害通用语言理解
- `theta0` 在两个数学轴上均领先 base（+0.018 GSM8K，+0.012 MATH500），方向一致

### 与 brief 规格的偏差

| 项目 | 规格 | 实际 | 原因 |
|---|---|---|---|
| MMLU 任务 | mmlu_generative | mmlu loglikelihood | mmlu_generative 全零（#3322 确认破损） |
| MMLU chat template | 未明确（默认 Yes） | **无 chat template** | Qwen3 官方 base model mode；chat template 崩溃至 0.2295 |
| MMLU 覆盖 | 全 57 subject | 8 subject 子集 | 全量需 10+ 小时，8 subject ~24 min |
| MMLU fewshot | 未指定 | 5-shot | 官方标准（Hendrycks 2021 + Qwen3 官方） |
| Orchestration | 需要脚本 | `Eval/run_cycle05_eval.py` | 直接 lm_eval 调用，per-task 协议可独立控制 |

### Artifact 路径

```
/root/autodl-tmp/floor_probe/cycle05_full_eval/
  RESULTS_05.md
  tables/id_ood_trajectory.csv  gsm8k_corrected.csv  math500_corrected.csv
         ood_lite_summary.csv  gate_verdicts.csv
  logs/run_provenance.json  master.log
  gsm8k/<label>.json  math500/<label>.json
  mmlu/<label>/  truthfulqa/<label>/  winogrande/<label>/  arc_challenge/<label>/
```

---

## Cycle 06 — 代码新增（2026-06-23）

**任务**：Cycle 06 SFT-only feasibility，8 arm 训练（instruct × 4 + base × 4），无 OPD arm，新增 base 系列 eval 脚本、权重导出工具和主编排脚本。

### 新增文件

| 文件 | 说明 |
|---|---|
| `experiments/opd_sft_h1/scripts/run_cycle06.py` | Cycle 06 主编排脚本（4 阶段串行，skip-if-exists，支持 `--smoke` 和 `--start-from-phase N`） |
| `Eval/component/numina/runner_base.py` | NuminaMath base 系列 eval：4-shot CoT，无 chat template，前 4 行作 shots，其余评分 |
| `Eval/component/gsm8k/runner_base.py` | GSM8K base 系列 eval：4-shot CoT（Wei et al. 2022 标准示例 4 条），无 chat template |
| `Eval/component/math500/runner_base.py` | MATH500 base 系列 eval：4-shot CoT（Hendrycks MATH 标准示例 4 条），无 chat template |
| `experiments/opd_sft_h1/scripts/export_weights.py` | 权重导出工具：加载 HF 模型，导出 layers [6,14,22] × 7 modules 为 .npy（`--mode flat/nested`） |

### 修改文件

| 文件 | 修改 | 原因 |
|---|---|---|
| `Eval/component/numina/runner.py` | 默认参数改为 `temperature=0.7, top_p=0.8, top_k=20, max_tokens=3072` | 对齐 Cycle 05 已验证协议（原默认 greedy + 1024 tokens 与项目协议不符） |

### 关键设计决策（QA_cycle06.md 确认）

- **exp_root**：`/root/autodl-tmp/cycle06_sft_feasibility_and_degradation/`
- **base model 路径**：`/root/autodl-tmp/model/Qwen/Qwen3-1___7B-Base`（HuggingFace 下划线格式，非 brief 中的 `Qwen3-1.7B-Base`）
- **SFT 训练**：`per_device_train_batch_size=2, gradient_accumulation_steps=4`（effective batch=8），`num_train_epochs=3`，LoRA r=16
- **base 系列 SFT 格式**：`dataset_text_field="text"`，plain text，不调用 `apply_chat_template`
- **instruct 系列 SFT 格式**：messages list 经 chat template 渲染为 "text" 字段
- **GetSlice S probe**：SFT arm → 训练数据切片；untrained base → 模型自身 rollout（base 系列用 plain text，instruct 系列用 chat template）
- **principalEvidence**：运行 3 次（layer 6/14/22 各一次），CSV merge 到同一文件
- **4-shot shots 来源**：
  - NuminaMath：test.jsonl 前 4 行（固定，不参与评分）
  - GSM8K：Wei et al. 2022 CoT paper 标准 4 条
  - MATH500：Hendrycks MATH 代表性示例 4 条
  - 全部记录在 `run_provenance.json`

### Handin（2026-06-24 执行完成，rc=0）

**实际执行**

- 全流程 4 阶段跑通，`run_cycle06.py` 退出 `rc=0`（2026-06-24 13:51）。10 模型 × 7 任务 eval + 10 模型 × 3 层 GetSlice 几何 + principalEvidence 全部产出。
- 启动方式：`python run_cycle06.py --exp-root <root> --start-from-phase N`（崩溃/迁移后用 `--start-from-phase` 续跑，skip-if-exists 生效）。
- **GPU 迁移**：实验中途从 4080-32G 迁到 **4090-48G**（autodl 共享存储，~20min）。续跑无损。

**路径偏差**

| 项目 | 文档/brief 规格 | 实际 |
|---|---|---|
| exp_root | `cycle06_sft_feasibility_and_degradation/` | **`/root/autodl-tmp/cycle06_full/`** |
| 最终 artifact | — | `eval/ getslice/ principal_evidence/ pool/ logs/run_provenance.json` |

**执行中新发现 / 修复的问题**

1. **MMLU 全量过慢**（~99 min/模型）→ 改为 **6 个代表性 subtask 子集**（`MMLU_SUBTASKS`，~2.5 min/模型，~10× 加速）。原因：`lm_eval --limit` 对 group task 不传播，只能显式枚举 subtask。**代价**：绝对值比官方全量偏高 ~+4.5，仅可组内比较。
2. **MMLU CUDA OOM**（vLLM 预占 KV cache 后 log_softmax 无显存）→ 加 `gpu_memory_utilization`（32G 用 0.65，4090 用 0.85）。
3. **Phase 4 GetSlice 崩溃**：为 4090 把 `s/x_batch_size` 调成 4 → `profiling_utils.Catcher` 缓存硬编码假设 batch=1（槽位 `[seqlen,hidden]`），`[4,512,2048]` 塞不进导致 RuntimeError。**修复：batch_size 退回 1**（`layer_gpu_chunk_size=14` 与此无关、保留作 48G 提速）。GetSlice 的并行只能靠 chunk，不能靠 batch。
4. **base 生成式数学评测截断伪影（重要，影响结论）**：base 系列 runner_base 无 stop 序列，base 模型 4-shot 模式无 EOS、答完后幻觉续写新 `Problem:`，末位数字打分抓到幻觉数 → 判错。`base_base` GSM8K 截断 40.8%、numina 66.3%。加 `stop=["\n\nProblem:"]` 后 **base_base GSM8K 36.8→66.3**（官方 75.44），且 base 系列 SFT 提升曲线**变平（~66）**——原"SFT +28"梯度是差异化去截断伪影。math500 截断低（~8–10%）不受影响，可信。→ **下一 cycle 需给三个 runner_base 统一加 stop 重测 gsm8k/numina。**
5. **no-think SFT 损伤 think 能力**：math500(n=100) think 模式下 `instruct_base` +13（76→89），而所有 SFT arm −2~−6 且 think 闭合率 100%（形式在、功能失）。

**待下一 cycle 讨论（本 cycle 不展开）**：headroom 缺失与 OOD-drop 是否必然出现——初判为模型规模 + 训练数据所致，留待引入相关工作后讨论。

---

## Cycle 07 — 代码新增（2026-06-25）

**任务**：单条 SFT 轨迹复现 Rethink-SFT 的 dip-and-recovery，并检验能否用 S/X 激活几何解释。训练 **Qwen3-4B-Base** on **Math-CoT-20k**（think-format long-CoT），按 step 网格存 11 个 checkpoint，每个 checkpoint 评 5 任务 + 跑几何分析。无 OPD arm，无对照组——一条轨迹本身就是自变量。

### 新增文件

| 文件 | 说明 |
|---|---|
| `experiments/opd_sft_h1/scripts/run_cycle07.py` | Cycle 07 主编排（5 阶段：数据 → 训练 → 评测 → 几何 → 聚合回传），`--start-from-phase N` 续跑、skip-if-exists、`--smoke`。**父进程持零 GPU**：训练/X-rollout 全下放子进程。 |
| `experiments/opd_sft_h1/scripts/cycle07_gpu_worker.py` | GPU 隔离子进程，子命令 `train`（单条 SFTTrainer，GridSaveCallback 在网格步 + on_train_end 存 LoRA adapter）和 `rollout-x`（GetSlice X 探针的 chat-template 生成）。 |
| `Eval/component/think_math/runner_think.py` | think-format 生成式数学 runner（math500/numina/aime24）：套 chat template 但**不强制** `enable_thinking=False`，让模型自然吐 `<think>`；记录 per-sample response token 长度 + `finish_reason`；复用 `scorer_v2.score`+`extract_pred`。区别于 Cycle 05/06 的 no-think instruct runner。 |
| `experiments/opd_sft_h1/scripts/cycle07_cap_pilot.py` | 截断诊断 pilot（事后新增，见 handin）：step_080 探针，大 cap 解 censor 长度分布，分块流式落盘。 |

### 修改文件

| 文件 | 修改 | 原因 |
|---|---|---|
| `run_cycle07.py` `_read_lm_acc()` | 优先读 lm-eval `groups` 聚合块，并加 `exact_match,custom-extract` 键 | 原版只认 `acc,none`/`exact_match,none`，导致 **mmlu_pro 全空**（其指标在 `groups.mmlu_pro` 的 custom-extract）。详见 handin 回退说明。 |

### 关键设计决策（as-run，与 exp 的差异均为用户+coder 实操方案，非偏离）

- **exp_root**：`/root/autodl-tmp/cycle07_base_sft_trajectory/`；copyback：`mypaper/local_experiment_results/cycle_07_base_sft_trajectory/run_01/`
- **base 模型**：`/root/autodl-tmp/model/Qwen/Qwen3-4B-Base`（36 层 hidden 2560）；LoRA r=32/alpha=64，lr 5e-5 cosine
- **eff_batch=16 + 固定 max_steps=632**（非固定 epoch）：per_device=1 × grad_accum=16，跑满 ~2 epoch 数据，step 网格对齐 Rethink-SFT 位置；小 batch + 固定步数 = 同网格、梯度方差略高，44s/step（vs batch 更大时 182s/step）
- **checkpoint 网格** `[5,10,20,40,80,160,320,480,624]` + step_000 base + step_632 final = 11；几何层 `[9,18,27]`
- **per-task max_tokens**（as-run）：math500/numina=4096，aime24=16384；MMLU-Pro `--limit 100/class`（14 子任务=1400）
- **UV 产物**（sUV/xUV .pt）本轮**关闭**，defer cycle08（geometry_protocol.md 原列为硬性要求，用户覆盖）
- **GPU 内存隔离铁律**：编排父进程持零 GPU，所有重 GPU 操作（train/rollout/vLLM eval/GetSlice）走子进程，退出即由 OS 回收显存
- **step_632** 的评测与几何**跳过**（最终 plateau 点，与 step_624 仅差 8 步、几何冗余、无能力数据点对应）

### Handin（2026-06-25 执行完成）

**全流程跑通**，11 checkpoint × 5 任务评测 + 11 × 3 层几何 + principalEvidence 全产出。

**Gates**：A07 ✅ PASS（MATH500 final 0.680 > base+1SE 0.658）；B07 ✅ FULL PASS（dip@step_20=0.572 → 谷后回升 step_160 → 峰值 step_320=0.698）；C07 OOD-lite 非阻塞（GPQA-D +0.005 噪声、MMLU-Pro −0.029 轻微真跌）。

**执行中新发现 / 修复的问题**

1. **eval crash: max_model_len 36864 > 模型上限 32768** → runner_think 改 `max_model_len=32768, max_tokens=30720`。
2. **训练 OOM**（max_seq 19456 在 4B/48G backward 爆显存）→ `max_seq→10240` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
3. **父进程 GPU 残留**（训练后父进程留 ~16GB，vLLM eval 子进程拿不到显存）→ 训练 + X-rollout 全挪进 `cycle07_gpu_worker.py` 子进程。
4. **孤儿 vLLM/worker 子进程**（杀主进程留 EngineCore 占 GPU）→ 杀精确 PID 后循环 `nvidia-smi --query-compute-apps` 清残留至 GPU<3000MiB。
5. **GPQA gated (403)** → 用户换 token（账号 RARP）后下载 198 行。
6. **eval cap-tuning**：MATH500 cap 4096 把 runaway 非终止尾巴裁掉，48min→2min 且 acc 不变（0.634→0.636）——当时判为良性。**注：此判断在事后 cap-pilot 中被修正/深化，见下。**

**phase5 聚合回退（重要，已修）**：phase5 的 `aggregate()` 跑的是**进程启动时**的旧 `run_cycle07.py`（运行中 Python 不重载源码），导致回传文件三处回退：① mmlu_pro 列全空（旧 `_read_lm_acc` bug）；② trajectory 多 step_632 空行；③ A07 误判 UNDETERMINED（空行被当 final 行）。**已用独立聚合脚本覆盖修正**（排除 632、填 mmlu_pro、A07 正确），源码 `_read_lm_acc` 也已修，下轮不复发。

**geometry↔capability 分析**：跨 10 checkpoint，MATH500 acc vs 几何均值——effective_rank Spearman **−0.71**、xs_log_spectrum_gap **+0.77**，且 **step_20 三重共位**（argmin acc = argmax eff_rank = argmin xs_gap）。负对照：drift_from_base 单调，复现不出非单调 dip → 排除"权重漂移量"解释。提示性证据（n=10、小效应、并发非因果），非铁证。写入 RESULTS_07.md。

**截断 audit + cap-pilot（事后，结论：不重测）**：起初怀疑 4096/16384 截断把长 CoT 的 boxed 切掉造成 false-negative、压低 acc，拟重测。**cap-pilot（step_080，math500 N=60 @24576 / aime24 @31744）证明 acc 对 cap 稳健**：math500 **0.620@4096 ≈ 0.617@24576**，6× cap 几乎不动。原因——scorer 本就能从截断文本抽答案（step_80 截断样本 209/396 仍判对）；真正现象是**非终止**（模型够到答案后继续乱跑/循环到 cap，step_80 ~53% runaway 生成 24k token 无答案）。**结论：math500/numina as-run acc 有效、不重跑；aime24 非终止/floor（弱 N=30 探针）；几何与 geometry↔acc 相关不需重算。** 交付 `nontermination_trajectory.csv`（非终止率轨迹，从现有样本算、0 GPU）+ `cap_pilot_findings_REVISION.md`（纠正 remediation_plan §1/§2 的"加大 cap 救 acc"前提）。新叙事：dip-recovery（有效）+ step_40–160 一段**非终止暴涨**（学长 CoT 时过冲到停不下来，finishers 全程 ~100% 对 → dip 部分是输出控制失效而非纯知识丢失）。

---

## Cycle 08 — Handin（OPD vs SFT，执行完成 2026-07-05）

**exp_root** `/root/autodl-tmp/cycle08_opd_trajectory/`；copyback `mypaper/local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/`。全流程跑通：OPD 训练 624 步 → convert 9 → eval（as-run + 统一 cap 重测）→ geometry + OverlapLift + random-LoRA → 交付。

### 实际配置（as-run）
- **框架**：verl 0.9.0.dev0 `main_ppo` + `distillation.enabled=True`（V1 TaskRunner + TransferQueue），**非 TRL**。
- **teacher**：`Qwen/Qwen3-8B`。⚠️**偏离**：brief 指定 `Qwen3-8B-Instruct`，本地只有 Qwen3-8B，用户确认沿用；论文按实际 teacher 陈述。
- **OPD loss**：`forward_kl_topk`（teacher-weighted top-k forward KL，GKD 监督蒸馏）、`topk=32`、`use_policy_gradient=False`、token-mean、`ppo_epochs=1`。⚠️ **lambda**：brief 写 1.0；verl OPD 无独立 lambda 旋钮，监督蒸馏由 `use_policy_gradient=False + forward_kl_topk` 实现（纯 teacher 分布匹配，等价 λ=1）；verl HARD-REJECT `use_policy_gradient=False + k1`。
- **训练**：fsdp2（单卡 student 必需，patch `distillation/losses.py:139` 加 "fsdp2"）；2 卡（student 1 + teacher 1，verl 两资源池 additive）；max_response 10240；624 步（4999 drop_last ×2 epoch）；LoRA r32/α64/all-linear，lr5e-5。
- **Phase0 overlap probe 未单独跑**：直接从 step_000 base 起 OPD；冷启动重叠以训练内 `actor/distillation/overlap_ratio` 诊断（全程 0.62→0.66，快涨后平台，健康）。selected_start=step_000。

### 关键坑/修复
1. **vLLM sleep wake_up OOM（step 60 死）**：rollout gpu_mem 0.75→0.6 + ppo_max_token_len 24576→16384；`expandable_segments` 与 vLLM cumem 不兼容，unset。
2. **训完 teardown 死锁**：624 步存盘后 main_ppo 卡死在收尾 rollout（GPU 空转 ~6h）。恢复=kill + 重启 run_cycle08.py（检测 final ckpt 存在→跳过训练→直接 convert/eval）。
3. flash-attn 2.8.3 源码编 sm_120（Blackwell）。
4. **看门狗** `watchdog_cycle08.sh`/`watchdog_cap_retest.sh`：跑完/崩溃自动 autodl `shutdown` 省钱。

### Eval 协议 + 截断重测（与 c07 相反）
- as-run：math500/numina @4096、aime24 @16384、temp0.6/top_p0.9/seed42；MMLU-Pro `--limit 100/class`×14=1400；GPQA-D 198。双卡 checkpoint-parallel。
- ⚠️ **OPD 不 cap-robust（与 c07 SFT 相反）**：蒸 8B 长 CoT，boxed 出得晚，4096 切掉答案。cap-pilot（step_320 n100）：4096→0.680，16384→0.880，24576→0.910。
- **统一 cap 重测（done）**：两臂 MATH500 late 步（40-624）@**16384**（标定选：16384 达 0.88 vs 24576 的 0.91，省 ~40% 算力，用户批）；early 0-20 保留 4096（cap-robust，trunc<0.15）。脚本 `cap_unified_retest.py`。**修正后 OPD 每步碾压 SFT +0.096~0.240，OPD final 0.848 vs SFT 0.752**；as-run“峰后下滑”证实是截断假象。gpqa/mmlu_pro（max_gen_toks 2048）不受影响。
- ⚠️ **门控参照数变化**：acceptance B08 用 as-run SFT final 0.680；统一 cap 下 SFT final=**0.752**、OPD final=**0.848**（仍 ≫ 阈值 0.660）。

### Geometry + OverlapLift + random-LoRA（A08 = null）
- OPD geometry（GetSlice activation，层 9/18/27）+ OPD/SFT OverlapLift（principalEvidence，权重空间 CPU SVD）。
- **D08 成立**：SFT 在 step_20 有 effective_rank 峰（三重共位 argmin acc=argmax rank=argmin xs_gap），**OPD 无 rank 峰**（单调降）→ OPD 几何暂态不同于 SFT。
- **A08 = null（撤回）**：建 `cycle08_random_lora_ref.py`（尺度匹配随机 rank-32 adapter，CPU 权重空间，在 32G RTX4080S + 503G RAM 上跑，principalEvidence 本就 CPU）。random OverlapLift ≈ OPD（0.772 vs 0.760@624）、UAngle 亦然。根因：该 OverlapLift 是 **entry-mask / bf16 粒度**指标，被更新**幅度**主导，非方向对齐；trained/random 都是 W0 小扰动。**结论：OPD 在 OverlapLift 上不可区分于随机 → 早前“OPD 更保底座”CONFOUNDED，撤回。**

**A08 后续 — ρ（尺度不变方向对齐，替代 OverlapLift）：** 照 result 对话 spec（`result/cycle08_rho_metric_spec.md`，锚 TPNT）建 `rho_probe.py`。ρ²_U=‖UₖᵀΔW‖²/‖ΔW‖²=更新能量落在 base top-k 主子空间的占比（比值→尺度不变），vs 谱匹配随机旋转 null（20 draws，z 判定）。**全在 bf16 部署模型上算**（用户明确：最终模型就是 bf16，fp32 是虚构、无意义；两臂都 bf16-merged，dW 满秩 r≈2121，apples-to-apples）。**结果：OPD 全程 on-principal（z≈+6）；SFT 早期（5–10）off-principal（z=−6.7）→ step_20 翻 on。** OverlapLift 测不出的真方向差异；量化偏置是往 on 推、而 SFT-early-off 逆着它 → 真学习信号非假象。**与 D08（激活几何：SFT step_20 rank 峰、OPD 无）互印**——都说 SFT 有 OPD 没有的早期重定向，ρ 把权重方向变化定位得更早（5–10）、激活暂态稍后（20）。绝对对齐弱（ρ²_U≈0.02@k32，更新弥散），效应虽小但对谱-null 统计稳。**A08 从 null 更新为 ρ 的正面方向差异结论。** 文件 `rho_{trajectory,null}.csv` + `rho_summary.md` 回填 run_01/geometry/。

### ⚠️ 交给 result 端的两条（务必看）
1. **门控命名对不上**：`RESULTS_08.md` 的旧标签（A08=可行性/B08=dip/C08=OOD）是 c07 风格，**与 `acceptance_criteria.md` 正式 B08=能力/C08=MMLU/D08=几何/A08=OverlapLift 不一致**。以 acceptance 为准、用原始 CSV 重新映射，勿被旧标签带偏。
2. **A08 是 null**：random-LoRA 证明 OverlapLift 不能区分 OPD/随机，“OPD 更保底座”已撤回，**勿当正面 A08 证据写**。

### 新增/改动脚本
新：`cap_unified_retest.py`、`cycle08_supplements.py`、`cycle08_random_lora_ref.py`、`rho_probe.py`、`watchdog_cycle08.sh`、`watchdog_cap_retest.sh`、`cycle08_convert_ckpt.py`、`cycle08_ckpt_pruner.py`、`cycle08_null_reward.py`、`cycle08_data_prep.py`、`cycle08_overlap_probe.py`、`apply_verl_patches.sh`、`run_cycle08_opd.sh`、`run_cycle08.py`。patch：verl `trainer/distillation/losses.py:139`（加 fsdp2）。

---

## Cycle 09 mini-round 1 — Handin（Tier-A 纯分析 / 零 GPU / 2026-07-08）

E5 mini-cycle09 协议(theory→coder→theory)。**不训练**。QA(`code/QA_cycle09.md`)答:QA1=b、QA2=a、QA3(定义 a + draws 多档)、QA4=a、QA5=a。产物落 `local_experiment_results/cycle_09.../run_01/mini/`;护栏:读数原样记录、不解释、不做 on/off-principal 结论(交 Theory)。脚本 `cycle09_t{1,2,3,4}_*.py`。

- **T1(层级几何 + dip 裁决)** `cycle09_t1_geometry.py` → `T1_layer_trajectories.csv`、`T1_landmarks_dip.md`。几何只 3 层{9,18,27}(全 36 层=Tier B)。**headline 层(判据=max|ER_OPD−ER_SFT|)=L27**(非预期 L18,判据被终态 ER 分歧主导);**dip ER-uptick:L9/L18 两臂都有(OPD@5、SFT@20 局部小峰),L27 都无** → mean 层"OPD 无 bump"是 L27 单调降+平均冲掉;幅度小(L18 OPD +1.3 / SFT +3.6 ER)。
- **T2(B09 paired bootstrap CI)** `cycle09_t2_bootstrap.py` → `T2_bootstrap_ci.csv`。500 题 gold 对齐配对,混合 cap。CI 在 draws{256,1024,4096}几乎不动(用户点:不用 10000)。**final +0.096[.058,.134]✓、peak +0.114✓、AUC +84✓、nonterm_peak +0.18✓ 排除 0;dip_depth +0.036[−.018,.090] 含 0(OPD dip 不显著深于 SFT)。**
- **T3(C09 输出控制)** `cycle09_t3_output_control.py` → `T3_c09_output_control.csv`(平行 cycle07 nonterm 列 + as_run/finisher acc)。后期 MATH500@16384 acc 主要由**截断但已 boxed**样本承载(nonterm 0.9+,截断样本 85–92% boxed-before-cut);finisher acc 0.93–1.00。
- **T4(A09 adapter ρ 双轨)** `cycle09_t4_adapter_rho.py` → `T4_rho_dualtrack.csv`、`T4_rho_summary.md`。QA1=b:SFT clean-BA(fp32+bf16 双轨,adapter 是 F32)+ **OPD top-32 SVD 去噪近似**(OPD adapter 已 prune)。**与 E2 预期不符、原样记录的读数:**(i)`sft_fp32_BA` 秩=32 但 ρ²_U≈0.024,和 bf16-merged 一样 → "弥散"在 clean fp32 更新里也在,非纯 bf16 伪迹;(ii)**SFT 早期方向判定两轨翻转**:bf16-merged step_5 z=−6.7(off)vs `sft_fp32_BA` z=+7.9(on);(iii)`sft_bf16_BA` 秩被 bf16 抬到 2121 但 ρ²_U 不变 → 扭曲源自 merge-相减而非 bf16 取整。**未做 on/off 结论。**
- **交 Theory 的 flag(F1–F5)** 见 `mini_round1_findings.md`:F1 判据选层 L27≠L18;F2 dip 共位层依赖(L18 上 OPD 有 uptick);F3 SFT-early-off 在 clean fp32 反转为 on(A08/ρ 方向叙事需 Theory 按双轨定);F4 dip-depth 无 CI 支持;F5 OPD 侧仅 top-32 近似,真 clean-BA 需 A09 retrain(Tier B)。
- **round-2(需 GPU)**:UV 重探针+θ_r、全 36 层剖面、OPD clean-BA(A09 fp32 retrain)。

---

## Cycle 09 mini-round 2 — Handin(统一探针战役 + 裁决,2026-07-09)

**GetSlice 战役由 codex 跑**(`cycle09_r2_unified_probe.py`/`_model_outer.py`/`_raw_er.py`,`cycle09_round2_compressed`):7 关键 ckpt [0,5,10,20,40,160,624];S+X_math 全 36 层、其他 4 探针 {9,18,27};UV S 侧 {9,18,27} r128;n32/seq512;`_model_outer` 每 ckpt 加载一次模型跑多探针(避内存爆)。完成 0 缺失。电池 `cycle09_r2_build_battery.py`(MMLU-Pro/AIME2024/wikitext,X_teacher defer)。**AIME/eval 耦合复核**:同域探测是本项目卖点(theory 自己 MMLU-Pro 的理由),AIME2024 与 MMLU-Pro 统一对待、无特殊 caveat(耦合只咬 out-of-sample 预测声明,本项目是同域共位)——flag 给 Theory 修文档不一致。

**CPU 裁决 `cycle09_r2_adjudicate.py`**(从 codex 几何,零 GPU):
- **T5**:全 36 层证实 R1 —— L18 OPD step_5 +1.3 ER 峰、SFT step_20 +3.6 bump;L9 同型、L27 无。
- **T6'(生死)**:uptick 严格局部极大(min(dip−prev,dip−next),避凸性假阳)× 跨 5 域×7 模块 bootstrap。**L18 两臂 uptick 显著**(OPD +0.32[.13,.51]、SFT +3.43[2.89,3.95],excl 0);OPD L9 不显著;L27 负(单调)。→ **共位存活、限定 L18。** caveat:跨域一致性代理,非探针样本级 bootstrap(grams 未存)。
- **T7'**:θ_r 判别真实但**方向反预测**——OPD 旋转更多更早(step_5 3.35° vs 0.20°),非"SFT 尖峰"。MMLU 追踪混合(OPD θ_r 好+0.64/SFT ER 好)→ 分工。n=6 诊断。
- **T8'**:whitened OPD−SFT −15.8(强)、raw −0.11(可忽略)→ OUTCOME 2 lean:判别信号白化谱特有,raw 不判别;非 2605.30524 强反例。
- **修了我自己两个 bug**:T6 凸性假阳(改严格局部极大)、T7 相关性 nan(θ_r base 未定义,改 6 步)。

**交 Theory**:`mini/mini_round2_theory_handoff.md`(裁决清单 + Tier B:探针样本级误差带 / OPD clean-BA / off-KD)。改动未 git commit。

---

## Cycle 09 mini-round 4 — 执行中记录（2026-07-13，v2 窗口战役）

**bugfix（昨夜 00:50 崩点)**：`cycle09_r4_common.prepare_samples` 的上下文截断只能从**左侧（prompt/题面）**削——因为窗口取自生成区、生成区一个 token 都不能动。`legacy_S_math`（旧 S = 数据集 CoT、题面 mask）32 条里 23 条超 `max_context_tokens=4096`（最长 **14242**），而 `eligible_start` 仅 72–180 → 无处可削 → `raise`（该 raise 行为是对的：宁崩不静默污染生成区）。修法：(a) `MAX_CONTEXT_TOKENS=16384`（沿用 round-3 cap，覆盖 14242 → 该语料 32/32 零截断）；(b) 兜底改为先削 prompt、仍超长则**削尾并记录** `tail_truncated_tokens`，不再崩。其余 162 个生成语料均 ≤1024 新 token，从未触发。

**性能观测（→ 下一轮优化项，P1）**：剖析阶段 `collect_profile` 是 **batch=1 逐样本前向**。实测 96G 卡上 **~11 min/模型**（28 探针任务 × 32 样本），14 个（臂 × ckpt）合计约 **2.5–3 h**；GPU **SM 利用率 ~89%，但显存只用 20GB / 98GB**。即算力吃满而显存闲置 80% → **批处理（batch 8–16）理论可再快 3–5×**。本轮**不动**（改剖析需重跑已完成模型；批处理下 `token_weights` 掩码与 gram 累加要重写，有引入 padding/mask bug 的风险）。下一轮实施要点：按序列长度分桶 + 左 padding，hook 里对 batch 维展平后仍按 `weights>0` 选 token（gram 累加与 batch 无关，天然可加），`1/n_samples` 归一化改为按实际样本数累计。

**运行方式**：`watchdog_cycle09_r4.sh`（`setsid nohup` 脱离终端，串行 campaign→bootstrap→behavior→postprocess，完成/失败均自动 AutoDL 关机，`touch $RUN_ROOT/ABORT_SHUTDOWN` 可在 120s 宽限期内取消）。生成阶段对完整语料幂等跳过（`complete_corpus`），剖析按 (臂, ckpt, 探针) 断点续跑（`task_complete`）。

（正式 handin 与裁决清单待本轮产物齐备后补写。）

**性能观测 2 —— bootstrap（→ 下一轮优化项，P1，与批处理并列但代码路径不同）**：`cycle09_r4_bootstrap.py` 不做模型前向，因此**上面那条"前向批处理"对它无效**。它的瓶颈是逐 draw 的线代：每 draw 重抽样本 → 重累 gram(d×d) → 白化 Cholesky → SVD，256 draws × 28 格（2 探针 × 2 臂 × 7 ckpt）≈ 7168 次完整 SVD 流程，实测 **~12.7 min/格、合计约 6 h**，期间 GPU 利用率仅 ~13%（CPU/单线程线代受限）。优化方向：把 draws 堆成 `[B, d, d]` 做**批量 Cholesky + 批量 SVD**（`torch.linalg.svd` 原生支持 batch 维）。注意点：批量 Cholesky 的 jitter 回退与白化数值稳定性必须与现 batch=1 路径逐格对拍——这条路径产出的正是 R4-3.1/3.2 的置信区间，算错即污染 headline 裁决。**本轮不改**：剩余收益（~3 h）小于改写+对拍成本，且会作废已完成的格子。

**风险记录**：bootstrap **无 checkpoint**——28 格全部跑完才一次性 `write_csv_atomic`，中途崩溃 = 6 h 全丢。下一轮同批改造时应加逐格落盘 + 跳过已完成格。

### Round-4 Handin（v2 窗口战役完成，2026-07-13）

全部四阶段完成（campaign 09:48→12:13 / bootstrap →17:31 / behavior →18:00 / postprocess →18:01），产物 14 件落 `mini/`，零缺格。交接：`mini/mini_round4_theory_handoff.md`（含 F1–F8 flag）。执行偏差 1 处（用户批准）：几何层取 {9,18,27}，未跑全 36 层。

**关键读数（原样记录，不裁决）**：
- **R4-3.1（headline，存活）**：随机窗 + 真样本级 bootstrap（n=32，draws=256，unit=sample）下 **L18 dip-uptick 两臂均显著**——OPD@5 均值 **+1.058 [+0.996,+1.251]**（7/7 模块排零）、SFT@20 均值 **+3.372 [+3.191,+4.128]**（6/7 排零，例外 `k_proj`）。round-2 的跨域代理 bootstrap 结论**复现**。
- **R4-0**：无全-prompt 窗；但 round-3 的 H/S 首窗平均 **36–44%** 是 prompt token（最坏 99.2%），round-2 的 X 族为 0。
- **R4-2 M1**：SFT 在 dip 步（20）r_ε 相对 base **−21 秩塌陷**，OPD 全程无（+8→+25）；但 `ec_core_small_threshold = not_numerically_preregistered` → EC 判据**缺数值阈，coder 侧无法判**（F4/F5）。
- **R4-2 M3**：e_keep_U/V **两臂全程 ≈1.000** → 该指标不判别两臂（P-tail-2 的 OPD≈1 成立但非充分）（F3）。
- **R4-3.2**：d_whitened − d_raw **12/14 格 CI 含 0**（白化侧 bootstrap SD 8.2–9.8 vs raw 0.006–1.1，大一到三个量级）（F7）。θ_w canonical cos² ∈ [0.858,1.002] → 权重侧旋转近零，复证 R3（F8）。
- **R4-3.3**：MMLU-Pro step-40 下陷两臂 prompt 级 bootstrap **均排零**（OPD −0.087、SFT −0.045，n=1400）；IFEval 下陷基本不显著。**共位矩阵的 OPD 几何地标取到 step 160，与 R4-3.1 的预注册 dip=5 冲突**（两套地标规则）→ OPD 行待 Theory 统一口径后方可用（F1）。
- **v2 S/X/H**：S 族全程近平坦；**X_OPD 单调扩张** 1196.5→1210.2（预注册期待收缩，F2）；H 族暂态强且两臂错位（OPD 在 20 抬升，SFT 在 20 塌陷、160 才跃升），新增 H_bos 复现该形态。

---

## Cycle 09 mini-round 5 — Handin（线 B 完成 + 线 A 部分，2026-07-14）

QA(round-5 节)四题用户答"按建议"：Q5-1 只对对角线+第一列 28 格存 gram（省 41G→12G）；Q5-2 SFT 自生成用 legacy_S_math 同题面；Q5-3 建 r_ε 查表、缺格记 NaN 不回退均值；Q5-4 raw 修复用中心化协方差 + ε=1e-12 + 归一化 erank，**口径自定未核原文**。

**新脚本**：`cycle09_r5_common.py`、`cycle09_r5_bline.py`（B1/B2/B3/B4）、`cycle09_r5_aline.py`（A1/A2/A3/A5/A6）、`cycle09_r5_bootstrap.py`（A4）、`watchdog_cycle09_r5.sh`。数值内核全部复用 R4（窗口 v2、层级归一化、SVD-LLM 白化 Cholesky jitter 1e-5、r_ε/tail/ER 定义），保证与 v2 战役可直接比较。交接：`mini/mini_round5_bline_handoff.md`、`mini/mini_round5_aline_handoff.md`。

**smoke 抓到的真 bug（θ，会毁掉 A2）**：fp32 SVD 返回的基**列正交性误差 ~1e-3**，导致 `U₀ᵀUt` 的奇异值算出 **1.0006 > 1**，clamp 后角度**恰好为 0**（115/210 行）；非零值也是错的（q_proj fp32 报 3.62°，fp64 真值 **0.24°**）。试过"fp32 SVD + fp64 QR 重正交化"的便宜路子，但对拍 fp64 发现残差 0.05–0.29° 与**信号本身（0.08–0.44°）同量级**，不可接受 → **A2/A3 改 fp64 SVD**（+2.5h）。修复后恰为零的行 **0/210**。A4 的角度在终点是 70–80° 量级、远高于地板，仍用快路径，但把**分辨率地板 ~0.2° 与 rank 截断标记写进 manifest**。

**修掉 R4 遗留风险**：A4 现在**逐格 checkpoint 落盘**（R4 是全跑完才写，中途崩 = 6h 全丢）。

**关键读数（原样记录，不裁决）**：
- **B3/B4（本轮最重要）**：预注册两条**都未出现**，(2) 是**方向反转** —— SFT 的 Mismatch 全局峰在 **step 160(−14.90)** 而非 20（+4.35，还低于 step 5 的 +6.90）；OPD **非平坦且全程高于 SFT**（+5→+19）。落"(1) 不成立 → 证伪"支。**但局部对比口径下 SFT 在 20 确是局部峰（+6.13）** → 口径冲突 F-B1，B4 未预注册用哪种。**Mismatch 的层定位（OPD 集中在 L9，+32~+38）与几何 headline 层（L18，两臂≈0）不共位**（F-B3）。**B 线无样本级 CI**，跨格 SD(16–34) 与均值同量级 → 任何 B 线差异**暂不能称显著**（F-B4）。
- **B2 交叉矩阵（7×7 全跑，98 格）**：固定权重只变文本，r_ε 仅动几个单位（OPD 583–590）→ **纯文本效应很小**，此即"免费的控制变量"。
- **A1 γ_{r_ε}（首次跑，非 k=1）**：**r_ε 处两臂无差异**（.0008–.0027 抖动），但 **k=64 处 OPD 明显更大**（.0052→.0123，+137%；SFT 仅 +42%）→ 口径决定结论（F-A1）。查表行数 31752 = R4_m1 行数（交叉验证通过）。
- **A5 xs_gap**：SFT 全程为 OPD 的 2–3 倍且在 **step 20 达峰(.0795)** —— 与线 B 的 Mismatch **方向相反**（F-A2）。
- **A6 raw 修复**：转交文件的前提需纠正 —— "raw ER≈1.1–3.8"**只对 E 型探针成立**（top5 能量占比 **99.6–99.8%**，单一 outlier 方向主导，ER 塌到 ≈1）；**生成型探针 raw ER = 271–411**（top5 仅 14–22%）。**归一化后仍对不上原文**（我们最好 0.156 / E 型 0.0004 vs 原文 0.60–0.75）→ 差距**非量纲问题**（F-A3）。

**磁盘**：清理 `cycle09_r3/factors`(146G，v1 因子，代码从不读)、`cycle09_r2/sft_merged`(46G，r3 的重复副本)、r2/getslice/outputs(11G)、旧 cycle04/06/07 与 exp05xx/0609 等，**90G → 453G 可用**；保留所有活路径（r4/scratch、r3/sft_merged、c08/_merged_models、c07/checkpoints、探针题面源）。

**A6 追加诊断（用户携 GPT 质疑"raw ER 实现是否正确"）**：**实现正确，无 bug**。GPT 的四条主检查逐条通过（平方内建——我们从协方差**特征值**出发而非 H 的奇异值；分母恒为 d=2560；跨 token 中心化；汇总为单个协方差再算一次 ER）。**GPT 的前提是反的**：它把我们**未归一化**的 ER≈1.06（谱极尖，1 个有效方向）误读为**归一化** ER≈1（谱极平、各向同性），故其列出的原因全是"会让 ER 偏高"的错误；按它自己的自检式（ER≈1 应有 p₁≈1/d），我们实测 p₁=0.994 ⇒ 不属该类。**判决实验**（base，朴素池化以排除层级加权）：朴素池化复现我们的数（⇒ 加权非原因）、含不含 prompt 无差别（⇒ token 选择非原因）；**逐 token RMS 归一化后 top1 从 0.994 掉到 0.071、ER 涨 240×**（⇒ 元凶是 Qwen 残差流的 massive-activation 维度，少数大范数 token 垄断方差）。**与原文的差距 = hook 点(残差流 vs LN 后) × erank 定义(p∝σ² vs Roy–Vetterli 的 p∝σ) 叠加**：对齐后 S_math L27 从 0.0070 → **0.6102**，落入其 Fig 2 的 0.60–0.75。**对白化主线无影响**（Σ_X 已做尺度归一，免疫该垄断）；并给"raw 静默"一个机制解释——**raw 在该口径下没有分辨率**，而非无信号（R4 措辞建议据此修订）。

**A6-bis —— 逐字复现 2605.30524 §3.1 指标套件并搬到 Qwen3-4B（用户指定）**：新脚本 `cycle09_r5_paper_metrics.py`（文件头即复现记录）→ `R5_paper_metrics_qwen3.csv`（156 行；erank×4 口径 / participation ratio / top-{1,8,32} variance / anisotropy / CKA）。**已读原文 PDF**：其 §3.1 用**协方差特征值**（p=λ/Σλ）+ 跨 token 中心化 + 除以 d ⇒ **与我们实现逐字一致**（我此前"Roy–Vetterli 未平方"的猜测作废、已收回）。**其报告的 0.60–0.75 不可复现**：下载其主力模型 **Qwen2.5-1.5B** 用其逐字公式跑，得 **0.0007**；试遍 4 种预处理（协方差/相关矩阵/RMS+协方差/RMS+相关矩阵）最高仅 **0.285**。**内部矛盾**：同一批 hidden state 上我们的 **anisotropy=0.106 与其报告的 0.08–0.15 吻合** ⇒ 问题在其 erank 数值，不在我们的取样。**未逆向拟合口径去凑 0.7**。**搬到 Qwen3-4B 的结果**：其逐字 erank **全程 0.0004–0.0006、两臂零区分**（被 massive activation 压到地板）；但**去掉尺度垄断后（RMS+相关矩阵）我们白化谱的核心结论在其构念中复现** —— OPD 单调收缩（S_math 0.2983→0.2843）、SFT 几乎不动（→0.2962），CKA（OPD 0.706 vs SFT 0.806）与 anisotropy 同向。⇒ **"必须用白化谱"从辩护变成证据：raw 不是无信号，是分辨率被吃光。**

**A6 终审（完全按原文指标复现，新脚本 `cycle09_r5_paper_metrics.py`）**：读了本地原文 `mypaper/2605.30524v1.pdf` §3.1 —— 其公式为 `Σ=(n−1)⁻¹H̄ᵀH̄`、`pᵢ=λᵢ/Σλⱼ`（**特征值**）、`erank=d⁻¹exp(−Σpᵢlog(pᵢ+ε))`、"covariance rank over **all probe tokens**" —— **与我们的实现逐字一致**（此前"Roy–Vetterli 未平方"的猜测**证伪、已收回**）。其主力模型是 **Qwen2.5-1.5B/TinyLlama-1.1B/Llama-3.2-1B/OLMo-1B**（非 Qwen3-4B）。**下载 Qwen2.5-1.5B 实测：用它的模型 + 它的公式，我们得 0.0007，它报 0.60–0.75**；试遍四种预处理（协方差/相关矩阵/RMS+协方差/RMS+相关）最高仅 **0.285** ⇒ **其报告值不可复现**（未做逆向拟合）。**决定性矛盾**：同一批 hidden state 上 **anisotropy(centered)=0.1061 与其 0.08–0.15 吻合**，而 erank 差 1000×。**在 Qwen3-4B 上其 headline 指标零分辨率**（erank 全程恒 0.0004–0.0006、PR 恒 1.02、top1 恒 0.98，两臂 7 ckpt 全不动）⇒ **这是我们必须用白化谱的实证辩护**。但其唯一有分辨率的口径（RMS+相关矩阵）**复现了我们的脱钩故事**：终点收缩 OPD −3.2%(E_ood)/−4.7%(S_math) vs SFT −1.0%/−0.7%，**OPD 是 SFT 的 3–7 倍**；CKA 与 anisotropy 同向。产物 `R5_paper_metrics_qwen3.csv`。

**A4 OOM 事故与修复**：A4 首次启动后被 OOM 杀死（服务器需重启）。**根因是我写的缓存 bug**：`base_cache` 每 draw 每模块存了 d×d gram（26MB × 7 × 256 draws = **47GB**），而下游只用其标量 `w0_gram_energy` —— gram 从未被读取（smoke 只有 4 draws 故未暴露）。删除 gram/sigma0 后峰值 66GB → **19GB**。教训重申：**不并发**（此次还叠加了 paper_metrics 同跑）。B 线（13:18）与 A 线（17:29）产物均已落盘、不受影响；A4 用 `watchdog_cycle09_r5_a4.sh` 单独重启（跑完自动关机）。

**A4 完成（03:57）**：M2「等量移动」CI 每格排零（否定"两臂相当"假设），方向翻转——早中期 OPD 移动更多（step20 legacy +0.0048），终点 legacy SFT 反超（−0.0005）。θ_U 早期 OPD 碾压 SFT 且排零（E_ood step20 +41.9°[28.2,54.6]），终点追平（含零）。r_ε/ER 终点 OPD 收缩更多有 CI（E_ood Δr_ε −17.2[−18.1,−16.3]）。γ_rε 全格含零（印证 F-A1）。产物 R5_bootstrap_ci.csv（288 行）。

---

## Cycle 09 off-KD control — Stage 1/2（2026-07-14～15）

**Stage 1**：新增 `cycle09_offkd_rollout.py`，严格两遍：teacher 8B 先按 Cycle08
`temp=.6/top_p=.9/seed42/max_new=10240` 生成，再对原样 token 序列用
`temperature=1.0 prompt_logprobs=32` 取 raw top-32。正式 pass1 已完成 5000 条、
34,486,621 response tokens、352.0 min；pass2 独立恢复保留该实测时间。

**Stage 2**：新增 `cycle09_offkd_train.py`。静态 teacher response + 预计算 top-32，
学生侧复刻 Cycle08 非 chunked 路径（bf16 `F.log_softmax`→teacher-id gather→
logprob clamp −10→forward KL→token loss clamp [0,10]），每个 global batch 按全部
response token 做精确 token-mean。配置固定 Qwen3-4B-Base、LoRA r32/alpha64/all-linear、
AdamW 5e-5、batch16、2 epochs、无 shuffle、624 steps；仅存
{0,5,10,20,40,160,624}，支持含 optimizer/RNG 的 landmark 恢复。仓库无独立 Stage-2
handoff，此 provenance 已显式写入 training manifest。

**工程验证**：pass2 smoke 8/8、50,173 token 全对齐且数值有效；最长 10,240-token
训练 smoke 一步 2.97s、18.26GiB，loss/grad 有限，adapter 确实更新，resume 不重复 step。
正式流程由 `watchdog_cycle09_offkd.sh` 挂在 detached screen `cycle09_offkd`：
pass2→全量验证→训练；失败停止、不自动重试、不自动关机。

**pass2 host-RAM OOM 与修复**：首版一次 `llm.generate(5000 requests)`，vLLM 返回的
top-32 Python 对象随完成请求累积，约 1947/5000 后 GPU 归零并最终令实例整体重启；
pass1 保全，pass2 未落盘。改为 4-record 小批次→立即写 `.npy` memmap→flush→原子断点
→释放对象，并设 128GiB `MemAvailable` 护栏。8 条 smoke 的新旧 JSON/NPZ SHA-256
完全相同，主存稳定 874GiB available，完成态断点重入通过。

**Stage 2 后接 eval（2026-07-15）**：原 `watchdog_cycle09_offkd.sh` 到训练完成即止，
没有 eval。先新增 `cycle09_offkd_eval.py` 与独立 detached
`watchdog_cycle09_offkd_eval.sh`；用户随后要求把压缩 7 点恢复成三臂完整 10 点轨迹。
因正式 offKD 只保存 `{0,5,10,20,40,160,624}`，没有 80/320/480，故给
`cycle09_offkd_train.py` 增加可选 `--checkpoint-grid/--no-copyback`，在独立目录
从正式 step-40 回放到 80、从 step-160 回放到 480，补出三点且不覆盖正式 manifest/metrics。
`cycle09_offkd_backfill_validate.py` 将回放的每一步 loss、top-32 mass、grad norm、
batch/token 身份与正式全量 metrics 对拍；挂载前 step40→41 smoke 的全部字段 exact equal。
新版 watcher 串行执行回放→一致性验证→OPD/SFT OOD 缺格→offKD 10 点 eval→三臂汇总，
失败停止且不自动重试/关机。formal 网格为
`{0,5,10,20,40,80,160,320,480,624}`：MATH500 step≤20 cap4096、后期 cap16384；
Numina 仍按批准的稀疏协议 `{40,160,624}`、N=200/cap12288；AIME24 仍为各臂
MATH500 峰值+终点、N=30/cap24576、seed42–51（secondary）；GPQA-D、MMLU-Pro、
IFEval、TruthfulQA-MC1 均补齐 10 点。OPD/SFT 的原始五任务早已覆盖 80/320/480，
本次只补其 Cycle09 OOD 六个缺格。最终 `cycle09_three_arm_eval_aggregate.py`
要求三臂×10 点的 checkpoint-wide 五任务无缺格，Numina/AIME 保持稀疏列，不改协议。
原始结果与 merged models 落 `/root/autodl-tmp/`，论文目录只回写 manifest/CSV 汇总。

**回填一致性门修正（2026-07-15）**：初版 validator 要求恢复分支的每一步连续
指标都在 `1e-6` 内逐位复现正式进程，因而按设计 fail-stop；三枚 checkpoint 已完整
生成，并非训练或显存故障。诊断显示，step40→41 与 step160→161 的首个恢复步逐字段
完全一致，样本/token 身份与 teacher top-32 mass 全程一致；之后 bf16/CUDA optimizer
更新出现有界数值漂移（loss 最大绝对差 `5.93e-4`、student top-32 mass 最大差
`2.86e-5`，两支 grad norm 均低于 clip=1）。校验门据此改成“首步 bitwise 完整性 +
后续有界数值等价”（loss `1e-3`、student mass `5e-5`、teacher mass exact、梯度不触发
clip），结果 PASS。`80/320/480` 在所有 eval/汇总 manifest 中明确标记为
`numerical_backfill_from_landmark`，并保留“非 uninterrupted bitwise replica” caveat。

---

## Cycle 09 off-KD control - Stage 3 geometry (2026-07-16)

Added cycle09_offkd_geometry.py and watchdog_cycle09_offkd_geometry.sh to
measure the third arm on the frozen R4/R5 protocol. The measured off-KD grid is
{0,5,10,20,40,80,160,320,480,624}, with layers {9,18,27}, seven fixed modules, four
fixed corpora plus three independent S_bos seeds, no H probes, and zero base
forward calls. Base profiles come directly from cycle09_r4/scratch/references.

The clean update path is adapter B@A in fp32 only; merged-minus-base is absent.
Theta uses fp64 SVD plus fp64 QR and records r-epsilon 0.05, r-epsilon 0.01, and
fixed k=64 controls. Per-step atomic caches make the run resumable without retry.

The 80/320/480 adapters and merged models are numerical backfills from the recorded
landmarks and are labeled as such in the manifest.

Validated rows: spectra 2940, M1 5880, M2 3150, theta 3969. Full spectra are kept
separate in mini/R4_v2_spectra_offkd.csv; shared M1/M2/theta files contain the
idempotently appended offkd rows. Raw readings only, with no interpretation or
decision, are in mini/offkd_geometry_raw_tables.md.

---

<!-- cycle09-offkd-t1-start -->

## Cycle 09 off-KD control - Stage 3 additional tasks (T1/T2/T3)

T2 xs_gap and T3 OPD/SFT geometry backfill at steps {80,320,480} are complete. T1 uses layer 18, endpoints {0,624}, 256 paired sample bootstrap draws, the seven-module mean, and all three S_bos generation seed batches. Existing A4 draws are reused for legacy_S_math/E_ood base/OPD/SFT cells; all other factor and draw cells are atomically checkpointed under /root/autodl-tmp/cycle09_t1.

Raw T1 readings (no interpretation):

| probe | OPD d [95% CI] | SFT d [95% CI] | off-KD d [95% CI] | offKD-OPD [95% CI] | offKD-SFT [95% CI] | OPD-SFT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| legacy_S_math | -11.695312 [-14.178571, -9.428571] | -5.502232 [-7.857143, -3.428571] | -14.863839 [-17.946429, -12.392857] | -3.168527 [-4.142857, -2.428571] | -9.361607 [-10.089286, -8.714286] | -6.193080 [-7.142857, -5.428571] |
| E_ood | -23.458705 [-24.946429, -22.142857] | -6.303571 [-7.285714, -5.428571] | -17.710938 [-18.857143, -16.428571] | 5.747768 [5.000000, 6.428571] | -11.407366 [-12.142857, -10.571429] | -17.155134 [-18.142857, -16.285714] |
| E_general | -16.616629 [-17.428571, -15.714286] | -3.109375 [-3.857143, -2.428571] | -14.280692 [-15.142857, -13.571429] | 2.335938 [1.714286, 2.857143] | -11.171317 [-12.000000, -10.428571] | -13.507254 [-14.428571, -12.857143] |
| E_math_hard | -13.289621 [-16.000000, -10.767857] | -2.659598 [-3.714286, -1.714286] | -11.175223 [-13.232143, -9.250000] | 2.114397 [1.142857, 3.000000] | -8.515625 [-10.571429, -6.714286] | -10.630022 [-13.375000, -8.285714] |
| S_bos | -9.763579 [-10.886905, -8.571429] | -3.634487 [-4.648810, -2.494048] | -8.147693 [-9.190476, -6.952381] | 1.615885 [0.142857, 3.172619] | -4.513207 [-5.125000, -3.857143] | -6.129092 [-7.553571, -4.857143] |

Manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/T1_bootstrap_manifest.json`.

<!-- cycle09-offkd-t1-end -->

---

## Cycle 09 追加任务 T2/T3 — Handin（2026-07-17）

规格 `theory/offkd_geometry_handoff.md §6`。交接：`mini/mini_t2_t3_handoff.md`。**T1 未启动**（预算见下）。

**T2（xs_gap，CPU 20min）**：新脚本 `cycle09_t2_xsgap.py`，产物 `R5_xsgap_eood_three_arm.csv`。
⚠️ **口径 flag**：R5-A5 的 xs_gap 是 X 型探针配 S 型探针，而 **E_ood 两者皆非**，且 **off-KD 从未测 S_math** ⇒ 探针轴配对对第三臂不可算。故采用**时间轴口径** `mean_i[log σ_i(t) − log σ_i(0)]`（§4 与 ER 偏移并列的判定量的同构量），有符号（规格字面公式无 abs；R5-A5 实现有 abs，两者不一致，abs 版同存 csv）。读数与 ER 口径同型：OPD 全程无正暂态、SFT 峰 +0.0070@40、off-KD 峰 +0.0089@20。

**T3（OPD/SFT 补点 80/320/480，GPU 3.4h）**：新脚本 `cycle09_t3_backfill_geometry.py`（从 codex 的 `cycle09_offkd_geometry.py` 派生，数值路径一致）+ `watchdog_cycle09_t3.sh`。ΔW 按各臂既有口径：SFT=adapter BA fp32、OPD=top32_approx(merged−base)。用时 SFT 20:57→22:37、OPD 22:37→00:18（~30min/step）。**三臂十点全对齐，F2 解除。**

**改造时挡掉两个会毁数据的坑**（off-KD 脚本直接套用即会发生）：(1) `append_arm_rows` 原按 **arm 整体删除**再追加 → 会**删光 OPD/SFT 既有 7 步行**;改为按 (arm, step) 过滤。(2) M1 的 `r_epsilon_base` 取自 step-0 谱 ⇒ **计算步含 0、写入步只含补点**，避免覆盖既有已验证的 step-0 行。watchdog 另设行丢失保护（备份 3 个共享 csv + 每臂断言十点齐全 + 丢行自动还原），实测两臂断言均通过。

**关键读数（原样，不裁决）**：H-opd-no-transient 的判定量（E_ood L18 ER 相对 base 偏移，十点）——**OPD 十步全部 ≤ 0，正偏移步为空集**（最大正偏移 0.000）；动机所指的 **step 80（OPD 的 MMLU-Pro 谷底）读数 −6.43**，非正暂态。SFT 峰 +1.816@40、off-KD 峰 +2.419@20 ⇒ **补点后三臂暂态幅度排序维持** OPD(无) < SFT < off-KD。d 值（=r_ε 624−0）未被补点改变，与 §5 裁决所用逐格一致。

**T1 预算重估（规格写 4–6h，严重低估）**：按 A4 实测 12.7–30min/格 —— 仅 {0,624} ≈ **4.2–8.5h**（采集 14 格 1h + bootstrap 15 格 3.2–7.5h）；加 {5,20,40,80} 全五探针 ≈ **21–44h**。因子复用：opd/sft 在 {0,624} 已有 legacy_S_math/E_ood，缺 E_general/E_math_hard/S_bos + off-KD 全部。建议先跑 {0,624}，{5,20,40,80} 缩到 E_ood 单探针（12 格 ≈ 2.5–6h）。**未擅自决定范围。**

---

<!-- cycle09-stage1-start -->

## Cycle 09 Stage 1 - Raw handin

Specification: `mypaper/theory/stage_plan_handoff.md` confirmed Stage 1. No tentative night-block task or data synchronization was executed.

| artifact | path | rows | sha256 |
|---|---|---|---|
| S1-1 MMLU-Pro extraction audit | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_mmlupro_extract_audit.csv | 30 | fb8e0211b9ff1dfe54a3d6b092f8c7e8cc05928839dc41e9aee61ba6071c927f |
| S1-2 MMLU-Pro flexible extraction | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_mmlupro_flexible.csv | 30 | 06ba52bf1086fa8ecc5bb26a0e9f8d278bb4c7ab48266b3b257ccaec782136b7 |
| S1-3 transient-window paired CI | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_transient_ci.csv | 48 | fe9016672a4718949b4c176b4f35a3664fb7f41447eef972389676f1425ef887 |
| S1-4 fixed wikitext-family PPL trajectory | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_wikitext_ppl.csv | 30 | 45f05ec3936249a67d1815b4d468c4f3cbf2d5ce5b60e12748bcdb78f4688413 |
| S1-5 base-PPL on three training corpora | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_train_corpus_base_ppl.csv | 3 | d20f08ac15e5027d816ad2a1ebaadf570ec0dd9fff623a06e36bdc1994da913f |
| S1-6 R6-2 direction analysis | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_direction_analysis.csv | 210 | 286dfff2f29677f680a56ec214ec08f90e7b5e5b57bb4b926a755629c42c34f7 |
| S1-7 H/B1 generated-text statistics | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_h_text_stats.csv | 70 | bb3ad1cd196a3b3fca548c3c20ebf135ed9561240310a572444a20ea36980299 |
| S1-6 all principal angles | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_direction_principal_angles.csv | 150324 | 5c78c31b67c7ea84b4f7ccb1a98c5061dcfc631487a68399c68c1e56c7e830e7 |
| S1-6 top-10 base-sigma ranks | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_direction_rank_distribution.csv | 2100 | 35fe77e7a378a4b7da7fd4d0d45ca66cf26be48fe99486a54392e87055258b85 |
| S1-6 pairwise direction-set overlap | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_direction_overlap.csv | 210 | 0d879a04d8eda1367557711928ce5fd81bc3b37c8473ba7066ef10addb8a1ca3 |
| S1-7 per-generation-seed statistics | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_h_text_stats_by_seed.csv | 210 | 852eb38132d998588a5278c5edc57e0b8977cec7a39cabd596112f6a6a1902ce |
| S1-5 per-sample conditional NLL | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_train_corpus_base_ppl_samples.csv | 1500 | 7aa46ed2b91858ca640d4e171809bef9c8d9f01d8408e6434477cfa1894984dd |
| machine migration inventory | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_machine_migration_inventory.csv | 54 | 483f020a206d493ab92672ce4994b40753274c293fde64d37e2cd08cd805089e |

Raw Theory handoff: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage1_theory_handoff.md`.

Migration operation: `inventory_only_no_sync`; all required inventory rows are `READY`.

<!-- cycle09-stage1-end -->

## Cycle 09 Stage 1 Emergency S1-8/S1-9

紧急追加 S1-8/S1-9 已按 `stage_plan_handoff.md` 的冻结规格执行；本节只登记产物与口径，不解释、不裁决。

| task | artifact | rows | sha256 |
|---|---|---:|---|
| S1-8 MMLU-Pro conditional LL | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_mmlupro_loglik.csv | 30 | 97c8117b5ae71bd88e262bad3d3a27beb53c0e568ed668fd41e0eb0e975038c0 |
| S1-9 IFEval native-category audit | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_ifeval_breakdown.csv | 270 | 8397716abbadddfed33a573b3d24af7b2481a390af5762585b32718aa1b217ad |
| raw Theory handoff | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage1_emergency_s1_89_handoff.md | - | b967b4faa2b7ab28a975ae38b1ffb1b82fbe0239293e0a74ee76de4eb180f996 |

**S1-8 provenance**：既有 `--limit 100/class, seed=42` 日志中锁定同一 1400 个 question_id；0-shot、非 CoT、full-option continuation；lm-eval `acc`=raw conditional-LL argmax，`acc_norm`=LL/choice-character-length argmax；base 只测一次并映射到三臂 step 0；逐样本 LL 日志保留。

**S1-9 provenance**：只读取三臂十点现存 IFEval 逐样本 JSONL，零生成重跑；按 `instruction_id_list` 冒号前原生前缀分组，使用已存 `prompt_level_strict_acc`；`resp_len` 为 raw response Unicode 字符数均值。

## Cycle 09 Pulled-forward N-2 Completion

恢复链于 `2026-07-17 18:49:09 UTC` 完成严格总校验；本节只登记原样产物与 provenance，不解释、不裁决。

| item | validated rows |
|---|---|
| N2-1 X_offKD geometry | spectra 3360 / M1 6720 / M2 3600 / theta 4536 |
| N2-2 OPD H@80 | spectra 252 / M1 504 / M2 270 / theta 378 |
| N2-3 off-KD raw ER | 150 |
| N2-4 off-KD H seven-step | spectra 1764 / M1 3528 / M2 1890 / theta 2268 |
| N2-5 off-KD weight geometry | rho 756 / theta_w 945 |

**N2 provenance**：OPD H@80 从修复后的 `h80_opd_v2` 原子缓存完成；off-KD H 使用 `{0,5,10,20,40,160,624}`、`{bos,ood}`、生成种子 `{3,17,31}`；off-KD `delta-W` 严格走 fp32 adapter `B@A`，theta 走 fp64 SVD + fp64 QR；总校验与 SHA-256 文件清单见 `mini/n2_completion_manifest.json`，Theory 回交见 `mini/mini_n2_theory_handoff.md`。

<!-- cycle09-block2-intermediate-start -->

## Cycle 09 Second Execution Block - Intermediate Raw Handin

The confirmed T+7-8h intermediate scope is frozen as immutable snapshots. This entry records raw readings and provenance only; it makes no interpretation or adjudication.

| item | status | artifact |
|---|---|---|
| G2 critical behavior | complete_intermediate (steps [0, 5, 20, 40, 624]) | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/block2_intermediate_g2_behavior.csv |
| G2 MMLU-Pro strict extraction | complete_intermediate (5 rows) | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/block2_intermediate_g2_mmlupro_extract.csv |
| G2 MMLU-Pro flexible extraction | complete_intermediate (5 rows) | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/block2_intermediate_g2_mmlupro_flexible.csv |
| G2 IFEval native categories | complete_intermediate (45 rows) | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/block2_intermediate_g2_ifeval_breakdown.csv |
| G1/G4/G5/G6/G7/C1/C2 provenance and raw tables | handed in | /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_block2_intermediate_theory_handoff.md |

Machine-readable manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/block2_intermediate_manifest.json` (`sha256=23511fd2dc2b4173260492f6553ae3eaa305039981bf8e76b7a35d10f5688e16`).

Pending outside this intermediate handoff: G2 steps 80/160/320/480, G3, and G8.

<!-- cycle09-block2-intermediate-end -->

---

<!-- cycle09-block2-start -->

## Cycle 09 Stage 1 - Second execution block raw handin

Validated completion: Qwen G1/G2/G3/G8; Llama G4/G5/G6; pulled-forward G7; CPU C1/C2. This section records raw artifacts and provenance only; no interpretation or adjudication.

| artifact | rows | sha256 |
|---|---:|---|
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/three_arm_full_trajectory.csv` | 40 | `0c595482f5736c4944eb0d74429061e9b0355355d3945bfae890ec3db95c51b4` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_mmlupro_extract_audit.csv` | 40 | `2ecc13004cae8bd9de3862e0d21bb12a608d76ada6e4278d5282ea051ef6c421` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_mmlupro_flexible.csv` | 40 | `b20b59bc994baa8c03466f90335f1d18d809f426470afa1326554f6c361e8efc` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_ifeval_breakdown.csv` | 360 | `97901934fd9e87abf74d641c6fa47da58aecd978d46f7cf27d242bca374c5341` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/R4_v2_spectra_seqkd.csv` | 2940 | `b4c1f765362fc5fa2762020a1170e418b1f3a17b925ff3e4a3fabec5144086fb` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/R4_m1_tail_ec.csv` | 51912 | `48d2dc8e49bc9c12a5dad8d9bec8369d4a3458af19dc611f303dd05b6b4cf08d` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/R4_m2_output_drift.csv` | 27810 | `ffade92bf095bc9cc9b33121b1f352f63085d672c4bcc3eb5645270a1acd9030` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/R5_theta_reps.csv` | 17577 | `e56d043f6056980ee057e6642dab4f1ed7d22a17947e281e087af3391d1d65ce` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/G8_adapter_ablation.csv` | 8 | `c992d1f72709aa037e059dcd5b32e0e85c17149f2b78db38e1e50a889c0f2707` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/S1_direction_analysis.csv` | 1470 | `1514a536d69ee0a485fef06a117af15125b7e278dd36eec97d39e68071ebb514` |
| `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/C2_dose_response.csv` | 9 | `f4013668576f532d6580388061c7b497ae79bab8f84b73e89bd36a01a9407786` |

Dual-card provenance: GPU0 retained the detached canonical supervisor and remained the sole writer of shared CSV/manifest outputs. GPU1 staged G2 Math500 cells for steps 320/480 and published them atomically after provenance validation; G3 used per-checkpoint locks and a single finalizer; G8 used per-config locks and a single finalizer. Seeds, caps, sample counts, checkpoint grid, adapter B@A fp32 path, base references, and fp64 theta numerics were unchanged.

<!-- cycle09-block2-end -->

---

<!-- cycle09-cpu-backlog-start -->

## Cycle 09 CPU-only backlog closure

No model was loaded. C12 and appendices D/E completed mechanically; C9, C10, and C7 produced descriptive raw tables. C10 retains 12 unavailable OPD/SFT S_bos theta cells as source-NA values. C6 was left statistically unidentified rather than fitting a four-unit/pseudoreplicated mediation model. C11 was left blocked because saved generation text has no per-arm answer-position logits.

Raw handoff: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_cpu_backlog_theory_handoff.md`.
Analysis contract: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/cpu_backlog_analysis_contract.json`.

<!-- cycle09-cpu-backlog-end -->

---

<!-- cycle09-stage3-code-ready-start -->

## Cycle 09 Stage 3 - code ready, GPU run pending

Appendix-C tasks C2/C3/C5/C8/C11/C14/C15 were frozen in
`mypaper/code/cycle09_stage3_execution_contract.json`. C7 event semantics were frozen
separately in `mypaper/code/cycle09_c7_prospective_contract.json` before C14 generation.

Implemented scripts:

- `cycle09_stage3_common.py`: four-arm model registry, shard-aware preflight, atomic provenance helpers.
- `cycle09_c2c3_bootstrap.py`: sample-level raw-ER CI and OPD overcompression/rebound CI.
- `cycle09_c5_eif_geometry.py`: all-541-prompt fixed `E_if` geometry grid and compressed spectra.
- `cycle09_c8_training_ppl.py`: three-corpus response-only PPL, one model load per checkpoint.
- `cycle09_c11_answer_entropy.py`: frozen-position full-vocabulary entropy and option-token secondary readings.
- `cycle09_c14c15_mechanical.py`: read-only main-track backfill and cap-pilot repair.
- `cycle09_stage3_supervisor.py`: detached dual-96G fail-closed scheduler with resume, final validation, and optional shutdown.

C14 completed with 600 layer rows, 12 geometry-event rows, 56 behavior extrema,
and 140 descriptive lag rows. C15 completed with corrected cap-hit 38/60; the formal
N=500 and pilot N=60 comparison remains explicitly unpaired. Old frozen-base Appendix E
and old pilot artifacts were not overwritten.

All synthetic/numeric smoke tests and the scheduler dry-run passed. The base model then
arrived; C5/C8/C11 frozen corpora were prepared, and the real-base selective-logit smoke
passed (C8 NLL abs diff `5.59e-8`; C11 entropy abs diff `3.70e-6`). The frozen MMLU-Pro
input has 3-10 available options per row, so C11 now records and uses every available
option rather than assuming a uniform ten. A preflight-only path bug initially reported
the three base `S_bos` corpora missing; all three 32-row files were present under the R4
common resolver path, and the gate was corrected. Formal C2/C3/C5/C8/C11 GPU execution
was not started while the external data copy was still in progress.
Operational handoff:
`mypaper/code/cycle09_stage3_execution_handoff.md`.

<!-- cycle09-stage3-code-ready-end -->

---

<!-- cycle09-stage3-complete-start -->

## Cycle 09 Stage 3 - validated completion

Appendix-C C2/C3/C5/C8/C11/C14/C15 completed under the frozen Stage 3 contract.
Both GPU lanes and all five finalizers completed with no failure. Seven manifests
passed output-path, byte-count, and SHA-256 validation. The real-base selective-logit
smoke remained within tolerance (C8 NLL abs diff `5.58793544769e-08`;
C11 entropy abs diff `3.69548797607e-06`).

Raw readings and provenance are recorded in
`mypaper/code/cycle09_stage3_execution_handoff.md`. The final supervisor state
is `SHUTDOWN_CANCELLED`; the user-requested cancellation marker kept the server
running after validation.

<!-- cycle09-stage3-complete-end -->

---

---

<!-- cycle09-q1-stage-b-320-start -->

## Cycle 09 Q1 endpoint correction

Q1 alpha=.5 keeps its active Stage A exactly at steps `0..160`. A separate, gated
Stage B implementation now resumes only from a completed step-160 checkpoint and ends
exactly at step `320`; it cannot continue toward step 624. The Stage-B parquet reuses
the 2,560 Stage-A prompts in the same order and swaps each prompt's self/external
support assignment, preserving the per-batch 8:8 composition. No Stage-B training was
started by this configuration change. Llama P0 remains independently configured through
step 624.

<!-- cycle09-q1-stage-b-320-end -->

---

<!-- cycle09-q1-stage-a-start -->

## Cycle 09 Q1 alpha=.5 Stage A

Implemented the detached Stage-A postprocessing chain: checkpoint validation and source-separated support statistics; revision-pinned MATH/AIME25 acquisition with frozen deduplication; six-probe L18 whitened `r_epsilon` geometry; and keypoint MATH500/MMLU-Pro/IFEval evaluation. The supervisor runs a smoke gate before the formal two-GPU queue, keeps base references ahead of dependent checkpoint cells, remains resumable by completion artifacts, never shuts the instance down, and has no path to start the gated `160 -> 320` Stage B.

Raw handoff: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_qwen_alpha05_stage_a_handoff.md`.

<!-- cycle09-q1-stage-a-end -->


<!-- cycle09-llama-p0-stage-a-recovery-start -->

## Cycle 09 Llama OPD P0 Stage A - recovered terminal rollout gap

P0 Stage A was bounded at step 160 with checkpoint grid `0/5/10/20/40/80/160`.
All seven checkpoints passed the adapter completeness check. Raw rollout and training-metric
coverage is steps `1..159` (2,544 rows); step 160 has no raw rollout or metric row.
The final student rollout remained active for more than 45 minutes versus the earlier
2.5-3.5 minute batches, so its process group was stopped after the step-160 checkpoint
was persisted. The supervisor records `rc=-15`; recovery writes the terminal gap
explicitly and does not synthesize data.

Fixed two postprocess defects exposed by this recovery: the Stage-A checkpoint directory
and grid now use separate names, and mini support-stat CSV writes reuse the full union
of overall-row fields. All modified scripts compile; checkpoint inventory reports seven
complete cells. No geometry, behavioral evaluation, or `160->320` Stage B training was
started.

Raw handoff: `mini/mini_llama_opd_p0_stage_a_handoff.md`.

<!-- cycle09-llama-p0-stage-a-recovery-end -->

---

<!-- cycle09-llama-p0-stage-b-320-start -->

## Cycle 09 Llama P0 Stage B (160->320) - delivered raw handoff

The authorized Llama OPD extension resumed from step 160 and committed step 320.
The checkpoint inventory is complete on `{0,5,10,20,40,80,160,320}`. The terminal
asynchronous rollout/metric dump did not persist at either step 160 or step 320; both
are explicitly recorded as gaps after their checkpoints were durable, and no synthetic
rows were created. The terminal process group was stopped after the step-320 commit
(`rc=-15`) so the detached controller could continue.

Four-arm Llama behavior and geometry were then finalized at the seven-point early grid
`{0,5,20,40,80,160,320}`: behavior has 75 task rows and 225 IFEval category rows;
geometry has 9,912 `r_epsilon`, 2,478 tail-energy, 306 raw-representation, and 150
spectra-inventory rows. Geometry retained the existing adapter `B@A` fp32 delta-W
policy and per-checkpoint protocol. This entry records completion/provenance only.

Raw Theory handoff:
`mini/mini_llama_p0_stage_b_320_handoff.md`.
Machine-readable manifest:
`mini/llama_p0_stage_b_320_handoff_manifest.json`.

<!-- cycle09-llama-p0-stage-b-320-end -->
---

<!-- cycle09-stage3-h1-handoff -->

## Cycle 09 Stage3 H1 raw handoff

Completed raw-artifact package: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage3_H1_theory_handoff.md`. Machine-readable manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/stage3_H1_handoff_manifest.json`. The package records coverage, declared missing cells, and provenance without interpretation.
---

<!-- cycle09-stage3-h0-handoff -->

## Cycle 09 Stage3 H0 raw handoff

Completed raw-artifact package: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage3_H0_theory_handoff.md`. Machine-readable manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/stage3_H0_handoff_manifest.json`. The package records coverage, declared missing cells, and provenance without interpretation.

---

<!-- cycle09-stage3-h1-resync-v2 -->

## Cycle 09 Stage3 H1 corrected raw handoff

Rebuilt H1 after correcting the retained Qwen R4 schema mapping and Llama source filename. The prior derived package is archived under `/root/autodl-tmp/cycle09_stage3_followup/archive/`; the canonical H1 paths now point to the corrected raw package.
---

<!-- cycle09-stage3-h2-handoff -->

## Cycle 09 Stage3 H2 raw handoff

Completed raw-artifact package: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage3_H2_theory_handoff.md`. Machine-readable manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/stage3_H2_handoff_manifest.json`. The package records coverage, declared missing cells, and provenance without interpretation.
---

<!-- cycle09-stage3-h3-handoff -->

## Cycle 09 Stage3 H3 raw handoff

Completed raw-artifact package: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage3_H3_theory_handoff.md`. Machine-readable manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/stage3_H3_handoff_manifest.json`. The package records coverage, declared missing cells, and provenance without interpretation.
---

<!-- cycle09-stage3-h4-handoff -->

## Cycle 09 Stage3 H4 raw handoff

Completed raw-artifact package: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/mini_stage3_H4_theory_handoff.md`. Machine-readable manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/stage3_H4_handoff_manifest.json`. The package records coverage, declared missing cells, and provenance without interpretation.


---

<!-- cycle09-h5-checkpoint-retention -->

## Cycle 09 H5 checkpoint retention and disk recovery

H5 frozenSelf0-KD exhausted the data disk while attempting global_step_95.
The last durable native checkpoint is step 90. The formal launcher now starts
a checkpoint-retention guard: it preserves only the landmark grid
{5,20,40,80,160,320} plus the latest durable resume point, and never prunes
a directory at or above the latest durable marker while PPO is running.

The approved cleanup removed obsolete H5 intermediate checkpoints, the failed
step 95 write, validated smoke actor weights, Qwen alpha=.5 smoke weights, and
duplicate materialization-side top-32 arrays. Full cleanup provenance:
 /root/autodl-tmp/cycle09_stage3_followup/H5_frozen_self/H5_storage_cleanup_manifest.json.
---

<!-- cycle09-stage4-automation-smoke -->

## Cycle 09 Stage 4 automation and smoke

Implemented a restart-safe two-GPU Stage-4 supervisor for A0-A9. GPU0 owns Qwen
A1/A3 then Qwen A5/A6; GPU1 owns the corresponding Llama lane; CPU performs A0,
A2/A3 bootstrap preparation, A4 grouped checkpoint-held-out CV, A7 provenance, and
A9 schema after machine-readable cells land. Profile and metric cells are atomic and
idempotent. Retained-sample P1 profiles are disk-watermarked; the formal launcher is
detached and never requests instance shutdown.

The successful foreground smoke covered Qwen/Llama E_math step0/OPD-step20 profiles,
A1, centered A3, fixed-token A5, Qwen A6, A4, A7, and A9. It also verified direct
Llama adapter B@A fp32 delta construction, while the historical Qwen fallback is
explicitly tagged as an authorized fp32 effective-weight-difference exception.

Smoke manifest:
/root/autodl-tmp/cycle09_stage4_state_displacement/supervisor_state.json.

<!-- cycle09-stage4-automation-smoke-end -->

---

<!-- cycle09-stage4-storage-reclamation -->

## Cycle 09 Stage 4 storage reclamation

The formal DAG now skips a main profile whenever its metric manifest is complete; it
requeues a profile only when the metric is missing or a required centered cell remains.
After this restart dependency change, the user-authorized cleanup removed consumed
Stage4 main/smoke profiles and legacy R4 raw scratch caches. The cleanup manifest is
stored at /root/autodl-tmp/cycle09_stage4_state_displacement/audit/stage4_storage_cleanup_manifest.json.
It preserved base profiles, P1 retained factors, incomplete centered inputs, cells,
directions, result tables, checkpoints, and model sources.

<!-- cycle09-stage4-storage-reclamation-end -->

---

<!-- cycle09-m6-b0b2-raw-handoff -->

## Cycle 09 M6 completion and B0--B2 raw transfer

The AIME24-only M6 supplement queue completed all 17 of 17 queued GPU groups with no
non-complete cell. Its final raw artifacts contain 20 behavior rows and 1,610 geometry
rows; AIME25 remains explicitly excluded by the recorded scope revision. The raw
M6 table, queue provenance, hashes, and coverage are handed to Theory without
interpretation.

Also transferred are the B0/B1 dual-weight-object and matched-top-k provenance artifacts
plus B2's CPU-only direct-BA precompute: 50 inventory rows, 18 usable direct-BA cells,
and 576 singular-value rows. Six Qwen OPD direct-BA cells at steps 20/160/624 and
`o_proj`/`down_proj` are explicitly blocked because their serialized adapters are absent;
no merged-minus-base replacement was made. B2's GPU landmark audit, B3 smoke, and B4
rollback/proxy pilot have not yet begun.

Raw Theory handoff:
`mini/mini_cycle09_m6_b0b2_theory_handoff.md`.
Machine-readable manifest:
`mini/mini_cycle09_m6_b0b2_theory_handoff_manifest.json`.

<!-- cycle09-m6-b0b2-raw-handoff-end -->
---

<!-- cycle09-actual-output-o0-o5-handoff -->

## Cycle 09 O0--O5 actual checkpoint-output raw transfer

Completed the O0--O5 Llama minimum closure at the O5 gate: O0 forward inventory, O1 source-of-forward effective-weight selection, retained B2 direct-BA/merged smoke, O2 cumulative and stepwise geometry, O3 fixed-token full-vocabulary KL/NLL, O4 matched cells, and O5 checkpoint-grouped plus leave-arm-out raw tables. Final aggregation explicitly includes only formal `nall/capall` artifacts, excluding retained n=1/n=2 smoke outputs. O6 remains Theory-GO-gated and O7 did not run by design.

Raw Theory handoff:
`mini/mini_cycle09_actual_output_o0_o5_theory_handoff.md`.
Machine-readable manifest:
`mini/cycle09_actual_output_o0_o5_handoff_manifest.json`.

<!-- cycle09-actual-output-o0-o5-handoff-end -->

---

<!-- cycle09-relative-functional-contraction-handoff -->

## Cycle 09 full relative functional contraction raw transfer

Completed C1--C5 coverage/derivation/fixed-token forward/analysis pipeline. The final output queue is `140/140 complete`: the initially failed Qwen base step-0 cells (`E_general`, `E_math`, `E_ood`, `E_if`) were repaired and rerun before the aggregate, audit, manifest, and Theory handoff were regenerated. The handoff records declared `PENDING_UPSTREAM` and state-rank coverage gaps without interpretation.

Raw Theory handoff:
`mini/full_relative_functional_contraction_theory_handoff.md`.
Machine-readable manifest:
`mini/full_relative_functional_contraction_manifest.json`.

<!-- cycle09-relative-functional-contraction-handoff-end -->

---

<!-- cycle09-d5-d7-raw-handoff -->

## Cycle 09 D5-D7 raw transfer

D5 completed the fairness energy grid over 60 checkpoint groups: 6,720 module rows and 960 equal-seven cells, using the serialized BF16 merged checkpoint difference with FP32 subtraction, current-S as the main track, and fixed-S as a sensitivity. D6 and D7 then completed raw robustness, cumulative/signed/absolute NLL, detrending, grouped prediction, interaction, and stepwise tables. At this initial raw handoff, the separate D4.1 parity audit was still pending; the following D4.1 addendum resolves that audit. No D8 final-completion claim is made.

Raw Theory handoff: mini/d5_d7_raw_theory_handoff.md.
Machine-readable manifest: mini/d5_d7_raw_handoff_manifest.json.

<!-- cycle09-d5-d7-raw-handoff-end -->

---

<!-- cycle09-d4-parity-addendum -->

## Cycle 09 Qwen D4.1 parity-audit addendum

Ran the independent numerical reconstruction audit required after the 148/148 Qwen
merged-state matrix. Base step 0 and SFT/off-KD/seqKD step 160 all reproduce the
stored module ranks, equal-seven means, and epsilon ordering exactly: 528/528 raw
checks are PASS_EXACT. Qwen OPD steps 5/160/624 are explicitly recorded as 12
BLOCKED_MISSING_OPD_ADAPTER rows because their adapter B@A files are unavailable;
no merged-minus-base substitute was used. This is a raw parity transfer, not a D8
completion claim.

Raw Theory addendum:
mini/d4_parity_addendum_theory_handoff.md.

<!-- cycle09-d4-parity-addendum-end -->

---

<!-- cycle09-d8-completion-handoff -->

## Cycle 09 D3 v2 and D8 full relative-functional-contraction completion

Reran D3 after D4 using only the completed Qwen merged-state and matching fixed-token
outputs. The full-availability Model-C input is now 96 Llama plus 144 Qwen matched
state/output cells; all 54 grouped CV rows completed and 2,160 predictions were
written. The CPU-only D8 verifier checked D2-D7 coverage, rows, hashes, headline
table statuses, and the mechanical D7.1 branch code. It then issued the final
COMPLETE_CORE_MATRIX handoff, retaining the declared Qwen OPD parity boundary.

Raw Theory handoff:
mini/full_relative_functional_contraction_completion_handoff.md.
Machine-readable manifest:
mini/full_relative_functional_contraction_completion_manifest.json.

<!-- cycle09-d8-completion-handoff-end -->

---

<!-- cycle09-d10-llama-matched-numeric-handoff -->

## Cycle 09 D10 Llama matched-numeric parity and D10.5 downstream transfer

Implemented and ran the D10 Llama matched-numeric core on the 2 x 32G RTX 5090
machine after the data copy completed. The formal D10 core produced 100/100
state-output cells for Llama L14, four arms, six checkpoints, and four core
probes under BF16 load/forward, FP32 Gram accumulation, FP64 symmetric eig/SVD,
and fresh BF16 fixed-token output caches. Final raw counts are 2,800 module-rank
rows, 400 equal-seven rows, 100 fixed-token output rows, and 4 legacy-vs-matched
summary rows. The legacy parity summary reports exact-rank match fractions
0.599702--0.633929 and abs(rank-difference)<=1 fractions 0.909226--0.912202
across the four arms.

The same detached controller also ran a strict fixed-k source-principal `p_k`
raw transfer with `k={4,8,16,32}`: Llama `adapter_ba` produced 672 rows and Qwen
`bf16_merged_minus_base` produced 896 rows. This is not a full A3 claim for
PABS/NSS. The current 2 x 32G machine recorded A10 as
`HOLD_HARDWARE_INSUFFICIENT_FOR_2X96_PROTOCOL`; no seed43 training was launched.

D10.5 correction: implemented and ran
`experiments/opd_sft_h1/scripts/cycle09_d10_5_finalize.py --phase formal`,
replacing the legacy-Llama downstream pass with tables built from
`llama_d10_matched_numeric` and the Qwen D4 matched track. The D10.5 status is
`COMPLETE_D10_5_CORE_A2_A4_WITH_PARTIAL_A5_CENTERED_AGGREGATED`: 6,944
integrated module rows, 992 equal-seven rows, 248 output rows, 960 A4 feature
rows, 756 partial A5 centered rows, and 29 output files were written. The
historical `cycle09_contraction_completion.py --phase cpu-all` and
`cycle09_d5_d7_tables.py --phase formal` outputs are superseded for D10.5
purposes because they still read/label Llama as `legacy_llama`.

The D10.5 branch-code row is raw code `C`: Llama OPD signed-NLL Spearman
0.846087, Llama OPD detrended Spearman 0.445217, Llama OPD interaction 2.98801,
Qwen OPD signed-NLL Spearman 0.745946, Qwen SFT/off-KD/seqKD signed-NLL
Spearman 0.700245/0.801802/0.791248, Qwen OPD detrended Spearman 0.0777349,
and Qwen OPD interaction 0.438631. D10.5 NCD at eps=.05 is Llama OPD 10.250,
SFT 5.345, off-KD 2.746, seqKD 1.844; Qwen OPD 5.353, SFT 0.690, off-KD 2.646,
seqKD 3.206. The regenerated D10.5 same-grid shared-axis status is 17/18; the
older 18/18 statement remains the earlier D8/D10 readback claim, and six-probe
36/36 remains legacy sensitivity.

A5--A9 acceptance boundaries are now explicit: A5 current/fixed, A6 sample-tail-
epsilon robustness, A7 schema/provenance, and A8 trainer/top-32 audit are accepted
from existing Stage4 formal artifacts; A5 centered is only the partial centered
grid present on disk; A9 has legacy behavior protocol availability rather than a
new D10-specific item bootstrap. Full PABS/NSS mask claims are not asserted, and
stepwise KL is not reconstructed from aggregate D10/D4 output caches.

Raw handoff:
`mini/d10_llama_numeric_parity_handoff.md`.
Machine-readable manifest:
`mini/d10_llama_numeric_parity_manifest.json`.
D10.5 handoff:
`mini/d10_5_a2_a4_handoff.md`.
D10.5 manifest:
`mini/d10_5_a2_a4_manifest.json`.

<!-- cycle09-d10-llama-matched-numeric-handoff-end -->

---

<!-- cycle09-d11-pk-tpnt-handoff -->

## Cycle 09 D11 PK-TPNT deployed-weight correction and TPNT core transfer

Implemented and ran `experiments/opd_sft_h1/scripts/cycle09_d11_pk_tpnt.py`
for the frozen D11_PK_TPNT protocol. The run completed as
`COMPLETE_D11_CORE` on the 2 x 32G RTX 5090 machine without training, without
new free-generation behavior eval, and without recomputing the D10
`c_epsilon/r_epsilon` state track. Llama strict joint `p_k` was regenerated on
the official BF16 deployed merged-minus-base track; Qwen strict `p_k` reused the
existing deployed-effective table, as required by the updated protocol.

Raw output coverage: `d11_llama_merged_pk.csv` has 672 rows, matching
4 arms x 6 checkpoints x 7 modules x 4 fixed-k values. The adapter-BA versus
merged-BF16 audit has 672 rows. The TPNT principal-mask table has 2,016 rows
for two models, four arms, six checkpoints, seven modules, three source ranks,
and two mask densities. The angles/PABS/NSS table has 1,344 rows for two
models, four arms, six checkpoints, seven modules, and four angle-k values.
The same-cell feature matrix has 192 rows, and the grouped incremental
comparison table has 132 rows.

The TPNT random-null column uses a rank-32 Frobenius-norm-matched low-rank BF16
update mask with fixed deterministic seeds. E5--E7 optional enhancements were
not run and are not part of this core transfer. The D11 handoff reports raw
tables and mechanical status only; it does not adjudicate Theory claims.

Optional E5--E7 addendum: implemented and ran
`experiments/opd_sft_h1/scripts/cycle09_d11_e5_e7.py` after the E0--E4 inputs
were immutable. The optional addendum completed as `COMPLETE_D11_OPTIONAL_E5_E7`.
E5 layer robustness wrote 3,024 raw rows and 432 summary rows over Qwen
L9/L18/L27 and Llama L7/L14/L21 at checkpoints 20/160/320. E6 alpha sensitivity
wrote 2,016 raw rows and 288 summary rows, adding alpha 5% and 20% to the
headline-layer landmark cells while retaining the main 1%/10% rows for comparison.
E7 spectrum-matched random-subspace null wrote 1,008 aggregate rows and 144
summary rows using 10 fixed seeds per landmark cell and preserving the positive
singular values of the deployed BF16 merged-minus-base update. The core D11
manifest status was updated to `COMPLETE_D11_CORE_PLUS_OPTIONAL_E5_E7`.

Raw handoff:
`mini/d11_modelwise_and_crossmodel_summary.md`.
Machine-readable manifest:
`mini/d11_pk_tpnt_manifest.json`.
Optional E5--E7 handoff:
`mini/d11_e5_e7_optional_handoff.md`.
Optional E5--E7 manifest:
`mini/d11_e5_e7_manifest.json`.
Consolidated full handoff:
`mini/d11_pk_tpnt_full_handoff.md`.

<!-- cycle09-d11-pk-tpnt-handoff-end -->

<!-- cycle09-reviewer-robustness-start -->

## Cycle09 Reviewer Robustness Reuse Pass

按 `mypaper/code/cycle09_reviewer_robustness_handoff.md` 执行 reviewer-derived
稳健性专项。新增脚本
`experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py`。本轮严格为
zero training / no new forward：先跑 RR0 inventory，再只执行 RR0 标为
`READY_REUSE` 的 RR2/RR4/RR5/RR6；RR1A/RR1B 与 RR3 均未越过 GO 门槛。

RR0 产物落
`mini/reviewer_robustness/RR0_artifact_inventory.csv`、
`RR0_grid_coverage.csv`、`RR0_protocol_map.json`、`RR0_blockers.md`、
`RR0_task_estimates.csv`。覆盖状态：RR1A=96/96
`READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT`，RR1B=96/96 同样需要 formal
per-sample factor 或新 forward；RR2=60/96 `READY_REUSE`，Qwen
sft/offkd/seqkd 36 格缺 Stage4 direction spectrum；RR3=96/96
`READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT` 且需 explicit Theory GO；RR4=96/96
`READY_REUSE`；RR5=96/96 初始 presence gate `READY_REUSE`；RR6=Llama 48 格
`READY_REUSE`，Qwen 48 格按协议不适用。

READY_REUSE 后处理完成：

- RR2 写出 `RR2_spectrum_stability_module.csv`(1680 rows)、
  `RR2_spectrum_stability_equal7.csv`(240 rows)、
  `RR2_continuous_ordering.csv`(48 rows)、
  `RR2_continuous_output_links.csv`(20 rows) 与 manifest。只使用 Stage4
  direction singular values；Qwen 非 OPD 缺格保持 blocked。
- RR4 写出 `RR4_top32_retained_mass_summary.csv`(3 rows)、
  `RR4_top32_retained_mass_by_sequence.csv`(10000 rows)、
  `RR4_top32_coverage.csv`(4 rows) 与 manifest。raw-logprob sources:
  Llama model2_llama、Qwen offkd、Llama frozen_self；Qwen alpha05
  frozen_external 因 manifest 不能证明 raw full-vocab logprob，标为
  `BLOCKED_RENORMALIZED_TOPK`。
- RR5 exact-key join 后停止：`RR5_hybrid_coverage.csv` 显示 D11 192 格中 raw
  activation exact join 仅 76 格，且缺 Qwen OPD/SFT/seqKD 完整 arm；按协议不
  impute、不拟合，manifest 状态 `BLOCKED_INSUFFICIENT_COMMON_GRID`。
- RR6 写出 `RR6_frozen_self_text_stats.csv`(8 rows)、
  `RR6_frozen_self_paired_differences.csv`(2000 rows)、
  `RR6_frozen_self_coverage.csv`(4 rows) 与 manifest。只使用 Llama OPD vs
  frozen_self 的 math500 row-matched cells at steps {20,40,80,160};
  broader rollout/text pools未当作 paired。

Theory handoff:
`mini/reviewer_robustness/reviewer_robustness_theory_handoff.md`。

<!-- cycle09-reviewer-robustness-end -->

### 2026-07-27 Reviewer Robustness Correction Pass

- Script updated: `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py`.
- Old RR2 manifest marked `SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION`.
- Added RR2S state-spectrum preflight, RR2D displacement-spectrum auxiliary, RR5 Llama-only exact common-grid fitting, RR6 matched Math500 readout bootstrap, RR4 retained-mass provenance supplement, and RR1/RR3 shared forward preflight.
- Output handoff: `mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/reviewer_robustness_theory_handoff.md`.

### 2026-07-27 Reviewer Robustness Correction2 Reuse Pass

- Added `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_correction2.py`.
- Completed RR5 fold performance, checkpoint-demeaned descriptive correlations, and nested train-fold-only regularization.
- Completed Llama RR2S from D10 formal saved state spectra without model forward.
- Completed Llama RR3 centered audit from D10 formal profiles and BF16 merged deployment weights without model forward.
- Added RR2D equal-seven/order/output-link aggregation and task-specific RR1/RR2S/RR3 availability manifest.

### 2026-07-27 Reviewer Robustness Module-Level Summary

- Added `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_module_summary.py`.
- Wrote `RR_module_level_robustness_summary.csv`, `RR_module_level_robustness_cells.csv`, and manifest from formal RR2S/RR3 module tables.
- Explicitly marked nested RR5 as formal, fixed-regularization RR5 as parity only, old RR2 as superseded, and RR2D as auxiliary.

### 2026-07-28 Reviewer Robustness Module-Level Tie-Aware Correction

- Updated `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_module_summary.py` to separate unique OPD wins, tied maxima, and offline wins.
- Regenerated `RR_module_level_robustness_summary.csv`, `RR_module_level_robustness_cells.csv`, and manifest from formal RR2S/RR3 module tables.
- Explicitly marked nested RR5 as formal, fixed-regularization RR5 as parity only, old RR2 as superseded, and RR2D as auxiliary.

## EQUAL5_NON_QK Reuse Return: 2026-07-28T06:17:19.563050+00:00

- Task: measurement-side non-q/k equal-5 aggregation from existing module artifacts.
- Boundary: GPU used=false; new_forward=false; new_training=false; new_behavior_eval=false; new_svd=false.
- Command: `experiments/opd_sft_h1/scripts/cycle09_equal5_non_qk.py`
- Output directory: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk`
- Completed outputs:
  - `EQUAL5_coverage_inventory.csv`: 9156 rows
  - `EQUAL5_functional_cells.csv`: 6520 rows
  - `EQUAL5_functional_trajectories.csv`: 1304 rows
  - `EQUAL5_dominance_cells.csv`: 96 rows
  - `EQUAL5_dominance_summary.csv`: 12 rows
  - `EQUAL5_ncd.csv`: 32 rows
  - `EQUAL5_spectrum_robustness.csv`: 469 rows
  - `EQUAL5_support_controls.csv`: 10 rows
  - `EQUAL5_weight_baselines_module.csv`: 12000 rows
  - `EQUAL5_weight_baselines_aggregate.csv`: 2401 rows
  - `EQUAL5_output_link_correlations.csv`: 384 rows
  - `EQUAL5_checkpoint_demeaned_correlations.csv`: 24 rows
  - `EQUAL5_nested_metrics.csv`: 28 rows
  - `EQUAL5_nested_folds.csv`: 112 rows
  - `EQUAL5_nested_predictions.parquet`: 1792 rows
  - `EQUAL5_d11_parity_metrics.csv`: 41 rows
  - `EQUAL5_equal7_paired_comparison.csv`: 1306 rows
- Blocked items were recorded in `EQUAL5_coverage_inventory.csv` and `EQUAL5_manifest.json`; blocked rows were not reverse-engineered from equal-7 aggregates.

## EQUAL5_NON_QK Reuse Return: 2026-07-28T06:18:15.495260+00:00

- Task: measurement-side non-q/k equal-5 aggregation from existing module artifacts.
- Boundary: GPU used=false; new_forward=false; new_training=false; new_behavior_eval=false; new_svd=false.
- Command: `experiments/opd_sft_h1/scripts/cycle09_equal5_non_qk.py`
- Output directory: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk`
- Completed outputs:
  - `EQUAL5_coverage_inventory.csv`: 9156 rows
  - `EQUAL5_functional_cells.csv`: 6520 rows
  - `EQUAL5_functional_trajectories.csv`: 1304 rows
  - `EQUAL5_dominance_cells.csv`: 96 rows
  - `EQUAL5_dominance_summary.csv`: 12 rows
  - `EQUAL5_ncd.csv`: 32 rows
  - `EQUAL5_spectrum_robustness.csv`: 517 rows
  - `EQUAL5_support_controls.csv`: 10 rows
  - `EQUAL5_weight_baselines_module.csv`: 12000 rows
  - `EQUAL5_weight_baselines_aggregate.csv`: 2401 rows
  - `EQUAL5_output_link_correlations.csv`: 384 rows
  - `EQUAL5_checkpoint_demeaned_correlations.csv`: 24 rows
  - `EQUAL5_nested_metrics.csv`: 28 rows
  - `EQUAL5_nested_folds.csv`: 112 rows
  - `EQUAL5_nested_predictions.parquet`: 1792 rows
  - `EQUAL5_d11_parity_metrics.csv`: 41 rows
  - `EQUAL5_equal7_paired_comparison.csv`: 1306 rows
- Blocked items were recorded in `EQUAL5_coverage_inventory.csv` and `EQUAL5_manifest.json`; blocked rows were not reverse-engineered from equal-7 aggregates.

## 2026-07-28T08:56:15.838992+00:00 FAT-R1 S0 boundary gate

- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_outlink_round1.py`
- Output root: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1`
- Status: `BLOCKED_TOKEN_BOUNDARY_PROTOCOL`.
- Finding: MATH `\boxed{...}` span offset audit produced 3404 boundary-crossing rows; formal M1/M2 forward was not launched.

## 2026-07-28T11:09:19.826523+00:00 FAT-R1 output-link correction return

- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_outlink_round1.py`
- Output root: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2`
- Manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2/fat_r1_v2_manifest.json`
- Scope: confirmed first round only (`FAT-R1-S0`, `FAT-R1-M1`, `FAT-R1-M2`); no rollout/training/free generation.
- Numeric protocol: BF16 checkpoint forward, FP32 log-softmax/NLL/KL, exact full-vocabulary KL `D_KL(p0 || pt)`.

## 2026-07-28T13:13:51.319373+00:00 FAT-R1-v2 link return
- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_r1_v2_link.py`
- Output root: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_link`
- Manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_link/fat_r1_v2_link_manifest.json`
- Status: `PARTIAL_FAT_R1_LINK_PK_STEP10_EXCLUDED`
- Scope: CPU-only reuse linking of FAT-R1-v2 regional output cells with headline `c_epsilon` and deployed merged strict-joint `p_k`; no forward/training/Eval rerun.
- Fixed exclusions: Qwen `p_k` step10 kept missing and marked `QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION`.

## 2026-07-28T14:43:06.984242+00:00 FAT-R1-v2 equal-5 correction return
- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_r1_v2_equal5_final.py`
- Output root: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_link_equal5`
- Manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_link_equal5/fat_r1_v2_equal5_manifest.json`
- Status: `COMPLETE_EQUAL5_WITH_QWEN_PK_STEP10_EXCLUDED`
- Boundary: CPU-only reuse; no training/forward/rollout/Eval/new SVD.
- Headline aggregation: non-QK equal-5 modules `v/o/gate/up/down`; q/k excluded; equal-7 retained as sensitivity.
- Qwen step10 `p_k` remains excluded, not imputed.

## 2026-07-29T07:01:44.502715+00:00 FAT-R1-v2 MATH KL_C completion

- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_math_kl_c_completion.py`
- Output root: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_math_kl_c`
- Manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_math_kl_c/fat_r1_v2_math_kl_c_manifest.json`
- Scope: completed exact full-vocabulary `KL_C` for MATH500 C-region on the FAT-R1-v2 formal grid; no rollout/training/new geometry; original FAT-R1-v2 artifacts left intact.
- Numeric protocol: BF16 checkpoint forward, FP32 log-softmax/KL, `D_KL(p0 || pt)`, batch-local logits only.
