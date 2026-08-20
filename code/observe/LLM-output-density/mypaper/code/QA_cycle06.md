# QA: Cycle 06 — Code Execution Phase

```yaml
phase: code_execution
cycle: cycle_06_sft_feasibility_and_degradation
status: open
created: 2026-06-23
write_scope:
  - local_experiment_results/cycle_06_sft_feasibility_and_degradation/
  - code/code_evolution.md (append handin section after run)
  - code/current_code_structure.md (append change section after run)
read_scope:
  - readme.md
  - code/current_code_brief.md
  - code/current_code_structure.md
  - exp/experimental_methodology/ (4 files)
  - exp/current_experiment_design.md
  - exp/QA_cycle06_exp.md
```

本文件是 Cycle 06 代码对话启动前的 QA 审查清单。用户回答后，结论内化到执行计划，再开始写代码。

---

## 背景（已读文件摘要）

**当前任务（来自 current_code_brief.md）**：

- Phase 1：构建 n_sft_max=4096 的数据母池，生成 instruct 和 base 两种渲染版本
- Phase 2：训练 8 个 SFT 模型（instruct × 4 + base × 4），无 OPD arm
- Phase 3：评测 10 个模型（含 2 个未训练 base 参照行），instruct 系列沿用 Cycle 05 协议，base 系列需要新写 4-shot CoT 无 chat template 脚本
- Phase 4：GetSlice minimal geometry + principalEvidence.py 权重主成分分析

**代码库现状（来自 current_code_structure.md）**：

- 主入口：`experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py`（Cycle 04 风格单脚本全流程）
- Eval：`Eval/component/gsm8k/runner.py`、`math500/runner.py`、`numina/runner.py`（instruct 模式，已有）
- 确认：base 系列 math eval runner 尚不存在（brief 已明确 "New scripts needed"）
- 确认：权重导出工具尚不存在

**关键确认（来自文件系统验证）**：

- Qwen3-1.7B-Base checkpoint 存在于 `/root/autodl-tmp/model/Qwen/Qwen3-1___7B-Base`（注意：下划线替代点，HuggingFace 本地存储约定）
- Qwen3-1.7B instruct 在 `/root/autodl-tmp/model/Qwen/Qwen3-1.7B`

---

## 核心问题

### Q1：exp_root 路径

Cycle 04 的产物在 `/root/autodl-tmp/cycle04_opd_stability_gain/`。按照同样命名约定，Cycle 06 建议使用：

```
/root/autodl-tmp/cycle06_sft_feasibility_and_degradation/
```

**问题**：就按你说的使用这个路劲

---

### Q2：base 系列 math eval 的 4-shot 示例来源

brief 要求：
- **NuminaMath-test 4-shot**：从 test.jsonl 中选 4 个 exemplar（不参与评分）
- **GSM8K 4-shot**：standard math benchmark CoT demonstrations（e.g., Qwen3 tech report 或 Hendrycks MATH 标准 shots）
- **MATH500 4-shot**：同上

**候选方案**：

a. **程序化选取 NuminaMath，手工固定 GSM8K/MATH500**：NuminaMath 用 test.jsonl 前 4 行（非评分行），GSM8K/MATH500 用我选取的 4 个标准 CoT 示例（来自 GSM8K 官方 few-shot + MATH 论文示例），全部硬编码写入 `run_provenance.json`。

b. **全部程序化**：NuminaMath 同上，GSM8K/MATH500 直接从测试集自身取前 4 行作为 shots（前 4 行不参与 scoring，其余 N-4 行评测）。简单但偏离"标准 benchmark shots"。

c. **使用 Qwen3 官方 prompt**：Qwen3 tech report (arXiv:2505.09388) 提供的 base-model few-shot prompt 格式。如果你有原文或能确认具体示例来源，我按原文实现；否则退回方案 a。

**问题**：我认为选择a，但是现在是都是用4-shot还是有0-shot的测试？

---

### Q3：principalEvidence.py 覆盖层数

`run_principal_evidence()` 每次调用只处理一个 `target_layer`。brief 说明 "run for each of [6, 14, 22]"（样本 config 中 `target_layer: 14`，注释为 "run for each of [6,14,22]"）。

**选项**：

a. **全 3 层**：调用 3 次（layer 6 / 14 / 22），结果合并进同一个 `principal_evidence.csv`（利用函数内置的 CSV merge 逻辑，per-layer rows 会叠加）。产物更丰富，运行时间 × 3。

b. **仅 layer 14**（中间层，信息量最大）：只跑一层，简化产物，但 brief 明确说 "run for each"。

**问题**：确认方案 a

---

### Q4：脚本结构——单一编排脚本 vs 分阶段独立脚本

Cycle 04 使用单一主脚本 `run_opd_stability_gain_cycle04.py` 一键串行运行全流程（含 skip-if-exists），有 `--smoke` 模式。Cycle 06 共有 4 个阶段、8 个训练 + 10 × eval + 几何分析，单次运行可能需要数小时甚至多天。

**选项**：

a. **单脚本**（`scripts/run_cycle06.py`）：全流程编排，skip-if-exists，支持 `--smoke` 和 `--skip-phase1/2/3/4` 开关。出错后可以重启继续。风格与 Cycle 04 一致。

b. **分阶段脚本**：写 `phase1_pool.py`、`phase2_train.py`、`phase3_eval.py`、`phase4_geometry.py`，每个脚本独立可运行。灵活，可以单独重跑某阶段，但没有跨阶段的 skip 逻辑。

c. **混合**：一个单脚本 + `--phase {1,2,3,4}` 开关，可以只跑某一阶段，同时保留脚本的全局状态管理。

**问题**：使用a

---

### Q5：numina runner 默认参数差异

现有 `Eval/component/numina/runner.py` 的默认参数：
- `temperature=0.0`（greedy）
- `top_k=-1`（无限制）
- `max_tokens=1024`

brief 要求（Cycle 05 验证协议）：
- `temperature=0.7, top_p=0.8, top_k=20, max_gen_toks=3072`

runner 函数支持传参覆盖，但如果其他地方直接用 `runner.run(model=..., ...)` 不传这些参数，会用错误的默认值。

**选项**：

a. **修改 runner.py 的默认值**为 `temperature=0.7, top_p=0.8, top_k=20, max_tokens=3072`，对齐 Cycle 05 验证协议。
b. **在编排脚本中始终显式传参**，不改动 runner.py。

**问题**：方案a，这是在cycle05中确认的。

---

## 实现说明（不需要用户决定，信息备查）

以下是我已经确定的实现细节：

### 已确认的实现决策

1. **base 系列 runner 结构**：新建 `Eval/component/gsm8k/runner_base.py`、`math500/runner_base.py`、`numina/runner_base.py`，不修改现有 instruct runner。4-shot CoT 提示以 `Problem: {problem}\n\nSolution: {solution}` 格式构建，不调用 `apply_chat_template`。

2. **权重导出工具**：新建 `experiments/opd_sft_h1/scripts/export_weights.py`，加载合并后的 HuggingFace 模型，提取 `model.layers[N].{module}.weight`（Qwen3 架构），保存为 `.npy`。格式 `model_layers_{layer}_{module}_weight.npy`（module 名中的点号保留，如 `self_attn.q_proj`）。

3. **principalEvidence 目录结构**：
   ```
   {exp_root}/weights/
     instruct_base/                 ← base model weights (flat, no size_dir)
       model_layers_6_self_attn.q_proj_weight.npy ...
     instruct_sft_n512/
       512/                         ← size_dir name must be parseable int
         model_layers_6_self_attn.q_proj_weight.npy ...
   ```

4. **base_base 路径**：实际使用 `/root/autodl-tmp/model/Qwen/Qwen3-1___7B-Base`（下划线版），不用 brief 中的 `Qwen3-1.7B-Base`。在 handin 中记录偏差。

5. **SFTTrainer 数据格式**：
   - instruct 系列：`dataset_text_field=None`，使用 `messages` 格式（chat template rendering 交给 SFTTrainer）
   - base 系列：`dataset_text_field="text"`，plain text 直接输入，`packing=False`，不调用 `apply_chat_template`

---

## 结论（2026-06-23 确认）

| 问题 | 确认答案 | 备注 |
|---|---|---|
| Q1 exp_root | `/root/autodl-tmp/cycle06_sft_feasibility_and_degradation/` | 沿用 Cycle 04 命名约定 |
| Q2 4-shot 来源 | 方案 a：NuminaMath 取 test.jsonl 前 4 行；GSM8K/MATH500 用标准 CoT 示例（Wei et al. 2022 + Hendrycks MATH 标准示例，硬编码写入 provenance）| 所有 base 系列 math 均 4-shot；instruct 系列继续 0-shot；OOD-lite 由 lm_eval 自带 fewshot |
| Q3 principalEvidence 层数 | 全 3 层 [6, 14, 22]，各调用一次，CSV merge | |
| Q4 脚本结构 | 单脚本 `run_cycle06.py`，含 `--smoke` 和 `--start-from-phase N` | 风格与 Cycle 04 一致 |
| Q5 numina defaults | 修改 `numina/runner.py` 默认值为 temp=0.7/top_p=0.8/top_k=20/max_tokens=3072 | Cycle 05 已验证协议 |
