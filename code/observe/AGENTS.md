# `observe/`：既有项目与关键观察

本目录保存此前项目中可以被当前研究复用或对照的材料。当前优先继承和核验 `c_{\varepsilon}` 相对功能压缩指标；`C_{\gamma}` 观测方法作为次级谱 gap/稳定性指标一并保留。

## 维护规则

- 保留既有项目的代码、配置、数据说明和结果原貌，不把旧项目结果直接当作当前项目的实验结论。
- 整理 `c_{\varepsilon}` 或 `C_{\gamma}` 时至少说明：数学定义、计算输入、计算时机、输出含义、假设条件、数值稳定性和已知局限。
- 当前理论若使用这里的方法，应在 `../../human_read/theory/current_theory.md` 中明确引用来源和改动。
- 当前代码若实现或移植该观测方法，应在 `../CODE_OVERVIEW.md` 中记录模块位置、接口和与原实现的差异。
- 当前实验若验证该观测方法，应在对应的 `../../exp/<experiment_id>/` 中保留配置与远程证据索引，在 `../../human_read/result/` 中写人类可读的解释。

## `c_{\varepsilon}` 相对功能压缩指标索引

### 它是什么

`c_{\varepsilon}` 是当前比 `C_{\gamma}` 更重要的主观测量。它来自 activation-aware/SVDLLM whitening 后的功能秩 `r_{\varepsilon}`，用于描述某个模型状态相对 base 是否发生了 domain-conditioned 功能方向压缩。

先令

```text
A_{D,t,m} = W_{t,m} S_{D,t,m}
```

其中 `W_{t,m}` 是 checkpoint `t` 中 module `m` 的权重，`S_{D,t,m}` 是在 probe/domain `D` 上由 GetSlice profiling 得到的 activation scaling matrix。对 `A` 的奇异值 `sigma_i` 定义 `r_epsilon(A)` 为使最优 rank-k 近似的相对尾能量不超过 `epsilon` 的最小 rank：

```text
r_epsilon(A) = min k such that sum_{i>k} sigma_i^2 / sum_i sigma_i^2 <= epsilon
```

代码中等价写法是累计平方能量达到 `1 - epsilon` 时的 1-based index。正式阈值集合为 `{0.01, 0.025, 0.05, 0.10}`，headline 默认冻结 `epsilon = 0.05`，也就是保留 95% 能量。

然后在每个 module 内相对 base 归一化：

```text
c_{epsilon,D,t,m}
  = (r_{epsilon,D,0,m} - r_{epsilon,D,t,m}) / r_{epsilon,D,0,m}
```

解释：

- `c > 0`：相对 base 压缩，需要保留的功能方向变少。
- `c = 0`：相对 base 不变。
- `c < 0`：相对 base 膨胀，需要保留的功能方向变多。

主聚合必须 module-first：先对每个 module 用自己的 base rank 做归一化，再对模块等权平均。旧主口径是 equal-7：`q,k,v,o,gate,up,down`；后续更常用的 headline 是 equal-5 non-QK：`v,o,gate,up,down`，在结果表中常写作 `C5`。不要把“先平均 rank 再算比值”的 sensitivity 与主定义混称为同一个指标。

### 代码路径

核心定义和派生路径位于 `LLM-output-density/experiments/opd_sft_h1/scripts/`：

- `cycle09_actual_output_trajectory.py::rank_from_spectrum`：从奇异谱按累计平方能量计算 `r_epsilon`。
- `cycle09_relative_functional_contraction.py`：主相对功能压缩流水线；读取既有 `r_epsilon` 表，生成 `state_rank_delta`、`absolute_contraction`、`relative_functional_contraction_module`、`relative_functional_contraction_equal7` 等字段。
- `cycle09_equal5_non_qk.py`：把既有 module 级 `r_epsilon`/contraction 行转为 equal-5 non-QK headline，输出常用的 `c_equal5`/`C5` 口径。
- `mypaper/theory/stage_plan_handoff.md` 的 C1/C2：冻结了正式 schema、覆盖矩阵、epsilon 集合、headline layer 和“module-first 再聚合”的硬规则。
- `mypaper/theory/human_read-ch.md` 的 A.2/A.3 与 C.2/C.3：解释 `r_epsilon` 的最优 rank-k 近似含义、尺度/正交不变性、epsilon 稳健性和 equal-5/equal-7 敏感性。

### 如何获取

直接路径分为“先有 `r_epsilon`，再派生 `c_epsilon`”：

1. 用 GetSlice/geometry 在服务器上为每个 `model × arm × checkpoint × probe × layer × module × epsilon` 产生或读取 whitened state spectrum，即 `W_t S_{D,t,m}` 的奇异值。
2. 对每个奇异谱计算 `r_epsilon`：

```text
energy_i = sigma_i^2
r_epsilon = first k where cumulative_sum(energy)[k] >= (1 - epsilon) * total_energy
```

3. 严格匹配同一 `model × probe × layer × module × epsilon` 的 base rank `r_{epsilon,D,0,m}` 和当前 rank `r_{epsilon,D,t,m}`。
4. 计算：

```text
state_rank_delta = current - base
absolute_contraction = base - current
relative_functional_contraction_module = (base - current) / base
```

其中 denominator 必须大于 0；不得加任意小常数，也不得跨 module/probe 借 denominator。

5. 聚合：

```text
c_epsilon_equal7 = mean_m(relative_functional_contraction_module)
c_epsilon_equal5 = mean_m(relative_functional_contraction_module), m in {v,o,gate,up,down}
```

服务器脚本入口：

```bash
python <SERVER_BASE>/SOPPO/code/observe/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_relative_functional_contraction.py --phase audit
python <SERVER_BASE>/SOPPO/code/observe/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_relative_functional_contraction.py --phase derive
```

`audit` 生成 coverage/missing registry；`derive` 从已验证的 `r_epsilon` source rows 派生 module 审计表与 all-cells 聚合表。需要 output-link 分析时才继续跑 `aggregate-outputs`、`analyze` 等阶段；本地工作区不得运行这些项目脚本。

### 主要产物字段

```text
relative_functional_contraction_coverage.csv
relative_functional_contraction_missing_registry.csv
relative_functional_contraction_module_audit.csv
relative_functional_contraction_all_cells.csv
relative_functional_contraction_aggregation_sensitivity.csv
```

关键列：

```text
state_rank_base
state_rank_current
state_rank_delta
absolute_contraction
relative_functional_contraction_module
relative_functional_contraction_equal7
relative_functional_contraction_ratio_of_means_sensitivity
attention_group_relative_contraction
mlp_group_relative_contraction
```

在 output-link 或模型比较表中，`relative_functional_contraction_equal7` 常被重命名为 `c_epsilon`；equal-5 non-QK 口径常写作 `c_equal5`、`C5` 或类似字段名。

### 适用条件与局限

- `c_{\varepsilon}` 是状态几何观测量，不是能力分数，也不单独决定输出改善/恶化；只能作为与 cumulative KL、NLL、行为指标等连接的解释变量。
- 它测的是实际状态 `W_t S_{D,t}`，不得混入 merged-BF16 update rank、`r_epsilon(Delta W)` 或 strict source-principal `p_k`。
- 比较时必须保持 model、probe、sample/token protocol、layer、module、epsilon、current/fixed/centered whitening 口径一致。
- 统计观测单位是 `model × arm × checkpoint × probe × layer × epsilon`，module 行只用于审计和 module-first 聚合，不能把 module 当独立重复样本。
- headline 默认 `epsilon=.05`，但正式稳健性需要保留 `.01/.025/.05/.10`；若某个结论只在单一阈值成立，必须明确标注。
- `c_{\varepsilon}` 与 `C_{\gamma}` 不同：前者关注“需要多少功能方向”的相对压缩，后者关注 eigengap/子空间稳定性及 S/X 传输范数。当前项目优先使用 `c_{\varepsilon}`。

## `C_{\gamma}` 观测方法索引

### 它是什么

本目录沿用 `C_{\gamma}` 指代旧项目中围绕 GetSlice/SVDLLM whitening 谱 gap 得到的容量或稳定性观测量。需要区分两个层次：

- 底层 eigengap：对数据侧 `D` 的 whitened weight `WL_D`，先取奇异值 `sigma_1 >= sigma_2 >= ...`，定义

```text
gamma_k(WL_D) = gap_k(WL_D) = sigma_k(WL_D) - sigma_{k+1}(WL_D)
```

它是 Davis-Kahan top-k 子空间稳定性的分母代理；`gamma_k` 越大，top-k 子空间在扰动下越不容易旋转。旧理论笔记中确认的 "gamma" 指的就是这个 `sigma_k - sigma_{k+1}`，且 `gamma_s.jsonl` 只是 S-probe 文件名，不是指标本身。

- 上层 `C_{\gamma}` / `Gamma gap`：`Readme-dataSplit.md` 中的上层定义把 S/X 两侧 gap 和 profiling matrix 的传输范数组合起来：

```text
C_gamma^{(k)}(X <- S; layer, module)
  = || L_S^{-1} L_X ||_2 * gamma_k(WL_S) / gamma_k(WL_X)
```

其中 `L_S`、`L_X` 是 GetSlice profiling matrix 中对应 `layer/module` 的 scaling matrix，`WL_D = W @ L_D`。若只保留 `sMat/xMat` 奇异值 JSON 而没有保留 profiling matrix `.pt`，只能计算 `gamma_k(WL_D)` 或 `gamma_k(WL_S) / gamma_k(WL_X)`，不能计算完整 `C_{\gamma}`。

`k` 可以固定给出，也可以按谱能量自适应选择：在奇异谱中用 `sum(sigma_i^2)` 的累计能量达到指定比例时的 1-based index 作为 `k`。当前可核验的 Cycle 09 R5 代码使用 `epsilon = 0.05` 和 `0.01`，即 `r_eps_05` 保留 95% 能量、`r_eps_01` 保留 99% 能量；90% 只出现在早期理论笔记中作为示例，不是当前 R5 默认值。

### 代码路径

核心实现位于 `LLM-output-density/GetSlice/`：

- `GetSlice/slice.py`：命令行入口；读取配置，解析 S/X 任务，运行 profiling 与 whitening SVD，输出 `sMat_*.json`、`xMat_*.json`、可选 `sUV_*.pt/xUV_*.pt`。
- `GetSlice/utils/data_utils.py`：严格读取输入 JSONL。X 侧要求每行有 `output.text`；S 侧要求每行有 `question` 和 `answer`。
- `GetSlice/utils/profiling_utils.py`：生成 profiling matrix `L_D`。当前路径使用 SVDLLM-style Cholesky(+jitter) scaling matrix；历史 `gram_mat` 会在加载时转换。
- `GetSlice/utils/profiling_utils.py::whitening`：对每个 `layer/module` 计算 `W_scale = W @ L_D`，并保存其奇异值；这些奇异值就是计算 `gamma_k(WL_D)` 的输入。
- `Readme-dataSplit.md` 第 3.3 节：记录了 `dataAnalyse.gamma` 的上层公式、配置项和输出表名。当前本地快照没有 `run_analysis.py` 或 `dataAnalyse/` 源码，因此可核验路径以直接调用 `GetSlice/slice.py` 为准。

### 如何用 `LLM-output-density` 获取

推荐在保留 profiling matrix 的前提下运行 GetSlice；否则完整 `C_{\gamma}` 缺少 `||L_S^{-1}L_X||_2`。

1. 准备 X 侧 JSONL：

```json
{"output": {"text": "model generated text ..."}}
```

2. 准备 S 侧 JSONL。若 `s_jsonl_path` 是目录，GetSlice 会查找 `{s_jsonl_path}/{task}/{s_jsonl_file}`：

```json
{"question": "...", "answer": "..."}
```

3. 复制并修改 `LLM-output-density/GetSlice/config.json`。关键项：

```json
{
  "model": "/path/to/hf/model",
  "save_path": "/path/to/GetSliceGamma",
  "mode": "split_whitened_svd",
  "DEV": "cuda",
  "model_seq_len": 1024,
  "seed": 3,
  "target_layer": 14,
  "s_nsamples": 32,
  "s_jsonl_path": "/path/to/S_root",
  "s_jsonl_file": "gamma_s.jsonl",
  "x_nsamples": 32,
  "x_jsonl_path": "/path/to/X.jsonl",
  "cleanup_intermediate": false,
  "save_profile_s_path": "profile_s_{task}.pt",
  "save_profile_x_path": "profile_x_{task}.pt",
  "save_s_json_path": "sMat_{task}.json",
  "save_x_json_path": "xMat_{task}.json"
}
```

`mode` 可选值：`s_only_svd` 只算 S；`x_only_svd` 只算全局 X；`split_whitened_svd` 同时算每个 S task 与全局 X。若 `target_layer` 为 `null`，会跑全层；若是整数或 `"layer_14"`，只跑单层。

4. 运行：

```bash
cd <SERVER_BASE>/SOPPO/code/observe/LLM-output-density/GetSlice
python slice.py --config /path/to/config.json
```

5. 读取输出：

```text
{save_path}/{task}/layer_<n>/sMat_<task>.json
{save_path}/X/layer_<n>/xMat_X.json
{save_path}/{task}/layer_<n>/profile_s_<task>.pt
{save_path}/X/layer_<n>/profile_x_X.pt
```

如果没有设置 `target_layer`，路径中不含 `layer_<n>` 子目录。`sMat/xMat` 的结构是 `{layer_key: {module_name: [singular_values...]}}`；profile `.pt` 的结构是 `{layer_idx: {module_name: L_D_tensor}}`。

6. 后处理每个 `layer/module`：

```text
gamma_s = sigma_s[k-1] - sigma_s[k]
gamma_x = sigma_x[k-1] - sigma_x[k]
transfer = || solve(L_S, L_X) ||_2
C_gamma = transfer * gamma_s / gamma_x
```

实现时优先 `torch.linalg.solve(L_S, L_X)`；若 `L_S` 不可解，退回 `torch.linalg.pinv(L_S) @ L_X`。若 `gamma_x <= eps`，该 cell 写为 `NaN`，避免除零。

### 适用条件与局限

- `C_{\gamma}` 是观测/诊断指标，不是当前项目的新实验结论；引用时必须说明来自 `observe/LLM-output-density`。
- S/X 输入必须是同一模型、同一 tokenizer、同一 `model_seq_len`、同一 `layer/module` 下的 profiling 结果，否则 `L_S^{-1}L_X` 与 gap ratio 没有可比性。
- `s_nsamples/x_nsamples`、随机 `seed`、`target_layer`、S/X 文本格式都会影响谱；复现实验时必须记录这些配置。
- `cleanup_intermediate: true` 且未显式设置 `save_profile_*_path` 时，默认 profile `.pt` 可能被清理；这会导致无法计算完整 `C_{\gamma}`。
- `cholesky_jitter` 和 `svd_singular_floor` 是数值稳定参数，会改变病态 Gram 矩阵下的 `L_D`；跨实验比较时应保持一致。
- module 级 `C_{\gamma}` 可以再按 layer、module family、Source/DataSize 聚合；聚合规则必须另行记录，不能把均值、最大值、headline layer 混为同一个指标。
