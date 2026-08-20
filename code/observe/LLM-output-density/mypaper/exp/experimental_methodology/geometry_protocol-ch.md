# 几何探针协议（GetSlice，中文阅读版）

```yaml
view_type: stable_methodology_ch
source_file: geometry_protocol.md
last_organized: 2026-06-16
```

> 本文件是 `geometry_protocol.md` 的中文阅读版，内容与源文件保持一致。本文件讲"几何指标的原始数据怎么取"（喂给模型什么、在哪几层、用哪个探针）；指标本身的定义和解读见 `../../theory/geometry_metric_definitions-ch.md`。

## GetSlice 算的是什么

给定一个模型和一个探针（一组输入/输出文本），GetSlice 做一次前向传播，在目标层/模块上对激活做白化（Cholesky-jitter 协方差白化，`cholesky_jitter=1e-5`，`epsilon=1e-3`，`svd_singular_floor=0.0`），再对白化后的矩阵做 SVD 得到奇异谱（监督/"S"探针存成 `sMat_*.json`，共享/"X"探针存成 `xMat_X.json`）。前向传播用 `model_dtype=float16`，SVD 用 `uv_dtype=float32`。**默认不保存 UV/sketch 产物**（`save_s_uv_path=None`，`save_x_uv_path=None`）——这就是为什么 Cycle 04/05 无法支持 principal-angle/子空间重叠类结论；后续 cycle 如果需要，必须在训练前显式打开。

## 取哪些层、哪些模块

- 每次运行取 3 层，在 Qwen3-1.7B 共 28 层里选早/中/晚：**第 6、14、22 层**（`run_opd_stability_gain_cycle04.py` 中的 `GEOMETRY_LAYERS`）。
- 每层 7 个模块，和 LoRA 的 target modules 完全对应：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`。
- 探针前向传播的序列长度：`seqlen=512`（正式跑），`seqlen=64`（smoke）。每个(模型,探针)对的采样数：`s_nsamples=x_nsamples=16`（正式跑），`4`（smoke）。

## Dual-X / 按模型分的 S 设计

这部分最容易被误读，所以单独写清楚：

- **X 探针是冻结的，所有模型共用同一份。** 两个 X 都只由 `theta0` 生成一次，之后原样复用给每个模型测：
  - `X_prompt`：在探针 prompt 后接上 `theta0` 的 rollout 补全（即"prompt + theta0 的回答"）。
  - `X_bos`：`theta0` 从 BOS 出发的自由生成，完全不带 prompt。
  - 因为 X 是冻结的，不同模型在 X 条件下谱的差异，纯粹来自该模型自己的权重对同一固定输入的反应，不是输入本身的差异。
- **S 探针是按模型分的，反映该模型自己实际产出/训练用的内容，不是共享输入：**
  - `theta0` 的 S = **teacher**（Qwen3-4B）在探针 prompt 上的 rollout。
  - 每个 OPD arm 的 S = **该 arm 自己**在探针 prompt 上的 rollout（自生成、on-policy 文本）。
  - 每个 SFT 对照的 S = **它自己真实训练数据**的一个切片（它实际训练用的监督 teacher 补全文本），完全不是 rollout。
  - 也就是说"相对 `theta0` 的漂移"这件事，在 S 轴上其实是拿每个模型跟一个**类型不同**的基线比（teacher rollout vs 自身 rollout vs 静态训练切片）——这是已知的不对称，不是疏漏，对比 OPD arm 和 SFT arm 的漂移时要记住这一点。
- 探针 prompt 来自 NuminaMath 探针池（正式跑 `n_probe=32`，smoke `n_probe=4`），是 `model_and_data-ch.md` 中统一数据池的一部分。
- 完整交叉矩阵：每个模型都会对所有相关 S 来源、两种 X 变体、3 层 × 7 模块都测一遍——这就是结果文档里"S×model 交叉矩阵"/"三层几何"的含义。

## 输出目录结构

```text
getslice/outputs/{model_source}/step_{size}/S__{probe_source}/numina_math_probe/layer_N/sMat_*.json
getslice/outputs/{model_source}/step_{size}/X__{prompt|bos}/layer_N/xMat_X.json
```

`geometry_reader.py` 把这些 JSON 解析成长表（层、模块、奇异值）；`geometry_metrics.py` 据此算出 `effective_rank`、`spectral_gap`、`spectral_drift_from_theta0_diag`、`X_S_spectrum_level_gap`——具体公式和怎么解读见 `../../theory/geometry_metric_definitions-ch.md`。

## 未确认事项

- 基础 GetSlice 配置里的 `cleanup_intermediate=True` 是否曾经在后续需要重新分析之前就把某个探针的原始激活缓存删掉了，尚未核查；后续 cycle 如果需要算一个 `geometry_metrics.csv` 里还没有的新指标，先确认原始 `sMat_*.json`/`xMat_X.json` 是否还在，再判断是否真的需要重新跑。
