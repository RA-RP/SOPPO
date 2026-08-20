# LLM-output-density 指南

---

## ⚠️ 协议设计原则：先查官方，再查社区，被采纳则关联 related_work

**在设计评测协议或训练格式之前，必须按以下顺序查阅，不得跳过：**

1. **官方技术报告 / 官方文档**：模型的 arXiv 论文、HuggingFace model card、官方评测脚本。
2. **同系列 / 类似大小的其他模型**：若官方没有说明，参考同系列上一代或类似规模模型。
3. **社区实践**：lm-eval-harness GitHub Issues、HuggingFace Discussions、leaderboard 评测说明。

查阅后将来源写入协议文档；**若该来源被采纳进协议或设计，必须同步添加到 `mypaper/related_work/current_source_matrix.md`**——这是整个项目的 related work 文献库，协议来源与论文论点之间需要可追溯的 citation 链。详见 `mypaper/exp/README.md`。

---

## 1. 快速开始

对于在平台运行代码的用户，如果你使用的是autodl可以先执以下命令，如果不是请跳过。

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli login
your token
YOUR_HF_TOKEN
```

这方便后续下载包以及huggingface上的部分数据集

### 1.1 配置环境

> 配置为python3.10是很重要的

```bash
conda create -n density python=3.10 -y
conda activate density
```

安装依赖包

```bash
pip install -r requirements.txt
```

下载底模，你可以根据需求安装其它底模。但是请注意这里的安装路径是重要的，如果需要更改安装路径，请同时修改Density.yaml以及MultiTasks/Density_base.yaml中的base_model地址指向你的模型。

```bash
pip install modelscope
cd LLM-output-density
modelscope download --model Qwen/Qwen2.5-1.5B --local_dir model/Qwen
```

### 1.2 数据集准备与 prepare

当前默认数据集是本地 NuminaMath-1.5：

```text
/root/autodl-tmp/dataset/NuminaMath-1___5
```

原始 parquet 位于：

```text
/root/autodl-tmp/dataset/NuminaMath-1___5/data/*.parquet
```

核心字段为：

```text
problem, solution, answer, problem_type, question_type,
problem_is_valid, solution_is_valid, source, synthetic
```

训练语义是：

```text
input  = problem
target = solution
```

`answer` 不作为 SFT target，只作为 metadata 和下游评测答案。

#### 一次性 split

NuminaMath 不是 lm-eval 内置任务，所以项目先从本地数据中固定切出一个测试集。这个步骤是一次性数据前处理，不属于常规 `prepare` stage。

入口：

```text
Prepare/split_numina_math.py
```

默认输入：

```text
/root/autodl-tmp/dataset/NuminaMath-1___5/data/*.parquet
```

默认输出：

```text
/root/autodl-tmp/prepared/NuminaMath-1___5/
├── train.parquet
├── test.jsonl
└── split_meta.json
```

执行：

```bash
cd /root/LLM-output-density
/root/miniconda3/envs/density/bin/python Prepare/split_numina_math.py --overwrite
```

规则：

- `test.jsonl` 固定 2048 条，只包含 `problem` 和 `answer`。
- 测试集只从 `math-word-problem` 和 `MCQ` 中抽样。
- 测试集排除 `proof`、`notfound`、空答案和无效题目。
- 按 `(source, question_type)` 分层抽样，保留来源分布。
- `train.parquet` 保留原始字段并新增 `row_id`，排除被抽到测试集的行。
- 训练集只过滤明显无效或空 `problem/solution` 样本，proof 样本可以保留用于 SFT。

#### 常规 prepare

常规 `prepare` stage 只负责把主入口明确指定的训练文件缓存成统一 ShareGPT SFT 格式。它不做数据清洗、不扫描目录、不采样截断，也不从训练或评测配置反推数据来源；每个任务的样本数量由输入文件本身决定。

当前默认配置读取：

```text
/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet
```

配置位置：

```json
"prepare": {
  "dataset": {
    "cache_root": "/root/autodl-tmp/prepared/sft_cache",
    "dataset_alias": "sft_train",
    "task_inputs": {
      "NuminaMath-1___5": {
        "input_paths": [
          "/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet"
        ]
      }
    }
  }
}
```

主入口会把 `task_inputs` 写入 `prepare.dataset.tasks`，每个任务固定生成：

```text
/root/autodl-tmp/prepared/sft_cache/{task_id}/
├── train.jsonl
├── dataset_info.json
└── prepare_meta.json
```

`train.jsonl` 使用 ShareGPT 格式：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"answer":"...","source":"...","question_type":"...","row_id":"..."}}
```

NuminaMath 的转换规则是：

```text
problem  -> messages[0].content
solution -> messages[1].content
answer/source/question_type/problem_type/synthetic/row_id -> metadata
```

`dataset_info.json` 注册 LLaMA-Factory 的 `sharegpt` 格式，默认别名为 `sft_train`。`prepare_meta.json` 记录输入 fingerprint、样本数和配置；重跑时 fingerprint 和配置一致会复用缓存，如果输入变化且 `overwrite=false` 会报错。

### 1.3 Baseline 数据筛选

`baseline` stage 位于 `prepare` 之前，用来从 NuminaMath 训练集生成不同策略的训练子集。默认入口：

```bash
cd /root/LLM-output-density
python LLMOutputDensity.py --config config.json --stages baseline prepare train eval_vllm
```

默认方法为：

- `random`：按完整训练集 `source` 分布做分层配额，再稳定随机抽样 N 条。
- `ppl_cond_middle`：只计算 `PPL(solution | problem)`，每个 `source` 内按 PPL 排序后从中间 `70%` 区间稳定抽样；`sizes` 中的 N 是名义训练预算，实际输出约为 `round(N * 0.70)`，训练 epoch 会按 `round(base_epochs * N / actual_count)` 放大。
- `diversity`：用训练配置中的 `cutoff_len` 截断 `problem + solution`，取 layer14 hidden states 做 attention-mask mean pooling 和样本向量 L2 normalize，再用 MiniBatchKMeans 聚类；在 source 配额内尽量覆盖不同簇后抽样 N 条。
- `random_cfs`：先生成 N 条 `random` base，再额外混入 `round(N * cfs_ratio)` 条 CFS；CFS 标记为独立 `source: cfs`，训练 epoch 会按 base/total 比例缩小。

默认输出：

```text
/root/autodl-tmp/baseline/NuminaMath-1___5/{task_id}/
├── selected.parquet
├── cfs.jsonl                  # 仅 random_cfs
└── selection_meta.json
```

`baseline` 会自动把生成的任务注入 `prepare.dataset.tasks`、`train.tasks` 和 `eval.tasks`，继续复用现有 prepare/train/eval 主干。`ppl_cond_middle` 和 `diversity` 的长度不单独配置，统一读取对应训练 size 的 `cutoff_len`；GetSlice 的 `model_seq_len` 本轮仍保持自身配置。

### 1.4 Train/SFT

训练阶段已经从旧 `LoRA/` 目录迁移到 `Train/`：

```text
Train/
├── train.py
├── Lora.yaml
├── Full.yaml
└── components/
    └── train_runner.py
```

训练入口：

```bash
cd /root/LLM-output-density
python Train/train.py --config config.json
```

主流程中也可以通过 `LLMOutputDensity.py` 或 `MultiDensity.py` 调用 `train` stage。核心配置在 `config.json`：

```json
"train": {
  "method": "lora",
  "config_paths": {
    "lora": "Train/Lora.yaml",
    "full": "Train/Full.yaml"
  },
  "finetune_root": "/root/autodl-tmp/finetuneOutput/Qwen2.5",
  "merge_root": "/root/autodl-tmp/mergeOutput/Qwen2.5"
}
```

`train.method` 支持：

- `lora`：读取 `Train/Lora.yaml`，运行 LLaMA-Factory SFT，选择 best checkpoint，再执行 export merge，最终模型写到 `merge_root`。
- `full`：读取 `Train/Full.yaml`，运行全参 SFT，不执行 merge；训练产物直接在 `finetune_root` 下作为 full finetune 输出使用。

每个任务的输出结构：

```text
/root/autodl-tmp/finetuneOutput/Qwen2.5/{dataset}/{max_samples}/
├── finetune_{task_id}.yaml
├── train_meta.json
├── trainer_state.json
└── checkpoint-*/
```

LoRA 额外输出：

```text
/root/autodl-tmp/mergeOutput/Qwen2.5/{dataset}/{max_samples}/
├── merge_{task_id}.yaml
└── merged model files
```

`train_meta.json` 记录训练方法、base model、prepare cache、训练配置 hash、输出路径、best checkpoint 和 merge 信息。它是后续热重启、审计和 eval 定位模型的重要依据。

只生成训练配置、不启动训练：

```bash
python Train/train.py --config config.json --dry-run
```

也可以临时切换方法：

```bash
python Train/train.py --config config.json --method full
```

### 1.5 Eval/Test

当前默认下游任务是本地 NuminaMath 测试集，不再使用 GSM8K 作为默认 ID 任务。

本地 lm-eval task 定义：

```text
Eval/tasks/
├── numina_math_local.yaml
└── numina_math_utils.py
```

`numina_math_local.yaml` 使用 lm-eval 原生本地任务机制：

```yaml
task: numina_math_local
dataset_path: json
dataset_kwargs:
  data_files:
    test: /root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl
output_type: generate_until
process_results: !function numina_math_utils.process_results
```

主配置：

```json
"eval": {
  "lm_tasks": ["numina_math_local"],
  "include_path": "Eval/tasks",
  "vllm_model_len": 4096
}
```

评测入口：

```text
Eval/allRunEval_vLLM.py
```

它会把 `eval.include_path` 传给 lm-eval：

```bash
lm_eval --tasks numina_math_local --include_path Eval/tasks ...
```

答案判断逻辑在 `numina_math_utils.py`：

- 优先提取模型输出中最后一个 `\boxed{...}`。
- 用括号计数解析 boxed 内容，支持 `\frac{1}{2}`、`\sqrt{\frac{3}{5}}` 这类嵌套 LaTeX。
- 如果没有 boxed，回退到 `Final Answer:`、`Answer:` 或最后一行。
- 对预测和 gold 做轻量 normalize，包括去 `$`、`\$`、逗号、空白、末尾标点、`\left/\right` 等。
- 第一版指标仍叫 `exact_match`，便于结果汇总。

评测输出：

```text
/root/autodl-tmp/exp0506/origin/_combined/{dataset}_{max_samples}.json
/root/autodl-tmp/exp0506/origin/numina_math_local/{dataset}_{max_samples}.json
/root/autodl-tmp/exp0506/csv_results/numina_math_local_results.csv
```

### 1.6 运行
如果你要下载 Hugging Face 模型或继续运行内置 Hugging Face benchmark，需要先登录 Hugging Face。当前默认 NuminaMath 本地评测本身不需要从 Hub 下载测试集。

```bash
huggingface-cli login
<your token>
```

我们提供两个运行入口

- 最基础的入口 **LLMOutputDensity.py** 根据 `Density.yaml` 或 `config.json` 中的配置执行指定 stage。
- 批量化运行入口 **MultiDensity.py** 读取 `config.json`，按 tasks 分批生成 `Density.yaml`，再调用 `LLMOutputDensity.py`。

推荐先运行一次 split，然后按 stage 执行：

```bash
cd /root/LLM-output-density
/root/miniconda3/envs/density/bin/python Prepare/split_numina_math.py --overwrite
```

```bash
python LLMOutputDensity.py --config config.json --stages baseline prepare train eval_vllm
```

批量入口：

```bash
python MultiDensity.py --config config.json --stages baseline prepare train eval_vllm
```

这是一个完整的流程，你可以选择部分执行。

数据fix单独执行的命令行

```bash
/root/miniconda3/envs/density/bin/python -c '... from Eval.component.Eval2Res import eval2res ...; config["eval"]["lm_tasks"] = ["mmlu", "truthfulqa"]; eval2res(config)'
```

### 1.7 结果输出

这里只介绍最终结果的存储位置，项目运行过程中产生的矩阵在2.3项目结构进一步介绍

## dataAnalyse：exp0509 精简分析流水线

`dataAnalyse` 现在只负责“已经得到的数据”的整理、展示和相关性分析，不再负责 ER、DeltaW、SVD consistency、INVScore、scaling law、Goodhart 等旧分析。需要从模型输出或矩阵结果中计算新特征时，优先放到 `MyFunc/analyse` 或其它对应模块中完成，再把结果 CSV 交给 `dataAnalyse` 合并。

当前默认实验根目录为：

```text
/root/autodl-tmp/exp0509
```

所有输入和输出路径都基于这个 `{root}`。默认底模目录名为 `Qwen2.5`，统一主键为：

```text
Source, DataSize
```

其中当前 `Source` 包含旧模型名和统一后的 `dataset_0601..0604`、`dataset_0701..0704`。

### 当前代码结构

```text
dataAnalyse/
├── main.py
└── config.json

MyFunc/analyse/
├── __init__.py
├── er_features.py
└── main.py
```

- `dataAnalyse/main.py`：精简主入口，生成主表、宏观图、相关性面板、相关性 CSV 和热图。
- `dataAnalyse/config.json`：当前实验根目录、底模名、ER run 目录、WSI 权重、谱截断参数和 `extra_feature_csvs` 接口。
- `MyFunc/analyse/er_features.py`：从 GetSlice 输出中整理 ER、谱偏斜、相对 X 的谱 KL 和 UV 旋转角，生成可被 `dataAnalyse` 合并的特征表。

已经删除旧的 `dataAnalyse/components`，旧 stage1/stage2 组织方式不再作为当前分析入口。

### 执行方式

默认读取 `dataAnalyse/config.json`：

```bash
cd /root/LLM-output-density
/root/miniconda3/envs/density/bin/python dataAnalyse/main.py
```

也可以显式指定实验根目录、底模和 ER run：

```bash
cd /root/LLM-output-density
/root/miniconda3/envs/density/bin/python dataAnalyse/main.py \
  --root /root/autodl-tmp/exp0509 \
  --base-model Qwen2.5 \
  --er-run-dir offPolicyData
```

增加未来额外特征 CSV 时：

```bash
/root/miniconda3/envs/density/bin/python dataAnalyse/main.py \
  --extra-feature-csvs analyseOutput/tables/your_extra_features.csv
```

`extra_feature_csvs` 中的路径可以写绝对路径，也可以写相对 `{root}` 的路径。CSV 只要包含 `Source, DataSize` 和数值列，就会自动进入相关性分析。

### 输入数据

| 数据类型 | 读取位置 | 读取规则 | 当前状态 |
| --- | --- | --- | --- |
| Benchmark | `{root}/csv_results/*.csv` | 读取 GSM8K、MMLU、Winograd、TruthfulQA、Ethics 的 CSV，并统一到 `Source, DataSize` | 已实现 |
| Eval Loss | `{root}/finetuneOutput/{base_model}/{Source}/{DataSize}/trainer_state.json` | 从 `log_history` 中选择最小 `eval_loss`，同时记录对应 step | 已实现 |
| ER 基础数据 | `{root}/offPolicyData/ranked_dataset_er.csv` | 兼容读取历史 source 级 ER 聚合结果；当前优先用 GetSlice source 级 `sMat` 重新计算 | 兼容回退 |
| ER chunk 明细 | `{root}/offPolicyData/dataSets/{Source}/chunk_quota.csv` 或旧 `ranked_chunk_datasets` | 兼容历史 chunk 级分析；exp0509 当前主流程直接使用 `dataSets/{Source}` 的 source 级 GetSlice 输出 | 兼容回退 |
| GetSlice S 谱数据 | `{root}/offPolicyData/GetSliceOutput/{Source}/layer_14/sMat_{Source}.json` | 读取 source 级奇异值，计算 `EffectiveRank`、`SpectralSkewness` 和相对 X 的 `SigmaKL_*` | 已实现 |
| GetSlice S UV 数据 | `{root}/offPolicyData/GetSliceOutput/{Source}/layer_14/sUV_{Source}.pt` | 读取 source 的 U/S/VT，用于和 X 计算 U、VT 旋转角 | 已实现 |
| GetSlice X 基线 | `{root}/offPolicyData/GetSliceOutput/X/layer_14/xMat_X.json` 和 `xUV_X.pt` | 使用 `x_jsonl_path` 指向的全局 X 数据作为参考分布/参考方向 | 已实现 |
| analyseMat 特征 | `{root}/analyseMat/differenceMat/{Source}/{DataSize}/...` | 未来从指定 layer 和模块中计算一个或多个数值特征，再按 `Source, DataSize` 合并 | 未完成 |
| AnalyseMat 相关工作证据 | `{root}/analyseMat/principalEvidence/principal_evidence.csv` | 针对配置指定单层，计算 update mask、principal mask overlap 和 principal angle | 已实现 |
| 通用额外特征 | 任意 CSV | 包含 `Source, DataSize` 与数值列即可，通过 `extra_feature_csvs` 合并 | 接口已预留 |

当前 benchmark 指标映射如下：

| 文件 | 输入列 | 输出列 |
| --- | --- | --- |
| `gsm8k_results.csv` | `Strict_Match` | `GSM8K` |
| `mmlu_results.csv` | `results_mmlu_acc` | `MMLU` |
| `winogrande_results.csv` | `Accuracy` | `Winograd` |
| `truthfulqa_results.csv` | `results_truthfulqa_mc1_acc,none` | `TruthfulQA_MC1` |
| `truthfulqa_results.csv` | `results_truthfulqa_mc2_acc,none` | `TruthfulQA_MC2` |
| `ethics_results.csv` | `ethics_cm_acc` | `Ethics_CM` |
| `ethics_results.csv` | `ethics_deontology_acc` | `Ethics_Deontology` |
| `ethics_results.csv` | `ethics_justice_acc` | `Ethics_Justice` |
| `ethics_results.csv` | `ethics_utilitarianism_acc` | `Ethics_Utilitarianism` |
| `ethics_results.csv` | `ethics_virtue_acc` | `Ethics_Virtue` |

### 主表与 WSI

主表输出到：

```text
{root}/analyseOutput/tables/source_datasize_metrics.csv
```

当前列为：

```text
Source, DataSize, EvalLoss, BestStep,
GSM8K,
MMLU, Winograd, TruthfulQA_MC1, TruthfulQA_MC2,
Ethics_CM, Ethics_Deontology, Ethics_Justice, Ethics_Utilitarianism, Ethics_Virtue,
WSI
```

主表只保留 Eval Loss 和 Benchmark 在 `Source, DataSize` 上的交集。

WSI 当前只用于 OOD 指标，不包含 GSM8K。当前公式为：

```text
WSI(row) = sum_i weight_i * max(0, reference_i - value_i) / reference_i
```

其中 `i` 是 OOD 指标，`reference_i` 默认是当前表中该指标的最大值；如果未来存在 `Source == "human"` 且 `DataSize` 为最大数据量，则优先用 human 行作为参考。WSI 越低越好，`0` 表示所有参与计算的 OOD 指标都达到或超过参考值。

当前 WSI 权重为：

| 指标 | 权重 |
| --- | --- |
| `MMLU` | `1.0` |
| `Winograd` | `1.0` |
| `TruthfulQA_MC1` | `0.75` |
| `TruthfulQA_MC2` | `0.75` |
| `Ethics_CM` | `0.3` |
| `Ethics_Deontology` | `0.3` |
| `Ethics_Justice` | `0.3` |
| `Ethics_Utilitarianism` | `0.3` |
| `Ethics_Virtue` | `0.3` |

当前 OOD 指标都是 accuracy 类指标，默认数值越大越好。如果之后加入 loss、error rate 等数值越小越好的指标，需要先转换方向，或者给 WSI 增加 metric direction 配置。

### ER 特征输出

`MyFunc/analyse` 会输出两个表：

```text
{root}/analyseOutput/tables/er_features.csv
{root}/analyseOutput/tables/er_feature_modules.csv
```

`er_features.csv` 是按 `Source, DataSize` 聚合后的表：

```text
Source, DataSize,
EffectiveRankMean, EffectiveRankStd, EffectiveRankMin, EffectiveRankMax,
SpectralSkewnessMean, SpectralSkewnessStd,
SigmaKL_top50Mean, SigmaKL_bottom50Mean, SigmaKL_allMean,
UAngleDeg_top50Mean, VTAngleDeg_top50Mean,
...Std,
ModuleCount, FeatureKeyMode
```

当前 ER 是 source-only 特征，没有真实 DataSize。因此会复制到该 Source 在主表中实际存在的所有 DataSize，并标记：

```text
FeatureKeyMode = source_only_replicated
```

`er_feature_modules.csv` 是模块明细表：

```text
Source, ChunkId, Layer, Module,
EffectiveRank, SpectralSkewness,
SigmaKL_top50, SigmaKL_bottom50, SigmaKL_all,
UAngleDeg_top50, VTAngleDeg_top50,
UsedCount, SMatPath, UVPath
```

其中 `EffectiveRank` 使用奇异值归一化后的熵指数：

```text
p_i = s_i / sum_j s_j
EffectiveRank = exp(-sum_i p_i * log(p_i))
```

`SpectralSkewness` 当前定义为前 `top_k` 个奇异值占总奇异值和的比例：

```text
SpectralSkewness = sum(top_k singular values) / sum(all singular values)
```

`SigmaKL_*` 比较 source 的奇异值分布相对于 X 的变化，当前方向为：

```text
KL(source || X) = sum_i p_source(i) * log(p_source(i) / p_X(i))
```

其中 `p` 是截断后奇异值归一化得到的分布。KL 会按 `p_source(i)` 加权，因此 source 中权重高的位置变化会被放大；如果 source 在某个位置权重大而 X 权重很小，贡献会明显变大。`MyFunc/analyse` 本身不内置这些截断值，当前由 `dataAnalyse/config.json` 控制，默认 `topK_sigma = [50, -50, 0]`，分别对应：

```text
top50, bottom50, all
```

`UAngleDeg_*` 和 `VTAngleDeg_*` 比较 source 与 X 的奇异向量方向。U 使用 U 的列向量，VT 使用 VT 的行向量；由于 SVD 向量有符号不唯一性，计算角度时使用 `abs(dot)` 消除正负号翻转影响，再对截断范围内的向量角度取平均。当前由 `dataAnalyse/config.json` 控制，默认 `topK_degree = [50]`。

这几类特征仍然是 source-only 特征，会按主表中同 Source 的所有 DataSize 复制，并用 `FeatureKeyMode = source_only_replicated` 标记。相关性计算检测到这个标记后，会按每个 DataSize 单独计算矩阵，避免复制数据被当成独立样本。

### GetSlice 配置说明

当前 exp0509 实际使用的配置不是模板文件 `MyFunc/GetSlice/config.json`，而是：

```text
/root/autodl-tmp/exp0509/offPolicyData/getslice_config.json
```

关键配置为：

```text
save_path = /root/autodl-tmp/exp0509/offPolicyData/GetSliceOutput
s_jsonl_path = /root/autodl-tmp/exp0509/offPolicyData/dataSets
x_jsonl_path = /root/LLM-output-density/MyFunc/dataset/X/Qwen.jsonl
mode = split_whitened_svd
target_layer = 14
save_s_json_path = sMat_{task}.json
save_s_uv_path = sUV_{task}.pt
save_x_json_path = xMat_{task}.json
save_x_uv_path = xUV_{task}.pt
skip_existing_outputs = true
cleanup_intermediate = true
```

GetSlice 的热启动是最终输出级别：如果某个 task 的所有配置输出都存在且非空，就跳过该 task。默认中间 profile/cache 会在最终输出写完后删除，因此中断在中间步骤时通常需要重跑该 task。

当前 cluster 版本的 GetSlice 输入并不是从完整 cluster 中直接采样，也不是先随机选单条记录。流程是先在每个 cluster 中通过 `stage_sample` 生成 `er_samples/cluster_xxxx` 子集，再由 GetSlice 读取该子集内所有文本，拼接成一个整体文本池，然后随机截取 `model_seq_len` token window。这个语义会让长文本更容易被采到，采样窗口也可能跨记录边界；后续如果要改成记录级采样，需要单独调整 GetSlice 数据构造逻辑。

### 宏观图

宏观图输出到：

```text
{root}/analyseOutput/figures/
```

当前输出：

| 文件 | 含义 |
| --- | --- |
| `macro_metrics_over_datasize.png` | 所有宏观指标随 DataSize 变化的总览图 |
| `macro_GSM8K.png` | GSM8K 随 DataSize 变化 |
| `macro_EvalLoss.png` | EvalLoss 随 DataSize 变化 |
| `macro_WSI.png` | WSI 随 DataSize 变化 |
| `macro_MMLU.png`、`macro_Winograd.png` 等 | 各 OOD 指标随 DataSize 变化 |

### 相关性分析

相关性输出到：

```text
{root}/analyseOutput/correlation/
```

分析入口会先保存完整相关性面板：

```text
{root}/analyseOutput/correlation/correlation_panel.csv
```

相关性不是写死指标名，而是基于 `correlation_panel.csv` 的表格内容自动选择：

- 选择所有数值列。
- 排除 `DataSize`。
- 非数值列会自然排除，例如 `Source`、`FeatureKeyMode`。
- 全空列和常数列会自动排除。
- 对剩余列同时计算 Pearson 和 Spearman。

由于 ER 是复制到每个 DataSize 的 source-only 特征，为避免复制数据导致伪相关，当前会按每个 DataSize 单独计算相关性。输出示例：

```text
pearson_datasize_200.csv
pearson_datasize_200.png
spearman_datasize_200.csv
spearman_datasize_200.png
pearson_datasize_800.csv
spearman_datasize_800.csv
pearson_datasize_1800.csv
spearman_datasize_1800.csv
```

如果未来合并的所有特征都有真实 `Source, DataSize` 粒度，且没有 `source_only_replicated` 标记，则会计算全局相关性：

```text
pearson_global.csv
spearman_global.csv
```

### 当前不再保留的旧功能

以下旧 `dataAnalyse` 功能已经从当前入口中移除：

- `prepare_data` 分阶段中间表。
- `stage1/stage2` 目录结构。
- scaling laws。
- Goodhart/alignment 分析。
- DeltaW dynamics 图片。
- SVD consistency。
- INVScore。
- 旧的 `Model, DataSize` 命名体系。

如果之后确实需要恢复其中某个能力，建议作为独立模块重新接入，并输出包含 `Source, DataSize` 的特征 CSV，而不是恢复旧的 stage pipeline。

### AnalyseSVD 删除状态

`AnalyseSVD` 已从仓库中删除，也不再接入 `LLMOutputDensity.py` 主入口。当前主实验链路中的 SVDLLM 风格谱提取使用 `MyFunc/GetSlice`。后续与相关工作比较所需的矩阵格式和数值修正，应集中放在 `AnalyseMat` 中完成，并由 `dataAnalyse` 消费最终 CSV。

### AnalyseMat 相关工作证据

`AnalyseMat` 现在可以额外计算一类用于相关工作比较的参数空间证据。它不读取历史 `differenceMat` JSON，而是直接读取底模和微调后模型的 `.npy` 权重，针对配置指定的一层计算：

- `update mask`：将 `W0` 和 `W+` 都转成 `bfloat16` 后比较，不同的位置视为可见更新。
- `principal mask`：对底模权重 `W0` 做 SVD，使用 rank-k 重构矩阵绝对值的 top ratio/top count 位置。
- `mask overlap`：统计 update mask 和 principal mask 的重合程度。
- `principal angles`：比较 `W0` 与 `W+` 的 top-k 左/右奇异子空间旋转角度。

配置示例：

```yaml
analyse:
  related_work:
    enable: true
    target_layer: 0
    target_modules: null
    principal_rank_k: 50
    principal_top_ratio: 0.01
    principal_top_count: null
    update_mask_rule: bf16
    save_png: true
    plot_max_side: 768
    output_root: null
```

默认只分析 `target_layer` 下所有二维 `weight` 矩阵，自动跳过 bias、layernorm 等一维张量。`target_modules` 可以设为模块白名单，例如：

```yaml
target_modules: ["self_attn_q_proj_weight", "self_attn_o_proj_weight"]
```

输出目录默认是 `analyse.output_root` 的 sibling：

```text
/root/autodl-tmp/analyseMat/principalEvidence
```

主要输出：

```text
principal_evidence.csv
details/{Source}/{DataSize}/layer_{N}_{module}.json
figures/{Source}/{DataSize}/layer_{N}_{module}.png
```

`principal_evidence.csv` 包含 `Source, DataSize, Layer, Module` 以及 overlap、density、lift、Jaccard、U/V principal angle 统计列。要接入 `dataAnalyse`，把它作为额外特征 CSV 传入即可：

```bash
/root/miniconda3/envs/density/bin/python dataAnalyse/main.py \
  --extra-feature-csvs /root/autodl-tmp/analyseMat/principalEvidence/principal_evidence.csv
```

注意：`principal_evidence.csv` 是模块级表。如果 `target_modules=null`，同一个 `Source, DataSize` 下会有多行；若需要严格按 `Source, DataSize` 唯一主键进入 `dataAnalyse`，请先指定单个模块，或后续将模块级结果聚合成唯一主键表。

### 待完成事项

1. 通用 `analyseMat/differenceMat` 对齐接口还未实现。当前已实现的是相关工作证据 `principal_evidence.csv`；如果之后需要从 `{root}/analyseMat/differenceMat/{Source}/{DataSize}` 的历史 JSON 中抽取其它通用特征，还需要单独补接口。
2. 通用 `{root}/GetSliceOutput/{Source}` 与 `{root}/GetSliceOutput/{Source}/{DataSize}` 两种读取方式尚未作为当前主入口实现。exp0509 当前不保留 `GetSliceOutput` 中间产物，之后需要时应重新生成。
3. ID 能力指标暂未保留。当前主表包含下游任务 GSM8K 和 OOD 指标，不包含 ID 指标。
4. WSI 暂未支持每个指标的方向配置。当前假设参与 WSI 的 OOD 指标都是越大越好。
5. 相关性指标选择已经是表格驱动，但如果某些数值列不应参与相关性，例如 `BestStep`，之后可以增加 `exclude_correlation_columns` 配置。

## GetSlice：SVDLLM 风格激活谱提取

`MyFunc/GetSlice` 用于对现有 `S` / `X` 数据执行 SVDLLM step1 风格的 profiling 与 whitening。它不会训练模型，只读取数据、前向模型、累积每层每个 Linear 的输入激活矩阵，并输出奇异值与奇异向量。

### 执行入口

```bash
cd /root/LLM-output-density/MyFunc/GetSlice
python slice.py --config config.json
```

### 执行流程

流程分为两步：

1. `profiling`

   对每个 layer、每个 Linear，hook 输入激活 `X`，累积：

   ```text
   raw_scaling_diag_matrix = sum X^T X
   ```

   profiling 阶段的浮点数使用与 SVDLLM 保持一致：激活缓存跟随模型 dtype，hook 中用 `float32` 计算并累积 Gram，分解前再转为 `float64`。然后优先使用 SVDLLM 风格的 Cholesky：

   ```text
   profiling_mat = cholesky(raw_scaling_diag_matrix)
   ```

   如果 Cholesky 失败，则仿照 SVDLLM 加入正定补偿项：

   ```text
   lambda = -min_eig(raw_scaling_diag_matrix) + cholesky_jitter
   profiling_mat = cholesky(raw_scaling_diag_matrix + lambda I)
   ```

   如果第二次 Cholesky 仍失败，才使用 `eigh` 作为最终兜底。

2. `whitening`

   对每个 Linear 权重 `W` 计算：

   ```text
   W_scale = W @ profiling_mat
   U, S, VT = svd(W_scale)
   ```

   `sMat/xMat` JSON 只保存奇异值 `S`，`sUV/xUV` PT 保存完整 `U/S/VT`。

### 运行模式

在 `config.json` 中通过 `mode` 控制：

```json
"mode": "s_only_svd"
```

可选值：

- `s_only_svd`：只执行 S。按 `tasks` 枚举任务，每个 task 单独输出到 `{save_path}/{task}`。
- `x_only_svd`：只执行全局 X 一次，输出到 `{save_path}/X`。
- `split_whitened_svd`：先按 `tasks` 执行所有 S，再执行全局 X 一次。X 不会随每个 task 重复计算。

### 任务与数据路径

`tasks` 支持单个、数组和空值：

```json
"tasks": "Grok4_fast"
```

```json
"tasks": ["Grok4_fast", "Gemini3"]
```

```json
"tasks": null
```

当 `tasks` 为 `null`、空字符串或空数组时，会枚举 `s_jsonl_path` 下所有包含 `s_jsonl_file` 的子目录。

S 数据路径按下面规则拼接：

```text
{s_jsonl_path}/{task}/{s_jsonl_file}
```

例如：

```json
"s_jsonl_path": "/root/LLM-output-density/MyFunc/dataset/S",
"s_jsonl_file": "gsm8k.jsonl"
```

会读取：

```text
/root/LLM-output-density/MyFunc/dataset/S/Grok4_fast/gsm8k.jsonl
```

X 数据是全局数据文件：

```json
"x_jsonl_path": "/root/LLM-output-density/MyFunc/dataset/X/Qwen.jsonl"
```

### 关键参数

```json
"model": "/root/LLM-output-density/model/Qwen",
"save_path": "/root/autodl-tmp/exp0504/GetSliceOutput",
"DEV": "cuda",
"model_seq_len": 1024,
"seed": 3,
"target_layer": null,
"layer_gpu_chunk_size": 1,
"single_layer_task_group_size": 4,
"s_nsamples": 64,
"x_nsamples": 64,
"activation_cache_device": "cuda",
"uv_dtype": "float32",
"svd_singular_floor": 0.0,
"cholesky_jitter": 0.00001,
"cleanup_intermediate": true,
"skip_existing_outputs": true
```

参数说明：

- `target_layer`：为 `null` 时保持全层流程；设为整数或 `"layer_N"` 时只对指定 transformer 层执行 profiling 与 whitening，层号从 0 开始。
- `layer_gpu_chunk_size`：仅单层模式使用，控制 replay 时每次把多少个连续 transformer 层放入 GPU；`1` 等价原 streaming 行为，调大可减少搬运但更占显存。
- `single_layer_task_group_size`：仅单层 S 多任务模式使用，控制每组 pending task 数量；可设为正整数或 `"all"`，最后一组可小于该值。
- `activation_cache_device`：`cuda` 更快，`cpu` 更省显存。
- `uv_dtype`：保存 `U/VT` 的 dtype，默认 `float32`。`S` 保存为 CPU tensor。
- `cholesky_jitter`：Cholesky 失败时的 SVDLLM 风格正定补偿，默认 `0.00001`。
- `svd_singular_floor`：只在最终 `eigh` 兜底时作为特征值下界，默认 `0.0`。
- `cleanup_intermediate`：为 `true` 时会删除默认 token cache 和默认 profiling_mat 中间文件；如果要保留 profiling_mat，请设为 `false` 或显式指定 `save_profile_s_path/save_profile_x_path`。
- `skip_existing_outputs`：为 `true` 时，如果某个 task 或全局 X 的最终输出文件已经完整存在且非空，就直接跳过；设为 `false` 可强制重跑。

### 输出文件

默认输出路径都在：

```text
{save_path}/{task}/
```

如果启用单层模式，默认输出路径会增加一层：

```text
{save_path}/{task}/layer_{N}/
```

S 输出示例：

```text
/root/autodl-tmp/exp0504/GetSliceOutput/Grok4_fast/sMat_Grok4_fast.json
/root/autodl-tmp/exp0504/GetSliceOutput/Grok4_fast/sUV_Grok4_fast.pt
```

全局 X 输出示例：

```text
/root/autodl-tmp/exp0504/GetSliceOutput/X/xMat_X.json
/root/autodl-tmp/exp0504/GetSliceOutput/X/xUV_X.pt
```

单层输出示例：

```text
/root/autodl-tmp/exp0504/GetSliceOutput/Grok4_fast/layer_5/sMat_Grok4_fast.json
/root/autodl-tmp/exp0504/GetSliceOutput/Grok4_fast/layer_5/sUV_Grok4_fast.pt
/root/autodl-tmp/exp0504/GetSliceOutput/X/layer_5/xMat_X.json
```

输出配置：

```json
"save_s_json_path": "sMat_{task}.json",
"save_x_json_path": "xMat_{task}.json",
"save_s_uv_path": "sUV_{task}.pt",
"save_x_uv_path": "xUV_{task}.pt"
```

这些字段也可以写成绝对路径；如果是相对路径，则默认放到 `{save_path}/{task}` 下，单层模式则放到 `{save_path}/{task}/layer_{N}` 下。

## offPolicyData：基于聚类/分块与可配置指标的数据筛选前处理

`offPolicyData` 用于把现有 S 数据从“直接按来源训练”改成“先聚类或分块、再按可配置指标排序、最后合成候选训练集”的实验前处理流程。它不修改原始数据、不覆盖 `MyFunc/GetSlice/config.json`，也不调用 `dataAnalyse/main.py`；打分与排序逻辑在 `offPolicyData/components/common.py` 中独立完成。

整体流程：

```text
读取 S 数据 QA
-> 用本地 Qwen 抽取 QA embedding
-> 按每类约 100 条 QA 聚类
-> 每个类保存为独立数据集目录
-> 每类混合采样代表样本
-> 自动生成独立 GetSlice 配置并运行 GetSlice
-> 从 sMat JSON 计算每类/每块指标
-> 按 ranking metric 的容量感知正态分布合成 K 个候选数据集
```

### 目录结构

```text
offPolicyData/
├── config.json
└── run_pipeline.py
```

运行后输出到：

```text
{output_root}/{run_name}/
```

例如当前配置：

```json
"output_root": "/root/autodl-tmp/exp0506/offPolicyData/runs",
"run_name": "qwen_layer_cluster_er"
```

实际输出根目录为：

```text
/root/autodl-tmp/exp0506/offPolicyData/runs/qwen_layer_cluster_er
```

### 执行方式

建议使用项目的 `density` 环境：

```bash
cd /root/LLM-output-density
/root/miniconda3/envs/density/bin/python offPolicyData/run_pipeline.py \
  --config offPolicyData/config.json \
  --stages all
```

第一次运行建议分阶段执行，方便检查中间产物：

```bash
cd /root/LLM-output-density

/root/miniconda3/envs/density/bin/python offPolicyData/run_pipeline.py \
  --config offPolicyData/config.json \
  --stages embed cluster sample getslice_config
```

确认自动生成的 `getslice_config.json` 无误后，再运行较重的 GetSlice 与聚合步骤：

```bash
/root/miniconda3/envs/density/bin/python offPolicyData/run_pipeline.py \
  --config offPolicyData/config.json \
  --stages getslice er aggregate
```

可选阶段：

| stage | 功能 | 主要产物 |
| --- | --- | --- |
| `embed` | 读取 QA 并用 Qwen 抽 embedding | `records_manifest.jsonl`, `embeddings.npy` |
| `cluster` | MiniBatchKMeans 聚类并做容量约束分配 | `clusters/`, `clusters.csv`, `cluster_labels.npy` |
| `sample` | 每类抽代表样本送入 ER 分析 | `er_samples/`, `er_samples.csv` |
| `getslice_config` | 生成独立 GetSlice 配置副本 | `getslice_config.json` |
| `getslice` | 调用 `MyFunc/GetSlice/slice.py` | `GetSliceOutput/` |
| `er` | 从 `sMat_*.json` 独立计算类级/块级指标并排序 | `cluster_er.csv`, `cluster_er_modules.csv` |
| `aggregate` | 按当前 ranking metric 的 rank 合成 K 个候选数据集 | `ranked_datasets/dataset_*/` |

如果同一个 `run_name` 已经有非空输出，默认会停止，避免旧结果混入新结果。重新实验时建议改 `run_name`；确认要覆盖时可设置：

```json
"overwrite_outputs": true
```

### Ranking Metric

默认保持旧行为，按 EffectiveRank 从大到小排序：

```json
"ranking": {
  "metric": "effective_rank",
  "direction": "desc",
  "topk_kl": {
    "top_k": 50
  }
}
```

如果希望用相对 X 的谱 KL 进行筛选：

```json
"ranking": {
  "metric": "topk_KL",
  "direction": "desc",
  "topk_kl": {
    "top_k": 50
  }
}
```

`topk_KL` 使用：

```text
KL(source || X)
```

其中 source 是当前 chunk/cluster 的 `sMat_*.json`，X 是 `GetSliceOutput/X/**/xMat_X.json`。`top_k` 支持：

```text
50      -> 前 50 个奇异值
-50     -> 后 50 个奇异值
0/all   -> 全部奇异值
```

当 `metric = "topk_KL"` 时，`getslice_config` 阶段会自动把 GetSlice mode 设置为 `split_whitened_svd`，生成 X 基线 `xMat_X.json`；UV 输出会默认关闭，避免生成不需要的大 `.pt` 文件。输出 CSV 会继续保留旧的 `er_rank` 和 ER 列，同时新增 `rank_metric`、`rank_score`、`rank_score_mean`、`rank_score_std`。这里的 `er_rank` 为兼容旧代码保留，含义是“当前 ranking metric 的排序名次”。

### 输入数据

输入来自：

```json
"source_root": "/root/LLM-output-density/MyFunc/dataset/S",
"sources": ["GLM_5", "GPT5_4", "GPT5_4_nano", "Grok4", "Grok4_fast"],
"task_file": "gsm8k.jsonl"
```

实际读取路径为：

```text
{source_root}/{source}/{task_file}
```

每条记录要求至少包含：

```json
{"question": "...", "answer": "..."}
```

脚本会清洗 answer 中的 `__HEREDOC__` / `__APPEND__` 占位符为 GSM8K 风格的 `<<...>>`，并保留 `source`、`source_line`、`global_id` 等追踪字段。

### Embedding 与聚类

embedding 配置：

```json
"embedding": {
  "model_path": "/root/LLM-output-density/model/Qwen",
  "device": "cuda",
  "dtype": "float16",
  "batch_size": 8,
  "max_length": 512,
  "layer": "getslice_target"
}
```

每条 QA 会拼成：

```text
Question: {question}
Answer: {answer}
```

然后送入本地 Qwen 完整 forward，并通过 `output_hidden_states=True` 取指定层的 hidden states 做 mean pooling 与 L2 normalize。这里的指定层不是“只跑这一层”：如果 `layer=14`，文本会经过 embedding 层和 transformer layer 0 到 14，再取 layer 14 输出。

`layer` 支持：

| 取值 | 含义 |
| --- | --- |
| `"getslice_target"` | 默认值，读取 GetSlice base config 中的 `target_layer`；若其为 `null` 则退回最后一层 |
| `"input"` | 只取 token embedding 输出 |
| `"last"` | 取最后一层输出 |
| 整数 | 取对应 transformer 层输出，层号从 0 开始 |

聚类配置：

```json
"clustering": {
  "target_cluster_size": 100,
  "batch_size": 1024,
  "max_iter": 100
}
```

- `target_cluster_size` 控制每类约多少条 QA。脚本先计算 `n_clusters = ceil(N / target_cluster_size)`，再把样本按距离中心的偏好做容量约束分配，让每类尽量接近目标大小。
- `batch_size` 和 `max_iter` 是 `MiniBatchKMeans` 的训练参数，只影响聚类中心训练，不影响 Qwen embedding batch、GetSlice `s_nsamples` 或最终数据集大小。

每个类会保存为兼容数据集结构：

```text
clusters/cluster_0000/
├── gsm8k.jsonl
├── gsm8k-train.jsonl
├── dataset_info.json
└── cluster_meta.json
```

其中 `gsm8k-train.jsonl` 是旧 GSM8K 流程下的训练视图，`gsm8k.jsonl` 保留追踪 metadata。新版 Math-CoT SFT 流程不再直接读取这种 source 目录；需要由主入口把要训练的数据文件显式写入 `prepare.dataset.tasks[*].input_paths`，再由常规 `prepare` 缓存成固定的 ShareGPT `train.jsonl`。

### 类代表样本与 GetSlice

采样配置：

```json
"sampling": {
  "class_sample_size": null,
  "center_fraction": 0.5
}
```

- `class_sample_size=null` 表示继承 GetSlice base config 中的 `s_nsamples`。例如原 GetSlice 配置里 `s_nsamples=32`，则每类最多抽 32 条代表样本。
- `center_fraction=0.5` 表示一半样本取离类中心最近的 QA，另一半从剩余 QA 中按固定 seed 随机抽取。

代表样本保存为：

```text
er_samples/cluster_0000/
├── gsm8k.jsonl
├── gsm8k-train.jsonl
└── dataset_info.json
```

GetSlice 配置：

```json
"getslice": {
  "base_config": "/root/LLM-output-density/MyFunc/GetSlice/config.json",
  "config_out": "getslice_config.json",
  "output_dir": "GetSliceOutput",
  "script": "/root/LLM-output-density/MyFunc/GetSlice/slice.py"
}
```

脚本会读取 `base_config`，生成独立副本：

```text
{output_root}/{run_name}/getslice_config.json
```

原始 `MyFunc/GetSlice/config.json` 不会被覆盖。独立副本会继承原配置中的 `model`、`target_layer`、`s_nsamples`、`model_seq_len`、`layer_gpu_chunk_size` 等参数，只覆盖：

```json
"s_jsonl_path": "{output_root}/{run_name}/er_samples",
"s_jsonl_file": "gsm8k.jsonl",
"tasks": ["cluster_0000", "cluster_0001", "..."],
"mode": "s_only_svd",
"save_path": "{output_root}/{run_name}/GetSliceOutput"
```

因此 `getslice.output_dir` 的完整路径是：

```text
{output_root}/{run_name}/{getslice.output_dir}
```

### 类级 ER 计算

`er` 阶段会读取 GetSlice 输出的 `sMat_*.json`，不进入 `dataAnalyse/main.py`。每个模块的 EffectiveRank 公式与 `dataAnalyse` 保持一致：

```text
p_i = abs(s_i) / sum(abs(s_i))
EffectiveRank = exp(-sum(p_i * log(p_i)))
```

类级代表 ER 为该类 `sMat` 中所有目标层模块 `EffectiveRank` 的均值。输出：

```text
cluster_er.csv
cluster_er_modules.csv
```

`cluster_er.csv` 主要列：

| 列 | 含义 |
| --- | --- |
| `cluster_id` | 类 ID |
| `er_rank` | 按 `cluster_er_mean` 从高到低排序后的 rank，0 表示最高 ER 类 |
| `cluster_er_mean` | 类级 ER 均值 |
| `cluster_er_std/min/max` | 模块级 ER 的统计量 |
| `module_count` | 参与该类 ER 聚合的模块数 |

### 合成候选数据集

聚合配置：

```json
"aggregation": {
  "num_datasets": 4,
  "dataset_size": 3000,
  "min_per_class": 1,
  "capacity_alpha": 0.5,
  "active_classes": null,
  "active_class_multiplier": 1.5,
  "rank_sigma": null,
  "sigma_coverage_z": 3.0
}
```

输出为 K 个候选数据集：

```text
ranked_datasets/
├── dataset_001/
├── dataset_002/
├── dataset_003/
└── dataset_004/
```

每个 `dataset_xxx` 都包含：

```text
gsm8k.jsonl
gsm8k-train.jsonl
dataset_info.json
source_stats.csv
cluster_quota.csv
ranking_meta.json
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `num_datasets` | 生成多少个候选数据集 |
| `dataset_size` | 每个候选数据集包含多少条 QA |
| `min_per_class` | 每类至少贡献多少条，保证覆盖所有类 |
| `active_class_multiplier` | 自动控制主要采样范围的倍数 |
| `active_classes` | 手动指定主要覆盖多少个类；为 `null` 时自动计算 |
| `capacity_alpha` | 类大小权重指数，`0` 表示不考虑类大小，`1` 表示完全按类容量加权 |
| `rank_sigma` | 手动指定 ER rank 空间中的 sigma；为 `null` 时自动计算 |
| `sigma_coverage_z` | 自动 sigma 的覆盖系数，默认 `3.0` 表示用约 `±3σ` 覆盖主要活跃类 |

自动采样范围计算：

```text
median_class_size = median(每类样本数)
min_needed_classes = ceil(dataset_size / median_class_size)
active_classes = ceil(min_needed_classes * active_class_multiplier)
```

第 `j` 个候选数据集的 ER rank 中心：

```text
mu_j = j * (C - 1) / (num_datasets - 1)
```

其中 `C` 是类数量，`j` 从 0 开始。因此：

```text
dataset_001       -> 中心在最高 ER 类
dataset_middle    -> 中心在中间 ER 类
dataset_last      -> 中心在最低 ER 类
```

如果 `rank_sigma=null`，脚本会根据 `active_classes` 和 `sigma_coverage_z` 自动反推 sigma，并考虑边界情况。例如 `dataset_001` 的中心在最左侧，没有更高 ER 的左侧类，sigma 会只向右展开，避免半个正态分布落到无效区域。

每类最终权重：

```text
gaussian_weight_i = exp(-0.5 * ((rank_i - mu_j) / sigma)^2)
capacity_weight_i = class_size_i ^ capacity_alpha
final_weight_i = gaussian_weight_i * capacity_weight_i
```

配额分配：

```text
quota_i = min_per_class + floor(剩余名额 * final_weight_i / sum(final_weight))
quota_i <= class_size_i
```

如果某些类容量不足，未用完的名额会继续分配给还有容量的类，直到每个候选数据集严格达到 `dataset_size`。

`cluster_quota.csv` 用于检查每类实际贡献：

| 列 | 含义 |
| --- | --- |
| `cluster_id` | 类 ID |
| `er_rank` | 类 ER 排名 |
| `cluster_er_mean` | 类级 ER |
| `class_size` | 类内总样本数 |
| `gaussian_weight` | 只由 ER rank 正态分布给出的权重 |
| `capacity_weight` | 类大小权重 |
| `final_weight` | 二者相乘后的最终权重 |
| `quota` / `used_count` | 该类被分配并实际使用的 QA 数 |
| `source_counts` | 该类被选中样本的来源统计 |

`source_stats.csv` 用于检查每个合成数据集来自哪些源：

```text
dataset_id,source,count,ratio
dataset_001,Grok4,742,0.2473
dataset_001,GLM_5,611,0.2037
...
```

### 后续接入 LoRA/Eval

每个合成候选数据集要送入新版 LoRA/SFT 时，应先作为显式输入交给 `prepare`，由 `prepare` 写入固定缓存目录：

```yaml
prepare:
  dataset:
    tasks:
      - task_id: dataset_001
        input_paths:
          - /path/to/dataset_001.parquet
        dataset_alias: sft_train
```

随后 LoRA/SFT 只读取 `/root/autodl-tmp/prepared/sft_cache/dataset_001/train.jsonl` 和同目录下的 `dataset_info.json`。如果仍在使用旧 GSM8K/offPolicyData 实验脚本，可以保留旧 source 目录结构；但新版 Math-CoT SFT 流程不再依赖 `dataset_root` 扫描。

后续 LoRA/Eval 不由 `offPolicyData` 自动执行。推荐先分别训练：

```text
dataset_001: 偏高 ER 类
dataset_002/003: 中间 ER 类
dataset_004: 偏低 ER 类
```

再用原有 LoRA、Eval、dataAnalyse 流程比较最终指标，判断类级 ER 排序是否真的对应更好的训练数据选择。
