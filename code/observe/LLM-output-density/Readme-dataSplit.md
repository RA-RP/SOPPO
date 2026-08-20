# 代码流程及其配置文件

## 1. baseline / build_sources 阶段

这个阶段负责把原始大数据集切成后续训练真正会消费的小训练集。它是一次性的 CPU/GPU 数据构造阶段，入口是 `run_build_sources.py`。

### 1.1 术语

`baseDataset`：原始基础数据集，例如 `NuminaMath-1___5`。它来自 `configs/global.yaml` 里的 `baseDatasets`，每个 entry 需要包含 `name`、`source_prefix`、`train_path`。

`Source`：真正用于训练、eval、analyse 的训练数据源。它是 build_sources 阶段物化出来的数据集名，例如：

```text
NuminaMath-random
NuminaMath-ppl_mid70
NuminaMath-diversity
NuminaMath-cfs50
NuminaMath-offPolicyData_kl_high
NuminaMath-offPolicyData_kl_mid
NuminaMath-offPolicyData_kl_low
```

其中 `kl_high` 表示取 topk_KL 较高的样本，`kl_low` 表示取 topk_KL 较低的样本。这里的 topk_KL 分数先在簇上计算，再回填给簇内样本，最终 high/mid/low 是样本级分档，不是简单按簇切分。

`DataSize`：实验标签中的训练样本量，来自 `configs/global.yaml` 的 `sizes`。除 `cfs` 外，每个方法最终训练样本数都等于 `DataSize`；`cfs` 会额外混入 `round(DataSize * cfs.ratio)` 条 CFS 样本。

`NuminaMath_source`：`train.parquet` 每条样本内部的来源列，用于按来源保持比例、分来源聚类和分来源抽样。它不是 manifest 里的训练 `Source`。

### 1.2 运行顺序

NuminaMath 不是 lm-eval 内置任务，所以先手动从本地数据中固定切出训练池和测试集：

```bash
python Prepare/split_numina_math.py --config configs/numina_math_split.yaml
```

然后运行数据构造入口：

```bash
python run_build_sources.py --config configs/global.yaml
```

最后 GPU 机器消费 manifest：

```bash
python run_gpu.py --config configs/global.yaml
```

如果只跑某个基础数据集，可以使用：

```bash
python run_gpu.py --config configs/global.yaml --base-datasets NuminaMath-1___5 --sizes 100
```

### 1.3 配置来源

`configs/global.yaml` 是主配置：

- `home`：实验产物根目录。
- `models`：模型维度，例如 `Qwen2.5` 及其 `base_model_path`。
- `baseDatasets`：原始基础数据集列表，每项包含 `name`、`source_prefix`、`train_path`。
- `sizes`：默认 DataSize 列表，例如 `[100, 200]`。
- `stages`：GPU pipeline 执行阶段。

`configs/stages/build_sources.yaml` 控制数据构造阶段：

- `data_construction.output_root`：统一 sources 输出根目录，当前是 `{home}/sources`。
- `data_construction.manifest_path`：manifest 路径，当前是 `{home}/sources/manifest.json`。
- `data_construction.run.methods`：默认运行哪些构造方法，当前是 `[baseline, offPolicyData]`。
- `data_construction.methods.baseline.methods`：baseline 子方法列表，当前是 `[random, ppl_cond_middle, diversity, cfs]`。
- `data_construction.methods.baseline.output_root`：baseline 产物目录，当前是 `{home}/baseline`。
- `data_construction.methods.baseline.diversity.target_cluster_size`：按每个 `NuminaMath_source` 聚类时的目标簇大小。
- `data_construction.methods.offPolicyData.pipeline`：offPolicyData 后端，当前是 `cluster_kl`。
- `data_construction.methods.offPolicyData.config_path`：offPolicyData 详细配置，当前是 `offPolicyData/config.json`。

`offPolicyData/config.json` 控制 KL 排序流程：

- `pipeline`：当前为 `cluster_kl`。
- `task_file`：当前为 `dataset.jsonl`。旧流程中出现过的 `gsm8k.jsonl` 是历史命名，新产物不应再使用。
- `kl_sample`：从共享簇中抽多少样本用于 GetSlice。
- `getslice`：GetSlice 的基础配置、输出目录、脚本路径和 X 数据路径。
- `ranking.metric`：当前排序指标为 `topk_KL`。
- `ranking.topk_kl.top_k`：计算 topk_KL 时使用的 top K。
- `aggregation.bands`：固定为 `["high", "mid", "low"]`。
- `aggregation.dataset_size`：配置文件中保持为 `0`，表示 offPolicyData 的指标阶段不绑定某个 DataSize。实际聚合时由 `run_build_sources.py` 按 `configs/global.yaml` 的 `sizes` 注入。

### 1.4 产物

baseline 产物写在：

```text
{home}/baseline/
```

每个方法和 size 会生成一个独立目录，例如：

```text
{home}/baseline/NuminaMath-random__100/
{home}/baseline/NuminaMath-diversity__200/
```

主要文件：

- `selected.parquet`：baseline 主训练样本。
- `cfs.jsonl`：仅 `cfs` 使用，表示额外拼接进最终训练集的 CFS 样本。
- `selection_meta.json`：记录配置、输入指纹、实际样本数、来源分布、cache 路径等信息。
- `baseline_tasks_meta.json`：baseline 本次生成的 task 汇总。

cache 目录：

- `{home}/baseline/_ppl_cache/`：PPL 打分缓存。
- `{home}/baseline/_diversity_cache/`：diversity embedding 缓存。
- `{home}/baseline/_diversity_cluster_cache/{baseDataset}/{cache_hash}/`：按 `NuminaMath_source` 分组的共享聚类缓存，baseline diversity 和 offPolicyData 共用。

offPolicyData 产物写在：

```text
{home}/offPolicyData/runs/
```

`cluster_kl` 会读取共享聚类缓存，运行 GetSlice，计算 topk_KL，并固定聚合出 `kl_high`、`kl_mid`、`kl_low` 三个 ranked datasets。

offPolicyData 的 run 目录不带 DataSize，例如：

```text
{home}/offPolicyData/runs/NuminaMath_offpolicy_topk_kl/
```

这个目录表示一套可复用的 offPolicyData 指标结果，不应该再出现 `NuminaMath_offpolicy_topk_kl_size100` 这类按 size 命名的主 run 目录。

其中这些目录和文件是 size-independent 指标产物，只需要计算一次：

```text
clusters/
kl_samples/
GetSliceOutput/
cluster_kl.csv
cluster_kl_modules.csv
clusters_index.json
```

`clusters/` 是从 baseline diversity 的共享聚类缓存物化出来的簇任务；`kl_samples/` 是每个簇中用于 GetSlice/KL 估计的样本任务。它们的任务文件名都应是 `dataset.jsonl`。

不同 DataSize 只在下面生成不同抽取结果：

```text
{home}/offPolicyData/runs/NuminaMath_offpolicy_topk_kl/ranked_datasets/size100/kl_high/
{home}/offPolicyData/runs/NuminaMath_offpolicy_topk_kl/ranked_datasets/size200/kl_high/
```

因此，新增 `DataSize=500` 时，正常情况下只需要复用已有 `cluster_kl.csv` 再生成：

```text
{home}/offPolicyData/runs/NuminaMath_offpolicy_topk_kl/ranked_datasets/size500/
```

不应该重新聚类，也不应该重新运行 GetSlice 或重新计算 KL，除非输入训练池、base model、GetSlice/X 配置、聚类参数或 KL 参数发生变化。

统一 manifest 写在：

```text
{home}/sources/manifest.json
```

manifest entry 里核心字段包括：

- `task_id`：`{Source}__{DataSize}`。
- `source` / `dataset`：真实训练 Source。
- `base_source`：原始 baseDataset。
- `size`：DataSize。
- `train_sample_count`：真实训练样本数，`cfs` 会大于 `size`。
- `method`：构造方法，例如 `baseline_random`、`baseline_diversity`、`offPolicyData_cluster_kl`。
- `input_paths`：prepare 阶段要读取的训练样本路径。
- `dataset_alias`：写入 LLaMA Factory 数据集时使用的 alias。

### 1.5 baseline 子方法

`random`：按 `NuminaMath_source` 的原始比例随机抽样。最终训练样本数等于 `DataSize`。

`ppl_cond_middle`：先计算 PPL，再按 PPL 排序取中间区间。`middle_fraction=0.70` 会生成 Source `NuminaMath-ppl_mid70`。最终训练样本数等于 `DataSize`。

`diversity`：先计算 embedding，再对每个 `NuminaMath_source` 单独聚类，随后按来源比例做多样性抽样。`target_cluster_size` 控制大致簇大小和簇数量，不控制最终每个来源抽多少条；最终每个 `NuminaMath_source` 的样本数由 `DataSize` 和原始来源比例决定。`kmeans_batch_size/max_iter/n_init` 控制 MiniBatchKMeans。

`cfs`：先从原始训练池抽 `DataSize` 条，再额外拼接 `round(DataSize * cfs.ratio)` 条 CFS。比如 `cfs.ratio=0.5` 且 `DataSize=100`，最终训练样本数是 `150`，manifest 中 `size=100`、`train_sample_count=150`。

### 1.6 offPolicyData 子方法

当前 offPolicyData 入口由 `run_build_sources.py` 调用，配置来自 `configs/stages/build_sources.yaml` 和 `offPolicyData/config.json`。

`cluster_kl` 的流程是：

1. 读取 baseline diversity 生成的共享 cluster cache。
2. 将共享 cluster cache 物化成 `clusters/`，并从每个簇抽样生成 `kl_samples/`。
3. 调用 GetSlice 得到 S/X 相关中间结果。
4. 计算每个簇的 `topk_KL` 排序分数，写入 `cluster_kl.csv`。
5. 将簇分数回填到簇内样本，在每个 `NuminaMath_source` 内按样本级 KL 从高到低切成 `high/mid/low` 三段。
6. 每个 band 按原始 `NuminaMath_source` 比例抽到 `DataSize` 条。
7. 将 `NuminaMath-offPolicyData_kl_high/mid/low` 注册进 manifest。

offPolicyData 与 baseline 是同级比较方法。它不是 baseline 的子方法，也不是 GPU loop 的一部分；它和 random、ppl_mid70、diversity、cfs 一样，最终都通过 manifest 暴露为真实训练 Source。

注意：第 1 到第 4 步与 DataSize 无关，应复用已有产物；第 5 到第 7 步才按不同 DataSize 生成训练数据。baseline 的 PPL 和 diversity embedding/cluster 也使用 cache，默认 `cutoff_len=4096`，不会因为 DataSize 不同而重复打分。

如果已经有：

```text
{home}/offPolicyData/runs/NuminaMath_offpolicy_topk_kl/cluster_kl.csv
```

再次运行：

```bash
python run_build_sources.py --config configs/global.yaml --methods offPolicyData --sizes 100 200
```

应看到日志提示复用 KL 指标，并只补齐或复用 `ranked_datasets/size100`、`ranked_datasets/size200`。

### 1.7 与后续 GPU pipeline 的衔接

`run_gpu.py` 先按 `baseDatasets × sizes` 构造 entry filters。这里的 `(baseDataset, DataSize)` 只是 manifest 筛选条件，不是 GPU 执行任务。随后 `resolve_entries()` 会查 manifest，把 filters 展开为真实训练 Source entries。每个 baseDataset/DataSize filter 可能展开为多个真实训练 Source，例如 random、diversity、cfs、kl_high、kl_mid、kl_low。

展开后的真实 entries 按 `gpu_batch_size` 分批执行。配置优先级：

```text
CLI --batch-size > global.yaml 的 gpu_batch_size > 默认 8
```

`gpu_batch_size` 的单位是**真实训练 Source 数量**，不是 baseDataset×DataSize 对数量。例如 `gpu_batch_size: 1` 且一个 baseDataset 对应 7 个 Source 时，会分成 7 个 batch，每次只跑 1 个 Source。

batch 循环的核心流程：

```text
entry_filters = product(baseDatasets, sizes)       # manifest 筛选条件，不是执行任务
all_entries = resolve_entries(eff, entry_filters)  # 展平为真实 Source×DataSize entries
for batch in chunks(all_entries, batch_size):      # 按 gpu_batch_size 分批
    batch_cfg = inject_resolved_entries(eff, batch)  # 注入 tasks 到 cfg
    for stage in stages:
        stage_dispatch[stage](batch_cfg)
```

后续目录和结果表也遵循这个含义：

- 模型输出目录中的 Source 是真实训练 Source。
- eval CSV 中的 `Source` 是真实训练 Source。
- eval CSV 中的 `DataSize` 是实验标签 `size`。
- 如需记录原始基础数据集，应使用 `BaseDataset` 字段，不要复用 `Source`。

### 1.8 常见误区

`configs/global.yaml` 顶层 `sources` 已废弃，必须使用 `baseDatasets`。

`NuminaMath_source` 不是训练 Source，它是 `train.parquet` 内部每条样本的来源字段。

`run_gpu.py` 不会生成 baseline 或 offPolicyData 数据。缺少 `{home}/sources/manifest.json` 时，应先运行 `run_build_sources.py`。

GPU loop 的执行单位不是 `baseDataset × DataSize`，而是 manifest 中真实的 `Source × DataSize` entry。`baseDataset × DataSize` 只用于筛选 manifest。

offPolicyData 主 run 目录不应带 DataSize。`.../NuminaMath_offpolicy_topk_kl_size100/` 是旧结构；新结构中 size 只能出现在 `ranked_datasets/size100/` 这一层。

offPolicyData 的任务文件不应再叫 `gsm8k.jsonl`。新结构统一使用 `dataset.jsonl`，避免把通用数学数据集误写成 GSM8K。

`analyseMat` 阶段的 `extract_safetensors_to_npy` 曾有递归 bug：当底模 npy 目录不存在时，函数递归调用自身尝试提取底模，但递归在 npy 目录创建之前发生，导致无限递归。已修复为内联提取，不再递归。

## 2. GPU loop：prepare/train/eval/analyseMat/delete

### 2.1 prepare

prepare 阶段负责把 manifest entry 指向的训练样本文件转换成 LLaMA Factory 可直接读取的 ShareGPT SFT cache。入口是 `Prepare/allPrepare.py` 的 `run_prepare()`，核心实现是 `Prepare/component/prepareDataset.py`。

prepare 不负责 baseline/offPolicyData 的数据构造，也不决定跑哪些 Source。真实 Source 和 DataSize 已经在 `run_build_sources.py` 写入 manifest，随后由 `run_gpu.py` 的 `inject_resolved_entries()` 注入到当前 batch 的配置里：

```text
prepare.dataset.tasks:
  - task_id: NuminaMath-random__100
    input_paths:
      - {home}/baseline/NuminaMath-random__100/selected.parquet
    dataset_alias: sft_train
```

主要配置文件是 `configs/stages/prepare.yaml`：

- `prepare.dataset.enable`：是否执行 SFT cache 构造。
- `prepare.dataset.cache_root`：prepare 产物根目录，当前是 `{home}/prepared/sft_cache`。
- `prepare.dataset.dataset_alias`：写进 `dataset_info.json` 的数据集别名，默认 `sft_train`。
- `prepare.dataset.overwrite`：是否重建已有 cache。
- `prepare.dataset.task_inputs`：无 manifest fallback 时使用的输入路径；正常 manifest 流程中会被当前 batch 覆盖。
- `prepare.eval_cache`：可选的 benchmark 预下载配置，不影响训练数据 prepare。

读入文件来自 manifest entry 的 `input_paths`。这些文件通常是：

```text
{home}/baseline/<Source>__<DataSize>/selected.parquet
{home}/baseline/<Source>__<DataSize>/cfs.jsonl
{home}/offPolicyData/runs/<run>/ranked_datasets/size<DataSize>/<band>/selected.parquet
```

prepare 支持读取 `.parquet`、`.jsonl`、`.json`。它会把 NuminaMath 的 `problem/solution` 或 `question/answer`，以及已有 ShareGPT 风格的 `messages`，统一规范成：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

训练用 `train.jsonl` 只保留 `messages`。baseline/offPolicyData 的逐条来源字段不会写进训练 JSONL 的顶层字段，避免 Hugging Face `datasets` 在读取混合来源样本时因为 metadata schema 不一致而失败；这些逐条 provenance 会旁路保存到 `record_metadata.jsonl`。

输出目录按 `task_id` 存放：

```text
{home}/prepared/sft_cache/<task_id>/
├── train.jsonl
├── record_metadata.jsonl
├── dataset_info.json
└── prepare_meta.json
```

其中 `dataset_info.json` 注册 `sft_train -> train.jsonl`，供 LLaMA Factory 训练读取；`record_metadata.jsonl` 与 `train.jsonl` 行号一一对应，只用于追溯来源，不参与训练读取；`prepare_meta.json` 记录输入文件 fingerprint、输出格式版本、样本数等信息。train 阶段会校验这里的样本数是否和 manifest 中的 `train_sample_count` 一致，尤其是 `cfs` 这种真实训练样本数大于 `DataSize` 的情况。

注意：`prepare_meta.json` 中的 `request.format_version` 当前为 `4`。这个版本修复了 `cfs` 训练集的 metadata schema 冲突：旧版 `train.jsonl` 会把 base 样本和 CFS 样本的不同 metadata struct 写进同一个 JSONL，Hugging Face `datasets` 读取时可能报 `Couldn't cast array of type struct...`。如果某个旧 SFT cache 仍是旧格式，prepare 会提示格式版本变化；此时只需要重建对应 task 的 prepare cache，不需要重跑 baseline/offPolicyData 数据构造。

重建 prepare cache 时，不需要重跑 `run_build_sources.py`。常用做法是临时把 `configs/stages/prepare.yaml` 中的 `prepare.dataset.overwrite` 改为 `true`，然后只跑 prepare stage：

```bash
python run_gpu.py --config configs/global.yaml --stages prepare
```

这个命令会按当前 `global.yaml` 的 `baseDatasets × sizes` filters 展开 manifest，并重建命中的真实 `Source × DataSize` prepare cache。重建完成后应把 `prepare.dataset.overwrite` 改回 `false`，避免后续 GPU loop 反复重建缓存。

如果直接跑完整 `python run_gpu.py --config configs/global.yaml`，会按 manifest 展开当前配置命中的所有真实 `Source × DataSize` entry，并继续执行 train/analyse/eval/delete。

此外，评测 benchmark 可以提前缓存，入口是：

```bash
python Prepare/component/prefetchEvalTasks.py --config configs/global.yaml --offline false
```

这个命令会读取 layered `configs/global.yaml`，默认使用 `eval.lm_tasks`；如果没有 `eval.lm_tasks`，则从 `eval.target_metrics[].json_task` 推导任务。它只预下载/检查 lm-eval 所需 benchmark 缓存，不生成训练 Source，也不修改 manifest。

也可以使用 `--no-offline`，它等价于 `--offline false`。`--overwrite` 只表示忽略本项目写出的 `eval_cache_meta.json` 并重新执行一次缓存检查，不会主动清空 Hugging Face datasets/hub cache；已经完整缓存的数据集通常仍会被 Hugging Face 复用。

### 2.2 train

train 阶段负责对当前 batch 中每个真实 `Source × DataSize` entry 启动一次 SFT，并把可加载模型物化到统一目录，供后续 `analyseMat` 和 `eval_vllm` 使用。入口是 `Train/train.py`，核心实现是 `Train/components/train_runner.py`。

训练使用 LLaMA Factory，实际命令由代码生成：

```text
llamafactory-cli train <finetune_yaml>
```

train 阶段不再自己决定数据来源。它读取的是 `run_gpu.py` 注入后的：

```text
train.tasks:
  - dataset: NuminaMath-random
    max_samples: 100
    task_id: NuminaMath-random__100
    train_sample_count: 100
```

其中：

- `dataset`：真实训练 Source，用于目录命名。
- `max_samples`：实验标签 DataSize，用于目录命名和 eval/analyse 标签。
- `task_id`：对应 prepare cache 的目录名。
- `train_sample_count`：真实训练样本数；普通方法等于 `DataSize`，`cfs` 会大于 `DataSize`。

主要配置分两层：

- `configs/stages/train.yaml`：编排配置，决定 `train.method`、模板路径、输出根目录。
- `Train/Full.yaml` / `Train/Lora.yaml`：LLaMA Factory 原生训练参数模板，包括 seed 以外的 batch size、epoch、learning rate、cutoff_len、eval/save/logging step 等。

`configs/stages/train.yaml` 当前关键字段：

- `train.method`：`full` 或 `lora`。
- `train.config_paths.full`：全参训练模板，当前是 `Train/Full.yaml`。
- `train.config_paths.lora`：LoRA 训练模板，当前是 `Train/Lora.yaml`。
- `train.finetune_root`：训练过程输出目录，当前是 `{home}/finetuneOutput/{model}`。
- `train.model_output_root`：统一可加载模型目录，当前是 `{home}/modelOutput/{model}`。
- `train.merge_root`：旧 LoRA 时代的兼容字段，后续不应作为 eval/analyse 的主入口。

每个任务会生成：

```text
{home}/finetuneOutput/{model}/{Source}/{DataSize}/
├── finetune_<task_id>.yaml
├── train_meta.json
├── trainer_state.json
├── checkpoint-*/
└── log_output/

{home}/modelOutput/{model}/{Source}/{DataSize}/
└── 可被 vLLM / analyseMat 加载的模型文件
```

`finetuneOutput` 是训练过程目录，保存日志、checkpoint 和训练记录；`modelOutput` 是后续 eval/analyseMat 的统一模型输入目录。`deleteMid` 默认会在当前 batch 完成后删除 `modelOutput` 和大体积训练权重，只保留轻量训练记录。

GPU loop 现在带 per-entry 热重启状态机。`run_gpu.py` 会先按 manifest 展开真实 `Source × DataSize` entry，再逐个检查：

- `finetuneOutput/{model}/{Source}/{DataSize}/train_meta.json` 或 `all_results.json`：判断训练是否完成过。
- `modelOutput/{model}/{Source}/{DataSize}`：判断可加载模型是否仍存在。
- `finetuneOutput/.../checkpoint-*` 或 `trainer_state.json.best_model_checkpoint`：判断能否从 checkpoint 物化或 resume。
- `csv_results/{model}/target_metrics_results.csv`：判断 eval target metrics 是否已经完成。
- `analyseMat/{model}/principalEvidence/principal_evidence.csv`：判断 PE 是否已经完成。
- `analyseMat/{model}/npy/{Source}/{DataSize}`：判断是否能跳过 safetensors→npy，只补 PE。

可先用 dry-run 查看决策，不执行任何 stage：

```bash
python run_gpu.py --config configs/global.yaml --resume-dry-run
```

常见决策包括：

- `skip`：train/eval/PE 已完成，且模型权重已被清理，直接跳过。
- `deleteOnly`：train/eval/PE 已完成，但 `modelOutput` 还在，只执行 `deleteMid`。
- `resultOnly`：已有 eval JSON，但 CSV 缺失，只重建 `target_metrics_results.csv`，不要求模型目录存在。
- `materializeThenEval`：训练完成且 checkpoint 仍在，但 `modelOutput` 缺失，先物化模型再补 eval/analyse。
- `resumeTrain`：训练未完成但 checkpoint 可用，从 checkpoint resume。
- `fullRestart`：缺少可恢复产物，从可用的最早阶段重新开始。

注意：`all_results.json` 只代表训练曾经完成，不代表模型权重还在。若 `deleteMid` 已经删除了 `modelOutput` 和 checkpoint，而 eval/PE 又缺失，则代码会避免盲目假装可恢复；对于已经有 eval 结果但缺少 PE 且无法恢复权重的旧任务，会跳过并在 dry-run 中标记为 `skipMissingModel`。

分为两个子功能。

#### 2.2.1 Full

Full 表示全参数 SFT。当前 `configs/stages/train.yaml` 中 `train.method: full` 时，会读取 `Train/Full.yaml`。

`Train/Full.yaml` 中的关键训练参数包括：

- `finetuning_type: full`
- `cutoff_len: 4096`
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 16`
- `learning_rate: 1.0e-5`
- `num_train_epochs: 3`
- `bf16: true`
- `flash_attn: fa2`
- `val_size: 0.1`
- `save_total_limit: 2`
- `load_best_model_at_end: true`
- `metric_for_best_model: eval_loss`

Full 的读入数据是 prepare 阶段生成的：

```text
{home}/prepared/sft_cache/<task_id>/train.jsonl
{home}/prepared/sft_cache/<task_id>/dataset_info.json
```

训练时会生成本次任务专属的 `finetune_<task_id>.yaml`，其中会写入：

```text
model_name_or_path: <base_model>
dataset: sft_train
dataset_dir: {home}/prepared/sft_cache/<task_id>
output_dir: {home}/finetuneOutput/{model}/{Source}/{DataSize}
```

Full 训练结束后，代码会优先从 `trainer_state.json` 读取 best checkpoint；如果没有明确 best checkpoint，则回退到最新 `checkpoint-*`。随后把该 checkpoint 复制到：

```text
{home}/modelOutput/{model}/{Source}/{DataSize}
```

因此后续 `eval_vllm` 和 `analyseMat` 不需要知道这是 Full 还是 LoRA，只读取统一的 `modelOutput`。

#### 2.2.2 LoRA

LoRA 表示参数高效微调。将 `train.method` 改成 `lora` 时，会读取 `Train/Lora.yaml`。

`Train/Lora.yaml` 中的关键训练参数包括：

- `finetuning_type: lora`
- `lora_target: all`
- `lora_rank: 16`
- `cutoff_len: 1024`
- `per_device_train_batch_size: 8`
- `gradient_accumulation_steps: 4`
- `learning_rate: 3.0e-5`
- `num_train_epochs: 20`
- `bf16` 未显式写入，按 LLaMA Factory 和环境默认行为处理。

LoRA 的读入数据和 Full 一样，也是 prepare cache：

```text
{home}/prepared/sft_cache/<task_id>/train.jsonl
{home}/prepared/sft_cache/<task_id>/dataset_info.json
```

LoRA 训练结束后会先选择 best checkpoint，然后额外生成：

```text
{home}/finetuneOutput/{model}/{Source}/{DataSize}/merge_<task_id>.yaml
```

并执行：

```text
llamafactory-cli export <merge_yaml>
```

export 会把 base model + LoRA adapter 合并成可直接加载的完整模型，输出到：

```text
{home}/modelOutput/{model}/{Source}/{DataSize}
```

这就是当前统一接口的关键：Full 是复制 best/full checkpoint 到 `modelOutput`，LoRA 是 export merged model 到 `modelOutput`。两者后续都走同一个 eval/analyseMat 路径。


### 2.3 analyseMat
这个阶段负责从训练后的模型权重中提取 safetensors 转 npy，并可选执行 principalEvidence（单层主成分证据分析）。入口是 `AnalyseMat/analyseMat.py` 的 `run_analyse()`。

`loader.py` 会将配置中的 `analyseMat` 自动映射为 `analyse`，因此代码内部统一读 `config["analyse"]`。

分为两个子步骤：

#### 2.3.1 safetensors → npy 提取（Step 1，总是执行）

从 `{home}/modelOutput/{model}` 读取训练后的模型 safetensors，逐层转为 numpy `.npy` 文件，输出到 `{home}/analyseMat/{model}/npy`。

如果底模 npy（`{home}/originMat/{model}`）不存在，会自动从 `base_model`（底模 safetensors 路径）提取。底模提取是内联执行的，不经过递归。

配置在 `configs/stages/analyseMat.yaml` 的 `analyseMat` 下：

- `input_model_root`：训练后模型目录，当前是 `{home}/modelOutput/{model}`。
- `npy_output_root`：npy 输出目录，当前是 `{home}/analyseMat/{model}/npy`。
- `base_model_npy_dir`：底模 npy 目录，当前是 `{home}/originMat/{model}`。
- `prefix_with_stem`：npy 文件名是否带 safetensors 文件名前缀，默认 `false`。
- `overwrite`：是否覆盖已存在的 npy 文件，默认 `false`。
- `keep_layer_keywords`：只保留名称中包含指定关键词的层；`null` 表示全部保留。

#### 2.3.2 principalEvidence（Step 2，按配置执行）

当 `related_work.enable` 为 `true` 时，对指定 `target_layer` 执行单层主成分证据分析。产物写入 `{home}/analyseMat/{model}/principalEvidence`。

配置在 `configs/stages/analyseMat.yaml` 的 `analyseMat.related_work` 下：

- `enable`：是否执行 PE 分析，默认 `true`。
- `target_layer`：目标层号。
- `target_modules`：目标 module 列表；`null` 表示全部。
- `principal_rank_k`：主成分秩 K。
- `principal_top_ratio`：主成分保留比例。
- `update_mask_rule`：掩码更新规则，当前为 `bf16`。
- `save_png`：是否保存可视化 PNG。
- `plot_max_side`：PNG 最大边长。
- `output_root`：PE 产物目录。

### 2.4 eval

eval 阶段配置在 `configs/stages/eval.yaml`。当前 `eval.lm_tasks` 对应一次 lm-eval/vLLM 启动时真正执行的任务列表，`eval.target_metrics` 对应最终需要记录到汇总 CSV 的指标表。

`enable_eval` 和 `enable_result` 是两个独立开关：

- `enable_eval: true`：运行 lm-eval/vLLM，生成 combined JSON，并拆分到各 benchmark 目录。
- `enable_result: true`：读取已有 benchmark JSON，生成统一的 `target_metrics_results.csv`。

旧的 per-task clean/getResult CSV 流程已删除；当前不再使用 `eval.result.mmlu.target_keys`，也不再默认生成 `mmlu_results.csv`、`gsm8k_results.csv` 这类旧表。后续分析应以 `target_metrics_results.csv` 为准。

当前目标指标包括：

```text
GSM8K: gsm8k / exact_match,flexible-extract
MATH500: hendrycks_math500 / exact_match,none
AIME24: aime24 / exact_match,none
MMLU: mmlu / acc,none
MMLU-STEM: mmlu(groups.mmlu_stem) / acc,none
MMLU-Humanities: mmlu(groups.mmlu_humanities) / acc,none
MMLU-Social Sciences: mmlu(groups.mmlu_social_sciences) / acc,none
MMLU-Other: mmlu(groups.mmlu_other) / acc,none
TruthfulQA-MC1: truthfulqa_mc1 / acc,none
TruthfulQA-MC2: truthfulqa_mc2 / acc,none
WinoGrande: winogrande / acc,none
IFEval prompt strict: ifeval / prompt_level_strict_acc,none
IFEval instruction strict: ifeval / inst_level_strict_acc,none
HumanEval: humaneval / pass@1,create_test
```

正式评测时，每个真实训练 Source 的模型只启动一次 combined lm-eval，然后拆分到各 benchmark 目录。`target_metrics_results.csv` 会按 `eval.target_metrics` 读取对应 JSON 字段；其中 MMLU 的 STEM/Humanities/Social Sciences/Other 来自 lm-eval 输出里的 `groups`。

项目里仍保留了本地 NuminaMath 测试任务 `Eval/tasks/numina_math_local.yaml`，但当前默认 `eval.lm_tasks` 没有包含它，因此不会自动评测。若要加入本地 NuminaMath，需要把 `numina_math_local` 加到 `eval.lm_tasks`，并在 `eval.target_metrics` 中增加对应记录。

为了支持严格离线评测，项目在 `Eval/tasks/` 中覆盖了部分 lm-eval 内置 task：

```text
Eval/tasks/hendrycks_math500.yaml
Eval/tasks/aime24.yaml
Eval/tasks/ifeval.yaml
```

这些 task 的数据分别来自：

```text
Eval/tasks/data/hendrycks_math500/test.jsonl
Eval/tasks/data/aime24/train.jsonl
Eval/tasks/data/ifeval/train.jsonl
```

原因是 `HuggingFaceH4/MATH-500`、`Maxwell-Jia/AIME_2024`、`google/IFEval` 即使数据 Arrow 已经在 Hugging Face cache 中，严格离线时也可能无法通过 Hub dataset id 完成 metadata/module 解析。使用本地 JSONL task 后，正式 eval 在 `HF_HUB_OFFLINE=1` 下不再依赖这些 Hub id。

`Eval/tasks/aime24.yaml` 将 `max_gen_toks` 固定为 `2048`，以匹配当前 `eval.vllm_model_len: 4096` 的评测设置。不要恢复 lm-eval 内置的 `32768`，否则 vLLM 会因为生成长度超过模型上下文而把 prompt 截空。

`Eval/tasks/ifeval.yaml` 使用本地 `ifeval_utils.process_results` 包一层官方 IFEval 评分逻辑。原因是 JSON/datasets 会把嵌套 `kwargs` 中的整数读成 `3.0` 这类 float，而官方 IFEval checker 有些字段会作为列表下标使用；wrapper 会在评分前把整数型 float 转回 int。

如果机器需要提前准备 benchmark，可以先联网预下载缓存。首次准备或新增 benchmark 时建议使用在线模式：

```bash
python Prepare/component/prefetchEvalTasks.py --config configs/global.yaml --offline false
```

`prefetchEvalTasks.py` 会读取 layered `configs/global.yaml`，默认复用 `eval.lm_tasks`；如果没有配置 `eval.lm_tasks`，则从 `eval.target_metrics` 的 `json_task` 推导要预下载的任务。`eval.include_path` 也会同步传给 lm-eval 的 `TaskManager`。

缓存相关参数含义：

- `--offline false`：允许联网下载缺失 benchmark；也可以写成 `--no-offline`。
- `--offline true`：只使用本地缓存，缺失 benchmark 会报错；也可以写成 `--offline`。
- `--overwrite`：忽略 `{home}/logs/prepare/eval_cache_meta.json`，重新让 lm-eval/datasets 加载任务并检查缓存。它不会删除 `/root/.cache/huggingface/datasets` 或 `/root/.cache/huggingface/hub`，因此已经缓存完整的数据集通常不会重复下载。

预缓存后必须再做一次严格离线验证，因为“缓存目录存在”不等于 `datasets.load_dataset()` 在 `HF_HUB_OFFLINE=1` 下能够解析成功。验证命令：

```bash
export HF_ALLOW_CODE_EVAL=1
python Prepare/component/prefetchEvalTasks.py --config configs/global.yaml --offline --overwrite
```

如果看到：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
[DONE] 评测缓存预下载完成
```

说明当前 `eval.lm_tasks` 里的 benchmark 可以严格离线加载。当前已验证通过的默认任务是：

```text
gsm8k
hendrycks_math500
aime24
mmlu
truthfulqa_mc1
truthfulqa_mc2
winogrande
ifeval
humaneval
```

因此当前 `configs/stages/eval.yaml` 默认使用严格离线：

```yaml
hf_hub_offline: true
hf_datasets_offline: true
```

这样正式 eval 会完全使用本地缓存，不会再访问 Hugging Face Hub。若后续新增 benchmark、清空 `/root/.cache/huggingface`、升级 lm-eval 导致 dataset revision 变化，先临时改回：

```yaml
hf_hub_offline: false
hf_datasets_offline: false
```

然后重新运行在线预缓存和严格离线验证。

缓存位置由 `configs/stages/eval.yaml` 控制：

```yaml
hf_home: /root/.cache/huggingface
hf_datasets_cache: /root/.cache/huggingface/datasets
```

如果网络不能直接访问 Hugging Face，可以在运行前设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python Prepare/component/prefetchEvalTasks.py --config configs/global.yaml --offline false
```

HumanEval 需要额外注意：它的 `pass@1` 指标会使用 `code_eval`，该 metric 会执行模型生成的 Python 代码。当前默认配置包含 `humaneval`，因此需要两层显式确认：

```yaml
allow_code_eval: true             # 给 lm-eval 子进程设置 HF_ALLOW_CODE_EVAL=1
confirm_run_unsafe_code: true     # 给 lm-eval 命令追加 --confirm_run_unsafe_code
```

这两个字段在 `configs/stages/eval.yaml` 的 `eval` 下。只建议在隔离环境中开启这些开关。若不想执行代码评测，应同时从 `eval.lm_tasks` 和 `eval.target_metrics` 中移除 HumanEval。

当前默认不包含 Minerva-MATH500。原因是 `minerva_math500` 的 `math_verify` 指标需要 `lm-eval[math]` 和 `antlr4-python3-runtime==4.11.*`，而当前训练环境使用的 `llamafactory==0.9.3` 会引入 `OmegaConf==2.3.0`，它要求 `antlr4-python3-runtime==4.9.*`。为了保持单个 `density` 环境稳定，默认保留 MATH500/AIME/GSM8K/MMLU/TruthfulQA/WinoGrande/IFEval/HumanEval，移除 Minerva-MATH500。

当前推荐单环境依赖组合写在 `requirements.txt`：

```text
llamafactory==0.9.3
lm_eval==0.4.12
antlr4-python3-runtime==4.9.3
```

如果以后重新启用 Minerva-MATH500 或其它依赖 `math_verify` 的任务，需要重新评估 LLaMA Factory、OmegaConf、lm-eval math extra 和 antlr4 的版本组合。

建议安装/修复环境时使用：

```bash
conda activate density
pip install -r requirements.txt --force-reinstall
```

### 2.5 deleteMid

`deleteMid` 是每个 GPU batch 的最后阶段，配置在 `configs/stages/deleteMid.yaml`。它的作用域是当前 batch 注入的真实 `Source × DataSize` tasks，而不是整个 `baseDataset` 或整个实验目录。

默认删除：

- `{home}/modelOutput/{model}/{Source}/{DataSize}` 整个目录。这个目录只作为 eval/analyseMat 的临时可加载模型目录；deleteMid 执行时 eval/analyseMat 已经结束。
- `{home}/finetuneOutput/{model}/{Source}/{DataSize}` 中的 checkpoint、模型权重、optimizer/scheduler/rng 等训练态大文件。
- `{home}/analyseMat/{model}/npy`。

默认保留 finetune 里的轻量记录：

```text
train_meta.json
trainer_state.json
trainer_log.jsonl
train_results.json
eval_results.json
all_results.json
finetune_*.yaml
training_*.png
log_output/**
cleanup_meta.json
```

`cleanup_meta.json` 会记录本次清理删除了哪些路径、释放了多少字节、保留了哪些文件。删除 `modelOutput` 后，如果要重新 eval 或 analyseMat，必须重新训练或重新物化模型。

## 3. dataAnalyse：本地 CPU 汇总分析

`dataAnalyse` 是训练、`analyseMat`、`eval` 全部完成后的本地后处理阶段。它不启动训练，不启动 vLLM，也不重新生成 baseline/offPolicyData 数据；默认 ER 路径只读取已经存在的实验产物，构建统一主表、Source 数据谱特征、可选 PE 特征、宏观曲线和相关性结果。若开启 `dataAnalyse.gamma.enabled`，它会额外调用 `MyFunc/GetSlice` 重算 S/X whitening 产物，用于计算 Γ gap 指标。

推荐入口是：

```bash
python run_analysis.py --config configs/global.yaml
```

`run_analysis.py` 会读取 layered `configs/global.yaml`，取其中的 `home`、`models` 和 `dataAnalyse` 配置，然后调用 `dataAnalyse/main.py`。

### 3.1 输入数据

主表输入包括：

- `{home}/finetuneOutput/{model}/{Source}/{DataSize}/trainer_state.json`：读取训练过程中的 `eval_loss`，取最小值作为 `EvalLoss`，并记录对应 `BestStep`。
- `{home}/csv_results/{model}/target_metrics_results.csv`：新版 eval 汇总表。当前默认指标包括 `GSM8K`、`MATH500`、`AIME24`、`MMLU`、`MMLU-STEM`、`MMLU-Humanities`、`MMLU-Social Sciences`、`MMLU-Other`、`TruthfulQA-MC1`、`TruthfulQA-MC2`、`WinoGrande`、`IFEval prompt strict`、`IFEval instruction strict`、`HumanEval`。
- 旧版 `{home}/csv_results/*.csv`：仍兼容读取 `gsm8k_results.csv`、`mmlu_results.csv`、`winogrande_results.csv`、`truthfulqa_results.csv`、`ethics_results.csv`，用于历史 exp0509 分析。

Source 数据谱特征输入优先来自 manifest：

```text
{home}/sources/manifest.json
```

每个 manifest entry 的 `input_paths` 指向真实训练 Source 数据，例如：

```text
{home}/baseline/<Source>__<DataSize>/selected.parquet
{home}/baseline/<Source>__<DataSize>/cfs.jsonl
{home}/offPolicyData/runs/<run>/ranked_datasets/size<DataSize>/<band>/dataset.jsonl
```

如果 manifest 中没有可读输入，才回退到 prepare cache：

```text
{home}/prepared/sft_cache/<task_id>/train.jsonl
```

PE 特征是可选输入：

```text
{home}/analyseMat/{model}/principalEvidence/principal_evidence.csv
```

是否自动 join 由 `configs/stages/dataAnalyse.yaml` 中的 `auto_join_principal_evidence` 控制。

### 3.2 Source 数据谱 ER 特征

当前 `dataAnalyse` 默认：

```yaml
dataAnalyse:
  er_feature_source: source_max_datasize
```

含义是：

- `source_max_datasize`：每个 `Source` 只选择该 Source 下最大的 `DataSize`，读取这个最大 DataSize 的真实训练数据计算一次 ER，然后把结果复制到该 Source 的所有 DataSize。输出中 `FeatureKeyMode=source_only_replicated`，并用 `FeatureSourceDataSize` 记录实际用于计算的最大 DataSize。
- `source_datasize`：每个真实 `Source × DataSize` 都单独读取对应训练数据并计算 ER。输出中 `FeatureKeyMode=source_datasize`。

旧别名仅用于兼容：`dataset` 等价于 `source_datasize`，`datasize` 和 `auto` 等价于 `source_max_datasize`。当前 dataAnalyse 不再使用 GetSlice ER fallback；缺少 Source 数据输入时应直接报错，避免悄悄走旧分析路径。

Source 数据谱计算流程：

1. 按配置选择 `Source × DataSize` 或每个 Source 的最大 `DataSize`，再从 manifest 定位训练数据文件。
2. 读取 `.parquet`、`.jsonl`、`.json`、`.csv`。
3. 从 `problem/solution`、`question/answer` 或 ShareGPT `messages` 中抽取文本。
4. 构造文本谱矩阵并计算奇异值。若环境有 `sklearn`，使用 `HashingVectorizer + TruncatedSVD`；若没有 `sklearn`，使用内置 hash-cov fallback。
5. 根据奇异值计算：

```text
EffectiveRank = exp(-sum_i p_i * log(p_i))
SpectralSkewness = sum(top_k singular values) / sum(all singular values)
```

这里的 ER 是**Source 数据文本谱 ER**，不是旧版 GetSlice activation ER。CFS 与训练阶段保持一致，严格读取 manifest 中的全部 `input_paths`，也就是 `selected.parquet + cfs.jsonl`；其中 `selected.parquet` 只包含基础抽样部分，`cfs.jsonl` 是额外拼接的 CFS 样本，二者合起来才是 CFS 的真实训练集。

### 3.3 GetSlice Γ gap 指标

`dataAnalyse.gamma` 控制与 ER/KL 并列的 Γ gap 指标：

```yaml
dataAnalyse:
  gamma:
    enabled: true
    fixed_k: []
    energy_ratio: [10, 20, 40]
    getslice_config_path: /root/LLM-output-density/MyFunc/GetSlice/config.json
    getslice_save_path: "{home}/analyseOutput/GetSliceGamma"
    plot_singular_values:
      enabled: true
      output_dir: "{home}/analyseOutput/figures/gamma_singular_values"
      y_scale: log
      max_points: 0
```

指标定义为：

```text
gap_k(WL_D) = sigma_k(WL_D) - sigma_{k+1}(WL_D)
Gamma_{X<-S}^{(k)} = |L_S^{-1}L_X|_2 * gap_k(WL_S) / gap_k(WL_X)
```

这里的 `L_S`、`L_X` 来自 GetSlice profiling matrix，`WL_S`、`WL_X` 的奇异值来自 `sMat/xMat`。`dataAnalyse` 会从 manifest 的全部 `input_paths` 准备 S 侧 strict JSONL；因此 CFS 仍按真实训练输入读取 `selected.parquet + cfs.jsonl`，不是只读取 `cfs.jsonl`。

k 有两种来源：

- `fixed_k`：配置给出的 1-based k，S/X 使用同一个 k。
- `energy_ratio`：先在 `WL_X` 的奇异值中按 `sum(sigma^2)` 累积能量达到指定百分比选出 k，再让 `WL_S` 和 `WL_X` 共用这个 k。

`|L_S^{-1}L_X|_2` 的数值实现是 solve 优先：若 `L_S` 可解则使用 `torch.linalg.solve(L_S, L_X)`，否则退回 `torch.linalg.pinv(L_S) @ L_X`。若 `gap_k(WL_X) <= eps`，对应 Γ 输出 `NaN`，避免除零。

输出中：

- `tables/gamma_feature_modules.csv`：只包含 Γ 的 layer/module 明细。
- `tables/gamma_features.csv`：只包含 Γ 的 Source/DataSize 汇总。
- `tables/er_feature_modules.csv`：在原 dataset ER 明细后追加 Γ module 行。
- `tables/er_features.csv`：在原 ER 汇总列后追加 Γ 汇总列，用于相关性分析。
- `figures/gamma_singular_values/<layer>/<module>.png`：每张图对应同一个 layer/module，包含 X 和所有 Source 的 `WL_D` 奇异值分布曲线。

Γ 的 DataSize 语义跟 `er_feature_source` 保持一致：`source_datasize` 逐 `Source × DataSize` 计算；`source_max_datasize` 每个 Source 只用最大 DataSize 计算一次，再复制到该 Source 的所有 DataSize。

### 3.4 相关性 random baseline 归一化

相关性分析默认不直接使用原始数值列，而是先在每个 `Model × DataSize` 组内找到 Source 名包含 `random` 的行作为 baseline，再计算百分比波动：

```text
Metric_RandomDeltaPct = (Metric - Metric_random) / abs(Metric_random) * 100
```

配置为：

```yaml
dataAnalyse:
  correlation:
    exclude_sources: [cfs]
    module_correlations:
      enabled: true
      output_dir: modules
      feature_sources: [getslice_gamma]
    random_baseline:
      enabled: true
      source_contains: random
      suffix: _RandomDeltaPct
      scale: 100.0
      eps: 1.0e-12
```

`exclude_sources` 会在相关性计算前按 Source 名包含关系过滤指定来源，例如 `[cfs]` 会排除 `NuminaMath-cfs50`。`correlation_panel_raw.csv` 保留未过滤的原始 join 后数值；`correlation_panel_filtered_raw.csv` 保留过滤后的原始数值；`correlation_panel.csv` 和 Pearson/Spearman 相关性矩阵使用带 `_RandomDeltaPct` 后缀的归一化变量。若 random baseline 的某个变量接近 0，该变量的归一化值写为 `NaN`，避免除零。

`module_correlations` 会对 `gamma_feature_modules.csv` 中每个 `Layer × Module` 单独计算相关性；它复用同样的 `exclude_sources` 和 random baseline 归一化逻辑。输出位于 `correlation/modules/`，每个 module 会有独立的 Pearson/Spearman CSV 与热图，`correlation/modules/panels/` 中保留对应 module 的相关性输入面板。

### 3.5 WSI

`WSI` 从主表中的目标指标计算，公式为：

```text
WSI(row) = sum_i weight_i * max(0, reference_i - value_i) / reference_i
```

`reference_i` 默认取当前分析表中该指标的最大值；如果未来有 `Source == "human"` 且 `DataSize` 为最大数据量，则优先用 human 行作为参考。`WSI` 越低越好。

新版默认权重写在 `configs/stages/dataAnalyse.yaml`，覆盖 `MMLU`、MMLU 分组、`TruthfulQA-MC1/MC2`、`WinoGrande`、`IFEval` 和 `HumanEval`；同时保留旧列名 `Winograd`、`TruthfulQA_MC1/MC2` 和 `Ethics_*`，兼容历史 CSV。

### 3.6 输出结果

输出根目录默认是：

```text
{home}/analyseOutput/
```

主要产物：

```text
tables/source_datasize_metrics.csv
tables/er_features.csv
tables/er_feature_modules.csv
tables/gamma_features.csv
tables/gamma_feature_modules.csv
figures/gamma_singular_values/layer_*/<module>.png
figures/macro_metrics_over_datasize.png
figures/macro_<Metric>.png
correlation/correlation_panel.csv
correlation/correlation_panel_raw.csv
correlation/correlation_panel_filtered_raw.csv
correlation/modules/pearson_module_<layer>__<module>.csv
correlation/modules/spearman_module_<layer>__<module>.csv
correlation/modules/pearson_module_<layer>__<module>.png
correlation/modules/spearman_module_<layer>__<module>.png
correlation/pearson_global.csv
correlation/spearman_global.csv
correlation/pearson_global.png
correlation/spearman_global.png
```

`source_datasize_metrics.csv` 是训练 loss 和 benchmark 的主表。`er_features.csv` 是按 `Source, DataSize` 聚合后的数据谱特征表；若启用 Γ，也会包含 `LTransferNormMean`、`Gamma_*Mean`、`Gap*Mean` 等汇总列。`er_feature_modules.csv` 在 dataset 模式下记录每个 `Source × DataSize` 使用了哪些输入文件；若启用 Γ，也会追加 layer/module 级 Γ 明细。

## 4. TestExp 训练过程全层 GetSlice 实验

`TestExp/run_exp.py` 是一个独立的小实验入口，用来观察训练过程中不同 checkpoint 的全层 GetSlice 奇异值变化。它不接入主 pipeline，默认产物写到：

```text
/root/autodl-tmp/exp0524/
```

默认只跑 4 个 source，而不是全量 source：

```text
NuminaMath-random
NuminaMath-offPolicyData_kl_high
NuminaMath-offPolicyData_kl_low
NuminaMath-ppl_mid70
```

默认训练到 `step 510`，并分析 `170/340/510` 三个 checkpoint。这个设置贴近 exp0513 的历史训练 step，同时控制成本。exp0513 目前只可靠复用 prepared cache 和 trainer metadata；如果 `best_model_checkpoint` 指向的权重目录已经被清理，脚本会明确跳过，不会假装复用成功。

推荐先跑单 source 验证：

```bash
python TestExp/run_exp.py --source NuminaMath-random --stop-after-step 510 --checkpoint-steps 170 340 510
```

安全的重跑/检查模式：

```bash
python TestExp/run_exp.py --dry-run
python TestExp/run_exp.py --analyse-only --source NuminaMath-random --checkpoint-steps 170 340 510
python TestExp/run_exp.py --plot-only
python TestExp/run_exp.py --plot-only --comparison-only
```

主要输出：

```text
/root/autodl-tmp/exp0524/finetuneOutput/Qwen2.5/<Source>/10000/
/root/autodl-tmp/exp0524/getslice/base_model/
/root/autodl-tmp/exp0524/getslice/<Source>/step_<N>/
/root/autodl-tmp/exp0524/getslice/<Source>/exp0513_best/
/root/autodl-tmp/exp0524/figures/gamma_singular_values/
/root/autodl-tmp/exp0524/cache/
/root/autodl-tmp/exp0524/prepared/sft_cache/<task_id>/
```

其中 `exp0513_best` 只有在对应权重文件真实存在时才会生成；当前若 exp0513 checkpoint/modelOutput 已被清理，相关分析会自动跳过。`--plot-only` 可反复从已有 GetSlice JSON 重画图，不会重新训练或运行 GetSlice。
脚本会把当前 run 的 HuggingFace、datasets、torch 和 matplotlib 隐式缓存指向 `/root/autodl-tmp/exp0524/cache/`；训练前会把 exp0513 的轻量 prepared cache 复制到 `/root/autodl-tmp/exp0524/prepared/sft_cache/<task_id>/`，训练 YAML 的 `dataset_dir` 指向这份本地副本。GetSlice 的 profiling/token data `.pt` 默认生成在对应 `getslice/<checkpoint_label>/...` 目录下，正常结束后由 `cleanup_intermediate=true` 删除。
使用 `--clean-checkpoints` 时，脚本会在每个 source 训练完成后立刻分析该 source 的 checkpoint，并在对应 GetSlice JSON 写出后删除该 source 的 `checkpoint-*` 和训练根目录下的最终模型权重 shard，避免多个 source 的 18GB 级 checkpoint 同时堆积到磁盘爆满。

为了更直接比较训练过程中的奇异值变化，绘图阶段还会生成跨 `Source × step` 的汇总图：

```text
/root/autodl-tmp/exp0524/figures/gamma_singular_values/summary_layers/layer_<N>/<module>.png
```

默认汇总层是 `0, 7, 14, 21, 27`，module 仍然分开，每张图对应一个 `layer/module`。图中颜色区分 Source，线型区分 `base/step_170/step_340/step_510`。只想重画这类汇总图时使用：

```bash
python TestExp/run_exp.py --plot-only --comparison-only
```

也可以手动选择层和 step：

```bash
python TestExp/run_exp.py --plot-only --comparison-only --summary-layers 0 14 27 --summary-steps 170 510
```
