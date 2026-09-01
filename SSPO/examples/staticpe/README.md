# StaticPE 独立原型

本目录及其对应源码属于 `SSPO/` 下的独立实验通路，不登记到 Round3。
数据、模型、训练和评价命令只能在已获授权的服务器执行环境运行。
服务器环境应在 `SSPO/` 根目录用 `pip install -r requirements.txt` 创建；
其中 `-e .` 会把当前 `src_sspo` 注册为 `llamafactory-cli`，不会再拉取并误用旧上游源码。

## 固定定义

- 初始化模型：`Qwen/Qwen3-1.7B`，revision
  `b9352fbb8ce704292730cf54b3b1dceb2a808738`。
- 环境：`transformers==4.51.3`、`tokenizers==0.21.1`、
  `huggingface-hub==0.30.2`；Qwen3 不支持仓库原来的 Transformers 4.46.1。
- 数据：UltraFeedback `0.1` + UltraChat `0.1`。
- 无标签候选：UltraChat 原始回答与初始化 Qwen3 的一次性贪心生成；两者按稳定哈希随机交换 A/B 位置。
- 训练：非思考 `qwen3` 模板、DPO `beta=0.1`、`lambda=0.1`、1 epoch。
- 每设备物理 micro-batch 为 `4`；一行数据对应两个回答序列，因此每卡每次前向实际拼接 8 条回答序列。
- 使用 2 张 GPU、`gradient_accumulation_steps=8` 时，常规意义下的 optimizer batch 为 `4 * 2 * 8 = 64` 行。评估不做梯度累积，因此全局 eval batch 为 `4 * 2 = 8` 行。
- 双流采样器在整轮范围内保持原数据的标注/无标签总比例（仅有整数取整误差），每个 micro-batch 至少有 2 条无标签数据。原始标注比例低于 `1/4`，因此少量 batch 允许没有标注数据，否则无法同时保持原比例且不重采样。
- PE 的 `c_1/c_2` 是非线性的 batch 统计：当前实现先在两卡间可微汇总，所以每次从 `4 * 2 = 8` 行物理全局 micro-batch 中取其中的无标签子集计算 PE（按当前比例通常约 6 条）；随后对 8 个 micro-batch 的梯度进行累积。它不等价于先把 64 行中的全部无标签数据一次性合并后再计算一组 `c_1/c_2`。

对无标签对，令

```text
p_i = sigmoid(beta * ((log pi(A)-log pi_ref(A)) - (log pi(B)-log pi_ref(B))))
q_i = [p_i, 1-p_i]
c_1 = sum_i p_i q_i / (sum_i p_i + eps)
c_2 = sum_i (1-p_i) q_i / (sum_i (1-p_i) + eps)
L_PE = 0.5 * (||c_1-[1,0]||_1 + ||c_2-[0,1]||_1)
L = (L_DPO + lambda * L_PE) / (1 + lambda)
```

`c_1/c_2` 在所有已初始化的分布式进程之间进行可微汇总。若某个物理 micro-batch 暂时缺少标注行，该批次的 DPO 项为 0，但固定使用 `(L_DPO + lambda * L_PE) / (1 + lambda)`，不会临时放大 PE 权重。

训练日志分别记录：

- `staticpe/loss_dpo`、`staticpe/loss_pe`：组合前的两项原始损失；
- `staticpe/loss_dpo_weighted`、`staticpe/loss_pe_weighted`：除以 `1 + lambda` 后各自对总目标的实际贡献；
- `staticpe/loss_total`：两项相加后的当前 StaticPE 总损失。

评估阶段对应指标带 `eval_` 前缀。

训练正常结束后，完整的按 step 历史保存在输出目录的
`trainer_state.json` 中，具体位于 `log_history` 数组；当前配置每 20 个
optimizer step 记录一次。`plot_loss: true` 还会为上述 DPO、PE、加权分项
和总损失分别生成 `training_staticpe_loss_*.png` 曲线。`train_results.json`
主要是训练结束汇总，不代替 `trainer_state.json` 的完整过程曲线。

## 服务器执行顺序

先运行原始 UltraFeedback/UltraChat 预处理，再冻结候选 B：

```bash
python preprocessing_data/preprocessing_ultrachat.py --fb 0.1 --ch 0.1
python preprocessing_data/generate_staticpe_candidates.py
```

候选脚本会生成：

- `data/ultra_combined_fb0.1_ch0.1_staticpe.json`
- 同名 `.manifest.json`
- `data/dataset_info.json` 中的 `ultra_combined_fb0.1_ch0.1_staticpe` 条目

然后训练：

```bash
export PYTHONPATH="$(pwd)/src_sspo${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="0,1"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
llamafactory-cli train examples/train/qwen3-1.7b-it/staticpe/fb0.1_ch0.1/lora_Qwen3-1.7B_staticpe_lr1e-05_beta0.1_lambda0.1_cutoff1024_ep1_tb4_eb4_ga8_rank8.yaml
```

生成 AlpacaEval 2.0 的 805 条模型回答：

```bash
python examples/staticpe/generate_alpacaeval_outputs.py \
  --adapter_name_or_path /path/to/staticpe/adapter \
  --output_file /path/to/exports/staticpe_alpacaeval2.json
```

最后调用官方 evaluator。API key 只通过服务器环境变量传入，不写进脚本或日志：

```bash
export OPENAI_API_KEY=...
bash examples/staticpe/run_alpacaeval2.sh \
  /path/to/exports/staticpe_alpacaeval2.json \
  /path/to/runs/staticpe_alpacaeval2
```

主报告读取 evaluator 产生的 `leaderboard.csv` 中的
`length_controlled_winrate`（LC）和普通 `win_rate`，原始逐样本输出与 annotations 留在服务器。
