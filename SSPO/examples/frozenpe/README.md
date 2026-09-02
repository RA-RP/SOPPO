# FrozenPE

`FrozenPE`是 Round4 v2 中原 `StaticPE` 原型的正式名称：labeled 数据使用
DPO，UltraChat 无标签行使用原始回答 A 与冻结初始化 Qwen3 贪心生成的回答 B。
候选 B 只在授权的 A100 环境生成一次；训练参数并不被冻结。

目标为：

```text
L = (L_DPO + 0.1 * L_PE) / 1.1
```

其中 `p_i = sigmoid(0.1 * ((s_A-s_ref,A) - (s_B-s_ref,B)))`，随后由所有
global physical micro-batch 中的 `p_i` 形成 `c_1/c_2` 和 L1 PE 项。

训练日志：`frozenpe/loss_dpo`、`frozenpe/loss_pe`、二者的加权贡献及
`frozenpe/loss_total`。这些分项只用于同一方法的机制诊断。

## 服务器顺序

```bash
python preprocessing_data/preprocessing_ultrachat.py --fb 0.1 --ch 0.1
python preprocessing_data/generate_frozenpe_candidates.py --fail_on_drop
llamafactory-cli train examples/train/qwen3-1.7b-it/frozenpe/fb0.1_ch0.1/<exact-yaml>.yaml
```

通用 AlpacaEval 输出生成与 judge 流程在 `examples/evaluation/`；A100 只生成
request bundle，4090 使用私有 judge profile 进行 API judge 与 WR/LC 聚合。
