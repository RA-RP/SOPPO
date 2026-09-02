# Round4 v2 代码交接总览

## 状态

- Cycle：`cycle-20260901-01` / Round4；当前阶段：`CODE_IMPLEMENTATION`。
- 依据：`r4-theory-v2.0`、`round4-exp-v2.0`，均于2026-09-02获用户明确通过。
- 当前候选：`round4-code-v2.0.0`（本次交接的仓库`HEAD`）；服务器上传、环境创建、smoke和formal均锁定，等待用户明确确认该exact commit可以提交服务器。
- 历史：`af6dac4`只完成环境/资产准备；`6b010b8`修复无效行过滤。旧预处理失败证据和旧 prepared 目录保留，不被覆盖。

## 实现映射

| 规范 | 入口/实现 | 产物或关键约束 |
| --- | --- | --- |
| DPO label-only | `SSPO/examples/train/.../dpo/` | UltraFeedback、reference、effective batch16 |
| SSPO | `trainer.py::sspo_loss` | 既有DPO-base、threshold/prior、shared单回答UltraChat |
| StaticPE | `trainer.py::staticpe_loss` | SimPO `beta=10,gamma=2`、单回答、detached EMA、无reference、`staticpe/*` telemetry |
| FrozenPE | `trainer.py::frozenpe_loss` | 原StaticPE双候选DPO-base、reference、`frozenpe/*` telemetry |
| 参数合同 | `hparams/finetuning_args.py` | `pe_contract`阻止二者数据/目标混用 |
| batch构造 | `data/sampler.py`、`data/processors/pairwise.py` | StaticPE/SSPO使用`unlabeled`；FrozenPE使用`unlabeled_pair` |
| 固定候选B | `preprocessing_data/generate_frozenpe_candidates.py` | 仅FrozenPE需要；空/重复候选 fail closed |
| formal配置 | `examples/train/make_yaml.py`及四目录YAML | DPO/SSPO/StaticPE/FrozenPE，epoch1、LoRA r8、BF16 |
| A100生成 | `examples/evaluation/generate_alpacaeval_outputs.py` | non-thinking Qwen3、冻结instruction顺序、manifest SHA |
| smoke | `code/scripts/round4/03_prepare_smoke.py`、`03_run_smoke_a100.sh`、`03_validate_smoke.py` | A100训练/merge/reload/生成；不读取API key、不调用judge |

## StaticPE目标与状态

`staticpe_loss`先对labeled行计算长度归一化 SimPO，再以 chosen/rejected/unlabeled
三个分数的全局、detach 后EMA统计归一化无标签分数。只有PE分子/分母和
`p=sigmoid(z)`保留梯度；EMA不会进入计算图。目标固定为
`(L_SimPO + 0.1 L_PE) / 1.1`。每个global physical micro-batch至少需要两个
unlabeled响应；共同labeled eval不含unlabeled时只记录SimPO诊断且不更新EMA。
EMA state随checkpoint写入`staticpe_ema_state.json`并在resume恢复。

## 日志与评价合同

- StaticPE：`staticpe/loss_simpo`、`loss_pe`、两项weighted、`loss_total`、p分布、
  `c1/c2`、EMA均值/标准差、clip rate与样本计数。
- FrozenPE：`frozenpe/loss_dpo`、`loss_pe`、两项weighted、`loss_total`、p、
  `c1/c2`和候选reward摘要。
- SSPO保留`sspo/loss_labeled`、`loss_unlabeled`、`loss_total`、gamma等旧指标。
- A100写出不可变生成输出和GPU phase摘要；4090读取request bundle，以一个冻结的
  judge profile调用API并写回 aggregate-only 的WR/LC结果。API key和完整endpoint
  不进入仓库、日志或A100。

## 静态复核与服务器待验

本地仅执行标准库AST/YAML/JSON/grep/bash语法等静态检查，未导入项目、未读取数据或
模型，也未运行训练、数值、GPU、生成或judge任务。服务器必须验证：参数解析、StaticPE
数值/梯度与DDP一致性、EMA resume、两流sampler、2-step四臂full-chain smoke、A100→4090
request/result绑定和正式805条流水线。

在用户确认此exact commit前，任何服务器动作均不允许开始。
