# Round4 v2.0.5 代码与服务器执行总览

## 状态

- Cycle：`cycle-20260901-01` / Round4；当前阶段：`SERVER_EXECUTION`。
- 依据：`r4-theory-v2.0`、`round4-exp-v2.0`，均于2026-09-02获用户明确通过。
- 已验证 smoke 代码：`cb2d54c`。A100-2四臂2-step full-chain smoke通过：训练、eval、adapter、merge/reload、2条生成与4090真实API judge均完成；StaticPE与FrozenPE的PE分支均有非零训练诊断。2条样本的WR/LC仅验证接口，不能用于比较方法。
- 正式执行代码：`71b0a6cf347a86bed9ac1a0ae6b68f0ee3dc2500`（本总览对应提交）。其训练语义不变，新增`code/scripts/round4/05_prepare_formal.py`和`05_run_formal_a100.sh`，将每个formal运行绑定到数据、模型、exact commit与不可变输出路径；用户已明确授权连续执行。
- 历史：用户于2026-09-02批准并执行`6afebd3`、`92259df`、`98dc1aa`与`67ebed0`。前三者分别在包导入、torchrun入口和Trainer batch接口停止；`67ebed0`已进入模型前向，却因Qwen3在`transformers==4.51.3`中将 decoder layer 调用包装为`functools.partial`，旧梯度检查点包装器假设`func.__self__`存在而停止。均未完成优化 step。`6b010b8`修复无效行过滤；旧失败产物保留，不被覆盖。

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
| 4090 judge配置 | `code/scripts/round4/04_run_api_judge_4090.sh`、`examples/evaluation/judge_{profiles,credentials}.example.json` | profile保存模型、`max_tokens`/`temperature`/logprob参数与变量名；权限600的仓库外credentials JSON保存多组key/base URL；环境变量可覆盖 |
| formal准备与运行 | `code/scripts/round4/05_prepare_formal.py`、`05_run_formal_a100.sh` | 服务器生成计划、按base→四方法顺序运行；训练输出后生成805条并写入judge request，不在A100保存API key |

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
  judge profile调用API并写回 aggregate-only 的WR/LC结果。profile存于
  `$HOME/.config/soppo/judge_profiles.json`，同目录权限600的
  `judge_credentials.json`为多个profile提供key/base URL；环境变量优先覆盖。API key
  不进入仓库、日志或A100。

## 静态复核与服务器待验

本地仅执行标准库AST/YAML/JSON/grep/bash语法等静态检查，未导入项目、未读取数据或
模型，也未运行训练、数值、GPU、生成或judge任务。服务器必须验证：参数解析、StaticPE
数值/梯度与DDP一致性、EMA resume、两流sampler、2-step四臂full-chain smoke、A100→4090
request/result绑定和正式805条流水线。

服务器验证已完成`cb2d54c`四臂2-step full-chain smoke。`71b0a6c`的新增formal启动器只完成本地静态shell检查；实际计划生成、正式训练、805条生成和4090评价均必须在服务器留下独立证据。正式运行中的失败保持产物并记录，不以服务器现场改参绕过冻结合同。
