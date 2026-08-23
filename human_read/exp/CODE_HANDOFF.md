# 第二轮 rollout PE：实验设计到代码交接合同

## 当前状态

- Cycle：`cycle-20260818-01`
- 第一轮 Experiment：`exp-20260819-01-mvp`（冻结基线，只读引用）
- 第二轮 Experiment：待定，必须使用独立 experiment_id，不得写入第一轮结果目录
- 实验设计：`current_experiment.md` v0.6 已同步为两轮边界；第二轮只新增 rollout 相关 PE 实验
- 当前阶段：`SERVER_EXECUTION`
- 代码交接：训练实现基线 `e047ce7` 已确认；standalone平台适配基线 `e4eb95d`；1/2/4卡等价执行档位基线`cf6bb99`已完成本地静态复核、尚待服务器验证
- 服务器执行：`AUTHORIZED`（2026-08-21）
- 当前平台适配：2026-08-22 新增 `scripts/standalone/` 无 Slurm 独占服务器入口；尚待该服务器验证
- 正式代码说明：`../../code/CODE_OVERVIEW.md`

## 第二轮新增实验清单

第二轮只新增以下两条实验，不重跑第一轮任何轨迹：

1. `SOPPO-PE-sft-rollout-exp`：SFT 参考回复 + 当前策略 rollout 回复构造动态候选对，并在该候选集合上计算 PE。
2. `SOPPO-PE-rollout-only-exp`：去掉 SFT 参考回复，只使用 rollout 生成候选回复，检验 SFT 锚点是否必要。

第一轮的 DPO-10、DPO-100、静态 PE、Pseudo-target/SSPO 类对照全部作为冻结基线只读引用。第二轮必须使用独立 experiment_id、独立输出根目录和独立命令入口，便于后续从两台机器直接合并聚合结果。

第一轮 MVP 实现和静态复核已经完成，并作为冻结基线保留。第二轮代码只允许复用公共模块并新增 rollout 相关入口、配置和脚本；不得修改、重跑或覆盖第一轮 MVP 的 DPO-10、DPO-100、静态 PE、Pseudo-target/SSPO 类对照及其结果。

## 第一轮冻结合同与第二轮新增范围

- 数据：继续使用 30k 的 2,700/300/24,000/3,000 隔离 split；DPO-100 oracle 为 26,700。
- 模型：Qwen3-4B、bf16/SDPA、thinking off、response-only、max length 2048。
- LoRA：r8、alpha16、dropout0、all q/k/v/o/gate/up/down projections；不使用 QLoRA或全参训练。
- 第一轮 DPO：DPO-10 与 DPO-100 均已作为冻结基线，只读引用，不在第二轮重跑。
- 第一轮 SSPO/Pseudo 类对照：作为冻结基线，只读引用，不在第二轮重跑。
- PE：margin-free pair probability、exact global population、epsilon1e-8；responsibility 和 denominator 默认不断梯度。
- 第二轮动态：只新增 `SOPPO-PE-sft-rollout-exp` 与 `SOPPO-PE-rollout-only-exp`；两者使用完全相同的 paper `gamma_t`，gamma0=1、gamma_min=2700/26700、decay=.01，只替换 unlabeled candidate construction。
- 第一轮 static PE：四条 PE lambda `{.1,.3,.5,1.0}` 已作为冻结基线，只读引用，不在第二轮重跑。
- batch：logical global64；1/2/4卡分别用梯度累积16/8/4，DPO每卡microbatch固定4；joint每step始终全局8 labeled pairs +56 unlabeled pairs。为避免2048长度OOM，梯度执行统一限制为每rank最多2 pair的backward subbatch；DPO按`2+2`，joint logical `3/4`按`2+1`/`2+2`回传，损失归一化、optimizer step与PE exact-global population不变。三卡被fail-closed拒绝。
- 优化：SSPO/PE 2 epochs、lr1e-5；AdamW、wd0、cosine、warmup.1、clip1、seed42。
- checkpoint：全部保留 LoRA adapter；DPO 20 step、SSPO/PE 40 step加 final；无 optimizer state。
- 第二轮新增轨迹只有两条：`SOPPO-PE-sft-rollout-exp` 与 `SOPPO-PE-rollout-only-exp`。两 DPO、SSPO/Pseudo、静态 PE 等第一轮轨迹不得重复训练、不得覆盖。
- headroom 与 oracle：DPO-10、DPO-100 与 static lambda 选择均来自第一轮冻结结果；第二轮只读引用，不重新选择。
- strong smoke：使用与正式训练相同的1/2/4卡档位、bf16/2048及各split最长真实样本；旧集群要求所选数量的A800，standalone必须记录实际SKU；覆盖五种配置、backward subbatch、KDE/PE、finite checks、adapter round-trip。
- Slurm 路由：账户只获批`gpu` partition且拒绝`sbatch --hold`/typed `--gres`；统一使用`-G N`，辅助阶段申请1卡，smoke/正式训练由`--formal-gpus 1|2|4`选择，直接提交`afterok` DAG并在中途提交失败时回滚。
- Slurm worker 路径：由提交器显式 export `SOPPO_CLUSTER_SCRIPT_DIR`，避免 batch 副本从 `/var/spool/slurmd` 错误解析 `job_env.sh`；状态工具使用真正的 array task ID 字段。
- 版本/节点保护：新DAG提交时从指定Git commit导出不含`.git`的独立源码快照，生成全文件SHA-256 manifest；每个worker启动时复核快照commit、manifest及内容，因而不依赖后续可变checkout。该机制加入前的旧DAG仍按原逻辑核对共享checkout。默认排除已发生NCCL卡死的`gn005`与既有排除节点`gn021`，可由`SOPPO_EXCLUDE_NODES`显式覆盖。
- standalone 平台：不改变上述训练合同；后台controller串行复用相同stage worker，由`SOPPO_TRAIN_GPU_IDS`数量选择1/2/4卡训练、单卡后处理，以原子registry记录档位和状态并在任一失败时阻断下游。实际GPU不静默假定为A800，但每卡默认至少79000 MiB，卡名与torch CUDA写入硬件证据。
- 数据入口：提交前重验30k行数/SHA、跨 split ID、公开隐藏标签和私有标签精确连接，摘要进入白名单。
- 下游：第二轮单独导出聚合结果；最终合并阶段只读取第一轮与第二轮各自的白名单聚合结果，不回写任一原始产物。

## 实现映射

| 合同 | 主要实现 |
| --- | --- |
| 配置/超参/batch fail-closed | `src/config.py`, `configs/mvp/*.yaml` |
| LoRA 加载、保存、重载 | `src/model/model_utils.py` |
| DPO 与 mean response score | `src/model/dpo_loss.py` |
| 第一轮 SSPO KDE/EMA/hard risk/scheduler | `src/model/sspo_loss.py`（冻结，只读引用，不改写） |
| exact-global PE | `src/model/pe_loss.py`（可复用公共实现；第二轮新增 candidate construction 与入口） |
| DDP joint trainer | `src/training/trainer.py` |
| Qwen3 response-only token边界 | `src/data/dataset.py` |
| headroom/static selection | `src/training/selectors.py` |
| strong smoke | `scripts/cluster/03_smoke.sh` |
| 第一轮 final runs | `03_preexperiment.sh`, `04_lambda_search.sh`, `05_run_main.sh`（冻结，不改写） |
| 第二轮 rollout runs | 新增独立配置、命令和入口；不得复用会覆盖第一轮输出的 experiment_id 或目录 |
| adapter evaluator/GetSlice | `src/evaluation/evaluator.py`, `observe/.../model_utils.py` |
| 一次提交 DAG | `scripts/cluster/submit_all.sh` |
| 独占服务器顺序长链 | `scripts/standalone/start_pipeline.sh`, `run_pipeline.sh` |
| 独占服务器状态/精确停止 | `scripts/standalone/status_pipeline.sh`, `stop_pipeline.sh` |

## 第二轮实现边界与交接条件

本地只允许纯文本编辑、`bash -n`、`git diff --check` 和静态路径/旧接口搜索；不运行 Python import、pytest、数据、模型、训练、评价或 GPU 工作。第二轮只新增代码、配置和命令，不修改第一轮 MVP 代码路径的既有语义。服务器 tests 与 strong smoke 必须在后续明确授权后完成。

代码总览需明确第二轮两条新增 rollout 实验的实现、默认值、产物、与第一轮冻结基线的只读合并方式、已知限制、静态复核和服务器待验证项；完成这些仍只意味着可以请求服务器执行授权，不等于已经获得授权。
