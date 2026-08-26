# 当前代码交接：Round3五方法实现

> 当前唯一活动阶段为Round3 `SERVER_EXECUTION`。服务器strong smoke暴露入口与CUDA确定性实现缺陷后已停止；用户于2026-08-26明确允许Codex commit/push `round3-code-candidate-v0.4`、部署其exact commit、从新attempt重跑完整strong smoke，并在全部门禁通过后直接挂载formal。

## Round3交接状态

- 代码版本：服务器已测v0.3 exact commit为`ed1bfca002799f11ea1bad29f6f06e2e15fdd565`；v0.4 diff已获commit/push授权，本次提交产生的HEAD即待部署exact commit
- 获批实验：`round3-exp-v1.4`，用户于2026-08-26明确批准方案B及本地修改
- 方案B：冻结`test_prefs`选择前隔离3条empty-rejected；保持1K validation，剩余997条全部作为独立test，不从train补样
- 当前实现：隔离的`src/round3/`、`configs/round3/`、`scripts/round3/`；完整映射见`../../code/CODE_OVERVIEW.md`
- 覆盖范围：双源确定性数据与malformed audit、Qwen3-1.7B非量化LoRA、DPO-1K、GitHub-loss SSPO、DPO-8K、两个双vLLM动态PE、完整训练态checkpoint、共同1K selection与独立997-pair双head final test
- 明确排除：PE-static、AlpacaEval/MT-Bench、QLoRA、自动pruner及Round1/Round2观测入口
- 当前验证：v0.3已通过两个环境、6项合同测试、data v2与reference cache；两个strong-smoke attempt分别在入口模块路径与SSPO数值重放门禁停止。deterministic诊断使loss、LoRA最大绝对差和最大相对差全部为0
- 当前修复：v0.4把`PYTHONPATH`置于strong-smoke所有Python调用之前，并在trainer/verifier进入CUDA前强制`CUBLAS_WORKSPACE_CONFIG=:4096:8`、PyTorch deterministic algorithms、TF32关闭；科学合同与容差未改变
- 运行证据：见`../../exp/round3-20260826-01/README.md`和`../../exp/round3-20260826-02/README.md`；五方法strong smoke、双vLLM与storage projection仍无完整通过证据
- Round2保护：只读证据确认其正式任务在step590停止且step580/589/590保留，两个pruner未运行；本交接不授权删除或覆盖其checkpoint
- 测试分工：Codex独占`SOPPO/`设计与内容修订权限；GLM只按`../../code/scripts/round3/GLM_VALIDATION_GUIDE.md`机械部署exact reviewed commit、执行命令并回传证据，禁止现场编辑源码或自行处理失败

本候选还补齐了实现者不能留给运行者猜测的细节：显式public Git ref到模型/数据full SHA的服务器解析证据、sample ID源行反向审计、reference-cache输入/模型/tokenization绑定、SSPO下一batch round-trip数值容差、final selected checkpoint完整性复核、进程PID/PGID/starttime绑定，以及把source cache/strong-smoke留存纳入空间投影。这些是fail-closed复现与执行安全门禁，不改变五方法科学合同。

## Round2历史交接（只读）

## 当前状态

- Cycle：`cycle-20260818-01`
- 第一轮 Experiment：`exp-20260819-01-mvp`（冻结基线，只读引用）
- 第二轮默认 Experiment：`exp-20260823-01-round2-tp2`；正式运行必须使用新的唯一 ID
- 实验设计：v0.6完整基线固定于Git commit `d338eb5bedef16d83a42790c3faa97f8f404315b`并由`experiment_archive.md`索引；当前`current_experiment.md`已专用于Round3，不反向改写本Round2交接
- 历史阶段：曾进入`SERVER_EXECUTION`；2026-08-26只读核验确认第一方法在step590停止、第二方法未启动，不是全局活动阶段。
- 历史代码交接：用户于2026-08-24明确确认 commit `c2c9069a0b1a1187c8e709729b33b15aaec8c454`；其后的多个失败attempt用于修复环境、DTensor/TP hook、vLLM生命周期和rollout长度门禁。
- 历史执行状态：后续版本曾通过server tests、两条方法的真实TP2 strong smoke并启动正式Round2；正式experiment已核验为`exp-20260824-05-round2-tp2`，controller在step590以exit 143停止，第二方法未启动，仍无final结果。
- 当前本地源码基线：方案B修改前共同clean基线为`33f0eea632ba93ede616650484f72c57f35742c6`；方案B实现后来出现在本地`8eedca51067162562cade72408930c8b2321ffda`，而当前GLM指南修订仍未提交。服务器checkout仍须按实时证据处理。
- 当前平台适配：3×4090固定GPU0–1 Transformers native TP-LoRA、GPU2 vLLM；两条正式方法串行而非并行。
- 当前运行边界：不得热改潜在运行中的Round2共享checkout、停止控制器或改写正式配置。双vLLM副本和Qwen3-1.7B现已仅获Round3本地实现授权，仍不能热应用到Round2，也尚未获Round3服务器执行授权。
- 现场运行、耗时、显存、磁盘、checkpoint清理和待决策项：`../code/ROUND2_LIVE_HANDOFF.md`。
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
- checkpoint：第一轮冻结合同为DPO每20 step、SSPO/PE每40 step加final；Round2在线rollout每step必须先发布一个LoRA adapter，且无optimizer state。2026-08-26核验时第一方法保留20个checkpoint（含step580/589/590，best为step480），两个外置pruner进程均不存在；不得继续清理。
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
| 第二轮 TP 训练 | `src/round2/tp_trainer.py`, `tp_backend.py`, `run_tp.py`；TP=2/PP=1/DP=1，以checkpoint完整shape核对每rank普通本地Tensor shard，不依赖DTensor类型 |
| 在线候选与不可变 adapter handoff | `src/round2/run_rollout.py`, `queue_protocol.py`；每步 READY/SHA adapter、56-prompt request/response |
| 固定单回复锚点生成与隔离 | `src/round2/prepare_sft_anchor.py`, `sft_schema.py`, `scripts/round2/00_prepare_sft_anchor.sh`；从公开 response_a 确定性派生，24k ID/prompt/response 精确连接，禁止 label/pair 字段 |
| 第二轮 GPU 等待 / strong smoke / 长链 | `scripts/round2/02_wait_for_idle_gpus.sh`, `02_strong_smoke.sh`, `start_all.sh`, `run_all.sh`, `status_all.sh`, `stop_all.sh` |
| 第二轮独立评价与无样本导出 | `src/round2/evaluate.py`, `aggregate.py`, `scripts/round2/03_evaluate.sh`, `04_aggregate.sh` |
| adapter evaluator/GetSlice | `src/evaluation/evaluator.py`, `observe/.../model_utils.py` |
| 一次提交 DAG | `scripts/cluster/submit_all.sh` |
| 独占服务器顺序长链 | `scripts/standalone/start_pipeline.sh`, `run_pipeline.sh` |
| 独占服务器状态/精确停止 | `scripts/standalone/status_pipeline.sh`, `stop_pipeline.sh` |

## 第二轮实现边界与交接条件

本地只允许纯文本编辑、`bash -n`、`git diff --check` 和静态路径/旧接口搜索；不运行 Python import、pytest、数据、模型、训练、评价或 GPU 工作。第二轮只新增代码、配置和命令，不修改第一轮 MVP 代码路径的既有语义。服务器 tests 与 strong smoke 必须在用户确认本次未提交 diff 并形成 clean commit 后完成。

2026-08-23 用户已补齐开放决定：固定单回复锚点来自第一轮24k公开 unlabeled 的已随机换位 `response_a`；采样冻结为 `temperature=0.7/top_p=0.8/top_k=20/min_p=0`。这些实现随 `c2c9069` 于2026-08-24获用户明确确认，且服务器两个环境已安装验证。

随后用户要求在三张卡暂时被占用时先挂起本实验。实现把只读等待器插入 `server_tests → strong_smoke` 之间：默认每30秒查询 resolved config 指定的三张4090，要求90秒稳定无 compute PID、显存使用不超过1024MiB、利用率不超过5%，并持续核对 clean Git commit；它只写原子 `gpu_wait.json`，绝不发送信号。无调度器条件下无法原子锁卡，因此放行后仍由原有 preflight 再次失败关闭。该门禁已进入获服务器启动授权的 `f4601c8`。

首次启动在pytest collection因缺少 `datasets` 失败；第二次启动暴露旧DTensor门禁和EngineCore清理遗漏；第三次启动暴露PEFT/Transformers TP-hook接口错位。用户提交兼容修复 `d03a116` 后，第四次启动的SFT+rollout已完成单步真实TP2/PE并记录每rank约8.65GB allocated峰值。rollout-only生成两条候选/提示时，`min_tokens=512`没有保证全部112条都恰好达到512，因而在训练前被最坏长度门禁拒绝。这些均为历史attempt，不是当前正式任务状态。

后续修复版本已完成两条方法的真实TP2 strong smoke并进入正式长链。正式合同每方法856个optimizer step；2026-08-24性能快照在第一方法step35/856，最近20步平均286.1秒，其中rollout生成165.4秒。2026-08-26只读核验则确认该方法已在step590停止、第二方法未启动；两者都是带时间证据，任何新动作前仍须按`../code/ROUND2_LIVE_HANDOFF.md`重新核验。

代码总览已经记录第二轮两条新增rollout实验的实现、默认值、产物、与第一轮冻结基线的只读合并方式和已知限制。此段历史授权只覆盖当时启动的4B Round2运行；QLoRA或新增Round2 `C_epsilon`仍未授权。双rollout副本和1.7B模型后来只在独立Round3合同下解锁本地实现，不能热应用到Round2，且Round3服务器执行仍未授权。
