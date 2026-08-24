# SOPPO 第一轮 MVP 代码总览与第二轮 rollout 边界

## 0. 状态

- Cycle：`cycle-20260818-01`
- 第一轮 Experiment：`exp-20260819-01-mvp`（冻结基线，只读引用）
- 第二轮默认 Experiment：`exp-20260823-01-round2-tp2`；正式运行时必须换成新的唯一 ID
- 设计依据：`../human_read/exp/current_experiment.md` v0.6；当前文档将第一轮 MVP 代码说明与第二轮 rollout 新增边界分开记录
- 当前阶段：`CODE_IMPLEMENTATION`（第二次服务器启动在首条strong smoke暴露TP证据门禁与vLLM清理问题，返回代码阶段修复）
- 已确认代码：第二轮 TP=2 + 单卡 vLLM commit `c2c9069a0b1a1187c8e709729b33b15aaec8c454` 已于2026-08-24获用户明确确认；服务器 clean checkout 与两个环境已核验
- 已执行代码：依赖修复进入 `f54f6f4d744d138c80f2309ec5e350f1d5a428b3` 后，`exp-20260824-02-round2-tp2` 已通过锚点/config/server tests/GPU等待和vLLM ready；首条strong smoke在训练加载后、optimizer step前被旧DTensor门禁误拒绝，失败清理还留下vLLM EngineCore
- 当前代码版本：工作区HEAD/origin出现尚未经用户审阅的 `2af290df6447bf541abf225f666bff1e34beddd4`，包含初版safetensors local-shape门禁、rollout独立进程组和strong-smoke子状态；其上仍有未提交补充修复：逐planned-module验证TP hooks、LoRA sharded-SUM/replicated-average global clip，以及进程组建立失败的安全边界。全部待用户审阅
- 服务器执行：当前修复形成clean commit并获确认前暂时 `LOCKED`；失败experiment保留，只能用新ID重启

第一轮本地只编辑纯文本源码、配置和说明。没有在本地安装/import 项目依赖，没有运行 pytest、数据、模型、训练、评价或 GPU 任务。第一轮运行正确性必须由获批后的服务器 tests/strong smoke 证明。第二轮不得改写第一轮 MVP 代码语义，只能复用公共模块并新增 rollout 相关入口、配置和脚本。

## 1. 第一轮冻结实现范围

第一轮已实现 Qwen3-4B、30k UltraFeedback、seed42 的八条最终 LoRA 轨迹；这些轨迹属于冻结基线，第二轮只读引用，不重跑、不覆盖：

```text
DPO-10
DPO-100
SSPO-hard-exp
SOPPO-PE-exp
SOPPO-PE-static lambda = 0.1 / 0.3 / 0.5 / 1.0
```

旧 SFT、hard-static、Pseudo-target、DPO-style PE、linear/exp-warmup lambda 和全参 FSDP 路径已删除。保留数据隔离、Qwen manifest、DPO reference cache、独立 test、L18 `C_epsilon` 和 fail-closed 顺序任务图。`scripts/cluster/` 保留旧 Slurm 适配；`scripts/standalone/` 只替换平台层并复用同一批 stage worker、Python 入口和冻结配置。

## 1.1 第二轮新增范围

第二轮只新增两条 rollout 相关 PE 实验：

```text
SOPPO-PE-sft-rollout-exp
SOPPO-PE-rollout-only-exp
```

DPO-10、DPO-100、SSPO-hard-exp、第一轮静态 PE 与第一轮 `SOPPO-PE-exp` 均作为冻结基线只读引用。第二轮必须使用独立 experiment_id、独立输出根目录和独立命令入口，最终合并阶段只读取两轮各自导出的聚合结果。

## 1.2 round2 TP=2 训练与独立 rollout 边界

第二轮不再依赖未提供的外部 Megatron/rollout entrypoint，也不复用第一轮 DDP trainer。4090 专用实现把三张卡固定分成两个资源池：

- GPU `0,1`：`src/round2/tp_trainer.py` 通过 `torchrun --nproc_per_node=2`、Transformers `tp_plan="auto"` 和 PEFT TP-LoRA 把 Qwen3-4B 权重切到两卡，`TP=2, PP=1, DP=1`。Transformers 5.4把权重表示为普通本地Tensor slices；启动门禁逐项读取原始safetensors header的完整shape，核对TP plan/device mesh和每rank预期local shape，拒绝把两份完整复制模型冒充TP。
- GPU `2`：`src/round2/run_rollout.py` 常驻一个 vLLM engine，每个 optimizer step 加载训练端刚发布的只读 LoRA adapter，生成候选后卸载该 adapter。
- 两个 Python 环境隔离：`requirements-round2-train.txt` 固定 Transformers 5.4+/PEFT 0.19+ 的 TP-LoRA 侧；`requirements-round2-rollout.txt` 固定 vLLM 0.9.2 侧，避免用一个环境强行满足不兼容依赖。
- 训练环境同时显式固定项目测试/数据导入所需的 `datasets==2.21.0` 与 `tqdm==4.67.1`；安装后不仅执行 `pip check`，还实际 import 并记录二者版本，避免仅依赖传递依赖造成 collection-time 缺包。
- GPU ID、Git commit、模型/data/固定锚点路径、采样参数均写入 resolved config。preflight 只信 resolved config，并核对 clean checkout、完整 commit、三张 4090 全空闲以及实际 `CUDA_VISIBLE_DEVICES`。

每个训练 step 的顺序是：发布当前 adapter → rank0 向 GPU2 发出 56-prompt 请求 → GPU2 原子写回候选对 → 两个 TP rank 在同一 56-pair population 上求 PE 系数 → 依次回传 8 labeled + 56 dynamic pairs → 一个 optimizer step → 发布下一版 adapter。队列由 `queue_protocol.py` 定义 request/response schema；只有含 `READY.json` 和 SHA-256 的完整 adapter 目录能被 rollout 读取。

Transformers 5.4/PEFT TP-LoRA同时包含本地sharded LoRA张量与跨rank replicated LoRA张量。梯度裁剪不能分别调用普通单rank `clip_grad_norm_`：实现先对sharded参数的平方范数做SUM，对replicated参数做SUM/world_size避免重复计数，再把同一个TP-global L2系数应用到两个rank；分类未覆盖全部可训练参数时fail-closed。

两条方法共享同一批 prompt、固定单回复锚点、采样超参和 current-policy 定义，但不共享实际生成结果：

- `SOPPO-PE-sft-rollout-exp` 每个 prompt 生成 1 条，组成 `fixed response_a anchor vs rollout_0`；
- `SOPPO-PE-rollout-only-exp` 每个 prompt 生成 2 条独立候选，组成 `rollout_0 vs rollout_1`。若只生成并复用同一条，候选会变成 A=B，`p_i=0.5`，无法检验 rollout-only。

第二轮 logical optimizer batch 始终为 8 labeled + 56 dynamic pairs。两张训练卡只承担模型 TP，不做 data parallel；每次物理前/反向仅 materialize 一个 preference pair，按完整 population 的精确一阶系数累计后只执行一次 optimizer step。该实现不改变 global64、PE 定义、两 epoch、lr1e-5 或 `gamma_t`。

2026-08-23 用户已冻结此前开放项：`00_prepare_sft_anchor.sh` 从第一轮24,000行公开 `unlabeled_train.jsonl` 的已随机换位 `response_a` 确定性生成单回复锚点；源数据不覆盖，已有派生目录只在 manifest、SHA及逐行 `response/ID/prompt` 全部一致时复用。该锚点不是独立高质量 SFT 语料，结果解释限于“固定历史回复锚点 vs 纯在线 rollout”。两条方法统一使用 Qwen3 non-thinking `temperature=0.7/top_p=0.8/top_k=20/min_p=0`，validator拒绝任何不同值，每步 request/response 都记录完整采样四元组。

round2 执行入口位于 `scripts/round2/`，配置位于 `configs/round2/`。`start_all.sh` 在独占服务器后台依次完成锚点生成/复核、server tests、两种方法的生产路径 strong smoke、正式训练、validation-selected 独立 test 评价与 sample-free Round2 聚合导出；`status_all.sh` 同时只读展示总链、strong smoke与formal子状态。每个vLLM worker运行在独立进程组中，正常/失败退出都先写STOP，超时后只对该worker组TERM/KILL，避免EngineCore残留；`stop_all.sh` 仍只终止整个experiment记录的控制器进程组。完整服务器命令见 `scripts/round2/EXECUTION_GUIDE.md`。

## 2. 关键实现

### 2.1 配置合同

`src/config.py` 递归加载 YAML、应用 dotted override，并 fail-closed 验证：

- 方法名只能是五类配置接口；static lambda 只能取四个预注册值；
- formal 模型必须 bf16/2048，LoRA 必须 r8/alpha16/dropout0/all projections；
- formal logical global batch 必须64；运行时只允许1/2/4卡并分别使用16/8/4次梯度累积，joint始终为全局8/56；梯度 backward subbatch 限制为每 rank 最多2 pair；
- DPO 固定1 epoch/lr1e-6/beta.1；SSPO/PE固定2 epochs/lr1e-5/beta10/margin2；
- AdamW/wd0/cosine/warmup.1/clip1/seed42；
- paper gamma、30k counts/ratios、reference cache与 label isolation 约束。

smoke 明确设置 `training.smoke_mode=true`，只缩小数据量和 optimizer step；序列上限与精度保持正式 bf16/2048，并构造 logical batch>backward subbatch 来覆盖显存分块路径，不改变损失定义。

### 2.2 LoRA 与 checkpoint

`src/model/model_utils.py`：

- 每次从冻结绝对路径、manifest 校验后的 Qwen3 base 离线加载；
- `peft.LoraConfig` 注入 q/k/v/o/gate/up/down；base 参数冻结；
- 启动时逐名验证只有预注册 target 的 `lora_A/lora_B` 可训练；adapter 重载还核对 rank/alpha/dropout/targets、base 与 manifest；
- 多卡档位使用 DDP，单卡档位不包 DDP；不再做 full-state FSDP gather；
- checkpoint 写 `adapter_model.safetensors`、`adapter_config.json`、tokenizer、`run_config.yaml` 和 `checkpoint_meta.json`；
- evaluator 可重新加载 adapter；GetSlice loader 识别 adapter 并在内存中 `safe_merge`。

adapter 不含 optimizer/scheduler state，`--init-checkpoint` 可继续微调参数，但不是 bit-exact resume；hard threshold EMA 也会重新初始化。

### 2.3 第一轮损失实现

- `src/model/dpo_loss.py`：response-only token-sum logp、response-token mean logp、任意 A/B label DPO。
- `src/model/sspo_loss.py`：SimPO labeled loss、margin-free PE pair probability、paper gamma、normalized fixed lambda、Gaussian KDE Bayes threshold、mean/std/threshold EMA 和 single-response hard logistic risk。
- KDE 使用论文的 200 grid；论文未给 bandwidth，代码冻结 per-class Scott rule并记录 bandwidth/risk/threshold。
- `src/model/pe_loss.py`：dense PE，加跨 rank完整 optimizer population 的精确一阶系数和第二次前向 surrogate。

SSPO-hard 对 unlabeled pair 的 A/B 独立打 hard label；PE 对 A/B 形成一个 direction-unknown pair probability。两者都看不到 hidden label。第二轮可复用 `pe_loss.py` 的 PE 公共实现，但必须新增 rollout candidate construction，不得改变第一轮静态 pair 语义。

### 2.4 Trainer 与 batch

`src/training/trainer.py` 提供统一 CLI。它从实际 `torchrun` world size 解析1/2/4卡执行档位，并在写出 resolved config 前重新冻结设备相关字段。三种档位为：

`src/data/dataset.py` 将 Qwen3 chat prompt 与 response+EOS 分别 tokenize 后拼接 token IDs，从构造上固定 response-only mask；不再假设 tokenizer 对“prompt”和“prompt+response”两次编码具有前缀稳定性。

| GPU数 | 梯度累积 | DPO每卡logical batch | joint每卡labeled | joint每卡unlabeled | 全局组成 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 4 | 8 | 56 | 8 L + 56 U |
| 2（YAML默认） | 8 | 4 | 4 | 28 | 8 L + 56 U |
| 4 | 4 | 4 | 2 | 14 | 8 L + 56 U |

joint 的 unlabeled pattern 分别为 `[3,4]×8`、`[3,4]×4`、`[3,4]×2`；每个档位都在偶数 microstep 另取一个 labeled pair。三卡因不能精确保持 global64 与8/56而 fail-closed 拒绝。

hard 第一遍收集全局 labeled winning/losing 与全部 response reward并更新 KDE/EMA，第二遍回传。PE 第一遍只收集56个全局 pair probability并求 exact coefficient，第二遍回传。DDP `no_sync` 只延迟通信到该 optimizer step 的最后一次 backward，不改变归一化。

正式 bf16/2048 在 A800 上实测发现 DPO logical microbatch=4 的峰值显存不足。实现保留上述 logical batch 与采样顺序，但把所有有梯度的 DPO/SimPO/hard/PE 第二遍限制为每 rank 最多2 pair 的 backward subbatch：DPO logical 4 为 `2+2`，joint logical `3/4` 为 `2+1`/`2+2`。每条损失继续除以完整本地 population，DDP 仍只在 optimizer step 最后同步。PE 第一遍仍一次覆盖完整全局56 pair并产生同一组 `dL_PE/dp_i`，所以这一修复不改变目标函数或 PE population 语义。

validation：DPO 用 reference delta；SSPO/PE 用 margin-free SimPO mean-logp delta。best 依次按更高 accuracy、更低 Brier；每次 metrics 都保留 score type、loss weights、global batch与 hard/PE诊断。

### 2.5 选择、评价与观测

- `src/training/selectors.py`：共同 raw mean-logp score 下 DPO-10 对训练前显式禁用 adapter 的冻结 base 的 .05 headroom gate，并核对前后 score type/样本数；四条 static lambda validation-only selector。DPO-100只作为 oracle。
- `src/evaluation/evaluator.py`：独立 adapter 加载；只有 DPO 读取 reference cache，只有此入口读取 test private labels。
- `src/evaluation/aggregate.py`：第一轮要求八个评价完整，报告 dynamic controlled 与 validation-selected static 差值，不在 test 上挑最好方法。
- `src/round2/evaluate.py` 与 `aggregate.py`：只在训练结束后读取 validation-selected adapter；前者在单卡上读取独立3000 test私有标签并把逐样本预测留在服务器，后者只比较两条 Round2 方法并导出无样本聚合。第一轮冻结摘要留到结果交接时只读合并，不回写任一轮产物。
- `src/evaluation/c_epsilon.py`：Qwen L18，epsilon `{.01,.025,.05,.10}`，module-first equal5/equal7。
- `src/data/audit_prepared_data.py`：提交前重验30k行数、SHA-256、跨 split ID、公开隐藏标签和私有标签精确连接；审计摘要进入回传白名单。
- `observe/.../GetSlice/utils/model_utils.py`：相对原工具唯一的当前项目兼容改动是 adapter-aware offline load + in-memory safe merge。

## 3. 第一轮冻结超参

| 项 | DPO-10 / DPO-100 | SSPO-hard / SOPPO-PE |
| --- | ---: | ---: |
| epoch | 1 | 2 |
| lr | 1e-6 | 1e-5 |
| loss beta | DPO 0.1 | SimPO 10 |
| margin | — | 2 |
| logical global batch | 64（每卡4×累积16/8/4×卡数1/2/4；backward subbatch≤2） | 64 = 8 L + 56 U（backward subbatch≤2） |
| max seq len | 2048 | 2048 |
| LoRA | r8/alpha16/dropout0 | 同左 |
| optimizer | AdamW, wd0 | 同左 |
| scheduler | cosine, warmup .1 | 同左 |
| epochs | 1 | 2 |

SSPO/PE exp：`gamma0=1`、`gamma_min=2700/26700`、`decay=.01`。PE static使用 normalized lambda `{.1,.3,.5,1.0}`。hard：prior.5、EMA.95、KDE grid200、Scott bandwidth。PE：epsilon1e-8、L1、denominator不detach。

## 4. 第一轮服务器入口与不重复的八条训练

旧共享集群入口保留在 `scripts/cluster/EXECUTION_GUIDE.md`：环境、模型、数据准备后由 `submit_all.sh` 提交 Slurm `afterok` DAG；每次提交从指定Git commit导出一个不含`.git`的源码快照并记录全文件SHA-256 manifest，worker只运行该快照，因此不同commit的DAG可以并存而不共享可变checkout。节点级故障恢复继续使用registry-scoped `cancel_pipeline.sh`与`submit_from_dpo.sh`，不得按共享账户整批取消任务。

当前独占服务器入口为 `scripts/standalone/EXECUTION_GUIDE.md`，基目录由 repo 位置反推并要求 `SOPPO/` 与静态 `ICLR/` 平级：

1. `standalone/00_server_setup.sh`：使用可配置 conda 或 Python 3.10 venv 建立路径环境，不读取旧集群 module；
2. `standalone/02_download_model.sh` 与 `02_prepare_data.sh`：完整产物存在时只按 manifest/audit 复核，否则下载并生成；
3. `standalone/start_pipeline.sh`：用 `nohup + setsid` 启动后台顺序控制器；`status_pipeline.sh` 读取原子 registry 与日志，`stop_pipeline.sh` 只操作 registry 核验过的进程组；
4. `standalone/run_pipeline.sh`：从 `SOPPO_TRAIN_GPU_IDS` 选择1/2/4卡训练、单卡后处理，一次只运行一个 arm；每一步成功才进入下一步，任一步失败立即停止；
5. standalone controller 仍通过 legacy `SOPPO_CLUSTER_SCRIPT_DIR` 注入新的 runtime helper，从而直接复用 `cluster/01_server_tests.sh`、`03_smoke.sh` 以及 `02_finalize_inputs.sh` 到 `08_aggregate.sh` 的 stage body；它不调用 `sbatch/squeue/scancel`。

两种平台都会在开始前重验模型 manifest、30k 数据、clean Git checkout，并锁定完整 commit。旧服务器仍在排队的同一实验必须先按旧 registry 精确取消，避免新旧服务器产生两个同名运行。

第一轮共同逻辑任务图：

```text
tests -> strong smoke -> oracle/reference
      -> stage03 [DPO-10, DPO-100] -> DPO-10 vs frozen-base headroom
      -> stage04 [4 static PE] -> static selection
      -> stage05 [hard-exp, PE-exp]
      -> C_epsilon [base + 8]
      -> evaluation [8] -> aggregate/export
```

stage03/04/05 合计正好八条 first-round final trajectories，都写在 `runs/<experiment>/main/`，不存在预实验后重复训练。第二轮不得复用该 experiment_id 或输出目录。

## 5. checkpoint、产物与空间

- DPO：每20 step及 final；DPO-10预计约42 steps，保留20/40/final。
- SSPO/PE：每40 step及 final；全部为小型 LoRA adapter，显著小于旧全模型 checkpoint估算。
- `C_epsilon` 每轨迹最多选10点；合并只发生在当前进程内。
- server 保存 data/model/cache/adapters/raw logs/private predictions/C_epsilon raw。
- 本地只回传 summary JSON/CSV/Markdown、聚合图表、配置、manifest、环境摘要、registry 与路径索引。

## 6. strong smoke

`03_smoke.sh` 从正式 split 中选取字符长度最大的真实样本，以正式 bf16/2048 压测；它使用与正式训练相同的1/2/4进程档位，并随 world size 扩展 fixture。旧集群要求所选数量的A800，standalone 默认要求每卡至少79000 MiB。tokenization gate 要求 labeled-train 与 unlabeled-train 都至少有一个序列实际达到2048截断上限，validation只记录长度而不强制截断：

- 真实 Qwen3 offline/manifest 与 response mask；
- reference cache；
- LoRA base-frozen、所选1/2/4卡执行路径；
- DPO-10、DPO-100、hard-exp、PE-exp、PE-static各一步；
- hard KDE/EMA、exact-global PE、finite gradient；
- PE adapter保存后重载再训练一步。

正式任务另有运行时 GPU gate。standalone 不把 SKU 名称静默写死为 A800，会把实际卡名、显存和 torch CUDA 版本写入 hardware CSV；若实际 SKU 不同，结果交接必须披露。smoke通过只表示工程接口闭环，不代表30k训练一定不会出现后期数值或wall-time问题。

第二轮在 strong smoke 前增加 `round2/02_wait_for_idle_gpus.sh`。后台控制器完成数据/配置/tests 后即可无限等待，不要求用户保持 SSH：默认每30秒只读查询 resolved config 的GPU0–2，要求完整90秒无 compute PID、每卡已用显存不超过1024MiB且利用率不超过5%；同时持续核对4090/23GiB硬件和 clean Git commit。状态原子写入 `gpu_wait.json`，停止仍由实验专属进程组完成。等待器不发送信号，也不替代正式 preflight；无调度器导致的放行后抢卡会被 preflight 失败关闭。

随后运行独立的 `round2/02_strong_smoke.sh`。它不复用第一轮五方法 smoke，而是让两条 rollout 方法各自完成一个生产路径 optimizer step：真实 TP=2 Qwen3 LoRA、8+56 完整 population、物理 pair subbatch=1、最长真实 labeled 样本的 2048 backward、每条 rollout 强制512 token、adapter 保存/哈希/vLLM回载、在线候选构造、PE反传、optimizer step、adapter 再发布与8条最长 validation评价。它必须在正式两条轨迹之前成功，且服务器仍须验证实际 wall time 与峰值显存。

## 7. 静态复核与服务器待验证

本地复核范围：shell `bash -n`、`git diff --check`、旧接口/方法/路径静态搜索、第一轮入口未被改写、第二轮只有两种方法。根据本地边界，不运行 Python import、pytest、数据、模型或 GPU。服务器已验证tests、GPU等待和vLLM base ready；修复后的checkpoint-backed TP shape门禁、PEFT TP-LoRA保存/optimizer/clip、vLLM adapter回载、失败清理、24GB峰值显存、数值、512-token rollout和完整长链仍必须由新strong smoke验证。

2026-08-24 已完成的本地静态复核：`bash -n code/scripts/round2/*.sh`、工作区与 index 的 `git diff --check` 均通过；锚点与 GPU 等待脚本具备执行位。未在本地执行 Python/pytest、数据、模型或GPU任务。`02_wait_for_idle_gpus.sh` 的真实 `nvidia-smi` 输出解析、等待状态和放行到 preflight 的行为仍须在服务器验证。

已知风险：

- 2048长度和 hard/PE两次前向会增加 wall time；实际耗时以 strong smoke和首个formal日志校准。
- 当前账户没有 `cpu`/`gpu_test` 关联，且集群只接受 `-G N` 而拒绝 typed `--gres`；辅助阶段在 `gpu` partition 申请1卡，smoke/正式训练由 `--formal-gpus 1|2|4` 选择，卡不足时由 Slurm 排队。
- Slurm 会把 batch script 复制到 `/var/spool/slurmd`；提交器通过 `SOPPO_CLUSTER_SCRIPT_DIR` 显式传递仓库中的真实脚本目录，worker 不再从 spool 路径寻找 `job_env.sh`。
- 新提交的Slurm worker会核对提交时冻结的源码快照manifest和完整Git commit，不再读取可变checkout；快照位于各自`runs/<experiment>/pipeline/source/`。在该机制加入前已经提交的旧DAG仍依赖共享checkout，必须保持其锁定commit直到结束。
- standalone 由普通后台进程而非调度器托管；`nohup + setsid` 可跨 SSH 断开，但服务器重启不会自动恢复。registry 记录 PID/PGID与逐阶段状态，当前 checkpoint 仍不支持 exact resume。
- standalone 默认使用两张约80GB卡并串行运行所有 arm，也可把 GPU ID 列表切到1张或4张；global batch与8/56不变。实际新服务器环境、driver、所选卡间通信和完整 strong smoke 尚待服务器验证。
- 2026-08-21 两条 DPO array arm 分别在 `gn014` 发生 backward OOM、在 draining 的 `gn005` 发生 DDP/NCCL timeout。修复后默认排除 `gn005,gn021`，启用 expandable-segments allocator，并由 full-length smoke 验证 backward subbatch。
- SSPO论文未给KDE bandwidth，Scott rule是明确记录的复现决定。
- v0.6将已有pair拆成两个SSPO unpaired response，是数据形态适配，不等同于论文使用UltraChat single-response corpus。
- 单种子不能支持显著性结论；`C_epsilon`不是因果证据。
- 第二轮 adapter 每 step 都必须发布给在线 rollout，因此会保留大量 LoRA checkpoint；当前不保存 optimizer/scheduler state，不支持 bit-exact 热恢复。
- 第二轮固定锚点、采样四元组、依赖和GPU等待已在 `f54f6f4` 的服务器尝试中通过；当前需完成TP shape门禁与vLLM独立清理修复的代码交接。

旧 Slurm 路径的静态复核与部分服务器门禁已有证据；Round2 两个环境、server tests、GPU wait和vLLM base ready已有服务器证据，但首条TP/vLLM strong smoke尚未完成optimizer step，正式训练未执行，不能把部分启动成功写成训练验证成功。
