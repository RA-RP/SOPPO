# SOPPO Round3 五方法实现总览

## 0. 当前状态与授权边界

- Cycle：`cycle-20260818-01` / Round3
- 实现候选：`round3-code-candidate-v0.4`
- 获批理论/实验：`r3-theory-v0.9` / `round3-exp-v1.4`
- 当前唯一活动阶段：`SERVER_EXECUTION`
- 用户批准：用户于2026-08-26明确批准方案B数据勘误与本地修改，即1,000 validation +997 independent test、畸形行确定性审计且不从train补样
- 代码交接：**用户于2026-08-26明确允许Codex commit/push v0.4、部署新exact commit、完整strong smoke通过后直接挂载formal**
- 版本：阶段C/data/reference验证的旧服务器commit为`ed1bfca002799f11ea1bad29f6f06e2e15fdd565`；v0.4 exact commit由本次获批提交产生。physical subbatch与存储投影仍须由完整strong smoke解析
- 本地边界：本地只静态编辑源码、YAML、shell与Markdown，没有在本地安装/import依赖或运行Python、pytest、数据、模型、训练、评价、聚合或GPU任务。阶段C测试只在4090-3服务器执行；Codex没有commit/push
- Round2边界：只读证据确认正式任务已在step590停止，step580/589/590保留、第二方法未启动、两个pruner未运行；旧环境已删除但runs/checkpoints不得删除或覆盖

当前实现严格隔离在`src/round3/`、`configs/round3/`与`scripts/round3/`。第一轮和Round2入口保持历史语义，不作为Round3 trainer或rollout worker。

## 1. 实现范围

Round3实现五个且仅五个方法：

```text
dpo_1k
sspo_code_loss_stratified_ultrachat_2df9e9a
dpo_8k
dpo_pe_sft_rollout
dpo_pe_rollout_only
```

共同模型是非量化ModelScope `Qwen/Qwen3-1.7B` post-trained版本，native non-thinking、BF16 forward/autocast、FP32 LoRA/optimizer state。所有方法均为1 epoch/250 optimizer steps，每25步保存一次，共10个durable checkpoints/方法。GPU0单进程训练；只有两个动态PE方法在GPU1/2各启动一个独立vLLM replica。

明确没有实现：

- `DPO+PE-static`：只登记到Round5；
- AlpacaEval/MT-Bench generation、judge或API调用：只登记到Round4；
- SSPO paper-v3 KDE、Bayes threshold、threshold EMA或joint reward statistics；
- QLoRA、DDP、TP、三卡loss分片、keep-N或自动checkpoint pruner；
- Round1/Round2的`C_epsilon`/`C_gamma`观测。Round3获批设计没有该观测，不能顺手迁入。

## 2. 规范到模块的映射

| 获批合同 | 当前实现 |
| --- | --- |
| 双源数据、独立namespace与确定性隔离 | `src/round3/data.py` |
| 显式Git ref解析为不可变full SHA | `scripts/round3/00_resolve_revisions.sh`、`preflight.py` |
| Qwen3-1.7B不可变文件/revision manifest | `src/round3/model_manifest.py`及公共`src/model/model_manifest.py`的兼容验证分支 |
| 五方法配置与fail-closed超参 | `src/round3/config.py`、`validate_config.py`、`configs/round3/*.yaml` |
| DPO total-response-logp beta .1 | `src/round3/losses.py::dpo_objective` |
| PE mean-response-logp beta10、exact 28、L1、denominator不断梯度 | `src/round3/losses.py::pe_objective`及公共`src/model/pe_loss.py` |
| GitHub commit `2df9e9a` SSPO code loss | `GitHubSSPOState`、`github_sspo_objective` |
| 单GPU逻辑population两遍精确梯度 | `src/round3/trainer.py` |
| 双replica路由、step/adapter SHA/ACK屏障 | `queue_protocol.py`、`rollout_worker.py` |
| 当前policy原子staging adapter | `checkpoint.py::publish_staging_adapter` |
| 10个完整训练态checkpoint与resume | `checkpoint.py::save_durable_checkpoint/load_training_state` |
| 冻结reference cache | `reference_cache.py` |
| 共同1K DPO selection与更早step tie-break | trainer内selection、`selection.py`独立复核 |
| independent 997-pair双head final test | `final_evaluate.py` |
| sample-free同head聚合 | `aggregate.py` |
| 3×4090、clean commit、数据/模型/磁盘门禁 | `preflight.py`、`storage_gate.py` |
| 五方法strong smoke与projected peak | `scripts/round3/03_strong_smoke.sh`、`project_storage.py` |
| standalone长链与只读状态 | `scripts/round3/run_all.sh`、`start_all.sh`、`status_all.sh` |
| GLM只测试/部署、不改源码的分阶段作业单 | `scripts/round3/GLM_VALIDATION_GUIDE.md` |

## 3. 数据与模型流

`data.py`只允许在服务器读取两个显式full commit SHA：

```text
ultrafeedback_binarized/train_prefs
  ├─ SHA排序8,000 master ── 前1,000 limited
  └─ 不直接提供validation/test

ultrafeedback_binarized/test_prefs
  ├─ 先隔离3条empty-rejected source rows
  ├─ 独立namespace先选1,000 validation
  └─ 剩余恰好997条全部进入independent test

ultrachat_200k/train_sft
  └─ 排除全部paired prompt并去重后选7,000 singles
```

`sample_id`严格包含dataset ID、resolved revision、split、prompt ID和source row index；排序key是`SHA256(namespace || NUL || 42 || NUL || sample_id)`。paired A/B位置使用独立hash，test公开文件没有label，私有label单独留在服务器。畸形行在选择前隔离到无原始文本的`malformed_source_rows.jsonl`；data manifest v2固定source/valid/malformed/reason aggregates、12,197行audit及17,997行view source manifest。preflight重算文件SHA、source identity、reason aggregate、1K前缀嵌套、public/private ID连接和四个view的canonical prompt零交叉。

模型下载脚本要求调用者显式提供resolved revision。manifest记录所有顶层模型/tokenizer文件SHA、Qwen3类型、层数和special-token摘要；配置、preflight和rollout sampling再次核对revision、pad/eos ID与文件manifest。

冻结reference对8K train、1K train、1K validation和997 test预计算response-token总log-prob；cache manifest绑定模型文件、tokenization contract、数据文件SHA和代码commit。方案B使用仓库外`dual_source_v2`与`reference_qwen3_1.7b_dual_source_v2`，不删除失败遗留的空v1目录。原始cache是服务器产物，不回传本地。

## 4. 一个optimizer step

### DPO-1K / DPO-8K

依次消费4/32个pair。第一遍按服务器strong smoke确认的physical subbatch收集完整logical batch的policy total response logp；在小型leaf tensor上计算精确DPO系数；第二遍逐physical subbatch做vector-Jacobian product，clip后只执行一次optimizer update。

### GitHub-loss SSPO

每步固定4 labeled pairs+28 UltraChat singles，共36条response。实现先收集完整4 chosen、4 rejected、28 unpaired mean logp，再严格按chosen→rejected→unpaired调用同一running state：首次chosen直接初始化，之后momentum .95，population variance加`1e-8`，normalized值clamp `[-5,5]`。threshold是本步normalized chosen最小值且保留GitHub源码中的梯度路径；不存在KDE或threshold EMA。目标是`gamma_t*SimPO+(1-gamma_t)*unpaired risk`，首步`t=0`。

### 两个动态PE

训练先原子发布current LoRA staging adapter。每个generation job按`method/step/sample_id/draw_index`稳定路由到replica 0或1；两个replica都必须先核对并回传相同`(method_id, optimizer_step, adapter_sha256)`，缺任一ACK即fail closed。SFT+rollout每source生成1条；rollout-only按两个独立draw/seed生成2条，不能复制文本或跨方法/step共享。

训练收齐28个candidate pairs后，同时在4个labeled pair上计算reference-DPO、在完整28-pair logical population上计算PE，联合为`(DPO+0.1*PE)/1.1`。第一遍求完整目标对response score的系数，第二遍累计到同一个optimizer update；physical subbatch不能创建额外statistics更新或optimizer step。

## 5. Checkpoint、选择和final test

durable checkpoint固定为steps `25..250`每25步一次。每个目录包含：

- PEFT adapter/tokenizer与resolved run config；
- optimizer、scheduler、Python/NumPy/Torch CPU/CUDA RNG和global step；
- SSPO额外`running_mean/running_var`及固定state超参；
- adapter和training-state SHA-256、base/model manifest/commit/config绑定。

动态current-policy staging adapter与durable checkpoint分开。staging目录按step不可变发布并在运行期间保留；当前实现没有自动删除逻辑。任何后续清理都必须在结果与保留策略批准后另行执行。

每个durable checkpoint只在共同1,000-pair validation上计算reference-DPO beta .1 NLL。SSPO selection前后state SHA必须相同；non-finite checkpoint记录无效但不替代数值，十个全部无效则方法工程失败。`selection.py`再次按原始`(loss, earlier_step)`复核`best.json`。

final evaluation只加载每方法`best.json`指向的一个checkpoint以及frozen base，在独立997 pair上同时输出：

- `dpo_reference_delta_beta_0.1`；
- `raw_mean_logp_delta_beta_10`。

每个head含tie-half-credit Accuracy、report-only clamp NLL、Brier、ECE-15、固定分位数/confidence/collapse描述项和概率和。逐样本双head预测与private labels留在服务器；`aggregate.py`只输出无样本聚合，并只做同head差值，不生成综合分数。

## 6. 配置与固定超参

| 项 | Round3值 |
| --- | --- |
| 模型 | `Qwen/Qwen3-1.7B` post-trained，non-thinking |
| LoRA | r8/alpha16/dropout0，q/k/v/o/gate/up/down，非量化 |
| precision | BF16 forward/autocast，FP32 trainable/optimizer |
| optimizer | AdamW，lr `1e-5`，betas `.9/.999`，eps `1e-8`，wd0 |
| scheduler | cosine，warmup `.1`，clip1 |
| 长度 | total2048，prompt1024左截断，completion1024右截断 |
| steps/checkpoint | 250；每25步保存/eval |
| DPO | total response logp，reference beta `.1` |
| PE | mean response logp，beta10，epsilon `1e-8`，L1，denominator不detach，lambda `.1` |
| SSPO | SimPO beta10/margin2，prior `.5`，EMA `.95`，clip5，gamma floor `.125`/decay `.001` |
| rollout | temp`.7`/top-p`.8`/top-k20/min-p0/repetition1/presence0，max new1024 |
| seed | data/train base seed42；rollout另绑定step/sample/draw |
| deterministic backend | `CUBLAS_WORKSPACE_CONFIG=:4096:8`、PyTorch deterministic algorithms、TF32关闭；用于formal训练与checkpoint重放一致性 |
| GPU | train GPU0；dynamic replicas GPU1、GPU2 |

source YAML中的`physical_pair_subbatch=1`是待服务器production-path strong smoke确认的保守候选，不是本地验证事实。formal resolved config必须携带strong-smoke投影的`projected_peak_bytes`。

## 7. 服务器阶段入口与逐项门禁

当前重新进入`SERVER_EXECUTION`。阶段A/C、data v2与reference cache已形成服务器证据并保留；v0.4已获commit/push与重新部署授权，但以下入口仍必须按“完整strong smoke成功后才启动formal”的顺序执行：

```text
00_prepare_data.sh
02_prepare_reference_cache.sh
03_strong_smoke.sh
run_all.sh / start_all.sh
04_evaluate.sh / 05_aggregate.sh
```

环境、模型、dataset revisions与experiment ID全部要求显式输入。`03_strong_smoke.sh`对五方法各执行一个完整logical population的production step，写出完整训练态代表checkpoint；两个动态方法同时覆盖双replica ACK与staging handoff。`project_storage.py`据实际checkpoint/staging/queue尺寸、最大生成文本上界、数据源parquet/Arrow cache、保留的strong-smoke产物和平台日志投影完整Round3 peak。formal只在一次性`free_bytes >= 2*projected_peak_bytes`门禁通过后解析配置；门禁和脚本都不删除Round2或其他产物。

`status_all.sh`只读controller、五个state/best、metrics尾部、`nvidia-smi`和`df`，并明确显示自动pruner关闭。`stop_all.sh`默认仅预览；即使将来明确授权`--execute`，也只向本experiment记录且重新核对的controller进程组发送TERM，不删除checkpoint。

完整服务器阶段顺序与二次授权门禁见`scripts/round3/EXECUTION_GUIDE.md`。手册存在不构成data、strong smoke或formal授权。

GLM执行服务器工作时另以`scripts/round3/GLM_VALIDATION_GUIDE.md`为操作边界：阶段A只读核验后必须先回传；部署、CPU tests、data v2和strong smoke分别授权，任何失败立即停止，禁止现场改源码或自行进入formal。

## 8. 静态复核与服务器待验证

方案B服务器阶段C、data v2和reference cache已通过。`round3-20260826-01`的DPO-1K smoke训练成功后因入口未导出`PYTHONPATH`而无法调用verifier；保留失败证据后，`round3-20260826-02`的DPO-1K及checkpoint验证通过，SSPO单步训练/checkpoint成功但独立重放的LoRA更新最大绝对/相对差为`1.3404528544924688e-6`/`1.99992835521698`，超过`1e-7`/`1e-6`合同，后续方法与formal未启动。服务器诊断在不改loss/batch/容差下启用deterministic algorithms和CUBLAS workspace后，loss、参数绝对/相对差均为0；据此形成v0.4修复。

服务器代码交接后必须依序验证：

1. candidate依赖安装、`pip check`、实际版本与environment freeze；
2. dataset/model resolved revision、source/cache/file SHA和canonical隔离；
3. CPU loss/config/protocol tests；
4. 五方法完整logical-population strong smoke、最长真实样本、finite loss/gradient与显存；
5. SSPO save/load缺失state fail-closed、selection state不变和下一batch数值round-trip（loss绝对差`<=1e-7`，LoRA参数最大绝对/相对差`<=1e-7/1e-6`）；
6. 两个vLLM副本对同一adapter ACK、draw路由、独立生成与LoRA卸载；
7. checkpoint optimizer/scheduler/RNG resume、十个durable保留且无pruner；
8. projected storage及两倍free门禁。

当前是**获批重新部署、尚待完整生产路径验证的实现候选**。v0.4在trainer与verifier进入CUDA前统一确定性后端，并修复strong-smoke verifier的模块路径；未放宽预注册容差、未改变loss、batch、长度或方法。用户已授权在新exact commit、新experiment attempt从头验证，并在五方法strong smoke、双vLLM与存储投影全部通过后直接启动formal。

## 9. Round1/Round2历史实现（非当前入口）

以下内容保留用于追溯旧代码，不描述Round3授权或当前服务器实时状态。

### 历史状态

- 第一轮 Experiment：`exp-20260819-01-mvp`（冻结基线，只读引用）
- Round2 TP=2 + 单vLLM实现与现场交接见`../human_read/code/ROUND2_LIVE_HANDOFF.md`
- Round2最新只读证据为`exp-20260824-05-round2-tp2`在step590停止；step580/589/590保留、best step480、第二方法未启动、pruner未运行。核验时服务器checkout为`33f0eea`，但该checkout快照不替代controller内的运行provenance；下文更早快照也不能覆盖该证据

第一轮本地只编辑纯文本源码、配置和说明。没有在本地安装/import 项目依赖，没有运行 pytest、数据、模型、训练、评价或 GPU 任务。第一轮运行正确性必须由获批后的服务器 tests/strong smoke 证明。第二轮不得改写第一轮 MVP 代码语义，只能复用公共模块并新增 rollout 相关入口、配置和脚本。

### 9.1 第一轮冻结实现范围

第一轮已实现 Qwen3-4B、30k UltraFeedback、seed42 的八条最终 LoRA 轨迹；这些轨迹属于冻结基线，第二轮只读引用，不重跑、不覆盖：

```text
DPO-10
DPO-100
SSPO-hard-exp
SOPPO-PE-exp
SOPPO-PE-static lambda = 0.1 / 0.3 / 0.5 / 1.0
```

旧 SFT、hard-static、Pseudo-target、DPO-style PE、linear/exp-warmup lambda 和全参 FSDP 路径已删除。保留数据隔离、Qwen manifest、DPO reference cache、独立 test、L18 `C_epsilon` 和 fail-closed 顺序任务图。`scripts/cluster/` 保留旧 Slurm 适配；`scripts/standalone/` 只替换平台层并复用同一批 stage worker、Python 入口和冻结配置。

### 9.2 第二轮新增范围

第二轮只新增两条 rollout 相关 PE 实验：

```text
SOPPO-PE-sft-rollout-exp
SOPPO-PE-rollout-only-exp
```

DPO-10、DPO-100、SSPO-hard-exp、第一轮静态 PE 与第一轮 `SOPPO-PE-exp` 均作为冻结基线只读引用。第二轮必须使用独立 experiment_id、独立输出根目录和独立命令入口，最终合并阶段只读取两轮各自导出的聚合结果。

### 9.3 Round2 TP=2训练与独立rollout边界

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

### 9.4 关键实现

#### 历史配置合同

`src/config.py` 递归加载 YAML、应用 dotted override，并 fail-closed 验证：

- 方法名只能是五类配置接口；static lambda 只能取四个预注册值；
- formal 模型必须 bf16/2048，LoRA 必须 r8/alpha16/dropout0/all projections；
- formal logical global batch 必须64；运行时只允许1/2/4卡并分别使用16/8/4次梯度累积，joint始终为全局8/56；梯度 backward subbatch 限制为每 rank 最多2 pair；
- DPO 固定1 epoch/lr1e-6/beta.1；SSPO/PE固定2 epochs/lr1e-5/beta10/margin2；
- AdamW/wd0/cosine/warmup.1/clip1/seed42；
- paper gamma、30k counts/ratios、reference cache与 label isolation 约束。

smoke 明确设置 `training.smoke_mode=true`，只缩小数据量和 optimizer step；序列上限与精度保持正式 bf16/2048，并构造 logical batch>backward subbatch 来覆盖显存分块路径，不改变损失定义。

#### 历史LoRA与checkpoint

`src/model/model_utils.py`：

- 每次从冻结绝对路径、manifest 校验后的 Qwen3 base 离线加载；
- `peft.LoraConfig` 注入 q/k/v/o/gate/up/down；base 参数冻结；
- 启动时逐名验证只有预注册 target 的 `lora_A/lora_B` 可训练；adapter 重载还核对 rank/alpha/dropout/targets、base 与 manifest；
- 多卡档位使用 DDP，单卡档位不包 DDP；不再做 full-state FSDP gather；
- checkpoint 写 `adapter_model.safetensors`、`adapter_config.json`、tokenizer、`run_config.yaml` 和 `checkpoint_meta.json`；
- evaluator 可重新加载 adapter；GetSlice loader 识别 adapter 并在内存中 `safe_merge`。

adapter 不含 optimizer/scheduler state，`--init-checkpoint` 可继续微调参数，但不是 bit-exact resume；hard threshold EMA 也会重新初始化。

#### 第一轮损失实现

- `src/model/dpo_loss.py`：response-only token-sum logp、response-token mean logp、任意 A/B label DPO。
- `src/model/sspo_loss.py`：SimPO labeled loss、margin-free PE pair probability、paper gamma、normalized fixed lambda、Gaussian KDE Bayes threshold、mean/std/threshold EMA 和 single-response hard logistic risk。
- KDE 使用论文的 200 grid；论文未给 bandwidth，代码冻结 per-class Scott rule并记录 bandwidth/risk/threshold。
- `src/model/pe_loss.py`：dense PE，加跨 rank完整 optimizer population 的精确一阶系数和第二次前向 surrogate。

SSPO-hard 对 unlabeled pair 的 A/B 独立打 hard label；PE 对 A/B 形成一个 direction-unknown pair probability。两者都看不到 hidden label。第二轮可复用 `pe_loss.py` 的 PE 公共实现，但必须新增 rollout candidate construction，不得改变第一轮静态 pair 语义。

#### 历史trainer与batch

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

#### 历史选择、评价与观测

- `src/training/selectors.py`：共同 raw mean-logp score 下 DPO-10 对训练前显式禁用 adapter 的冻结 base 的 .05 headroom gate，并核对前后 score type/样本数；四条 static lambda validation-only selector。DPO-100只作为 oracle。
- `src/evaluation/evaluator.py`：独立 adapter 加载；只有 DPO 读取 reference cache，只有此入口读取 test private labels。
- `src/evaluation/aggregate.py`：第一轮要求八个评价完整，报告 dynamic controlled 与 validation-selected static 差值，不在 test 上挑最好方法。
- `src/round2/evaluate.py` 与 `aggregate.py`：只在训练结束后读取 validation-selected adapter；前者在单卡上读取独立3000 test私有标签并把逐样本预测留在服务器，后者只比较两条 Round2 方法并导出无样本聚合。第一轮冻结摘要留到结果交接时只读合并，不回写任一轮产物。
- `src/evaluation/c_epsilon.py`：Qwen L18，epsilon `{.01,.025,.05,.10}`，module-first equal5/equal7。
- `src/data/audit_prepared_data.py`：提交前重验30k行数、SHA-256、跨 split ID、公开隐藏标签和私有标签精确连接；审计摘要进入回传白名单。
- `observe/.../GetSlice/utils/model_utils.py`：相对原工具唯一的当前项目兼容改动是 adapter-aware offline load + in-memory safe merge。

### 9.5 第一轮冻结超参

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

### 9.6 第一轮服务器入口与不重复的八条训练

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

### 9.7 历史checkpoint、产物与空间

- DPO：每20 step及 final；DPO-10预计约42 steps，保留20/40/final。
- SSPO/PE：每40 step及 final；全部为小型 LoRA adapter，显著小于旧全模型 checkpoint估算。
- `C_epsilon` 每轨迹最多选10点；合并只发生在当前进程内。
- server 保存 data/model/cache/adapters/raw logs/private predictions/C_epsilon raw。
- 本地只回传 summary JSON/CSV/Markdown、聚合图表、配置、manifest、环境摘要、registry 与路径索引。

### 9.8 历史strong smoke

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

### 9.9 历史静态复核与服务器待验证

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
- 第二轮固定锚点、采样四元组、依赖、GPU等待、vLLM ready和失败清理已获服务器证据；当前需完成PEFT 0.19.1 / Transformers 5.4.0 TP-hook兼容修复的代码交接。

以下是Round2某一历史attempt当时的静态复核快照：旧Slurm路径与部分服务器门禁已有证据，SFT+rollout完成过单步TP/PE，而当时rollout-only尚未完成optimizer step。后续Round2已在step590停止且没有final result；本段不能当作Round3验证证据。
