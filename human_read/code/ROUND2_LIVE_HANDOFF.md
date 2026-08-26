# Round2 现场交接：3×4090 正式运行、资源瓶颈与待决策项

> 2026-08-26状态勘误：本文主体是2026-08-24的历史现场快照，不再表示全局当前阶段。后续只读核验确认正式experiment为`exp-20260824-05-round2-tp2`，controller已在step590以exit 143停止；第一方法保留step580/589/590且best指向step480，第二方法未启动，两个keep-20 pruner PID均为陈旧记录且进程不存在。旧Round2环境已被外部操作删除，但run/checkpoint仍保留。不得删除或覆盖这些产物；任何新执行仍须重新核验实时GPU、磁盘、进程与checkout。

> 快照日期：2026-08-24（Asia/Shanghai）。这是服务器现场与讨论结论的交接记录，不替代服务器实时状态。新 agent 必须先运行只读查询，不得根据本文猜测当前 step、PID、experiment ID 或 Git commit。

## 1. 权限与边界

- 当前 cycle：`cycle-20260818-01`。
- 快照时活动阶段：`SERVER_EXECUTION`；用户当时已经在3×4090 standalone服务器启动获批的Qwen3-4B Round2长链。
- 用户反复明确：Codex 不得自行 `git commit`、`git push`；只保留工作区 diff 供用户审阅和手工提交。
- 当前正在运行的4B实验不得被QLoRA或1.7B讨论静默改写。任何模型、量化、并行或数据合同变化都必须使用新experiment ID，并先回到实验/代码门禁。
- 服务器`/data`只剩约111G是整个共享文件系统的状态，不等于这些空间由本用户占用；不得删除其他用户、其他项目或其他experiment的文件。

## 2. 已冻结的 Round2 科学设置

两条正式方法串行执行，不并行：

1. `soppo_pe_sft_rollout_exp`；
2. `soppo_pe_rollout_only_exp`。

共同合同：

- base：`Qwen/Qwen3-4B`，bf16、SDPA、thinking off、`max_seq_len=2048`；
- 标准LoRA：`r=8/alpha=16/dropout=0`，目标为q/k/v/o/gate/up/down projections，不是QLoRA；
- 数据：UltraFeedback 30k；2,700 labeled train、300 validation、24,000 unlabeled prompt、3,000 private test；
- 每个optimizer step：8 labeled pair +56 dynamic pair，global64；
- 2 epochs，`floor(24000/56)×2=856` optimizer steps/方法；
- optimizer：AdamW，lr `1e-5`、wd0、cosine、warmup0.1、clip1、seed42；
- PE：完整56-pair population，`epsilon=1e-8`、L1、denominator不detach；
- GPU0–1：Transformers native TP=2、PP=1、DP=1；物理pair subbatch=1；
- GPU2：单个常驻vLLM，`temperature=.7/top_p=.8/top_k=20/min_p=0/max_new_tokens=512`；
- SFT+rollout每prompt生成1条并与固定`response_a`锚点组成pair；rollout-only每prompt生成2条独立候选。

第一轮MVP与Round2的关键差别不是数据规模或epoch，而是MVP使用冻结数据中的既有A/B，不进行在线生成；Round2每step都等待current-policy rollout。

## 3. 服务器现场快照

用户在2026-08-24提供的正式运行证据：

- 一次时间统计时第一条方法为step35/856；最近20步平均总耗时286.1秒；
- 其中vLLM `generate`平均165.4秒（57.8%），训练、TP通信、adapter发布等差值120.7秒（42.2%）；
- 按第一条方法相同速度计算，第一条剩余下限约2天17小时，两条剩余下限约5天13小时；由于第二条每prompt生成2条且估算未计完整validation/test，现实总剩余可能约6–8天；
- 随后一次`nvidia-smi`快照：GPU0约18824MiB/100%、GPU1约18834MiB/65%、GPU2约19520MiB/0%；GPU0–1处于训练或validation，GPU2已保留vLLM模型/KV cache但在等待下一请求；
- 训练模型刚加载时每rank约4–5GB，符合4B bf16权重经TP=2切半；运行高水位约18.8GB还包含长序列激活、FP32大词表log-softmax、backward临时量、CUDA/NCCL workspace与PyTorch reserved cache；
- 18.8GB实测使“当前实现直接改成单卡bf16”存在较高OOM风险，不能未经最长样本strong smoke直接切换；
- 共享文件系统快照为`/dev/sdb1 3.6T/3.3T/111G/97%`。无论占用来自谁，本实验都只能使用当时剩余空间。

以上都是带时间的快照。服务器真实experiment ID与commit未在本地证据中得到最后确认；新agent先读取：

```bash
export SERVER_BASE=/data/youchang/youchang
cd "$SERVER_BASE/SOPPO/code/scripts/round2"
bash status_all.sh
cat "$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID/controller.json"
git -C "$SERVER_BASE/SOPPO" rev-parse HEAD
git -C "$SERVER_BASE/SOPPO" status --short
```

运行期间不要`git pull`、切换checkout或编辑服务器正在使用的仓库。

## 4. 训练数据与步数口径

每个方法的数据池为2,700条labeled train与24,000条unlabeled prompt。300 validation和3,000 test不进梯度。

每个方法累计：

```text
856 × 8  = 6,848 labeled pair exposures
856 × 56 = 47,936 dynamic pair exposures
856 × 64 = 54,784 total pair exposures
```

SFT+rollout全程生成47,936条新回复；rollout-only全程生成95,872条新回复，但两者都形成47,936个dynamic pair。每epoch因drop-last使用23,968个prompt occurrence，随机舍弃32个，第二epoch重新shuffle。

## 5. checkpoint、磁盘与best语义

当前仓库代码每个optimizer step都必须先发布current adapter供下一次rollout读取，因此每个方法默认产生`step_000000`至`step_000856`，共857个adapter目录；两个方法共1,714个。

- checkpoint是PEFT LoRA adapter，不包含完整4B base，也不包含optimizer/scheduler/RNG state；不能宣称bit-exact热恢复；
- `best.json`不是独立复制的best权重，只是指向某个`policy/step_xxxxxx`；首次在step40 validation后产生；
- `complete.json`指向final checkpoint；
- 单checkpoint按10.84M LoRA参数和重复tokenizer估计约55–65MiB；全部保留约90–115GB，当前磁盘余量下不安全。

用户要求运行中外置清理策略改为“每20步保留一个”，而不是每40步。安全语义应为：

- 保留step0；
- 保留20、40、60……840；
- 保留当前step及前一步，绝不删除vLLM可能仍读取的adapter；
- 额外读取并保护`best.json`和`complete.json`所指路径；
- 不碰`.step_xxxxxx.partial`；
- 只允许删除当前experiment两个明确`policy/`目录中的非关键点。

聊天中给过外置pruner命令，但用户没有回传启动确认。新agent不得声称它正在运行；先核对：

```bash
export RUN_DIR="$SERVER_BASE/runs/$ROUND2_EXPERIMENT_ID"
ls -l "$RUN_DIR"/checkpoint_pruner*.pid "$RUN_DIR"/checkpoint_pruner*.log 2>/dev/null
for pidfile in "$RUN_DIR"/checkpoint_pruner*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid="$(cat "$pidfile")"
    ps -o pid,ppid,state,etime,args -p "$pid"
done
```

如果旧keep-40 pruner仍在，必须先用PID文件和命令行指纹确认只命中pruner，再停止它；不得用`pkill python`。step20若已经被删，当前训练无法重建，不得伪造。

keep-20后两个方法约保留88个里程碑adapter，预计约5–7GB，另加rollout JSON、日志和评价产物。

## 6. Round2当前没有C_epsilon阶段

要区分：

- 训练中PE loss确实使用数值稳定项`epsilon=1e-8`；
- Round2 `run_all.sh`没有第一轮MVP的`C_epsilon prepare/raw/derive`轨迹分析，顺序是smoke→两条formal→independent evaluation→aggregate。

如果未来给Round2新增`C_epsilon`，必须作为获批的新实现；keep-20里程碑足以提供多点轨迹，但当前运行不会自动生成`c_epsilon_trajectory.csv`。

## 7. 为什么慢，以及当前运行不能热优化什么

当前每step先在线生成，再执行两遍PE：第一遍对64个pair逐个计算A/B score，第二遍又对每个pair的A/B逐个forward/backward，约256次细粒度模型调用。TP通信被这些小调用反复支付。

当前已启动的Python进程不能安全热修改：

- physical subbatch；
- vLLM `max_num_seqs`；
-量化方式；
-模型大小；
-两副本rollout路由。

直接修改共享checkout不会改变已加载进程，反而可能使第二条方法的clean-Git/config-hash/preflight失败。当前两条核心方法必须使用相同实现和commit，不能只优化第二条。

## 8. 尚未批准的优化候选

### 8.1 Qwen3-4B QLoRA单卡训练 + 两张卡rollout

优点：4-bit NF4显著降低训练基础权重显存，并释放GPU1用于第二个rollout replica。

限制：

- QLoRA主要解决显存，4-bit反量化不保证比bf16训练更快；
- 它改变第一轮冻结的“标准bf16 LoRA”合同；
- 若trainer使用NF4 base而vLLM仍使用bf16 base，rollout不再是严格相同的current policy；要保持一致必须验证vLLM同量化base、动态LoRA和生成协议；
- 两张rollout卡应运行两个独立data-parallel replica并按稳定sample ID分56个prompt，不应把能单卡容纳的4B小模型做vLLM TP=2；rollout-only同一prompt的两条候选应留在同一replica。

粗略性能估计：约170–235秒/step，改善有限且工程/解释风险较高。未获用户批准。

### 8.2 Qwen3-1.7B标准LoRA + 单卡训练 + 两个vLLM replica

优点：同时减少训练与生成计算，保持标准bf16 LoRA和trainer/rollout policy一致；Qwen3-1.7B为1.7B参数、28层。

代价：模型容量改变，不能把1.7B Round2直接与冻结的4B MVP做同模型差值；若作为正式主线，需要新experiment/cycle，并至少重跑1.7B frozen base、DPO-10/headroom、SSPO-hard-exp、SOPPO-PE-exp以及两条rollout方法。`C_epsilon`层位也需重新讨论。

粗略性能估计：两副本rollout下约90–140秒/step（rollout+SFT），all-rollout约120–190秒/step，两条约2.5–4天。未获用户批准。

### 8.3 保留4B标准bf16 LoRA的工程候选

如果4B MVP直接可比性优先，先研究单卡bf16 LoRA的显存优化，而不是立即QLoRA：避免完整FP32 `log_softmax`中间量、批量/融合score计算、批量validation，并用最长样本完整PE strong smoke验证是否低于约22GB。当前18.8GB/TP-rank证据意味着单卡是否可行仍不确定。

### 8.4 当前讨论形成的建议（不是批准）

- 如果首要目标是把Round2墙钟时间显著降下来，优先把`Qwen3-1.7B + 单卡标准LoRA + 两个vLLM data-parallel replica`作为下一套完整实验重新设计；它同时减少训练和生成成本，预期收益大于仅用QLoRA节省训练显存。
- 如果首要目标是与现有4B MVP做最干净的同模型受控比较，应继续当前4B，或先实现4B score/validation批量化与显存优化；不要只为释放一张rollout卡就默认切QLoRA。
- `4B QLoRA + 双rollout replica`位于两者之间：可能提速，但改变量化合同且current-policy一致性需要额外验证，不是当前首选。
- 最终选择尚未得到用户明确批准；下一位agent必须先报告实时运行状态和可回收成本，再与用户作决定。

## 9. 给下一位agent的第一轮只读检查

1. 完整阅读根`AGENTS.md`、`SOPPO/AGENTS.md`、本文件、`human_read/exp/CODE_HANDOFF.md`、`human_read/exp/experiment_archive.md`中的Round2 v0.6条目、`code/CODE_OVERVIEW.md`与`machine/standalone_3x4090_server.md`；当前`human_read/exp/current_experiment.md`已专用于Round3，不是本次运行依据。
2. 让用户提供或在服务器执行`status_all.sh`、`controller.json`、两个`state.json`、最新metrics尾部和`nvidia-smi`；不要猜当前step。
3. 核实keep-20 pruner是否真的启动、旧keep-40是否仍在、step20是否存在。
4. 在用户明确决定“继续当前4B / 新4B优化 / 新1.7B设计”前，不停止当前训练、不修改服务器checkout、不提交代码。
5. 任何停止操作只能使用当前experiment记录的`stop_all.sh`与PID/进程组指纹；任何删除只能限定在当前experiment已核验的非关键adapter目录。

## 10. 可复制给下一位code agent的开场白

```text
你将接手本地研究工作区 /Users/rarp/Desktop/ICLR。请先完整阅读根 AGENTS.md、SOPPO/AGENTS.md、SOPPO/human_read/code/ROUND2_LIVE_HANDOFF.md、SOPPO/human_read/exp/CODE_HANDOFF.md、SOPPO/human_read/exp/experiment_archive.md中的Round2 v0.6条目、SOPPO/code/CODE_OVERVIEW.md、machine/CURRENT_STATE.md 和 machine/standalone_3x4090_server.md；不要把专用于Round3草案的`human_read/exp/current_experiment.md`当成本次运行依据。再向我复述你理解的职责、当前阶段、正在运行的4B Round2合同、服务器风险与尚未批准的决策。当前3×4090服务器上可能仍有正式长链运行；在获得实时证据前不要猜experiment ID、commit、step或PID，不要停止任务、修改服务器checkout或删除checkpoint。先只读核验 status_all.sh、controller.json、两个state.json、metrics尾部、nvidia-smi、df和keep-20 pruner状态。QLoRA、双vLLM副本和Qwen3-1.7B只是候选方案，不是执行授权。尤其注意：未经我审阅，不得自行git commit或git push；本地只允许静态阅读/编辑与文本检查，测试、数据、模型和GPU工作只能在相应服务器阶段授权下执行。
```
