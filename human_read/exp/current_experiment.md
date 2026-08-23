# 当前实验设计：SOPPO v0.6 SSPO-aligned 30k MVP

## 0. 状态与门禁

- Cycle ID：`cycle-20260818-01`
- Experiment ID：`exp-20260819-01-mvp`
- 理论依据：`../theory/current_theory.md` v0.2，已于 2026-08-19 明确通过
- 实验版本：v0.6 SSPO-aligned 30k MVP
- 实验批准：用户于 2026-08-21 逐项确认 SSPO 对齐、DPO headroom、LoRA、两种 PE 权重和八条最终轨迹，并明确要求开始编码
- 当前唯一活动阶段：`CODE_IMPLEMENTATION`
- 代码交接：第一轮冻结基线保持不变；第二轮 3×4090 TP=2 + 单卡 vLLM 适配尚未提交、尚待用户审阅
- 当前入口：第二轮草案为 `../../code/scripts/round2/EXECUTION_GUIDE.md`；第一轮 cluster/standalone 入口保持冻结
- 服务器执行：第二轮 `LOCKED`。本次实现获得代码修改授权，但尚未获得新代码交接确认；不得把第一轮 2026-08-21 的执行授权自动沿用到第二轮 TP 实现

v0.6 替代 v0.5 的 SFT/Pseudo/DPO+PE 方案。第一轮静态 PE 已完成并冻结；本文件当前只约束后续第二轮 rollout 相关扩展。30k 数据及隔离合同不变，训练目标、超参、LoRA、batch、checkpoint 和任务图以本文件为准。

## 1. 问题与最终对照

核心问题：第一轮 MVP 已经完成并冻结 DPO-10、DPO-100、静态 PE 与已有伪标签/SSPO 类对照后，第二轮只新增 rollout 相关 PE 实验：SFT+rollout 是否优于 rollout-only，并相对第一轮冻结基线提供增量。

第二轮新增轨迹只有两条：

1. `SOPPO-PE-sft-rollout-exp`；
2. `SOPPO-PE-rollout-only-exp`。

DPO-10、DPO-100、静态 PE 与已有伪标签/SSPO 类对照均属于第一轮 MVP 冻结结果，只读引用，不在第二轮重跑。第二轮只比较动态 rollout 相关主线与其消融；`SOPPO-PE-sft-rollout-exp` 与 `SOPPO-PE-rollout-only-exp` 共享 labeled loss、优化超参、batch 和同一 `gamma_t`，只替换 unlabeled objective。

## 2. 数据合同

第一轮 MVP 的结果目录与第二轮 rollout 扩展的结果目录必须分离存储，不得互相覆盖；第二轮不得重跑或覆盖 DPO-10、DPO-100、静态 PE 与已有伪标签/SSPO 类对照。后续最终合并只读取两边导出的聚合结果，不回写原始实验产物。

继续复用已经冻结的 `<SERVER_BASE>/data/ultrafeedback/mvp-v0.5-30k/`，不复制或覆盖数据：

| split | 数量 | 占总量 | 用途 |
| --- | ---: | ---: | --- |
| labeled train | 2,700 | 9% | DPO-10、SSPO/PE labeled risk |
| labeled validation | 300 | 1% | checkpoint、lambda 与 headroom 选择 |
| unlabeled train | 24,000 pairs | 80% | SSPO/PE；入口看不到 label |
| independent test | 3,000 | 10% | 仅独立 evaluator 可读私有 label |

`DPO-100` 使用隔离生成的 26,700 条 oracle train（2,700 + 24,000），不使用 validation/test 梯度。公共 unlabeled/test 中 A/B 已随机换位；SSPO-hard 将每个 pair 的 A、B 当作两个独立 unpaired response，二者都可能被阈值判为 winning 或 losing，绝不读取隐藏方向。

## 3. 模型、LoRA 与序列

- 冻结 base：ModelScope `Qwen/Qwen3-4B`，服务器路径 `<SERVER_BASE>/models/Qwen3-4B`。
- 第一轮冻结环境使用 `transformers==4.51.3`；所有路径均保持 `trust_remote_code=false`、`enable_thinking=false`。第二轮 TP 专用依赖版本见 §7.1，不覆盖第一轮环境。
- 正式精度 bf16、SDPA、gradient checkpointing、`max_seq_len=2048`、response-only mask。
- 所有训练统一标准 LoRA；不使用 QLoRA，不更新 base 参数。
- LoRA：`r=8`、`alpha=16`、`dropout=0`、`bias=none`，目标为 `q/k/v/o/gate/up/down_proj`。

`alpha/r=2` 是 LoRA update 的缩放；它不改变 adapter rank。所有 checkpoint 保存为 PEFT adapter，并绑定 base 路径、模型 manifest 与 resolved config。adapter 全部保留；评价和 `C_epsilon` 在内存中加载/合并，不批量保存 full merged model。

## 4. 损失函数

### 4.1 DPO-10 与 DPO-100

使用标准 reference-based DPO，reference 为冻结初始 Qwen3，response log-prob 使用 token sum：

```text
Delta_DPO = beta_DPO [(log pi_a - log pi_ref,a) - (log pi_b - log pi_ref,b)]
L_DPO = -log sigmoid(direction(label) * Delta_DPO)
beta_DPO = 0.1
```

### 4.2 SSPO/PE 的 labeled risk

与 SSPO 论文的 SimPO 设置一致，reward 使用 response-token mean log-prob：

```text
r_theta(x,y) = beta_S * mean_response_logp_theta(y|x)
L_L = -log sigmoid(r_w - r_l - margin)
beta_S = 10
margin = 2
```

### 4.3 SSPO-hard-exp

每个 optimizer step 在完整全局 minibatch 上：

1. 对 labeled winning/losing 和 unlabeled A/B response 计算 reward；
2. 用全体 response reward 更新 mean/std EMA，`m=0.95`；
3. 用 normalized labeled winning/losing reward 的 Gaussian KDE 最小化 Bayes risk；prior 固定 `0.5`；
4. threshold 也做 `m=0.95` EMA；
5. unlabeled 单 response 高于 threshold 为 pseudo-winning，否则 pseudo-losing；
6. 使用 threshold-centered logistic BCE，并按 prior 加权。

论文明确 Gaussian KDE 和每 step 200 个搜索点，但没有给 bandwidth；v0.6 将各类分布的 Scott rule 冻结为复现决定，不把它冒充论文原文超参。

### 4.4 SOPPO-PE

对每个无标签 pair：

```text
p_i = sigmoid(beta_S * (mean_logp_a - mean_logp_b))
q_i = [p_i, 1-p_i]
e_hat+ = sum_i p_i q_i / (sum_i p_i + epsilon)
e_hat- = sum_i (1-p_i) q_i / (sum_i (1-p_i) + epsilon)
L_PE = 0.5 (||e_hat+ - [1,0]||_1 + ||e_hat- - [0,1]||_1)
epsilon = 1e-8
```

`p_i` 不使用 SimPO margin，否则会因 A/B 位置引入偏置。PE responsibility 不 detach；denominator 默认也不 detach。实现必须在完整 56-pair global unlabeled population 上两次前向，第一次跨 rank 得到精确 `dL_PE/dp_i`，第二次回传，不允许 microbatch 近似。

### 4.5 两种混合权重

动态受控比较使用论文 scheduler：

```text
gamma_t = max(gamma_min, gamma0 * exp(-decay * t))
L_exp = gamma_t L_L + (1-gamma_t) L_U
gamma0 = 1
gamma_min = 2700 / (2700+24000) = 0.1011235955
decay = 0.01
```

其中 `L_U` 在 hard arm 是 pseudo-labeled risk，在 PE arm 是 `L_PE`。固定 PE 使用归一化凸组合：

```text
L_static = 1/(1+lambda) L_L + lambda/(1+lambda) L_PE
lambda in {0.1, 0.3, 0.5, 1.0}
```

禁止把 static lambda 直接写成未归一化的 `L_L + lambda L_PE`，也禁止用旧 linear/exp-warmup lambda 替代论文 `gamma_t`。

## 5. 优化与 batch

所有正式训练使用单节点1/2/4张约80GB GPU、LoRA、seed 42；多卡使用DDP，单卡不包DDP。旧集群实证卡为A800；standalone保持同一设备档位与显存下限，并单独记录实际SKU。卡数只是执行档位，不是实验变量；一次实验的八条轨迹必须全部使用同一个档位：

| 路径 | epoch | lr | global batch | 组成 |
| --- | ---: | ---: | ---: | --- |
| DPO-10 | 1 | 1e-6 | 64 | per-device 4 × accumulation 16/8/4 × devices 1/2/4 |
| DPO-100 | 1 | 1e-6 | 64 | 同上 |
| SSPO-hard-exp | 2 | 1e-5 | 64 | 每 step 8 labeled pairs + 56 unlabeled pairs |
| PE-exp/static | 2 | 1e-5 | 64 | 与 hard 完全相同 |

1/2/4卡的梯度累积分别为16/8/4。joint 的每rank unlabeled pattern分别为`[3,4]×8`、`[3,4]×4`、`[3,4]×2`，在每个偶数microstep另取1个labeled pair；三种档位都合计8 labeled +56 unlabeled。三卡不能精确保持该合同，因此不支持。optimizer为AdamW，weight decay 0，cosine schedule，warmup ratio 0.1，max grad norm 1.0。

显存执行合同：上述 logical batch、optimizer step 和损失归一化全部不变；2048 长度的梯度前向/反向按每 rank 最多 2 pair 的 backward subbatch 顺序累积，只有完整 logical optimizer batch 的最后一次 backward 触发 DDP 同步。DPO 的 logical 4 拆为 `2+2`，joint 的 logical `3/4` 拆为 `2+1`/`2+2`。PE 第一遍仍在完整 56-pair global unlabeled population 上求精确系数，第二遍只是用同一组系数分块回传，因此不构成 PE microbatch 近似。

## 6. checkpoint 与选择

- adapter checkpoint 不自动删除，不保存 optimizer/scheduler state，因此不宣称 bit-exact resume。
- DPO 每 20 optimizer step及 final 保存；DPO-10 因而至少有三个轨迹点。
- SSPO/PE 每 40 step及 final 保存；`C_epsilon` 每条轨迹最多均匀选 10 点。
- best checkpoint 只按 validation accuracy、再按更低 Brier 选择。
- headroom 使用共同的 margin-free mean-response-logp A/B score，要求 `DPO-10` 选中 checkpoint 的 validation accuracy 至少比同一路径训练前、显式禁用 adapter 的冻结 Qwen3 base 高 `0.05`；前后还必须核对同一 validation 样本数和 score type。不再用 SFT。DPO reference delta 在初始化时恒为零，不能用于这个 base-before/after 比较。
- static lambda 只在四条 static arm 之间按 validation accuracy、Brier、较小 lambda 的确定性顺序选择；test 不参与选择。

## 7. 强 smoke 与一次启动的任务图

strong smoke 在账户获批的 `gpu` partition 请求与正式训练相同档位的1/2/4×A800、90分钟，并从各 split 选择字符长度最大的真实样本，以bf16/2048运行。它覆盖：

- 冻结 Qwen3 manifest、离线加载和 chat mask；
- LoRA trainable/base-frozen 合同；
- 所选1/2/4卡路径和真实前反向；
- DPO-10、DPO-100、SSPO-hard-exp、PE-exp、PE-static 各一步；
- KDE/EMA/threshold、exact-global PE、finite loss/gradient；
- adapter 保存后重新加载并再训练一步。

共享集群保留 `submit_all.sh`/Slurm `afterok` DAG 作为旧执行适配。当前独占服务器不使用调度器，由 `standalone/start_pipeline.sh` 启动后台顺序控制器；同一逻辑任务图中每一步成功才运行下一步，任一失败都会立即停止：

```text
CPU tests
  -> strong smoke
  -> oracle/reference cache
  -> DPO-10 + DPO-100 final trajectories
  -> DPO-10 vs frozen-base headroom gate
  -> four PE-static final trajectories
  -> validation-only static-lambda selection
  -> SSPO-hard-exp + SOPPO-PE-exp final trajectories
  -> C_epsilon prepare/8 trajectories/derive
  -> independent test evaluation of all 8
  -> aggregate + whitelist export
```

standalone运行不依赖SSH会话，也不产生排队任务；默认两张GPU，也可由GPU ID列表选择1张或4张，其中一张用于串行后处理，并要求每张卡至少79000 MiB。实际GPU SKU、显存、torch CUDA版本和所选卡数必须写入registry邻接的hardware CSV；如果新服务器并非原计划的A800，结果交接必须把它列为执行环境差异。训练目标、global batch、8/56和八条轨迹不因平台或卡数档位变化。

## 7.1 第二轮 3×4090 执行剖面

用户已确认第二轮使用 GPU0–1 对 Qwen3-4B 做真正 tensor parallel、GPU2 独立运行 vLLM rollout；不得把两张训练卡实现成 DDP 两份模型，也不得用 QLoRA、缩短2048序列或改变8+56来规避显存。

- 训练：`TP=2, PP=1, DP=1`，bf16、SDPA、LoRA r8/alpha16/all projections；每次物理 forward/backward 为1个 preference pair，但一次 optimizer step 仍精确覆盖8 labeled +56 dynamic pairs。
- 环境：训练侧因 PEFT 官方 TP-LoRA 接口要求而隔离使用 Transformers 5.4+/PEFT 0.19+；GPU2 的 vLLM 0.9.2 使用另一隔离环境。该版本适配不改变 Qwen3 权重、tokenization、loss 或冻结超参，第一轮 `envs/youc` 不被覆盖。
- 在线性：每个 optimizer step 都先发布不可变 current-policy adapter，再由 GPU2 生成该步候选；不是预先用base批量生成后重复使用。
- SFT+rollout：每个 prompt 使用一个固定 SFT response 与一个 current-policy rollout。
- rollout-only：每个 prompt 必须从该方法自己的 current policy 独立采样两条候选；不能把同一条复制到A/B两侧，也不能在两条已经分叉的训练轨迹之间共享实际输出。
- 两条正式方法顺序运行并使用独立输出目录；第一轮结果只读引用。

下面三项尚未预注册，因而第二轮 formal execution 仍锁定：

1. 与24,000个 frozen unlabeled prompt 一一对应的 label-free 单回复 SFT corpus 来源；
2. rollout temperature；
3. rollout top-p。

代码只实现严格 schema 和必填入口，不从隐藏 preference label 选择 chosen response，也不自行填经验默认值。用户确认上述三项并完成本次代码交接后，才可形成 clean commit 并在3×4090服务器依次执行 preflight、server tests 和 production-path strong smoke。

## 8. 评价与解释边界

独立 test 输出 accuracy、Brier、ECE 与 confidence distribution。DPO 用 reference delta 评分；SSPO/PE 用 margin-free `beta_S(mean_logp_a-mean_logp_b)` 评分。两种 score 只在各自模型上形成 A/B 概率，最终指标可并列表达，但必须保留 `score_type`。

重点预注册差值：

- `PE-exp - DPO-10`；
- `PE-exp - SSPO-hard-exp`（核心动态受控比较）；
- validation-selected static PE 与 DPO-10 / SSPO-hard-exp 的差值；
- `DPO-10 - frozen base` 的共同口径 validation headroom；
- `DPO-100 - DPO-10` 仅作为独立 test oracle gap，不称为 headroom。

`C_epsilon` 继续观测 Qwen3 L18 的 equal5/equal7 轨迹。单种子结果只能表述为探索性趋势；不得给出显著性结论，也不得仅凭 `C_epsilon` 宣称因果机制。

## 9. 失败关闭与批准记录

模型/数据/cache manifest 不符、Git checkout 不干净、卡数不是1/2/4或与registry/smoke档位不符、型号不符、标签泄漏、batch不是64或joint不是8/56、非有限loss/gradient、PE population不完整、headroom失败、adapter/best checkpoint缺失、`C_epsilon` cell不完整或导出含样本字段，均非零退出并阻断下游。

- 2026-08-21：用户明确删除 SFT headroom，并进一步纠正：headroom 应检验 DPO-10 是否确实强于训练前 base；DPO-100 只作为 oracle。
- 2026-08-21：用户明确 `max_seq_len=2048`，并要求解释和冻结 LoRA。
- 2026-08-21：用户要求训练损失、epoch、lr 与 SSPO 论文对齐。
- 2026-08-21：用户明确核心比较同时包含 paper exponential `gamma_t` 与 normalized fixed lambda PE。
- 2026-08-21：用户确认上述内容无问题并要求开始正式编码。
- 2026-08-22：服务器执行阶段中，用户明确要求在不改变训练目标与全局batch的前提下支持1/2/4卡，并可通过少量shell配置快速切换；该项作为等价执行适配实现，不新增方法臂或科学比较。

这些批准只解锁 v0.6 `CODE_IMPLEMENTATION`；不自动授权 SFTP 上传、模型/数据操作或 Slurm 提交。
