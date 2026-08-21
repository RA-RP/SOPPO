# 当前实验设计：SOPPO v0.6 SSPO-aligned 30k MVP

## 0. 状态与门禁

- Cycle ID：`cycle-20260818-01`
- Experiment ID：`exp-20260819-01-mvp`
- 理论依据：`../theory/current_theory.md` v0.2，已于 2026-08-19 明确通过
- 实验版本：v0.6 SSPO-aligned 30k MVP
- 实验批准：用户于 2026-08-21 逐项确认 SSPO 对齐、DPO headroom、LoRA、两种 PE 权重和八条最终轨迹，并明确要求开始编码
- 当前唯一活动阶段：`CODE_IMPLEMENTATION`
- 下一入口：`../../code/CODE_OVERVIEW.md`
- 服务器执行：`LOCKED`；代码静态交接完成后仍需用户另行明确授权

v0.6 替代 v0.5 的 SFT/Pseudo/DPO+PE 方案。30k 数据及隔离合同不变，训练目标、超参、LoRA、batch、checkpoint 和任务图以本文件为准。

## 1. 问题与最终对照

核心问题：在只有 10% 偏好标签时，population-level PE 能否比与 SSPO 论文设置对齐的 instance-level hard pseudo-label 更有效，并向 DPO-100 oracle 靠近。

八条最终轨迹为：

1. `DPO-10`；
2. `DPO-100`；
3. `SSPO-hard-exp`；
4. `SOPPO-PE-exp`；
5. `SOPPO-PE-static-0.1`；
6. `SOPPO-PE-static-0.3`；
7. `SOPPO-PE-static-0.5`；
8. `SOPPO-PE-static-1.0`。

不再训练 SFT、hard-static、linear scheduler 或旧的 DPO-style pseudo-target。`SSPO-hard-exp` 与 `SOPPO-PE-exp` 共享 labeled loss、优化超参、batch 和同一 `gamma_t`，只替换 unlabeled objective；这是核心受控比较。四条 static PE 用于判断固定混合是否优于动态 curriculum。

## 2. 数据合同

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
- `transformers==4.51.3`，`trust_remote_code=false`，`enable_thinking=false`。
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

所有正式训练使用单节点 2×A800、DDP LoRA、seed 42：

| 路径 | epoch | lr | global batch | 组成 |
| --- | ---: | ---: | ---: | --- |
| DPO-10 | 1 | 1e-6 | 64 | per-device 4 × accumulation 8 × 2 |
| DPO-100 | 1 | 1e-6 | 64 | 同上 |
| SSPO-hard-exp | 2 | 1e-5 | 64 | 每 step 8 labeled pairs + 56 unlabeled pairs |
| PE-exp/static | 2 | 1e-5 | 64 | 与 hard 完全相同 |

8/56 的实现为每 rank 每 step 8 个 microstep：unlabeled size pattern `[3,4,3,4,3,4,3,4]`，在 microstep `[0,2,4,6]` 另取 1 个 labeled pair；两 rank 合计 8 labeled + 56 unlabeled。optimizer 为 AdamW，weight decay 0，cosine schedule，warmup ratio 0.1，max grad norm 1.0。

显存执行合同：上述 logical batch、optimizer step 和损失归一化全部不变；2048 长度的梯度前向/反向按每 rank 1 pair 的 backward subbatch 顺序累积，只有完整 logical optimizer batch 的最后一次 backward 触发 DDP 同步。PE 第一遍仍在完整 56-pair global unlabeled population 上求精确系数，第二遍只是用同一组系数分块回传，因此不构成 PE microbatch 近似。

## 6. checkpoint 与选择

- adapter checkpoint 不自动删除，不保存 optimizer/scheduler state，因此不宣称 bit-exact resume。
- DPO 每 20 optimizer step及 final 保存；DPO-10 因而至少有三个轨迹点。
- SSPO/PE 每 40 step及 final 保存；`C_epsilon` 每条轨迹最多均匀选 10 点。
- best checkpoint 只按 validation accuracy、再按更低 Brier 选择。
- headroom 使用共同的 margin-free mean-response-logp A/B score，要求 `DPO-10` 选中 checkpoint 的 validation accuracy 至少比同一路径训练前、显式禁用 adapter 的冻结 Qwen3 base 高 `0.05`；前后还必须核对同一 validation 样本数和 score type。不再用 SFT。DPO reference delta 在初始化时恒为零，不能用于这个 base-before/after 比较。
- static lambda 只在四条 static arm 之间按 validation accuracy、Brier、较小 lambda 的确定性顺序选择；test 不参与选择。

## 7. 强 smoke 与一次提交 DAG

strong smoke 在账户获批的 `gpu` partition 请求 2×A800、90 分钟，并从各 split 选择字符长度最大的真实样本，以 bf16/2048 运行。它覆盖：

- 冻结 Qwen3 manifest、离线加载和 chat mask；
- LoRA trainable/base-frozen 合同；
- 两 rank DDP 和真实前反向；
- DPO-10、DPO-100、SSPO-hard-exp、PE-exp、PE-static 各一步；
- KDE/EMA/threshold、exact-global PE、finite loss/gradient；
- adapter 保存后重新加载并再训练一步。

服务器执行时，`submit_all.sh` 在账户唯一获批的 `gpu` partition 直接提交完整 `afterok` DAG；集群不允许普通用户 `sbatch --hold`，因此任一中途提交失败都会触发已提交 job 的自动取消：

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

所有依赖为 `afterok`；排队和运行不依赖 SSH 会话。formal job 必须恰好识别请求数量的 A800。

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

模型/数据/cache manifest 不符、Git checkout 不干净、卡数/型号不符、标签泄漏、batch 不是 64 或 joint 不是 8/56、非有限 loss/gradient、PE population 不完整、headroom 失败、adapter/best checkpoint 缺失、`C_epsilon` cell 不完整或导出含样本字段，均非零退出并阻断下游。

- 2026-08-21：用户明确删除 SFT headroom，并进一步纠正：headroom 应检验 DPO-10 是否确实强于训练前 base；DPO-100 只作为 oracle。
- 2026-08-21：用户明确 `max_seq_len=2048`，并要求解释和冻结 LoRA。
- 2026-08-21：用户要求训练损失、epoch、lr 与 SSPO 论文对齐。
- 2026-08-21：用户明确核心比较同时包含 paper exponential `gamma_t` 与 normalized fixed lambda PE。
- 2026-08-21：用户确认上述内容无问题并要求开始正式编码。

这些批准只解锁 v0.6 `CODE_IMPLEMENTATION`；不自动授权 SFTP 上传、模型/数据操作或 Slurm 提交。
