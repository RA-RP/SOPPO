# 当前实验草案：Round4 DPO、SSPO 与 StaticPE

## 0. 版本与门禁

- Cycle：`cycle-20260901-01` / Round4
- 实验草案：`round4-exp-draft-v0.2`
- 理论来源：`../theory/current_theory.md` `r4-theory-v0.2`
- 当前唯一活动阶段：`THEORY_DISCUSSION`
- 理论状态：讨论中，尚未获得用户明确通过
- 实验状态：**锁定草案**；尚未进入 `EXP_DISCUSSION`，更未批准
- 代码、commit/push、服务器测试、镜像构建、训练与评价：全部锁定
- Round3：旧五方法 formal 已完成并作行政结项；拟议的 DPO-reward extension 未运行，不并入 Round4

本文件用于提前暴露 Round4 的实验选择，不表示这些选择已经定稿。理论通过后才逐项讨论并冻结本设计。

## 1. 研究问题

在同一 `Qwen/Qwen3-1.7B`、同一 UltraFeedback/UltraChat 数据比例和统一训练预算下比较：

1. 仅使用有标签偏好对的 DPO；
2. 同时利用有标签与无标签样本的 SSPO；
3. 同样使用双流数据、以条件编码偏好概率构造 PE 正则的 StaticPE。

核心问题是：StaticPE 是否能利用无标签数据获得超出 label-only DPO 的收益，并相对 SSPO 展现不同的稳定性或泛化行为。本轮是单模型、单轮次的探索性比较；若只跑一个 seed，不宣称统计显著性。

## 2. 方法臂

| 方法 | 训练数据 | 训练目标 | 当前待定项 |
| --- | --- | --- | --- |
| DPO | UltraFeedback labeled pairs | DPO-base | 无标签数据不进入训练 |
| SSPO | UltraFeedback labeled + UltraChat unlabeled | 与冻结实现一致的 labeled/unlabeled 联合目标 | 正式采用 DPO-base 还是 SimPO-base 尚待理论确认 |
| StaticPE | 同一 labeled + unlabeled 双流 | `(L_DPO + 0.1 L_PE) / 1.1` | PE 是 physical microbatch 统计还是 optimizer-step 完整 population 统计 |

StaticPE 的无标签样本由两条固定候选构成：

- A：UltraChat 原始回答；
- B：冻结初始化 `Qwen/Qwen3-1.7B` 及生成配置后预生成一次的回答。

令 DPO-base reward 得到候选 A 胜出的概率 `p_i`，并令 `q_i=[p_i,1-p_i]`：

$$
c_1=\frac{\sum_i p_iq_i}{\sum_i p_i+\epsilon},\qquad
c_2=\frac{\sum_i(1-p_i)q_i}{\sum_i(1-p_i)+\epsilon},
$$

$$
L_{PE}=\frac12\left(\lVert c_1-[1,0]\rVert_1+\lVert c_2-[0,1]\rVert_1\right),
\qquad
L_{StaticPE}=\frac{L_{DPO}+0.1L_{PE}}{1.1}.
$$

`c_1/c_2`使用上述条件编码归一化形式；`epsilon`、reward beta、是否跨 gradient-accumulation 合并 population 必须在理论/实验批准前冻结。

## 3. 模型、数据与共同设置

- 模型：`Qwen/Qwen3-1.7B`，不是 `qwen/Qwen-3-1.7B`；执行时冻结 resolved revision、模型/tokenizer文件校验值和模板。
- 模板：Qwen3 native chat template，默认显式关闭 thinking；训练、merge 后生成和 AlpacaEval 使用同一对话语义。
- 数据：UltraFeedback 与 UltraChat 均按现有默认比例 `0.1`；StaticPE/SSPO共享相同 labeled IDs 和 unlabeled IDs，DPO只使用同一 labeled view。
- epoch：1。
- StaticPE：`lambda=0.1`。
- GPU：每个训练臂固定使用2卡；不允许因目标平台有8卡而让某一方法获得额外数据并行优势。
- eval physical batch：三方法候选均为每设备4；服务器显存检查不通过时，只允许调低并记录，不改变训练目标。

数据必须在服务器冻结并输出去敏 manifest。任何 malformed、去重、split 或实际样本数变化都要先反映到 resolved config；不能根据预估 step 数反向删样本。

## 4. 正式 batch 与 step 口径

用户于2026-09-01明确选择 DPO 全局有效 `batch=16`。候选正式配置为：

| 方法 | per-device train | GPU | gradient accumulation | 全局有效 batch | per-device eval | epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DPO | 1 | 2 | 8 | **16** | 4 | 1 |
| SSPO | 4 | 2 | 8 | **64** | 4 | 1 |
| StaticPE | 4 | 2 | 8 | **64** | 4 | 1 |

这里的全局有效 batch 是 `per_device_train_batch_size × GPU数 × gradient_accumulation_steps`。DPO 的16不是 minibatch；其物理全局 minibatch 是2。

按当前默认10%数据与 `val_size=0.1` 的粗略规模，DPO预计约344个 optimizer steps，SSPO/StaticPE约379步，差距约10%。这只是设计估算；正式 step 只能由冻结后的有效 train rows、sampler和丢弃规则计算。原则是：

- 三方法都完整训练1 epoch；
- 不重复或截断数据来伪造完全相同 step；
- 报告 optimizer steps、样本数、response token 数和 wall time，不能只以 step 数宣称算力完全匹配。

## 5. two-stream 与损失统计

SSPO/StaticPE使用 two-stream sampler，使一个完整 epoch 内总体 labeled:unlabeled 数量与冻结数据量一致；同时尽量让每个 physical batch 包含两类数据。若某个 physical batch 因整数和分布式切分不含某一类，必须明确定义并测试该分支。

StaticPE 当前 legacy 原型是在每个两卡 physical microbatch 的无标签子集上分别形成 `c_1/c_2`，再累计8次梯度。这与在全局有效64行上只计算一次 PE 不等价。正式设计必须在以下两者中选择：

1. `microbatch-PE`：接受当前实现语义，简单且显存较省；
2. `optimizer-population-PE`：跨8次累积收集完整无标签 population 后精确计算一次，语义更贴近“所有无标签数据”，但实现和显存更复杂。

在该项冻结前，不得将现有原型称作正式 StaticPE 数学实现。

## 6. 训练与评价记录

训练期至少记录：

- 三方法共同：总 loss、learning rate、grad norm、epoch、optimizer step、训练/验证 loss；
- DPO：`dpo_loss`、chosen/rejected reward、reward margin、accuracy；
- SSPO：labeled loss、unlabeled loss、混合权重/阈值/归一化状态及各自有效样本数；
- StaticPE：`dpo_loss`、`pe_loss`、归一化后的 joint loss、`p`与`c_1/c_2`摘要、labeled/unlabeled数量；
- 所有指标只使用 detach 后标量用于展示，不改变梯度聚合。

评价分为三层：

1. 训练内 eval loss：用于监测训练，不等价于最终能力评价；
2. merge/reload 检查：LoRA merge 后离线重载，确认生成和权重校验值可复现；
3. AlpacaEval 2.0：完整正式集为805条；三方法加冻结 base 使用同一 generation config、同一 evaluator、同一 judge 和同一缓存规则，报告 win rate 与 length-controlled win rate（LC）。

正式 AlpacaEval 版本和 judge 仍待冻结。当前原型固定 `alpaca_eval==0.6.2`，不能在不同方法间漂移版本。API key、逐样本输出与原始 judge 响应只留服务器/密钥系统，不进Git或镜像。

## 7. 4090-3 镜像与数据准备

4090-3不再运行Round4训练smoke，只承担获批exact commit的镜像准备和联网数据暂存：

1. 拉取用户批准的exact clean commit；
2. 实时核验Docker/BuildKit权限、独立scratch空间和内部镜像仓库连通性；
3. 依据冻结dependency lock构建无凭据镜像，使用非`latest` tag并记录digest；
4. 配置可被FusionOne正常创建的安全启动入口：只初始化环境/保持容器可进入，不自动启动训练；
5. 下载冻结revision的数据集，生成source manifest、文件字节数和SHA-256；
6. 通过SSH/rsync类通道把数据直接传到A100仓库外的目标数据目录；
7. 在目标端重算并逐文件比对SHA-256，只有完全一致才标记transfer complete。

数据不能进入Git或镜像层，传输中断必须可续传且不能把半文件误认作完整数据。4090-3现有历史快照显示当前SSH落点缺少容器构建工具且数据盘空间紧张；若实时复核仍如此，该落点无法完成第2–3步，必须由用户先补充构建权限/scratch或改用获授权制作面。不能删除旧实验产物来强行腾空间。

## 8. FusionOne 正式执行候选

用户于2026-09-01确认亲自验证目标平台存在8张A100，并决定先创建/占用其中2张。本实验据此登记为用户验证的目标资源事实，但单卡显存、拓扑、容器映射、CPU/内存和挂载仍由 preflight 采集。

镜像流程候选：4090-3构建无凭据镜像 → 使用非`latest` tag并记录digest → 内部仓库录入 → 先创建2×A100容器 → 核验硬件/挂载/数据SHA → 在A100上完成全部smoke → 获得单独formal授权后启动训练。

三方法共享这同一组2张A100并顺序执行，不再候选6卡并发。每个方法先运行独立smoke：恰好2个optimizer steps、至少一次eval、adapter保存、LoRA merge、新进程离线reload、少量固定Alpaca指令生成和一次judge API调用；SSPO/StaticPE fixture必须同时覆盖labeled与unlabeled分支。smoke使用独立目录，只验证链路，不产生论文结果，也不覆盖formal batch合同。训练任务需要的平台占位算法和容器设置只属于执行面，不在Git保存内部接入信息。

## 9. 预先判断与报告边界

主要结果是相对冻结 base 的 AlpacaEval 2.0 LC win rate；普通 win rate、训练/验证 loss、reward统计、长度和成本为辅助结果。

探索性判断：

- StaticPE若在共同评价上稳定高于DPO，且PE没有坍缩/non-finite，支持“无标签条件编码提供额外训练信号”的候选解释；
- StaticPE若与DPO相当或更差，需结合PE梯度、candidate质量和population定义判断，不能直接断言理论无效；
- SSPO与StaticPE比较只有在SSPO base、数据顺序、有效batch和评价合同冻结后才可归因于目标函数差异；
- 单seed的小差异只报告为观察，不表述为稳定优势。

`C_gamma`如保留，只作为观测量，不进入损失、选点或主结论，除非后续理论另行批准。

## 10. 产物与边界

服务器保留：数据/模型、checkpoint、adapter、merged model、逐样本生成、Alpaca逐样本输出、judge原始响应、完整日志、镜像构建日志和内部证据。

本地只回传：resolved config、无样本级聚合Markdown/JSON/CSV、汇总图表、环境摘要、commit/image/model/data校验值和去敏远程证据索引。

不得把账号、密码、token、内部地址、原始数据、模型或逐样本回答写进Git、镜像层或聊天回答。

## 11. 理论批准前未决项

1. SSPO正式使用 DPO-base 还是 SimPO-base；当前建议选择DPO-base，以缩小与DPO/StaticPE的非目标差异。
2. StaticPE选择 `microbatch-PE` 还是精确 `optimizer-population-PE`。
3. AlpacaEval是否纳入冻结 base，并冻结具体版本、judge与生成参数。
4. Round4只跑seed42，还是为三方法增加多个seed。
5. MT-Bench是否纳入；当前建议本轮先只做AlpacaEval 2.0。
6. A100侧Qwen3模型的来源：复用已核验只读挂载，或由4090-3按独立model manifest传输；不得放进镜像层。

以上项目属于理论/实验选择。用户明确通过 `r4-theory-v0.2` 前，本草案不能解锁代码修改或任何服务器操作。
