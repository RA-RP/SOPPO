# 当前实验设计：Round4 DPO、SSPO 与 StaticPE

## 0. 版本与门禁

- Cycle：`cycle-20260901-01` / Round4
- 实验版本：`round4-exp-v1.0`
- 理论来源：`../theory/current_theory.md` `r4-theory-v1.0`（2026-09-01用户明确通过）
- 当前唯一活动阶段：`SERVER_EXECUTION`
- 实验状态：**APPROVED**
- 用户确认：2026-09-01，用户明确授权直接进入code阶段并允许在当前Round4边界内冻结剩余工程选择
- 代码交接：`round4-code-v1.0.1`，2026-09-01用户要求继续；首次4090执行暴露的`-e .`过滤缺陷已作无研究语义变化的脚本修复
- Round3：旧五方法 formal 已完成并作行政结项；拟议的 DPO-reward extension 未运行，不并入 Round4

本文件是Round4代码实现所依据的冻结实验合同；实现若改变以下研究语义，必须返回实验讨论并重新批准。

## 1. 研究问题

在同一 `Qwen/Qwen3-1.7B`、同一 UltraFeedback/UltraChat 数据比例和统一训练预算下比较：

1. 仅使用有标签偏好对的 DPO；
2. 同时利用有标签与无标签样本的 SSPO；
3. 同样使用双流数据、以条件编码偏好概率构造 PE 正则的 StaticPE。

核心问题是：StaticPE 是否能利用无标签数据获得超出 label-only DPO 的收益，并相对 SSPO 展现不同的稳定性或泛化行为。本轮是单模型、单轮次的探索性比较；若只跑一个 seed，不宣称统计显著性。

## 2. 方法臂

| 方法 | 训练数据 | 训练目标 | 冻结口径 |
| --- | --- | --- | --- |
| DPO | UltraFeedback labeled pairs | DPO-base | 无标签数据不进入训练 |
| SSPO | UltraFeedback labeled + UltraChat unlabeled | DPO-base labeled branch + SSPO unlabeled mechanism | 不采用SimPO-base |
| StaticPE | 同一 labeled + unlabeled 双流 | `(L_DPO + 0.1 L_PE) / 1.1` | 跨两卡同步的physical-microbatch PE |

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

`c_1/c_2`使用上述条件编码归一化形式，`epsilon=1e-8`、reward `beta=0.1`；每次forward跨两卡同步当前physical micro-batch的unlabeled统计，不跨8次gradient accumulation合并population。

## 3. 模型、数据与共同设置

- 模型：`Qwen/Qwen3-1.7B`，不是 `qwen/Qwen-3-1.7B`；执行时冻结 resolved revision、模型/tokenizer文件校验值和模板。
- 模板：Qwen3 native chat template，默认显式关闭 thinking；训练、merge 后生成和 AlpacaEval 使用同一对话语义。
- 数据：UltraFeedback 与 UltraChat 均按现有默认比例 `0.1`；StaticPE/SSPO共享相同 labeled IDs 和 unlabeled IDs，DPO只使用同一 labeled view。
- epoch：1。
- StaticPE：`lambda=0.1`。
- GPU：每个训练臂固定使用2卡；不允许因目标平台有8卡而让某一方法获得额外数据并行优势。
- eval physical batch：三方法均为每设备4；服务器显存检查不通过时，只允许调低并记录，不改变训练目标。
- Python/CUDA环境：CPython 3.12.x、PyTorch 2.5.1 CUDA 12.4 wheel，运行于`cuda12.4-cudnn-devel-ubuntu22.04-py312-ssh`既有镜像；最终resolved patch版本与wheel SHA写入环境manifest。
- 随机种子：三方法与候选生成统一seed42；本轮不据单seed结果宣称统计显著性。

数据必须在服务器冻结并输出去敏 manifest。任何 malformed、去重、split 或实际样本数变化都要先反映到 resolved config；不能根据预估 step 数反向删样本。

## 4. 正式 batch 与 step 口径

用户于2026-09-01明确选择 DPO 全局有效 `batch=16`。正式配置为：

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

StaticPE固定在每个两卡global physical microbatch的无标签子集上同步形成一组`c_1/c_2`，再累计8次梯度。它与在全局有效64行上只计算一次PE不等价；代码、指标和论文记录统一称为`physical-microbatch-PE`。

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

AlpacaEval固定`alpaca_eval==0.6.2`、`weighted_alpaca_eval_gpt4_turbo`、`get_length_controlled_winrate`；三方法与frozen base共享同一805条数据、生成配置、judge实际解析值和缓存规则。API key、逐样本输出与原始judge响应只留服务器/密钥系统，不进Git或镜像。MT-Bench退出Round4。

## 7. 4090-3 镜像与数据准备

4090-3不再运行Round4训练smoke，只承担获批exact commit的离线依赖、联网数据与模型暂存：

1. 拉取用户批准的exact clean commit；
2. 复用FusionOne已有`cuda12.4-cudnn-devel-ubuntu22.04-py312-ssh`镜像，不再把4090-3的Docker能力作为前置条件；
3. 依据冻结dependency lock构建CPython 3.12/Linux x86_64离线wheelhouse并记录逐文件SHA-256；
4. 在A100既有镜像内部创建新的Round4 venv，禁止直接复制旧Python 3.10 venv；
5. 下载冻结revision的数据集，生成source manifest、文件字节数和SHA-256；
6. 通过SSH/rsync类通道把wheelhouse、数据和冻结模型分别传到A100仓库外目标目录；
7. 在目标端重算并逐文件比对SHA-256，只有完全一致才标记transfer complete。

wheelhouse、数据和模型不能进入Git或镜像层；传输必须可续传且不能把半文件误认作完整文件。4090-3空间在执行前继续核验，生成物落在仓库外并分别使用manifest闭环。

## 8. FusionOne 正式执行候选

用户于2026-09-01确认亲自验证目标平台存在8张A100，并决定先创建/占用其中2张。本实验据此登记为用户验证的目标资源事实，但单卡显存、拓扑、容器映射、CPU/内存和挂载仍由 preflight 采集。

执行流程：FusionOne复用既有py312/CUDA12.4镜像 → 先创建2×A100容器 → 从4090-3传入离线wheelhouse、数据和模型 → 核验硬件/挂载/逐文件SHA → 在镜像内新建Round4 venv → 完成全部smoke → 获得单独formal授权后启动训练。

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

## 11. 批准与代码入口

本设计已冻结：SSPO使用DPO-base、StaticPE使用`physical-microbatch-PE`、frozen base加入AlpacaEval、seed42、MT-Bench退出、模型与数据从4090-3按独立manifest传至A100。用户于2026-09-01明确授权`round4-code-v1.0`提交并完成4090-3任务；当前处于`SERVER_EXECUTION`，代码入口为`../../code/CODE_OVERVIEW.md`。4090-3不训练，formal训练仍按本文件约定单独授权。
