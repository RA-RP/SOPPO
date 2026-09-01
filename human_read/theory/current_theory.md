# 当前理论：Round4 StaticPE、SSPO与label-only DPO

## 0. 版本、状态与边界

- Cycle：`cycle-20260901-01` / Round4
- 理论版本：`r4-theory-v1.0`
- 当前唯一活动阶段：`SERVER_EXECUTION`
- 状态：`APPROVED`
- 用户确认：2026-09-01，用户明确授权“直接到code阶段，想改什么改什么”；该授权按当前Round4理论与实验对象记录为明确通过，并授权在不改变研究问题的前提下冻结剩余工程选择
- Round3边界：五方法formal结果只读保留；两个DPO-reward extension不再执行，不能用“未运行”推断方法效果
- 代码交接：`round4-code-v1.0`于2026-09-01获用户明确批准提交并完成4090-3任务；4090-3只负责离线依赖、冻结数据/模型与manifest，不执行训练

## 1. 研究问题

Round4在同一Qwen3-1.7B初始化和同一10%双源数据合同下比较三种方法：

1. `DPO-label-only`：只使用有标签UltraFeedback preference pairs；
2. `SSPO`：在相同labeled来源上加入UltraChat single-response风险；
3. `StaticPE`：在相同labeled来源上加入由固定candidate pair构造的conditional-encoding PE。

核心问题是：当labeled DPO信号、模型、数据来源、epoch和通用优化超参尽量对齐时，StaticPE对固定无标签候选群体施加的结构约束，能否比不使用无标签数据的DPO和SSPO的single-response伪风险得到更好的训练行为与生成式指令遵循结果。

Round4的首要比较不是“谁使用更多token更高效”，因为DPO没有UltraChat分支，三方法训练计算量天然不同。公平性主轴是共享labeled IDs、共同初始化、共同labeled DPO定义和共同最终评价；数据暴露量、optimizer steps和token compute必须分别披露。

## 2. 共同模型与数据

### 2.1 模型

- 模型：`Qwen/Qwen3-1.7B`
- 初始化revision：`b9352fbb8ce704292730cf54b3b1dceb2a808738`
- 模板：native Qwen3 non-thinking；训练、候选B生成、merge后推理和Alpaca生成保持同一模板语义
- 训练：非量化LoRA，候选共同值为r8、all linear、BF16、lr `1e-5`、cosine、warmup `0.1`、epoch 1

固定revision的目的不是宣称该commit具有额外质量，而是确保StaticPE候选B生成和三方法训练从完全相同的模型/tokenizer快照开始。

### 2.2 数据

- labeled：`HuggingFaceH4/ultrafeedback_binarized`的10% preference数据；三个方法共享同一行ID、chosen/rejected、split与顺序合同
- unlabeled：`HuggingFaceH4/ultrachat_200k`的10% single-response数据；SSPO和StaticPE共享同一行ID，DPO完全不读取该view
- validation：三方法共享同一held-out labeled view，不能从各自训练混合池临时切出不同评价样本

“数据保持一致”定义为共享来源、冻结revision和共享labeled IDs；不意味着DPO必须读取无标签行。否则DPO会被静默改造成另一种半监督方法。

## 3. 三种目标函数

### 3.1 共同DPO-base

对response tokens上的总log-prob定义

$$
s_\theta(x,y)=\sum_{t\in\mathrm{response}}\log\pi_\theta(y_t\mid x,y_{<t}),
$$

冻结初始化reference为$\pi_{\mathrm{ref}}$，则一条labeled pair的logit为

$$
d_i=\beta[(s_\theta^+-s_{\mathrm{ref}}^+)-(s_\theta^--s_{\mathrm{ref}}^-)],
\qquad \beta=0.1,
$$

$$
L_{\mathrm{DPO}}=\operatorname{mean}_i[-\log\sigma(d_i)].
$$

DPO-label-only只优化该目标。StaticPE的labeled branch和SSPO的labeled branch也严格使用该DPO-base目标，使三方法主要只差无标签机制；Round4不采用作者默认SimPO-base。

### 3.2 SSPO

SSPO保留作者GitHub code-loss的主要无标签机制：对labeled winning/losing与UltraChat single response计算reward标准化，以labeled winning reward的最小值作为threshold，在prior-weighted的两侧风险之间选择unlabeled loss，并通过$\gamma_t$混合：

$$
L_{\mathrm{SSPO}}=\gamma_t L_{\mathrm{labeled}}+(1-\gamma_t)L_{\mathrm{unlabeled}}.
$$

$L_{\mathrm{labeled}}$固定采用DPO-base。running mean/variance、threshold、prior和$\gamma_t$属于SSPO内部机制，不与StaticPE的$\lambda$等同。

### 3.3 StaticPE

每条UltraChat记录固定两个candidate：

- candidate A：数据中的原始UltraChat assistant回答；
- candidate B：冻结初始化Qwen3-1.7B以non-thinking贪心配置预先生成一次的回答；
- A/B位置由独立稳定hash交换，训练期间不刷新candidate。

令

$$
r_A=\beta(s_\theta^A-s_{\mathrm{ref}}^A),\qquad
r_B=\beta(s_\theta^B-s_{\mathrm{ref}}^B),
$$

$$
p_i=\sigma(r_A-r_B),\qquad q_i=[p_i,1-p_i].
$$

对当前PE统计群体定义条件编码：

$$
c_1=\frac{\sum_i p_iq_i}{\sum_i p_i+\epsilon},\qquad
c_2=\frac{\sum_i(1-p_i)q_i}{\sum_i(1-p_i)+\epsilon},
$$

$$
L_{\mathrm{PE}}=\frac12\left(\lVert c_1-[1,0]\rVert_1+\lVert c_2-[0,1]\rVert_1\right),
\qquad \epsilon=10^{-8},
$$

$$
L_{\mathrm{StaticPE}}=\frac{L_{\mathrm{DPO}}+0.1L_{\mathrm{PE}}}{1.1}.
$$

PE不读取candidate的真实优劣标签；它鼓励模型产生内部一致、可分离的方向概率。初始化policy等于reference时所有$p_i=0.5$，首个非对称更新由labeled DPO分支提供。

### 3.4 PE统计群体的冻结语义

当前legacy原型在每个两卡全局physical micro-batch上可微汇总$p_i$并计算一组$c_1/c_2$，再对8个micro-batch累积梯度。formal设置下每个全局physical batch有8行，其中约6行unlabeled；这不等价于把effective batch64中的约50条unlabeled一次性合并后只计算一组$c_1/c_2$。

Round4固定采用`physical-microbatch-PE`：每次forward先跨当前两张DDP卡聚合该physical micro-batch中的unlabeled样本，形成一组全局$c_1/c_2$；一个optimizer step优化8个physical-microbatch PE目标的梯度累计。它明确不等同于在effective batch64上只形成一个PE目标，报告和代码不得混用两种口径。

## 4. Batch、step与公平性

用户已决定：

- SSPO/StaticPE：每设备4、2 GPU、梯度累积8，effective batch64行；
- DPO：每设备1、2 GPU、梯度累积8，effective batch16 pairs；
- eval：每设备4；
- 三方法：epoch 1。

较小DPO batch用于缩小因数据量较少造成的optimizer-step差距，同时保持两卡和梯度累积8一致。按预期数据量和`val_size=0.1`粗算，SSPO/StaticPE约379 steps、DPO约344 steps。Round4不重复DPO数据或截断联合数据来强行制造完全相同步数；正式报告必须给出冻结数据后的精确steps、response数和token数。

## 5. 可证伪预测

| 编号 | 预测 | 支持证据 | 削弱或反驳条件 |
| --- | --- | --- | --- |
| R4-P1 | label-only DPO应复现稳定的labeled preference学习 | train/eval DPO loss下降、held-out reward accuracy提高 | loss/accuracy不优于初始化且无工程错误 |
| R4-P2 | StaticPE若提供有效结构信号，PE loss应在不破坏DPO分支的情况下下降 | DPO与PE分项、$p$分布、$c_1/c_2$、held-out eval | PE下降但DPO/eval/Alpaca恶化，或$p$直接坍缩到单侧 |
| R4-P3 | SSPO的价值依赖labeled threshold形成可靠分离 | labeled/unlabeled loss、threshold、pseudo-side比例 | threshold不稳定且SSPO不优于DPO |
| R4-P4 | 若StaticPE优于SSPO，差异应表现为更稳定的无标签结构而非仅更长回答 | LC win rate、普通win rate、长度、PE/SSPO telemetry | 只提高普通win rate而LC不升，或完全由长度解释 |
| R4-P5 | 固定candidate B可复现但可能引入来源捷径 | A/B来源方向、交换后概率、candidate长度与质量聚合 | 模型只按A/B来源分离且与评价质量无关 |

单种子只支持探索性比较，不使用“显著优于”措辞。若三方法token compute差距较大，结果解释为既定训练预算下的整体方法比较，而不是样本效率因果结论。

## 6. 评价理论

### 6.1 训练内评价

三方法在共同labeled eval view上记录DPO-style eval loss/reward accuracy；训练目标分项只作机制诊断，不能把SSPO/PE内部loss直接横向比较为统一分数。

### 6.2 AlpacaEval 2.0

三种selected/final模型候选都在相同805条指令上生成一次回答，主要报告length-controlled win rate，同时报告普通win rate和平均输出长度。LC用于降低回答长度对judge偏好的混杂，但仍是自动裁判结果，不替代人类评价。

正式评价固定`alpaca_eval==0.6.2`、`weighted_alpaca_eval_gpt4_turbo`和`get_length_controlled_winrate`；数据revision、reference outputs、generation参数、judge实际解析值与API可用性进入执行manifest。三方法与frozen base完全共用该合同，不能训练后只升级其中一条结果。

MT-Bench退出Round4，不分配生成或judge预算。

## 7. $C_{\gamma}$角色

Round4主问题是三种训练目标与生成式表现，不把$C_{\gamma}$作为训练目标、checkpoint selector或成功标准。若后续加入，它只能作为三种selected模型的同层、同epsilon、同样本view观测量，并必须单独预算与服务器聚合；当前不因历史工具存在而自动运行。

## 8. 适用边界与风险

- StaticPE固定B把生成器质量固化为数据构造偏差；结果不能外推到在线刷新candidate。
- SSPO与StaticPE看到UltraChat而DPO不看，因此整体差异同时包含无标签机制和额外文本暴露。
- DPO effective batch16与联合方法64，梯度噪声和step数不完全一致；必须披露而不能称为严格compute matched。
- Alpaca自动judge受provider、model alias、prompt和reference变化影响；真实API smoke通过前不宣称可复现。
- merged model只用于推理/评价交付，训练证据仍绑定原adapter与base revision。
- 4090-3只承担镜像与数据中转准备，不再承担Round4训练smoke；其构建权限、scratch和网络连通性必须实时验证。
- 三方法smoke与formal都在同一批先占用的2张A100上顺序执行；smoke通过只说明工程闭环，不代表formal已获授权。
- 从4090-3经SSH传入A100的数据和冻结Qwen3模型必须落在Git仓库和镜像层之外，分别以source manifest、字节数和SHA-256闭环；模型不得进入镜像层。

## 9. 冻结决定

用户已决定：Round4三方法、Qwen3-1.7B、UltraFeedback/UltraChat各10%、DPO只用labeled、StaticPE `lambda=0.1`、epoch1、两卡、SSPO/StaticPE effective batch64、DPO effective batch16、eval每卡4、FusionOne 8×A100资源存在及AlpacaEval 2.0。执行顺序改为4090-3先准备镜像和数据并通过SSH传输，优先创建/占用2张A100，随后所有训练smoke和formal均在这2张卡上顺序进行；不再在4090-3执行训练smoke。

其余选择冻结为：SSPO labeled branch使用DPO-base；StaticPE使用跨两卡同步的`physical-microbatch-PE`；frozen base进入同合同AlpacaEval作为训练前headroom；本轮只运行seed42并按探索性结果报告；MT-Bench退出Round4；冻结Qwen3模型与数据一样由4090-3生成独立manifest并校验传至A100，且不进入镜像层。

`r4-theory-v1.0`已于2026-09-01获得用户明确通过，实验设计同步冻结为`round4-exp-v1.0`；`round4-code-v1.0`随后完成交接，当前进入`SERVER_EXECUTION`。
