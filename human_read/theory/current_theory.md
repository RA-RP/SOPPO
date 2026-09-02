# 当前理论：Round4 FrozenPE、StaticPE、SSPO 与 label-only DPO

## 0. 版本、状态与边界

- Cycle：`cycle-20260901-01` / Round4
- 理论版本：`r4-theory-v2.0`
- 当前唯一活动阶段：`SERVER_EXECUTION`
- 状态：`APPROVED`
- 用户确认：2026-09-02，用户明确确认 v2 方法方案、损失定义、超参数、A100/4090流水线及执行顺序，并要求“按着这个计划开始正式执行”。
- 版本关系：`r4-theory-v1.0`保留为历史合同；其双回答 DPO-base `StaticPE` 自 v2 起更名为 `FrozenPE`。v2 新增的单回答 SimPO-base 方法使用名称 `StaticPE`。
- Round3边界：Round3结果与旧动态PE实现仅作历史参考，不构成本轮方法效果证据。

## 1. 研究问题

在同一 Qwen3-1.7B 初始化、同一冻结的 UltraFeedback/UltraChat 10% 数据合同、同一 epoch 与共同优化设置下，比较：

1. `DPO-label-only`：只用 UltraFeedback labeled preference pairs；
2. `SSPO`：DPO-base labeled branch 加 UltraChat 单回答阈值风险；
3. `FrozenPE`：DPO-base labeled branch 加冻结双回答候选的 PE；
4. `StaticPE`：SimPO-base labeled branch 加单回答、连续 soft-responsibility PE。

四臂共享初始化、labeled IDs、模板、训练时长和最终评价合同。DPO不读取无标签训练 view；其余三臂共享同一filtered UltraChat IDs。由于 `StaticPE` 同时改变 labeled base 与无标签机制，未加入 SimPO-label-only 对照时，只能将它解释为整体组合方法，不把差异单独归因于 PE。

## 2. 共同模型、数据与训练合同

- 模型：`Qwen/Qwen3-1.7B`，revision `b9352fbb8ce704292730cf54b3b1dceb2a808738`。
- 模板：Qwen3 native non-thinking；训练、FrozenPE candidate B、merge后生成与AlpacaEval保持一致。
- labeled：`HuggingFaceH4/ultrafeedback_binarized` 10% filtered preference pairs；四臂共享同一有效行ID与held-out labeled eval view。
- unlabeled：`HuggingFaceH4/ultrachat_200k` 10% filtered单回答；SSPO、FrozenPE、StaticPE共享同一有效行ID。
- 优化：LoRA r8、all linear、BF16、learning rate `1e-5`、cosine、warmup `0.1`、epoch 1、seed42。
- batch：DPO每设备1、2 GPU、GA8、effective16；SSPO/FrozenPE/StaticPE每设备4、2 GPU、GA8、effective64；所有方法eval每设备4。
- PE统计语义：FrozenPE和StaticPE都在当前两卡 global physical micro-batch 上形成一次 PE，再累计8次梯度；不把一个effective batch64合并为单一PE population。

## 3. 四种目标函数

### 3.1 DPO 与 SSPO 的共同 DPO-base

令 response total log-prob 为

$$
s_\theta(x,y)=\sum_{t\in\mathrm{response}}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

$$
L_{\mathrm{DPO}}=\operatorname{mean}_i\left[-\log\sigma\left(0.1[(s_\theta^+-s_{\mathrm{ref}}^+)-(s_\theta^--s_{\mathrm{ref}}^-)]\right)\right].
$$

SSPO保持既有DPO-base、EMA reward normalization、chosen-min threshold、prior `0.5`和gamma scheduler：`gamma_0=1.0`、`gamma_decay=0.001`、`gamma_min=0.2273`。这些不是FrozenPE或StaticPE的超参数。

### 3.2 FrozenPE：冻结双回答 DPO-base PE

每条UltraChat单回答构造固定候选：A为原始回答，B为冻结初始化Qwen3以non-thinking贪心配置生成一次的回答；A/B由稳定hash交换，训练期间不刷新。

$$
r_A=0.1(s_\theta^A-s_{\mathrm{ref}}^A),\qquad
r_B=0.1(s_\theta^B-s_{\mathrm{ref}}^B),\qquad
p_i=\sigma(r_A-r_B).
$$

令 $q_i=[p_i,1-p_i]$，并定义

$$
c_1=\frac{\sum_i p_iq_i}{\sum_i p_i+\epsilon},\qquad
c_2=\frac{\sum_i(1-p_i)q_i}{\sum_i(1-p_i)+\epsilon},
$$

$$
L_{\mathrm{PE}}=\frac12\left(\lVert c_1-[1,0]\rVert_1+\lVert c_2-[0,1]\rVert_1\right),\qquad
L_{\mathrm{FrozenPE}}=\frac{L_{\mathrm{DPO}}+0.1L_{\mathrm{PE}}}{1.1}.
$$

`FrozenPE`中的“冻结”指候选B及其生成器，不指冻结训练参数。

### 3.3 StaticPE：单回答 SimPO-base PE

对每条response采用长度归一化 log-prob：

$$
\bar{s}_\theta(x,y)=\frac1{|y|}\sum_{t\in\mathrm{response}}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

对labeled pair，固定 SimPO `beta=10.0`、`gamma=2.0`：

$$
L_{\mathrm{SimPO}}=\operatorname{mean}_i\left[-\log\sigma\left(10(\bar{s}_\theta^+-\bar{s}_\theta^-)-2\right)\right].
$$

对每条无标签单回答 $u_i$，以当前global physical micro-batch中chosen、rejected和unlabeled的分数更新跨卡一致、不可微的EMA mean/variance：momentum `0.95`、clip范围 $[-5,5]$、$\epsilon=10^{-8}$。令

$$
z_i=\operatorname{clip}\left(\frac{10\bar{s}_\theta(x_i,u_i)-\operatorname{stopgrad}(\mu_t)}{\sqrt{\operatorname{stopgrad}(v_t)}+\epsilon},-5,5\right),\qquad p_i=\sigma(z_i).
$$

随后对这些 $p_i$ 使用与FrozenPE相同的 $q_i,c_1,c_2,L_{\mathrm{PE}}$，但不使用threshold、prior、hard pseudo label或正负划分：

$$
L_{\mathrm{StaticPE}}=\frac{L_{\mathrm{SimPO}}+0.1L_{\mathrm{PE}}}{1.1}.
$$

$p_i$是相对于当前reward分布的soft responsibility，不是校准后的“回答正确概率”。EMA统计不参与反向传播，PE分子、分母与 $p_i$ 保持可微。PE需要至少两个全局unlabeled响应；不足时属于实现/数据合同失败。

## 4. 评价与流水线

训练中使用共同held-out labeled view记录DPO或SimPO的raw preference诊断；不同base的loss不作为统一横向能力分数。StaticPE的EMA在eval中不更新，PE训练分项只作机制观测。

最终评价采用冻结805条AlpacaEval 2.0 instructions。A100-2完成每个方法的训练、merge/reload和输出生成，立即写入不可变request bundle；4090-3拉取bundle、调用一个冻结的primary judge profile并本地计算普通 WR 与 LC WR，再把aggregate-only result bundle回传A100。API judge可和A100训练下一方法重叠，但不允许中间结果改变后续训练、checkpoint选择或超参数。

执行顺序为：`frozen base → DPO → SSPO → StaticPE → FrozenPE`。FrozenPE candidate B在进入FrozenPE前生成一次；四个训练臂均从相同base独立开始。

一个profile可配置model、base URL和独立key reference；不同profile的结果使用不同evaluation ID，不能混为同一主指标。非官方GPT-4 Turbo profile的指标明确标为“AlpacaEval 2.0-compatible custom-judge WR/LC”。

## 5. 可证伪预测与边界

| 编号 | 预测 | 支持证据 | 削弱/反驳条件 |
| --- | --- | --- | --- |
| R4-P1 | DPO在共同labeled eval上形成稳定偏好学习 | DPO loss、margin、accuracy | 非工程故障下不改善 |
| R4-P2 | FrozenPE可降低PE诊断而不损害DPO branch | DPO/PE分项、p/c摘要、最终LC | PE坍缩或最终能力恶化 |
| R4-P3 | StaticPE可在不生成candidate B的情况下形成非退化连续结构信号 | SimPO/PE分项、EMA、p分布、finite梯度 | p长期单侧、NaN或仅长度增长 |
| R4-P4 | SSPO的行为依赖threshold机制 | threshold、pseudo-side、loss telemetry | threshold不稳且无收益 |
| R4-P5 | 仅普通WR提升而LC不提升时，不能归因于真实质量提升 | WR、LC、长度共同报告 | 结果主要由长度解释 |

单种子只支持探索性比较。FrozenPE与StaticPE、SSPO与StaticPE并非严格单变量消融；token compute、reference forward和candidate数量必须随结果报告。

## 6. 执行边界

- `af6dac4`的环境和资产传输证据、以及预处理失败证据均保留；尚无Round4训练、merge、生成或judge结果。
- `6b010b8`的确定性无效行过滤修复必须进入v2代码，但不作为独立执行版本。
- 本地仅编辑、静态复核和代码交接；依赖、数据、模型、测试、smoke、训练、生成、judge和LC均在服务器完成。
- v2代码形成新的exact commit并获得用户明确交接确认前，不上传、不重建环境、不运行服务器任务。
