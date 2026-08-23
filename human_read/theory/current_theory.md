# 当前理论：有限偏好监督下的 Label-Encoding 结构监督

## 版本信息

- Cycle ID：`cycle-20260818-01`
- 理论版本：**v0.2**
- 当前阶段：`THEORY_DISCUSSION` → **已完成，通过**
- 状态：**用户已明确确认通过**
- 用户确认状态：**已通过**
- 用户确认日期：**2026-08-19**
- 下游门禁：**`EXP_DISCUSSION` 已解锁**；`CODE_IMPLEMENTATION`、上传、`SERVER_EXECUTION`、`RESULT_HANDOFF` 与 `NEXTCYCLE_DISCUSSION` 仍锁定
- 最近更新：2026-08-19（整合 QA 决策，更新为 v0.2；MVP 静态 PE 已归入第一轮，后续第二轮只做动态 rollout 相关实验）
- 上一版本：v0.1（初始理论草稿）
- 证据等级：方法假设与实验前推导，尚无本项目实验结果
- 原始来源：
  - `../../../idea/基于有限偏好监督的动态偏好优化方法.docx`
  - 用户提供的 LERM 结构监督与 MVP 重定位附件；其内容已随工作草稿保存在 `../../../idea/理论与MVP工作草稿.md`
- QA 记录：`theory_qa_current.md` 与 `theory_qa_archive.md`

## 1. 一句话理论

在偏好标签有限时，与其把模型对每个无标签回复对的当前预测转化为硬或软伪目标，不如只用这些预测在群体层面估计两类偏好方向的 label encoding，并用少量真实偏好对锚定方向。MVP 已经用静态 PE 完成第一轮验证；后续第二轮只围绕 rollout 相关信息展开，检验 SFT+rollout 与 rollout-only 是否带来额外增益。

## 2. 研究问题与核心定位

高质量成对偏好数据昂贵，而监督微调数据通常只包含单个参考回复，无法直接作为标准偏好对使用。本研究希望用少量可靠偏好标签学习方向锚点，再从大量方向未知的回复对中提取不依赖逐样本伪标签的训练信号。

当前最核心、也最先需要回答的问题是：

> 不给无标签回复对分配 instance-level pseudo target，仅利用其整体 label-encoding structure，能否改善 preference learning？

这里的核心贡献候选不是笼统地“把 LERM 当作先验”，而是把监督单位从单个样本的目标，改为无标签回复对群体的偏好编码结构：

$$
\text{instance-level pseudo target}
\quad\longrightarrow\quad
\text{population-level label-encoding structural supervision}.
$$

需要精确说明的是：结构方法仍然使用每个样本的预测概率作为 soft responsibility，但不把这个概率或其阈值结果当作该样本应拟合的目标。训练约束来自按 responsibility 聚合后的类别编码。

## 3. 研究路线：两轮实验与 rollout 消融

两份初始材料描述的是同一研究方向的两个实验层次，但这里要明确区分：**第一轮 MVP 已完成静态 PE**，后续第二轮只补 rollout 相关实验，不再把静态 PE 当作新的主项。

1. **第一轮：静态 PE MVP。** 在已有完整标签的 preference dataset 上人为隐藏大部分训练标签，只检验 label-encoding 结构监督本身是否提供有效信息。本轮不引入 on-policy 采样，只使用静态无标签回复对；它已经承担了“无 rollout 的结构监督能否工作”的验证。
2. **第二轮主线：动态 SFT+rollout。** 在同一理论框架下，使用 SFT 参考回复与当前策略在线生成回复构造方向未知候选对，并按同样的 PE 规则计算群体级编码与联合损失；这回答“rollout 带来的动态候选信息是否提供额外增益”。
3. **第二轮消融：rollout-only。** 进一步去掉 SFT 参考回复，只保留 rollout 生成的候选回复，用来检验 SFT 参考锚点是否必要。该消融用于回答“PE 是否必须依赖 SFT+rollout，而不是纯 rollout”。

这种并置式设计保留了 rollout 相关的动态信息，同时明确把静态 PE 视为已完成的 MVP 部分，避免把数据来源、采样策略和损失函数的变化混为一谈。

## 4. 统一符号与偏好概率

### 4.1 模型与隐式偏好得分

以监督微调模型初始化待优化策略 $\pi_{\theta}$，并保留冻结的参考模型 $\pi_{\mathrm{ref}}$。对输入 $x$ 和回复 $y=(y_1,\ldots,y_T)$，序列对数概率为

$$
\log \pi_{\theta}(y\mid x)
=
\sum_{t=1}^{T}\log \pi_{\theta}(y_t\mid x,y_{<t}).
$$

定义相对参考模型的隐式偏好得分

$$
r_{\theta}(x,y)
=
\beta\left[
\log\pi_{\theta}(y\mid x)
-
\log\pi_{\mathrm{ref}}(y\mid x)
\right],
$$

其中 $\beta>0$ 控制偏好区分尺度与策略偏离参考模型的程度。

对任意回复对 $(x_i,y_{i1},y_{i2})$，定义

$$
\Delta_i
=
r_{\theta}(x_i,y_{i1})-r_{\theta}(x_i,y_{i2}),
\qquad
p_i
=
\sigma(\Delta_i)
=
P_{\theta}(y_{i1}\succ y_{i2}\mid x_i).
$$

对应的二维预测编码为

$$
\mathbf q_i
=
\begin{bmatrix}
p_i\\
1-p_i
\end{bmatrix}.
$$

### 4.2 数据划分与静态候选对

第一轮 MVP 已完成静态 PE；合并后的主实验这里只把它作为已验证的基线，并从带有真实方向的 preference dataset 构造

$$
D=D_L\cup D_U\cup D_{\mathrm{test}},
$$

其中：

- $D_L$ 保留少量真实偏好方向，用于 DPO 锚定；
- $D_U$ 在静态 PE 主线中只暴露已有回复对，不生成新回复；它提供 label-encoding 所需的静态候选对；
- $D_{\mathrm{test}}$ 的真实方向仅用于最终评价。

初始建议比例为 $10\%:80\%:10\%$，但它是首轮实验提案而不是理论常数。划分时必须防止 prompt 或近重复样本跨集合泄漏；隐藏标签后还应随机化回复位置，避免原数据的 chosen/rejected 排列泄漏方向。

真实偏好锚定损失写为

$$
L_{\mathrm{DPO}}(D_L)
=
-\mathbb E_{(x,y^w,y^l)\sim D_L}
\log\sigma\left[
r_{\theta}(x,y^w)-r_{\theta}(x,y^l)
\right].
$$

## 5. Label-Encoding 结构监督

将两个偏好方向的理想编码定义为

$$
\mathbf e_+
=
\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\mathbf e_-
=
\begin{bmatrix}0\\1\end{bmatrix}.
$$

对无标签批次中的样本，以 $p_i$ 与 $1-p_i$ 作为属于两个方向的 soft responsibility，估计条件编码

$$
\widehat{\mathbf e}_+
=
\frac{\sum_i p_i\mathbf q_i}{\sum_i p_i},
\qquad
\widehat{\mathbf e}_-
=
\frac{\sum_i(1-p_i)\mathbf q_i}{\sum_i(1-p_i)}.
$$

定义偏好编码结构损失

$$
L_{\mathrm{PE}}(D_U)
=
\frac{1}{2}
\left(
\left\|\widehat{\mathbf e}_+-\mathbf e_+\right\|_1
+
\left\|\widehat{\mathbf e}_--\mathbf e_-\right\|_1
\right).
$$

合并后的主线总目标为

$$
\boxed{
L
=
L_{\mathrm{DPO}}(D_L)
+
\lambda L_{\mathrm{PE}}(D_U)
},
$$

其中 $D_U$ 在第一轮 MVP 中表示已有无标签回复对；在第二轮中则分别表示 SFT+rollout 候选对与纯 rollout 候选对。$L_1$ 距离是初始方案，最终是否优于 $L_2$、KL 或其他结构距离需要由消融验证。

### 5.1 与伪目标方法的本质差异

硬伪标签会把

$$
\widetilde z_i=\mathbb{1}[p_i>0.5]
$$

作为单个样本的目标；soft-target 方法仍会要求样本拟合某个连续目标。二者都可能把早期错误固化为 confirmation bias。

本方法不声明“第 $i$ 个回复对一定属于哪一侧”，而是要求无标签回复对总体形成可分的两类偏好编码结构。它使用了个体预测来计算 responsibility，因此不是完全不依赖个体预测；更准确的说法是：**不信任个体预测足以成为监督目标，只把它用于构造群体级结构约束。**

## 6. 合并式实验框架：第一轮静态 PE 已完成，第二轮聚焦 rollout

### 6.1 对照组

第一轮 MVP 已经包含 DPO-10%、Pseudo-target/SSPO 类基线、静态 PE 与 DPO-100% 等参照。第二轮**不重跑这些已有臂**，只新增两条 rollout 相关 PE 实验：

1. **DPO + PE（SFT+rollout）**：对每个 SFT prompt 由当前策略在线生成 $y_j^o$，构造 $D_U(\theta)=\{(x_j,y_j^s,y_j^o)\}$，并在该动态候选集合上计算同样的 $L_{\mathrm{PE}}$。
2. **DPO + PE（rollout-only）**：进一步去掉 $y_j^s$，只保留 rollout 生成的候选回复，用来检验 SFT 参考锚点是否必要。

DPO-10%、DPO-100%、静态 PE 与 Pseudo-target/SSPO 等均作为第一轮冻结基线只读引用，不作为第二轮新任务。第二轮的目的不是再验证静态 PE 或重新建立上下界，而是把“rollout 是否提供关键增量”“SFT 参考是否必要”放在同一条主线里比较，并与第一轮结果分离存储、分离命名。

第一轮可使用一个完整标注的 preference dataset，例如附件提出的 UltraFeedback。具体数据版本、过滤、划分和模型选择属于实验设计，必须在运行前固定。

### 6.2 核心指标

测试集偏好方向准确率为

$$
\operatorname{Acc}
=
\frac{1}{|D_{\mathrm{test}}|}
\sum_i
\mathbb{1}\!\left[(p_i>0.5)=z_i\right].
$$

概率质量使用 Brier score：

$$
\operatorname{Brier}
=
\frac{1}{|D_{\mathrm{test}}|}
\sum_i(p_i-z_i)^2.
$$

第一轮不需要用 AlpacaEval 取代上述机制指标。生成质量评估可以在核心机制成立后补充。

### 6.3 成功、失败与决策标准

- **最低有效性条件**：DPO + PE 在独立测试集上稳定优于 DPO-10%，说明无标签回复对提供了可利用信息。
- **核心贡献条件**：DPO + PE 稳定优于公平调参的 Pseudo-target，说明群体结构监督比逐样本伪目标更可靠。
- **上界参照**：与 DPO-100% 的差距用于判断尚有多少标签信息未被结构方法利用。
- “稳定优于”不能由单次随机种子或单个切分决定；重复次数、不确定性和判断阈值应在实验设计中预先约定。
- 若最低有效性条件不成立，应先研究 $L_{\mathrm{PE}}$ 在 preference space 中为何失效，而不是立即加入 on-policy。
- 若仅优于 DPO-10% 而不优于 Pseudo-target，只能说明结构正则可能有帮助，不能支撑“优于 instance-level target”的核心论点。

## 7. 第二阶段：动态 SFT 候选对扩展

只有第一阶段支持核心命题后，才进入动态版本。给定少量真实偏好数据

$$
D_P=\{(x_i,y_i^w,y_i^l)\}_{i=1}^{N_P}
$$

和监督微调数据

$$
D_S=\{(x_j,y_j^s)\}_{j=1}^{N_S},
$$

对每个 SFT prompt 使用当前策略在线生成

$$
y_j^o\sim\pi_{\theta}(\cdot\mid x_j),
$$

并构造方向未知的动态候选集合

$$
D_U(\theta)
=
\{(x_j,y_j^s,y_j^o)\}_{j=1}^{N_S}.
$$

$y_j^s$ 只是 SFT 参考回复，不能预设为 chosen；$y_j^o$ 也不能预设为 rejected。定义

$$
p_j
=
\sigma\left[
r_{\theta}(x_j,y_j^s)-r_{\theta}(x_j,y_j^o)
\right]
$$

后，仍在批次层面估计“参考回复更受偏好”和“在线回复更受偏好”两类编码，并联合优化

$$
L
=
L_{\mathrm{DPO}}(D_P)
+
\lambda L_{\mathrm{PE}}(D_U(\theta)).
$$

随着 $\theta$ 更新，在线回复、候选对分布、偏好概率和编码估计共同变化，从而形成

$$
\text{在线生成}
\rightarrow
\text{候选对构造}
\rightarrow
\text{偏好编码估计}
\rightarrow
\text{联合更新}
$$

的动态闭环。这里要检验的不是某个固定“第二阶段”本身，而是：在 rollout 引入后，SFT 参考锚点是否仍然提供额外信息，以及动态候选对是否优于静态候选对。

## 8. 关键假设与可证伪预测

| 编号 | 假设 | 可观察预测 | 反驳或削弱条件 |
| --- | --- | --- | --- |
| H1 | 少量真实偏好对足以锚定共享的偏好方向 | 第一轮静态 PE 的测试集 Acc 高于 DPO-10%，且 Brier 不恶化 | 多切分、多随机种子下无稳定提升 |
| H2 | 群体级结构约束比逐样本伪目标更能抑制早期错误累积 | 第一轮静态 PE 优于公平实现的硬/软 Pseudo-target | Pseudo-target 持续持平或更优 |
| H3 | rollout 引入后，SFT 参考锚点仍可能提供额外信息 | 第二轮 SFT+rollout 在受控增量实验中优于 rollout-only，并相对第一轮冻结基线有增量 | rollout-only 持平或更优，说明 SFT 锚点不必要 |
| H4 | 第一轮静态 PE 与第二轮 rollout 实验可分离存储、独立复现 | 两轮实验目录、命令和产物彼此隔离，且 MVP 代码冻结不被改写 | 两轮结果混写、命令复用不独立或覆盖既有产物 |

## 9. 与相关工作的暂定关系

- LERM 的原始目标是 label-insufficient classification；本研究借鉴的是“从无标签样本总体预测估计类别编码”的结构，而不是把分类任务或原方法原封不动搬到 preference optimization。[LERM / arXiv:2406.02862](https://arxiv.org/abs/2406.02862)
- 附件将 SSPO、CW-PO、Semi-DPO 和 RE-PO 列为 instance-level pseudo-label、weak-label 或 confidence 路线的近邻工作。这里暂时只把它们作为待对照对象；每项方法究竟监督什么、与本目标是否同构，仍需逐篇核验 `../../../相关工作/` 中的原文。
- “尚无工作把 LERM 机制直接用于 LLM preference optimization”目前只能作为待系统检索的创新性假设，不能写成已经证明的 novelty claim。distribution-level regularization、class-prior matching、prototype learning 和熵/多样性正则都是必须继续排查的邻近方向。

## 10. MVP 阶段与第二轮约束（v0.2 新增）

基于三轮 QA 讨论，以下约束已确认并预注册：

### 10.1 数据与预处理
- **数据集**：UltraFeedback（MVP 专注单一数据集）
- **数据规模**：10k 子集（1k labeled : 8k unlabeled : 1k test）
- **标签隐藏**：
  - 随机交换回复位置，避免位置泄漏
  - 确保同一 prompt 的不同回复对不跨集合
  - 需要无监督 baseline（只用 $D_U$）验证标签隐藏有效性

### 10.2 模型与预实验
- **模型**：Qwen2.5-4B-Instruct
- **Headroom 预实验**：
  - 用 1k 样本验证 DPO-100% 显著优于 SFT baseline
  - 如果 headroom < 5%，换成 Qwen2.5-1.5B 或更难数据子集

### 10.3 超参数策略
- **固定**：$\beta = 0.1$（或 0.5，从文献常见值选择）
- **搜索**：$\lambda \in [0.1, 0.3, 0.5, 1.0]$，在 validation set（从 $D_L$ 划出 10%）上选择
- **$\lambda$ scheduler**（可选消融）：
  - 主实验：Fixed-$\lambda$
  - 如果时间允许：Warmup-$\lambda$（从 0 线性增长）和 Decay-$\lambda$（指数衰减）
- **共享**：所有方法使用相同 lr, batch size, epochs
- **预注册**：避免实验后调优

### 10.4 Baseline 设置
- **第一轮冻结基线（只读引用，不重跑）**：DPO-10%、Pseudo-target/SSPO 类基线、静态 DPO+PE、DPO-100%
- **第二轮新增实验**：DPO+PE（SFT+rollout）与 DPO+PE（rollout-only）

### 10.5 成功标准
- **第一轮 MVP**：静态 PE，3 个随机种子，统计显著性 p < 0.05
- **第二轮 rollout 实验**：独立命名、独立存储、独立命令；不改写 MVP 代码，只复用公共模块并新增实现
- **第一轮最低有效性**：DPO+PE 优于 DPO-10%
- **第一轮核心贡献**：DPO+PE 优于 Pseudo-target
- **第二轮核心比较**：SFT+rollout vs rollout-only；DPO-10%、DPO-100%、静态 PE 与 Pseudo-target/SSPO 等第一轮结果只作冻结基线，不重跑
- **度量**：Acc 和 Brier 分别报告，差异 ≤1% 视为无实质差异

### 10.6 撤退标准
- **时间盒**：1-2 周诊断分析
- **第一轮触发条件**：静态 DPO+PE 与 DPO-10% 无显著差异或更差（3 种子一致）
- **第一轮撤退动作**：静态 PE 不成立时先暂停，不自动把 rollout 当作补救；第二轮只在第一轮结论清楚后单独推进

### 10.7 `C_{\gamma}` 状态
- **优先级**：第二优先级（仅在 MVP 核心机制成立后考虑）
- **当前处理**：暂时从理论中移除，待核验后再加入

## 11. 实现细节澄清（v0.2 新增）

### 11.1 梯度路径
- **第一轮静态 PE**：让梯度穿过 responsibility $p_i$（不使用 stop_gradient），作为已完成基线
- **第二轮 rollout 实验**：沿用同样的责任定义，若训练出现数值不稳定或震荡，再对分母 $\sum_i p_i$ 使用 detach

### 11.2 数值稳定性
- **第一轮静态 PE**：随机采样 + 足够大的 batch size（128-256）
- **第二轮 rollout 实验**：如果出现退化（$p_i$ 全部接近 0.5 或 1），考虑分层采样
- **监控指标**：观测 $\sum_i p_i$ 和 $\sum_i (1-p_i)$ 的分布

### 11.3 训练流程
每个训练步：
1. 从 $D_L$ 采样 batch，计算 $L_{\text{DPO}}$
2. 从 $D_U$ 采样 batch，用**当前权重 $\theta$** 实时计算 $p_i$
3. 估计 $\widehat{\mathbf{e}}_+, \widehat{\mathbf{e}}_-$，计算 $L_{\text{PE}}$
4. 反向传播 $L = L_{\text{DPO}} + \lambda L_{\text{PE}}$

Pseudo-target baseline 对齐此流程：每步实时计算 $p_i$ 并生成硬标签。

## 12. 理论边界、风险与未决问题

### 12.1 结构目标本身

- $p_i$ 同时参与 responsibility 与 prediction encoding，梯度是否穿过 responsibility、是否需要 stop-gradient 尚未确定，必须做实现消融。
- 当某一方向的责任质量 $\sum_i p_i$ 或 $\sum_i(1-p_i)$ 很小时，编码估计可能数值不稳定；是否加入 $\varepsilon$、跨批次统计或最小质量约束需要预先定义。
- 一热理想编码隐含“偏好方向应高度可分”的假设；对于真实歧义或接近无差别的回复对，这可能造成过度自信。
- 当前损失可能主要推动置信度变尖，而未必保证两类都有合理覆盖。必须同时观测方向质量、预测分布、熵、校准与坍缩，而不能只看训练损失。
- mini-batch 条件编码是否能稳定代表总体结构，依赖批大小、采样方式和类别比例。

### 12.2 数据与评估

- 人为隐藏标签的 MVP 只证明“已存在的无标签偏好对”是否可用，不直接证明 SFT 单回复可以安全转化为动态候选对。
- 标签隐藏必须真正隔离方向信息；回复位置、文件字段、缓存和预处理都可能泄漏 chosen/rejected。
- Preference Acc/Brier 检验的是偏好判别机制，不等同于最终生成质量、帮助性或安全性。
- Pseudo-target 基线必须共享相同初始化、标注预算、数据和计算预算，否则核心比较无效。

### 12.3 模型与优化

- 隐式偏好得分可能受回复长度、tokenization 和参考模型选择影响。
- $\beta$、$\lambda$、标签比例、无标签批次大小和更新频率可能改变结论，需要区分核心结论与特定超参数现象。
- 少量标签若无法提供可靠方向锚点，结构责任会继承系统性偏差。

### 12.4 尚待完成

- 核验相关工作的精确监督单位和 novelty 边界。
- 明确 `C_{\gamma}` 的定义及其是否能诊断结构学习。
- 把本理论转化为 `../exp/current_experiment.md` 中预注册的首轮 MVP 设计。
- 在代码实现前确定梯度路径、数值稳定策略和防标签泄漏检查。
