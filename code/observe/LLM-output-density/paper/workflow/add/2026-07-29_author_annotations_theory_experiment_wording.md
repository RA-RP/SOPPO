# 当前论文批注：理论解释、实验设计与术语调整

日期：2026-07-29

来源：用户对当前中文版论文的行内批注，原始附件为
`/root/.codex/attachments/ffc3df50-ddb4-430a-9ce4-64f5dcb87aaf/pasted-text.txt`。

状态：讨论结论已登记；附件中编号 5 缺失，等待用户补充。在收到明确的“编写论文”指令前，不据此修改中英文正文、Supplement、图片或压缩包。

## 1. 批注完整性

附件中实际存在的编号为：

\[
1,\ 2,\ 3,\ 4,\ 6,\ 7,\ 8,\ 9,\ 10.
\]

因此当前共有九条可执行批注，而不是十条。编号 5 从附件中缺失，不能根据上下文自行推断。用户补充后，应追加到本文件并重新检查其与其余决定是否冲突。

## 2. 批注 1：反推裸权重谱隐含的输入假设

### 用户意见

在介绍 \(A_{D,t}=W_tS_{D,t}\) 消除坐标尺度错配后，应进一步反推：从这一视角看，直接分析裸权重 \(W_t\) 隐含了怎样的输入假设。

### 最终判断

该意见应采纳，而且属于正文理论核心。

直接对 \(W_t\) 做 SVD 和 Frobenius 低秩近似，相当于在近似目标中使用单位输入二阶矩：

\[
\Sigma_{D,t}=I,\qquad S_{D,t}=I.
\]

对各向同性输入 \(z\)，有

\[
\mathbb E_{z:\,\mathbb E[zz^\top]=I}
\left\|(W_t-\widetilde W)z\right\|_2^2
=
\left\|W_t-\widetilde W\right\|_F^2.
\]

因此，裸权重谱使用的是标准欧氏输入度量：所有输入方向在近似目标中被视为互不相关、具有单位尺度，并获得相同输出误差权重。真实域激活通常并不满足该度量。

\(W_tS_{D,t}\) 的作用不是把一个外部激活特征事后拼接到权重上，而是把裸权重隐含的各向同性输入度量替换成模型在域 \(D\) 上实际经历的输入二阶矩。这样，权重奇异方向的大小才与该域上的本层输出能量直接对应。

### 正文与 Supplement 分工

- 正文：明确写出“裸权重是 \(S=I\) 的特殊情形”，并给出上述期望误差等式；
- Supplementary Sec. A：展开坐标变换、秩亏支撑空间与更一般输入度量的证明；
- 不把 fixed whitening 写成因果分离，因为当前激活本身就是当前模型状态的一部分。

## 3. 批注 2：解释为什么丢弃小奇异值产生最小 Frobenius 损失

### 用户意见

在引用 Eckart--Young 定理前，应展示为什么丢弃较小的奇异值及其对应奇异向量，会得到最小 Frobenius norm 损失。

### 最终判断

该意见应采纳。正文需要保留证明的关键两步，而不是只引用定理名称。

令

\[
A_{D,t}
=
U\operatorname{diag}(\sigma_1,\ldots,\sigma_q)V^\top,
\qquad
\sigma_1\ge\cdots\ge\sigma_q,
\]

并定义 top-\(k\) 截断

\[
A_{D,t}^{(k)}
=
\sum_{i=1}^{k}\sigma_i u_iv_i^\top.
\]

由奇异向量的正交性，

\[
\left\|A_{D,t}-A_{D,t}^{(k)}\right\|_F^2
=
\sum_{i>k}\sigma_i^2.
\]

Eckart--Young 定理进一步保证，对任意 rank-\(k\) 矩阵 \(B\)，

\[
\left\|A_{D,t}-B\right\|_F^2
\ge
\sum_{i>k}\sigma_i^2.
\]

再结合

\[
\mathbb E\|W_th-\widetilde Wh\|_2^2
=
\|(W_t-\widetilde W)S_{D,t}\|_F^2,
\]

即可得到：保留最大的 \(k\) 个功能奇异方向、丢弃较小方向，会最小化域条件下的期望本层输出误差。

### 正文与 Supplement 分工

- 正文：保留截断形式、尾部平方和以及最优不等式；
- Supplementary Sec. A：给出完整证明、重奇异值、秩亏和坐标不变性处理。

## 4. 批注 3：解释为什么逐模块相对自身基线归一化

### 用户意见

在定义 equal-5 相对功能压缩前，应解释为什么每类投影需要相对自己的 step-0 基线归一化。

### 最终判断

该意见应采纳。

七类线性投影具有不同的矩阵形状、输入维度和基线功能秩。若直接平均

\[
r_{\varepsilon,D,a,0,j}
-
r_{\varepsilon,D,a,t,j},
\]

初始功能秩较大的模块会因为拥有更多可减少的 direction 而自然占据更大权重。它回答的是原始 direction-count 的总变化，不适合表示不同投影类型的典型相对重组程度。

逐模块除以自身基线：

\[
c_{\varepsilon,D,a,t,j}
=
\frac{
r_{\varepsilon,D,a,0,j}
-
r_{\varepsilon,D,a,t,j}
}{
r_{\varepsilon,D,a,0,j}
},
\]

把变化转化为各投影相对自身功能模式预算的比例。随后 equal-5 等权平均回答：

> 一个典型的 non-QK 投影类型，相对自身基线发生了多大功能压缩？

### 必须保留的边界

- equal-5 不是功能能量归因；
- 它不能说明哪个模块贡献了多少总压缩能量；
- q/k 的基线功能秩较小，因此少量 direction 变化会在相对比例中获得较高杠杆，这也是独立报告 q/k 与 equal-7 sensitivity 的原因之一。

## 5. 批注 4：把训练设置写成逐步收紧的控制链

### 用户意见

当前实验设计只是连续列举 OPD、SFT、off-KD、seqKD、\(\alpha=.5\) 和 frozenSelf0-KD，像“报菜名”。应先介绍 OPD 与 SFT 的差异，再说明其他训练设置如何逐步收紧和消融解释。

### 最终判断

该意见应采纳，并重构整个“模型、训练设置与输入域”小节的开头。

推荐逻辑如下。

首先把 OPD 与 SFT 作为两个主要端点：

- OPD 使用 current-student rollout，并在这些前缀上匹配 dense teacher distribution；
- SFT 使用外部 reference reasoning sequence，并进行 hard next-token supervision；
- 二者同时改变 sequence producer、训练 support 和监督目标，因此直接比较只能描述总体差异，不能承担单因素因果解释。

随后按研究问题引入控制：

1. **off-KD：sequence source。**保持 dense teacher forward KL，但将 current-student rollout 替换为 frozen-teacher sequence，用于检查 on-policy sequence exposure 的作用。
2. **\(\alpha=.5\)：exposure fraction。**在相同 forward-KL 下改变 current-student sequence exposure 比例，检查轨迹是否沿 off-KD 到 OPD 的方向有序移动。
3. **frozenSelf0-KD：rollout refresh。**保留 student-generated sequence 的来源，但将 step-0 student rollout 永久冻结，只移除持续 refresh，用于区分“曾经使用学生序列”和“持续使用当前学生序列”。
4. **seqKD：target distribution。**与 off-KD 使用完全相同的 teacher sequences、顺序和优化设置，仅将 dense teacher target 改为 hard next-token target，用于检查 soft/hard target 在 matched support 下的作用。

### 写作规则

- 中文统一使用“训练设置”或“训练条件”，不使用含义模糊的“训练臂”；
- 不把 SFT 与 OPD 写成干净的一因素控制；
- 不用额外的训练控制表重复这一逻辑，优先用连贯正文说明。

## 6. 批注 5：附件缺失

附件从编号 4 直接跳到编号 6，没有编号 5 的批注。

该项保持待补，不自行构造内容。

## 7. 批注 6：详细介绍四个固定输入域

### 用户意见

不能只列出 general text、held-out math、MMLU-Pro 和 IFEval。正文应说明每个数据来源是什么、为什么选择它，以及它在论文中回答什么问题。

### 最终判断

该意见应采纳。正文当前篇幅允许增加必要的数据语义，不能把所有信息都压入 Supplement。

四个核心 probe 的建议介绍如下。

1. **General domain。**使用冻结的 Wikitext 文本切片，而不是笼统称为“Wikipedia 数据集”。它提供非数学、非指令型的一般文本锚点，用于检查功能轨迹是否只是数学训练域或 benchmark 文本的局部现象。
2. **Held-out mathematics。**使用从 Hendrycks MATH 构造、与 MATH500 和训练数据排除重叠后的冻结题面。它提供与训练任务同属数学域、但未作为训练样本或行为评测题目的外部输入分布。
3. **MMLU-Pro。**使用冻结的问题与选项文本，覆盖多学科知识和推理。它用于观察模型在跨任务知识/推理域中的内部功能状态，不等同于在该 probe 上计算 accuracy。
4. **IFEval。**使用冻结的 instruction-following constraint prompts，不包含 checkpoint 的生成回答。它用于观察模型在格式和指令约束域上的输入条件功能状态，而不是直接代替 IFEval pass rate。

### 正文与 Supplement 分工

- 正文：每个域至少说明来源、文本内容和选择理由；
- Supplement：列出 exact source、版本、样本数、抽样 seed、去重规则、tokenization、窗口、mask 和 manifest hash；
- 在核对最新 manifest 前，不把不同模型或不同审计轨的有效样本数写成统一值；
- 明确几何 probe、fixed-token output 和自由生成 benchmark Eval 是三个不同层次。

## 8. 批注 7：删除 methods 中重复的 single-seed 句子

### 用户意见

删除：

> 每个模型、每条训练设置当前只有一个独立 seed，因此全文不把 cellwise 一致性解释为训练随机性的置信区间。

该限制已在 Limitations 中说明。

### 最终判断

接受删除这句，但保留此前的统计依赖说明：

> probe、domain、module 和 checkpoint 是同一条训练轨迹上的相关测量，而不是独立训练重复。

两句话作用不同：

- 前一句防止把 cells 当作独立训练重复，属于统计协议；
- single-seed 属于外部有效性和随机性限制，放在 Limitations 更自然。

最终正文应避免重复陈述 single-seed，但不能删除 cell dependence 的协议说明。

## 9. 批注 8：替换 “gold completion”

### 用户意见

确认 `gold completion` 是否为自造术语，并寻找更专业的表达。

### 最终判断

`gold completion` 并非完全不可理解，但在本文中不够准确。尤其对 MATH500，数据集参考推理不一定是唯一正确的推理路径，称为 gold 容易暗示唯一性。

正文统一使用：

- `fixed reference sequence`；
- 或 `fixed reference completion`。

推荐英文句式：

> We evaluate each checkpoint with teacher forcing on fixed reference sequences.

更完整时可写：

> For MMLU-Pro, the reference output is deterministically constructed from the correct answer; for MATH500, we use the dataset-provided reference solution.

推荐中文：

> 我们在固定参考序列上对各 checkpoint 进行 teacher forcing。

或者：

> MMLU-Pro 使用由正确答案确定的固定参考输出，MATH500 使用数据集提供的参考解答；所有 checkpoint 均在相同参考序列上进行 teacher forcing。

## 10. 批注 9：替换 “gold option”

### 用户意见

`gold option` 与 `gold completion` 类似，不够专业，需要替换。

### 最终判断

本文测量的不是整个正确选项文本，而是输出中编码正确选项标签的 token span。建议统一使用：

- `correct answer-choice label`；
- 更精确时使用 `correct option-label token(s)`。

区域 \(A\) 推荐定义为：

> the token span encoding the correct answer-choice label

中文版：

> 表示正确选项标签的 token 区域。

因此 MMLU-Pro 的区域可写为：

- prompt \(P\)；
- pre-answer formatting tokens \(F\)；
- correct answer-choice label \(A\)；
- termination token \(T\)。

## 11. 批注 10：删除“峰值--过冲--回弹”

### 用户意见

“我们不主张统一的峰值--过冲--回弹曲线”属于过时版本留下的内容，不应出现在当前论文中。

### 最终判断

完全删除，不保留同义改写。这句话像版本更新记录或对旧 claim 的辩护，不符合当前论文的自然叙事。

推荐过渡逻辑：

> 两种模型的具体时间形态有所不同，但在共同早期窗口中呈现一致的训练设置排序。为检验这一排序是否被域平均掩盖，我们定义连续压缩边际……

结合已经收敛的图表方案：

- Figure 1 合并为双模型 `2 x 4` 构念/轨迹图；
- 原完整逐域连续 margin panel 进入 Supplement；
- 正文使用 \(M_{\min}^{\mathrm{early}}\) 保留“不存在域平均掩盖的排序反转”这一证明；
- \(M_{\min}^{\mathrm{early}}\) 与 NCD 合并到同一张轨迹摘要表；
- NCD 描述完整轨迹的压缩深度与持续时间，不能替代早期排序。

## 12. 本轮修改对论文结构的影响

### 理论部分

- “白化坐标中的同一权重映射”增加裸权重的各向同性输入度量解释；
- Eckart--Young 结论前增加截断 SVD 的两步关键推导；
- “功能秩与轨迹量”增加逐模块基线归一化的统计含义和非能量归因边界。

### 实验设计

- “模型、训练设置与输入域”重写为从 OPD/SFT 总体差异到 sequence source、exposure、refresh 和 target 的控制链；
- 四个固定输入域分别介绍 provenance、文本内容和研究角色；
- single-seed 限制只留在 Limitations，cell dependence 保留在统计协议。

### 区域输出

- `gold completion` 改为 `fixed reference sequence/completion`；
- `gold option` 改为 `correct answer-choice label`；
- 区分 MMLU-Pro 的确定性参考输出与 MATH500 的 dataset-provided reference solution。

### 结果

- 删除所有“峰值--过冲--回弹”版本历史；
- 使用共同早期排序直接进入 \(M\) 或 \(M_{\min}^{\mathrm{early}}\)；
- 与 `/root/lecture/add/2026-07-29_construct_comparison_figure_rr5_revision.md` 中的 Figure 1 合并和 margin+NCD 表格方案保持一致。

## 13. 最终执行检查

收到“编写论文”指令后，统一执行时需要：

1. 先补齐用户编号 5 的批注；
2. 从最新 `human_read-ch.md` 和 probe manifest 核对数据来源、样本数与去重；
3. 同步修改中英文理论推导；
4. 同步重写中英文实验设计；
5. 统一 reference sequence 与 answer-choice label 术语；
6. 删除旧峰形叙述；
7. 更新相关图号、表号和 Supplementary section 引用；
8. 检查新增理论文字与 Supplementary Sec. A 的证明不矛盾；
9. 编译检查页数与公式排版；
10. 最后统一同步 Overleaf 上传目录和 ZIP。
