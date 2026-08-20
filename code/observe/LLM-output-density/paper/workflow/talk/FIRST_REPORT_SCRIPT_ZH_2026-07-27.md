# 第一次研究汇报讲稿

建议时长：15--18 分钟  
配套材料：`/root/LLM-output-density/mypaper/theory/human_read-ch.md` v13  
使用方式：主讲只看 `human_read` 正文第 1--7 节；附录仅在被追问时打开。

## 0. 开场：我在研究什么（约 1 分钟）

我先用一句话概括这项工作。

我想研究的不是 OPD 最后能把准确率提高多少，而是当训练序列由当前模型持续生成和刷新时，模型在
特定输入域上实际调用的内部计算模式会怎样变化。

现有分析通常位于两个端点。一端直接分析权重更新，例如更新范数、秩、谱和参数子空间；另一端直接
观察最终输出，例如准确率、KL、NLL、格式和终止行为。前者能告诉我们参数在哪里改变，却不知道这些
方向是否会被真实输入访问；后者能告诉我们行为发生了什么，却把中间所有层和模块压缩成了最终结果。

所以这篇工作的核心问题是：能不能定义一个有明确数学意义的内部功能状态，把当前权重和特定输入域
实际访问的激活方向结合起来，并沿训练 checkpoint 观察它如何演化？

此时可展示：`human_read` 开头的“当前最高层结论”。

## 1. 为什么需要这个观察对象（约 2 分钟）

我不是想构造一个介于权重和输出之间的折中分数，而是想测量一个不同的对象：模型在某个输入域上
实际执行的局部计算。

假设某层的权重是 \(W_t\)，输入激活为 \(h_t\)。即使两个参数方向具有相同的更新幅度，如果其中一个
方向几乎不会被输入访问，它们的功能意义也可能完全不同。反过来，一个很小的权重变化，如果发生在
高频访问的输入方向上，也可能产生明显的局部输出变化。

纯输出同样不能替代这个对象。准确率或者 KL 可以说明模型最终变了多少，但很难说明变化位于哪一层、
哪个模块，或者内部计算是在减少有效方向、增加有效方向，还是只发生方向旋转。

因此，我们希望回答的是：

> 特定输入域实际访问的方向经过当前权重映射以后，需要多少个主要功能模式才能保留绝大部分局部
> 输出能量？

这也是本文所说“域条件功能模式”的含义。

此时可展示：`human_read` §3.1 的理论来源链。

## 2. 方法定义（约 2 分钟）

对 checkpoint \(t\) 和输入域 \(D\)，我们首先计算激活的二阶矩

\[
\Sigma_{D,t}=\mathbb E[h_t h_t^\top],
\qquad
S_{D,t}S_{D,t}^\top=\Sigma_{D,t}.
\]

然后构造

\[
A_{D,t}=W_tS_{D,t}.
\]

对它进行 SVD 后，每个奇异三元组可以看成一个域条件输入--输出功能模式，奇异值平方表示该模式承载
的本层输出能量。我们定义 \(r_\varepsilon\) 为保留至少 \(1-\varepsilon\) 输出能量所需的最小模式数。
正文使用 \(\varepsilon=0.05\)。

这个定义不是纯经验启发。对任意低秩近似 \(\widetilde W\)，都有

\[
\mathbb E_D\|(W_t-\widetilde W)h\|_2^2
=
\|(W_t-\widetilde W)S_{D,t}\|_F^2.
\]

因此，对 \(W_tS_{D,t}\) 做截断 SVD，恰好对应当前域二阶矩下最小化期望本层输出误差的低秩近似。
所以它在这个局部线性问题中是精确定义的，不需要 Fisher 或 Hessian 近似。

这里也要马上说明边界：它只对当前层的线性映射和二阶输入度量完整，不包含后续非线性、残差路径和
最终 token readout。因此我们把它称为局部功能状态，不把它称为完整模型能力。

此时可展示：`human_read` §1.3 和 §2.1。

## 3. 实验为什么不只是 OPD 对 SFT（约 2 分钟）

如果只比较 OPD 和 SFT，我们无法判断差异来自 sequence support、训练目标还是数据来源。因此目前的
实验形成了一条逐步收紧的控制链。

OPD 使用 current student rollout 和 teacher forward-KL。off-KD 保留同样的 KL objective，但改用
冻结的 teacher rollout。seqKD 与 off-KD 使用完全相同的 teacher 序列，只把 dense KL 换成
hard-label CE。SFT 使用外部 reference CoT。

除此之外，我们还有两个更直接的 exposure 对照。

第一是 \(\alpha=.5\)，将 current-self 与冻结 support 各占一半，用来观察 exposure 比例变化是否让
功能轨迹沿 off-KD 到 OPD 的方向移动。

第二是 Llama 上完成的 frozenSelf0-KD。它让 step-0 student 只生成一次 rollout，之后永久冻结；
teacher KL、prompt pool、LoRA 和优化器都与 OPD 对齐。这样 OPD 与 frozen-self 的主要差别就是
current-student support 是否持续刷新。

这个比较识别的是 current-support-refresh bundle 的总效应。刷新同时可能改变长度、EOS、重复率和
风格，因此我们不声称已经把它拆成一个纯粹的 freshness 通道。

此时可展示：`human_read` §1.1 的训练臂表格。

## 4. 主要结果一：OPD 更早进入跨域功能压缩（约 2 分钟）

第一个主要发现是，不同训练范式具有可区分的 checkpoint-wise 功能轨迹。

我们在 Qwen 和 Llama 两个模型家族上，使用相同数值协议重新计算了四条核心训练臂。在预先固定的
共同早期窗口 \(t\in\{20,40,80\}\) 中，OPD 在四个核心 probes 的 24 个模型--域--checkpoint
比较单元里，有 23 个单元的功能压缩最深。唯一例外是 Qwen 的 IFEval step20，而且只相差
0.143 个 direction，属于近似持平。

如果把共同训练范围内的负压缩沿时间积分，Qwen 和 Llama 上 OPD 的负压缩暴露也都是最大的。

这里“功能压缩”只表示：相对于 step0，为保留同样比例的本层输出能量所需的有效模式数减少。它不
等价于知识消失，也不等价于准确率下降。

两个模型共享的是 OPD 较早、较一致地进入跨域压缩，而不是完全相同的曲线形状。Qwen 有局部恢复，
Llama 则持续加深，所以我们已经放弃了统一的“过冲--恢复”叙事。

此时可展示：`human_read` §2.3 或 §6.2--6.3。

## 5. 主要结果二：current support refresh 是重要组织因素（约 2 分钟）

仅有 OPD 与 off-KD，还不能区分 student/teacher 生成风格和在线刷新。frozen-self 给出了更直接的
比较。

在 Llama 的 headline 层和 \(\varepsilon=0.05\) 下，比较 OPD 与 frozenSelf0-KD：

- 六个 probes、五个后续 checkpoints 中，OPD 在 29/30 个单元压缩更深；
- 如果只看五个固定外部 probes，则是 25/25；
- 在 step160，五个外部 probes 上 OPD 都比 frozen-self 多压缩约 4.9 到 13.4 个 directions。

与此同时，Qwen 的 \(\alpha=.5\) 在多数严格可比单元中位于 off-KD 到 OPD 之间，但存在明显正交
残差，因此 exposure 的影响有顺序结构，却不是一个简单的一维线性剂量函数。

这两组结果共同支持：on-policy 的主要差异位于 current support exposure，持续刷新是形成 OPD
功能压缩轨迹的重要组织因素。由于 frozen-self 目前只有 Llama 的单条训练轨迹，这仍然是强描述性
证据，而不是随机种子意义上的普遍因果定律。

此时可展示：`human_read` §6.3。

## 6. 为什么这个指标不等同于已有权重几何（约 2 分钟）

一个自然问题是：既然已经有 update rank、source-principal projection 和奇异向量旋转，为什么还要
计算 \(W_tS_{D,t}\)？

我们已经在相同的 192 个 cells、相同 checkpoint-grouped folds 上做了公平比较。结果并不是所有
权重指标都失效。严格的 source-principal \(p_k\) 是很强的基线：预测 cumulative KL 的 held-out
\(R^2\) 是 0.578，而 \(c_\varepsilon\) 单独是 0.504。二者结合后提高到 0.598；在识别 OPD
训练臂时，\(p_k\) 的 AUC 是 0.836，\(c_\varepsilon\) 是 0.720，组合后达到 0.894。

因此我们的结论不是功能秩全面优于裸权重。更准确的解释是：

- \(p_k\) 描述更新写入预训练 source 主坐标的位置；
- \(c_\varepsilon\) 描述特定输入域上当前实现的功能模式预算；
- 二者观察的是互补对象。

\(c_\varepsilon\) 相对 raw update energy 和当前 TPNT-style suite 有明显增量；相对强 \(p_k\) 的
pooled 增量较小，而且主要来自 Llama。这个模型依赖性需要如实保留。

此时可展示：`human_read` §3.3 或 §6.6。

## 7. 它与最终输出有什么关系，又不能解释什么（约 2 分钟）

第二个自然问题是：如果最终关心的是模型行为，为什么不直接看 KL、NLL 或准确率？

双模型四臂的完整结果显示，相对功能压缩 \(c_\varepsilon\) 与 cumulative KL 和 absolute NLL
普遍强同动。这说明它确实记录了模型相对 base 的累计输出 departure，而不是一个与输出无关的内部
统计量。

但它不能替代输出分析。最清楚的例子是 Qwen 的 off-KD 和 seqKD：两者使用完全相同的 teacher
序列，功能 rank path 的 Pearson 相关达到 0.985，平均只差 1.426 个 directions；可是它们在
MATH、截断、EOS 和严格格式上明显分叉。

这说明 sequence support 可以组织相近的功能模式预算，而 objective 仍然可以改变这些功能状态最终
如何被 token readout 使用。我们把它称为 support--readout separation。

我们原本还假设只有 OPD 的压缩会与 signed NLL 恶化对齐，但完整 Qwen 四臂结果否定了这个跨模型
命题。因此当前保留的结论是：功能压缩稳定追踪无符号 output departure，至于变化对 reference
likelihood 是有利还是不利，则依赖模型和训练臂。

此时可展示：`human_read` §2.4--2.6 和 §6.4--6.7。

## 8. 我认为当前的三个主要贡献（约 1 分钟）

目前我把贡献收束为三点。

第一，我们把 activation-aware low-rank approximation 从静态压缩工具转化为 checkpoint-wise 的
域条件功能模式观察空间，并给出 \(r_\varepsilon\) 的局部最优输出误差解释。

第二，我们建立了 OPD、frozen-self、off-KD、seqKD、SFT 和 \(\alpha=.5\) 的控制链，分别约束
current support refresh、生成来源和 objective 的解释。

第三，我们发现 OPD 在两个模型家族上具有一致的早期跨域功能压缩支配，并通过 frozen-self 将主差异
进一步收紧到 current support refresh bundle；同时明确了它与强权重基线互补、与输出相关但不能
充分决定行为的边界。

所以本文不是提出新的训练方法，而是提供一种观察 post-training 内部功能演化的方法和一组关于
on-policy exposure 的经验规律。

## 9. 局限与希望讨论的问题（约 1 分钟）

当前最大的限制有四个：所有训练臂只有一个独立 seed；实验都是 LoRA；指标是局部层级而不是端到端
功能；frozen-self 识别了 refresh bundle，但没有拆分长度、EOS、重复和风格等中介。

我希望这次主要讨论三个问题：

1. 论文应该以“新的功能几何观察空间”为第一定位，还是更突出“on-policy support refresh 机制”？
2. 当前方法、控制链和跨模型规律是否已经构成足够集中的论文贡献？
3. 剩余资源最值得投入独立 seed、进一步的 refresh 机制拆分，还是集中完成论文表达与图表？

我的总结是：

> 裸权重告诉我们参数改在了哪里，纯输出告诉我们行为变成了什么；域条件功能模式描述的是特定输入域
> 实际访问的方向经过当前权重以后形成了怎样的局部计算结构。沿这个空间追踪，我们看到 current
> support refresh 系统性地组织了 OPD 的早期跨域功能压缩轨迹。

## 被打断时的简短回答

### 这不就是 SVD-LLM 吗？

\(WS_D\) 的局部输出误差解释来自 activation-aware compression，我们明确承认这一方法血缘。本文的
贡献是把它从一次性压缩准则改造成沿 post-training checkpoint 追踪的域条件状态变量，并用 matched
on/off-policy 控制研究功能轨迹。

### 为什么不用 Fisher？

Fisher 通过预测分布或 loss 的局部二阶近似衡量参数敏感性；我们不近似端到端 loss，而是精确描述
指定线性模块在当前域二阶矩下的局部输出近似。前者更接近全局敏感性但依赖近似，后者更局部但定义
精确，二者回答的问题不同。

### \(W_tS_{D,t}\) 不是混合了权重和激活变化吗？

是的，对状态描述而言这是有意的，因为我们要测的就是当前权重与当前访问方向共同形成的 realized
computation。它不能单独做因果归因；fixed whitening 等消融用于分析来源，但 headline 指标描述联合
功能状态。

### 压缩是不是说明能力下降？

不是。压缩只表示主要本层输出能量集中到更少的功能模式。off-KD/seqKD 的行为分叉以及 signed NLL
的模型依赖性都说明，压缩不是能力损伤的充分条件。

### 23/24 和 29/30 能当作统计显著性吗？

不能。这些是预先冻结网格上一致性的描述，不是独立训练重复。当前每个 model×arm 只有一条训练
轨迹，所以论文把它们称为 strong descriptive evidence，并把独立 seed 列为首要限制。
