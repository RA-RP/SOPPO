# 面向在策略蒸馏的匹配增益诊断：推理稳定性与激活条件谱的最小研究

## 摘要

在策略蒸馏（on-policy distillation, OPD）越来越多地被用于将较大教师模型的推理行为迁移到较小学生模型中。其核心做法是让学生模型在自身策略采样得到的状态上接受教师监督。然而，在实际研究中，关键问题往往不只是 OPD 是否能提升某个同分布 benchmark，而是它能否避免持续监督微调（continued supervised fine-tuning, continued SFT）中可能出现的分布外退化和表示漂移。

本文提出一个最小诊断协议来研究这一问题。该协议在同分布增益匹配的条件下比较 OPD-like 训练与 continued SFT，并将任务指标与激活条件谱诊断结合起来。谱诊断基于冻结 prompt 探针和训练目标探针计算。

在一个初步的 TRL 实验中，我们使用 Qwen3-1.7B 作为学生模型、Qwen3-4B 作为教师模型。一个 OPD-like 运行与最近的 SFT 对照在 GSM8K 增益上相差 0.010。在这一匹配条件下，OPD-like 模型具有更低的 OOD-lite penalty（0.0100 vs. 0.0182）、更低的 worst OOD-lite drop（0.0100 vs. 0.0150），以及显著更低的 layer-14 七个目标模块平均谱漂移（0.0179 vs. 0.0488）。OPD-like 模型在所有七个测量模块上的漂移都低于匹配 SFT 对照。

这些结果不能证明 OPD 提升了 GSM8K 准确率：当前评估只使用 200 个样本，只有一个 OPD 设置，并且几何分析只覆盖一层。本文的贡献是一个谨慎且可复现的诊断框架：在任务表现匹配时，OPD-like 训练相较 continued SFT 可以表现出更干净的稳定性特征。这一发现推动下一步更严格的实验：更高样本数评估、多个 OPD 设置、多 seed、以及多层几何分析。

## 1. 引言

知识蒸馏是将高能力教师模型压缩到较小学生模型中的常见方法。但是，在自回归语言模型中，蒸馏天然存在一个训练-推理不匹配问题：学生模型在训练时看到的是一类状态，而在推理时访问的是由自身生成行为诱导出的状态。在策略蒸馏通过让学生模型生成轨迹，并在学生实际访问的状态上使用教师反馈，来缓解这一不匹配问题。

本文研究一个更窄但更实际的诊断问题。假设一个小型推理模型已经经过 cold-start 蒸馏。从同一个起点出发，我们应该继续使用 OPD-like 训练，还是使用普通 SFT 在额外的监督推理轨迹上继续训练？直接比较 accuracy 是不够的。SFT 可能将模型推向较窄的答案轨迹分布，即使它在邻近 benchmark 上看起来保持或提升了表现，也可能损害训练风格之外的推理行为。另一方面，当教师-学生重叠较弱时，OPD 也可能昂贵且不稳定。因此，更有用的问题是一个条件性问题：

> 在同分布表现相当的条件下，哪一种训练路径能更干净地保留分布外行为和模型几何？

我们为此提出一个最小匹配增益诊断。第一，从同一个起始 checkpoint 训练多个 continued-SFT 对照和至少一个 OPD-like 模型。第二，根据同分布推理 benchmark 上的增益匹配 OPD 与 SFT。第三，用 OOD-lite 任务指标和激活条件谱诊断评估匹配后的模型对。

匹配增益步骤避免了一个常见混淆：一个方法有更低的 OOD 退化，可能只是因为它学得更少。谱诊断则进一步追问：任务层面的差异是否伴随着模型在探针分布下发生变化方式的可测差异。

我们的初步实验使用 Qwen3-1.7B 作为学生模型，Qwen3-4B 作为教师模型，prompt 池来自 NuminaMath。训练包括基于 TRL 的 OPD-like 训练和 continued SFT 对照。主匹配对是使用 800 个 rollout prompt 的 OPD-like 训练和使用 256 个监督样本的 SFT。它们的 GSM8K 增益相差 0.010，低于预注册的 valid-match 阈值 0.02。

在这一匹配下，OPD-like 训练具有更低的 OOD-lite penalty 和 worst OOD-lite drop。它相对于起始 checkpoint 的 layer-14 激活条件谱漂移也明显更低，不论是七个模块平均，还是逐个 attention 与 MLP projection 模块查看，结论都一致。

这一结果必须谨慎解读。它不是完整 OPD 复现，不是广泛 OOD 泛化声明，也不是 OPD 提升 GSM8K 准确率的证据。初步实验只用了 `eval_limit=200`，只有一个 OPD 超参数设置，几何分析也只覆盖一层。它的价值在于闭合了一个最小工程循环，并明确了更尖锐的下一轮实验：在更低评估噪声、更多 OPD 设置或 seed、更不饱和的起点以及更广泛几何测量下，检验同样的稳定性信号是否仍然存在。

本文贡献如下：

1. 提出一个匹配增益诊断协议，用于在同分布推理表现相当的条件下比较 OPD-like 训练与 continued SFT。
2. 给出一种激活条件谱分析，将 OOD-lite 退化与相对起始 checkpoint 的谱漂移联系起来。
3. 在一个初步 TRL 实验中显示：在有效 GSM8K 匹配下，OPD-like 训练相较 continued SFT 可以保留更干净的稳定性特征。
4. 给出一个明确的后续实验 gate，用于从“稳定性声明”推进到可能的“提升 + 稳定性声明”。

## 2. 相关工作

### 在策略蒸馏

Generalized Knowledge Distillation 让学生模型在自身生成的输出序列上训练，并使用教师对这些序列的反馈，从而直接针对教师生成训练输出与学生推理输出之间的分布不匹配问题。近期 OPD survey 和机制研究则从学生采样轨迹、反馈粒度和稳定性条件等角度组织这一系列方法。

与本文最相关的机制观点是：成功的 OPD 会在学生访问状态上，使学生逐渐对齐教师支持的高概率 token；而较差的教师-学生重叠可能导致 OPD 失败。这也是本文采用 cold-start 加 OPD-like 设置的原因：学生需要足够接近教师支持区域，使 on-policy 反馈具有意义。

### 微调与推理退化

SFT 高效且简单，但它对推理能力的影响可能是混合的。关于 fine-tuning 与 chain-of-thought reasoning 的近期工作报告了一个现象：任务特定微调可能降低推理表现，并影响推理过程的 faithfulness，尤其是在较小模型上。这并不意味着 SFT 总是有害。但它意味着 continued-SFT baseline 应该被显式评估是否产生退化，而不应被视为中性的“额外训练量”。

本文使用多个 SFT 数据量，正是因为监督轨迹拟合程度可能同时改变同分布表现和 OOD 行为。

### 适应过程的谱视角

谱分析已经成为理解 fine-tuning 如何改变模型的有用工具。例如，LoRA 与 full fine-tuning 的比较研究显示，不同微调方式可以产生不同的奇异向量结构，并将这些结构与遗忘联系起来。

本文的分析不同于直接检查原始权重矩阵。我们先用探针文本收集激活统计，再在该激活分布下对线性权重进行条件化，然后计算奇异谱。这使得诊断对模块在特定分布下如何被使用更加敏感。更广泛的动机是一致的：如果两种适应方法获得了相近任务表现，但沿着不同谱路径移动模型，那么这些路径可能解释 retention 和 generalization 的差异。

### 表示层 OPD

OPRD 提出将在策略蒸馏从输出空间 token 分布移动到 hidden-state alignment。本文没有实现表示层监督。然而，我们的结果与这一方向兼容：如果 output-space OPD-like 训练已经伴随更低的激活条件漂移，那么表示层诊断以及未来的表示层目标都是自然延伸。

## 3. 问题设置

令 $\theta_0$ 表示一个 cold-start 学生 checkpoint，它通过在小规模 prompt 集上从教师蒸馏得到。从 $\theta_0$ 出发，我们比较两类训练方法。

第一类是 OPD-like 训练：当前学生模型采样 completion，并在这些采样轨迹上接收 token-level 教师监督。第二类是 continued SFT：学生模型在固定监督推理轨迹上训练。

令 $m(\theta)$ 表示同分布推理指标，这里是 GSM8K accuracy。定义相对共同起点的增益为：

$$
g(\theta) = m(\theta) - m(\theta_0)
$$

对于每个 OPD-like 模型 $\theta_o$，我们根据绝对增益差选择最近的 SFT 模型 $\theta_s$：

$$
\Delta_g(\theta_o,\theta_s)=|g(\theta_o)-g(\theta_s)|
$$

当 $\Delta_g \leq 0.02$ 时，该模型对被视为 valid match。

核心稳定性问题是：

> 在同分布推理增益相当的模型之间，OPD-like 训练是否比 continued SFT 产生更低的 OOD-lite 退化和更低的激活条件谱漂移？

这一框架刻意区分两个声明。第一个是稳定性声明：相对于匹配 SFT 对照，OPD 更好地保留行为和几何。第二个是更强的提升声明：OPD 在保留稳定性的同时提升同分布 benchmark。当前实验只能支持第一个声明。

## 4. 方法

### 训练分支

实验使用一个统一 prompt 池，并将其切分为 training、held-out 和 probe 子集。Qwen3-1.7B 学生模型首先用 Qwen3-4B 教师模型在 512 个 prompt 上进行 cold-start。随后从同一个 $\theta_0$ 出发，训练一个 OPD-like 分支和四个 continued-SFT 对照。

OPD-like 分支使用 $\lambda=1.0$，消耗 800 个 rollout prompt。SFT 对照分别使用 256、512、1024 和 2048 个监督样本。所有分支使用相同 LoRA target modules，并从同一个 $\theta_0$ 出发。

### 任务指标

同分布匹配指标是 GSM8K。由于初步实验只使用 200 个样本作为评估限制，小于约 0.02 的差异不应被解释为可靠提升。

OOD-lite 表现由 average OOD score、penalty $p2$ 和 worst-drop 指标总结。具体任务组合遵循本地 Cycle 03 协议。在这版最早论文草稿中，关键不是 OOD 绝对分数，而是 OPD-like 与 SFT 模型在匹配条件下的比较。

### 激活条件谱诊断

对于每个选定层和目标线性模块，我们在探针文本上收集激活统计，并形成白化或激活条件化的权重矩阵。令 $W$ 为模块权重矩阵，$L_{\mathcal{D}}$ 为从探针分布 $\mathcal{D}$ 上的 activation Gram matrix 得到的 Cholesky factor。我们分析以下矩阵的奇异谱：

$$
A^{\mathcal{D}} = W L_{\mathcal{D}}
$$

该谱刻画了模块在探针分布活跃方向上的作用强度。

给定奇异值 $\sigma=(\sigma_1,\ldots,\sigma_n)$，effective rank 定义为：

$$
\operatorname{erank}(\sigma)=\exp\left(-\sum_i p_i \log p_i\right), \quad
p_i=\frac{\sigma_i}{\sum_j \sigma_j}
$$

相对起始 checkpoint 的漂移定义为 log-spectrum 距离的 RMS：

$$
\operatorname{drift}(\sigma,\sigma^0)
=\sqrt{\frac{1}{L}\sum_{i=1}^L
\left(\log(\sigma_i+\epsilon)-\log(\sigma_i^0+\epsilon)\right)^2}
$$

我们还跟踪 spectral gap，以及冻结参考探针 $X$ 与每个模型训练目标探针 $S$ 之间的有符号 log-spectrum level gap。

### 匹配增益稳定性诊断流程

1. 训练共同 cold-start checkpoint $\theta_0$。
2. 从 $\theta_0$ 训练 OPD-like 分支 $\{\theta_o\}$。
3. 从 $\theta_0$ 训练 SFT 对照 $\{\theta_s\}$。
4. 评估 GSM8K，并计算 $g(\theta)=m(\theta)-m(\theta_0)$。
5. 对每个 OPD-like 分支 $\theta_o$，选择：

   $$
   \theta_s^\star = \arg\min_{\theta_s}|g(\theta_o)-g(\theta_s)|
   $$

6. 如果 $|g(\theta_o)-g(\theta_s^\star)|\leq 0.02$，标记为 valid match。
7. 在 valid match 下比较 OOD-lite 退化。
8. 在 valid match 下比较激活条件谱漂移。

## 5. 初步实验

### 配置

表 1 总结当前最小运行配置。该设计有意保持较小规模：它的目标是在运行更大实验前，验证 OPD-like 训练、评估、匹配和几何分析的端到端流程是否能够闭合。

| 项目 | 取值 |
|---|---|
| Student | Qwen3-1.7B |
| Teacher | Qwen3-4B |
| Cold start | 512 prompts, OPD-like distillation |
| OPD-like arm | $\lambda=1.0$, 800 rollout prompts |
| SFT controls | 256, 512, 1024, 2048 examples |
| Main eval | GSM8K, limit 200 |
| Geometry | Layer 14, 7 target modules |
| Probe length | 512 tokens |
| LoRA | rank 16, alpha 32, dropout 0.05 |

表 1：最小 Cycle 03 配置。

### 任务轨迹

表 2 给出任务轨迹。最重要的观察是，在这个 setup 下，更大的 SFT 对照会使 GSM8K 退化，而 OPD-like 分支大致保持了起始 checkpoint 的 GSM8K 分数。由于评估限制只有 200，OPD-like 分数应被解释为“保持”，而不是“提升”。

| 模型 | Size | GSM8K | Gain | OOD $p2$ |
|---|---:|---:|---:|---:|
| SFT-1024 | 1024 | 0.295 | -0.140 | 0.0261 |
| SFT-2048 | 2048 | 0.310 | -0.125 | 0.0279 |
| SFT-512 | 512 | 0.380 | -0.055 | 0.0124 |
| OPD-$\lambda1$ | 800 | 0.430 | -0.005 | 0.0100 |
| $\theta_0$ | 512 | 0.435 | 0.000 | 0.0000 |
| SFT-256 | 256 | 0.440 | 0.005 | 0.0182 |

表 2：评估轨迹。Size 对 OPD-like 训练表示消耗的 rollout prompts，对 SFT 表示监督样本数。

### 匹配增益比较

OPD-$\lambda1$ 最近的 SFT match 是 SFT-256。GSM8K 增益差为：

$$
|-0.005 - 0.005| = 0.010
$$

该差异满足 valid-match 阈值。表 3 给出匹配后的 OOD-lite 比较。在匹配 GSM8K gain 的条件下，OPD-like 模型具有更低的 OOD-lite penalty 和更低的 worst OOD-lite drop。

| 指标 | OPD | SFT-256 | OPD--SFT |
|---|---:|---:|---:|
| GSM8K gain | -0.005 | 0.005 | -0.010 |
| OOD-lite penalty $p2$ | 0.0100 | 0.0182 | -0.0082 |
| Worst OOD-lite drop | 0.0100 | 0.0150 | -0.0050 |

表 3：匹配增益比较。对退化指标而言，负的 OPD--SFT 数值表示 OPD 更优。

### 谱稳定性

表 4 总结 layer-14 几何。OPD-like 模型的谱漂移为 0.0179，而 SFT-256 为 0.0488。随着监督训练数据量增加，SFT 对照显示出逐渐增加的漂移。

OPD-like 模型还使 $X$--$S$ spectrum level gap 更接近 0，这表明在 OPD-like 更新下，参考生成分布与训练目标分布之间保持了更高的一致性。

| 模型 | Effective rank | Drift | $X$--$S$ gap |
|---|---:|---:|---:|
| $\theta_0$ | 897.29 | 0.0000 | -0.0231 |
| OPD-$\lambda1$ | 892.66 | 0.0179 | -0.0185 |
| SFT-256 | 925.28 | 0.0488 | -0.0467 |
| SFT-512 | 927.27 | 0.0527 | -0.0467 |
| SFT-1024 | 930.36 | 0.0598 | -0.0479 |
| SFT-2048 | 930.84 | 0.0643 | -0.0492 |

表 4：Layer-14 激活条件谱总结，数值为七个目标模块平均。

模块级结果比平均值更强。如表 5 所示，OPD-like 训练在每个测量的 layer-14 模块上都比 SFT-256 有更低漂移。最大绝对差异出现在 attention output projection 以及 MLP gate/up projections。

这一模式与如下假设一致：continued SFT 将模型推向较窄的答案风格流形，而 OPD-like 训练执行的是更温和的分布对齐更新。

| 模块 | OPD | SFT-256 | Diff. |
|---|---:|---:|---:|
| attn.q | 0.0165 | 0.0391 | -0.0226 |
| attn.k | 0.0137 | 0.0355 | -0.0218 |
| attn.v | 0.0137 | 0.0357 | -0.0220 |
| attn.o | 0.0328 | 0.0792 | -0.0464 |
| mlp.gate | 0.0162 | 0.0597 | -0.0435 |
| mlp.up | 0.0158 | 0.0597 | -0.0439 |
| mlp.down | 0.0167 | 0.0324 | -0.0157 |

表 5：Layer-14 各模块相对 $\theta_0$ 的漂移。负差值表示 OPD drift 更低。

## 6. 讨论

初步结果支持一个保守版本的 K1 假设：在匹配同分布推理增益的条件下，OPD-like 训练可以比 continued SFT 产生更低的 OOD-lite 退化和更低的激活条件谱漂移。

匹配增益要求非常重要。没有这一要求，更低的 drift 可能被解释为 OPD 只是训练得更少。在这里，OPD-$\lambda1$ 和 SFT-256 的 GSM8K gain 很接近，但它们的 OOD-lite 和谱特征明显不同。

该结果也明确了我们还不知道什么。第一，OPD-like 训练并没有可靠地提升 GSM8K 超过 $\theta_0$。第二，起始 checkpoint 可能已经饱和：如果 $\theta_0$ 在当前有限评估下已经接近局部 GSM8K 上限，就很难检测改进。第三，几何证据目前只是 layer-14 证据。模块级一致性令人鼓舞，但机制声明仍需要多层证据，最好还需要奇异向量或 principal-angle artifact。

这些限制并不削弱当前结果的用途，而是明确了下一步实验的必要性。更严格的 follow-up 应该使用更不饱和的 $\theta_0$，增加 GSM8K 评估样本数或移除限制，加入至少一个额外 OPD 设置或 seed，并测量 early、middle、late 多层几何。

如果稳定性模式仍然存在，这项工作可以支持一个更强的声明：OPD-like state matching 相对于 continued SFT 能控制破坏性漂移。如果同一 setup 还能产生可靠 GSM8K gain，那么声明就可以从“稳定性”推进到“提升 + 稳定性”。

## 7. 局限性

这版最早草稿还有若干必须在 AAAI 投稿质量版本前解决的限制。

第一，实验只有一个 OPD-like 运行。第二，GSM8K 的 `eval_limit=200` 使小差异不可靠。第三，OOD-lite suite 作为本地退化探针是有用的，但不能建立广泛 OOD 泛化。第四，谱分析只限于 layer 14，并且尚未保存奇异向量 sketch 或 principal-angle 分析所需 artifact。第五，当前实现是 TRL OPD-like，而不是 verl 等标准 OPD 系统的完整复现。最后，当前证据支持的是匹配表现下的稳定性，而不是一般性能提升声明。

## 8. 结论

本文提出了一个最小匹配增益诊断，用于比较 OPD-like 训练与 continued SFT。在一个初步的 Qwen3-1.7B/Qwen3-4B 实验中，OPD-like 训练在 GSM8K gain 上与最近 SFT 对照匹配，同时表现出更低的 OOD-lite 退化和显著更低的激活条件谱漂移。

该结果最好被视为可行性与稳定性信号，而不是最终 OPD 性能声明。下一道研究 gate 已经明确：在更低评估噪声、多个 OPD 设置、更不饱和的起始 checkpoint 和多层几何条件下，检验同样的匹配增益稳定性模式是否成立。

## 参考文献占位

中文版本沿用英文稿的 `references.bib`。对应核心来源包括：

- Agarwal et al., On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.
- Li et al., Rethinking On-Policy Distillation of Large Language Models.
- Song and Zheng, A Survey of On-Policy Distillation for Large Language Models.
- Lobo et al., On the Impact of Fine-Tuning on Chain-of-Thought Reasoning.
- Shuttleworth et al., LoRA vs Full Fine-tuning: An Illusion of Equivalence.
- Yang et al., OPRD: On-Policy Representation Distillation.

