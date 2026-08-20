# 构念比较、主图合并与 RR5 表格调整

日期：2026-07-29

状态：讨论方案已基本收敛，等待 RR5 新结果；在收到明确的“编写论文”指令前，不据此修改正文、Supplement、图片或压缩包。

## 1. 本轮修改目标

本轮讨论解决的是论文前两个结果小节之间的衔接：

1. 先比较纯激活、纯权重与域条件功能谱分别观察到了什么；
2. 再利用同一张图说明 OPD 的共同早期功能压缩；
3. 用紧凑的表格保留原逐域早期边际图的证明作用；
4. 将训练设置识别与区域输出预测拆成两个不同的统计任务；
5. 补齐 Qwen 的严格共同网格 RR5 结果，使双模型比较协议一致。

这轮调整不改变域条件功能谱的理论定义、低秩输出误差解释、exposure/refresh 控制、matched-support 结果或 signed readout 的主结论。

## 2. Figure 1：合并构念比较与功能压缩轨迹

### 2.1 最终布局

将原计划的构念比较图与功能压缩主轨迹图合并为一张 `2 x 4` 双栏图：

| 模型 | Activation ER | TPNT | PABS | Functional contraction \(C_5\) |
|---|---|---|---|---|
| Qwen | (a) | (b) | (c) | (d) |
| Llama | (e) | (f) | (g) | (h) |

其中：

- 第一行展示 Qwen；
- 第二行展示 Llama；
- 四列分别展示 activation ER、TPNT、PABS 和 equal-5 功能压缩；
- 所有面板使用一致的训练设置颜色和 checkpoint 横轴约定；
- 不同指标保留各自有意义的纵轴，图注必须说明不能跨列比较原始数值幅度；
- 数据缺少某些 checkpoint 时只画实际观测点，不插值；
- \(p_k\) 不放入 Figure 1，避免与后面的区域 KL/NLL 预测比较重复。

### 2.2 两个结果小节如何使用同一张图

第一个结果小节引用 Figure 1(a--c, e--g)，讨论：

- ER 表示纯激活支撑空间的维数变化；
- TPNT 表示更新对 source principal coordinates 的覆盖及其相对 null 的富集；
- PABS 表示 source 与当前权重的主奇异子空间对齐；
- 三者分别观察激活、更新覆盖和权重基底旋转，但不能单独表达当前权重如何作用于域上实际访问的输入方向；
- 功能谱提供的是权重与输入访问共同决定的本层功能模式。

第二个结果小节引用 Figure 1(d, h)，讨论：

- 两个模型中 OPD 都在共同早期窗口进入更深的功能压缩状态；
- Qwen 与 Llama 的完整时间形态不同，因此不主张统一的峰值、过冲或恢复曲线；
- 后续 NCD 只概括深度与持续时间，不替代早期逐域排序。

### 2.3 Figure 1 的叙事目的

这张图不是方法结构示意图。它要让读者在同一批训练轨迹上直接看到：

> ER、TPNT、PABS 与功能谱观察的是不同层面的变化，而功能谱呈现了更清楚的训练设置分离。

Figure 1 主要承担轨迹形态和构念差异的展示，统计识别能力由 Table 1 承担。

## 3. 原早期边际 panel 的处理

原功能压缩图中的逐域连续边际 panel 不再保留在正文主图中。

该 panel 的独立视觉价值有限，但具有重要的稳健性作用：它证明 OPD 的早期压缩排序不是对多个域取平均后产生的假象。处理方式如下：

1. 完整逐模型、逐域、逐 checkpoint 的连续边际图移入 Supplement；
2. 正文保留一句自然语言结论并引用 Supplement；
3. 将最关键的证明量 \(M_{\min}^{\mathrm{early}}\) 与 NCD 合并到正文表格中。

早期连续边际定义为

\[
M_{m,D,t}
=
\min_{b\in\{\mathrm{SFT,offKD,seqKD}\}}
\Delta r_{m,b,D,t}^{(5)}
-\Delta r_{m,\mathrm{OPD},D,t}^{(5)}.
\]

正文表格使用

\[
M_{\min,m}^{\mathrm{early}}
=
\min_{D,\;t\in\{20,40,80\}} M_{m,D,t}.
\]

只要 \(M_{\min,m}^{\mathrm{early}}>0\)，就严格说明该模型在所有纳入比较的核心域和共同早期 checkpoint 上都没有发生 OPD 排序反转。这比在正文中使用“若干项中有若干项成立”的计数写法更简洁。

## 4. 合并早期排序与 NCD 的正文表格

建议将当前 NCD 表转置并扩展为：

| Model | \(M_{\min}^{\mathrm{early}}\) | OPD NCD | SFT NCD | off-KD NCD | seqKD NCD |
|---|---:|---:|---:|---:|---:|
| Qwen | 待从正式结果填入 | **50.423** | 9.554 | 31.017 | 37.793 |
| Llama | 待从正式结果填入 | **77.594** | 39.683 | 17.518 | 11.093 |

表注建议表达为：

> 共同早期排序与全轨迹压缩剂量。\(M_{\min}^{\mathrm{early}}\) 是在四个核心域和共同早期 checkpoint 上取得的最小 OPD 连续压缩边际；正值说明该范围内不存在被域平均掩盖的排序反转。NCD 在 \(\log(1+t)\) 时间上积分至 \(T=320\)，衡量基线以下压缩的深度与持续时间。前者以 directions 为单位，后者以 directions\(\times\)log-step 为单位，数值尺度不可直接比较。

注意：

- \(M_{\min}^{\mathrm{early}}\) 回答局部早期排序；
- NCD 回答完整轨迹的压缩深度与持续时间；
- 二者可以放在同一张表中，但必须明确单位不同，不能直接比较数值大小；
- 四个 \(\varepsilon\) 的完整稳健性以及高阈值 IFEval 的局部例外放入 Supplement，不挤入主表。

## 5. Table 1：严格共同网格上的训练设置识别

### 5.1 表格任务

Table 1 独立回答：

> 在完全相同的 cells 和 checkpoint-held-out folds 上，不同观察量能够多好地识别训练设置，尤其是 OPD？

推荐最终列为：

| Model | Activation suite \(A\) | TPNT | PABS | Functional \(C_5\) | Weight \(P_{k,5}\) |
|---|---:|---:|---:|---:|---:|
| Qwen | 待 RR5 | 待严格共同网格 | 待严格共同网格 | 待 RR5 | 待 RR5 |
| Llama | .556 | 待严格共同网格 | 待严格共同网格 | .743 | .688 |

表中的数值均应为 checkpoint-held-out AUC。TPNT 和 PABS 只有在使用相同 cells、相同 folds 和无信息泄漏的特征选择协议时才能进入此表。

### 5.2 Activation suite \(A\) 的准确含义

\(A\) 不是 ER、TPNT 和 PABS 的合集，也不是 ER 单项。

它是由八项纯激活特征组成的 raw-activation suite：

1. normalized entropy effective rank；
2. participation ratio；
3. top-1 explained-energy share；
4. top-8 explained-energy share；
5. top-32 explained-energy share；
6. raw anisotropy；
7. centered anisotropy；
8. relative-to-step-0 linear CKA。

因此，已有的 Llama `.556` 是整个 activation suite \(A\) 的 held-out AUC，不能写成“ER 的 AUC”。

表头必须写成 `Activation suite \(A\)`，不能只写 `A`。表注需要完整解释：

> \(A\) 表示由八项纯激活特征组成的 activation suite，包括归一化 entropy effective rank、participation ratio、top-1/top-8/top-32 解释能量占比、raw 与 centered anisotropy，以及相对 step 0 的 linear CKA。TPNT 和 PABS 是独立的权重空间特征组，不属于 \(A\)。

英文表注可写为：

> \(A\) denotes the eight-feature raw-activation suite: normalized entropy effective rank, participation ratio, top-1/top-8/top-32 explained-energy shares, raw and centered anisotropy, and linear CKA relative to step 0. TPNT and PABS are evaluated as separate weight-space feature blocks rather than components of \(A\).

### 5.3 Figure 1 与 Table 1 中 ER/A 的区别

- Figure 1 使用 ER 作为纯激活轨迹的代表性、可解释展示；
- Table 1 使用完整 activation suite \(A\) 进行样本外训练设置识别；
- 正文和图注必须明确这一差别，避免读者误以为 Figure 1 的 ER 单项直接产生了 Table 1 的 \(A\) 列。

### 5.4 TPNT/PABS 的公平比较要求

若 Table 1 加入 TPNT 和 PABS：

- 必须与 \(A\)、\(C_5\)、\(P_{k,5}\) 使用完全相同的 model/setting/checkpoint/probe cells；
- 必须使用相同 checkpoint-held-out folds；
- 标准化、特征筛选和正则化只能在训练 fold 内完成；
- TPNT 的 \(k,\alpha\) 和 PABS 的 \(k\) 不能根据全数据结果事后选择；
- 可以预先登记一个固定配置，也可以把多组配置作为 feature block，但后者仍需在 fold 内处理；
- 不能用 D11 的 descriptive parity 数值替代 RR5 的严格共同网格样本外结果。

## 6. RR5 当前需要补齐什么

原 RR5 的直接缺口是 Qwen 的严格共同网格：

\[
A,\qquad C_5,\qquad P_{k,5}.
\]

这项缺口目前直接阻碍 Table 1 的 Qwen 行和第一个结果小节中的双模型样本外总结，不阻碍：

- Figure 1 中已有的双模型 ER/TPNT/PABS 轨迹；
- OPD 功能压缩主轨迹；
- 区域 KL/NLL 输出预测表；
- exposure、refresh 和 matched-support 控制。

本轮最终决定是：补 RR5 时同时构造 TPNT 与 PABS 的严格共同网格 feature blocks。Table 1 最终比较双模型

\[
A/\mathrm{TPNT}/\mathrm{PABS}/C_5/P_{k,5}.
\]

这样 Table 1 才能完整对应 Figure 1 中展示的三类相关工作指标，而不是只比较 \(A/C_5/P_{k,5}\)。如果 TPNT/PABS 因 exact-key join 无法进入完全相同网格，不能用 D11 descriptive parity 数值填入正式表；应在记录中明确缺失原因，再决定是否将两列降入 Supplement。

## 7. 区域输出预测表

训练设置识别和区域输出预测必须拆成两张表，不能继续放在同一张多 panel 表中。

区域输出预测表：

- 同时展示 Qwen 和 Llama；
- 比较 equal-5 功能压缩 \(C_5\) 与最佳 \(P_{k,5}\)；
- 使用相同状态、模块集合和 checkpoint-held-out folds；
- 主要回答哪个内部坐标更能预测区域级 full-vocabulary KL；
- \(p_k\) 在这里作为强权重基线得到详细展示；
- 现有双模型区域结果不依赖新 RR5，可以独立保留。

Figure 1 不展示 \(p_k\)，Table 1 只给其训练设置识别摘要；\(p_k\) 与功能谱的任务依赖互补性主要在区域 KL/NLL 小节展开。

## 8. 正文各部分的具体影响

### 8.1 相关工作

需要平行说明：

- ER：纯激活空间的维度；
- TPNT：更新对 source principal coordinates 的覆盖；
- PABS：权重主奇异子空间旋转；
- \(p_k\)：更新写入 source 主子空间的位置。

这里只介绍研究对象和已有方法，不提前报告实验胜负。

### 8.2 实验设置/比较坐标

正文明确给出 ER、TPNT、PABS 和 \(p_k\) 的核心定义与直观含义。NSS、CKA、PR、anisotropy 等其余指标概括后放入 Supplement。

需要说明 Table 1 的严格共同网格、equal-5 聚合、checkpoint-held-out folds，以及 Figure 1 的 ER 与 Table 1 的 activation suite \(A\) 不是同一个特征粒度。

### 8.3 第一个结果小节

重写为“功能谱提供纯激活与纯权重坐标之外的训练轨迹信息”，按 ER、TPNT、PABS、功能谱的顺序形成完整构念比较。

这一节不能写成功能谱普遍优于全部权重指标。更准确的结论是：

> 功能谱在本文轨迹和无符号输出移动任务上提供了更清楚的训练结构；权重位置指标在部分训练设置识别或 signed readout 任务中仍具有互补价值。

摘要不需要加入单个模型上 \(p_k\) 的局部优势。

### 8.4 第二个结果小节

使用 Figure 1(d, h) 说明 OPD 的共同早期压缩，随后用合并后的 \(M_{\min}^{\mathrm{early}}+\mathrm{NCD}\) 表分别概括早期排序与全程剂量。

完整逐域边际图、阈值敏感性、层级和模块级结果转入 Supplement。

### 8.5 后续输出小节

区域 KL 小节使用独立预测表详细比较 \(C_5\) 和 \(P_{k,5}\)。signed NLL 与真实行为小节继续强调模型与训练目标依赖，不把无符号移动预测写成行为方向预测。

## 9. Supplement 需要登记的内容

至少包括：

1. 原正文早期连续 margin panel 的完整逐模型、逐域、逐 checkpoint 版本；
2. 四个 \(\varepsilon\) 下的早期排序与局部例外；
3. activation suite \(A\) 的八项定义、计算和完整轨迹；
4. TPNT 的不同 \(k,\alpha\)、null 构造和标准化结果；
5. PABS 的不同 \(k\)、层级和投影类型结果；
6. NSS 及其余 activation-only/weight-only audits；
7. RR5 的 exact-key join、缺失 cell 处理、fold 构造、标准化和嵌套选择协议；
8. equal-5 与 equal-7 的敏感性；
9. Figure 1 中没有展开的所有层、模块和 checkpoint。

## 10. 最终统一编写时的执行顺序

1. 等待并核验 RR5 新结果，首先确认 exact common cells 和 held-out 协议；
2. 决定 Table 1 使用最低版本还是包含 TPNT/PABS 的完整版本；
3. 生成 `2 x 4` Figure 1；
4. 将完整早期 margin 图移入 Supplement；
5. 计算并填入两个模型的 \(M_{\min}^{\mathrm{early}}\)；
6. 合并早期 margin 摘要与 NCD 表；
7. 重写比较坐标说明和前两个结果小节；
8. 拆分训练设置识别表与区域输出预测表；
9. 补充 Supplement 的完整协议、稳健性和全量图表；
10. 同步更新中英文正文、Supplement、图号、表号和交叉引用；
11. 编译检查页数、浮动位置、字体、矢量图和引用；
12. 最后统一更新 Overleaf 上传目录与 ZIP。

## 11. 写作约束

- 中文避免使用含义模糊的“臂”，统一写“训练设置”或“训练条件”；
- 不使用 `xx/xx` 的计数式表述来代替自然语言结论；
- 不使用 “headline”；
- 不把 Figure 1 的 ER 与 Table 1 的 activation suite \(A\) 混为一谈；
- 不把 D11 descriptive parity 写成 RR5 严格样本外证据；
- 不把 \(M_{\min}^{\mathrm{early}}\) 与 NCD 的数值直接比较；
- 不声称功能谱普遍支配所有权重坐标；
- 正文保留主结论与最关键证据，完整配置、层级、模块、阈值和轨迹进入 Supplement。

## 12. 当前执行边界

本文件是后续统一修改的需求记录，不代表论文已经按此方案完成。

此前中英文 `main` 文件中存在一次未经授权的 Table 1 占位与拆表改动。该部分不视为本轮方案已经实施；收到“编写论文”指令后，应依据本记录和届时最新 RR5 结果重新核对，而不是直接沿用该临时占位。

## 13. 补录范围：从“为什么只介绍 \(p_k\)”开始的最终决定

本文件应完整覆盖从以下问题开始的全部有效讨论：

> 为什么实验部分只介绍了 \(p_k\)，而没有介绍其他实际参与比较的相关工作指标？

从该问题之后，最终保留的决定如下：

1. 正文不能只定义 \(p_k\)，否则读者无法判断 ER、TPNT、PABS、NSS 等结果到底比较了什么；
2. ER、TPNT、PABS 与 \(p_k\) 是四个一级相关工作指标，正文必须给出核心定义和观察对象；
3. NSS、PR、top-share、anisotropy、CKA、raw update energy 等作为二级审计，正文概括，完整计算放 Supplement；
4. 第一个结果小节围绕 ER、TPNT、PABS 与功能谱的构念差异展开；
5. \(p_k\) 是较强的权重位置基线，不与 TPNT/PABS 一起写成“都没有信息”，而是在训练设置识别表中给出摘要，并在区域 KL/NLL 小节详细比较；
6. 原先构想的 Figure 1 方法结构示意图被放弃，改为真实双模型轨迹；
7. Figure 1 与功能压缩主图最终合并为 `2 x 4`；
8. 原早期逐域 margin panel 从正文图中移除，完整图进入 Supplement，最小 margin 与 NCD 合表；
9. 训练设置识别和区域输出预测是两个不同任务，必须拆成两张表；
10. Table 1 最终比较双模型 \(A/\mathrm{TPNT}/\mathrm{PABS}/C_5/P_{k,5}\)；
11. Table 1 中的 \(A\) 是八维 raw activation suite，不是 ER 单项，也不包含 TPNT/PABS；
12. Qwen 需要补严格共同网格 RR5；D11 只能作为 descriptive parity，不能代替正式样本外结果；
13. 正文引用单独提交的补充材料时统一使用 `Supplementary Material, Sec. X` 或 `Supplementary Sec. X`，不使用 `Appendix X`。

未列入上述清单的中间构图方案、重复表格方案和仅展示 Llama 的临时方案均视为已放弃。

## 14. 参与比较的指标总表

下表记录实际进入现有结果或稳健性结论的主要指标、计算对象、当前正文介绍状态以及最终写作位置。这里的“当前状态”是讨论发生时的论文状态，不表示最终版本应保持不变。

| 指标 | 观察对象与核心计算 | 在结果中的作用 | 当前正文是否充分定义 | 最终决定 |
|---|---|---|---|---|
| \(r_\varepsilon(W_tS_{D,t})\) | 保留 \(1-\varepsilon\) 功能能量所需的最小方向数 | 本文主状态量 | 是 | 理论与方法正文完整保留 |
| \(c_\varepsilon^{(5)}\)、\(C_5\) | 逐模块相对自身基线的功能压缩，再对 v/o/gate/up/down equal-5 | 轨迹、区域输出和 held-out 比较的主特征 | 基本充分 | 正文主结果；equal-7 为 Supplement sensitivity |
| Activation ER | 对 centered hidden-state covariance 的归一化特征值 \(q_i\) 计算 \(\exp(-\sum_iq_i\log q_i)\)，必要时再除 hidden width | Figure 1 的纯激活代表轨迹 | 仅名称和结果，公式不足 | 正文给核心公式；完整 activation suite 放 Supplement |
| Activation PR | \((\sum_i\lambda_i)^2/\sum_i\lambda_i^2\) | raw activation robustness | 否 | 正文一句列出，公式和全结果放 Supplement |
| Top-1/8/32 share | covariance 前 \(k\) 个特征值的能量占比 | raw activation suite \(A\) | 否 | Table 1 表注说明属于 \(A\)，细节放 Supplement |
| Raw/centered anisotropy | hidden representations 的方向集中程度 | raw activation suite \(A\) | 否 | Table 1 表注概括，定义放 Supplement |
| Linear CKA | \(\|X^\top Y\|_F^2/(\|X^\top X\|_F\|Y^\top Y\|_F)\) | 与 step 0 的表示方向相似性 | 仅名称和结果 | Table 1 表注说明属于 \(A\)，公式放 Supplement |
| Activation suite \(A\) | normalized ER、PR、top-1/8/32 share、raw/centered anisotropy、step-0 CKA 共八项 | 严格共同网格的 activation-only feature block | 没有解释 \(A\) 的组成 | Table 1 表头与表注必须完整定义 |
| \(p_k\)、\(P_{k,5}\) | \(\|U_k^\top\Delta WV_k\|_F^2/\|\Delta W\|_F^2\)，再按 equal-5 聚合或组成 \(k\)-block | 强 source-principal 权重位置基线；训练识别和区域输出预测 | 正文已有公式 | Table 1 摘要，区域 KL/NLL 小节详细比较 |
| TPNT | source rank-\(k\) 重建的 top-\(\alpha\) coordinate mask 与 update mask 的 coverage/lift；另有 spectrum-matched null \(z_{\rm TPNT}\) | 更新是否富集在 source principal coordinates | 仅名称，未给计算 | 正文给核心定义；\(k,\alpha\)、null 和全量结果放 Supplement |
| PABS | source/current top-\(k\) 左右奇异子空间 principal-angle cosine 的平均 | 权重主奇异子空间旋转 | 仅名称，未给计算 | 正文给 scalar 定义；完整 rank/layer 结果放 Supplement |
| NSS | 两端 sum-normalized top-32 奇异值谱的 L1/L2 距离 | 权重奇异值谱漂移 | 仅名称，未给计算 | 正文随 PABS 简述，公式与结果放 Supplement |
| Raw update energy | \(\|\Delta W\|_F^2\) 的逐模块聚合 | 参数移动幅度审计 | 未正式定义 | 不升为正文一级指标；Supplement 完整报告 |
| Activation-weighted update | \(\|\Delta WS_D\|_F^2\)，current 或 fixed whitening | 更新在域输入度量下的局部输出移动 | 理论讨论中不充分 | 作为 Supplement 构念边界，不与 state rank 混同 |
| Stable rank | \(\|A\|_F^2/\sigma_1^2\) | 功能谱连续统计稳健性 | 正文仅报告名称 | 主结果一句，定义和曲线放 Supplement |
| Functional entropy rank | \(\exp(-\sum_i p_i\log p_i)\)，\(p_i\) 为功能奇异能量占比 | 不依赖 \(\varepsilon\) 的连续谱稳健性 | 正文仅报告名称 | 主结果一句，定义和曲线放 Supplement |
| \(M_{m,D,t}\) | OPD 相对最接近离线训练设置的连续压缩边际 | 逐域验证早期 OPD 排序 | 正文已有定义 | 完整曲线移入 Supplement；正文表保留 \(M_{\min}^{\rm early}\) |
| NCD | 在 \(\tau=\log(1+t)\) 上积分基线以下压缩 | 汇总全轨迹深度与持续时间 | 有文字解释 | 与 \(M_{\min}^{\rm early}\) 合并为正文表 |
| Regional KL | base 到 checkpoint 的区域 full-vocabulary \(D_{\rm KL}(p_0\Vert p_t)\) | 无符号输出 departure | 是 | 独立输出预测表，比较 \(C_5\) 与 \(P_{k,5}\) |
| Signed/absolute NLL | gold token likelihood 的方向性或幅度变化 | 区分 output movement 与 readout valence | 是 | signed readout 小节保留，不能由 KL 结果替代 |

### 14.1 四个一级相关工作指标的关系

正文应给读者一个明确的构念地图：

- ER 回答激活访问空间是否收缩或铺开；
- TPNT 回答更新坐标是否富集在 source principal mask；
- PABS 回答 source 与 checkpoint 的主奇异子空间是否旋转；
- \(p_k\) 回答更新能量有多少同时写入 source top-\(k\) 左右奇异子空间；
- \(r_\varepsilon(W_tS_{D,t})\) 回答当前权重在指定域实际访问的输入度量下，需要多少功能方向保留主要本层输出能量。

这些量不是同一指标的不同归一化，不能通过比较原始曲线幅度来决定优劣。公平比较必须使用同一 cells、同一 folds 和明确的信息任务。

## 15. 第一个结果小节中文工作草稿

以下草稿记录本轮已经接受的行文逻辑。最终数值和表号必须等待新 RR5 后统一替换，不能直接把占位文字提交。

```latex
\subsection{功能谱提供纯激活与纯权重坐标之外的轨迹信息}

为了判断域条件功能谱是否只是已有几何量的重新表达，我们首先在相同训练设置和 checkpoint 上比较纯激活谱、source-principal 更新坐标以及权重奇异子空间的变化。它们分别回答模型访问了哪些表示方向、更新写入了哪些源权重坐标，以及主权重基底是否发生旋转；本文的功能谱则进一步询问，当前权重在指定输入域实际访问的方向上形成了怎样的本层输出模式。Figure~\ref{fig:construct-trajectories} 将这四类观察量放在同一组 Qwen 与 Llama 轨迹中。

纯激活结果并未复现功能谱中的训练设置分离。Figure~\ref{fig:construct-trajectories}a,e 展示 normalized activation ER：在训练过程中，hidden-state covariance 的总体铺开程度只有较小变化；Llama 的多个外部域上，OPD@160 的相对变化不足 $.003$，step-0 CKA 仍约为 $.9997$--$1.0000$。这说明输入表示访问的总体维数与方向可以基本保持稳定，而相同位置的功能秩已经明显下降。功能压缩因而不能简化为 activation collapse。

纯权重坐标给出另一侧的边界。TPNT 检查更新坐标是否富集在 source 低秩重建的 principal mask 上。Figure~\ref{fig:construct-trajectories}b,f 显示部分 checkpoint 存在高于 spectrum-matched null 的局部富集，但这种富集没有形成跨模型、训练设置和 checkpoint 稳定的 OPD 特异轨迹。PABS 则比较 source 与当前权重 top-$k$ 左右奇异子空间的 principal angles；Figure~\ref{fig:construct-trajectories}c,g 中其数值始终接近 1，配套 NSS 也很小，说明当前 LoRA 设置主要在近乎固定的主权重基底内改变，而不是通过大幅旋转该基底产生功能轨迹。这些结果并不否定 TPNT 或 PABS 的原生用途，而是表明更新覆盖和权重基底旋转不足以单独解释本文观察到的轨迹结构。

与之相比，Figure~\ref{fig:construct-trajectories}d,h 中的 equal-5 相对功能压缩清楚地区分了训练设置。这里的差别来自 $W_t$ 与域输入二阶矩的共同作用：纯权重指标不知道输入域实际访问哪些方向，纯激活指标也不知道当前权重如何放大、压低或组合这些方向；$W_tS_{D,t}$ 则把二者写入同一个具有本层输出能量含义的坐标。因此，功能谱提供的不是更多同类特征，而是介于参数位置和最终输出之间的功能状态。

我们进一步在严格共同 cells 上进行 checkpoint-held-out 训练设置识别。Table~\ref{tab:identification} 比较完整 raw activation suite $A$、TPNT、PABS、功能压缩 $C_5$ 与 source-principal block $P_{k,5}$。其中 $A$ 包含 normalized ER、PR、top-1/8/32 explained-energy shares、raw/centered anisotropy 和 relative-to-step-0 linear CKA，而不是 Figure~\ref{fig:construct-trajectories} 中 ER 单项的另一名称。[此处等待 Qwen RR5 与最终双模型统计后填写结果。] \(p_k\) 是本实验中的强权重位置基线；它与功能谱在训练设置识别和后续输出任务上提供任务依赖的互补信息，区域 KL/NLL 的详细比较将在后文展开。
```

### 15.1 该草稿保留的写作边界

- 第一小节内部可以明确说功能谱在本文轨迹上表现出显著区分能力；
- 不把 TPNT/PABS 写成普遍无效，只描述本项目 LoRA/deployed-BF16 设置下的结果；
- 不把 \(p_k\) 与 TPNT/PABS 一起概括为弱基线；
- 不在这里展开 \(p_k\) 的区域 KL/NLL 胜负，避免与后续 subsection 重复；
- 不把 Activation suite \(A\) 误写成 ER；
- 不根据不同量纲的视觉幅度比较指标优劣；
- Qwen 与 Llama 的 ER 纵轴可以各自解释，但不能把不同采集协议下的绝对幅度直接作跨模型比较。

## 16. Appendix 与 Supplementary Material 的最终称呼

AAAI AuthorKit 区分两类内容：

1. `Content Appendices`：接在主论文正文之后，属于主论文，计入页数限制；
2. `Supplementary Material / Technical Appendices`：是否允许及是否独立提交由当年会议政策决定。

本文已经决定让正文在 Conclusion 结束，并将额外证明、协议和完整实验作为单独 PDF 提交。因此采用第二类。

正文中的统一写法是：

- 首次或较正式：`Supplementary Material, Sec. A.1`；
- 后续简写：`Supplementary Sec. A.1`；
- 多节：`Supplementary Secs. C.2--C.3`。

中文版工作稿也沿用上述英文标签，避免最终翻译时改变交叉引用对象。

正文不要写：

- `Appendix A.1`；
- `see Appendix`；
- `附录 A.1`。

这些写法容易让读者理解为 appendix 与主论文在同一个 PDF 中。单独的 Supplement 可以继续使用 A、B、C、D 作为内部 section 编号，但正文应把它们称为 Supplementary sections。跨 PDF 引用通常是文本引用，不依赖主论文中的 LaTeX `\ref` 自动跳转。
