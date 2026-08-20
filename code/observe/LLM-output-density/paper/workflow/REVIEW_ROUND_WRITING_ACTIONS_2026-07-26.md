# 本轮审稿反馈：论文编写行动台账

更新时间：2026-07-26

## 一、用途与边界

本文档只管理本轮反馈中的论文编写任务：论证结构、主张措辞、方法说明、图表组织、相关工作和中英文同步。理论定义、实验设计、统计计算和新结果仍由事实层统一管理：

`mypaper/theory/human_read-ch.md`

本文件不是新的事实来源。任何新数字、新对照或主张强度变化，都必须先进入 `human_read-ch.md`，再进入中文稿和英文稿。

## 二、本轮共同判断

两类 reviewer 的意见可以归纳为：

1. `r_\varepsilon(W_tS_{D,t})` 具有正确、清楚的模块局部解释。
2. 创新不是发明 activation-aware 低秩算子，而是把它重构为 checkpoint-wise、domain-conditioned 的后训练诊断量。
3. OPD/off-KD、off-KD/seqKD 和 `alpha=.5` 比单纯 OPD--SFT 更有信息量。
4. 功能秩轨迹与行为读出可以分离，是可信且有信息量的经验发现。
5. 当前拒稿风险主要来自实验闭环、统计可信度和复现信息，而不是 idea 没有意义。

因此，论文编写不再大幅更换核心问题。目标是把已经成立的 idea 写成边界清楚、结果闭合、能够复现的投稿成品。

## 三、冻结的正面 claim architecture

论文的一句话定位应以发现结束，而不是以免责声明结束：

> 本文将激活感知低秩近似中的域加权算子重构为 checkpoint-wise 模块功能状态量，并通过匹配的多臂后训练轨迹揭示 OPD 在不同模型上共享的跨域功能压缩制度，以及该内部重组与输出分布移动和行为读出之间的结构关系。

全文只维护四个一级 claim：

1. **Functional observable。** `r_\varepsilon(W_tS_{D,t})` 将域输入统计和当前权重组合成一个可沿 checkpoint 追踪的模块功能状态量。
2. **Cross-model regime。** OPD 在 Qwen 与 Llama 上均呈现最强的早期跨域功能压缩和最大的共同 horizon 压缩暴露，具体时间形态依模型而异。
3. **Exposure evidence。** same-KL OPD--offKD 与 `alpha=.5` 干预把该压缩制度定位到 current-self generation pipeline，而不是简单的 hard/soft label 区别。
4. **Output and readout structure。** 相对功能压缩与输出分布移动共同演化；matched-support 结果进一步显示功能维数与行为读出是两个可分离但互补的后训练坐标。

其中 Claim 1--2 是论文不可动摇的核心，Claim 3--4 的具体强度由 `human_read-ch.md` 的最终证据状态决定。限制不再与每个 claim 绑定重复出现，而集中放置：

| Claim | 正文的主动表述 | 边界只出现的位置 |
|---|---|---|
| Functional observable | 模块功能状态量及其精确输出误差意义 | Method 定义后一句，说明 module-local 范围 |
| Cross-model regime | 两模型共享压缩制度、时间实现不同 | Results 统计单位一句；seed 问题只进 Limitations |
| Exposure evidence | current-self pipeline 的方向性证据 | Discussion 一次说明 pipeline bundle，不逐段道歉 |
| Output/readout structure | 内部功能状态、输出移动和读出可分离 | matched-support 结果一次；不再命名为 “boundary” |

### Caveat budget

- Abstract 最多保留一个范围限定，且最后一句必须回到正面贡献。
- Introduction 和 contribution bullets 不使用 `we do not claim`、`cannot` 或 `not sufficient`。
- 每个 Results 小节最多有一句说明统计范围，不在每个段落末尾追加反命题。
- Discussion 负责解释发现，不模拟 reviewer 问答。
- Limitations 集中容纳 seed、LoRA、module-local 和协议范围；同一限制不在其他章节重复。
- Conclusion 以发现和意义收束，不再重复完整免责声明。
- 同一 epistemic caveat 全文原则上不出现超过两次。

以下内容仍然是事实边界，但只作为作者内部约束保存，不需要逐条写进正文：`WS` 的 activation-aware 误差解释有既有方法血缘；current `W_tS_{D,t}` 是联合状态；三个 `alpha` 不构成完整连续剂量函数；相关性不等于因果效应。

## 四、可立即处理的写作任务

### 4.1 消除“未完稿”语气

正式正文不应出现或反复出现：

- `still consolidating`
- `remain in progress`
- `currently available`
- `until ... are complete`
- “仍在补算”“正式空位”“等待交付”

尚未完成的分析保留在源码注释和 `human_read-ch.md` 中。正文中的内容只能是“已有结果”或“当前范围之外”，不能把进行中的实验同时写成核心贡献和限制条件。

### 4.2 从防御性边界改为自然强度

正文按“结果 → 证据 → 含义”推进，不再按“结果 → reviewer 可能反驳什么 → 自我限制”推进。优先使用肯定句：

- `Functional geometry and behavioral readout diverge`，而不是 `Functional rank is not sufficient for behavior`。
- `The shared phenomenon is compression dominance, realized through model-specific temporal paths`，而不是 `this is not a universal trajectory template`。
- `The mixture provides directional exposure evidence`，而不是先强调它不能建立连续剂量函数。
- `Current whitening measures the realized joint functional state`，而不是把“混合 weight 与 activation”首先写成缺陷。

建议的章节标题映射：

| 当前标题 | 下一版标题 |
|---|---|
| `On-Policy Exposure Organizes Functional Trajectories` | `OPD Exhibits a Distinct Cross-Domain Compression Regime` |
| `A Necessary Boundary: Functional Rank Is Not Sufficient for Behavior` | `Functional Geometry and Behavioral Readout Diverge` |
| `Why Use a Module-Local Exact Scale?` | `A Functional View of Post-Training Dynamics` |
| `What On-Policy Exposure Means Here` | `Current-Self Exposure and the Compression Regime` |
| `Functional Compression Is Not Capability Compression` | `Functional Compression and Capability Can Decouple` |

最终语言审计需检查：

- 相邻段落不能都以 `but/cannot/does not/not a` 收尾；
- `boundary` 不进入标题和 contribution 名称；
- `rather than`、`we therefore do not claim` 和 “This establishes X; it does not establish Y” 不形成重复句式；
- Results 段落尽量以发现或解释含义结尾；
- reviewer 要求、pending 状态和项目管理语言不出现在成品正文。

### 4.3 明确方法血缘与新颖性

Related Work 和 Method 应连续说明：

1. DALR、SVD-LLM、ASVD、NSVD 等使用 activation statistics 改善静态低秩压缩。
2. 本文继承其 module-output error space。
3. 本文的增量是将该空间变成跨 checkpoint、跨域、跨训练臂的动态状态量，并在 OPD 轨迹上建立经验规律及其与输出和行为读出的结构关系。
4. Fisher/Jacobian/task-tangent 更接近端到端目标，但通常依赖梯度和局部全网近似；本文选择更窄、forward-only、模块局部精确的中间尺度。

所有新增文献必须先核实来源和结论，不能直接采用 agent reviewer 生成的题目与摘要。

### 4.4 提高复现性

Method 需要一个紧凑的算法式描述：

1. 冻结 domain probe 和 sample IDs。
2. 在指定 checkpoint、层和模块收集输入 token activations。
3. 按冻结 token mask 累积 uncentered Gram/second moment。
4. 构造 `S`，计算 `F=WS`。
5. 对 `F` 求奇异值，按累计平方能量计算 `r_\varepsilon`。
6. 先在模块内相对 step 0 形成变化，再对七个 module class 等权聚合。
7. 输出绝对 rank、`Delta r_\varepsilon`、`c_\varepsilon`、tail margin 和数值审计项。

正文或附录必须实际给出，而不是只写“由 manifest 冻结”：

- 每个 probe 的样本数和有效 token 数；
- token mask、采样方式和 activation hook 位置；
- covariance/Gram 估计及 `S` 的分解方式；
- eigenvalue floor、rank-deficient 和阈值 tie 的处理；
- storage/load/compute dtype；
- 优化器、scheduler、序列长度、rollout decoding、temperature 和停止条件；
- top-32 forward-KL 的归一化与尾部处理。

这些事实必须由 `human_read-ch.md` 提供。

### 4.5 简化指标解释

- 首次定义 `r_\varepsilon` 时立即解释为“保留 95% 模块输出能量所需的最少方向数”。
- 明确 `c_\varepsilon` 是相对 step 0 的即时收缩比例，不是随时间累计的量。
- 首次定义 NCD 时说明它只累计基线以下的 rank 缺口，并在 `log(1+t)` 上积分。
- 统一 NCD 单位：若 log-time 视为无量纲，可按 directions 报告，同时称为 trajectory area，避免被误解为行为损伤。
- signed NLL 在结果冻结前不进入摘要、贡献列表和结论。

### 4.6 重组主文图表

事实层交付相应结果后，主文至少展示：

- 一张紧凑 robustness 表或图：epsilon、层、七模块/attention--MLP、sample count；
- current/fixed/centered 的 headline 对照；
- `36/36` 和均值之外的 per-domain 或 per-module dispersion；
- `c_\varepsilon`--KL 的分臂、分域或时间控制结果及不确定性；
- Figure 1 中可识别的 probe/checkpoint 标记或分面，以及可用的不确定性带；
- generation-cap schedule 和双模型配对审计只在附录复现协议中登记；正文保留 cap-hit 作为终止行为读数，但不讨论这项提速配置。

完整网格放附录，正文保留直接回答 reviewer 质疑的摘要。

### 4.7 摘要、贡献与结论

在 Qwen 四臂交付前，不把部分完成的 `c_\varepsilon`--KL 写成完整跨模型贡献。若保持当前范围，不使用 “currently analyzed” 这样的施工语言，可写为：

> We evaluate the relation across four Llama arms and a complementary Qwen OPD trajectory.

贡献列表最多保留三项：

1. 动态、域条件的模块功能状态量；
2. 跨模型 OPD 早期功能压缩结果与 support-direction evidence；
3. 内部功能状态、output departure 与行为读出的结构关系。

某项缺少最终支持时，将其降为结果段内容，不保留为一级贡献。

### 4.8 压缩 Limitations

Limitations 写最终研究范围，不写项目待办，按四组组织：

1. 单轨迹、LoRA 和外部有效性；
2. module-local、current-state 与非因果解释；
3. probe、层、阈值和有限样本；
4. 跨模型 probe 与行为协议差异。

同一限制只出现一次，不在 Abstract、Results、Discussion 和 Limitations 中反复道歉。

### 4.9 Generation-cap 写作决定

- `4096/16384` 是经过 Qwen 与 Llama 配对审计的评测提速配置，不作为正文限制或解释变量。
- 正文不介绍 cap schedule，也不把跨 checkpoint 变化归因于该配置。
- 附录复现协议只需简要登记 schedule、配对审计范围和“headline 结论不变”。
- `cap-hit` 继续作为终止行为读数；它描述模型在给定评测预算下的停止行为，不等同于配置混杂。
- Qwen 配对审计的正式数字和 artifact 路由由 theory 线程先登记到 `human_read-ch.md`，写作线程随后忠实转入附录。

### 4.10 当前稿逐节去防御化审计

| 位置 | 当前问题 | 下一版处理 |
|---|---|---|
| Abstract 末句 | 以 `not a sufficient statistic` 收尾，贡献被免责声明压住 | 改为内部功能状态、输出移动和行为读出的统一正面意义 |
| Introduction 结果总览 | 每个发现都附带一层限制 | 直接给四项发现；范围留到对应结果和 Limitations |
| Contribution 3 | 名称是 `interpretive boundary` | 改成 output/readout structure 的正面贡献 |
| Related Work | 多次写 `do not claim` 和“代价是什么” | 用构念差异和研究问题分工定位，不进行自我辩护 |
| Method 定义后 | module-local 限制是必要的 | 全文唯一完整保留一次 |
| NCD 定义 | 立即强调不是 behavioral damage | 只解释它累计什么；行为含义放 Discussion |
| State decomposition | `not causal mediation` 必要但重复 | 公式后保留一次，后文不再重复 |
| Validation | `does not establish all weight metrics fail` | 删除反命题，直接展示与 raw activation/weight baselines 的经验差异 |
| Cross-model result | 用 `not a universal template` 结束 | 改为“shared regime, model-specific realization”正面结论 |
| `alpha=.5` result | 以不能建立连续函数结束 | 以 directional exposure evidence 收尾；剂量范围进 Limitations |
| KL result | 多句解释不能 calibration/causal | 正文保留时间控制证据；剩余范围集中到 Limitations |
| Matched-support result | 标题和段落都叫 boundary | 改写成功能几何与行为读出的分离发现 |
| Local readout | 以“不是 universal module”收尾 | 展示完整层组消融和答案通道证据，范围只在段首或图注说明 |
| Discussion | 三个小节主要负责否定误读 | 改成 observable 的意义、OPD regime 的解释、readout 分离的启示 |
| Limitations | 与前文重复同一 caveat | 只保留一次完整版本，压缩为两到三段 |
| Conclusion | 再次用 `rather than` 限制贡献 | 以动态功能诊断对后训练研究的价值收束 |

下一版段落采用以下基本节奏：先给发现，再给关键证据，最后说明它改变了我们对 OPD/post-training 的什么理解。只有当某个限制会直接改变该段数字的含义时，才在结果段就地说明。

## 五、等待事实层交付的写作任务

| 所需交付 | 论文收到后如何使用 |
|---|---|
| 独立 seed 结果 | seed-level 表格、CI 和主张强度 |
| current/fixed/centered 对照 | Method 构念说明和 robustness 表 |
| sample-count、bootstrap、tail-margin、dtype 审计 | Estimator robustness 与附录 |
| epsilon、跨层、七模块完整汇总 | 主文紧凑稳健性图表 |
| Qwen 四臂 `c_\varepsilon`--KL | 决定是否保留 output 关系为核心结果 |
| 时间控制、within-arm/domain、held-out 结果 | 排除 training-progress artifact |
| Qwen 配对 cap 审计的事实登记 | 仅进入附录复现说明，不改变正文 output/readout separation 结果 |
| 新 `alpha`、full-FT、Fisher/task-tangent | 可选增强，不由写作线程承诺 |

收到交付前，论文编写线程只准备插入位置和表述模板，不生成数字，不替实验判断结论。

## 六、主文与附录分配

主文必须保留：

- metric 的最短完整推导和适用边界；
- 四臂控制逻辑；
- 跨模型 headline；
- 一张 estimator/construct robustness 摘要；
- `c_\varepsilon`--output 中真正完成的结果；
- matched-support 的功能几何与行为读出分离结果；
- 可复现所必需的核心协议。

附录承载：

- 完整 probe x arm x checkpoint x layer x module 网格；
- 全 epsilon 表；
- sample-count、dtype 和 tail-margin 细节；
- extractor 定义与行为分解示例；
- 扩展相关性、held-out 和替代聚合；
- 训练与 rollout 的完整配置。

附录不能替代主文对核心稳健性的最小展示。

## 七、执行顺序

1. 等待 `human_read-ch.md` 冻结新增事实和允许主张。
2. 重写 Method 与 Reproducibility，消除“manifest 代替说明”。
3. 根据交付设计主文 robustness 表和补充材料。
4. 重写 Results，使每个一级主张紧邻证据、范围和不确定性。
5. 收紧 Discussion/Limitations 的因果语言和重复内容。
6. 补齐并核实 Related Work。
7. 最后重写 Introduction、Contributions、Abstract 和 Title。
8. 中文稿确认后同步英文稿并生成 reviewer-facing PDF。

## 八、当前状态判断

在不增加实验交付的情况下，写作仍可改善复现说明、主张边界和成品感，但很难单独解决 seed-level robustness 门槛。近期目标不是扩大理论，而是确保每个已完成结果都被完整展示，每个未完成问题都不以“施工中的核心贡献”出现在投稿稿件中。
