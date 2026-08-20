# Refresh 对照与 Matched-Support 展示方式调整

日期：2026-07-29

状态：讨论方案已确认，等待最终统一编写。在收到明确的“编写论文”指令前，不据此修改中英文正文、Supplement、图片或压缩包。

关联记录：

- `/root/lecture/add/2026-07-29_construct_comparison_figure_rr5_revision.md`
- `/root/lecture/add/2026-07-29_author_annotations_theory_experiment_wording.md`

## 1. 本轮讨论解决的问题

当前 “Exposure 与 Refresh 组织轨迹，但不决定读出” 小节存在四个问题：

1. frozenSelf0-KD 结论包含一句没有必要的自我辩护；
2. 对 freshness 边界的表述像审稿回复，不像论文自然叙述；
3. off-KD/seqKD 的 matched scatter 只能展示几何路径接近，不能完整展示 Eval、格式和截断错配；
4. frozenSelf0-KD 的逐域热力图占据较大空间，可以由更直接的逐域轨迹图替代。

本轮最终决定是：收紧两段正文；将 Figure 2 从 `2 x 2` 改为 `1 x 2`；删除 heatmap 与 matched scatter；使用一张双 panel 表格完整展示 matched-support 的几何与行为结果。

## 2. FrozenSelf 段落的行文调整

### 2.1 删除自说自话的前半句

当前表述：

> 因此首次使用 student-generated 数据不足以重现 OPD 轨迹，current-support refresh bundle 是更直接的组织因素。

删除：

> 因此首次使用 student-generated 数据不足以重现 OPD 轨迹

保留并直接写成：

> OPD 相对 frozenSelf0-KD 的压缩优势在五个固定外部域和全部后续 checkpoint 上持续存在，只有训练 support 在最终 checkpoint 出现单点例外。由此，current-support refresh bundle 是该轨迹更直接的组织因素。

理由：

- “首次使用 student-generated 数据不足……”是在回应一个正文没有明确提出的假设；
- frozenSelf0-KD 的直接结论是持续 refresh 的总体作用，无需先否定一个额外说法；
- 直接从对照结果进入组织因素，叙述更像论文。

### 2.2 将 freshness 边界改成自然论文口吻

当前表述：

> 这里的 bundle 同时包含 refresh 引起的长度、EOS、重复与风格变化，我们不把它缩写成已经被单独识别的抽象 freshness。

最终改为：

> 由于 refresh 同时改变 rollout 的长度、EOS、重复和风格，这一对照识别的是 refresh bundle 的总体作用，而不是纯粹的 freshness 效应。

该版本保留必要的构念边界，但不使用“我们不把它缩写成……”这种审稿回复式措辞。

### 2.3 合并后的中文工作稿

```latex
更直接的 Llama frozenSelf0-KD 对照只移除 current-student rollout 的持续刷新。OPD 相对 frozenSelf0-KD 的压缩优势在五个固定外部域和全部后续 checkpoint 上持续存在，只有训练 support 在最终 checkpoint 出现单点例外。由此，current-support refresh bundle 是该轨迹更直接的组织因素。由于 refresh 同时改变 rollout 的长度、EOS、重复和风格，这一对照识别的是 refresh bundle 的总体作用，而不是纯粹的 freshness 效应。
```

最终编写时需要根据 Figure 2 的新 panel 编号补入交叉引用。

## 3. Figure 2 的最终结构

### 3.1 从 `2 x 2` 缩减为 `1 x 2`

Figure 2 最终只保留两个 panel：

| Panel | 内容 | 回答的问题 |
|---|---|---|
| (a) | Qwen \(\alpha=.5\) exposure ordering | 改变 current-student exposure 后，轨迹是否沿 off-KD 到 OPD 的方向移动？ |
| (b) | Llama OPD--frozenSelf0-KD 的逐域 refresh margin | 持续 refresh 的压缩优势是否跨外部域和 checkpoint 保持？ |

删除当前：

- frozenSelf0-KD 逐域热力图；
- off-KD/seqKD matched scatter 及 cap-hit inset。

off-KD/seqKD 的完整结果转由正文表格承担。

### 3.2 Panel (b) 的推荐画法

不再同时画 OPD 与 frozenSelf0-KD 的域平均绝对轨迹，而直接画 paired margin：

\[
m_{D,t}^{\mathrm{refresh}}
=
c_{\mathrm{OPD},D,t}^{(5)}
-
c_{\mathrm{frozenSelf0},D,t}^{(5)}.
\]

其中：

- \(m_{D,t}^{\mathrm{refresh}}>0\) 表示 OPD 比 frozenSelf0-KD 更压缩；
- 五个固定外部域分别使用细线；
- 外部域等权均值使用粗线；
- train support 使用灰色虚线，明确展示最终 checkpoint 的单点例外；
- 添加 \(y=0\) 水平线；
- 各 checkpoint 只连接真实观测点，不插值额外状态。

优先展示全部五个固定外部域，而不是事后挑选部分域。五条外部域曲线数量可控，也避免 cherry-picking 质疑。

如果最终版面确实不足，最低可接受方案是：

- 正文画四个核心域与 train support；
- AIME25 放入 Supplement；
- 图注明确四个核心域是预先固定的主网格，而不是根据结果选择。

### 3.3 Panel (b) 图注必须说明

建议图注包含：

> 细线表示五个固定外部输入域，粗线表示其等域均值，灰色虚线表示训练 support。正值表示持续刷新的 OPD 相对 frozenSelf0-KD 具有更深的 equal-5 功能压缩；外部域在全部后续 checkpoint 上保持正值，训练 support 仅在最终 checkpoint 出现单点例外。完整逐模块、层级和阈值结果见 Supplementary Material。

### 3.4 Figure 2 的整体作用

新 Figure 2 只承担 exposure 与 refresh 两个机制控制：

- (a) 给出 exposure fraction 的有序移动；
- (b) 给出持续 refresh 的跨域配对优势。

它不再同时承担 support--readout separation。后者需要同时展示内部路径和行为输出，用表格比散点图更完整。

## 4. 删除原 Figure 2c 热力图的理由

原热力图的主要信息是：

- 五个固定外部域上的 frozenSelf0-KD--OPD margin 持续支持 OPD 更深；
- train support 在最终 checkpoint 出现一个负 margin。

这些信息可以由新的逐域 margin 轨迹更直接地表达。热力图的不足是：

- 颜色编码不如零线两侧的轨迹容易理解；
- 占据较大的双栏面积；
- 与 Panel (b) 的域平均曲线部分重复；
- 读者难以直接观察每个域随 checkpoint 的变化形态。

完整热力图可以保留在 Supplement，作为逐 cell 数值审计；正文不再需要。

## 5. 用表格替代 off-KD/seqKD matched scatter

### 5.1 替换理由

原 matched scatter 主要展示：

- Qwen pooled Pearson 约为 `.995`；
- Llama pooled Pearson 约为 `.944`；
- 两模型的 direction-MAE 较小；
- inset 只额外展示 MATH500 cap-hit 差。

它没有同时展示：

- 不同输入域上的路径一致性；
- MATH500 accuracy；
- MMLU-Pro strict/flexible 与 extract failure；
- IFEval strict readout；
- Qwen 的强行为分叉为何没有在 Llama 上同等复现。

因此散点图不足以独立支撑 support--readout separation。正文改用双 panel 表格。

## 6. Matched-Support 表 Panel A：逐域功能路径一致性

### 6.1 表格结构

Panel A 使用每个模型一行。每个域报告：

\[
\text{Pearson }r \;/\; \text{direction-MAE}.
\]

推荐结构：

| Model | General \(r/\mathrm{MAE}\) | Math \(r/\mathrm{MAE}\) | MMLU-Pro \(r/\mathrm{MAE}\) | IFEval \(r/\mathrm{MAE}\) | Pooled \(r/\mathrm{MAE}\) |
|---|---:|---:|---:|---:|---:|
| Qwen | .995 / 1.13 | .996 / 2.67 | .998 / 2.18 | .999 / 2.29 | .995 / 2.07 |
| Llama | -.771 / 2.53 | .971 / 1.73 | .968 / 2.23 | .824 / 2.40 | .944 / 2.23 |

数据口径：

- equal-5 non-QK；
- \(\varepsilon=.05\)；
- off-KD 与 seqKD 使用完全匹配的非零 checkpoint；
- Qwen 每域使用 steps 5/10/20/40/80/160/320/480/624；
- Llama 每域使用 steps 5/20/40/80/160/320；
- Pearson 衡量随 checkpoint 的线性同动；
- MAE 衡量逐 checkpoint 的绝对方向差。

底层来源：

`/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk/EQUAL5_functional_trajectories.csv`

### 6.2 Panel A 的解释

不能只写“两模型所有域路径都高度一致”。逐域结果显示：

- Qwen 四个核心域均具有很高 Pearson，且 MAE 较小；
- Llama 在 held-out math、MMLU-Pro 和 IFEval 上保持较高或中高同动；
- Llama General 的 Pearson 为负，但 MAE 仍约为 2.53 directions，说明两条轨迹的绝对功能预算接近，却没有相同的时间变化方向；
- pooled Pearson 会同时受到跨域水平和共同 checkpoint 结构影响，不能替代逐域检查。

更准确的正文结论是：

> off-KD 与 seqKD 在 Qwen 的四个核心域中形成高度一致的功能路径；Llama 的总体模式预算仍较接近，但时间同动具有域异质性。

## 7. Matched-Support 表 Panel B：终点行为

### 7.1 不使用成对斜杠行

早期草案把每个行为单元格写成 `off-KD / seqKD`。该形式容易让读者忘记顺序，也使 MMLU strict/flexible 单元格出现两层斜杠。

最终决定是：

- 单独增加 `Training setting` 列；
- 每个模型使用 off-KD、seqKD 两行；
- 不在行为单元格中使用训练设置配对斜杠；
- 表注不再承担解释数值顺序的任务。

### 7.2 最终结构

| Model | Setting | MATH Acc. | MATH Cap-hit | MMLU Strict | MMLU Flexible | MMLU Extract-fail | IFEval Prompt | IFEval Instruction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | off-KD | 79.4 | 4.8 | 35.4 | 57.1 | 47.3 | 23.1 | 36.5 |
|  | seqKD | 72.4 | 73.0 | 30.6 | 58.1 | 54.1 | 24.4 | 39.3 |
| Llama | off-KD | 8.2 | 91.8 | 14.2 | 16.4 | 50.3 | 19.6 | 33.2 |
|  | seqKD | 10.2 | 94.6 | 13.1 | 15.4 | 51.6 | 19.2 | 31.5 |

所有行为值使用百分数。

终点：

- Qwen：step 624；
- Llama：step 320。

推荐使用分组表头：

- `MATH500`：Accuracy、Cap-hit；
- `MMLU-Pro`：Strict、Flexible、Extract-fail；
- `IFEval`：Prompt-strict、Instruction-strict。

表格不再加入 mean tokens。cap-hit 已直接表达截断错误，mean length 可在正文或 Supplement 中作为一致性说明，避免主表过宽。

### 7.3 Panel B 的作用

该 panel 同时展示：

- 任务得分：MATH accuracy；
- 终止/截断错误：MATH cap-hit；
- 格式敏感读出：MMLU strict/flexible；
- 抽取/格式失败：MMLU extract-fail；
- 指令遵循：IFEval prompt/instruction strict。

Qwen 的主要分叉集中在 MATH500 cap-hit，accuracy 与 MMLU strict/extract 也存在差异；Llama 的同类行为差异明显更小。该跨模型差异用于限制 target distribution 的行为效应，不能写成跨模型固定规律。

## 8. Matched-Support 段落的最终逻辑

建议正文按以下顺序展开：

1. 说明 off-KD 与 seqKD 保持 teacher sequences、顺序、步数和 LoRA 配置相同，只改变 dense teacher target 与 hard next-token target；
2. 引用表格 Panel A，说明 Qwen 的逐域路径高度一致，Llama 的模式预算总体接近但具有 General 域异质性；
3. 引用表格 Panel B，说明 Qwen 出现明显终止/格式分叉，而 Llama 没有同等幅度复现；
4. 得出 support--readout separation，而不是声称 soft/hard target 在所有模型中产生固定行为差异。

推荐中文工作稿：

```latex
off-KD 与 seqKD 使用完全相同的 teacher sequences、样本顺序、训练步数和 LoRA 配置，只将 dense teacher target 替换为 hard next-token target。Table~\ref{tab:support-readout}A 显示，二者在 Qwen 的四个核心输入域中形成高度一致的 equal-5 功能路径；Llama 的总体功能模式预算仍较接近，但时间同动具有域异质性。行为读出并未被这些路径锁定：Qwen 的 seqKD 在终点出现显著更高的 MATH500 cap-hit，并伴随 accuracy、strict format 与 extraction 的变化，而 Llama 没有复现同等幅度的分叉（Table~\ref{tab:support-readout}B）。因此，相同 sequence support 可以组织相近的功能模式预算，但 target distribution 仍能改变具体读出；功能秩不是完整行为的充分统计量。
```

### 8.1 行文限制

- 不再使用“存在性反例限定解释边界”这种审稿回复式表达；
- 可以直接称为 `support--readout separation`；
- 不写成“两模型所有域路径完全一致”；
- 不因 Llama General Pearson 为负而否定其较小 direction-MAE；
- 不用 pooled Pearson 掩盖逐域异质性；
- 不把 Qwen 的 68.2 percentage-point cap-hit gap 外推为 soft/hard target 的跨模型固定效应。

## 9. 版面影响

当前 Figure 2 为双栏 `2 x 2`。新方案：

- Figure 2 缩为双栏 `1 x 2`；
- 删除 heatmap；
- 删除 matched scatter 与 inset；
- 增加一张双 panel 的 `table*`。

虽然新增表格，但它同时替代散点、cap-hit inset 和正文中零散的行为数字，整体信息密度更高。节省出的图面积可用于：

- 理论核心中裸权重各向同性假设与截断 SVD 的关键推导；
- 输入域与控制链的详细说明；
- 第一结果小节的 ER/TPNT/PABS 构念比较。

## 10. Supplement 安排

以下内容移入或保留在 Supplement：

1. frozenSelf0-KD 完整逐域 heatmap；
2. 所有六个 probe 的原始 paired margins；
3. 全部层、模块、\(\varepsilon\) 和 checkpoint 结果；
4. off-KD/seqKD 的原 matched scatter；
5. 完整行为轨迹，而不仅是终点；
6. MATH500 mean/median/p90 length、EOS、boxed rate 等终止细节；
7. MMLU-Pro 类别级 strict/flexible/extract audit；
8. IFEval 类别级结果；
9. Pearson/MAE 的计算协议与 matched-cell inventory。

正文引用统一使用：

- `Supplementary Material, Sec. X`；
- 简写为 `Supplementary Sec. X`。

不使用 `Appendix X`。

## 11. 最终统一执行检查

收到“编写论文”指令后：

1. 从最新 equal-5 数据重新验证 Panel A 的逐域 Pearson/MAE；
2. 从 canonical behavior files 验证 Panel B 的四行终点数值；
3. 重绘 Figure 2(a--b)；
4. 将 heatmap 和 matched scatter 移入 Supplement；
5. 创建双 panel matched-support 表；
6. 重写 frozenSelf 与 matched-support 两段中英文正文；
7. 更新 Figure 2 图注、表注、图号和交叉引用；
8. 检查表格宽度与 AAAI 字体要求；
9. 确认正文与 Limitations 中 support--readout 边界一致；
10. 最后统一同步 Overleaf 上传目录和 ZIP。
