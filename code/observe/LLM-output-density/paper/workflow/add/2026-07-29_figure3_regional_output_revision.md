# Figure 3 区域输出图重构

日期：2026-07-29

状态：讨论方案已确认，等待最终统一编写。在收到明确的“编写论文”指令前，不据此修改中英文正文、Supplement、图片或压缩包。

关联记录：

- `/root/lecture/add/2026-07-29_construct_comparison_figure_rr5_revision.md`
- `/root/lecture/add/2026-07-29_refresh_matched_support_figure_table_revision.md`
- `/root/lecture/add/2026-07-29_author_annotations_theory_experiment_wording.md`

## 1. 当前 Figure 3 的原始任务

当前 Figure 3 试图建立以下证据链：

\[
\text{域条件功能压缩}
\longrightarrow
\text{区域输出分布移动}
\longrightarrow
\text{最终行为读出}.
\]

原图包含：

1. Panel (a)：MMLU-Pro 与 MATH500 的 token 区域示意；
2. Panel (b)：每个模型和训练设置内，\(c_{.05}\) 与区域 full-vocabulary KL 的 Spearman 热力图；
3. Panel (c)：MMLU-Pro signed readout 与格式/抽取行为的 Spearman 热力图。

这条证据链必须保留，因为它使论文不止报告内部几何轨迹，还说明功能压缩与模型输出分布的实际变化相关，并同时限定其不能直接决定行为方向。

但当前三 panel 使用了三种不同性质的信息：

- Panel (a) 是方法定义；
- Panel (b) 是功能状态与无符号输出移动的关系；
- Panel (c) 是 signed token readout 与自由生成行为的关系。

三者没有共同统计对象或视觉语言，导致“闭环”主要存在于图注，而不是图本身。

## 2. 删除原 Panel (a)

### 2.1 最终决定

删除 token-region 彩色方块示意，不再让它占据正文主图三分之一宽度。

理由：

- 它不包含实验结果；
- 彩色块宽度可能让读者误以为对应真实 token 数量或区域长度；
- Methods 本来就需要正式定义区域切分；
- 删除后可以给两个结果 panel 更充足的横向空间。

### 2.2 区域定义的最终落点

区域定义不能只依赖图注，按以下方式分工。

Methods 中正式定义：

- MMLU-Pro：
  - prompt \(P\)；
  - pre-answer formatting tokens \(F\)；
  - correct answer-choice label \(A\)；
  - termination token \(T\)；
- MATH500：
  - prompt \(P\)；
  - reference reasoning before the final boxed answer \(C\)；
  - token-clean complete boxed-answer span \(B\)；
  - termination token \(T\)。

Methods 同时说明：

- 所有 checkpoint 在相同 fixed reference sequences 上 teacher forcing；
- MMLU-Pro 的参考输出由正确答案确定；
- MATH500 使用 dataset-provided reference solution；
- KL 主区域为 MMLU-Pro \(A/F/T\) 与 MATH500 \(B/T\)；
- signed NLL 另覆盖需要的 \(F-A\) 与 \(B-C\) contrasts。

Figure 3 caption 只解释实际出现在图中的缩写，不承担完整 tokenization/mask 定义。

Supplement 提供：

- exact token masks；
- tokenizer 边界处理；
- MATH boxed-answer token-clean construction；
- 样本数、macro aggregation 与缺失区域规则；
- whole-reference-stream sensitivity。

## 3. Figure 3 的新布局

删除原 Panel (a) 后，Figure 3 改为双栏 `1 x 2`：

| 新 panel | 原 panel | 内容 |
|---|---|---|
| (a) | 原 (b) | 相对功能压缩与区域 KL 的 within-trajectory Spearman |
| (b) | 原 (c) | signed readout 与格式/抽取行为的关系 |

两个 panel 分别承担：

- (a)：功能压缩是否追踪区域级无符号 output departure；
- (b)：这种联系是否足以决定 signed readout 与真实行为方向。

图的中心结论为：

> 功能压缩稳定追踪区域输出分布移动的幅度，但这种移动如何转化为格式、答案与终止行为具有模型和训练设置依赖性。

## 4. 新 Panel (a)：区域 KL 热力图

### 4.1 计算对象

对每个

\[
\text{model}
\times
\text{training setting}
\times
\text{domain}
\times
\text{region},
\]

在严格匹配的非 base checkpoint 上计算

\[
\rho_s
\left(
c_{.05}^{(5)},
D_{\mathrm{KL}}(p_0\Vert p_t)
\right).
\]

step 0 不进入相关，因为 \(c\) 与 departure 均为结构性零。

列对应：

- MMLU-Pro：\(A/F/T\)；
- MATH500：\(B/T\)。

行对应两个模型的四个训练设置。

### 4.2 视觉设计

当前 Panel (b) 使用以零为中心的红蓝 diverging palette，但全部数值均为正，且大多位于较高区间。这会导致整个图近乎同一深红色，无法形成有效视觉层次。

最终建议：

- 使用具有单调亮度变化的 sequential palette；
- 优先使用灰度后仍单调的 palette，例如 `cividis`；
- 保留每个 cell 内的两位小数；
- 根据背景亮度自动选择黑色或白色数字；
- colorbar 标出 Spearman \(\rho_s\)；
- 色域根据正式数据冻结，不能为了放大局部差别事后选择误导性范围；
- 图注或 colorbar 明确数值范围。

### 4.3 平直的分组列标签

删除倾斜标签。使用两级平直列标题：

| \multicolumn{3}{c}{MMLU-Pro} | \multicolumn{2}{c}{MATH500} |
|---|---|---|---|---|
| \(A\) | \(F\) | \(T\) | \(B\) | \(T\) |

这样：

- 不需要重复写 `MMLU A`、`MMLU F`；
- 避免 30--45 度倾斜文字；
- 减少图下方无效空白；
- 更容易与 Methods 中的区域定义对应。

倾斜标签在 AAAI 中并不违法，但在删除原 Panel (a) 后已没有必要。

## 5. 新 Panel (b)：Signed readout 边界

### 5.1 计算对象

当前 MMLU-Pro panel 展示两类相关：

\[
\rho_s
\left(
\Delta\mathrm{NLL}_{F-A},
\text{strict--flexible gap}
\right),
\]

以及

\[
\rho_s
\left(
\Delta\mathrm{NLL}_{F},
\text{extract failure}
\right).
\]

第一列检查格式相对答案的 signed token likelihood 变化是否与严格格式缺口同向；第二列检查格式 token 的 signed NLL 是否与自由生成中的答案抽取失败同向。

该 panel 的作用不是证明统一行为预测，而是展示相关符号随模型和训练设置改变，从而限定：

- unsigned KL 对 movement magnitude 的关系较稳定；
- signed token readout 与完整自由生成行为没有跨模型、跨训练设置的固定映射。

### 5.2 视觉设计

由于该 panel 同时包含正负值，可以保留以零为中心的 diverging palette，但必须：

- 在每个 cell 内保留带符号的数值；
- 使用零对称 color scale；
- 确保正负两侧在灰度下即使亮度相近，仍能通过 `+/-` 数字符号解读；
- 不依赖红/蓝颜色本身传达正负；
- 保持与 Panel (a) 一致的字体、cell 尺寸和 row order；
- 将完整 MATH signed readout 与更多行为关系放入 Supplement，正文 panel 明确标为 MMLU-Pro。

## 6. 模型缩写与行标签

当前行标签使用：

- `L-off-KD`；
- `L-OPD`；
- `Q-off-KD`；
- `Q-OPD`。

用户要求图注必须说明：

> L and Q denote Llama and Qwen, respectively.

更推荐的最终方案是取消缩写，直接写：

- `Llama / off-KD`；
- `Llama / OPD`；
- `Llama / seqKD`；
- `Llama / SFT`；
- `Qwen / off-KD`；
- `Qwen / OPD`；
- `Qwen / seqKD`；
- `Qwen / SFT`。

删除原 Panel (a) 后横向空间充足，直接标签比让读者查图注更清楚。

如果最终仍保留 `L/Q`，图注必须显式解释，不能默认读者自行推断。

## 7. 当前 Panel (c) 的裁切问题

当前导出的 Figure 3 中，右侧 panel 标题 `Readout boundary` 没有完整显示，右侧被 PDF media box 截断。

这不是阅读器问题，而是图片导出时布局与 media box 不匹配。

最终处理：

- 删除原 Panel (a)；
- 重新生成 `1 x 2` figure；
- 重新计算 panel 间距和外边距；
- 在图片生成程序中完成真实 crop；
- 检查最终 PDF media box；
- 不在 LaTeX `\includegraphics` 中使用 `trim`、`clip` 或 `viewport`。

AAAI AuthorKit 明确要求在图片生成阶段完成裁切，禁止依赖 LaTeX 的 trim/clip。

仅删除 Panel (a) 而不重新布局和导出，不能视为修复完成。

## 8. AAAI 图形合规要求

根据 `/root/lecture/AuthorKit27/AnonymousSubmission2027.pdf`：

- 图片格式必须是 PDF、PNG 或 JPEG；
- 图注位于图片下方，使用 10 pt roman；
- 图内标签与文字至少 9 pt；
- 图内字体必须嵌入；
- 不得包含 Type 3 字体；
- 线宽至少 0.5 pt；
- PDF bounding box 必须真实正确；
- 颜色必须满足 WCAG 2.0 对比要求；
- 论文可能以灰度打印；
- 图必须在不依赖颜色的情况下仍可解读；
- 不得使用 LaTeX trim/clip 修补图片边界。

热力图本身在 AAAI 中常见且合法；AuthorKit 不限制图形类型，也不禁止倾斜标签。合规关键是字体、可读性、边界、灰度与颜色冗余编码。

## 9. 当前 Figure 3 的灰度与字体检查

### 9.1 灰度检查结果

已将当前 Figure 3 转换为灰度检查：

- cell 内数值与正负号仍可阅读；
- Panel (a) 的区域颜色失去区分，但字母仍保留；
- 原 Panel (b) 大部分高正相关 cell 均变成相近深灰，视觉层次较弱；
- 原 Panel (c) 的正负主要依靠数字符号，而不是灰度色块区分。

因此：

- 当前图满足基本“无需颜色仍可读取数字”的要求；
- 但没有很好地利用灰度亮度表达数值结构；
- 新 Panel (a) 应改用单调亮度 sequential palette；
- 新 Panel (b) 必须依靠带符号数字提供冗余编码。

### 9.2 字体风险

对当前 Figure 3 PDF 执行 `pdffonts` 没有列出嵌入字体条目，说明文字可能在 Ghostscript 后处理时被转换为轮廓或其他非标准形式。

最终重绘时需要：

- 从 Matplotlib 直接导出 PDF；
- 使用 Helvetica 或 Times-compatible 字体；
- 嵌入 TrueType/Type 1/OpenType 字体；
- 用 `pdffonts` 检查最终 PDF；
- 确认没有 Type 3；
- 在论文实际插入尺寸下确认所有标签至少 9 pt。

## 10. Methods、Results 与 Caption 的分工

### 10.1 Methods 负责定义计算

Methods 应说明：

1. fixed reference sequences 与 teacher forcing；
2. MMLU-Pro \(P/F/A/T\) 和 MATH500 \(P/C/B/T\) 的区域定义；
3. full-vocabulary KL 的方向：
   \[
   D_{\mathrm{KL}}(p_0\Vert p_t);
   \]
4. 相关分析只使用 matched non-base checkpoints；
5. Panel (a) 的 within-model、within-training-setting Spearman 计算；
6. Panel (b) 的两个 signed NLL--behavior Spearman；
7. macro aggregation、checkpoint grouping 和 step-0 exclusion；
8. checkpoint-demeaning 与 held-out prediction 是额外分析，不与 raw within-trajectory Spearman 混为同一统计量。

完整 mask、样本数和敏感性进入 Supplement。

### 10.2 Results 负责解释证据

Unsigned output subsection 解释：

- \(c_{.05}^{(5)}\) 与区域 KL 在多数训练轨迹中同步；
- 该关系说明功能压缩追踪 output-distribution departure；
- raw correlation 可能包含共同训练进度，因此还需要 checkpoint-demeaned 和 checkpoint-held-out 结果；
- 与 \(p_k\) 的公平比较由独立表格承担。

Signed readout subsection 解释：

- KL 只表示移动幅度；
- signed NLL 区分 gold/reference token likelihood 改善或恶化；
- signed contrast 到 strict/flexible 或 extract failure 的符号随模型和训练设置变化；
- 功能模式预算、signed token readout 与完整自由生成行为是三个相关但不可合并的观察层。

### 10.3 Caption 只负责如何读图

图注不承担全部公式和统计协议，只需：

- 说明两个 panel；
- 解释列缩写；
- 说明行标签中的模型缩写或直接使用完整模型名；
- 指出 cell 数值为 Spearman；
- 指向 Methods 和 Supplement。

## 11. 推荐英文图注工作稿

```latex
\caption{Regional output relationships. (a) Within-trajectory Spearman correlations between equal-5 functional contraction and regional full-vocabulary KL over matched non-base checkpoints. Columns denote the MMLU-Pro correct-label ($A$), format ($F$), and termination ($T$) regions and the MATH500 boxed-answer ($B$) and termination ($T$) regions. (b) MMLU-Pro correlations of the format--answer signed-NLL contrast with the strict--flexible gap and of format NLL with extraction failure. L and Q denote Llama and Qwen, respectively; complete masks, checkpoint-demeaned analyses, and additional regions appear in the Supplementary Material.}
```

如果最终图中直接使用完整模型名，应删除最后一句中的 `L and Q denote...`。

## 12. 推荐中文图注工作稿

```latex
\caption{区域输出关系。（a）在每个模型和训练设置的匹配非 base checkpoint 上，equal-5 功能压缩与区域 full-vocabulary KL 的轨迹内 Spearman。各列依次表示 MMLU-Pro 的正确选项标签区域 $A$、格式区域 $F$、终止区域 $T$，以及 MATH500 的 boxed-answer 区域 $B$ 和终止区域 $T$。（b）MMLU-Pro 格式--答案 signed-NLL contrast 与 strict--flexible gap，以及格式 NLL 与 extract failure 的相关。L 和 Q 分别表示 Llama 与 Qwen；完整 mask、checkpoint-demeaned 分析和其他区域见 Supplementary Material。}
```

如果最终图中直接使用完整模型名，同样删除 `L 和 Q分别表示...`。

## 13. 与区域预测表的关系

Figure 3(a) 与区域预测表不能写成同一证据的重复版本：

- Figure 3(a)：描述每条训练轨迹内，功能压缩与 KL 是否随 checkpoint 同步；
- checkpoint-demeaned 分析：固定训练阶段后，比较不同训练设置之间是否仍同步；
- 区域预测表：在 checkpoint-held-out 条件下比较 \(C_5\) 与 \(P_{k,5}\) 的样本外预测能力。

三者分别回答：

1. 同一轨迹内是否共同变化；
2. 是否超出共同训练时钟；
3. 功能谱相对权重位置基线是否具有更好的样本外任务对齐。

正文必须按这一顺序说明，避免将 raw correlation 冒充 held-out prediction。

## 14. Supplement 安排

Figure 3 相关 Supplement 至少包括：

1. 完整 token-region mask 与样例；
2. whole-reference-stream sensitivity；
3. raw within-trajectory correlations 的完整数值表；
4. checkpoint-demeaned correlations；
5. checkpoint-held-out folds 与预测结果；
6. signed/absolute NLL 的完整 target 网格；
7. MATH500 signed \(B-C\) 结果；
8. MMLU-Pro strict/flexible/extract 的完整行为轨迹；
9. 不同 \(\varepsilon\)、层和模块聚合的敏感性；
10. Figure 3 中没有显示的其他区域与模型特定结果。

正文统一引用：

- `Supplementary Material, Sec. X`；
- 简写 `Supplementary Sec. X`。

不使用 `Appendix X`。

## 15. 最终统一执行检查

收到“编写论文”指令后：

1. 从最新 FAT/equal-5 结果重新验证 Figure 3 两个 heatmap 的数值；
2. 核对 step-0 是否全部排除；
3. 核对每行使用的 matched checkpoint 数；
4. 删除原 token-region schematic；
5. 重绘 `1 x 2` Figure 3；
6. 将 Panel (a) 改为 sequential grayscale-friendly palette；
7. 保留 Panel (b) 的零中心 diverging palette和带符号数字；
8. 使用平直分组列标签；
9. 优先将行标签改为完整模型名；
10. 修复右侧 media-box 裁切；
11. 检查图内字号、嵌入字体、Type 3、线宽和 WCAG/灰度可读性；
12. 在 Methods 中补充计算定义；
13. 在 Results 中区分 raw、demeaned 与 held-out 三层证据；
14. 同步更新中英文图注、正文引用和 Supplement；
15. 最后统一同步 Overleaf 上传目录和 ZIP。
