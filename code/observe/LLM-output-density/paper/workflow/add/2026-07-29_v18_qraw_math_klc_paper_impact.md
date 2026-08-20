# v18 QRAW 与 Math KL_C 纳入正文后的论文修改范围

日期：2026-07-29

## 0. 本记录的状态

本文件登记以下两项更新对论文的具体影响：

1. `human_read-ch.md` v18 已补齐 Qwen L18 的 raw-activation strict common grid；
2. 作者决定将 MATH500 CoT 区域的 exact full-vocabulary
   \(\mathrm{KL}_C\) 正式纳入论文，而不再维持 `PAPER_DEFERRED`。

本次只登记修改方案，**没有修改**：

- 中文正文 `aaai_opd_sft_trajectory_zh/main_zh.tex`；
- 英文正文 `aaai_opd_sft_trajectory_en/main_en.tex`；
- 中英文 Supplement；
- 任何图片、PDF 或 ZIP。

本记录更新了此前两份讨论记录中的局部状态：

- `2026-07-29_construct_comparison_figure_rr5_revision.md` 中的
  “Qwen RR5 pending” 已经过时；
- `2026-07-29_figure3_regional_output_revision.md` 中 Math 只展示 \(B/T\)
  的方案应更新为展示 \(C/B/T\)。

其余已确认的 Figure 3 设计仍然有效：删除 token-region 示意 panel，正文 Figure 3
采用一个无符号区域输出 panel 和一个 signed readout panel。

---

## 1. 新证据的准确含义

### 1.1 QRAW 补齐双模型构念比较

Qwen 新共同网格为：

\[
4\ \text{training settings}
\times 4\ \text{checkpoints}
\times 4\ \text{probes}
=64\ \text{states}.
\]

具体设置为：

- training settings：OPD、SFT、off-KD、seqKD；
- checkpoints：5、20、40、160；
- probes：general、IFEval、held-out math、MMLU-Pro；
- raw activation block \(A\)：ER、PR、top-1/8/32 share、
  raw/centered anisotropy、step-0 CKA；
- functional block \(C_5\)：non-QK equal-5
  \(c_{\varepsilon}^{(5)}\)，\(\varepsilon=.05\)；
- weight block \(P_{k,5}\)：\(p_4,p_8,p_{16},p_{32}\)，
  使用 deployed BF16 merged-minus-base；
- outer split：leave-one-checkpoint-group-out；
- 标准化与正则选择只使用 outer-training checkpoints。

Qwen 和 Llama 现在都具有各64个严格共同状态。按预先确定的 primary metric：

| 模型/任务 | Raw activation \(A\) | Functional \(C_5\) | Weight \(P_{k,5}\) | 判据 |
|---|---:|---:|---:|---|
| Llama cumulative KL | .104 | **.720** | -.349 | OOF \(R^2\) |
| Llama absolute NLL | .181 | **.738** | .436 | OOF \(R^2\) |
| Llama signed NLL | .022 | **.541** | -.364 | OOF \(R^2\) |
| Llama OPD | .556 | **.743** | .688 | OOF AUC |
| Qwen cumulative KL | -.575 | **.344** | .052 | OOF \(R^2\) |
| Qwen absolute NLL | .112 | .247 | **.261** | OOF \(R^2\) |
| Qwen signed NLL | -.284 | **.278** | -.264 | OOF \(R^2\) |
| Qwen OPD | .595 | **.708** | .521 | OOF AUC |

正确结论是：

- \(C_5\) 相对 raw activation \(A\) 在全部八个 primary comparisons 中均更好；
- \(C_5\) 相对 \(P_{k,5}\) 仅保留一个反例：
  Qwen absolute NLL 的 \(R^2\) 为 .247，对比 \(P_{k,5}\) 的 .261；
- 这支持 \(WS\) 不是 raw activation spectrum 的改名；
- 不支持“功能谱全面支配所有权重坐标”；
- Qwen OPD 的 \(C_5\) AUC=.708，但 log-loss=1.305，
  不能把排序判别优势写成概率校准或固定阈值分类优势。

### 1.2 Math KL_C 的新增含义

\(\mathrm{KL}_C\) 在 MATH500 固定 reference solution 中，
对 token-clean CoT 区域 \(C\) 计算：

\[
\mathrm{KL}_C
=D_{\mathrm{KL}}(p_0\Vert p_t).
\]

覆盖为：

- Qwen 和 Llama 共62个 model states；
- 每个 state 使用500道 MATH500；
- 共31,000条 sample rows；
- BF16 forward；
- FP32 `log_softmax` 与 KL；
- 先在每题的 \(C\) tokens 内平均，再对500题 sample-macro。

逐训练设置的 \(c_\varepsilon^{(5)}\)-\(\mathrm{KL}_C\) Spearman：

| 模型 | off-KD | OPD | seqKD | SFT |
|---|---:|---:|---:|---:|
| Llama | .943 | .771 | .943 | 1.000 |
| Qwen | .867 | .950 | .883 | .678 |

在每个 model-domain-checkpoint 内减去同期四种训练设置均值后：

| 模型 | \(\mathrm{KL}_C\) | \(\mathrm{KL}_B\) | \(\mathrm{KL}_B-\mathrm{KL}_C\) |
|---|---:|---:|---:|
| Llama | .875 | .902 | -.209 |
| Qwen | .573 | .440 | -.183 |

因此可以写：

> 相对功能压缩同时追踪 CoT 与 boxed-answer 区域各自的无符号输出移动，
> 但不决定输出移动主要分配到推理过程还是最终答案。

必须把

\[
\Delta\mathrm{KL}_{B-C}
=\mathrm{KL}_B-\mathrm{KL}_C
\]

称为 **regional KL contrast**，不能把它命名成新的 KL divergence。

---

## 2. 正文需要修改的部分

以下修改需在中文稿和英文稿中完全同步。

### 2.1 摘要

不加入具体相关系数或胜负计数，但需要检查当前摘要是否只写了
“格式、答案和终止区域”。若是，应扩展为：

> 推理、格式、答案和终止区域

构念比较可概括为：

> 双模型严格共同网格显示，该功能坐标保留了 raw activation spectrum
> 与 source-principal 权重位置不能单独提供的轨迹和输出移动信息。

摘要仍不写“全面优于所有权重指标”。

### 2.2 引言

当前三点贡献结构不需要改变，但需要两处收紧：

1. 第一项贡献的“双模型”证据已经完整，不再保留 Qwen 待补语气；
2. 第三项贡献从
   “格式、答案和终止区域”
   扩展为
   “推理、格式、答案和终止区域”。

引言仍只给概括性结果，不堆入 `.875/.573` 或八项逐格数值。

### 2.3 相关工作

章节结构不变。

“权重、激活与功能空间”小节可以更有根据地说明：

- raw activation block 单独不能复现 \(C_5\) 的双模型 held-out 结果；
- strict \(p_k\) 仍是强基线，并保留 Qwen absolute-NLL 反例；
- 本文贡献来自不同的观察空间，而不是通过否定全部权重或激活指标建立。

不需要为 \(\mathrm{KL}_C\) 新增 related-work 小节。

### 2.4 方法：区域输出与比较坐标

当前正文写的是：

> KL 使用 \(F/A/T\) 与 \(B/T\)。

应改为：

\[
\text{MMLU-Pro}: F/A/T,\qquad
\text{MATH500}: C/B/T.
\]

正文需要定义：

- \(F\)：正确答案标签之前的固定格式 tokens；
- \(A\)：正确 answer-choice label token；
- \(C\)：固定 reference solution 中 boxed answer 之前的 CoT；
- \(B\)：token-clean 的完整 boxed-answer span；
- \(T\)：termination token。

同时执行既有语言决定：

- `gold completion` 改为 `fixed reference completion`；
- `gold option` 改为 `correct answer-choice label`；
- `gold token` 改为 `reference token`。

正文只保留区域定义、KL方向和 macro 聚合的必要信息；
完整 mask、tokenizer 边界和样本协议进入 Supplement。

---

## 3. 结果第一小节与构念比较表

### 3.1 “功能谱提供不同于纯权重与纯激活的轨迹信息”

当前正文只报告 Llama common grid，必须改为双模型。

正文建议按以下顺序叙述：

1. raw activation、TPNT/PABS/NSS 等构念分别观察什么；
2. 严格共同状态上比较 \(A/C_5/P_{k,5}\)；
3. 两模型中 \(C_5\) 对 raw activation 的 primary comparisons 均更强；
4. strict \(p_k\) 保留 Qwen absolute-NLL 反例；
5. OPD AUC 的排序优势不等同于分类校准优势。

不要在自然语言中连续使用“8/8、7/8”作为版本账本式写法。
正文可改写为：

> 在两个模型的三个输出目标和训练设置识别中，
> \(C_5\) 均优于 raw activation block；相对 strict \(p_k\)，
> 唯一例外是 Qwen 的 absolute NLL。

具体计数和数值由表格或 Supplement 承担。

### 3.2 表 `tab:identification`

当前 Qwen 行为：

> RR5 共同网格结果待补

必须删除并填入：

| 模型 | Raw activation \(A\) | Functional \(C_5\) | Weight \(P_{k,5}\) |
|---|---:|---:|---:|
| Llama OPD held-out AUC | .556 | **.743** | .688 |
| Qwen OPD held-out AUC | .595 | **.708** | .521 |

表注改为：

> 两个模型均使用64个严格共同状态、相同 equal-5 模块集合和
> checkpoint-held-out folds。

这张表只负责“训练设置识别”，不把全部 KL/NLL 回归结果塞入同一表。
三个 output targets 的完整 \(A/C_5/P_{k,5}\) 表进入 Supplement，
正文结果段给出概括。

---

## 4. Figure 3 的修改

### 4.1 总体结构

仍采用此前已经确认的 `1 x 2`：

- Panel A：相对功能压缩与区域无符号 KL 的热力图；
- Panel B：signed readout 与真实行为边界。

原 token-region 示意 panel 删除。区域定义由 Methods 负责，
图注只解释缩写和统计单位。

### 4.2 Panel A

列结构更新为：

\[
\underbrace{F,\ A,\ T}_{\mathrm{MMLU-Pro}},
\qquad
\underbrace{C,\ B,\ T}_{\mathrm{MATH500}}.
\]

显示每个 model×training-setting 内，
\(c_{.05}^{(5)}\) 与区域 KL 跨 checkpoint 的 Spearman。

保持此前确认的作图要求：

- 使用灰度可辨的 sequential palette；
- 每格保留数值；
- 标签水平排列；
- 优先使用完整行名，如 `Llama / off-KD`；
- 若保留 L/Q，图注明确 L=Llama、Q=Qwen；
- 不能依赖颜色单独表达结论；
- 重新导出并检查右侧不再裁切；
- 最终字体不小于9 pt、无 Type 3、media box 正确。

### 4.3 Panel B

Panel B 的核心任务不变：

> 展示 signed regional readout 与自由生成行为的对应方向随模型和训练设置改变。

加入 \(\mathrm{KL}_C\) 后，需要在正文解释 Panel A 与 Panel B 的逻辑关系：

- Panel A：各区域移动多少；
- Panel B：移动方向、区域间分配和行为实现是否一致。

Panel B 不需要强行加入 \(\mathrm{KL}_{B-C}\)。
MATH 的 regional KL contrast 可由正文一句和 Supplement 表承担。

### 4.4 Figure 3 图注

图注至少定义：

- MMLU \(F/A/T\)；
- MATH \(C/B/T\)；
- 每格是固定 model×training-setting 内跨 checkpoint 的 Spearman；
- L/Q 的模型含义（若使用缩写）；
- signed panel 的两种行为轴。

计算细节不堆入图注，放在 Methods。

---

## 5. 无符号输出结果小节

### 5.1 主统计范围

原正文主区域是：

- MMLU \(F/A/T\)；
- MATH \(B/T\)；
- 共5类真实区域；
- 48条 model×training-setting×region 轨迹关系。

加入 \(\mathrm{KL}_C\) 后变为：

- MMLU \(F/A/T\)；
- MATH \(C/B/T\)；
- 共6类真实区域；
- 56条逐轨关系。

因此以下数字必须统一重算：

- 中位 Spearman；
- 达到预设强相关阈值的关系数；
- checkpoint-demeaned 的范围；
- 正文和 Supplement 中所有旧分母。

不能继续无说明地沿用“48条”“41/48”。

### 5.2 新增正文结论

正文至少加入：

1. Math CoT 区域的逐轨相关在两个模型、四种训练设置中均为正；
2. checkpoint-demeaned 后，Llama/Qwen 的
   \(c_\varepsilon\)-\(\mathrm{KL}_C\) Spearman 为 .875/.573；
3. \(\mathrm{KL}_C\) 和 \(\mathrm{KL}_B\) 各自与功能压缩正相关；
4. \(\mathrm{KL}_B-\mathrm{KL}_C\) 与压缩弱负相关；
5. 因此功能压缩更稳定地追踪区域移动幅度，
   不决定输出移动在 CoT 和 final answer 之间的分配。

### 5.3 旧整段 reference stream

旧 \(R\)-stream cumulative KL/NLL 继续作为 Supplement sensitivity。
它不因加入 \(C\) 而失效，也不能与 \(P/F/A/C/B/T\) 区域记号混用。

正文仍可保留一句：

> 将完整 fixed reference completion 合并为单一区域时，
> 无符号关系仍然存在。

---

## 6. 区域预测表与必须重算的统计

### 6.1 表 `tab:regional-prediction`

MATH 行从：

\[
B/T
\]

扩展为：

\[
C/B/T.
\]

正文表只放真实区域，不建议把
\(\mathrm{KL}_B-\mathrm{KL}_C\) 与真实 KL divergence 混在同一列。
regional KL contrast 进入 Supplement。

### 6.2 理论/实验侧必须补交

论文更新前还需要得到以下正式结果：

1. \(C_5\) 对 \(\mathrm{KL}_C\) 的 checkpoint-held-out
   \(R^2\)、MAE 和 prediction Spearman；
2. \(P_{k,5}\) 对 \(\mathrm{KL}_C\) 的相同结果；
3. 如将 \(\mathrm{KL}_B-\mathrm{KL}_C\) 纳入 grouped target family，
   还需对该 contrast 重跑同一比较；
4. expanded target family 下的新胜负总数；
5. 原“区域 KL 上 \(R^2\)/MAE 10/12胜”的新分母与新数值；
6. \(\varepsilon\in\{.01,.025,.05,.10\}\) 下的 Math-\(C\)
   关联敏感性；
7. 逐 outer fold 结果，尤其是低目标方差 checkpoint 的
   MAE/Spearman 与不稳定 \(R^2\)。

在这些结果返回前：

- 可以确定论文结构和区域定义；
- 不可以预填新 held-out 胜负数；
- 不可以继续把旧10/12描述成扩展后 target family 的结果。

---

## 7. Signed Readout 小节

结构不需要重写，但必须清楚区分：

\[
\Delta\mathrm{KL}_{B-C}
=\mathrm{KL}_B-\mathrm{KL}_C
\]

与

\[
\Delta\mathrm{NLL}_{B-C}
=\Delta\mathrm{NLL}_B-\Delta\mathrm{NLL}_C.
\]

前者比较两个区域的无符号 full-vocabulary movement；
后者比较 reference-token likelihood 的有符号相对变化。

建议的逻辑过渡是：

> 功能压缩与 CoT、答案和终止区域各自的无符号移动稳定同步；
> 但区域间的相对移动、reference-token likelihood 的改善/恶化方向，
> 以及自由生成行为并不存在跨模型、跨训练设置的固定符号映射。

这样能把 Figure 3 的两个 panel 连接起来，而不把所有结果压成一个行为机制。

---

## 8. 讨论、局限性与结论

### 8.1 讨论

增加或强化以下判断：

> 功能压缩描述主要本层输出能量集中到多少模式，
> 并追踪各输出区域移动了多少；
> 它不决定这些移动主要落在哪个区域，也不决定移动对任务是改善还是损伤。

这使“模式数量、区域分配、读出效价、序列级实现”形成清晰层级。

### 8.2 局限性

保留：

- 每种模型和训练设置只有一条独立训练轨迹；
- 每模型只有四个 RR5/QRAW checkpoint groups；
- AUC 不代表校准；
- 当前结论是 LoRA、headline layer 和固定 probe 下的结果。

删除或更新任何暗示“Qwen raw activation common grid 尚未完成”的句子。

### 8.3 结论

将“格式、答案和终止”扩展为：

> 推理、格式、答案和终止区域。

结论可进一步明确：

> 功能压缩与区域无符号输出移动之间的稳定关系，
> 和区域分配及行为效价的模型依赖性共同说明，
> 该指标提供的是裸权重与最终行为之间的功能观察层，
> 而不是端到端行为充分统计量。

---

## 9. Supplement 的详细修改

### 9.1 数学基础

无需因 \(\mathrm{KL}_C\) 改动。

### 9.2 实验与估计协议

加入：

- Math \(C\) span 的精确定义；
- token-clean mask 构造；
- 500题 sample-macro 聚合；
- 62 states、31,000 sample rows；
- BF16 forward、FP32 `log_softmax`/KL；
- \(D_{\mathrm{KL}}(p_0\Vert p_t)\) 方向；
- boxed-answer tokenizer 边界处理；
- artifact 和 manifest 路径。

### 9.3 区域无符号输出

加入：

- \(\mathrm{KL}_C\) 的八条逐训练设置相关；
- checkpoint-demeaned 的 Llama/Qwen `.875/.573`；
- 与 \(\mathrm{KL}_B\) 的并列比较；
- regional contrast
  \(\mathrm{KL}_B-\mathrm{KL}_C\) 的 `-.209/-.183`；
- expanded 56-relation 总统计。

### 9.4 严格构念比较

当前只展示 Llama 的 `tab:supp-rr5` 必须扩为双模型，完整登记：

- \(A/C_5/P_{k,5}\) standalone；
- \(A+C_5\)、\(P_{k,5}+C_5\) 的增量块；
- cumulative KL、absolute NLL、signed NLL；
- OPD AUC、log-loss、balanced accuracy；
- pooled OOF、逐 fold MAE/Spearman；
- Qwen absolute-NLL 与分类校准反例。

### 9.5 新区域的 held-out 比较

在实验侧完成第6.2节所列重算后，Supplement 加入：

- \(\mathrm{KL}_C\) 的 \(C_5/P_{k,5}\) 对照；
- 如采用，\(\mathrm{KL}_B-\mathrm{KL}_C\) 对照；
- expanded target-family 胜负汇总；
- epsilon sensitivity。

### 9.6 Signed Readout

把 KL contrast 与 signed NLL contrast 分表或分段展示，
不得使用相同符号而不加解释。

---

## 10. 不受本轮更新影响的部分

以下内容不因 QRAW 或 \(\mathrm{KL}_C\) 改变理论结构：

- 域条件功能谱的白化坐标解释；
- 输出二阶矩恒等式与最优低秩近似；
- non-QK equal-5 的定义；
- OPD 共同早期压缩支配；
- margin 与 NCD；
- \(\alpha=.5\) exposure 对照；
- OPD--frozenSelf0-KD refresh 对照；
- off-KD--seqKD support--readout separation；
- Figure 1 的主轨迹证据；
- Figure 2 的 exposure/refresh 证据。

这些部分可能因统一语言或此前独立讨论而修改，
但不是本轮 QRAW/\(\mathrm{KL}_C\) 更新直接造成的。

---

## 11. 最终执行顺序

正式“编写论文”时建议按以下顺序执行：

1. 理论/实验侧补交 Math-\(C\) 的 strict \(C_5/P_{k,5}\)
   held-out 和 epsilon-sensitivity 结果；
2. 冻结 expanded target family 和新分母；
3. 更新 Methods 的区域定义；
4. 更新构念比较正文和 `tab:identification`；
5. 重绘 Figure 3；
6. 更新区域输出正文和 `tab:regional-prediction`；
7. 更新 signed-readout 过渡、讨论与结论；
8. 扩写 Supplement 的 QRAW 与 Math-\(C\) 结果；
9. 同步英文稿；
10. 编译、检查页数、字体、灰度、交叉引用和 PDF media box；
11. 最后重新打包 Overleaf ZIP。

在作者明确说“编写论文”之前，本文件只作为累计修改记录使用。
