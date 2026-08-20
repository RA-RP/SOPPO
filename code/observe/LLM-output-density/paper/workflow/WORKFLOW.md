# 论文编写工作流

本文档规定当前论文项目的真实工作流，以及各文件之间的权威关系。核心原则是：**理论事实、中文论证和英文成品分层管理，不在三个文件中同时独立修改。**

## 一、三个层级

### 1. `human_read-ch.md`：理论与实验事实层

权威文件：

`mypaper/theory/human_read-ch.md`

该文件是论文的事实来源，负责记录：

- 核心研究问题、理论解释和机制假设；
- 指标的正式定义、适用范围和构念边界；
- 实验设计、训练条件、评测协议和统计口径；
- 已确认的实验结果和具体数值；
- 证据等级、可支持的主张和禁止使用的过度主张；
- 尚未完成的实验、当前局限和待验证问题；
- probe、arm、checkpoint 和数据集的统一命名。

任何涉及理论、实验事实或主张强度的修改，都必须先进入该文件。论文正文不得独立创造与 `human_read-ch.md` 不一致的新解释。

### 2. `main_zh.tex`：论文结构与表述层

权威文件：

`paper/zh/main_zh.tex`

该文件是论文讨论和迭代的主要工作区，负责：

- 确定论文标题、章节结构和论证顺序；
- 决定哪些证据进入正文、独立补充材料或暂不使用；
- 修改中文表述、段落衔接和概念解释；
- 组织图、表、公式和相关工作；
- 控制论文篇幅、叙事重点和审稿人阅读体验；
- 逐章讨论并形成稳定的论文版本。

`main_zh.tex` 可以压缩、重排或省略 `human_read-ch.md` 中的内容，但不能改变其中已经冻结的事实。若在写作过程中发现理论或实验描述需要改变，应先回到 `human_read-ch.md` 修正，再继续修改中文稿。

当前主稿采用“正文以 Conclusion/结论结束，随后自然接续参考文献”的结构，
不在 `main_zh.tex` 或 `main_en.tex` 中使用 `\appendix`。主张成立所必需的定义、
关键证明结论、稳健性和边界必须拆入相应正文章节；只有不影响正文自洽的扩展材料，
才可在后续建立独立的 Supplementary Document。

当前独立补充材料已经建立：

- `paper/zh/supplement_zh.tex`
- `paper/en/supplement_en.tex`

Supplement 与正文分别编译和提交，不通过 `\input`、`\include` 或主稿末尾的
`\appendix` 拼接。当前 Supplement 采用四层结构：

- Appendix A：数学基础与证明；
- Appendix B：实验、估计、数值和统计协议；
- Appendix C：完整功能轨迹、模块/阈值/层级稳健性与训练对照；
- Appendix D：完整输出指标、构念比较、signed readout 和行为边界。

正文只保留理解方法和支持主要结论所必需的关键等式与主证据；扩展证明、完整分解、负结果
和审计表进入 Supplement。正文通过 `Supplementary Sec. A.x--D.x` 关联独立补充材料，
但不依赖该关联才能读懂主要论证。

### 3. `main_en.tex`：最终英文成品层

权威文件：

`paper/en/main_en.tex`

该文件只用于生成最终英文论文，不参与日常理论讨论和结构探索。它负责：

- 将已经稳定的中文论文转写为学术英语；
- 按 AAAI 模板完成最终英文排版；
- 统一英文术语、语法、引用和图表标题；
- 生成提交、外部评审或预览所需的英文 PDF。

不得只在 `main_en.tex` 中修改理论、实验数字、章节逻辑或主张边界。若英文评审暴露出实质问题，应先把问题带回 `human_read-ch.md` 或 `main_zh.tex` 解决，随后重新同步英文稿。

## 二、唯一同步方向

论文内容按照以下方向流动：

```text
原始实验产物
    ↓
human_read-ch.md
理论、事实、协议与证据边界
    ↓
main_zh.tex
论文结构、论证和中文表述
    ↓
main_en.tex
最终英文成品
    ↓
AAAI PDF / 外部评审版本
```

原则上不允许反向维护。英文稿中发现的问题可以触发上游修改，但不能把英文稿直接变成新的事实来源。

## 三、修改入口

| 修改内容 | 首先修改的文件 |
|---|---|
| 新实验结果或数字修正 | `human_read-ch.md` |
| 训练、评测或统计协议变化 | `human_read-ch.md` |
| 指标定义、理论解释或机制假设 | `human_read-ch.md` |
| 主张增强、削弱或撤回 | `human_read-ch.md` |
| 章节顺序和论证结构 | `main_zh.tex` |
| 中文措辞和概念解释 | `main_zh.tex` |
| 图表选择、正文篇幅和内容取舍 | `main_zh.tex` |
| 英文翻译和学术英语润色 | `main_en.tex` |
| AAAI 英文排版与最终格式检查 | `main_en.tex` |

## 四、逐章修改流程

每一章按以下顺序处理：

1. 从 `human_read-ch.md` 提取该章允许使用的事实、定义和证据。
2. 在 `main_zh.tex` 中讨论并修改该章的结构与中文表述。
3. 检查该章的每项主要主张是否能回指到 `human_read-ch.md`。
4. 中文章节确认稳定后，再同步到 `main_en.tex`。
5. 英文同步只做忠实转写和语言优化，不重新设计理论故事。

当前建议的章节顺序为：

```text
Method
→ Results
→ Introduction
→ Related Work
→ Discussion and Limitations
→ Abstract and Title
→ English finalization
```

## 五、Reviewer 反馈处理

外部 reviewer 或 agent reviewer 的意见先分类，不直接修改英文成品：

- **理论、定义或实验问题**：先修改 `human_read-ch.md`；
- **论证、章节和信息组织问题**：修改 `main_zh.tex`；
- **纯英文语法、术语或排版问题**：可以修改 `main_en.tex`；
- **同时涉及事实和表达的问题**：先处理事实层，再处理中文稿，最后重新同步英文稿。

英文评审版本只是诊断工具，不改变三层文件的权威顺序。

本轮 reviewer 反馈中与论文编写有关的行动项统一登记在：

`paper/workflow/REVIEW_ROUND_WRITING_ACTIONS_2026-07-26.md`

该台账只管理论证、表述、复现说明、图表和相关工作。新实验、理论变化和统计结论仍先进入
`human_read-ch.md`，不得由写作台账直接写入正文。

## 六、冲突处理

当三个文件出现不一致时，按以下优先级处理：

```text
实验原始产物
> human_read-ch.md
> main_zh.tex
> main_en.tex
```

- 数字冲突时，以经过审计的实验产物和 `human_read-ch.md` 为准；
- 理论边界冲突时，以 `human_read-ch.md` 为准；
- 章节结构冲突时，以当前确认的 `main_zh.tex` 为准；
- 英文稿不得覆盖上游文件中的决定。

## 七、当前工作约定

- 日常讨论默认围绕 `human_read-ch.md` 和 `main_zh.tex` 展开。
- `main_zh.tex` 是逐章修改时直接阅读和编辑的论文。
- `main_en.tex` 只在中文论文达到阶段性稳定后更新。
- 每次生成新的英文评审版本前，应完成一次事实、数字、引用和章节同步检查。
- 英文稿不与中文稿并行自由演化。
