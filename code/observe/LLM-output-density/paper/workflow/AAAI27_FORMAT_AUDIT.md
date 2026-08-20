# AAAI 2027 格式审计

本记录依据 `AuthorKit27/AnonymousSubmission2027.tex` 与配套 `aaai2027.sty`，
用于约束主稿、补充材料和绘图产物。它不进入投稿源码包。

## 模板与页面

- 使用 `\documentclass[letterpaper]{article}` 和
  `\usepackage[submission]{aaai2027}`。
- 仅使用 PDFLaTeX；页面为 US Letter、双栏。
- 不修改模板的页边距、栏宽、字号、字距、行距、标题间距或基线。
- 不使用 `geometry`、`setspace`、`titlesec`、`\vspace`、强制换页等排版覆盖。
- 正文与 Supplement 独立编译；正文不包含 `\appendix`。
- 英文匿名稿不加载 CJK；中文稿仅用于内部迭代，不作为正式匿名投稿文件。

## 图表

- 图片只使用 PDF/PNG/JPG；当前所有论文图均为 PDF 1.5 矢量文件。
- 图片按最终宽度 7 inch 直接导出，不依靠 LaTeX 裁剪或缩放内部文字。
- 可见图中文字不小于 9 pt，线宽不小于 0.5 pt。
- 字体已转换为矢量轮廓；`pdffonts` 不报告 Type 3、CID 或未嵌入字体。
- 图意不只依赖颜色，同时使用 marker、线型、标签或表格。
- 图题位于图下方；表题位于表下方。
- 表格使用 10 pt，必要时使用 9 pt；不使用 `\resizebox` 缩放文字。

## 引用与 PDF

- 参考文献自然接在 Conclusion 之后，无强制另起一页。
- 使用 AAAI 2027 BibTeX 样式，不使用 `hyperref` 或内嵌 PDF bookmark。
- 英文主稿、英文 Supplement 均使用嵌入的 Type 1 字体。
- 编译日志不得包含 overfull box、未定义引用或未解析 citation。

## 当前产物

- 英文主稿：`aaai_opd_sft_trajectory_en/main_en.pdf`，7 页。
- 中文主稿：`aaai_opd_sft_trajectory_zh/main_zh.pdf`，7 页。
- 英文 Supplement：`aaai_opd_sft_trajectory_en/supplement_en.pdf`，6 页。
- 中文 Supplement：`aaai_opd_sft_trajectory_zh/supplement_zh.pdf`，5 页。

最终上传前仍需按当年投稿系统要求执行官方 PDF checker，并确认匿名信息、文件大小和
Supplement 是否作为独立附件提交。
