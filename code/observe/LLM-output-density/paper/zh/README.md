# AAAI 2027 Chinese Working Manuscript

```yaml
canonical_manuscript: main_zh.tex
fact_source: ../../mypaper/theory/human_read-ch.md
target_style: AAAI 2027 anonymous submission
supplement: supplement_zh.tex
```

## 文件

- `main_zh.tex`：论文结构与中文论证的工作主稿。
- `results_zh.tex`、`results_floats_zh.tex`：结果正文与图表定义。
- `supplement_zh.tex`：与正文分离编译的中文补充材料。
- `references.bib`：参考文献数据库。
- `figures/`：正文和补充材料编译所需的矢量图。
- `aaai2027.sty`、`aaai2027.bst`：自包含编译所需的 AAAI 格式文件。

理论、实验事实和 claim 边界首先进入
`../../mypaper/theory/human_read-ch.md`；本目录只负责论文结构与中文表述。
英文稿在中文结构阶段性稳定后同步。

在仓库根目录运行 `make paper-zh` 进行编译。

