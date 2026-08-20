# AAAI OPD Stability Draft 01

## Material Passport

```yaml
artifact_type: paper_draft
origin_skill: academic-research-suite/academic-paper
draft_stage: earliest_full_draft
target_style: AAAI anonymous submission style
source_cycles:
  - research_cycles/cycle_03_trl_opd_like
  - research_cycles/cycle_04_opd_stability_gain
date: 2026-06-15
status: draft_not_submission_ready
```

## 文件说明

这个文件夹是论文草稿区，不属于代码实现区，也没有修改 `mycode`。

当前文件：

```text
main.tex          # AAAI 风格英文论文初稿
references.bib    # BibTeX 引用
figures/          # 后续放图
notes/            # 修订、审稿、证据检查记录
```

## 编译说明

`main.tex` 使用 AAAI-26 风格写法：

```latex
\usepackage{aaai26}
\bibliographystyle{aaai26}
```

因此后续编译前需要把 AAAI 官方 author kit 里的这些文件放到本目录：

```text
aaai26.sty
aaai26.bst
```

AAAI-26 官方说明要求双栏、匿名、US Letter、技术正文最多 7 页，参考文献和 reproducibility checklist 可另计。当前草稿按匿名提交语气写，没有作者、单位、致谢。

## 当前论文定位

当前版本不是最终投稿稿。它的核心结论是谨慎的：

```text
TRL OPD-like feasibility result with preliminary K1 support.
```

允许写：

```text
Under matched GSM8K gain, OPD-like training showed lower OOD-lite degradation
and lower layer-14 activation-conditioned spectral drift than continued SFT.
```

不允许写：

```text
OPD improves GSM8K.
K1 is proven.
This is full OPD replication.
This establishes broad OOD generalization.
```

## 下一步修订优先级

1. 等 Cycle 04 结果回来后，替换或扩展当前 Cycle 03 preliminary result。
2. 加入真实 figures，而不是只用表格。
3. 增加 multi-layer geometry 和多 OPD 设置/seed。
4. 做 citation audit，确认每个引用支撑的句子不过度外推。
5. 使用 AAAI author kit 编译并检查页数。

