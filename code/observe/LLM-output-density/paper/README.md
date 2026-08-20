# Paper Workspace

This directory contains the source-controlled paper workspace.

- `../mypaper/theory/human_read-ch.md` is the source of theory and experimental facts.
- `zh/main_zh.tex` controls paper structure and Chinese-language argumentation.
- `en/main_en.tex` is the final English manuscript.
- `en/supplement_en.tex` and `zh/supplement_zh.tex` are independently compiled supplements.
- `workflow/` records the writing workflow, reviewer actions, and accepted revision plans.
- `scripts/plot_aaai27_figures.py` rebuilds the committed vector figures from `../paper_artifacts/`.

Build from the repository root:

```bash
make paper-en
make paper-supplement
make paper-zh
make checklist
```

LaTeX build products are ignored. The committed PDF files inside `en/figures/` and
`zh/figures/` are source assets, not compiled manuscript PDFs.

