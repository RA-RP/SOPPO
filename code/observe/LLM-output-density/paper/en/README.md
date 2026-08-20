# AAAI 2027 English Manuscript

```yaml
canonical_manuscript: main_en.tex
structure_source: ../zh/main_zh.tex
fact_source: ../../mypaper/theory/human_read-ch.md
supplement: supplement_en.tex
```

## Files

- `main_en.tex`: final English manuscript source.
- `results_en.tex`, `results_floats_en.tex`: results prose and float definitions.
- `supplement_en.tex`: independently compiled supplementary material.
- `references.bib`: bibliography.
- `figures/`: committed vector assets required by the manuscript and supplement.
- `aaai2027.sty`, `aaai2027.bst`: the formatting files required for a self-contained build.

The English manuscript is a downstream product. Theory, experimental numbers, and claim
boundaries must first be updated in `../../mypaper/theory/human_read-ch.md`; structural
changes should first be settled in the Chinese working manuscript.

Build from the repository root with `make paper-en` and `make paper-supplement`.

