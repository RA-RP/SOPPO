# Compact Paper Artifacts

This directory is a curated, numeric subset of the original Cycle 09 result store. It
supports continued paper editing, figure regeneration, and table-level auditing without
placing multi-gigabyte raw artifacts in Git.

## Included Evidence Families

- `equal5_non_qk/`: equal-5 functional trajectories, dominance, NCD, nested metrics, and weight baselines.
- `fat_outlink_round1_v2_link_equal5/`: regional KL/NLL joins and checkpoint-held-out comparisons.
- `qwen_raw_activation_rr5_q64/`: corrected Qwen raw-activation common-grid comparison.
- `reviewer_robustness/`: threshold, centering, continuous-rank, top-32, and paired-bootstrap audits.
- top-level Cycle 09 files: alpha exposure, frozen-self, M6, and D11 TPNT/PABS/NSS summaries.

## Excluded

- model and adapter weights;
- rollout or teacher-response JSONL files;
- prompt text and generated completions;
- activation caches and per-token tensors;
- large raw spectrum and tail-energy dumps;
- original training outputs.

`MANIFEST.csv` records every included file's size and SHA-256 hash. `SOURCE_MAP.csv`
records the original repository-relative location so that a local full artifact archive
can be reconciled with this compact copy.

The final figure script reads this directory through repository-relative paths:

```bash
python paper/scripts/plot_aaai27_figures.py
```

