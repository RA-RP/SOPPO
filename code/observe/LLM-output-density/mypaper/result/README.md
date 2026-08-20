# Current Results

```yaml
view_type: current_workspace_section
section: result
status: active
last_organized: 2026-06-16
```

This directory is the highest-level, most readable result entry point. It uses an incremental format: each file answers one question about the project's findings. Detailed numbers and evidence are in `../local_experiment_results/<cycle_name>/`.

## Current Files

| File | Question Answered |
|---|---|
| `incremental_log.md` | What did each cycle add or change? |
| `current_picture.md` | What do all currently valid results say together? |
| `claims_allowed.md` | What conclusions are supported, and what is not allowed? |

## Maintenance Notes

- `incremental_log.md` is append-only: add a new section per cycle, never rewrite past sections.
- `current_picture.md` reflects the current aggregate state: update it when new results land or old ones are corrected.
- `claims_allowed.md` is a living boundary: update allowed/not-allowed claims as results change.
- Do not erase negative results. Future cycles may qualify them, but historical claim boundaries should be traceable.
- Gate metrics and exploratory metrics must be kept separate.
- Link to `../local_experiment_results/` for raw tables and figures rather than embedding them here.
