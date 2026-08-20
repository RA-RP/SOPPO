# Current Theory

```yaml
view_type: current_workspace_section
section: theory
status: active
last_organized: 2026-06-16
```

This directory is the current theory and metric-definition entry point. It should remain self-contained: a reader should not need to open an old cycle merely to understand the current hypothesis or geometry metrics.

## Current Files

| File | Role | Main Source |
|---|---|---|
| `paper_thesis_framework.md` | **The paper's constitution**: frozen thesis sentence (2026-07-13) + claim tree (C1–C8 with evidence tiers ★/◐/○) + related-work positioning + pending-dependency ledger + section skeleton. Writing phase draws claims ONLY from this tree. | Cycle 09 thesis discussion |
| `current_theory_update.md` | Layered hypothesis: base-aware K1 (entry condition) → Cycle 06 SFT-feasibility precondition → Cycle 07 trajectory mechanism → Cycle 08 active H-OPD test → Cycle 09 Tier-0 contribution repositioning after the scoop (layered framing: sharpened observation-space core / mechanism-chain hypothesis / conditional LoRA supplement / measurement critique) + mini-round rulings & claim boundaries 1–20. | Cycle 05–09 theory updates |
| `geometry_metric_definitions.md` | GetSlice metric definitions + interpretation limits; Cycle 07 adds trajectory/subspace-stability metrics; Cycle 08 OverlapLift matched-LoRA relative signal (later retracted, see post-hoc note); Cycle 09 activates measured rotation θ_r (UV on), layer-resolved (L18-headline) reading, dual-ER constructs vs 2605.30524, probe battery v2.1, and the adapter-first fp32(B@A) weight-measurement correction. | Cycle 03 definitions + Cycle 04–09 usage |

## Maintenance Notes

- Stable definitions should be preserved across cycles.
- If a new experiment changes the interpretation of a metric, update the interpretation notes instead of deleting the metric.
- Subspace stability: γ_r = σ_r − σ_{r+1} (Davis–Kahan) is the capacity proxy; from Cycle 09 the measured rotation θ_r is also available (UV saving ON). Never write γ_r as if it were θ_r.
- Cycle 07 reframes OOD-drop as a checkpoint/optimization-stage property; do not write it as an intrinsic method property.
- OverlapLift is retracted as an on/off-principal signal (Cycle 08 post-hoc); the weight-direction metric is ρ per `result/cycle08_rho_metric_spec.md`, computed adapter-first as fp32(B@A) (Cycle 09 correction). No on/off-principal claim before A09 resolves.
- Any "compression" statement must name its construct: whitened output-relevant spectrum (ours, SVD-LLM lens) vs raw representational ER (2605.30524).
