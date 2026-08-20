# Geometry Probing Protocol (GetSlice)

```yaml
view_type: stable_methodology
section: exp
status: active
maintenance_rule: edit_in_place_as_methodology_changes_not_per_cycle
grounded_in:
  - /root/LLM-output-density/experiments/opd_sft_h1/scripts/run_opd_minimal_closure_v2.py (run_getslice_cross, build_probes_multi)
  - /root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/geometry_reader.py
  - /root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/geometry_metrics.py
  - /root/LLM-output-density/mypaper/theory/geometry_metric_definitions.md
last_organized: 2026-06-16
```

This file describes **how the raw data for the geometry metrics is collected**
(what gets fed into the model, at which layers, for which probe). For what
each resulting *metric* means and how to interpret it, see
`../../theory/geometry_metric_definitions.md` — that file is metric
definitions, this file is data collection.

## What GetSlice Computes

For a chosen model and a chosen probe (a set of input/output text), GetSlice
runs a forward pass, whitens the activations at the target layer/module
(Cholesky-jittered covariance whitening, `cholesky_jitter=1e-5`,
`epsilon=1e-3`, `svd_singular_floor=0.0`), and takes the SVD of the whitened
matrix to get a singular spectrum (`sMat_*.json` for a supervision/"S" probe,
`xMat_X.json` for a shared/"X" probe). `model_dtype=float16` for the forward
pass, `uv_dtype=float32` for the SVD. **UV/sketch artifacts are not saved by
default** (`save_s_uv_path=None`, `save_x_uv_path=None`) — this is why
principal-angle/subspace-overlap claims are unavailable in Cycle 04/05; a
future cycle that needs them must explicitly turn this on before training.

## Which Layers And Modules

- 3 layers per run, chosen as early/mid/late out of Qwen3-1.7B's 28
  transformer layers: **layer 6, layer 14, layer 22** (`GEOMETRY_LAYERS` in
  `run_opd_stability_gain_cycle04.py`).
- 7 modules per layer, matching the LoRA target modules exactly:
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Sequence length for the probe forward pass: `seqlen=512` (full run),
  `seqlen=64` (smoke). Number of probe samples per (model, probe) pair:
  `s_nsamples=x_nsamples=16` (full run), `4` (smoke).

## The Dual-X / Per-Model-S Design

This is the part most likely to be misread if undocumented, so it is spelled
out explicitly:

- **X probes are frozen and shared across every model in the comparison.**
  Both are generated once, by `theta0`, and reused as-is for every model's X
  measurement:
  - `X_prompt`: `theta0` rollout completions appended after the probe prompt
    (i.e. "prompt + theta0's answer to it").
  - `X_bos`: `theta0` freeform generation starting from BOS with **no**
    prompt at all.
  - Because X is frozen, any difference in the X-conditioned spectrum across
    models is purely due to the model's own weights reacting to the same
    fixed input, not to different inputs.
- **S probes are per-model and reflect what that model actually
  produces/trains on, not a shared input:**
  - `theta0`'s S = the **teacher**'s (Qwen3-4B) rollout on the probe prompts.
  - Each OPD arm's S = **that arm's own** student rollout on the probe
    prompts (i.e. self-generated, on-policy text).
  - Each SFT control's S = a slice of **its own real training data**
    (the actual supervised teacher-completion text it trained on), not a
    rollout at all.
  - This means "drift from `theta0`" on the S axis is comparing each model
    against a baseline built from a *different kind* of text per model role
    (teacher rollout vs self-rollout vs static training slice) — this is a
    known asymmetry, not an oversight, and should be kept in mind when
    comparing OPD-arm drift to SFT-arm drift.
- Probe prompts are drawn from a NuminaMath probe pool
  (`n_probe=32` for the full run, `4` for smoke), part of the same unified
  prompt pool described in `model_and_data.md`.
- Full cross matrix: every model is measured against every relevant S source
  and against both X variants, at all 3 layers × 7 modules — this is what
  "S×model cross matrix" / "three-layer geometry" refers to in result
  documents.

## Output Layout

```text
getslice/outputs/{model_source}/step_{size}/S__{probe_source}/numina_math_probe/layer_N/sMat_*.json
getslice/outputs/{model_source}/step_{size}/X__{prompt|bos}/layer_N/xMat_X.json
```

`geometry_reader.py` parses these JSON files into long-format rows (layer,
module, singular values); `geometry_metrics.py` computes `effective_rank`,
`spectral_gap`, `spectral_drift_from_theta0_diag`, `X_S_spectrum_level_gap`
from those rows — see `../../theory/geometry_metric_definitions.md` for the
exact formulas and how to read each metric.

---

## Cycle 07 Geometry Protocol (2026-06-24): Multi-Checkpoint Trajectory

Cycle 07 changes the geometry paradigm from **cross-arm comparison** (OPD vs SFT)
to **single-trajectory multi-checkpoint sampling** (10 checkpoints along the dip-and-recovery).

### Reference Checkpoint

`step_000` = untrained Qwen3-4B-Base. **No theta0 exists in Cycle 07.**
All drift metrics are computed relative to step_000. The X probe is generated once
from step_000 (replacing the prior theta0-generated X) and reused across all 10 checkpoints.

### Layer Selection

Qwen3-4B-Base has **36 transformer layers** (`model.config.num_hidden_layers = 36`).
Target layers: `{9, 18, 27}` ≈ 25%, 50%, 75% depth.

This differs from Cycle 04–06 which used `{6, 14, 22}` for 28-layer Qwen3-1.7B.
Do not mix layer indices across cycles.

### Probe Specification

- **S probe:** 32 samples from the Math-CoT-20k training subset (the 5k seed=42 sample used
  for training), tokenized to seqlen=512. Identical probe inputs reused across all 10 checkpoints.
- **X probe:** Same 32 samples, generated once from step_000 (base model forward pass).
  Fixed and reused across all checkpoints so any spectrum difference is purely from the
  model weights, not the input.

### UV/Sketch Artifact Saves (HARD REQUIREMENT)

At **every checkpoint**, explicitly set:
```python
save_s_uv_path = f"{EXP_ROOT}/geometry/uv_artifacts/step_{NNN:03d}/s_uv.npz"
save_x_uv_path = f"{EXP_ROOT}/geometry/uv_artifacts/step_{NNN:03d}/x_uv.npz"
```

These are required for `principalEvidence.py` OverlapLift. Not saving them was the
Cycle 04/05 failure mode — do not default to `None`.

### Per-Checkpoint Output

For each step in `{0, 5, 10, 20, 40, 80, 160, 320, 480, 624}`:

```
{EXP_ROOT}/geometry/
  geometry_metrics_step_{NNN:03d}.csv    ← effective_rank, spectral_gap, drift_from_base (per layer/module)
  principal_evidence_step_{NNN:03d}.csv  ← OverlapLift (per layer/module)
  uv_artifacts/step_{NNN:03d}/           ← raw UV saves (keep until run fully confirmed)
```

Merged output after all checkpoints: `principal_evidence_all.csv`
(step, layer, module, overlap_lift columns — for the trajectory plot).

### principalEvidence Configuration (Cycle 07)

```python
config = {
    "analyse": {
        "base_model_npy_dir": f"{EXP_ROOT}/weights/step_000/",
        "npy_output_root": f"{EXP_ROOT}/weights/",
        "related_work": {
            "enable": True,
            "target_layer": 18,          # run for each of {9, 18, 27}
            "target_modules": None,       # all 7 LoRA modules
            "principal_rank_k": 50,
            "principal_top_ratio": 0.01,
            "save_png": True,
        },
        "tasks": [
            [f"step_{NNN:03d}", str(NNN)]
            for NNN in [5, 10, 20, 40, 80, 160, 320, 480, 624]
        ],
    }
}
```

All comparisons are relative to step_000 (base), not to the previous checkpoint.

### Expected Trajectory Signatures (working hypotheses, not confirmed)

| Phase | Steps | Expected geometry signal |
|---|---|---|
| Dip | 5–40 | High effective_rank (broad update), high drift_from_base, low OverlapLift |
| Recovery | 80–320 | Effective_rank contracts, drift stabilizes, OverlapLift rises |
| Converged | 480–624 | Low effective_rank (specialized), stable drift, high OverlapLift |

Record what is actually observed — these are predictions from the dip-and-recovery theory,
not pre-confirmed facts for our LoRA variant.

---

## Open / Unconfirmed Items

- Whether `cleanup_intermediate=True` (set in the base GetSlice config) has
  ever caused a probe's raw activation cache to be deleted before a later
  re-analysis needed it has not been checked; if a future cycle needs to
  recompute a metric not already in `geometry_metrics.csv`, confirm the raw
  `sMat_*.json`/`xMat_X.json` files still exist before assuming a re-run is
  required.
