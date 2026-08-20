#!/usr/bin/env python3
"""
run_cycle08.py — Cycle 08: OPD (on-policy distillation) trajectory via verl, compared
to the Cycle 07 base-model SFT trajectory.

One OPD arm (Qwen3-4B-Base student, Qwen3-8B teacher) trained with verl, then evaluated
and geometrically analyzed on the SAME protocol / probes / checkpoint grid as Cycle 07,
so the OPD-vs-SFT dip-recovery and geometry are comparable step-for-step.

Phases (resumable via --start-from-phase, skip-if-exists throughout):
  Phase 2: train     — run verl OPD (run_cycle08_opd.sh) in the `verl` env; saves
                       {ckpt}/global_step_<N>/actor every SAVE_FREQ steps.   [needs >=2 GPUs]
  Phase 2b: convert  — verl checkpoint -> merged HF model per grid step (cycle08_convert_ckpt,
                       run with the verl python). Non-grid raw checkpoints are pruned.
  Phase 3: eval      — REUSES cycle07's runner_think (MATH500/Numina/AIME24) + lm_eval
                       (GPQA-D / MMLU-Pro), identical tasks/caps, on the merged models.
  Phase 4: geometry  — REUSES cycle07's GetSlice + export_weights + principalEvidence, on the
                       SAME S/X probes as cycle07 (byte-identical -> comparable geometry).
  Phase 5: aggregate — RESULTS_08.md + trajectory_scores.csv + an OPD-vs-SFT comparison table
                       (loads cycle07 trajectory_scores.csv).

Run with the DENSITY env python (same as cycle07: eval/geometry subprocesses use sys.executable).
Training + conversion shell out to the verl env explicitly. Data (prompts + probes) reuse the
EXACT seed=42 5000 sample from cycle07 (built by cycle08_data_prep.py).

Usage:
  python run_cycle08.py [--run-root P] [--copyback-root P] [--smoke] [--start-from-phase N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

# Reuse cycle07's eval/geometry/aggregation leaf helpers verbatim (DRY: identical protocol).
from scripts import run_cycle07 as c07  # noqa: E402
from scripts.cycle08_convert_ckpt import convert_step  # noqa: E402

ensure_dir = c07.ensure_dir
write_json = c07.write_json
read_jsonl = c07.read_jsonl
step_label = c07.step_label
binom_se = c07.binom_se

BASE_MODEL = c07.BASE_MODEL
VERL_PY = Path("/root/autodl-tmp/envs/verl/bin/python")
LAUNCH_SH = SIDECAR_ROOT / "scripts/run_cycle08_opd.sh"
CONVERT_PY = SIDECAR_ROOT / "scripts/cycle08_convert_ckpt.py"

RUN_ROOT_DEFAULT = Path("/root/autodl-tmp/cycle08_opd_trajectory")
COPYBACK_ROOT_DEFAULT = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_08_h_opd_vs_sft_comparison/run_01")
# Cycle 07 outputs we reuse: probes (for comparable geometry) + the SFT trajectory (for comparison).
CYCLE07_RUN_ROOT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory")
CYCLE07_COPYBACK = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_07_base_sft_trajectory/run_01")

PROMPTS_PARQUET = RUN_ROOT_DEFAULT / "data" / "opd_prompts_5k.parquet"
# Same evaluated checkpoint grid as cycle07. verl filters to 4999 rows and uses drop_last:
# floor(4999 / 16) * 2 epochs = 624 updates, matching the cycle07 evaluated final point.
CHECKPOINT_STEPS = [5, 10, 20, 40, 80, 160, 320, 480]
FINAL_STEP = 624
SAVE_FREQ = 5  # captures early grid steps; final 624 is saved by verl's last-step hook
TOTAL_EPOCHS = 2  # sweep the prompt sample twice; TOTAL_TRAINING_STEPS pins the exact stop

# Smoke = a small-but-FULL-fidelity mirror of the real run: same code paths (fsdp2, flash-attn,
# forward_kl_topk, null-reward, LoRA, online pruner, convert, eval, geometry, aggregate, env
# switching) at tiny scale. 16 prompts / batch 4 = 4 steps; save_freq=1 saves 1,2,3,4; grid
# {2,4} -> pruner deletes non-grid 1,3 (exercises the pruner); eval/geometry over {0,2,4}.
SMOKE_PARQUET = RUN_ROOT_DEFAULT / "data" / "opd_prompts_smoke16.parquet"
SMOKE_CHECKPOINT_STEPS = [2]
SMOKE_FINAL_STEP = 4
SMOKE_SAVE_FREQ = 1
SMOKE_MAX_RESPONSE = 512  # short for speed; the 10240-token OOM risk is a real-run/pilot concern

GEN_TASKS = c07.GEN_TASKS
LM_TASKS = c07.LM_TASKS
GEOMETRY_LAYERS = c07.GEOMETRY_LAYERS
N_PROBE = c07.N_PROBE
SEED = c07.SEED


# =============================================================================
# Phase 2: train (verl OPD)  [requires >=2 GPUs: 1 student + 1 dedicated teacher]
# =============================================================================
def phase2_train(run_root: Path, *, smoke: bool) -> dict:
    ckpt_root = ensure_dir(run_root / "checkpoints")
    if smoke:
        grid = list(SMOKE_CHECKPOINT_STEPS) + [SMOKE_FINAL_STEP]
        final_step, parquet = SMOKE_FINAL_STEP, SMOKE_PARQUET
        save_freq, total_epochs = SMOKE_SAVE_FREQ, 1
    else:
        grid = list(CHECKPOINT_STEPS) + [FINAL_STEP]
        final_step, parquet = FINAL_STEP, PROMPTS_PARQUET
        save_freq, total_epochs = SAVE_FREQ, TOTAL_EPOCHS
    info = {"ckpt_root": str(ckpt_root), "steps": grid, "total_steps": final_step,
            "save_freq": save_freq, "total_epochs": total_epochs}
    final_actor = ckpt_root / f"global_step_{final_step}" / "actor"
    if final_actor.exists():
        print(f"[TRAIN] final verl checkpoint exists ({final_actor}); skip training", flush=True)
        return info
    if not parquet.exists():
        raise FileNotFoundError(
            f"OPD prompt parquet missing: {parquet}. Run cycle08_data_prep.py first.")

    env = dict(os.environ)
    env["PATH"] = f"/root/autodl-tmp/envs/verl/bin:{env.get('PATH', '')}"
    env.update({
        "TRAIN_PARQUET": str(parquet),
        "VAL_PARQUET": str(parquet),
        "CKPT_DIR": str(ckpt_root),
        "SAVE_FREQ": str(save_freq),
        "TOTAL_EPOCHS": str(total_epochs),
        "TOTAL_TRAINING_STEPS": str(final_step),
        "USE_POLICY_GRADIENT": "False",   # supervised OPD (user-confirmed), cleanest vs SFT
    })
    if smoke:
        # tiny run, but REAL config (fsdp2, flash-attn, remove_padding, forward_kl_topk from the
        # launch-script defaults). Only scale shrinks: batch 4, short response, conservative mem.
        env.update({"TRAIN_BATCH_SIZE": "4", "PPO_MINI_BATCH_SIZE": "4",
                    "MAX_RESPONSE_LENGTH": str(SMOKE_MAX_RESPONSE), "MAX_PROMPT_LENGTH": "512",
                    "ROLLOUT_GPU_MEM_UTIL": "0.55", "TEACHER_GPU_MEM_UTIL": "0.70",
                    "PPO_MAX_TOKEN_LEN_PER_GPU": "4096"})
    # Online disk guard: verl saves ~16GB (FULL model) per checkpoint; save_freq=5 over 624
    # steps would be ~2TB. The pruner deletes completed non-grid checkpoints as they appear,
    # so only the ~9 grid checkpoints ever accumulate.
    pruner = subprocess.Popen(
        [sys.executable, str(SIDECAR_ROOT / "scripts/cycle08_ckpt_pruner.py"),
         "--ckpt-root", str(ckpt_root), "--grid", ",".join(str(s) for s in grid)])
    print(f"[TRAIN] launching verl OPD -> {ckpt_root} (>=2 GPUs required); pruner pid={pruner.pid}", flush=True)
    try:
        r = subprocess.run(["bash", str(LAUNCH_SH)], env=env, cwd=str(SIDECAR_ROOT / "scripts"))
    finally:
        pruner.terminate()
    if r.returncode != 0:
        raise RuntimeError(f"verl OPD training failed rc={r.returncode}")
    if not final_actor.exists():
        raise RuntimeError(f"training finished but final checkpoint missing: {final_actor}")
    return info


# =============================================================================
# Phase 2b: convert verl checkpoints -> merged HF models (verl env, via subprocess)
# =============================================================================
def _merged_dir(run_root: Path, step: int) -> Path:
    return run_root / "_merged_models" / step_label(step)


def _resolve_model(run_root: Path, step: int) -> Path:
    """Model path usable by vLLM/lm_eval/GetSlice. step 0 -> base; else converted merged dir."""
    if step == 0:
        return BASE_MODEL
    return _merged_dir(run_root, step)


def phase2b_convert(run_root: Path, train_info: dict, *, smoke: bool, prune_raw: bool = True) -> None:
    ckpt_root = Path(train_info["ckpt_root"])
    steps = [s for s in train_info["steps"] if s != 0]
    for step in steps:
        out = _merged_dir(run_root, step)
        if (out / "config.json").exists() and (any(out.glob("*.safetensors"))):
            print(f"[CONVERT] {step_label(step)} already merged; skip", flush=True)
            continue
        actor = ckpt_root / f"global_step_{step}" / "actor"
        if not actor.exists():
            print(f"[CONVERT] WARN: no verl checkpoint for step {step} ({actor}); skipping", flush=True)
            continue
        # Run converter under the verl python (model_merger needs torch2.9 + verl).
        cmd = [str(VERL_PY), str(CONVERT_PY), "--ckpt-root", str(ckpt_root),
               "--step", str(step), "--out-dir", str(out), "--base", str(BASE_MODEL)]
        print(f"[CONVERT] {step_label(step)} via verl model_merger", flush=True)
        r = subprocess.run(cmd, cwd=str(SIDECAR_ROOT / "scripts"))
        if r.returncode != 0:
            raise RuntimeError(f"convert failed rc={r.returncode} for step {step}")
        # reclaim the ~16GB raw verl checkpoint now that the merged HF model exists
        if not smoke:
            shutil.rmtree(ckpt_root / f"global_step_{step}", ignore_errors=True)
            print(f"[CONVERT] pruned raw global_step_{step} (~16GB; merged kept)", flush=True)

    if prune_raw and not smoke:
        # Reclaim disk: keep only grid global_step_* checkpoints (the merged models are the
        # deliverable; raw FSDP shards for non-grid steps are disposable).
        keep = {f"global_step_{s}" for s in train_info["steps"]}
        for d in ckpt_root.glob("global_step_*"):
            if d.name not in keep:
                shutil.rmtree(d, ignore_errors=True)
                print(f"[CONVERT] pruned non-grid raw checkpoint {d.name}", flush=True)


# =============================================================================
# Phase 3: eval (reuse cycle07 runner_think + lm_eval on the merged models)
# =============================================================================
def _eval_one_checkpoint(step: int, model_path: str, eval_root: str, lm_tasks: list,
                         smoke: bool, gpu_id: int) -> int:
    """Run ALL eval tasks for one checkpoint, pinned to a single GPU. Module-level so
    ProcessPoolExecutor can run several checkpoints concurrently (one per card). Each
    task's runner_think/lm_eval subprocess inherits CUDA_VISIBLE_DEVICES from here, and
    uses tensor_parallel_size=1 → identical per-model settings/results to cycle07, just
    2 checkpoints in flight at once. TP=2 is not used (4B doesn't benefit; comms overhead)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from scripts import run_cycle07 as _c07  # re-import in worker
    label = step_label(step)
    eroot = Path(eval_root)
    for t in GEN_TASKS:
        _c07._run_think_eval(t, model_path, label, ensure_dir(eroot / label / t), smoke=smoke,
                             n=_c07.TASK_N[t], max_tokens=_c07.TASK_MAXTOK[t],
                             max_model_len=_c07.TASK_MAXLEN[t])
    for (sub, lm_task, nfs, limit) in lm_tasks:
        out = eroot / label / sub
        if _c07._lm_eval_done(out):
            continue
        try:
            _c07._run_lm_eval(model_path, lm_task, ensure_dir(out), num_fewshot=nfs, limit=limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] lm_eval {lm_task} failed for {label} on gpu{gpu_id}: {exc}; continuing", flush=True)
    return step


def phase3_eval(run_root: Path, train_info: dict, *, smoke: bool, n_eval_gpus: int = 2) -> None:
    import concurrent.futures as _cf

    eval_root = ensure_dir(run_root / "eval")
    steps = [0] + [s for s in train_info["steps"]]
    lm_tasks = [] if smoke else LM_TASKS

    # Determine which checkpoints still need eval + resolve their model paths.
    todo = []
    for step in steps:
        label = step_label(step)
        gen_done = all((eval_root / label / t / f"{label}.json").exists() for t in GEN_TASKS)
        lm_done = all(c07._lm_eval_done(eval_root / label / sub) for (sub, *_r) in lm_tasks)
        if gen_done and lm_done:
            print(f"[SKIP] all eval done for {label}", flush=True)
            continue
        model_path = _resolve_model(run_root, step)
        if step != 0 and not (Path(model_path) / "config.json").exists():
            print(f"[EVAL] WARN: merged model missing for {label} ({model_path}); skipping", flush=True)
            continue
        todo.append((step, str(model_path)))

    if not todo:
        return
    # One checkpoint per GPU, up to n_eval_gpus in flight. Same results as cycle07 (TP=1 per
    # model), ~n_eval_gpus× faster wall-clock by using both 96G cards instead of idling one.
    workers = max(1, min(n_eval_gpus, len(todo)))
    print(f"[EVAL] {len(todo)} checkpoints across {workers} GPU(s) (checkpoint-parallel)", flush=True)
    with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_eval_one_checkpoint, step, mp, str(eval_root), lm_tasks, smoke, i % workers): step
                for i, (step, mp) in enumerate(todo)}
        for fut in _cf.as_completed(futs):
            step = futs[fut]
            try:
                fut.result()
                print(f"[EVAL] done {step_label(step)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[EVAL] ERROR {step_label(step)}: {exc}", flush=True)
                raise


# =============================================================================
# Phase 4: geometry (reuse cycle07 GetSlice/export/PE on the SAME probes)
# =============================================================================
def _reuse_cycle07_probes(run_root: Path) -> tuple[Path, Path]:
    """Reuse cycle07's exact S/X probes for byte-identical, comparable geometry."""
    s_root = CYCLE07_RUN_ROOT / "getslice/inputs/S"
    x_file = CYCLE07_RUN_ROOT / "getslice/inputs/X_base/x_probe.jsonl"
    if (s_root / "math_cot_probe" / "gamma_s.jsonl").exists() and x_file.exists():
        print(f"[GEO] reusing cycle07 probes: S={s_root}, X={x_file}", flush=True)
        return s_root, x_file
    # Fallback: rebuild from the (identical) cycle07 probe_rows via cycle07's builder.
    print("[GEO] cycle07 probes not found; rebuilding identical probes from cycle07 sample", flush=True)
    c07_meta = json.loads((CYCLE07_RUN_ROOT / "data_prep/data_meta.json").read_text())
    return c07._build_probes(run_root, c07_meta, n_probe=N_PROBE, max_new_tokens=512)


def _geometry_one_checkpoint(step: int, model_path: str, run_root: str, layers: list,
                             seqlen: int, n_probe: int, s_root: str, x_file: str, gpu_id: int) -> int:
    """GetSlice (all layers) + weight export for one checkpoint, pinned to one GPU. Module-level
    so several checkpoints run concurrently (one per card). GetSlice/export subprocesses inherit
    CUDA_VISIBLE_DEVICES; results are independent of which card, so the geometry is unchanged."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from scripts import run_cycle07 as _c07
    from scripts.export_weights import export_model_weights, MODULES as _MODULES
    label = step_label(step)
    rr = Path(run_root)
    for L in layers:
        _c07._getslice_checkpoint(Path(model_path), label, Path(s_root), Path(x_file), rr, L,
                                  seqlen=seqlen, n_probe=n_probe)
    w_dir = (rr / "weights" / label) if step == 0 else (rr / "weights" / label / str(step))
    export_model_weights(str(model_path), w_dir, layers=layers, modules=_MODULES)
    return step


def phase4_geometry(run_root: Path, train_info: dict, *, smoke: bool, n_geo_gpus: int = 2) -> None:
    import concurrent.futures as _cf
    from scripts.export_weights import MODULES

    weights_root = ensure_dir(run_root / "weights")
    layers = [18] if smoke else GEOMETRY_LAYERS
    n_probe = 4 if smoke else N_PROBE
    seqlen = 64 if smoke else 512
    steps = [0] + [s for s in train_info["steps"]]

    s_root, x_file = _reuse_cycle07_probes(run_root)

    todo = []
    for step in steps:
        label = step_label(step)
        gs_done = all(
            (run_root / "getslice/outputs" / label
             / f"layer_{L}/S/math_cot_probe/layer_{L}/sMat_math_cot_probe.json").exists()
            and (run_root / "getslice/outputs" / label
                 / f"layer_{L}/X/X/layer_{L}/xMat_X.json").exists()
            for L in layers)
        w_dir = (weights_root / label) if step == 0 else (weights_root / label / str(step))
        w_done = w_dir.exists() and len(list(w_dir.glob("*.npy"))) >= len(layers) * len(MODULES)
        if gs_done and w_done:
            print(f"[SKIP] geometry done for {label}", flush=True)
            continue
        model_path = _resolve_model(run_root, step)
        if step != 0 and not (Path(model_path) / "config.json").exists():
            print(f"[GEO] WARN: merged model missing for {label}; skipping", flush=True)
            continue
        todo.append((step, str(model_path)))

    if todo:
        workers = max(1, min(n_geo_gpus, len(todo)))
        print(f"[GEO] {len(todo)} checkpoints across {workers} GPU(s) (checkpoint-parallel GetSlice+export)",
              flush=True)
        with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_geometry_one_checkpoint, step, mp, str(run_root), layers, seqlen,
                              n_probe, str(s_root), str(x_file), i % workers): step
                    for i, (step, mp) in enumerate(todo)}
            for fut in _cf.as_completed(futs):
                step = futs[fut]
                try:
                    fut.result()
                    print(f"[GEO] done {step_label(step)}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[GEO] ERROR {step_label(step)}: {exc}", flush=True)
                    raise

    # Weight-space PE (OverlapLift) + spectral metrics are CPU/numpy -> keep sequential.
    if not smoke:
        c07._run_principal_evidence(run_root, weights_root, train_info["steps"], layers)
    c07._build_geometry_metrics(run_root, steps, layers)


# =============================================================================
# Phase 5: aggregate + OPD-vs-SFT comparison
# =============================================================================
def _load_cycle07_trajectory() -> list[dict]:
    csv_path = CYCLE07_COPYBACK / "trajectory_scores.csv"
    if not csv_path.exists():
        return []
    import csv as _csv
    with open(csv_path) as f:
        return list(_csv.DictReader(f))


def aggregate(run_root: Path, copyback_root: Path, train_info: dict, *, smoke: bool) -> None:
    eval_root = run_root / "eval"
    steps = [0] + [s for s in train_info["steps"]]
    lm_subs = [s for (s, *_r) in (LM_TASKS if not smoke else [])]

    traj_rows, rlen_rows = [], []
    for step in steps:
        label = step_label(step)
        row = {"step": step}
        for t in GEN_TASKS:
            jf = eval_root / label / t / f"{label}.json"
            if jf.exists():
                d = json.loads(jf.read_text())
                row[f"{t}_acc"] = round(d.get("acc", float("nan")), 4)
                row[f"{t}_se"] = round(d.get("stderr", float("nan")), 4)
                if t == "math500":
                    rlen_rows.append({"step": step,
                                      "mean_response_len_math500": round(d.get("mean_response_len", 0.0), 1)})
        for sub in lm_subs:
            acc, se = c07._read_lm_acc(eval_root / label / sub)
            row[f"{sub}_acc"] = round(acc, 4) if acc is not None else None
            row[f"{sub}_se"] = round(se, 4) if se is not None else None
        traj_rows.append(row)

    ensure_dir(copyback_root)
    c07._write_csv(copyback_root / "trajectory_scores.csv", traj_rows)
    c07._write_csv(copyback_root / "response_length_trajectory.csv", rlen_rows)
    geo_src = run_root / "geometry"
    if geo_src.exists():
        geo_dst = ensure_dir(copyback_root / "geometry")
        for f in geo_src.glob("geometry_metrics_*.csv"):
            shutil.copy2(f, geo_dst / f.name)
    pe_src = run_root / "principal_evidence"
    if pe_src.exists():
        for f in pe_src.glob("**/*.csv"):
            shutil.copy2(f, ensure_dir(copyback_root / "geometry") / f"principal_{f.parent.name}_{f.name}")

    _write_results_md(copyback_root, traj_rows, rlen_rows, train_info)
    _write_comparison(copyback_root, traj_rows)
    print(f"[AGG] copyback -> {copyback_root}", flush=True)


def _write_results_md(copyback_root: Path, traj, rlen, train_info) -> None:
    def g(row, k):
        v = row.get(k)
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else v

    base, final = traj[0], traj[-1]
    m0, mf = base.get("math500_acc"), final.get("math500_acc")
    a08 = "UNDETERMINED"
    if isinstance(m0, (int, float)) and isinstance(mf, (int, float)):
        thr = m0 + binom_se(m0, 500)
        a08 = f"{'PASS' if mf > thr else 'FAIL'} (final {mf:.3f} vs base+SE {thr:.3f})"
    dip = next((r["step"] for r in traj if r["step"] in (5, 10, 20, 40)
                and isinstance(r.get("math500_acc"), (int, float)) and isinstance(m0, (int, float))
                and r["math500_acc"] < m0), None)
    rec = next((r["step"] for r in traj if isinstance(r.get("math500_acc"), (int, float))
                and isinstance(m0, (int, float)) and r["step"] > 0 and r["math500_acc"] > m0), None)
    b08 = ("FULL PASS" if dip is not None and rec is not None else
           "PARTIAL" if rec is not None else "FAIL")
    lines = [
        "# RESULTS_08: Cycle 08 — OPD Trajectory (Qwen3-4B-Base <- Qwen3-8B teacher, verl)",
        "", "```yaml", "cycle: cycle_08_h_opd_vs_sft_comparison",
        f"student: {BASE_MODEL}", "teacher: /root/autodl-tmp/model/Qwen/Qwen3-8B",
        "method: on-policy distillation (verl, supervised forward_kl_topk, topk=32, token-mean, ppo_epochs=1)",
        f"total_steps: {train_info.get('total_steps')} (total_epochs={train_info.get('total_epochs')})",
        f"checkpoint_grid: {[0] + train_info.get('steps', [])}", "```", "",
        "## Gate Verdicts",
        f"- **A08 (feasibility, MATH500@final > base+1SE):** {a08}",
        f"- **B08 (dip-and-recovery):** {b08}  (dip step={dip}, recover step={rec})",
        "- **C08 (OOD-lite):** see GPQA-D / MMLU-Pro vs step_000 below.",
        "", "## Trajectory (acc)", "",
        "| step | math500 | numina | aime24 | gpqa | mmlu_pro |",
        "|---|---|---|---|---|---|",
    ]
    for r in traj:
        lines.append(f"| {r['step']} | {g(r,'math500_acc')} | {g(r,'numina_acc')} | "
                     f"{g(r,'aime24_acc')} | {g(r,'gpqa_acc')} | {g(r,'mmlu_pro_acc')} |")
    lines += ["", "## MATH500 Response Length (mean tokens)", "", "| step | mean_resp_len |", "|---|---|"]
    for r in rlen:
        lines.append(f"| {r['step']} | {r['mean_response_len_math500']} |")
    lines += ["", "See opd_vs_sft_comparison.md for the step-for-step OPD-vs-SFT table.", ""]
    (copyback_root / "RESULTS_08.md").write_text("\n".join(lines))


def _write_comparison(copyback_root: Path, opd_traj) -> None:
    """Step-for-step OPD (cycle08) vs SFT (cycle07) MATH500 + OOD comparison."""
    sft = _load_cycle07_trajectory()
    if not sft:
        (copyback_root / "opd_vs_sft_comparison.md").write_text(
            "# OPD vs SFT\n\ncycle07 trajectory_scores.csv not found; comparison unavailable.\n")
        return
    sft_by_step = {int(r["step"]): r for r in sft}

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    lines = ["# Cycle 08 OPD vs Cycle 07 SFT — step-for-step (same prompts, same eval)",
             "",
             "Both trajectories: Qwen3-4B-Base, seed=42 5000-prompt sample, eff_batch 16, ~2 epochs",
             "(OPD evaluated through step 624; SFT evaluated through step 624, trained to 632). "
             "OPD target = Qwen3-8B teacher distribution; SFT target = dataset answers.",
             "", "| step | MATH500 OPD | MATH500 SFT | Δ(OPD−SFT) | gpqa OPD | gpqa SFT | mmlu_pro OPD | mmlu_pro SFT |",
             "|---|---|---|---|---|---|---|---|"]
    for r in opd_traj:
        st = r["step"]
        s = sft_by_step.get(st, {})
        mo, ms = num(r.get("math500_acc")), num(s.get("math500_acc"))
        d = f"{mo - ms:+.3f}" if (mo is not None and ms is not None) else "—"
        def cell(x):
            return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else x)
        lines.append(
            f"| {st} | {cell(mo)} | {cell(ms)} | {d} | {cell(num(r.get('gpqa_acc')))} | "
            f"{cell(num(s.get('gpqa_acc')))} | {cell(num(r.get('mmlu_pro_acc')))} | "
            f"{cell(num(s.get('mmlu_pro_acc')))} |")
    lines += ["", "Δ>0 means OPD beats SFT at that step. Dip-recovery shape (cols 2-3) is the",
              "central H1 comparison; geometry/ CSVs carry the spectral side.", ""]
    (copyback_root / "opd_vs_sft_comparison.md").write_text("\n".join(lines))


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Cycle 08: OPD trajectory vs cycle07 SFT")
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--copyback-root", type=Path, default=COPYBACK_ROOT_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start-from-phase", type=int, default=2, choices=[2, 3, 4, 5],
                    help="2=train 2b=convert(in 2) 3=eval 4=geometry 5=aggregate")
    args = ap.parse_args()

    run_root = ensure_dir(args.run_root)
    smoke = args.smoke
    start = args.start_from_phase
    print(f"=== Cycle 08 === run_root={run_root} smoke={smoke} start_phase={start}", flush=True)

    info_path = run_root / "train_info.json"
    if start <= 2:
        print("\n=== Phase 2: Train (verl OPD) ===", flush=True)
        train_info = phase2_train(run_root, smoke=smoke)
        write_json(info_path, train_info)
        print("\n=== Phase 2b: Convert checkpoints ===", flush=True)
        phase2b_convert(run_root, train_info, smoke=smoke)
    else:
        train_info = json.loads(info_path.read_text())

    if start <= 3:
        print("\n=== Phase 3: Evaluation ===", flush=True)
        phase3_eval(run_root, train_info, smoke=smoke)
    if start <= 4:
        print("\n=== Phase 4: Geometry ===", flush=True)
        phase4_geometry(run_root, train_info, smoke=smoke)

    print("\n=== Phase 5: Aggregate + Copyback ===", flush=True)
    aggregate(run_root, args.copyback_root, train_info, smoke=smoke)
    print("\n=== Cycle 08 complete ===", flush=True)


if __name__ == "__main__":
    main()
