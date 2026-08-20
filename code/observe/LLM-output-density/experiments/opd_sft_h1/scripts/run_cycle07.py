#!/usr/bin/env python3
"""
run_cycle07.py — Cycle 07: Base-Model SFT Trajectory (Qwen3-4B-Base + Math-CoT-20k)

Single SFT trajectory (NO OPD arm, NO vLLM colocate). Four phases, resumable via
--start-from-phase, skip-if-exists throughout:

  Phase 1: data prep      — load Math-CoT-20k, sample 5000 (seed=42), render think-format
  Phase 2: train          — ONE SFTTrainer run on Qwen3-4B-Base; a GridSaveCallback saves a
                            LoRA adapter at steps {5,10,20,40,80,160,320,480,624}
  Phase 3: eval           — per checkpoint (step_000 base + 9 adapters): merge→temp dir, run
                            MATH500/NuminaMath/AIME24 (think-format, 32768 tok) + GPQA-D/MMLU-Pro
                            (lm_eval loglikelihood, no chat), record MATH500 response_length, delete merge
  Phase 4: geometry        — build S/X probes once; per checkpoint: merge→GetSlice (layers 9/18/27,
                            UV saved) + weight export → delete merge; then principalEvidence (OverlapLift)

Outputs: big/raw artifacts under --run-root (autodl-tmp); distilled CSVs + RESULTS_07.md +
provenance copied back to --copyback-root (mypaper/local_experiment_results/...).

Usage:
  python run_cycle07.py [--run-root PATH] [--copyback-root PATH] [--smoke] [--start-from-phase N]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR_ROOT = REPO_ROOT / "experiments/opd_sft_h1"

BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
DATA_PARQUET = Path("/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet")

RUN_ROOT_DEFAULT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory")
COPYBACK_ROOT_DEFAULT = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_07_base_sft_trajectory/run_01"
)

# Think-format instruction — identical to the Math-CoT-20k `message` user turn and to
# Eval/component/think_math/runner_think.py INSTR (single leading newline). Keep in sync.
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."

CHECKPOINT_STEPS = [5, 10, 20, 40, 80, 160, 320, 480, 624]
GEOMETRY_LAYERS = [9, 18, 27]            # 25% / 50% / 75% of Qwen3-4B-Base's 36 layers
N_TRAIN = 5000
N_PROBE = 32
SEED = 42

# Training uses a small effective batch (16) with a FIXED step count (632), not fixed epochs:
# this keeps the checkpoint grid on Rethink SFT's step positions while cutting wall-clock ~4x
# vs eff_batch 64. The run then consumes ~2 epochs of data (632*16/5000) at higher gradient
# variance (smaller batch). per_device=1 × grad_accum=16 = effective batch 16.
EFF_BATCH = 16
FIXED_STEPS = 632

THINK_RUNNER = REPO_ROOT / "Eval/component/think_math/runner_think.py"
GEN_TASKS = ["math500", "numina", "aime24"]
# Per-task generative eval size (0 = full set). NuminaMath capped at 256 (degradation-side
# ID check only) to bound the 32768-token × 10-checkpoint cost; MATH500 + AIME24 stay full.
TASK_N = {"math500": 0, "numina": 256, "aime24": 0}
# Per-task generation budget, tuned from the step_000/005 response-length distribution:
# math500/numina are bimodal (median ~400, p90 ~1100, then a ~5% non-terminating tail that
# runs to 30720), so a small cap only clips that tail (those items were already truncated/
# wrong). aime24 needs real headroom (p90 ~7800 — hard problems genuinely reason long).
# max_model_len = cap + prompt headroom; smaller max_model_len => higher vLLM concurrency => faster.
TASK_MAXTOK = {"math500": 4096, "numina": 4096, "aime24": 16384}
TASK_MAXLEN = {"math500": 6144, "numina": 6144, "aime24": 18432}
# lm_eval loglikelihood OOD tasks: (out_subdir, lm_task, num_fewshot, limit)
LM_TASKS = [
    ("gpqa", "gpqa_diamond_zeroshot", 0, None),
    ("mmlu_pro", "mmlu_pro", 0, 100),
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def step_label(step: int) -> str:
    return f"step_{step:03d}"


def binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


# =============================================================================
# Phase 1: Data Preparation
# =============================================================================
def phase1_data(run_root: Path, *, smoke: bool) -> dict:
    """Load Math-CoT-20k, sample N_TRAIN (seed=42), render think-format training text."""
    import pandas as pd
    from transformers import AutoTokenizer

    data_dir = ensure_dir(run_root / "data_prep")
    meta_path = data_dir / "data_meta.json"
    if meta_path.exists():
        print(f"[DATA] reuse {meta_path}", flush=True)
        return json.loads(meta_path.read_text())

    n_train = 64 if smoke else N_TRAIN
    df = pd.read_parquet(DATA_PARQUET)
    cols = df.columns.tolist()
    print(f"[DATA] parquet cols={cols} rows={len(df)}", flush=True)
    q_field = "question" if "question" in cols else ("problem" if "problem" in cols else cols[0])
    r_field = "response" if "response" in cols else ("answer" if "answer" in cols else cols[-1])

    df_s = df.sample(n=min(n_train, len(df)), random_state=SEED).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL), trust_remote_code=True)

    train_rows: list[dict] = []
    probe_rows: list[dict] = []
    for _, row in df_s.iterrows():
        question = str(row[q_field])
        response = str(row[r_field])
        messages = [
            {"role": "user", "content": question + INSTR},
            {"role": "assistant", "content": response},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        train_rows.append({"text": text})
        probe_rows.append({"question": question, "answer": response})

    train_jsonl = data_dir / "train_5k.jsonl"
    write_jsonl(train_jsonl, train_rows)
    probe_jsonl = data_dir / "probe_rows.jsonl"
    write_jsonl(probe_jsonl, probe_rows)

    # Format sanity check (brief Phase 1c)
    sample0 = train_rows[0]["text"]
    assert "<|im_start|>user" in sample0, "expected chat template user turn"
    assert "<|im_start|>system" not in sample0, "unexpected system prompt"
    has_think = "<think>" in sample0

    meta = {
        "n_train": len(train_rows), "seed": SEED,
        "q_field": q_field, "r_field": r_field, "parquet_cols": cols,
        "train_jsonl": str(train_jsonl), "probe_jsonl": str(probe_jsonl),
        "sample_has_think_tag": has_think,
        "data_parquet": str(DATA_PARQUET),
    }
    write_json(meta_path, meta)
    print(f"[DATA] {len(train_rows)} train rows -> {train_jsonl} "
          f"(q={q_field}, r={r_field}, think_tag={has_think})", flush=True)
    return meta


# =============================================================================
# Phase 2: Training (single run + grid-save callback)
# =============================================================================
GPU_WORKER = SIDECAR_ROOT / "scripts/cycle07_gpu_worker.py"


def _run_gpu_worker(subcommand: str, cfg: dict, cfg_path: Path) -> None:
    """Run a heavy GPU op (train / rollout-x) as an isolated subprocess.

    Keeps the orchestrator parent at ZERO GPU memory so its eval/geometry
    subprocesses can claim the full card (the residual-memory bug the smoke caught).
    """
    write_json(cfg_path, cfg)
    env = dict(os.environ, TMPDIR="/root/autodl-tmp/pip-tmp",
               HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentation OOM
    cmd = [sys.executable, str(GPU_WORKER), subcommand, "--config", str(cfg_path)]
    print(f"[WORKER] {subcommand} ({cfg_path.name})", flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"gpu_worker {subcommand} failed rc={r.returncode}")


def phase2_train(run_root: Path, data_meta: dict, *, smoke: bool) -> dict:
    """Train one SFT run (subprocess); grid-save LoRA adapters.

    Returns {'ckpt_root', 'steps', 'total_steps'}.
    """
    ckpt_root = ensure_dir(run_root / "checkpoints")
    # 19456 (3072+16384) OOMs a 4B on a single 48G card at backward. 10240 covers the
    # Math-CoT-20k response median (~7.6k tok) + prompt; the long tail is right-truncated.
    max_len = 256 if smoke else 10240
    n_train = data_meta["n_train"]

    if smoke:
        eff_batch, epochs, max_steps = 8, 1, -1
        total_steps = math.ceil(n_train / eff_batch) * epochs
        grid = [2, 4]
    else:
        # Fixed step count (not fixed epochs) at small effective batch: same 632-step grid
        # aligned to Rethink SFT, ~4x faster than eff_batch 64, ~2 epochs of data consumed.
        eff_batch, epochs, max_steps = EFF_BATCH, 1, FIXED_STEPS
        total_steps = FIXED_STEPS
        grid = list(CHECKPOINT_STEPS)
    if total_steps not in grid:
        grid = grid + [total_steps]
    print(f"[TRAIN] n_train={n_train} eff_batch={eff_batch} max_steps={max_steps} "
          f"=> total_steps={total_steps}; grid={grid}", flush=True)

    info = {"ckpt_root": str(ckpt_root), "steps": grid, "total_steps": total_steps}
    final_dir = ckpt_root / step_label(total_steps)
    if (final_dir / "adapter_config.json").exists():
        print(f"[TRAIN] final checkpoint exists ({final_dir}); skip training", flush=True)
        return info

    cfg = {
        "base_model": str(BASE_MODEL), "ckpt_root": str(ckpt_root),
        "train_jsonl": data_meta["train_jsonl"], "grid": grid,
        "eff_batch": eff_batch, "epochs": epochs, "max_steps": max_steps,
        "lr": 5e-5, "max_len": max_len,
        "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "seed": SEED,
    }
    _run_gpu_worker("train", cfg, ckpt_root / "_train_config.json")
    if not (final_dir / "adapter_config.json").exists():
        raise RuntimeError(f"training finished but final adapter missing: {final_dir}")
    print(f"[TRAIN] done -> {ckpt_root}", flush=True)
    return info


# =============================================================================
# Merge helper (LoRA adapter -> temp merged dir for vLLM / lm_eval / weight export)
# =============================================================================
def _merge_checkpoint(step: int, ckpt_root: Path, merged_root: Path) -> Path:
    """Return a model path usable by vLLM. step_000 -> base; else merge adapter to temp dir."""
    if step == 0:
        return BASE_MODEL
    from scripts.run_opd_minimal_closure import merge_lora_adapter
    adapter = ckpt_root / step_label(step)
    merged = merged_root / step_label(step)
    merge_lora_adapter(BASE_MODEL, adapter, merged)
    return merged


def _drop_merged(step: int, merged_root: Path) -> None:
    if step == 0:
        return
    d = merged_root / step_label(step)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"[CLEAN] removed {d}", flush=True)


# =============================================================================
# Phase 3: Evaluation
# =============================================================================
def _run_think_eval(task: str, model_path: str, label: str, outdir: Path, *, smoke: bool,
                    n: int = 0, max_tokens: int = 0, max_model_len: int = 0) -> None:
    result_json = outdir / f"{label}.json"
    if result_json.exists():
        print(f"[SKIP] {task}/{label}.json", flush=True)
        return
    cmd = [
        sys.executable, str(THINK_RUNNER),
        "--task", task, "--model", model_path, "--label", label, "--outdir", str(outdir),
    ]
    if smoke:
        cmd += ["--n", "2", "--max-tokens", "256", "--max-model-len", "2048"]
    else:
        if n > 0:
            cmd += ["--n", str(n)]
        if max_tokens > 0:
            cmd += ["--max-tokens", str(max_tokens)]
        if max_model_len > 0:
            cmd += ["--max-model-len", str(max_model_len)]
    env = dict(os.environ, TMPDIR="/root/autodl-tmp/pip-tmp",
               HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
    print(f"[THINK_EVAL] {task} {label}", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"think eval failed rc={r.returncode}: {task} {label}")


def _lm_eval_done(output_path: Path) -> bool:
    return output_path.exists() and bool(list(output_path.glob("**/results_*.json")))


def _run_lm_eval(model_path: str, task: str, output_path: Path, *,
                 num_fewshot: int = 0, limit: int | None = None) -> None:
    """Base-model loglikelihood eval: NO chat template, NO thinking."""
    model_args = (f"pretrained={model_path},dtype=bfloat16,"
                  f"gpu_memory_utilization=0.85,max_model_len=4096")
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm", "--model_args", model_args,
        "--tasks", task, "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto", "--seed", str(SEED),
        "--output_path", str(output_path),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1",
               TMPDIR="/root/autodl-tmp/pip-tmp")
    print(f"[LM_EVAL] {task} -> {output_path}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"lm_eval failed rc={r.returncode}: {task} {model_path}")


def phase3_eval(run_root: Path, train_info: dict, *, smoke: bool) -> None:
    ckpt_root = Path(train_info["ckpt_root"])
    merged_root = ensure_dir(run_root / "_merged_tmp")
    eval_root = ensure_dir(run_root / "eval")
    steps = [0] + train_info["steps"]
    # lm_eval OOD tasks (GPQA-D / MMLU-Pro) need their datasets downloaded; skipped in smoke.
    lm_tasks = [] if smoke else LM_TASKS

    for step in steps:
        label = step_label(step)
        gen_dirs = {t: eval_root / label / t for t in GEN_TASKS}
        lm_dirs = {sub: eval_root / label / sub for (sub, *_rest) in lm_tasks}
        gen_done = all((gen_dirs[t] / f"{label}.json").exists() for t in GEN_TASKS)
        lm_done = all(_lm_eval_done(lm_dirs[sub]) for (sub, *_rest) in lm_tasks)
        if gen_done and lm_done:
            print(f"[SKIP] all eval done for {label}", flush=True)
            continue

        model_path = str(_merge_checkpoint(step, ckpt_root, merged_root))
        try:
            for t in GEN_TASKS:
                _run_think_eval(t, model_path, label, ensure_dir(gen_dirs[t]), smoke=smoke,
                                n=TASK_N[t], max_tokens=TASK_MAXTOK[t], max_model_len=TASK_MAXLEN[t])
            for (sub, lm_task, nfs, limit) in lm_tasks:
                out = lm_dirs[sub]
                if _lm_eval_done(out):
                    continue
                # OOD lm_eval is the non-blocking C07 gate; a missing/gated dataset must not
                # abort the (blocking) math trajectory. Warn and continue on failure.
                try:
                    _run_lm_eval(model_path, lm_task, ensure_dir(out), num_fewshot=nfs, limit=limit)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] lm_eval {lm_task} failed for {label}: {exc}\n"
                          f"       (OOD dataset missing? download GPQA-D / MMLU-Pro). "
                          f"Continuing; C07 will be partial.", flush=True)
        finally:
            _drop_merged(step, merged_root)


# =============================================================================
# Phase 4: Geometry
# =============================================================================
_GETSLICE_BASE_CFG = {
    "tasks": ["math_cot_probe"],
    "DEV": "cuda",
    "layer_gpu_chunk_size": 12,
    "single_layer_task_group_size": 1,
    "epsilon": 0.001,
    "svd_singular_floor": 0.0,
    "cholesky_jitter": 0.00001,
    "activation_cache_device": "cuda",
    "uv_dtype": "float32",
    "cleanup_intermediate": True,
    "skip_existing_outputs": True,
    "model_dtype": "float16",
    "trust_remote_code": True,
    "s_batch_size": 1,   # profiling cache requires batch=1 (see Cycle 06 FINDING)
    "x_batch_size": 1,
    "save_s_json_path": "sMat_{task}.json",
    "save_x_json_path": "xMat_X.json",
    "save_s_pt_path": None,
    "save_x_pt_path": None,
    # Cycle 07: do NOT save UV/sketch artifacts. Per user, UV (for subspace-overlap /
    # principal-angle analysis) is deferred to Cycle 08. This cycle's geometry stores only
    # the singular spectra (sMat/xMat json) and the scalar metrics computed from them.
    "save_s_uv_path": None,
    "save_x_uv_path": None,
    "save_metrics_pt_path": None,
    "save_metrics_json_path": None,
    "seed": 3,
}


def _build_probes(run_root: Path, data_meta: dict, *, n_probe: int, max_new_tokens: int) -> tuple[Path, Path]:
    """Build the shared S probe (training slice) and X probe (base rollout), once.

    The X-probe rollout runs in the GPU worker subprocess so the orchestrator
    parent keeps zero GPU memory for the GetSlice subprocesses that follow.
    """
    gs_root = ensure_dir(run_root / "getslice")
    inputs_root = ensure_dir(gs_root / "inputs")

    # S probe: first n_probe training rows (question/answer). Pure file IO.
    s_dir = ensure_dir(inputs_root / "S" / "math_cot_probe")
    s_file = s_dir / "gamma_s.jsonl"
    s_root = inputs_root / "S"   # GetSlice reads {s_root}/{task}/gamma_s.jsonl
    if not s_file.exists():
        rows = read_jsonl(Path(data_meta["probe_jsonl"]))[:n_probe]
        write_jsonl(s_file, [{"question": r["question"], "answer": r["answer"]} for r in rows])
        print(f"[S PROBE] {len(rows)} rows -> {s_file}", flush=True)

    # X probe: same prompts, completions generated once from step_000 (base) via worker.
    x_file = inputs_root / "X_base" / "x_probe.jsonl"
    if not x_file.exists():
        cfg = {
            "base_model": str(BASE_MODEL), "prompts_jsonl": data_meta["probe_jsonl"],
            "out_jsonl": str(x_file), "n_probe": n_probe,
            "max_new_tokens": max_new_tokens, "instr": INSTR,
        }
        _run_gpu_worker("rollout-x", cfg, gs_root / "_xprobe_config.json")
    return s_root, x_file


def _run_slice(cfg_path: Path) -> None:
    env = dict(os.environ, TMPDIR="/root/autodl-tmp/pip-tmp",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
    cmd = [sys.executable, str(REPO_ROOT / "GetSlice/slice.py"), "--config", str(cfg_path)]
    print(f"[SLICE] {cfg_path.name}", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO_ROOT / "GetSlice"), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"GetSlice failed rc={r.returncode}: {cfg_path}")


def _getslice_checkpoint(model_path: Path, label: str, s_root: Path, x_file: Path,
                         run_root: Path, layer: int, *, seqlen: int, n_probe: int) -> None:
    gs_root = run_root / "getslice"
    out_root = ensure_dir(gs_root / "outputs")
    cfg_root = ensure_dir(gs_root / "configs")
    step_root = out_root / label

    base_cfg = dict(_GETSLICE_BASE_CFG)
    base_cfg["model_seq_len"] = seqlen
    base_cfg["target_layer"] = layer

    # GetSlice writes to {save_path}/{task}/layer_{N}/{s,x}Mat_{task}.json — note the extra
    # nested layer_{N} directory it creates under the task folder.
    s_save = step_root / f"layer_{layer}" / "S"
    s_mat = s_save / "math_cot_probe" / f"layer_{layer}" / "sMat_math_cot_probe.json"
    if not s_mat.exists():
        cfg = dict(base_cfg)
        cfg.update({"model": str(model_path), "save_path": str(s_save), "mode": "s_only_svd",
                    "s_nsamples": n_probe, "s_jsonl_path": str(s_root), "s_jsonl_file": "gamma_s.jsonl"})
        cfg_path = cfg_root / f"{label}__layer{layer}__S.json"
        write_json(cfg_path, cfg)
        _run_slice(cfg_path)

    x_save = step_root / f"layer_{layer}" / "X"
    x_mat = x_save / "X" / f"layer_{layer}" / "xMat_X.json"
    if not x_mat.exists():
        cfg = dict(base_cfg)
        cfg.update({"model": str(model_path), "save_path": str(x_save), "mode": "x_only_svd",
                    "x_nsamples": n_probe, "x_jsonl_path": str(x_file)})
        cfg_path = cfg_root / f"{label}__layer{layer}__X.json"
        write_json(cfg_path, cfg)
        _run_slice(cfg_path)


def phase4_geometry(run_root: Path, data_meta: dict, train_info: dict, *, smoke: bool) -> None:
    from scripts.export_weights import export_model_weights, MODULES

    ckpt_root = Path(train_info["ckpt_root"])
    merged_root = ensure_dir(run_root / "_merged_tmp")
    weights_root = ensure_dir(run_root / "weights")
    layers = [18] if smoke else GEOMETRY_LAYERS
    n_probe = 4 if smoke else N_PROBE
    seqlen = 64 if smoke else 512
    steps = [0] + train_info["steps"]

    s_root, x_file = _build_probes(run_root, data_meta, n_probe=n_probe,
                                   max_new_tokens=(64 if smoke else 512))

    modules = MODULES  # 7 LoRA modules with attribute paths
    for step in steps:
        label = step_label(step)
        # Skip if getslice json + weight npy already present for all layers
        gs_done = all(
            (run_root / "getslice/outputs" / label
             / f"layer_{L}/S/math_cot_probe/layer_{L}/sMat_math_cot_probe.json").exists()
            and (run_root / "getslice/outputs" / label
                 / f"layer_{L}/X/X/layer_{L}/xMat_X.json").exists()
            for L in layers)
        w_dir = (weights_root / label) if step == 0 else (weights_root / label / str(step))
        n_wfiles = len(layers) * len(modules)
        w_done = w_dir.exists() and len(list(w_dir.glob("*.npy"))) >= n_wfiles
        if gs_done and w_done:
            print(f"[SKIP] geometry done for {label}", flush=True)
            continue

        model_path = _merge_checkpoint(step, ckpt_root, merged_root)
        try:
            for L in layers:
                _getslice_checkpoint(Path(model_path), label, s_root, x_file, run_root, L,
                                     seqlen=seqlen, n_probe=n_probe)
            export_model_weights(str(model_path), w_dir, layers=layers, modules=modules)
        finally:
            _drop_merged(step, merged_root)

    # OverlapLift via principalEvidence (weight-space), each checkpoint vs step_000
    if not smoke:
        _run_principal_evidence(run_root, weights_root, train_info["steps"], layers)
    # geometry_metrics CSVs (effective_rank / spectral_gap / drift) from getslice json
    _build_geometry_metrics(run_root, [0] + train_info["steps"], layers)


def _run_principal_evidence(run_root: Path, weights_root: Path, steps: list[int], layers: list[int]) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from AnalyseMat.principalEvidence import run_principal_evidence
    pe_root = ensure_dir(run_root / "principal_evidence")
    base_dir = weights_root / step_label(0)            # flat export of step_000
    tasks = [[step_label(s), str(s)] for s in steps]   # finetuned at weights/{label}/{s}/
    for layer in layers:
        cfg = {"analyse": {
            "base_model_npy_dir": str(base_dir),
            "npy_output_root": str(weights_root),
            "related_work": {
                "enable": True, "output_root": str(pe_root / f"layer_{layer}"),
                "target_layer": layer, "target_modules": None,
                "principal_rank_k": 50, "principal_top_ratio": 0.01, "save_png": True,
            },
            "tasks": tasks,
        }}
        print(f"[PE] layer={layer}", flush=True)
        run_principal_evidence(cfg)

    # OverlapLift CSVs are written; the bulky per-checkpoint weight .npy (~11GB peak) are now
    # disposable. Remove them to reclaim disk (re-derivable by re-running the phase-4 export).
    for d in weights_root.glob("step_*"):
        shutil.rmtree(d, ignore_errors=True)
    print(f"[PE] cleaned weight npy under {weights_root}", flush=True)


def _build_geometry_metrics(run_root: Path, steps: list[int], layers: list[int]) -> None:
    """Compute effective_rank / spectral_gap / drift_from_base / xs_gap from getslice JSON."""
    from opd_sft_h1.geometry_metrics import (
        effective_rank, spectral_gap, log_spectrum_drift, xs_log_spectrum_gap)

    out_root = run_root / "getslice/outputs"
    geo_dir = ensure_dir(run_root / "geometry")

    def _load(label, layer, side):
        sub = (f"S/math_cot_probe/layer_{layer}/sMat_math_cot_probe.json" if side == "S"
               else f"X/X/layer_{layer}/xMat_X.json")
        p = out_root / label / f"layer_{layer}" / sub
        if not p.exists():
            return {}
        d = json.loads(p.read_text())
        layer_key = f"layer_{layer}"
        return d.get(layer_key, {})

    base_s = {L: _load(step_label(0), L, "S") for L in layers}
    for step in steps:
        label = step_label(step)
        rows = []
        for L in layers:
            s_mod = _load(label, L, "S")
            x_mod = _load(label, L, "X")
            for module, sigma_s in s_mod.items():
                sigma_x = x_mod.get(module)
                sigma_s0 = base_s.get(L, {}).get(module)
                rows.append({
                    "step": step, "layer": L, "module": module,
                    "effective_rank": effective_rank(sigma_s),
                    "spectral_gap": spectral_gap(sigma_s, 1),
                    "drift_from_base": (log_spectrum_drift(sigma_s, sigma_s0)
                                        if sigma_s0 is not None else None),
                    "xs_log_spectrum_gap": (xs_log_spectrum_gap(sigma_x, sigma_s)
                                            if sigma_x is not None else None),
                })
        if rows:
            _write_csv(geo_dir / f"geometry_metrics_{label}.csv", rows)
    print(f"[GEO] geometry_metrics CSVs -> {geo_dir}", flush=True)


# =============================================================================
# Aggregation + copyback
# =============================================================================
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _read_lm_acc(output_path: Path) -> tuple[float | None, float | None]:
    files = list(output_path.glob("**/results_*.json"))
    if not files:
        return None, None
    d = json.loads(sorted(files)[-1].read_text())
    # Grouped tasks (e.g. mmlu_pro, 14 subtasks) carry the aggregate in "groups";
    # single tasks (gpqa) carry it in "results". mmlu_pro reports under the
    # custom-extract metric, not ",none". Prefer the group aggregate.
    metric_keys = ("acc,none", "acc_norm,none", "exact_match,none", "exact_match,custom-extract")
    for block in (d.get("groups", {}), d.get("results", {})):
        for _task, metrics in block.items():
            for key in metric_keys:
                if key in metrics:
                    se_key = key.replace(",", "_stderr,", 1)
                    return float(metrics[key]), (float(metrics[se_key]) if se_key in metrics else None)
    return None, None


def aggregate(run_root: Path, copyback_root: Path, train_info: dict, data_meta: dict, *, smoke: bool) -> None:
    eval_root = run_root / "eval"
    steps = [0] + train_info["steps"]
    lm_subs = [s for (s, *_r) in (LM_TASKS if not smoke else [("gpqa",)])]

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
            acc, se = _read_lm_acc(eval_root / label / sub)
            row[f"{sub}_acc"] = round(acc, 4) if acc is not None else None
            row[f"{sub}_se"] = round(se, 4) if se is not None else None
        traj_rows.append(row)

    ensure_dir(copyback_root)
    _write_csv(copyback_root / "trajectory_scores.csv", traj_rows)
    _write_csv(copyback_root / "response_length_trajectory.csv", rlen_rows)
    # mirror geometry CSVs into copyback
    geo_src = run_root / "geometry"
    if geo_src.exists():
        geo_dst = ensure_dir(copyback_root / "geometry")
        for f in geo_src.glob("geometry_metrics_*.csv"):
            shutil.copy2(f, geo_dst / f.name)
    pe_src = run_root / "principal_evidence"
    if pe_src.exists():
        for f in pe_src.glob("**/*.csv"):
            shutil.copy2(f, ensure_dir(copyback_root / "geometry") / f"principal_{f.parent.name}_{f.name}")

    _write_results_md(copyback_root, traj_rows, rlen_rows, train_info, data_meta)
    _write_provenance(copyback_root, run_root, train_info, data_meta)
    _make_figures(copyback_root, traj_rows, rlen_rows)
    print(f"[AGG] copyback -> {copyback_root}", flush=True)


def _write_results_md(copyback_root: Path, traj, rlen, train_info, data_meta) -> None:
    def g(row, k):
        v = row.get(k)
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else v

    base = traj[0]
    final = traj[-1]
    m0 = base.get("math500_acc")
    mf = final.get("math500_acc")
    a07 = "UNDETERMINED"
    if isinstance(m0, (int, float)) and isinstance(mf, (int, float)):
        thr = m0 + binom_se(m0, 500)
        a07 = f"{'PASS' if mf > thr else 'FAIL'} (final {mf:.3f} vs base+SE {thr:.3f})"
    # B07: dip + recovery on math500
    dip = next((r["step"] for r in traj if r["step"] in (5, 10, 20, 40)
                and isinstance(r.get("math500_acc"), (int, float)) and isinstance(m0, (int, float))
                and r["math500_acc"] < m0), None)
    rec = next((r["step"] for r in traj if isinstance(r.get("math500_acc"), (int, float))
                and isinstance(m0, (int, float)) and r["step"] > 0 and r["math500_acc"] > m0), None)
    b07 = ("FULL PASS" if dip is not None and rec is not None else
           "PARTIAL" if rec is not None else "FAIL")

    lines = [
        "# RESULTS_07: Cycle 07 — Base-Model SFT Trajectory (Qwen3-4B-Base + Math-CoT-20k)",
        "", "```yaml",
        "cycle: cycle_07_base_sft_trajectory",
        f"model: {BASE_MODEL}",
        f"data: {DATA_PARQUET} (n_train={data_meta.get('n_train')}, seed={SEED})",
        f"total_steps: {train_info.get('total_steps')}",
        f"checkpoint_grid: {[0] + train_info.get('steps', [])}",
        "```", "",
        "## Gate Verdicts",
        f"- **A07 (feasibility, MATH500@final > base+1SE):** {a07}",
        f"- **B07 (dip-and-recovery):** {b07}  (dip step={dip}, recover step={rec})",
        "- **C07 (OOD-lite gain):** see GPQA-D / MMLU-Pro final vs step_000 below.",
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
    lines += ["", "Artifacts: trajectory_scores.csv, response_length_trajectory.csv, "
              "geometry/. Raw (per-sample jsonl, getslice activations, UV, weights) under run-root.", ""]
    (copyback_root / "RESULTS_07.md").write_text("\n".join(lines))


def _write_provenance(copyback_root: Path, run_root: Path, train_info, data_meta) -> None:
    def _git_hash():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    prov = {
        "cycle": "cycle_07_base_sft_trajectory",
        "git_hash": _git_hash(),
        "run_root": str(run_root),
        "base_model": str(BASE_MODEL),
        "data_meta": data_meta,
        "train_info": train_info,
        "checkpoint_grid": [0] + train_info.get("steps", []),
        "geometry_layers": GEOMETRY_LAYERS,
        "seed": SEED,
        "eval_protocol": {
            "generative": "chat template, think-format (no enable_thinking=False), "
                          "temp=0.6 top_p=0.9 max_tokens=32768",
            "lm_eval": "loglikelihood, no chat template, 0-shot; gpqa_diamond_zeroshot; "
                       "mmlu_pro --limit 500 --seed 42",
        },
        "runner": str(THINK_RUNNER),
    }
    write_json(copyback_root / "run_provenance.json", prov)


def _make_figures(copyback_root: Path, traj, rlen) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[FIG] matplotlib unavailable ({e}); skipping figures", flush=True)
        return
    fig_dir = ensure_dir(copyback_root / "figures")
    xs = [r["step"] for r in traj]

    def _series(key):
        return [r.get(key) if isinstance(r.get(key), (int, float)) else None for r in traj]

    plt.figure()
    plt.plot(xs, _series("math500_acc"), "o-")
    plt.xlabel("step"); plt.ylabel("MATH500 acc"); plt.title("MATH500 trajectory")
    plt.savefig(fig_dir / "trajectory_math500.png", dpi=120, bbox_inches="tight"); plt.close()

    plt.figure()
    for key, lab in [("gpqa_acc", "GPQA-D"), ("mmlu_pro_acc", "MMLU-Pro"), ("aime24_acc", "AIME24")]:
        plt.plot(xs, _series(key), "o-", label=lab)
    plt.xlabel("step"); plt.ylabel("acc"); plt.legend(); plt.title("OOD trajectory")
    plt.savefig(fig_dir / "trajectory_ood.png", dpi=120, bbox_inches="tight"); plt.close()

    if rlen:
        plt.figure()
        plt.plot([r["step"] for r in rlen], [r["mean_response_len_math500"] for r in rlen], "o-")
        plt.xlabel("step"); plt.ylabel("mean response tokens"); plt.title("MATH500 response length")
        plt.savefig(fig_dir / "response_length.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"[FIG] -> {fig_dir}", flush=True)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Cycle 07: Base-Model SFT Trajectory")
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--copyback-root", type=Path, default=COPYBACK_ROOT_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start-from-phase", type=int, default=1, choices=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    run_root = ensure_dir(args.run_root)
    smoke = args.smoke
    start = args.start_from_phase
    print(f"=== Cycle 07 === run_root={run_root} smoke={smoke} start_phase={start}", flush=True)

    if not (BASE_MODEL / "config.json").exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    if not DATA_PARQUET.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PARQUET}")

    # Phase 1
    if start <= 1:
        print("\n=== Phase 1: Data ===", flush=True)
        data_meta = phase1_data(run_root, smoke=smoke)
    else:
        data_meta = json.loads((run_root / "data_prep/data_meta.json").read_text())

    # Phase 2
    if start <= 2:
        print("\n=== Phase 2: Training ===", flush=True)
        train_info = phase2_train(run_root, data_meta, smoke=smoke)
        write_json(run_root / "train_info.json", train_info)
    else:
        train_info = json.loads((run_root / "train_info.json").read_text())

    # Phase 3
    if start <= 3:
        print("\n=== Phase 3: Evaluation ===", flush=True)
        phase3_eval(run_root, train_info, smoke=smoke)

    # Phase 4
    if start <= 4:
        print("\n=== Phase 4: Geometry ===", flush=True)
        phase4_geometry(run_root, data_meta, train_info, smoke=smoke)

    # Phase 5: aggregate + copyback
    print("\n=== Phase 5: Aggregate + Copyback ===", flush=True)
    aggregate(run_root, args.copyback_root, train_info, data_meta, smoke=smoke)
    print("\n=== Cycle 07 complete ===", flush=True)


if __name__ == "__main__":
    main()
