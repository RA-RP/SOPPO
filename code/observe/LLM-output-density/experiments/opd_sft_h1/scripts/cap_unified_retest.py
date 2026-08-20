#!/usr/bin/env python3
"""Unified-cap re-test of the generative math benchmarks for BOTH trajectories.

WHY: the as-run eval used max_tokens=4096 (math500/numina) / 16384 (aime24). That cap is
too short for OPD: on-policy distillation of the Qwen3-8B teacher gives the student a very
long CoT whose \\boxed{} answer is emitted LATE, so 4096 clips the answer and the late-step
MATH500/numina acc are biased LOW. (SFT is cap-robust at 4096 — short answers, box early —
per the c07 cap-pilot.) Verified: OPD step_320 MATH500 acc 0.680@4096 -> 0.910@24576 (+0.23,
23/100 flips wrong->right). To compare fairly, BOTH arms are re-tested at a UNIFIED longer cap.

WHAT: math500/numina @ cap 24576, aime24 @ cap 31744 (c07 cap-pilot: math trunc -> ~0 at 24576;
aime is longest). Records acc + trunc_rate (finish=="length") + mean_response_len as first-class.

WHERE TO RUN: a >=2 GPU box (2x96G). Uses the density env's vLLM. Checkpoint-parallel across GPUs.
  # DEFAULT = QUICK CHECK (agreed plan): MATH500 x late steps [40,80,160,320,480,624] x both arms
  /root/miniconda3/envs/density/bin/python cap_unified_retest.py
  # widen only if the quick check warrants it:
  ... --full                        # whole grid [0..624] x {math500,numina,aime24} x both arms
  ... --arms opd                    # one arm
  ... --steps 320,480,624 --tasks math500   # custom scope
Outputs -> --out (default /root/autodl-tmp/cap_unified_retest/), then aggregates a corrected
trajectory CSV + opd_vs_sft_unified.md. Idempotent: existing per-(arm,step,task) json are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR = REPO_ROOT / "experiments/opd_sft_h1"
sys.path.insert(0, str(SIDECAR))                       # so `scripts.run_opd_minimal_closure` imports
RUNNER = REPO_ROOT / "Eval/component/think_math/runner_think.py"
PY = sys.executable
BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")

GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
# QUICK CHECK (agreed plan): only the high-truncation late steps + MATH500 + both arms.
# step_40 (the apparent as-run peak 0.802) is included as an anchor to show the "decline" is a
# truncation artifact once corrected. Use --full for the whole grid / all 3 tasks.
FAST_STEPS = [40, 80, 160, 320, 480, 624]
FAST_TASKS = ["math500"]
# UNIFIED caps  ->  task: (n, max_tokens, max_model_len)   (0 = full task set)
TASKS = {
    "math500": (500, 16384, 17408),   # 16384 chosen via cap calibration (acc 0.88 vs 0.91@24576; user-approved trade for ~40% speedup)
    "numina":  (256, 24576, 25600),
    "aime24":  (0,   31744, 32768),
}
# as-run caps, for the aggregate delta table (what we are correcting FROM)
ASRUN_CAP = {"math500": 4096, "numina": 4096, "aime24": 16384}

ARMS = {
    "opd": {  # cycle08 — models already merged (raw verl ckpts were pruned)
        "run_root": Path("/root/autodl-tmp/cycle08_opd_trajectory"),
        "kind": "merged",
        "merged_dir": Path("/root/autodl-tmp/cycle08_opd_trajectory/_merged_models"),
    },
    "sft": {  # cycle07 — LoRA adapters, merge on the fly
        "run_root": Path("/root/autodl-tmp/cycle07_base_sft_trajectory"),
        "kind": "adapter",
        "ckpt_root": Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints"),
    },
}


def step_label(s: int) -> str:
    return f"step_{s:03d}"


# ---- per-worker GPU pinning (each worker owns one physical GPU for its lifetime) ----
def _init(counter, ngpu):
    with counter.get_lock():
        gpu = counter.value % ngpu
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)      # inherited by runner_think subprocesses
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_model(arm: str, step: int, tmp_root: Path):
    """Return (model_path, dir_to_drop_after). LoRA merge runs on CPU (no GPU context in worker)."""
    if step == 0:
        return str(BASE_MODEL), None
    cfg = ARMS[arm]
    if cfg["kind"] == "merged":
        return str(cfg["merged_dir"] / step_label(step)), None
    from scripts.run_opd_minimal_closure import merge_lora_adapter  # noqa: E402
    adapter = cfg["ckpt_root"] / step_label(step)
    merged = tmp_root / f"{arm}_{step_label(step)}"
    merge_lora_adapter(BASE_MODEL, adapter, merged)       # CPU merge -> HF dir
    return str(merged), merged


def run_one(job):
    arm, step, tasks, out_root = job
    label = step_label(step)
    tmp_root = out_root / "_merge_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        model, to_drop = _resolve_model(arm, step, tmp_root)
    except Exception as e:  # noqa: BLE001
        return f"{arm}/{label} MERGE-FAIL: {e}"
    done = []
    for t in tasks:
        n, cap, mml = TASKS[t]
        od = out_root / arm / label / t
        od.mkdir(parents=True, exist_ok=True)
        if (od / f"{label}.json").exists():
            done.append(f"{t}=skip"); continue
        cmd = [PY, str(RUNNER), "--task", t, "--model", model, "--label", label,
               "--n", str(n), "--max-tokens", str(cap), "--max-model-len", str(mml),
               "--gpu-mem", "0.85", "--temperature", "0.6", "--top-p", "0.9",
               "--seed", "42", "--outdir", str(od)]
        rc = subprocess.run(cmd, cwd=str(SIDECAR)).returncode
        done.append(f"{t}=rc{rc}")
    if to_drop is not None:
        shutil.rmtree(to_drop, ignore_errors=True)       # reclaim ~8G per merged SFT ckpt
    return f"{arm}/{label} [{os.environ.get('CUDA_VISIBLE_DEVICES')}] " + " ".join(done)


def aggregate(out_root: Path, arms, steps, tasks):
    def read(arm, step, task):
        p = out_root / arm / step_label(step) / task / f"{step_label(step)}.json"
        if not p.exists():
            return None
        d = json.load(open(p))
        return d.get("acc"), d.get("trunc_rate"), d.get("mean_response_len")

    csv_path = out_root / "cap_unified_trajectory.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["arm", "step"] + sum(([f"{t}_acc", f"{t}_trunc", f"{t}_meanlen"] for t in tasks), [])
        w.writerow(cols)
        for arm in arms:
            for s in steps:
                row = [arm, s]
                for t in tasks:
                    r = read(arm, s, t)
                    row += list(r) if r else ["", "", ""]
                w.writerow(row)
    print(f"[agg] wrote {csv_path}", flush=True)

    # OPD vs SFT unified MATH500 table (only if both arms present)
    if {"opd", "sft"} <= set(arms) and "math500" in tasks:
        md = out_root / "opd_vs_sft_unified.md"
        with open(md, "w") as f:
            f.write("# OPD vs SFT — UNIFIED cap re-test (MATH500 @ 24576, as-run was 4096)\n\n")
            f.write("| step | OPD acc | SFT acc | Δ(OPD−SFT) | OPD trunc | SFT trunc |\n")
            f.write("|---|---|---|---|---|---|\n")
            for s in steps:
                o = read("opd", s, "math500"); sf = read("sft", s, "math500")
                if not o or not sf:
                    continue
                f.write(f"| {s} | {o[0]:.3f} | {sf[0]:.3f} | {o[0]-sf[0]:+.3f} | {o[1]:.2f} | {sf[1]:.2f} |\n")
        print(f"[agg] wrote {md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="opd,sft")
    ap.add_argument("--steps", default="")   # empty -> FAST_STEPS (quick check) unless --full
    ap.add_argument("--tasks", default="")   # empty -> FAST_TASKS (quick check) unless --full
    ap.add_argument("--full", action="store_true",
                    help="whole grid + all 3 tasks (default is the QUICK CHECK: MATH500 x late steps)")
    ap.add_argument("--n-gpus", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/cap_unified_retest"))
    ap.add_argument("--agg-only", action="store_true", help="skip eval, just re-aggregate")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    steps = ([int(x) for x in args.steps.split(",") if x != ""] if args.steps
             else (GRID if args.full else FAST_STEPS))
    tasks = ([t for t in args.tasks.split(",") if t] if args.tasks
             else (["math500", "numina", "aime24"] if args.full else FAST_TASKS))
    print(f"[retest] mode={'FULL' if args.full else 'QUICK-CHECK'}  arms={arms} steps={steps} tasks={tasks}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.agg_only:
        jobs = [(arm, s, tasks, args.out) for arm in arms for s in steps]
        print(f"[retest] {len(jobs)} (arm,step) jobs x {tasks} across {args.n_gpus} GPU(s); "
              f"caps: {{ {', '.join(f'{t}:{TASKS[t][1]}' for t in tasks)} }}", flush=True)
        counter = mp.Value("i", 0)
        with ProcessPoolExecutor(max_workers=args.n_gpus, initializer=_init,
                                 initargs=(counter, args.n_gpus)) as ex:
            for r in ex.map(run_one, jobs):
                print("[retest]", r, flush=True)
    aggregate(args.out, arms, steps, tasks)
    print("[retest] DONE", flush=True)


if __name__ == "__main__":
    main()
