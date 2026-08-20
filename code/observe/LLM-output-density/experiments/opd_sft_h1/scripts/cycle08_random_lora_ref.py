#!/usr/bin/env python3
"""Cycle 08 A08 supplement: random-LoRA reference OverlapLift (CPU / weight-space).

The A08 gate compares OPD vs SFT OverlapLift against a RANDOM-LoRA null of the same
base / rank / target modules, module-wise scale-matched to the trained adapter. This
is pure weight-space linear algebra (SVD subspace overlap) — reuses export_weights +
AnalyseMat.principalEvidence, no model forward / vLLM / GPU generation.

For each grid step:
  1. export the OPD merged model's weight matrices (temp) -> trained Wp
  2. dW = Wp - W0 (base);  build a random rank-R update dR = B@A (N(0,1)) scaled so
     ||dR||_F == ||dW||_F, then bf16-round W0+dR (mirror bf16-trained storage granularity)
  3. write W0+dR as the "finetuned" npy under weights/rand_{step}/{step}/
Then run principal_evidence(base=step_000, tasks=rand_*) -> random OverlapLift trajectory.
Expected: random OverlapLift ~ 0 (random subspace vs base principal subspace), which
anchors that OPD/SFT (0.34-0.76) sit far above chance.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/root/LLM-output-density")
SIDECAR = REPO_ROOT / "experiments/opd_sft_h1"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIDECAR))
from scripts.export_weights import export_model_weights, MODULES  # noqa: E402
from AnalyseMat.principalEvidence import run_principal_evidence    # noqa: E402

BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
RUN_ROOT = Path("/root/autodl-tmp/cycle08_opd_trajectory")
MERGED = {  # arm -> merged-model dir producing trained weights for scale-matching
    "opd": RUN_ROOT / "_merged_models",
    "sft": None,  # SFT uses LoRA adapters merged on the fly; opd-scale null is enough for A08
}
WEIGHTS = RUN_ROOT / "weights"
PE_OUT = RUN_ROOT / "principal_evidence_random"
LAYERS = [9, 18, 27]
LORA_RANK = 32
STEPS = [5, 10, 20, 40, 80, 160, 320, 480, 624]
SEED = 0


def _bf16_round(a: np.ndarray) -> np.ndarray:
    import torch
    return torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()


def main():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    base_dir = WEIGHTS / "step_000"

    # 1. base weights (re-export if the post-PE cleanup removed them)
    export_model_weights(str(BASE_MODEL), base_dir, layers=LAYERS, modules=MODULES)

    rng = np.random.default_rng(SEED)
    merged_root = MERGED["opd"]
    for s in STEPS:
        rand_dir = WEIGHTS / f"rand_{s:03d}" / str(s)
        if rand_dir.exists() and len(list(rand_dir.glob("*.npy"))) >= len(LAYERS) * len(MODULES):
            print(f"[rand] step {s}: exists, skip", flush=True)
            continue
        merged = merged_root / f"step_{s:03d}"
        if not (merged / "config.json").exists():
            print(f"[rand] step {s}: merged model missing at {merged}, skip", flush=True)
            continue
        tmp = WEIGHTS / f"_tmp_opd_{s:03d}"
        print(f"[rand] step {s}: exporting trained weights for scale ...", flush=True)
        export_model_weights(str(merged), tmp, layers=LAYERS, modules=MODULES)
        rand_dir.mkdir(parents=True, exist_ok=True)
        for L in LAYERS:
            for mod_name, _ in MODULES:
                fn = f"model_layers_{L}_{mod_name}_weight.npy"
                W0 = np.load(base_dir / fn).astype(np.float32)
                Wp = np.load(tmp / fn).astype(np.float32)
                dW = Wp - W0
                scale = float(np.linalg.norm(dW))
                out, inp = W0.shape
                A = rng.standard_normal((LORA_RANK, inp)).astype(np.float32)
                B = rng.standard_normal((out, LORA_RANK)).astype(np.float32)
                dR = B @ A
                nrm = float(np.linalg.norm(dR))
                if nrm > 0:
                    dR *= scale / nrm            # match trained update Frobenius norm
                Wr = _bf16_round(W0 + dR)         # mirror bf16-trained storage granularity
                np.save(rand_dir / fn, Wr)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[rand] step {s}: wrote random-LoRA weights -> {rand_dir}", flush=True)

    # 3. principal evidence for the random references
    tasks = [[f"rand_{s:03d}", str(s)] for s in STEPS]
    for L in LAYERS:
        cfg = {"analyse": {
            "base_model_npy_dir": str(base_dir),
            "npy_output_root": str(WEIGHTS),
            "related_work": {
                "enable": True, "output_root": str(PE_OUT / f"layer_{L}"),
                "target_layer": L, "target_modules": None,
                "principal_rank_k": 50, "principal_top_ratio": 0.01, "save_png": False,
            },
            "tasks": tasks,
        }}
        print(f"[rand-PE] layer={L}", flush=True)
        run_principal_evidence(cfg)

    # 4. aggregate random OverlapLift per step (mean over layers*modules)
    import csv, glob, collections, statistics as st
    data = collections.defaultdict(list)
    for f in glob.glob(str(PE_OUT / "layer_*" / "principal_evidence.csv")):
        for r in csv.DictReader(open(f)):
            data[r["Source"]].append(r)
    out_csv = RUN_ROOT / "geometry" / "random_lora_overlap_lift.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "mean_overlap_lift", "mean_jaccard", "mean_uangle_mean_deg", "n"])
        for s in STEPS:
            rows = data.get(f"rand_{s:03d}", [])
            if not rows:
                continue
            def m(c):
                v = [float(r[c]) for r in rows if r[c] not in ("", "nan")]
                return st.mean(v) if v else float("nan")
            w.writerow([s, f"{m('OverlapLift'):.4f}", f"{m('Jaccard'):.4f}",
                        f"{m('UAngleMeanDeg'):.4f}", len(rows)])
    print(f"[rand] wrote {out_csv}", flush=True)
    print("[rand] DONE", flush=True)


if __name__ == "__main__":
    main()
