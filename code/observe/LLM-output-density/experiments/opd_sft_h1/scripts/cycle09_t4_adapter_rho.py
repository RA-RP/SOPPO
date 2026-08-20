#!/usr/bin/env python3
"""Cycle 09 mini T4: A09 adapter-first rho [QA1=b]. Zero training.
Tracks:
  SFT-fp32-BA   : dW = (alpha/r)*B@A from the fp32 adapter (rank<=32, clean)          -> "process" track
  SFT-bf16-BA   : same dW rounded to bf16                                             -> bf16 storage effect
  OPD-top32     : top-32 SVD of (bf16-merged - base)  (APPROXIMATION, OPD adapter pruned)
Compare to the bf16-merged rho already in rho_summary.md. rho^2 = ||Uk^T dW||^2/||dW||^2 (scale-invariant),
vs spectrum-matched random-rotation null (20 draws, z-verdict). fp32 linear algebra.
NOTE (guards): no on/off-principal conclusion here — dual-track numbers returned to Theory.
Outputs T4_rho_dualtrack.csv + T4_rho_summary.md to mini/.
"""
import glob, json, shutil, struct, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path("/root/LLM-output-density"); SIDE = REPO / "experiments/opd_sft_h1"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SIDE))
from scripts.export_weights import export_model_weights, MODULES  # noqa: E402

BASE_NPY = Path("/root/autodl-tmp/cycle08_opd_trajectory/weights/step_000")
OPD_MERGED = Path("/root/autodl-tmp/cycle08_opd_trajectory/_merged_models")
SFT_ADAPTER = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")
WTMP = Path("/root/autodl-tmp/cycle08_opd_trajectory/weights")
OUT = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
           "cycle_09_aaai_competitiveness_completion/run_01/mini")
LAYERS = [9, 18, 27]
STEPS = [5, 10, 20, 40, 80, 160, 320, 480, 624]
KS = [32, 128]
KMAX = 256
N_DRAWS = 20
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALING = 64 / 32  # alpha/r, use_rslora=False


def npy(L, mod): return f"model_layers_{L}_{mod}_weight.npy"


def base_svd():
    cache = {}
    for L in LAYERS:
        for mod, _ in MODULES:
            W0 = torch.from_numpy(np.load(BASE_NPY / npy(L, mod))).to(torch.float32).to(DEV)
            U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
            km = min(KMAX, U.shape[1])
            cache[(L, mod)] = (U[:, :km].contiguous(), Vh[:km, :].T.contiguous(), W0)
    return cache


def rho2(dW, U, V, dW_sq):
    A = U.T @ dW; B = dW @ V; C = A @ V
    cAu = torch.cumsum((A * A).sum(1), 0); cBv = torch.cumsum((B * B).sum(0), 0); C2 = C * C
    out = {}
    for k in KS:
        kk = min(k, U.shape[1])
        out[k] = (float(cAu[kk - 1] / dW_sq), float(cBv[kk - 1] / dW_sq), float(C2[:kk, :kk].sum() / dW_sq))
    return out


def null_z(dW, U, V, rng):
    S = torch.linalg.svdvals(dW); r = int((S > 1e-6 * S[0]).sum().item()); Sr = S[:r]
    m, n = dW.shape
    acc = {k: [] for k in KS}
    for _ in range(N_DRAWS):
        Pp = torch.linalg.qr(torch.randn(m, r, device=DEV, generator=rng))[0]
        Qp = torch.linalg.qr(torch.randn(n, r, device=DEV, generator=rng))[0]
        dWr = (Pp * Sr) @ Qp.T
        rr = rho2(dWr, U, V, float((dWr * dWr).sum()))
        for k in KS:
            acc[k].append(rr[k][0])  # U component
    return r, {k: (float(np.mean(acc[k])), float(np.std(acc[k]))) for k in KS}


def sft_dW(step, L, mod, cache):
    sd = load_file(SFT_ADAPTER / f"step_{step:03d}" / "adapter_model.safetensors")
    pre = f"base_model.model.model.layers.{L}.{mod}"
    A = sd[f"{pre}.lora_A.weight"].to(torch.float32)   # (r, in)
    B = sd[f"{pre}.lora_B.weight"].to(torch.float32)   # (out, r)
    dW = SCALING * (B @ A)
    return dW.to(DEV)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = base_svd()
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    rows = []  # track, step, layer, module, k, rho2_U, null_mean, null_std, z, dW_rank

    def record(track, step, L, mod, dW):
        U, V, _ = cache[(L, mod)]
        dW_sq = float((dW * dW).sum())
        if dW_sq < 1e-20:
            return
        real = rho2(dW, U, V, dW_sq)
        r, nul = null_z(dW, U, V, gen)
        for k in KS:
            ru = real[k][0]; nm, ns = nul[k]
            z = (ru - nm) / ns if ns > 0 else float("nan")
            rows.append([track, step, L, mod, k, f"{ru:.5f}", f"{nm:.5f}", f"{ns:.5f}", f"{z:+.2f}", r])

    for step in STEPS:
        # SFT dual-track from adapter
        for L in LAYERS:
            for mod, _ in MODULES:
                dW = sft_dW(step, L, mod, cache)
                record("sft_fp32_BA", step, L, mod, dW)
                record("sft_bf16_BA", step, L, mod, dW.to(torch.bfloat16).to(torch.float32))
        # OPD top-32 denoise from merged export
        tmp = WTMP / f"_t4_opd_{step:03d}"
        tmp.mkdir(parents=True, exist_ok=True)
        export_model_weights(str(OPD_MERGED / f"step_{step:03d}"), tmp, layers=LAYERS, modules=MODULES)
        for L in LAYERS:
            for mod, _ in MODULES:
                W0 = cache[(L, mod)][2]
                Wp = torch.from_numpy(np.load(tmp / npy(L, mod))).to(torch.float32).to(DEV)
                dW = Wp - W0
                Ud, Sd, Vhd = torch.linalg.svd(dW, full_matrices=False)
                dW32 = (Ud[:, :32] * Sd[:32]) @ Vhd[:32, :]   # rank-32 approximation
                record("opd_top32_approx", step, L, mod, dW32)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[T4] step {step} done", flush=True)

    import csv
    with open(OUT / "T4_rho_dualtrack.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "step", "layer", "module", "k", "rho2_U", "null_mean", "null_std", "z", "dW_rank"])
        w.writerows(rows)

    # aggregate mean over layer*module, k=32
    import collections, statistics
    agg = collections.defaultdict(list); zagg = collections.defaultdict(list); ragg = collections.defaultdict(list)
    for tr, step, L, mod, k, ru, nm, ns, z, r in rows:
        if int(k) == 32:
            agg[(tr, step)].append(float(ru)); zagg[(tr, step)].append(float(z)); ragg[tr].append(int(r))
    lines = ["# Cycle 09 mini T4 — A09 adapter rho, dual-track (k=32, mean over layer×module). NO on/off conclusion (→Theory).\n",
             "Tracks: sft_fp32_BA (clean rank≤32 process), sft_bf16_BA (bf16-rounded), opd_top32_approx "
             "(APPROX: top-32 of bf16-merged−base; OPD adapter pruned). Compare bf16-merged rho in rho_summary.md "
             "(SFT≈0.34→0.68, OPD≈0.37→0.76 at k32).\n",
             f"dW numerical rank (mean): " + ", ".join(f"{t}≈{statistics.mean(ragg[t]):.0f}" for t in ragg) + ".\n",
             "\n| step | sft_fp32 ρ²_U (z) | sft_bf16 ρ²_U (z) | opd_top32 ρ²_U (z) |",
             "|---|---|---|---|"]
    for step in STEPS:
        def cell(tr):
            if (tr, step) not in agg: return "NA"
            return f"{statistics.mean(agg[(tr,step)]):.3f} ({statistics.mean(zagg[(tr,step)]):+.1f})"
        lines.append(f"| {step} | {cell('sft_fp32_BA')} | {cell('sft_bf16_BA')} | {cell('opd_top32_approx')} |")
    (OUT / "T4_rho_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[T4] wrote T4_rho_dualtrack.csv ({len(rows)}) + T4_rho_summary.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
