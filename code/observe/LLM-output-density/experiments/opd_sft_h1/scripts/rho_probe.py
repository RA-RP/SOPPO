#!/usr/bin/env python3
"""Cycle 08 A08 re-do: scale-invariant directional alignment rho^2 (+ spectrum-matched random null).

Replaces the confounded OverlapLift (entry-mask/bf16-magnitude) with a RATIO metric that is scale-invariant:
how much of the update dW's energy falls in the base W0's top-k principal singular subspace.

  rho2_U (k)  = ||U_k^T dW||_F^2      / ||dW||_F^2
  rho2_V (k)  = ||dW V_k||_F^2        / ||dW||_F^2
  rho2_UV(k)  = ||U_k^T dW V_k||_F^2  / ||dW||_F^2

Null = spectrum-matched random rotation: dW=P S Q^T (numeric rank r), draw Haar-random P',Q',
dW_rand=P' S Q'^T (same singular values, random direction). rho2 scale-invariant -> only spectrum/rank match.

Pure linear algebra on already-exported / re-derivable weights. fp32 throughout. No training / no inference.
Outputs geometry/{rho_trajectory.csv, rho_null.csv, rho_summary.md}. Spec: mypaper/result/cycle08_rho_metric_spec.md
"""
from __future__ import annotations
import csv, shutil, sys, math
from pathlib import Path
import numpy as np
import torch

REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SIDE))
from scripts.export_weights import export_model_weights, MODULES  # noqa: E402
from scripts.run_opd_minimal_closure import merge_lora_adapter    # noqa: E402

BASE = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
RUN = Path("/root/autodl-tmp/cycle08_opd_trajectory")
WROOT = RUN / "weights"
BASE_NPY = WROOT / "step_000"
OUT = RUN / "geometry"
LAYERS = [9, 18, 27]
STEPS = [5, 10, 20, 40, 80, 160, 320, 480, 624]          # step_0: dW=0 -> NA, skipped
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_DRAWS = 20
KMAX = max(KS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = {
    "opd": {"kind": "merged", "root": RUN / "_merged_models"},
    "sft": {"kind": "adapter", "root": Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")},
}


def npy_name(L, mod): return f"model_layers_{L}_{mod}_weight.npy"


def rho2_from_proj(dW, U, V, dW_sq):
    """A=U^T dW, B=dW V, C=A V ; cumulative rho2 over KS. U:(m,KMAX) V:(n,KMAX)."""
    A = U.T @ dW              # (KMAX, n)
    B = dW @ V               # (m, KMAX)
    C = A @ V                # (KMAX, KMAX)
    Au = (A * A).sum(dim=1)  # per-row energy -> cumsum over k gives ||U_k^T dW||^2
    Bv = (B * B).sum(dim=0)  # per-col energy
    cAu = torch.cumsum(Au, 0)
    cBv = torch.cumsum(Bv, 0)
    C2 = C * C
    out = {}
    for k in KS:
        kk = min(k, U.shape[1])
        ruv = C2[:kk, :kk].sum()
        out[k] = (float(cAu[kk - 1] / dW_sq), float(cBv[kk - 1] / dW_sq), float(ruv / dW_sq))
    return out


def load_base_svd():
    """SVD(W0) once per (layer,module) -> cached U_kmax, V_kmax (fp32, on DEV)."""
    cache = {}
    for L in LAYERS:
        for mod, _ in MODULES:
            W0 = torch.from_numpy(np.load(BASE_NPY / npy_name(L, mod))).to(torch.float32).to(DEV)
            U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
            kmax = min(KMAX, U.shape[1])
            cache[(L, mod)] = {"U": U[:, :kmax].contiguous(), "V": Vh[:kmax, :].T.contiguous(),
                               "W0": W0}
    return cache


def export_arm_step(arm, step, tmp):
    cfg = ARMS[arm]
    label = f"step_{step:03d}"
    if cfg["kind"] == "merged":
        model_path = cfg["root"] / label
    else:
        merged = tmp / "_merged_sft"
        merge_lora_adapter(BASE, cfg["root"] / label, merged)   # base + LoRA adapter -> HF dir
        model_path = merged
    export_model_weights(str(model_path), tmp, layers=LAYERS, modules=MODULES)
    if cfg["kind"] == "adapter":
        shutil.rmtree(tmp / "_merged_sft", ignore_errors=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(list(BASE_NPY.glob("*.npy"))) < len(LAYERS) * len(MODULES):
        export_model_weights(str(BASE), BASE_NPY, layers=LAYERS, modules=MODULES)
    print(f"[rho] device={DEV}; SVD(W0) cache ...", flush=True)
    cache = load_base_svd()

    # scale-invariance self-check on one cell
    c0 = cache[(LAYERS[0], MODULES[0][0])]
    dwt = torch.randn_like(c0["W0"]) * 0.01
    r1 = rho2_from_proj(dwt, c0["U"], c0["V"], float((dwt * dwt).sum()))[32]
    r10 = rho2_from_proj(dwt * 10, c0["U"], c0["V"], float(((dwt * 10) ** 2).sum()))[32]
    assert all(abs(a - b) < 1e-4 for a, b in zip(r1, r10)), f"scale-invariance FAILED {r1} {r10}"
    print(f"[rho] scale-invariance OK (rho2@k32 {r1[0]:.4f} == {r10[0]:.4f})", flush=True)

    traj_rows, null_rows = [], []
    for arm in ARMS:
        for step in STEPS:
            tmp = WROOT / f"_rho_tmp_{arm}_{step:03d}"
            tmp.mkdir(parents=True, exist_ok=True)
            print(f"[rho] {arm} step {step}: exporting Wp ...", flush=True)
            try:
                export_arm_step(arm, step, tmp)
            except Exception as e:  # noqa: BLE001
                print(f"[rho] {arm} step {step} EXPORT FAIL: {e}", flush=True); shutil.rmtree(tmp, True); continue
            for L in LAYERS:
                for mod, _ in MODULES:
                    c = cache[(L, mod)]
                    Wp = torch.from_numpy(np.load(tmp / npy_name(L, mod))).to(torch.float32).to(DEV)
                    dW = Wp - c["W0"]
                    dW_sq = float((dW * dW).sum())
                    if dW_sq < 1e-12:
                        continue
                    real = rho2_from_proj(dW, c["U"], c["V"], dW_sq)
                    S = torch.linalg.svdvals(dW)
                    r = int((S > 1e-6 * S[0]).sum().item())
                    Sr = S[:r]
                    m, n = dW.shape
                    acc = {k: {"U": [], "V": [], "UV": []} for k in KS}
                    for _ in range(N_DRAWS):
                        Pp = torch.linalg.qr(torch.randn(m, r, device=DEV))[0]
                        Qp = torch.linalg.qr(torch.randn(n, r, device=DEV))[0]
                        dWr = (Pp * Sr) @ Qp.T
                        rr = rho2_from_proj(dWr, c["U"], c["V"], float((dWr * dWr).sum()))
                        for k in KS:
                            acc[k]["U"].append(rr[k][0]); acc[k]["V"].append(rr[k][1]); acc[k]["UV"].append(rr[k][2])
                    for k in KS:
                        ru, rv, ruv = real[k]
                        traj_rows.append([arm, step, L, mod, k, f"{ru:.5f}", f"{rv:.5f}", f"{ruv:.5f}",
                                          f"{math.sqrt(dW_sq):.4f}", r])
                        def ms(x): return (float(np.mean(x)), float(np.std(x)))
                        um, us = ms(acc[k]["U"]); vm, vs = ms(acc[k]["V"]); uvm, uvs = ms(acc[k]["UV"])
                        null_rows.append([arm, step, L, mod, k, f"{um:.5f}", f"{us:.5f}",
                                          f"{vm:.5f}", f"{vs:.5f}", f"{uvm:.5f}", f"{uvs:.5f}", N_DRAWS])
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"[rho] {arm} step {step}: done (dW_rank sample r={r})", flush=True)

    with open(OUT / "rho_trajectory.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "step", "layer", "module", "k",
                                       "rho2_U", "rho2_V", "rho2_UV", "dW_fro", "dW_rank"]); w.writerows(traj_rows)
    with open(OUT / "rho_null.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "step", "layer", "module", "k", "rho2_U_mean", "rho2_U_std",
                                       "rho2_V_mean", "rho2_V_std", "rho2_UV_mean", "rho2_UV_std", "n_draws"]); w.writerows(null_rows)
    print(f"[rho] wrote rho_trajectory.csv ({len(traj_rows)}) + rho_null.csv ({len(null_rows)})", flush=True)
    _summary(traj_rows, null_rows)
    print("[rho] DONE", flush=True)


def _summary(traj_rows, null_rows):
    import collections
    real = {(r[0], int(r[1]), int(r[4])): [] for r in traj_rows}
    for r in traj_rows:
        real[(r[0], int(r[1]), int(r[4]))].append((float(r[5]), float(r[6]), float(r[7])))
    nul = collections.defaultdict(list)
    for r in null_rows:
        nul[(r[0], int(r[1]), int(r[4]))].append((float(r[5]), float(r[6])))  # U mean,std
    ranks = collections.defaultdict(list)
    for r in traj_rows:
        ranks[r[0]].append(int(r[9]))
    lines = ["# Cycle 08 — rho^2 directional alignment (scale-invariant) vs spectrum-matched random null\n",
             f"n_draws={N_DRAWS}, fp32, device={DEV}. rho2_U reported (mean over layer*module). "
             "Verdict = mean z-score over cells: z=(real-null_mean)/null_std; >2 on-principal, <-2 off, else indistinguishable.\n",
             f"Numerical rank of dW (mean over cells): OPD r≈{np.mean(ranks['opd']):.0f}, SFT r≈{np.mean(ranks['sft']):.0f}.\n"]
    for k in (32, 128):
        lines.append(f"\n## k = {k}\n")
        lines.append("| step | OPD ρ²_U | OPD null(m±s) | OPD z→verdict | SFT ρ²_U | SFT null(m±s) | SFT z→verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for step in STEPS:
            cells = {}
            for arm in ("opd", "sft"):
                rv = real.get((arm, step, k), []); nv = nul.get((arm, step, k), [])
                if not rv or not nv:
                    cells[arm] = None; continue
                ru = np.array([x[0] for x in rv]); nm = np.array([x[0] for x in nv]); ns = np.array([x[1] for x in nv])
                z = np.mean((ru - nm) / np.where(ns > 0, ns, np.nan))
                verdict = "on" if z > 2 else ("off" if z < -2 else "indist")
                cells[arm] = (ru.mean(), nm.mean(), ns.mean(), z, verdict)
            def fmt(c): return "NA" if c is None else f"{c[0]:.3f}"
            def fmtn(c): return "NA" if c is None else f"{c[1]:.3f}±{c[2]:.3f}"
            def fmtz(c): return "NA" if c is None else f"{c[3]:+.1f}→{c[4]}"
            o, s = cells["opd"], cells["sft"]
            lines.append(f"| {step} | {fmt(o)} | {fmtn(o)} | {fmtz(o)} | {fmt(s)} | {fmtn(s)} | {fmtz(s)} |")
    (OUT / "rho_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[rho] wrote {OUT/'rho_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
