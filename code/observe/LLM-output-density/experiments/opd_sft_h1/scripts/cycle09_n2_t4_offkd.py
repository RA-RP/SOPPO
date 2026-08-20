#!/usr/bin/env python3
"""N-2d / task 5: off-KD weight-space rho dual-track + theta_w canonical angles (CPU only).

Math is copied verbatim from cycle09_t4_adapter_rho.py (rho2 / null_z, KS, KMAX, N_DRAWS,
SEED) and cycle09_r3_theta_w.py (angle_summary), so the offkd rows are comparable with the
existing sft/opd tracks cell by cell. dW = scaling * B@A on the off-KD fp32 adapter — never
merged-minus-base. Runs on CPU by design: the GPU belongs to another job.
Rows are appended per (track|arm, step); existing rows are never dropped.
"""
import csv, json, math
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file

torch.set_num_threads(20)
DEV = "cpu"
BASE_NPY = Path("/root/autodl-tmp/cycle08_opd_trajectory/weights/step_000")
OFFKD = Path("/root/autodl-tmp/cycle09_offkd")
MINI = Path("/root/LLM-output-density/mypaper/local_experiment_results/"
            "cycle_09_aaai_competitiveness_completion/run_01/mini")
LAYERS = [9, 18, 27]
MODULES = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
           "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
STEPS = [5, 10, 20, 40, 80, 160, 320, 480, 624]
KS = [32, 128]; KMAX = 256; N_DRAWS = 20; SEED = 0
RANKS = [8, 16, 32, 64, 128]
SCALING = 64/32

def npy(L, mod): return f"model_layers_{L}_{mod}_weight.npy"

def adapter_dir(step):
    d = OFFKD/"checkpoints"/f"checkpoint-{step:06d}"
    if (d/"adapter_model.safetensors").exists(): return d
    for p in sorted((OFFKD/"checkpoint_backfill").glob("*/")):
        c = p/f"checkpoint-{step:06d}"
        if (c/"adapter_model.safetensors").exists(): return c
    raise FileNotFoundError(f"no off-KD adapter for step {step}")

def dW_of(step, L, mod):
    sd = load_file(adapter_dir(step)/"adapter_model.safetensors")
    pre = f"base_model.model.model.layers.{L}.{mod}"
    A = sd[f"{pre}.lora_A.weight"].to(torch.float32)
    B = sd[f"{pre}.lora_B.weight"].to(torch.float32)
    return (SCALING*(B@A)).to(DEV)

def rho2(dW, U, V, dW_sq):
    A = U.T@dW; B = dW@V; C = A@V
    cAu = torch.cumsum((A*A).sum(1),0); cBv = torch.cumsum((B*B).sum(0),0); C2 = C*C
    return {k: (float(cAu[min(k,U.shape[1])-1]/dW_sq), float(cBv[min(k,V.shape[1])-1]/dW_sq),
                float(C2[:min(k,U.shape[1]),:min(k,V.shape[1])].sum()/dW_sq)) for k in KS}

def null_z(dW, U, V, rng):
    S = torch.linalg.svdvals(dW); r = int((S > 1e-6*S[0]).sum().item()); Sr = S[:r]
    m, n = dW.shape; acc = {k: [] for k in KS}
    for _ in range(N_DRAWS):
        Pp = torch.linalg.qr(torch.randn(m, r, generator=rng))[0]
        Qp = torch.linalg.qr(torch.randn(n, r, generator=rng))[0]
        dWr = (Pp*Sr)@Qp.T
        rr = rho2(dWr, U, V, float((dWr*dWr).sum()))
        for k in KS: acc[k].append(rr[k][0])
    return r, {k: (float(np.mean(acc[k])), float(np.std(acc[k]))) for k in KS}

def angle_summary(base, cur, rank):
    cos = torch.linalg.svdvals(base[:, :rank].T @ cur[:, :rank]).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos))
    return float(ang.max()), float(ang.mean()), float(cos.min())

def append_rows(path, new, key_cols, fields):
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            keys = {tuple(str(r[c]) for c in key_cols) for r in new}
            old = [r for r in rd if tuple(str(r[c]) for c in key_cols) not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows(old + new)
    return len(new)

def main():
    print("[N2-T4] base SVD cache (CPU)", flush=True)
    cache = {}
    for L in LAYERS:
        for mod in MODULES:
            W0 = torch.from_numpy(np.load(BASE_NPY/npy(L, mod))).to(torch.float32)
            U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
            cache[(L, mod)] = (U[:, :min(KMAX, U.shape[1])].contiguous(),
                               Vh[:min(KMAX, Vh.shape[0]), :].T.contiguous(), W0, U, Vh.T)
        print(f"[N2-T4]   base SVD layer {L} done", flush=True)

    gen = torch.Generator().manual_seed(SEED)
    rho_rows, th_rows = [], []
    for step in STEPS:
        for L in LAYERS:
            for mod in MODULES:
                Uk, Vk, W0, U0f, V0f = cache[(L, mod)]
                dW = dW_of(step, L, mod)
                for track, d in (("offkd_fp32_BA", dW),
                                 ("offkd_bf16_BA", dW.to(torch.bfloat16).to(torch.float32))):
                    dsq = float((d*d).sum())
                    if dsq < 1e-20: continue
                    real = rho2(d, Uk, Vk, dsq); r, nul = null_z(d, Uk, Vk, gen)
                    for k in KS:
                        ru = real[k][0]; nm, ns = nul[k]
                        rho_rows.append({"track": track, "step": step, "layer": L, "module": mod,
                                         "k": k, "rho2_U": f"{ru:.5f}", "null_mean": f"{nm:.5f}",
                                         "null_std": f"{ns:.5f}",
                                         "z": f"{(ru-nm)/ns:+.2f}" if ns > 0 else "nan",
                                         "dW_rank": r})
                # theta_w: W_t = W0 + dW (fp32 adapter track)
                Ut, St, Vht = torch.linalg.svd(W0 + dW, full_matrices=False)
                for rank in RANKS:
                    umx, umn, ucos = angle_summary(U0f, Ut, rank)
                    vmx, vmn, vcos = angle_summary(V0f, Vht.T, rank)
                    th_rows.append({"arm": "offkd", "step": step, "layer": L, "module": mod,
                                    "rank": rank, "source_kind": "offkd_clean_fp32_ba",
                                    "solver": "full_svd_fp32",
                                    "left_max_angle_deg": f"{umx:.8f}", "left_mean_angle_deg": f"{umn:.8f}",
                                    "left_min_cosine": f"{ucos:.8f}",
                                    "right_max_angle_deg": f"{vmx:.8f}", "right_mean_angle_deg": f"{vmn:.8f}",
                                    "right_min_cosine": f"{vcos:.8f}",
                                    "update_frobenius": f"{float(torch.linalg.matrix_norm(dW)):.8f}"})
        print(f"[N2-T4] step {step} done ({len(rho_rows)} rho / {len(th_rows)} theta rows)", flush=True)

    n1 = append_rows(MINI/"T4_rho_dualtrack.csv", rho_rows, ["track","step","layer","module","k"],
                     ["track","step","layer","module","k","rho2_U","null_mean","null_std","z","dW_rank"])
    n2 = append_rows(MINI/"R3_theta_w.csv", th_rows, ["arm","step","layer","module","rank"],
                     ["arm","step","layer","module","rank","source_kind","solver",
                      "left_max_angle_deg","left_mean_angle_deg","left_min_cosine",
                      "right_max_angle_deg","right_mean_angle_deg","right_min_cosine","update_frobenius"])
    print(f"[N2-T4] appended rho={n1} theta={n2}", flush=True)

if __name__ == "__main__":
    main()
