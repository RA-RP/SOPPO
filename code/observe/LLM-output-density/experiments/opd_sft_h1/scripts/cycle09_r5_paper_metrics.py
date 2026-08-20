#!/usr/bin/env python3
"""Cycle 09 Round 5 — 2605.30524 §3.1 metric suite, implemented verbatim, run on Qwen3-4B.

Paper's definitions (Representation Collapse in Sequential Post-Training, §3.1):

    H = H_l^t(X_probe) in R^{n x d}   (token hidden-state matrix, layer l, checkpoint t)
    H_bar = H - 1 mu^T                (centering over tokens)
    Sigma = (n-1)^{-1} H_bar^T H_bar  (covariance)
    p_i   = lam_i / sum_j lam_j       (eigenvalues of Sigma)
    erank(Sigma) = d^{-1} exp( - sum_i p_i log(p_i + eps) )     <- normalized effective rank
    PR(Sigma)    = (sum_i lam_i)^2 / sum_i lam_i^2              <- participation ratio
    top-k variance for k in {1, 8, 32}
    anisotropy   = average pairwise cosine, before and after centering
    CKA(H_s,H_t) = ||H_s^T H_t||_F^2 / (||H_s^T H_s||_F ||H_t^T H_t||_F)

REPRODUCTION RECORD (2026-07-14, before running this on our checkpoints):
Their Fig 2 reports base-checkpoint normalized erank ~0.60-0.75. We could NOT reproduce
that level from the stated definition — not even on their own primary model
(Qwen2.5-1.5B, 28 layers, d=1536), using their verbatim formula and four reasonable
preprocessing conventions:

    convention                          Qwen2.5-1.5B L14      Qwen3-4B L18
    covariance (verbatim §3.1)              0.0007               0.0006
    correlation (per-dim standardized)      0.0948               0.2205
    per-token RMS-norm + covariance         0.1343               0.1261
    per-token RMS-norm + correlation        0.2853               0.2310
    paper's reported range                  0.60 - 0.75            --

Meanwhile our anisotropy reproduction *does* agree with theirs (0.12-0.20 vs their
0.08-0.15) on the same hidden states. We therefore run their metric verbatim (the
`covariance` convention is primary, matching §3.1 literally) and report all four
conventions per cell, so trajectories can be compared in shape even though the
absolute level of their erank could not be reproduced. No convention was chosen to
make our numbers match theirs.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_r4_common as c4
import cycle09_r4_campaign as camp
import cycle09_r5_common as c5

PROBES = {
    "S_math": "corpora/generated/S/math/gen_seed_3.jsonl",
    "E_ood": "corpora/fixed/E_ood.jsonl",
    "E_general": "corpora/fixed/E_general.jsonl",
    "legacy_S_math": "corpora/fixed/legacy_S_math.jsonl",
}
EPS = 1e-12


def _erank(lam: torch.Tensor, d: int) -> float:
    p = lam / lam.sum().clamp_min(EPS)
    entropy = -(p * torch.log(p + EPS)).sum()
    return float(torch.exp(entropy) / d)


def _spectrum(M: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of (n-1)^-1 M^T M, via the d x d gram (cost independent of n)."""
    n = M.shape[0]
    cov = (M.T @ M) / max(n - 1, 1)
    return torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)


def paper_metrics(H: torch.Tensor) -> dict[str, float]:
    """All four conventions + PR + top-k variance + anisotropy, from one token matrix."""
    H = H.double()
    n, d = H.shape
    out: dict[str, float] = {"n_tokens": n, "hidden_dim": d}

    centered = H - H.mean(0, keepdim=True)
    lam = _spectrum(centered)
    p = lam / lam.sum().clamp_min(EPS)
    out["erank_covariance_verbatim"] = _erank(lam, d)
    out["participation_ratio"] = float(lam.sum().square() / lam.square().sum().clamp_min(EPS))
    out["pr_normalized"] = out["participation_ratio"] / d
    for k in (1, 8, 32):
        out[f"top{k}_variance_share"] = float(p[:k].sum())

    standardized = centered / centered.std(0, keepdim=True).clamp_min(1e-8)
    out["erank_correlation"] = _erank(_spectrum(standardized), d)

    rms = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-6) * np.sqrt(d)
    rms_centered = rms - rms.mean(0, keepdim=True)
    out["erank_rmsnorm_covariance"] = _erank(_spectrum(rms_centered), d)

    rms_std = rms_centered / rms_centered.std(0, keepdim=True).clamp_min(1e-8)
    out["erank_rmsnorm_correlation"] = _erank(_spectrum(rms_std), d)

    # anisotropy: average pairwise cosine, before and after centering (paper §3.1)
    sample = H if n <= 2048 else H[torch.randperm(n)[:2048]]
    for tag, matrix in (("raw", sample), ("centered", sample - H.mean(0, keepdim=True))):
        unit = matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        gram = unit @ unit.T
        m = gram.shape[0]
        off = (gram.sum() - gram.diagonal().sum()) / (m * (m - 1))
        out[f"anisotropy_{tag}"] = float(off)
    return out


@torch.no_grad()
def run(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    corpora = {
        name: c4.read_jsonl(c5.R4_ROOT / path)[: args.n_samples]
        for name, path in PROBES.items()
    }
    rows: list[dict[str, Any]] = []
    base_hidden: dict[tuple[str, int], torch.Tensor] = {}

    for arm in args.arms:
        for step in args.steps:
            if step == 0 and arm == "sft":
                continue  # step 0 is the shared base
            model = camp.load_model(c4.model_path(arm, step), args.device)
            print(f"[paper-metrics] {arm}/{c4.step_label(step)}", flush=True)
            try:
                for probe, samples in corpora.items():
                    per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in args.layers}
                    for row in samples:
                        ids = torch.tensor(
                            row["full_token_ids"][: args.max_tokens],
                            device=args.device,
                        ).unsqueeze(0)
                        out = model(
                            input_ids=ids, output_hidden_states=True, use_cache=False
                        )
                        for layer in args.layers:
                            per_layer[layer].append(
                                out.hidden_states[layer + 1][0].float()
                            )
                        del out
                    for layer in args.layers:
                        H = torch.cat(per_layer[layer])
                        metrics = paper_metrics(H)
                        key = (probe, layer)
                        if arm == "opd" and step == 0:
                            base_hidden[key] = H.double().cpu()
                        cka = float("nan")
                        if key in base_hidden:
                            B = base_hidden[key].to(args.device)
                            width = min(B.shape[0], H.shape[0])
                            Hs, Ht = B[:width].double(), H[:width].double()
                            num = torch.linalg.matrix_norm(Hs.T @ Ht) ** 2
                            den = (
                                torch.linalg.matrix_norm(Hs.T @ Hs)
                                * torch.linalg.matrix_norm(Ht.T @ Ht)
                            ).clamp_min(EPS)
                            cka = float(num / den)
                            del B, Hs, Ht
                        rows.append(
                            {
                                "arm": arm,
                                "step": int(step),
                                "probe": probe,
                                "layer": layer,
                                "cka_vs_base": cka,
                                **metrics,
                                "primary_convention": "erank_covariance_verbatim",
                                "reproduction_note": (
                                    "paper's 0.60-0.75 level NOT reproducible from its own "
                                    "definition, not even on its own model (Qwen2.5-1.5B): "
                                    "see module docstring"
                                ),
                            }
                        )
                        del H, per_layer[layer]
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                camp.unload_model(model)

    c4.write_csv_atomic(
        args.mini_root / "R5_paper_metrics_qwen3.csv", rows, list(rows[0].keys())
    )
    print(f"[paper-metrics] rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-root", type=Path, default=c5.MINI_ROOT)
    parser.add_argument("--arms", default=",".join(c5.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, c5.STEPS)))
    parser.add_argument("--layers", default=",".join(map(str, c5.LAYERS)))
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.arms = tuple(a for a in args.arms.split(",") if a)
    args.steps = tuple(int(s) for s in args.steps.split(",") if s)
    args.layers = tuple(int(s) for s in args.layers.split(",") if s)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
