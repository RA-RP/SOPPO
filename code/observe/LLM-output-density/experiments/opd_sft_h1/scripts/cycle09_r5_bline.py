#!/usr/bin/env python3
"""Cycle 09 Round 5 — line B (process line): dip as a compression-path fork.

B1  SFT training-domain self-generation (same 32 math prompts as legacy_S_math,
    so the arm contrast is not confounded by prompt pool).
B2  Cross matrix: weights of checkpoint i x text of checkpoint j, full 7x7,
    both arms, layers {9,18,27}, 7 modules, 3 generation seeds.
B3  Mismatch(t) = r_eps(W_t, X_t) - r_eps(W_t, X_0)   (diagonal - first column).
B4  Compare against the pre-registered H-mismatch predictions (no adjudication).

Windowing/normalization/whitening are inherited verbatim from the R4 v2 campaign.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cycle09_r4_common as c4
import cycle09_r4_campaign as camp
import cycle09_r5_common as c5

# Pre-registered predictions (B4) — written before the run, must not be edited.
PREREG = {
    "H-mismatch": {
        "1": "SFT Mismatch(t) peaks at step 20 (its ID dip step) and falls back on recover",
        "2": "OPD Mismatch(t) is flat throughout and significantly below the SFT peak",
        "branch_both": "dip mechanism (distribution mismatch / compression-path fork) holds",
        "branch_only_1": "mechanism holds for SFT; OPD needs off-KD split",
        "branch_not_1": "falsified: Mismatch unrelated to dip; fall back to correlational narrative",
    }
}


# ---------------------------------------------------------------- B1 generation
def generate_sft_selfgen(args) -> None:
    from vllm import LLM, SamplingParams

    banks = c4.prompt_banks(args.n_samples)
    math_bank = banks["math"]

    pending = []
    for step in [s for s in args.steps if int(s) != 0]:
        for seed in args.generation_seeds:
            target = c5.x_corpus_path("sft", step, seed)
            if camp.complete_corpus(target, args.n_samples):
                continue
            pending.append((target, step, seed))
    if not pending:
        print("[B1] all SFT self-generation corpora complete", flush=True)
        return

    by_step: dict[int, list] = {}
    for target, step, seed in pending:
        by_step.setdefault(int(step), []).append((target, seed))

    tokenizer = None
    for step, jobs in sorted(by_step.items()):
        model_dir = c4.model_path("sft", step)
        print(f"[B1] loading SFT {c4.step_label(step)} for {len(jobs)} seed batch(es)", flush=True)
        llm = LLM(
            model=str(model_dir),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=False,
            seed=c4.WINDOW_SEED,
        )
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
        try:
            for target, seed in jobs:
                prompts, params = [], []
                for item in math_bank:
                    prompts.append(
                        camp.formatted_prompt(
                            tokenizer, item["prompt"], item["instruction"], "math"
                        )
                    )
                    params.append(
                        SamplingParams(
                            temperature=c4.TEMPERATURE,
                            top_p=c4.TOP_P,
                            max_tokens=c4.MAX_NEW_TOKENS,
                            seed=c4.stable_seed(seed, "X", "math", item["sample_id"]),
                        )
                    )
                outputs = llm.generate(prompts, params)
                rows = []
                for item, formatted, output in zip(math_bank, prompts, outputs):
                    text = output.outputs[0].text
                    prompt_ids = list(output.prompt_token_ids)
                    gen_ids = list(output.outputs[0].token_ids)
                    rows.append(
                        {
                            "sample_id": item["sample_id"],
                            "probe_type": "X",
                            "domain": "math",
                            "source_kind": "sft_selfgen_training_domain",
                            "arm": "sft",
                            "step": int(step),
                            "generation_seed": int(seed),
                            "per_request_seed": c4.stable_seed(
                                seed, "X", "math", item["sample_id"]
                            ),
                            "prompt_text": item["prompt"],
                            "formatted_prompt": formatted,
                            "generation_text": text,
                            "prompt_token_ids": prompt_ids,
                            "generation_token_ids": gen_ids,
                            "full_token_ids": prompt_ids + gen_ids,
                            "eligible_start": len(prompt_ids),
                            "eligible_end": len(prompt_ids) + len(gen_ids),
                            "finish_reason": output.outputs[0].finish_reason,
                            "generation_config": {
                                "temperature": c4.TEMPERATURE,
                                "top_p": c4.TOP_P,
                                "max_new_tokens": c4.MAX_NEW_TOKENS,
                                "prompt_pool": "legacy_S_math questions (same 32 as static reference)",
                            },
                        }
                    )
                c4.write_jsonl_atomic(target, rows)
                print(f"[B1] {target} n={len(rows)}", flush=True)
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "task": "B1 SFT training-domain self-generation",
        "prompt_pool": "legacy_S_math questions (identical to the static X_SFT reference)",
        "prompt_pool_shared_with_opd_rollouts": True,
        "n_samples": args.n_samples,
        "steps": [int(s) for s in args.steps if int(s) != 0],
        "generation_seeds": list(args.generation_seeds),
        "temperature": c4.TEMPERATURE,
        "top_p": c4.TOP_P,
        "max_new_tokens": c4.MAX_NEW_TOKENS,
        "x0_column": "base math rollouts (R4 X/opd/step_000), shared by both arms",
    }
    c4.write_json_atomic(args.mini_root / "R5_selfgen_manifest.json", manifest)


# ------------------------------------------------------------- B2 cross matrix
def cell_rows(
    *,
    weight_arm: str,
    weight_step: int,
    text_arm: str,
    text_step: int,
    seed: int,
    model,
    profile,
    layers: list[int],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scales = camp.scaling_by_group(profile, layers, device)
    spectra_rows: list[dict[str, Any]] = []
    sigma_store: dict[str, dict[str, list[float]]] = {}
    try:
        for layer in layers:
            layer_key = f"layer_{layer}"
            sigma_store[layer_key] = {}
            for module in c4.MODULES:
                group = c4.MODULE_TO_GROUP[module]
                weight = camp.module_at(model, layer, module).weight.detach().to(
                    device=device, dtype=torch.float32
                )
                sigma = torch.linalg.svdvals(weight @ scales[layer][group])
                sigma_list = sigma.cpu().tolist()
                sigma_store[layer_key][module] = sigma_list

                row: dict[str, Any] = {
                    "weight_arm": weight_arm,
                    "weight_step": int(weight_step),
                    "text_arm": text_arm,
                    "text_step": int(text_step),
                    "generation_seed": int(seed),
                    "cell_kind": (
                        "diagonal"
                        if int(weight_step) == int(text_step)
                        else "static_x0"
                        if int(text_step) == 0
                        else "off_diagonal"
                    ),
                    "layer": layer,
                    "module": module,
                    "track": "per_checkpoint",
                    "n_samples": profile["n_samples"],
                    "effective_rank": c4.effective_rank(sigma_list),
                }
                for epsilon in c5.EPSILONS:
                    rank = c4.functional_rank(sigma_list, epsilon)
                    tag = f"{epsilon:.2f}".split(".")[1]
                    row[f"r_eps_{tag}"] = rank
                    row[f"gamma_r_eps_{tag}"] = c5.spectral_gap(sigma_list, rank)
                for r in (32, 64, 128, 256):
                    row[f"tail_energy_r{r}"] = c4.tail_energy(sigma_list, r)
                row[f"gamma_k{c5.FIXED_RANK_CONTROL}"] = c5.spectral_gap(
                    sigma_list, c5.FIXED_RANK_CONTROL
                )
                spectra_rows.append(row)
                del weight, sigma
        # A6 support: raw (unwhitened) residual-stream covariance, per checkpoint.
        raw_rows = []
        for layer in layers:
            second = profile["residual_second"][layer].to(device=device, dtype=torch.float32)
            mean = profile["residual_mean"][layer].to(device=device, dtype=torch.float32)
            covariance = second - torch.outer(mean, mean)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).cpu().numpy()[::-1]
            raw_rows.append(
                {
                    "weight_arm": weight_arm,
                    "weight_step": int(weight_step),
                    "text_arm": text_arm,
                    "text_step": int(text_step),
                    "generation_seed": int(seed),
                    "layer": layer,
                    "n_samples": profile["n_samples"],
                    "raw_er_unnormalized": c4.effective_rank(eigenvalues),
                    "raw_er_normalized": c5.normalized_effective_rank(eigenvalues),
                    "raw_top5_eigen_share": c5.top_eigen_share(eigenvalues, 5),
                    "raw_dim": int(eigenvalues.size),
                    "raw_trace": float(eigenvalues.sum()),
                }
            )
            del second, mean, covariance
    finally:
        scales.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return spectra_rows, raw_rows, sigma_store


def cross_matrix(args, tokenizer) -> None:
    layers = list(args.layers)
    for weight_arm in args.arms:
        for weight_step in args.steps:
            todo = []
            for text_step in args.steps:
                for seed in args.generation_seeds:
                    task_id = c5.cell_task_id(weight_arm, text_step, seed)
                    target = c5.measurement_path(weight_arm, weight_step, task_id)
                    if target.exists() and not args.force:
                        continue
                    todo.append((text_step, seed, task_id, target))
            if not todo:
                print(f"[B2 skip] {weight_arm}/{c4.step_label(weight_step)}", flush=True)
                continue

            model = camp.load_model(c4.model_path(weight_arm, weight_step), args.device)
            print(
                f"[B2] weights={weight_arm}/{c4.step_label(weight_step)} cells={len(todo)}",
                flush=True,
            )
            try:
                for text_step, seed, task_id, target in todo:
                    corpus = c5.x_corpus_path(weight_arm, text_step, seed)
                    if not corpus.exists():
                        raise FileNotFoundError(f"missing cross-matrix corpus: {corpus}")
                    samples = c4.prepare_samples(
                        corpus,
                        tokenizer,
                        corpus_id=task_id,
                        window_seed=c4.WINDOW_SEED,
                        max_context_tokens=args.max_context_tokens,
                    )
                    if args.measurement_n > 0:
                        samples = samples[: args.measurement_n]
                    keep = c5.keep_gram(weight_step, text_step)
                    profile = camp.collect_profile(
                        model,
                        samples,
                        layers,
                        args.device,
                        keep_factors=False,
                        keep_residual_samples=False,
                    )
                    spectra_rows, raw_rows, sigma_store = cell_rows(
                        weight_arm=weight_arm,
                        weight_step=weight_step,
                        text_arm="base" if int(text_step) == 0 else weight_arm,
                        text_step=text_step,
                        seed=seed,
                        model=model,
                        profile=profile,
                        layers=layers,
                        device=args.device,
                    )
                    c4.write_json_atomic(
                        target,
                        {
                            "schema_version": 1,
                            "weight_arm": weight_arm,
                            "weight_step": int(weight_step),
                            "text_step": int(text_step),
                            "generation_seed": int(seed),
                            "task_id": task_id,
                            "corpus_path": str(corpus),
                            "layers": layers,
                            "n_samples": profile["n_samples"],
                            "hierarchical_normalization": (
                                "window token mean -> sample window mean -> sample equal mean"
                            ),
                            "spectra_rows": spectra_rows,
                            "raw_rows": raw_rows,
                            "sigma": sigma_store,
                        },
                    )
                    if keep:
                        gpath = c5.gram_path(weight_arm, weight_step, task_id)
                        gpath.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "n_samples": profile["n_samples"],
                                "grams": {
                                    layer: {
                                        group: gram.cpu()
                                        for group, gram in profile["grams"][layer].items()
                                    }
                                    for layer in layers
                                },
                            },
                            gpath,
                        )
                    print(
                        f"[Cell] {weight_arm}/W{weight_step} x X{text_step}/g{seed} "
                        f"n={profile['n_samples']} gram={'kept' if keep else 'no'}",
                        flush=True,
                    )
                    del profile, sigma_store
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                camp.unload_model(model)


# --------------------------------------------------------- B3 / B4 summarize
def summarize(args) -> None:
    import pandas as pd

    spectra, raws = [], []
    for path in sorted(c5.RUN_ROOT.glob("measurements/*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        spectra.extend(payload["spectra_rows"])
        raws.extend(payload["raw_rows"])
    if not spectra:
        raise RuntimeError("no B2 measurements found")

    cross = pd.DataFrame(spectra)
    c4.write_csv_atomic(
        args.mini_root / "R5_cross_matrix.csv",
        cross.to_dict("records"),
        list(cross.columns),
    )
    raw = pd.DataFrame(raws)
    c4.write_csv_atomic(
        args.mini_root / "R5_raw_er_fixed_ckpt.csv",
        raw.to_dict("records"),
        list(raw.columns),
    )

    # B3 Mismatch(t) = r_eps(W_t, X_t) - r_eps(W_t, X_0), per layer/module/seed.
    diag = cross[cross.weight_step == cross.text_step]
    static = cross[cross.text_step == 0]
    keys = ["weight_arm", "weight_step", "generation_seed", "layer", "module"]
    metrics = ["r_eps_05", "r_eps_01", "effective_rank", "tail_energy_r32", "tail_energy_r64"]
    merged = diag.merge(
        static[keys + metrics],
        on=keys,
        suffixes=("_own", "_x0"),
        how="inner",
    )
    for metric in metrics:
        merged[f"mismatch_{metric}"] = merged[f"{metric}_own"] - merged[f"{metric}_x0"]
    mismatch_cols = keys + [f"{m}_own" for m in metrics] + [f"{m}_x0" for m in metrics] + [
        f"mismatch_{m}" for m in metrics
    ]
    mismatch = merged[mismatch_cols]
    c4.write_csv_atomic(
        args.mini_root / "R5_mismatch.csv",
        mismatch.to_dict("records"),
        list(mismatch.columns),
    )

    # B4: report the trajectory against the pre-registered branches (no verdict).
    traj = (
        mismatch.groupby(["weight_arm", "weight_step"])["mismatch_r_eps_05"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mismatch_r_eps_05_mean", "std": "mismatch_r_eps_05_sd"})
    )
    rows = []
    for arm in sorted(traj.weight_arm.unique()):
        sub = traj[traj.weight_arm == arm].sort_values("weight_step")
        series = dict(zip(sub.weight_step, sub.mismatch_r_eps_05_mean))
        nonzero = {s: v for s, v in series.items() if int(s) != 0}
        peak_step = max(nonzero, key=lambda s: abs(nonzero[s])) if nonzero else None
        for _, r in sub.iterrows():
            rows.append(
                {
                    "arm": arm,
                    "step": int(r.weight_step),
                    "mismatch_r_eps_05_mean": float(r.mismatch_r_eps_05_mean),
                    "mismatch_r_eps_05_sd": float(r.mismatch_r_eps_05_sd),
                    "n_cells": int(r["count"]),
                    "peak_step_by_abs": int(peak_step) if peak_step is not None else None,
                    "prereg_sft_peak_step": 20,
                    "prereg_note": (
                        "coder reports readings only; adjudication belongs to Theory"
                    ),
                }
            )
    c4.write_csv_atomic(
        args.mini_root / "R5_b4_prereg_readout.csv", rows, list(rows[0].keys())
    )
    c4.write_json_atomic(args.mini_root / "R5_b4_prereg.json", PREREG)
    print(f"[Summarize] cross={len(cross)} mismatch={len(mismatch)} raw={len(raw)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-root", type=Path, default=c5.RUN_ROOT)
    parser.add_argument("--mini-root", type=Path, default=c5.MINI_ROOT)
    parser.add_argument("--arms", default=",".join(c5.ARMS))
    parser.add_argument("--steps", default=",".join(map(str, c5.STEPS)))
    parser.add_argument("--layers", default=",".join(map(str, c5.LAYERS)))
    parser.add_argument("--generation-seeds", default=",".join(map(str, c5.GENERATION_SEEDS)))
    parser.add_argument("--n-samples", type=int, default=c4.N_GENERATED)
    parser.add_argument("--measurement-n", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=c4.MAX_CONTEXT_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.82)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.arms = tuple(a for a in args.arms.split(",") if a)
    args.steps = tuple(int(s) for s in args.steps.split(",") if s)
    args.layers = tuple(int(s) for s in args.layers.split(",") if s)
    args.generation_seeds = tuple(int(s) for s in args.generation_seeds.split(",") if s)

    if args.smoke:
        args.run_root = args.run_root / "smoke"
        args.mini_root = args.mini_root / "smoke_r5"
        args.steps = (0, 5)
        args.layers = (18,)
        args.generation_seeds = (3,)
        args.n_samples = 4
        args.measurement_n = 4
        args.max_context_tokens = 4096

    c5.RUN_ROOT = args.run_root  # keep path helpers in sync (incl. smoke)
    args.mini_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.generate = args.cross = args.summarize = True

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))

    if args.generate:
        generate_sft_selfgen(args)
    if args.cross:
        cross_matrix(args, tokenizer)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
