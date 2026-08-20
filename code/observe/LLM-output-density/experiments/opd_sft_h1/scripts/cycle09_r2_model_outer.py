#!/usr/bin/env python3
"""Cycle09 Round2 compressed runner with model/checkpoint as the outer loop.

This runner keeps the expensive model load at the arm/checkpoint granularity:
  1. load one model
  2. run full36 spectra for S and X_math
  3. run landmark spectra for other X probes on {9,18,27}
  4. run S-side UV on {9,18,27}
  5. run raw residual ER on selected probes/layers
  6. unload model, then move to the next checkpoint

Layer scope is intentionally mixed: full36 only for the core S+X_math branch;
all other branches stay on the three landmark layers unless explicitly widened.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path('/root/LLM-output-density')
SIDE = REPO / 'experiments/opd_sft_h1'
GETSLICE = REPO / 'GetSlice'
for item in (REPO, SIDE, GETSLICE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts import cycle09_r2_raw_er as raw  # noqa: E402
from scripts import cycle09_r2_unified_probe as u  # noqa: E402
from utils.data_utils import get_token_data_from_jsonl  # noqa: E402
from utils.profiling_utils import (  # noqa: E402
    profile_svdllm_low_resource,
    profile_svdllm_single_layer_group,
    whitening,
)

CORE_X_PROBES = ['X_math']
RAW_ER_PROBES = ['X_math', 'X_ood_knowledge']


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(',') if x.strip()]


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def load_model(model_path: Path, seqlen: int):
    from utils.model_utils import get_model_from_huggingface

    model, tokenizer = get_model_from_huggingface(
        model_id=str(model_path),
        torch_dtype='float16',
        trust_remote_code=True,
        cache_dir=None,
    )
    model = model.eval()
    model.seqlen = int(seqlen)
    model.config.use_cache = False
    return model, tokenizer


def s_loader(tokenizer, args: argparse.Namespace):
    return get_token_data_from_jsonl(
        jsonl_path=str(u.S_ROOT / 'math_cot_probe/gamma_s.jsonl'),
        tokenizer=tokenizer,
        nsamples=args.n_samples,
        seqlen=args.seqlen,
        seed=u.SEED,
        batch_size=1,
        cache_file=str(u.cache_path('s', 'math_cot_probe', args.n_samples, args.seqlen)),
        mode='s',
    )


def x_loader(tokenizer, probe: str, path: Path, args: argparse.Namespace):
    return get_token_data_from_jsonl(
        jsonl_path=str(path),
        tokenizer=tokenizer,
        nsamples=args.n_samples,
        seqlen=args.seqlen,
        seed=u.SEED,
        batch_size=1,
        cache_file=str(u.cache_path('x', probe, args.n_samples, args.seqlen)),
        mode='x',
    )


def run_full_s(model, tokenizer, model_path: Path, arm: str, step: int, args: argparse.Namespace) -> None:
    out = u.full_s_json(arm, step)
    if out.exists() and out.stat().st_size > 0:
        print(f'[Skip full36 S] {arm} {u.step_label(step)}', flush=True)
        return
    print(f'[Full36 S] {arm} {u.step_label(step)}', flush=True)
    loader = s_loader(tokenizer, args)
    profile = profile_svdllm_low_resource(
        model_name=str(model_path),
        model=model,
        calib_loader=loader,
        dev='cuda',
        singular_floor=0.0,
        activation_cache_device=args.activation_cache_device,
        cholesky_jitter=0.00001,
    )
    sigma, _ = whitening(
        model_name=str(model_path),
        model=model,
        profiling_mat=profile,
        dev='cuda',
        uv_dtype='float32',
        return_uv=False,
    )
    u.write_json(out, sigma)
    print(f'[Save full36 S] {out}', flush=True)
    del loader, profile, sigma
    cleanup_cuda()


def run_full_x_math(model, tokenizer, model_path: Path, arm: str, step: int, probes: dict[str, Path], args: argparse.Namespace) -> None:
    if 'X_math' not in probes:
        print('[Warn] X_math probe missing; skip full36 X_math', flush=True)
        return
    out = u.full_x_json(arm, step, 'X_math')
    if out.exists() and out.stat().st_size > 0:
        print(f'[Skip full36 X_math] {arm} {u.step_label(step)}', flush=True)
        return
    print(f'[Full36 X_math] {arm} {u.step_label(step)}', flush=True)
    loader = x_loader(tokenizer, 'X_math', probes['X_math'], args)
    profile = profile_svdllm_low_resource(
        model_name=str(model_path),
        model=model,
        calib_loader=loader,
        dev='cuda',
        singular_floor=0.0,
        activation_cache_device=args.activation_cache_device,
        cholesky_jitter=0.00001,
    )
    sigma, _ = whitening(
        model_name=str(model_path),
        model=model,
        profiling_mat=profile,
        dev='cuda',
        uv_dtype='float32',
        return_uv=False,
    )
    u.write_json(out, sigma)
    print(f'[Save full36 X_math] {out}', flush=True)
    del loader, profile, sigma
    cleanup_cuda()


def run_landmark_x(model, tokenizer, model_path: Path, arm: str, step: int, probes: dict[str, Path], args: argparse.Namespace) -> None:
    # X_math landmark values are read from its full36 output. Only non-core probes need standalone landmark JSON.
    non_core = {name: path for name, path in probes.items() if name not in CORE_X_PROBES}
    for layer in args.layers:
        pending = {
            name: path
            for name, path in non_core.items()
            if not u.landmark_output_available(arm, step, layer, name)
        }
        if not pending:
            print(f'[Skip landmark X] {arm} {u.step_label(step)} L{layer}', flush=True)
            continue
        print(f'[Landmark X] {arm} {u.step_label(step)} L{layer} probes={list(pending)}', flush=True)
        loaders = {name: x_loader(tokenizer, name, path, args) for name, path in pending.items()}
        profiles = profile_svdllm_single_layer_group(
            model_name=str(model_path),
            model=model,
            calib_loaders_by_task=loaders,
            dev='cuda',
            target_layer=layer,
            layer_gpu_chunk_size=args.layer_gpu_chunk_size,
            singular_floor=0.0,
            activation_cache_device=args.activation_cache_device,
            cholesky_jitter=0.00001,
        )
        for probe, profile in profiles.items():
            sigma, _ = whitening(
                model_name=str(model_path),
                model=model,
                profiling_mat=profile,
                dev='cuda',
                uv_dtype='float32',
                return_uv=False,
            )
            out = u.landmark_x_json(arm, step, layer, probe)
            u.write_json(out, sigma)
            print(f'[Save landmark X] {out}', flush=True)
            del sigma
        del loaders, profiles
        cleanup_cuda()


def run_uv_s(model, tokenizer, model_path: Path, arm: str, step: int, args: argparse.Namespace) -> None:
    pending_layers = [
        layer for layer in args.layers
        if not (u.uv_s_json(arm, step, layer).exists() and u.uv_s_pt(arm, step, layer).exists())
    ]
    if not pending_layers:
        print(f'[Skip UV S] {arm} {u.step_label(step)} all landmark layers', flush=True)
        return
    loader = s_loader(tokenizer, args)
    for layer in pending_layers:
        print(f'[UV S] {arm} {u.step_label(step)} L{layer} top-{args.uv_rank}', flush=True)
        profiles = profile_svdllm_single_layer_group(
            model_name=str(model_path),
            model=model,
            calib_loaders_by_task={'math_cot_probe': loader},
            dev='cuda',
            target_layer=layer,
            layer_gpu_chunk_size=args.layer_gpu_chunk_size,
            singular_floor=0.0,
            activation_cache_device=args.activation_cache_device,
            cholesky_jitter=0.00001,
        )
        sigma, uv = whitening(
            model_name=str(model_path),
            model=model,
            profiling_mat=profiles['math_cot_probe'],
            dev='cuda',
            uv_dtype='float32',
            return_uv=True,
        )
        u.write_json(u.uv_s_json(arm, step, layer), sigma)
        u.ensure_dir(u.uv_s_pt(arm, step, layer).parent)
        import torch
        torch.save(u.top_rank_uv(uv, args.uv_rank), u.uv_s_pt(arm, step, layer))
        print(f'[Save UV S] {u.uv_s_pt(arm, step, layer)}', flush=True)
        del profiles, sigma, uv
        cleanup_cuda()
    del loader
    cleanup_cuda()


def run_raw_er(model, tokenizer, model_path: Path, arm: str, step: int, probes: dict[str, Path], args: argparse.Namespace) -> None:
    if args.skip_raw_er:
        return
    wanted = parse_str_list(args.raw_probes, RAW_ER_PROBES)
    pending = {
        name: probes[name]
        for name in wanted
        if name in probes and not raw.raw_er_json(arm, step, name).exists()
    }
    if not pending:
        print(f'[Skip rawER] {arm} {u.step_label(step)}', flush=True)
        return
    print(f'[rawER] move full model to cuda for {arm} {u.step_label(step)} probes={list(pending)}', flush=True)
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    model.config.use_cache = False
    for probe, path in pending.items():
        result = raw.compute_raw_er_loaded(
            model=model,
            tokenizer=tokenizer,
            dev=dev,
            probe=probe,
            probe_path=path,
            layers=args.layers,
            nsamples=args.raw_n_samples,
            seqlen=args.seqlen,
        )
        result.update({'arm': arm, 'step': step, 'model': str(model_path)})
        raw.write_json(raw.raw_er_json(arm, step, probe), result)
        print(f'[Save rawER] {raw.raw_er_json(arm, step, probe)}', flush=True)
    model.to('cpu')
    cleanup_cuda()


def run_one_model(arm: str, step: int, probes: dict[str, Path], args: argparse.Namespace) -> None:
    model_path = u.model_path_for(arm, step)
    if not (model_path / 'config.json').exists():
        raise FileNotFoundError(f'model missing: {model_path}')
    print('\n' + '=' * 72, flush=True)
    print(f'[MODEL] {arm} {u.step_label(step)} -> {model_path}', flush=True)
    print('=' * 72, flush=True)
    model, tokenizer = load_model(model_path, args.seqlen)
    try:
        run_full_s(model, tokenizer, model_path, arm, step, args)
        run_full_x_math(model, tokenizer, model_path, arm, step, probes, args)
        run_landmark_x(model, tokenizer, model_path, arm, step, probes, args)
        run_uv_s(model, tokenizer, model_path, arm, step, args)
        run_raw_er(model, tokenizer, model_path, arm, step, probes, args)
    finally:
        del model, tokenizer
        cleanup_cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=u.DEFAULT_RUN)
    parser.add_argument('--mini-root', type=Path, default=u.DEFAULT_MINI)
    parser.add_argument('--arms', default='opd,sft')
    parser.add_argument('--steps', default=','.join(map(str, u.KEY_STEPS)))
    parser.add_argument('--layers', default=','.join(map(str, u.LANDMARK_LAYERS)))
    parser.add_argument('--probes', default=','.join(u.LANDMARK_X_PROBES))
    parser.add_argument('--raw-probes', default=','.join(RAW_ER_PROBES))
    parser.add_argument('--n-samples', type=int, default=u.DEFAULT_N)
    parser.add_argument('--raw-n-samples', type=int, default=None)
    parser.add_argument('--seqlen', type=int, default=u.DEFAULT_SEQLEN)
    parser.add_argument('--uv-rank', type=int, default=128)
    parser.add_argument('--layer-gpu-chunk-size', type=int, default=12)
    parser.add_argument('--activation-cache-device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--skip-raw-er', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    args.arms = parse_str_list(args.arms, ['opd', 'sft'])
    args.steps = parse_int_list(args.steps, u.KEY_STEPS)
    args.layers = parse_int_list(args.layers, u.LANDMARK_LAYERS)
    if args.raw_n_samples is None:
        args.raw_n_samples = args.n_samples

    u.configure_roots(args.run_root, args.mini_root)
    raw.configure_roots(args.run_root, args.mini_root)
    u.ensure_dir(u.GS)
    u.ensure_dir(u.MINI)
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')
    os.environ.setdefault('no_proxy', '127.0.0.1,localhost')

    probes = u.select_probes(parse_str_list(args.probes, u.LANDMARK_X_PROBES))
    if args.check_inputs:
        u.check_inputs(args)
    print('[Plan] model-outer compressed Round2', flush=True)
    print(f'  arms={args.arms}', flush=True)
    print(f'  steps={args.steps}', flush=True)
    print('  full36=S + X_math', flush=True)
    print(f'  landmark_layers={args.layers}', flush=True)
    print(f'  landmark_x={list(probes)} (X_math served by full36)', flush=True)
    print(f'  raw_er={parse_str_list(args.raw_probes, RAW_ER_PROBES)} skip={args.skip_raw_er}', flush=True)
    if args.dry_run:
        return

    for arm in args.arms:
        for step in args.steps:
            run_one_model(arm, step, probes, args)


if __name__ == '__main__':
    main()
