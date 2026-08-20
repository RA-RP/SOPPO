#!/usr/bin/env python3
"""Cycle 09 mini Round 2 raw residual-stream ER runner.

Default raw-ER scope is deliberately narrow for the compressed Round2 budget:
  * checkpoints: 0,5,10,20,40,160,624
  * layers: 9,18,27
  * probes: X_math and X_ood_knowledge

Use --probes/--layers to broaden this later. The runner loads each model once
per arm/checkpoint and evaluates all requested probes before unloading it.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path('/root/LLM-output-density')
SIDE = REPO / 'experiments/opd_sft_h1'
DEFAULT_RUN = Path('/root/autodl-tmp/cycle09_r2')
DEFAULT_MINI = REPO / 'mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini'
BASE_MODEL = Path('/root/autodl-tmp/model/Qwen/Qwen3-4B-Base')
OPD_MERGED = Path('/root/autodl-tmp/cycle08_opd_trajectory/_merged_models')
SFT_CKPT = Path('/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints')
SFT_EXISTING_MERGED = Path('/root/autodl-tmp/cycle07_base_sft_trajectory/_merged_tmp')
X_MATH = Path('/root/autodl-tmp/cycle07_base_sft_trajectory/getslice/inputs/X_base/x_probe.jsonl')
X_BOS_CANDIDATES = [
    Path('/root/autodl-tmp/exp0609/opd_minimal_03_v2/getslice/inputs/X_bos/x_probe.jsonl'),
    Path('/root/autodl-tmp/cycle04_opd_stability_gain/getslice/inputs/X_bos/x_probe.jsonl'),
    Path('/root/autodl-tmp/cycle04_smoke/getslice/inputs/X_bos/x_probe.jsonl'),
]
KEY_STEPS = [0, 5, 10, 20, 40, 160, 624]
LANDMARK_LAYERS = [9, 18, 27]
DEFAULT_PROBES = ['X_math', 'X_ood_knowledge']
SEED = 3

RUN = DEFAULT_RUN
GS = RUN / 'getslice'
MINI = DEFAULT_MINI

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SIDE) not in sys.path:
    sys.path.insert(0, str(SIDE))
from opd_sft_h1.geometry_metrics import effective_rank  # noqa: E402


def configure_roots(run_root: Path | None, mini_root: Path | None) -> None:
    global RUN, GS, MINI
    RUN = run_root or DEFAULT_RUN
    GS = RUN / 'getslice'
    MINI = mini_root or DEFAULT_MINI


def step_label(step: int) -> str:
    return f'step_{int(step):03d}'


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(',') if x.strip()]


def battery_input(name: str) -> Path:
    local = GS / 'inputs' / name / 'x_probe.jsonl'
    if local.exists():
        return local
    return DEFAULT_RUN / 'getslice' / 'inputs' / name / 'x_probe.jsonl'


def probe_paths() -> dict[str, Path]:
    paths = {
        'X_math': X_MATH,
        'X_ood_knowledge': battery_input('X_ood_knowledge'),
        'X_general': battery_input('X_general'),
        'X_math_hard': battery_input('X_math_hard'),
    }
    for candidate in X_BOS_CANDIDATES:
        if candidate.exists():
            paths['X_bos'] = candidate
            break
    return {name: path for name, path in paths.items() if path.exists()}


def model_path_for(arm: str, step: int) -> Path:
    if step == 0:
        return BASE_MODEL
    if arm == 'opd':
        return OPD_MERGED / step_label(step)
    if arm == 'sft':
        existing = SFT_EXISTING_MERGED / step_label(step)
        if (existing / 'config.json').exists():
            return existing
        merged = RUN / 'sft_merged' / step_label(step)
        if (merged / 'config.json').exists():
            return merged
        from scripts.run_opd_minimal_closure import merge_lora_adapter
        merge_lora_adapter(BASE_MODEL, SFT_CKPT / step_label(step), merged)
        return merged
    raise ValueError(f'unknown arm: {arm}')


def maybe_drop_sft(step: int, drop: bool) -> None:
    if not drop or step == 0:
        return
    path = RUN / 'sft_merged' / step_label(step)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f'[Clean] removed {path}', flush=True)


def raw_er_json(arm: str, step: int, probe: str) -> Path:
    return RUN / 'raw_er' / arm / step_label(step) / f'{probe}.json'


def load_x_texts(path: Path) -> list[str]:
    texts = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            output = obj.get('output')
            if not isinstance(output, dict) or 'text' not in output:
                raise ValueError(f'{path}:{line_no} missing output.text')
            text = str(output.get('text', '')).strip()
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f'No usable text in {path}')
    return texts


def build_batches(tokenizer, texts: list[str], nsamples: int, seqlen: int, seed: int):
    import torch

    total = '\n\n'.join(texts)
    if len(total) <= seqlen + 1:
        raise ValueError(f'Insufficient text length for seqlen={seqlen}')
    rng = random.Random(seed)
    batches = []
    attempts = 0
    while len(batches) < nsamples:
        attempts += 1
        if attempts > nsamples * 50:
            raise RuntimeError('failed to build enough raw ER windows')
        start = rng.randint(0, len(total) - seqlen - 1)
        enc = tokenizer(total[start:start + seqlen * 10], return_tensors='pt')
        if enc.input_ids.shape[1] < seqlen:
            continue
        ids = enc.input_ids[:, :seqlen].to(dtype=torch.long)
        batches.append({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
    return batches


def load_model(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if dev.type == 'cuda' else torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to(dev).eval()
    model.config.use_cache = False
    return model, tokenizer, dev


def compute_raw_er_loaded(model, tokenizer, dev, probe: str, probe_path: Path, layers: list[int], nsamples: int, seqlen: int) -> dict[str, Any]:
    import torch

    batches = build_batches(tokenizer, load_x_texts(probe_path), nsamples, seqlen, SEED)
    hidden = int(model.config.hidden_size)
    stats = {
        layer: {
            'n': 0,
            'sum': torch.zeros(hidden, dtype=torch.float64),
            'gram': torch.zeros((hidden, hidden), dtype=torch.float64),
        }
        for layer in layers
    }

    with torch.no_grad():
        for idx, batch in enumerate(batches, start=1):
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            for layer in layers:
                h = out.hidden_states[layer + 1].detach().float().reshape(-1, hidden).cpu().to(torch.float64)
                stats[layer]['n'] += h.shape[0]
                stats[layer]['sum'] += h.sum(0)
                stats[layer]['gram'] += h.T @ h
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f'[rawER] {probe} batch {idx}/{len(batches)}', flush=True)

    eig_device = dev if dev.type == 'cuda' else torch.device('cpu')
    layer_rows = {}
    for layer in layers:
        n = max(int(stats[layer]['n']), 1)
        mean = stats[layer]['sum'] / n
        cov = stats[layer]['gram'] / n - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        cov = cov / float(hidden)
        eig = torch.linalg.eigvalsh(cov.to(eig_device)).detach().cpu().clamp_min(0).numpy()[::-1]
        layer_rows[str(layer)] = {
            'raw_effective_rank': effective_rank(eig),
            'raw_spectral_gap': float(eig[0] - eig[1]) if eig.shape[0] > 1 else None,
            'eig_sum': float(np.sum(eig)),
            'token_count': int(stats[layer]['n']),
        }
        del cov
    return {'probe': probe, 'probe_path': str(probe_path), 'n_samples': nsamples, 'seqlen': seqlen, 'layers': layer_rows}


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    probes_all = probe_paths()
    wanted = parse_str_list(args.probes, DEFAULT_PROBES)
    probes = {k: probes_all[k] for k in wanted if k in probes_all}
    missing = [k for k in wanted if k not in probes_all]
    if missing:
        print(f'[Warn] raw ER requested probes missing and skipped: {missing}', flush=True)
    if not probes:
        raise FileNotFoundError(f'No requested probes found: {wanted}')

    current_sft_step: int | None = None
    for arm in args.arms:
        for step in args.steps:
            if arm == 'sft' and current_sft_step not in (None, step) and args.drop_sft_merged:
                maybe_drop_sft(current_sft_step, True)
            if arm == 'sft':
                current_sft_step = step
            model_path = model_path_for(arm, step)
            pending = {probe: path for probe, path in probes.items() if not raw_er_json(arm, step, probe).exists()}
            if not pending:
                print(f'[Skip rawER] {arm} {step_label(step)} all requested probes exist', flush=True)
                continue
            print(f'[Model] rawER {arm} {step_label(step)} -> {model_path}', flush=True)
            model, tokenizer, dev = load_model(model_path)
            try:
                for probe, path in pending.items():
                    print(f'[rawER] {arm} {step_label(step)} {probe}', flush=True)
                    result = compute_raw_er_loaded(model, tokenizer, dev, probe, path, args.layers, args.n_samples, args.seqlen)
                    result.update({'arm': arm, 'step': step, 'model': str(model_path)})
                    write_json(raw_er_json(arm, step, probe), result)
            finally:
                del model, tokenizer
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    if args.drop_sft_merged and current_sft_step is not None:
        maybe_drop_sft(current_sft_step, True)


def summarize_t8(args: argparse.Namespace) -> None:
    t5 = MINI / 'T5_full_layer_profile.csv'
    rows = []
    if t5.exists() and t5.stat().st_size > 0:
        with open(t5, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append({
                    'construct': 'whitened_weight_er',
                    'arm': row['arm'],
                    'step': row['step'],
                    'layer': row['layer'],
                    'probe': row['probe'],
                    'module': row['module'],
                    'effective_rank': row['effective_rank'],
                    'spectral_gap': row.get('spectral_gap', ''),
                    'source_path': row.get('s_json', ''),
                })
    missing = []
    probes = parse_str_list(args.probes, DEFAULT_PROBES)
    for arm in args.arms:
        for step in args.steps:
            for probe in probes:
                path = raw_er_json(arm, step, probe)
                if not path.exists():
                    missing.append({'arm': arm, 'step': step, 'probe': probe, 'path': str(path)})
                    continue
                data = read_json(path)
                for layer, vals in data.get('layers', {}).items():
                    rows.append({
                        'construct': 'raw_residual_stream_er',
                        'arm': arm,
                        'step': step,
                        'layer': layer,
                        'probe': probe,
                        'module': 'residual_stream',
                        'effective_rank': f"{float(vals['raw_effective_rank']):.8f}",
                        'spectral_gap': '' if vals.get('raw_spectral_gap') is None else f"{float(vals['raw_spectral_gap']):.8f}",
                        'source_path': str(path),
                    })
    write_csv(MINI / 'T8_dual_er.csv', rows)
    write_csv(MINI / 'cycle09_r2_missing_raw_er.csv', missing)
    print(f"[T8] wrote {MINI / 'T8_dual_er.csv'} rows={len(rows)}", flush=True)
    print(f'[Check] missing raw ER cells={len(missing)}', flush=True)


def status(args: argparse.Namespace) -> None:
    probes = parse_str_list(args.probes, DEFAULT_PROBES)
    print('[Status] raw ER roots:')
    print(f'  RUN={RUN}')
    print(f'  MINI={MINI}')
    print('[Status] raw ER probes:')
    for name, path in sorted(probe_paths().items()):
        print(f'  {name}: {path}')
    done = sum(
        int(raw_er_json(arm, step, probe).exists())
        for arm in args.arms
        for step in args.steps
        for probe in probes
    )
    expected = len(args.arms) * len(args.steps) * len(probes)
    print(f'[Status] raw ER json files={done}/{expected}')
    print(f"[Status] T8 exists={(MINI / 'T8_dual_er.csv').exists()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--mini-root', type=Path, default=DEFAULT_MINI)
    parser.add_argument('--arms', default='opd,sft')
    parser.add_argument('--steps', default=','.join(map(str, KEY_STEPS)))
    parser.add_argument('--layers', default=','.join(map(str, LANDMARK_LAYERS)))
    parser.add_argument('--probes', default=','.join(DEFAULT_PROBES))
    parser.add_argument('--n-samples', type=int, default=32)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--drop-sft-merged', action='store_true')
    args = parser.parse_args()
    configure_roots(args.run_root, args.mini_root)
    args.arms = parse_str_list(args.arms, ['opd', 'sft'])
    args.steps = parse_int_list(args.steps, KEY_STEPS)
    args.layers = parse_int_list(args.layers, LANDMARK_LAYERS)
    ensure_dir(MINI)
    if args.status:
        status(args)
    if args.run:
        run(args)
    if args.summarize:
        summarize_t8(args)
    if not (args.status or args.run or args.summarize):
        parser.print_help()


if __name__ == '__main__':
    main()
