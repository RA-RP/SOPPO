#!/usr/bin/env python3
"""T-PK joint source-principal metric with explicit update construction.

``adapter_ba`` measures the process-level fp32 LoRA update.  The separately
labelled ``bf16_merged_minus_base`` mode measures final stored/deployed Qwen
weights.  It is never reported as an adapter BA recovery.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import cycle09_r4_campaign as campaign
import cycle09_stage3_followup_common as c

ROOT = c.scoped_run('H2_tpk')
def output_path(family: str) -> Path:
    return ROOT / f'T_PK_{family}.csv'

def preflight_path(family: str) -> Path:
    return ROOT / f'T_PK_{family}_adapter_preflight.json'

def manifest_path(family: str) -> Path:
    return ROOT / f'T_PK_{family}_manifest.json'
QWEN_BASE = Path('/root/autodl-tmp/model/Qwen/Qwen3-4B-Base')
LLAMA_BASE = Path('/root/autodl-tmp/model/Meta/modelscope/Llama-3.2-3B')
QWEN_MERGED_ROOTS = {
    'opd': Path('/root/autodl-tmp/cycle08_opd_trajectory/_merged_models'),
    'alpha05': Path('/root/autodl-tmp/cycle09_block3/qwen_alpha05/_merged_models'),
}


def verl_checkpoint(arm: str, step: int) -> Path | None:
    roots = {
        'opd': Path('/root/autodl-tmp/cycle08_opd_trajectory/checkpoints'),
        'alpha05': Path('/root/autodl-tmp/cycle09_block3/qwen_alpha05/checkpoints'),
    }
    root = roots.get(arm)
    candidate = root / f'global_step_{step}' / 'actor/model_world_size_1_rank_0.pt' if root else None
    return candidate if candidate and candidate.is_file() else None


def hf_adapter(arm: str, step: int) -> Path | None:
    roots = {
        'sft': Path('/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints'),
        'offkd': Path('/root/autodl-tmp/cycle09_offkd/checkpoints'),
        'seqkd': Path('/root/autodl-tmp/cycle09_seqkd/checkpoints'),
    }
    root = roots.get(arm)
    if not root:
        return None
    candidate = root / (f'step_{step:03d}' if arm == 'sft' else f'checkpoint-{step:06d}')
    return candidate if candidate and (candidate / 'adapter_model.safetensors').is_file() else None


def adapter_source(family: str, arm: str, step: int) -> dict[str, Any]:
    if step == 0:
        return {'complete': True, 'kind': 'zero', 'family': family, 'arm': arm, 'step': step}
    if family == 'qwen3_4b':
        verl = verl_checkpoint(arm, step)
        if verl:
            meta = verl.parent / 'lora_train_meta.json'
            return {'complete': meta.is_file(), 'kind': 'verl_pt', 'path': str(verl), 'meta': str(meta),
                    'family': family, 'arm': arm, 'step': step}
        adapter = hf_adapter(arm, step)
        if adapter:
            return {'complete': True, 'kind': 'safetensors', 'path': str(adapter),
                    'family': family, 'arm': arm, 'step': step}
    resolved = c.discover_adapter(family, arm, step)
    if resolved.get('complete'):
        return resolved | {'kind': 'safetensors', 'path': resolved['adapter']}
    return resolved | {'kind': 'missing'}


def qwen_merged_path(arm: str, step: int) -> Path | None:
    """Return an actual saved Qwen BF16 merge when one is retained."""
    root = QWEN_MERGED_ROOTS.get(arm)
    path = root / f'step_{int(step):03d}' if root else None
    if path and (path / 'config.json').is_file() and (
        (path / 'model.safetensors.index.json').is_file() or (path / 'model.safetensors').is_file()
    ):
        return path
    return None


def delta_source(family: str, arm: str, step: int, mode: str) -> dict[str, Any]:
    """Resolve the declared update object without silently changing estimands."""
    if mode == 'adapter_ba':
        return adapter_source(family, arm, step) | {'delta_mode': mode}
    if mode != 'bf16_merged_minus_base':
        raise ValueError(f'unknown delta mode {mode!r}')
    if family != 'qwen3_4b':
        raise ValueError('bf16_merged_minus_base is currently defined only for Qwen')
    if step == 0:
        return {'complete': True, 'kind': 'zero', 'family': family, 'arm': arm, 'step': step,
                'delta_mode': mode, 'native_source': 'shared_base'}
    merged = qwen_merged_path(arm, step)
    if merged:
        return {'complete': True, 'kind': 'merged_bf16', 'path': str(merged),
                'family': family, 'arm': arm, 'step': step, 'delta_mode': mode,
                'native_source': 'saved_merged_bf16'}
    adapter = adapter_source(family, arm, step)
    if not adapter.get('complete'):
        return adapter | {'delta_mode': mode, 'native_source': 'unavailable'}
    return adapter | {'kind': 'simulated_merged_bf16', 'process_kind': adapter['kind'],
                      'delta_mode': mode,
                      'native_source': 'adapter_merge_quantized_to_bf16'}


def preflight(families: list[str], arms: list[str], steps: list[int], mode: str = 'adapter_ba') -> dict[str, Any]:
    rows = []
    for family in families:
        allowed = c.QWEN_ARMS if family == 'qwen3_4b' else c.LLAMA_ARMS
        for arm in arms:
            if arm not in allowed:
                continue
            for step in steps:
                rows.append(delta_source(family, arm, step, mode))
    payload = {'schema_version': 1, 'task': 'T-PK update-source preflight', 'delta_mode': mode,
               'created_utc': c.utc_now(), 'cells': rows,
               'complete': all(row.get('complete') for row in rows)}
    if len(families) != 1:
        raise ValueError('preflight requires exactly one family for an immutable output path')
    c.atomic_json(preflight_path(families[0]), payload)
    return payload


def scaling(config: dict[str, Any]) -> float:
    rank = float(config['r']); alpha = float(config['lora_alpha'])
    return alpha / math.sqrt(rank) if config.get('use_rslora', False) else alpha / rank


def suffixes(layer: int, module: str) -> tuple[str, str]:
    prefix = f'layers.{layer}.{module}'
    return (f'{prefix}.lora_A.weight', f'{prefix}.lora_B.weight')


def key_for(state: dict[str, torch.Tensor], layer: int, module: str, which: str) -> str:
    endings = (f'layers.{layer}.{module}.lora_{which}.weight',
               f'layers.{layer}.{module}.lora_{which}.default.weight')
    matches = [key for key in state if any(key.endswith(end) for end in endings)]
    if len(matches) != 1:
        raise KeyError(f'expected one lora_{which} key for L{layer} {module}; found={matches}')
    return matches[0]


def local_tensor(value: Any) -> torch.Tensor:
    """Materialize a world-size-one VERL DTensor without its stale DeviceMesh."""
    if hasattr(value, '_local_tensor'):
        value = value._local_tensor
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'expected tensor-like LoRA weight, found {type(value)!r}')
    return value

def adapter_delta(source: dict[str, Any], layer: int, module: str, device: str) -> torch.Tensor | None:
    kind = source.get('process_kind', source['kind'])
    if kind == 'zero':
        return None
    if kind == 'verl_pt':
        state = torch.load(source['path'], map_location='cpu', weights_only=False)
        config = json.loads(Path(source['meta']).read_text(encoding='utf-8'))
    elif kind == 'safetensors':
        adapter = Path(source['path'])
        state = load_file(adapter / 'adapter_model.safetensors')
        config = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    else:
        raise RuntimeError(f"no adapter source: {source}")
    left = local_tensor(state[key_for(state, layer, module, 'B')]).to(device=device, dtype=torch.float32)
    right = local_tensor(state[key_for(state, layer, module, 'A')]).to(device=device, dtype=torch.float32)
    return scaling(config) * (left @ right)


def merged_weight_key(layer: int, module: str) -> str:
    return f'model.layers.{layer}.{module}.weight'


def merged_tensor(root: Path, key: str) -> torch.Tensor:
    index_path = root / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        shard = root / index['weight_map'][key]
    else:
        shard = root / 'model.safetensors'
    with safe_open(str(shard), framework='pt', device='cpu') as handle:
        return handle.get_tensor(key)


def load_delta(
    source: dict[str, Any], layer: int, module: str, device: str,
    base_weight_bf16: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build the declared update in fp32 arithmetic from its declared source."""
    if source['kind'] == 'zero':
        return None
    if source['kind'] in ('verl_pt', 'safetensors'):
        return adapter_delta(source, layer, module, device)
    if base_weight_bf16 is None:
        raise ValueError(f"{source['kind']} requires the saved BF16 base weight")
    if source['kind'] == 'merged_bf16':
        merged = merged_tensor(Path(source['path']), merged_weight_key(layer, module))
        return merged.to(device=device, dtype=torch.float32) - base_weight_bf16.to(dtype=torch.float32)
    if source['kind'] == 'simulated_merged_bf16':
        process_delta = adapter_delta(source, layer, module, device)
        if process_delta is None:
            return None
        try:
            # This is the BF16 deployment object for arms whose full merge was
            # not retained: merge into the retained BF16 base, then subtract it.
            deployed = base_weight_bf16 + process_delta.to(dtype=torch.bfloat16)
            return deployed.to(dtype=torch.float32) - base_weight_bf16.to(dtype=torch.float32)
        finally:
            del process_delta
    raise RuntimeError(f"unknown delta source: {source}")


def load_qwen_checkpoint(arm: str, step: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Load the declared deployed Qwen checkpoint without a disk-side merge.

    OPD and alpha=.5 retain their final BF16 merged models and are loaded
    directly.  The remaining arms retain adapters only, so their deployed
    object is reproduced in memory by merging the adapter into the same BF16
    base.  Callers must persist the returned provenance; the latter is not an
    assertion that an historical full merge still exists on disk.
    """
    if int(step) == 0:
        return campaign.load_model(QWEN_BASE, device), {
            'delta_mode': 'bf16_merged_minus_base',
            'native_source': 'shared_base',
            'checkpoint_materialization': 'base_bf16',
        }
    source = delta_source('qwen3_4b', arm, int(step), 'bf16_merged_minus_base')
    if not source.get('complete'):
        raise RuntimeError(f'cannot load Qwen checkpoint: {source}')
    if source['kind'] == 'merged_bf16':
        return campaign.load_model(Path(source['path']), device), {
            'delta_mode': source['delta_mode'],
            'native_source': source['native_source'],
            'checkpoint_materialization': 'saved_merged_bf16',
        }
    if source['kind'] != 'simulated_merged_bf16':
        raise RuntimeError(f'unsupported Qwen checkpoint source: {source}')
    from peft import PeftModel

    base = campaign.load_model(QWEN_BASE, device)
    try:
        wrapped = PeftModel.from_pretrained(base, source['path'], is_trainable=False)
        model = wrapped.merge_and_unload(safe_merge=True)
        model.config.use_cache = False
        model.eval()
        return model, {
            'delta_mode': source['delta_mode'],
            'native_source': source['native_source'],
            'checkpoint_materialization': 'adapter_merge_quantized_to_bf16',
            'process_kind': source['process_kind'],
        }
    except Exception:
        campaign.unload_model(base)
        raise


def base_path(family: str) -> Path:
    return QWEN_BASE if family == 'qwen3_4b' else LLAMA_BASE


def source_weight(model: Any, layer: int, module: str, device: str) -> torch.Tensor:
    return campaign.module_at(model, layer, module).weight.detach().to(device=device, dtype=torch.float32)


def source_weight_bf16(model: Any, layer: int, module: str, device: str) -> torch.Tensor:
    return campaign.module_at(model, layer, module).weight.detach().to(device=device, dtype=torch.bfloat16)


def principal_basis(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left, _, right_h = torch.linalg.svd(weight, full_matrices=False)
    return left, right_h.T


def pks(
    left: torch.Tensor,
    right: torch.Tensor,
    delta: torch.Tensor,
    fractions: list[float],
    fixed_ks: list[int] | None = None,
) -> list[dict[str, float]]:
    denominator = torch.sum(delta.square()).clamp_min(1e-30)
    out = []
    specs = (
        [(max(1, min(left.shape[1], int(k))), max(1, min(left.shape[1], int(k))) / left.shape[1], "fixed_k")
         for k in (fixed_ks or [])]
        if fixed_ks else
        [(max(1, min(left.shape[1], int(round(fraction * left.shape[1])))), fraction, "rank_fraction")
         for fraction in fractions]
    )
    for k, fraction, rank_spec_kind in specs:
        core = left[:, :k].T @ delta @ right[:, :k]
        out.append({'rank_fraction': fraction, 'k': k, 'rank_spec_kind': rank_spec_kind,
                    'p_k': float(torch.sum(core.square()) / denominator)})
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    fractions = [float(value) for value in args.fractions.split(',') if value.strip()]
    fixed_ks = [int(value) for value in args.ks.split(',') if value.strip()] if args.ks else None
    if fixed_ks and any(item <= 0 for item in fixed_ks):
        raise ValueError('ks must be positive integers')
    if not fixed_ks and (not fractions or any(not 0 < item <= 1 for item in fractions)):
        raise ValueError('fractions must be comma-separated values in (0,1]')
    family = args.family; layers = [int(value) for value in args.layers.split(',')]
    arms = [value.strip() for value in args.arms.split(',') if value.strip()]
    steps = [int(value) for value in args.steps.split(',') if value.strip()]
    check = preflight([family], arms, steps, args.delta_mode)
    missing = [row for row in check['cells'] if not row.get('complete')]
    if missing:
        raise RuntimeError(f'update-source preflight failed for {args.delta_mode}: {missing}')
    model = campaign.load_model(base_path(family), args.device)
    rows: list[dict[str, Any]] = []
    bases: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    try:
        for arm in arms:
            for step in steps:
                source = delta_source(family, arm, step, args.delta_mode)
                for layer in layers:
                    for module in c.MODULES if args.modules == 'all' else args.modules.split(','):
                        key = (layer, module)
                        if key not in bases:
                            weight = source_weight(model, layer, module, args.device)
                            base_bf16 = source_weight_bf16(model, layer, module, args.device)
                            bases[key] = (*principal_basis(weight), base_bf16)
                            del weight
                        left, right, base_bf16 = bases[key]
                        delta = load_delta(source, layer, module, args.device, base_bf16)
                        if delta is None:
                            continue
                        try:
                            for value in pks(left, right, delta, fractions, fixed_ks):
                                rows.append({'family': family, 'arm': arm, 'step': step, 'layer': layer,
                                             'module': module, **value,
                                             'delta_construction': args.delta_mode,
                                             'native_source': source.get('native_source', 'adapter_BA')})
                        finally:
                            del delta
                            torch.cuda.empty_cache()
    finally:
        bases.clear()
        campaign.unload_model(model)
    output = output_path(family); preflight_file = preflight_path(family); manifest = manifest_path(family)
    c.atomic_csv(output, rows)
    payload = {'schema_version': 1, 'status': 'complete', 'task': 'joint source-principal p_k',
               'family': family, 'delta_mode': args.delta_mode, 'fractions': fractions, 'fixed_ks': fixed_ks,
               'rows': len(rows), 'output': c.artifact(output),
               'preflight': c.artifact(preflight_file), 'created_utc': c.utc_now()}
    c.atomic_json(manifest, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'run'), required=True)
    parser.add_argument('--family', choices=('qwen3_4b', 'llama3_2_3b'), default='llama3_2_3b')
    parser.add_argument('--arms', default='opd,offkd')
    parser.add_argument('--steps', default='20')
    parser.add_argument('--layers', default='14')
    parser.add_argument('--modules', default='self_attn.q_proj')
    parser.add_argument('--fractions', default='0.05')
    parser.add_argument('--ks', default='', help='comma-separated fixed k values; overrides --fractions when set')
    parser.add_argument('--delta-mode', choices=('adapter_ba', 'bf16_merged_minus_base'), default='adapter_ba')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.phase == 'preflight':
        result = preflight([args.family], args.arms.split(','), [int(x) for x in args.steps.split(',')], args.delta_mode)
    else:
        result = run(args)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
