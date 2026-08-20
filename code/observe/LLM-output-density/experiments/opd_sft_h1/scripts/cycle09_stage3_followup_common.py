#!/usr/bin/env python3
"""Shared immutable-run helpers for Cycle09 Stage3 follow-up analyses.

Adapter BA remains the process-level update object.  A separately labelled
``bf16_merged_minus_base`` track is permitted only where the task explicitly
asks for the final stored/deployed weight difference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO = Path('/root/LLM-output-density')
AUTODL = Path('/root/autodl-tmp')
MINI = REPO / 'mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini'
RUN_ROOT = AUTODL / 'cycle09_stage3_followup'
SCRIPT_DIR = REPO / 'experiments/opd_sft_h1/scripts'
DENSITY_PYTHON = Path('/root/miniconda3/envs/density/bin/python')

QWEN_ARMS = ('opd', 'sft', 'offkd', 'seqkd', 'alpha05')
LLAMA_ARMS = ('opd', 'sft', 'offkd', 'seqkd')
QWEN_COMMON_STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ALPHA_STEPS = (0, 5, 20, 40, 80, 160, 320)
LLAMA_STEPS = (0, 5, 20, 40, 80, 160, 320)
HEADLINE_LAYER = {'qwen3_4b': 18, 'llama3_2_3b': 14}
MODULES = (
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
)

# These paths are a registry, not a fallback hierarchy.  A caller must report
# exactly which adapter was selected.  Native merged-difference construction is
# implemented explicitly in the consuming task, never inferred here.
ADAPTER_ROOTS: dict[str, tuple[Path, ...]] = {
    'qwen3_4b:offkd': (AUTODL / 'cycle09_offkd/checkpoints', AUTODL / 'cycle09_offkd/checkpoint_backfill'),
    'llama3_2_3b:opd': (AUTODL / 'cycle09_block3/llama_models/adapters/opd',),
    'llama3_2_3b:sft': (AUTODL / 'cycle09_block3/llama_models/adapters/sft',),
    'llama3_2_3b:offkd': (AUTODL / 'cycle09_block3/llama_models/adapters/offkd',),
    'llama3_2_3b:seqkd': (AUTODL / 'cycle09_block3/llama_models/adapters/seqkd',),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {'path': str(path), 'complete': False, 'bytes': 0, 'sha256': None}
    return {'path': str(path), 'complete': path.stat().st_size > 0, 'bytes': path.stat().st_size,
            'sha256': sha256_file(path)}


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def immutable_run(name: str) -> Path:
    root = RUN_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    return root

def scope_root() -> Path:
    """Return the formal root or an explicitly isolated smoke/partition root."""
    scope = os.environ.get('CYCLE09_STAGE3_SCOPE', 'formal').strip()
    if scope in ('', 'formal'):
        return RUN_ROOT
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', scope):
        raise ValueError(f'invalid CYCLE09_STAGE3_SCOPE={scope!r}')
    category = 'partitions' if scope.startswith('partition_') else 'smoke'
    root = RUN_ROOT / category / scope
    root.mkdir(parents=True, exist_ok=True)
    return root


def scoped_run(name: str) -> Path:
    root = scope_root() / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def step_label(step: int) -> str:
    return f'checkpoint-{int(step):06d}'


def discover_adapter(family: str, arm: str, step: int) -> dict[str, Any]:
    """Return exactly one fp32 adapter BA source or a structured missing status."""
    key = f'{family}:{arm}'
    candidates: list[Path] = []
    for root in ADAPTER_ROOTS.get(key, ()):
        direct = root / step_label(step)
        if (direct / 'adapter_model.safetensors').is_file():
            candidates.append(direct)
        # Backfill layouts are intentionally discoverable, but are not guessed.
        candidates.extend(sorted(
            item.parent for item in root.glob(f'**/{step_label(step)}/adapter_model.safetensors')
        ))
    unique = sorted({item.resolve() for item in candidates})
    if len(unique) != 1:
        return {'family': family, 'arm': arm, 'step': int(step), 'complete': False,
                'reason': 'missing_adapter' if not unique else 'ambiguous_adapter',
                'candidates': [str(item) for item in unique]}
    adapter = unique[0]
    config = adapter / 'adapter_config.json'
    weights = adapter / 'adapter_model.safetensors'
    if not config.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        return {'family': family, 'arm': arm, 'step': int(step), 'complete': False,
                'reason': 'incomplete_adapter', 'adapter': str(adapter)}
    return {'family': family, 'arm': arm, 'step': int(step), 'complete': True,
            'adapter': str(adapter), 'config': artifact(config), 'weights': artifact(weights),
            'delta_construction': 'fp32_lora_scaling_times_BA'}



def load_llama_checkpoint(arm: str, step: int, device: str):
    """Load a Llama checkpoint from its retained PEFT adapter without a disk merge."""
    import cycle09_block3_common as block3
    import cycle09_r4_campaign as campaign

    if int(step) == 0:
        return campaign.load_model(block3.LLAMA_STUDENT, device)
    source = discover_adapter('llama3_2_3b', arm, int(step))
    if not source.get('complete'):
        raise RuntimeError(f'cannot load Llama adapter checkpoint: {source}')
    from peft import PeftModel

    base = campaign.load_model(block3.LLAMA_STUDENT, device)
    wrapped = PeftModel.from_pretrained(base, source['adapter'], is_trainable=False)
    model = wrapped.merge_and_unload(safe_merge=True)
    model.config.use_cache = False
    model.eval()
    return model

def source_geometry_files() -> list[tuple[str, Path]]:
    return [
        ('qwen_r4', MINI / 'R4_m1_tail_ec.csv'),
        ('qwen_alpha05', MINI / 'qwen_alpha05_r_epsilon.csv'),
        ('llama', MINI / 'llama_early_320_r_epsilon.csv'),
    ]


def canonical_geometry() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Normalize existing per-checkpoint geometry without silently filling cells."""
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for label, path in source_geometry_files():
        item = artifact(path) | {'source': label}
        inventory.append(item)
        if not item['complete']:
            continue
        frame = pd.read_csv(path)
        lower = {column.lower(): column for column in frame.columns}
        probe_col = lower.get('probe') or lower.get('probe_family')
        arm_col = lower.get('arm')
        step_col = lower.get('step')
        layer_col = lower.get('layer')
        module_col = lower.get('module')
        epsilon_col = lower.get('epsilon')
        if not all((probe_col, arm_col, step_col, layer_col, module_col, epsilon_col)):
            inventory[-1]['schema_error'] = 'required geometry columns absent'
            continue
        if label == 'qwen_r4':
            delta_col = lower.get('r_epsilon_delta') or lower.get('delta_from_base')
            value_col = lower.get('r_epsilon') or lower.get('r_epsilon_current')
            family = 'qwen3_4b'
        elif label == 'qwen_alpha05':
            delta_col = lower.get('r_epsilon_delta') or lower.get('delta_from_base')
            value_col = lower.get('r_epsilon')
            family = 'qwen3_4b'
        else:
            delta_col = lower.get('delta_from_base') or lower.get('r_epsilon_delta')
            value_col = lower.get('r_epsilon')
            family = 'llama3_2_3b'
        if not delta_col or not value_col:
            inventory[-1]['schema_error'] = 'rank/delta column absent'
            continue
        selected = pd.DataFrame({
            'family': family,
            'arm': frame[arm_col].astype(str),
            'step': pd.to_numeric(frame[step_col], errors='raise').astype(int),
            'probe': frame[probe_col].astype(str),
            'layer': pd.to_numeric(frame[layer_col], errors='raise').astype(int),
            'module': frame[module_col].astype(str),
            'epsilon': pd.to_numeric(frame[epsilon_col], errors='raise'),
            'r_epsilon': pd.to_numeric(frame[value_col], errors='raise'),
            'delta_from_base': pd.to_numeric(frame[delta_col], errors='raise'),
            'source': label,
        })
        if 'track' in frame.columns:
            selected['track'] = frame['track'].astype(str)
        else:
            selected['track'] = 'unknown'
        if label == 'qwen_alpha05':
            selected['arm'] = 'alpha05'
        frames.append(selected)
    if not frames:
        return pd.DataFrame(), inventory
    result = pd.concat(frames, ignore_index=True)
    result = result[(result['epsilon'] == 0.05) & (result['track'].isin(('per_checkpoint', 'per_checkpoint_only')))]
    return result.sort_values(['family', 'arm', 'step', 'probe', 'layer', 'module'], kind='stable'), inventory


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return '| status |\n|---|\n| no rows |\n'
    rows = frame.loc[:, columns].fillna('').astype(str)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines.extend('| ' + ' | '.join(value.replace('|', '\\|') for value in row) + ' |' for row in rows.itertuples(index=False, name=None))
    return '\n'.join(lines)
