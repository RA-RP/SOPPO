#!/usr/bin/env python3
"""Raw M3 stability, sample-count, and matched-timing audit.

This script deliberately does not fit a lag model or invent missing geometry.
It freezes the available epsilon and finite-sample artifacts, then records the
exact behavior--geometry intersections that a later descriptive lead-lag
analysis is allowed to use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_stage3_followup_common as c


ROOT = c.scoped_run('H4_increment')
EPSILON = ROOT / 'M3_epsilon_stability.csv'
SAMPLES = ROOT / 'M3_sample_count_inventory.csv'
LEAD_LAG = ROOT / 'M3_lead_lag_coverage.csv'
MANIFEST = ROOT / 'M3_audit_manifest.json'

MINI = c.MINI
PROBE_CORE = c.scope_root() / 'H2_probe_core/PROBE_CORE_r_epsilon.csv'


def artifact_or_missing(path: Path, role: str) -> dict[str, Any]:
    row = c.artifact(path) | {'role': role}
    if not row['complete']:
        row['reason'] = 'artifact_not_retained'
    return row


def normalise_epsilon(path: Path, label: str, family: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lower = {column.lower(): column for column in frame.columns}
    probe = lower.get('probe') or lower.get('probe_family')
    rank = lower.get('r_epsilon') or lower.get('r_epsilon_current')
    delta = lower.get('delta_from_base') or lower.get('r_epsilon_delta')
    required = [probe, lower.get('arm'), lower.get('step'), lower.get('layer'), lower.get('module'), lower.get('epsilon'), rank, delta]
    if not all(required):
        raise ValueError(f'{path} lacks a required epsilon column')
    selected = pd.DataFrame({
        'family': family,
        'arm': frame[lower['arm']].astype(str),
        'step': pd.to_numeric(frame[lower['step']], errors='raise').astype(int),
        'probe': frame[probe].astype(str),
        'layer': pd.to_numeric(frame[lower['layer']], errors='raise').astype(int),
        'module': frame[lower['module']].astype(str),
        'epsilon': pd.to_numeric(frame[lower['epsilon']], errors='raise'),
        'r_epsilon': pd.to_numeric(frame[rank], errors='raise'),
        'delta_from_base': pd.to_numeric(frame[delta], errors='raise'),
        'source': label,
    })
    if 'track' in frame:
        selected = selected[frame['track'].astype(str).isin(('per_checkpoint', 'per_checkpoint_only', 'per_checkpoint_S_Dt'))]
    return selected


def epsilon_rows() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sources = [
        ('qwen_r4', MINI / 'R4_m1_tail_ec.csv', 'qwen3_4b'),
        ('qwen_alpha05', MINI / 'qwen_alpha05_r_epsilon.csv', 'qwen3_4b'),
        ('llama', MINI / 'llama_early_320_r_epsilon.csv', 'llama3_2_3b'),
    ]
    inventory = [artifact_or_missing(path, label) for label, path, _ in sources]
    frames = [normalise_epsilon(path, label, family) for label, path, family in sources if path.is_file()]
    if PROBE_CORE.is_file():
        exact = pd.read_csv(PROBE_CORE)
        exact = exact.rename(columns={'delta_from_base': 'delta_from_base'}).copy()
        exact['source'] = 'probe_core_exact'
        frames.append(exact[['family', 'arm', 'step', 'probe', 'layer', 'module', 'epsilon', 'r_epsilon', 'delta_from_base', 'source']])
        inventory.append(artifact_or_missing(PROBE_CORE, 'probe_core_exact'))
    else:
        inventory.append(artifact_or_missing(PROBE_CORE, 'probe_core_exact'))
    if not frames:
        raise RuntimeError('no retained epsilon artifact is readable')
    combined = pd.concat(frames, ignore_index=True)
    # Exact MATH500 landmarks are semantically distinct from historical
    # same-named probes and take precedence only where they exist.
    exact_keys = set(tuple(row) for row in combined.loc[combined.source == 'probe_core_exact', ['family', 'arm', 'step', 'layer', 'module', 'epsilon']].itertuples(index=False, name=None))
    historic_math = (combined.probe == 'E_math') & (combined.source != 'probe_core_exact')
    mask = []
    for _, row in combined.iterrows():
        key = (row.family, row.arm, int(row.step), int(row.layer), row.module, float(row.epsilon))
        mask.append(not (row.probe == 'E_math' and row.source != 'probe_core_exact' and key in exact_keys))
    combined = combined.loc[mask].copy()
    summary = combined.groupby(['family', 'arm', 'step', 'probe', 'layer', 'epsilon'], as_index=False).agg(
        r_epsilon_module_mean=('r_epsilon', 'mean'),
        delta_from_base_module_mean=('delta_from_base', 'mean'),
        n_modules=('module', 'nunique'),
        source_count=('source', 'nunique'),
    )
    return summary.sort_values(['family', 'probe', 'arm', 'step', 'epsilon'], kind='stable'), inventory


def behavior_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    qwen = MINI / 'three_arm_full_trajectory.csv'
    if qwen.is_file():
        frame = pd.read_csv(qwen)
        mapping = [('math500_acc', 'math500_accuracy', 'E_math'), ('mmlu_pro_exact_match', 'mmlu_pro_exact_match', 'E_mmluPro'), ('ifeval_prompt_strict', 'ifeval_prompt_strict', 'E_if')]
        for column, outcome, probe in mapping:
            if column not in frame:
                continue
            for _, item in frame.dropna(subset=[column]).iterrows():
                rows.append({'family': 'qwen3_4b', 'arm': str(item.arm), 'step': int(item.step), 'outcome': outcome, 'probe': probe, 'behavior_value': float(item[column]), 'behavior_source': str(qwen)})
    llama = MINI / 'llama_early_320_behavior.csv'
    if llama.is_file():
        frame = pd.read_csv(llama)
        mapping = {'math500': ('math500_accuracy', 'E_math', 'accuracy'), 'mmlu_pro': ('mmlu_pro_flexible', 'E_mmluPro', 'flexible_accuracy'), 'ifeval': ('ifeval_prompt_strict', 'E_if', 'prompt_strict_accuracy')}
        for task, (outcome, probe, column) in mapping.items():
            if column not in frame:
                continue
            for _, item in frame[(frame.task == task) & frame[column].notna()].iterrows():
                rows.append({'family': 'llama3_2_3b', 'arm': str(item.arm), 'step': int(item.step), 'outcome': outcome, 'probe': probe, 'behavior_value': float(item[column]), 'behavior_source': str(llama)})
    return pd.DataFrame(rows, columns=['family', 'arm', 'step', 'outcome', 'probe', 'behavior_value', 'behavior_source'])


def lead_lag_coverage(epsilon: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    behavior = behavior_rows()
    geometry = epsilon[epsilon.epsilon == .05].groupby(['family', 'arm', 'step', 'probe'], as_index=False).agg(
        geometry_delta=('delta_from_base_module_mean', 'mean'),
        geometry_modules=('n_modules', 'max'),
        geometry_sources=('source_count', 'max'),
    )
    joined = behavior.merge(geometry, on=['family', 'arm', 'step', 'probe'], how='left')
    joined['coverage'] = joined.geometry_delta.notna().map({True: 'matched', False: 'missing_geometry'})
    missing = []
    for _, row in joined[joined.coverage != 'matched'].iterrows():
        missing.append({'family': row.family, 'arm': row.arm, 'step': int(row.step), 'probe': row.probe, 'outcome': row.outcome, 'reason': 'no exact retained matched geometry cell'})
    return joined.sort_values(['family', 'outcome', 'arm', 'step'], kind='stable'), missing


def run() -> dict[str, Any]:
    epsilon, source_inventory = epsilon_rows()
    sample_paths = [
        (MINI / 'R3_er_sample_bands.csv', 'historical_r_epsilon_sample_bands'),
        (MINI / 'R3_er_sample_bands_draws.json', 'historical_r_epsilon_bootstrap_draws'),
        (MINI / 'C15_cap_pilot_samples_corrected.csv', 'cap_pilot_sample_table'),
    ]
    samples = [artifact_or_missing(path, role) for path, role in sample_paths]
    lead_lag, missing_cells = lead_lag_coverage(epsilon)
    c.atomic_csv(EPSILON, epsilon.to_dict('records'), list(epsilon.columns))
    c.atomic_csv(SAMPLES, samples, ['role', 'path', 'complete', 'bytes', 'sha256', 'reason'])
    c.atomic_csv(LEAD_LAG, lead_lag.to_dict('records'), list(lead_lag.columns))
    payload = {
        'schema_version': 1,
        'status': 'complete_with_declared_missing_cells',
        'task': 'M3 raw stability/sample-count/matched-timing audit',
        'policy': {
            'epsilon': 'module-equal summary of retained per-checkpoint artifacts; exact E_math supersedes historical same-named cells only where present',
            'sample_count': 'inventory only; no recomputation from unrecoverable sample factors',
            'lead_lag': 'coverage and matched raw cells only; no fitted timing or mediation interpretation',
        },
        'source_inventory': source_inventory,
        'sample_inventory': samples,
        'missing_cells': missing_cells,
        'outputs': [c.artifact(path) for path in (EPSILON, SAMPLES, LEAD_LAG)],
        'created_utc': c.utc_now(),
    }
    c.atomic_json(MANIFEST, payload)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all',), required=True)
    parser.parse_args()
    print(json.dumps(run(), indent=2))
