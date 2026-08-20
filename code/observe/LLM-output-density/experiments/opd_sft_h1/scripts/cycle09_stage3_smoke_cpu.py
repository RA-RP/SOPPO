#!/usr/bin/env python3
"""Isolated synthetic CPU checks for the Cycle09 Stage3 follow-up pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import cycle09_stage3_followup_common as c

ROOT = c.scoped_run('runner')
FIXTURES = ROOT / 'fixtures'
MANIFEST = ROOT / 'cpu_smoke_manifest.json'


def require_existing() -> list[dict]:
    paths = [
        c.scope_root() / 'H1_resync/H1_resync_manifest.json',
        c.scope_root() / 'contracts/contracts_manifest.json',
        c.scope_root() / 'H2_tpk/T_PK_llama3_2_3b_manifest.json',
        c.scope_root() / 'H2_tpk/T_PK_qwen3_4b_manifest.json',
        c.scope_root() / 'H2_white/T_WHITE_llama3_2_3b_manifest.json',
        c.scope_root() / 'H2_white/T_WHITE_qwen3_4b_manifest.json',
        c.scope_root() / 'H2_sub/T_SUB_llama3_2_3b_manifest.json',
        c.scope_root() / 'H2_sub/T_SUB_qwen3_4b_manifest.json',
    ]
    results = []
    for path in paths:
        payload = c.read_json(path, {})
        if payload.get('status') != 'complete':
            raise RuntimeError(f'incomplete prerequisite smoke manifest: {path}')
        results.append(c.artifact(path))
    return results


def support_smoke() -> dict:
    import cycle09_stage3_support as support

    cells = []
    for arm_index, arm in enumerate(('opd', 'offkd')):
        for step in (20, 40):
            target = FIXTURES / 'support' / f'{arm}_{step}.jsonl'
            rows = []
            for index in range(3):
                rows.append({
                    'output': f'{arm} step {step} answer {index} token token',
                    'finished': index != 2,
                    'truncated': index == 2,
                    'source_kl': 0.1 * (arm_index + 1) + index / 100,
                    'source_loss': 0.2 * (arm_index + 1) + index / 100,
                })
            c.atomic_jsonl(target, rows)
            cells.append({'family': 'llama3_2_3b', 'arm': arm, 'step': step, 'path': str(target)})
    input_manifest = FIXTURES / 'support_inputs.json'
    c.atomic_json(input_manifest, {'schema_version': 1, 'status': 'frozen', 'cells': cells})
    return support.run(input_manifest)


def increment_smoke() -> dict:
    import cycle09_stage3_increment as increment

    arms = ('opd', 'offkd', 'seqkd', 'sft')
    steps = (5, 20, 40, 80, 160)
    probes = ('E_mathHeld', 'E_if')
    geometry_rows = []
    behavior_rows = []
    for arm_index, arm in enumerate(arms):
        for step_index, step in enumerate(steps):
            for probe_index, probe in enumerate(probes):
                geometry_rows.append({
                    'arm': arm,
                    'step': step,
                    'probe': probe,
                    'A_D': arm_index * 0.3 + step_index * 0.05 + probe_index * 0.01,
                    'G_D': arm_index * 0.2 + step_index * 0.04,
                    'p_k': arm_index * 0.1 + step_index * 0.03 + probe_index * 0.02,
                })
                behavior_rows.append({
                    'arm': arm,
                    'step': step,
                    'probe': probe,
                    'outcome': 'math500_accuracy' if probe == 'E_mathHeld' else 'ifeval_prompt_strict',
                    'value': 0.35 + arm_index * 0.04 + step_index * 0.02 + probe_index * 0.01,
                })
    geometry = pd.DataFrame(geometry_rows)
    behavior = pd.DataFrame(behavior_rows)
    splits = increment.frozen_splits(list(steps), 20260723)
    track_a_oof, track_a_summary = increment.track_a(geometry, splits)
    track_b_oof, track_b_summary = increment.track_b(geometry, splits, behavior)
    outputs = []
    for name, frame in (
        ('track_a_oof.csv', track_a_oof),
        ('track_a_summary.csv', track_a_summary),
        ('track_b_oof.csv', track_b_oof),
        ('track_b_summary.csv', track_b_summary),
    ):
        if frame.empty:
            raise RuntimeError(f'{name} is empty')
        target = ROOT / name
        c.atomic_csv(target, frame.to_dict('records'))
        outputs.append(c.artifact(target))
    return {'status': 'complete', 'outputs': outputs}


def optional_gate_smoke() -> dict:
    import cycle09_stage3_frozen_self as frozen
    import cycle09_stage3_mediator as mediator

    bad_gate = FIXTURES / 'bad_h4_gate.json'
    c.atomic_json(bad_gate, {'status': 'complete', 'gate': 'H3'})
    try:
        frozen.prepare(bad_gate)
    except RuntimeError:
        rejected_bad_gate = True
    else:
        raise RuntimeError('H5 accepted a non-H4 gate')
    h4_gate = FIXTURES / 'h4_gate.json'
    c.atomic_json(h4_gate, {'schema_version': 1, 'status': 'complete', 'gate': 'H4'})
    contract = frozen.prepare(h4_gate)
    total_effect = c.scope_root() / 'H5_frozen_self/frozen_self_total_effect.csv'
    rows = []
    for arm_index, arm in enumerate(('opd', 'offkd')):
        for step in (20, 40):
            rows.append({'family': 'llama3_2_3b', 'arm': arm, 'step': step,
                         'A_D': 0.1 * arm_index + step / 1000,
                         'G_D': 0.2 * arm_index + step / 1000,
                         'total_effect': 0.3 * arm_index + step / 1000})
    c.atomic_csv(total_effect, rows)
    h5_gate = FIXTURES / 'h5_gate.json'
    c.atomic_json(h5_gate, {'schema_version': 1, 'status': 'complete', 'gate': 'H5'})
    mediation = mediator.run(h5_gate)
    return {'status': 'complete', 'rejected_bad_gate': rejected_bad_gate,
            'h5_contract_status': contract['status'], 'h6_status': mediation['status']}


def main() -> None:
    payload = {'schema_version': 1, 'status': 'complete', 'task': 'Stage3 isolated CPU smoke',
               'existing': require_existing(), 'support': support_smoke(),
               'increment': increment_smoke(), 'optional_gates': optional_gate_smoke(),
               'created_utc': c.utc_now()}
    c.atomic_json(MANIFEST, payload)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

