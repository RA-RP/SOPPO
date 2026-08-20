#!/usr/bin/env python3
"""Detached smoke-only gate for Cycle09 Stage3 follow-up scripts.

It may wait for Q1 to finish, but it never starts a formal experiment.  A
formal supervisor must be launched separately after a smoke manifest is
complete and a user/Theory GO is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cycle09_stage3_followup_common as c

ROOT = c.immutable_run('smoke')
STATUS = ROOT / 'status.json'
MANIFEST = ROOT / 'smoke_manifest.json'
LOG_ROOT = ROOT / 'logs'
Q1_STATUS = Path('/root/autodl-tmp/cycle09_block3/stageb_320/status.json')
Q1_POSTPROCESS = c.MINI / 'qwen_alpha05_stage_b_320_handoff_manifest.json'
PYTHON = Path('/root/miniconda3/envs/density/bin/python')
SCRIPTS = (
    'cycle09_stage3_followup_common.py',
    'cycle09_stage3_resync.py',
    'cycle09_stage3_tpk.py',
    'cycle09_stage3_followup_smoke.py',
)


def q1_ready() -> tuple[bool, str]:
    handoff = c.read_json(Q1_POSTPROCESS, {})
    if handoff.get('status') == 'complete':
        return True, 'q1_stageb_handoff_complete'
    state = c.read_json(Q1_STATUS, {})
    return False, f"q1_not_ready:{state.get('state', 'missing')}"


def run_command(name: str, argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log = LOG_ROOT / f'{name}.log'
    started = c.utc_now()
    with log.open('a', encoding='utf-8') as handle:
        handle.write(f'[{started}] RUN {" ".join(argv)}\n')
        handle.flush()
        completed = subprocess.run(argv, cwd=c.REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return {'name': name, 'argv': argv, 'returncode': completed.returncode, 'started_utc': started,
            'finished_utc': c.utc_now(), 'log': c.artifact(log)}


def static() -> dict[str, Any]:
    rows = []
    for script in SCRIPTS:
        rows.append(run_command(f'compile_{script[:-3]}', [sys.executable, '-m', 'py_compile', str(c.SCRIPT_DIR / script)], os.environ.copy()))
    complete = all(row['returncode'] == 0 for row in rows)
    payload = {'schema_version': 1, 'kind': 'static', 'status': 'complete' if complete else 'failed',
               'results': rows, 'created_utc': c.utc_now()}
    c.atomic_json(ROOT / 'static_manifest.json', payload)
    if not complete:
        raise RuntimeError('static smoke failed; inspect logs')
    return payload


def gpu_core() -> dict[str, Any]:
    # The first real forward-free smoke is a strict fp32 BA/SVD calculation on
    # one Llama module.  Later task entries are appended only after their
    # implementation is compiled; this prevents a stale queue from claiming coverage.
    command = [str(PYTHON), str(c.SCRIPT_DIR / 'cycle09_stage3_tpk.py'), '--phase', 'run',
               '--family', 'llama3_2_3b', '--arms', 'opd,offkd', '--steps', '20', '--layers', '14',
               '--modules', 'self_attn.q_proj', '--fractions', '0.05', '--device', 'cuda:0']
    env = os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES': '0', 'PYTHONUNBUFFERED': '1'})
    row = run_command('tpk_llama_one_cell', command, env)
    payload = {'schema_version': 1, 'kind': 'gpu_core_partial',
               'status': 'complete' if row['returncode'] == 0 else 'failed',
               'coverage': 'T-PK only; queue will expand after T-WHITE/T-SUB/PROBE-CORE implementation',
               'results': [row], 'created_utc': c.utc_now()}
    c.atomic_json(ROOT / 'gpu_core_partial_manifest.json', payload)
    if row['returncode'] != 0:
        raise RuntimeError(f"GPU smoke failed; inspect {row['log']['path']}")
    return payload


def wait_then_smoke(interval: int) -> dict[str, Any]:
    c.atomic_json(STATUS, {'status': 'waiting_for_q1', 'created_utc': c.utc_now(), 'q1_status': str(Q1_STATUS)})
    while True:
        ready, reason = q1_ready()
        if ready:
            break
        c.atomic_json(STATUS, {'status': 'waiting_for_q1', 'updated_utc': c.utc_now(), 'reason': reason})
        time.sleep(interval)
    static_payload = static()
    gpu_payload = gpu_core()
    payload = {'schema_version': 1, 'status': 'complete', 'mode': 'smoke_only',
               'q1_gate': reason, 'static': static_payload, 'gpu': gpu_payload,
               'formal_experiments_started': False, 'created_utc': c.utc_now()}
    c.atomic_json(MANIFEST, payload)
    c.atomic_json(STATUS, {'status': 'complete', 'manifest': c.artifact(MANIFEST), 'updated_utc': c.utc_now()})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('static', 'gpu-core', 'wait'), required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--execute', action='store_true', help='required for any smoke action')
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit('smoke is disabled by default; pass --execute only after Q1 ends or an explicit interruption')
    result = static() if args.phase == 'static' else gpu_core() if args.phase == 'gpu-core' else wait_then_smoke(args.poll_seconds)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
