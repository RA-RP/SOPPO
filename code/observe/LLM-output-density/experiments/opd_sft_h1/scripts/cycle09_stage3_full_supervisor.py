#!/usr/bin/env python3
"""Detached controller for the approved Cycle09 Stage3 queue.

The controller owns every remaining GPU and CPU stage after H0.  It never
depends on a second detached shell supervisor: a failed child marks the stage
failed in this status file with its log path, rather than silently stranding a
GPU queue.  The user has explicitly authorized H5 frozen-self after H4; H6
mediator and supplemental work still require a later Theory GO.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cycle09_stage3_followup_common as c


PYTHON = '/root/miniconda3/envs/density/bin/python'
SCRIPTS = c.SCRIPT_DIR
ROOT = c.RUN_ROOT / 'full_supervisor'
LOGS = ROOT / 'logs'
STATUS = ROOT / 'status.json'
PID_FILE = ROOT / 'supervisor.pid'
STOP_FILE = ROOT / 'STOP_AFTER_CURRENT'
POLL_SECONDS = 20

H0_STATUS = c.RUN_ROOT / 'H0_q1_finalize/status.json'
H0_PID = c.RUN_ROOT / 'H0_q1_finalize/supervisor.pid'
GPU0_STATUS = c.RUN_ROOT / 'gpu0_core/status.json'
GPU0_PID = c.RUN_ROOT / 'gpu0_core/supervisor.pid'
MERGE_STATUS = c.RUN_ROOT / 'probe_merge/status.json'
MERGE_PID = c.RUN_ROOT / 'probe_merge/supervisor.pid'

H1 = c.RUN_ROOT / 'H1_resync/H1_resync_manifest.json'
TPK_Q = c.RUN_ROOT / 'H2_tpk/T_PK_qwen3_4b_manifest.json'
WHITE_Q = c.RUN_ROOT / 'H2_white/T_WHITE_qwen3_4b_manifest.json'
SUB_Q = c.RUN_ROOT / 'H2_sub/T_SUB_qwen3_4b_manifest.json'
PROBE = c.RUN_ROOT / 'H2_probe_core/PROBE_CORE_manifest.json'
H2 = c.MINI / 'stage3_H2_handoff_manifest.json'
H3 = c.MINI / 'stage3_H3_handoff_manifest.json'
H4 = c.MINI / 'stage3_H4_handoff_manifest.json'
H5_FROZEN = c.RUN_ROOT / 'H5_frozen_self/FROZEN_SELF_manifest.json'
H5 = c.MINI / 'stage3_H5_handoff_manifest.json'


def payload(path: Path) -> dict[str, Any]:
    return c.read_json(path, {}) if path.is_file() else {}


def complete(path: Path, **expected: Any) -> bool:
    value = payload(path)
    if not str(value.get('status', '')).startswith('complete'):
        return False
    return all(value.get(key) == expected_value for key, expected_value in expected.items())


def pid_alive(path: Path) -> bool | None:
    try:
        pid = int(path.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Supervisor:
    def __init__(self) -> None:
        ROOT.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        previous = payload(STATUS)
        self.state: dict[str, Any] = previous if previous.get('schema_version') == 1 else {
            'schema_version': 1,
            'auto_shutdown': False,
            'policy': {
                'scope': 'H0 through H5 frozen-self',
                'qwen_delta': 'bf16_merged_minus_base',
                'opd_alpha_source': 'saved_merged_bf16',
                'other_qwen_arms_source': 'adapter_merge_quantized_to_bf16',
                'post_H5': 'stop_and_await_theory_go_for_H6_or_supplements',
            },
            'started_utc': c.utc_now(),
            'stages': {},
        }
        self.state.setdefault('auto_shutdown', False)
        self.state['policy'] = {
            'scope': 'H0 through H5 frozen-self',
            'qwen_delta': 'bf16_merged_minus_base',
            'opd_alpha_source': 'saved_merged_bf16',
            'other_qwen_arms_source': 'adapter_merge_quantized_to_bf16',
            'post_H5': 'stop_and_await_theory_go_for_H6_or_supplements',
        }
        self.state.setdefault('stages', {})
        self.state.update({'pid': os.getpid(), 'updated_utc': c.utc_now(), 'state': 'starting'})
        self.write()

    def write(self) -> None:
        self.state['updated_utc'] = c.utc_now()
        c.atomic_json(STATUS, self.state)

    def stage(self, name: str, state: str, **extra: Any) -> None:
        self.state.setdefault('stages', {}).setdefault(name, {}).update({'state': state, 'updated_utc': c.utc_now(), **extra})
        self.write()

    def halted(self) -> None:
        if STOP_FILE.is_file():
            self.state['state'] = 'stopped_after_current'
            self.write()
            raise SystemExit(f'stop requested by {STOP_FILE}')

    def wait_external(self, name: str, status_path: Path, pid_path: Path | None = None, *, artifact: Path | None = None) -> None:
        self.stage(name, 'waiting_external', status_path=str(status_path), artifact=str(artifact) if artifact else None)
        while True:
            self.halted()
            status = payload(status_path).get('status', '')
            if str(status).startswith('complete'):
                if artifact is not None and not artifact.is_file():
                    raise RuntimeError(f'{name} claims complete but lacks {artifact}')
                self.stage(name, 'complete_external')
                return
            if status in ('failed', 'cancelled', 'stopped'):
                raise RuntimeError(f'{name} external supervisor status={status!r}: {status_path}')
            if pid_path is not None and pid_alive(pid_path) is False:
                raise RuntimeError(f'{name} supervisor exited before completion: {status_path}')
            time.sleep(POLL_SECONDS)

    def wait_artifact(self, name: str, artifact: Path, status_path: Path, pid_path: Path) -> None:
        """Return at an independently validated artifact, without idling for a later partition."""
        self.stage(name, 'waiting_artifact', status_path=str(status_path), artifact=str(artifact))
        while True:
            self.halted()
            if complete(artifact):
                self.stage(name, 'complete_artifact')
                return
            status = payload(status_path).get('status', '')
            if status in ('failed', 'cancelled', 'stopped'):
                raise RuntimeError(f'{name} external supervisor status={status!r}: {status_path}')
            if pid_alive(pid_path) is False:
                raise RuntimeError(f'{name} supervisor exited before producing {artifact}')
            time.sleep(POLL_SECONDS)

    def command_env(self, *, gpu: int | None, scope: str | None = None) -> dict[str, str]:
        """Construct a child environment without inheriting stale run scopes."""
        env = os.environ.copy()
        env.pop('CYCLE09_STAGE3_SCOPE', None)
        if scope:
            env['CYCLE09_STAGE3_SCOPE'] = scope
        env['PYTHONUNBUFFERED'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        return env

    def run(self, name: str, argv: list[str], *, gpu: int | None = None, scope: str | None = None) -> None:
        self.halted()
        log = LOGS / f'{name}.log'
        env = self.command_env(gpu=gpu, scope=scope)
        self.stage(name, 'running', argv=argv, gpu=gpu, scope=scope, log=str(log))
        with log.open('a', encoding='utf-8') as handle:
            handle.write(f'[{c.utc_now()}] START {" ".join(argv)}\n')
            handle.flush()
            child = subprocess.Popen(argv, cwd=c.REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
            self.stage(name, 'running', argv=argv, gpu=gpu, scope=scope, log=str(log), child_pid=child.pid)
            rc = child.wait()
            handle.write(f'[{c.utc_now()}] END rc={rc}\n')
        if rc:
            self.stage(name, 'failed', returncode=rc, log=str(log))
            raise RuntimeError(f'{name} failed rc={rc}; log={log}')
        self.stage(name, 'complete', log=str(log))

    def run_parallel(self, jobs: list[tuple[str, list[str], int | None, str | None]]) -> None:
        """Launch independent GPU jobs together, then surface every child result."""
        self.halted()
        active: list[tuple[str, subprocess.Popen[Any], Any, Path]] = []
        try:
            for name, argv, gpu, scope in jobs:
                log = LOGS / f'{name}.log'
                env = self.command_env(gpu=gpu, scope=scope)
                self.stage(name, 'running', argv=argv, gpu=gpu, scope=scope, log=str(log))
                handle = log.open('a', encoding='utf-8')
                handle.write(f'[{c.utc_now()}] START {" ".join(argv)}\n')
                handle.flush()
                child = subprocess.Popen(argv, cwd=c.REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
                self.stage(name, 'running', argv=argv, gpu=gpu, scope=scope, log=str(log), child_pid=child.pid)
                active.append((name, child, handle, log))

            failures: list[str] = []
            for name, child, handle, log in active:
                rc = child.wait()
                handle.write(f'[{c.utc_now()}] END rc={rc}\n')
                handle.flush()
                if rc:
                    self.stage(name, 'failed', returncode=rc, log=str(log))
                    failures.append(f'{name} rc={rc}; log={log}')
                else:
                    self.stage(name, 'complete', log=str(log))
            if failures:
                raise RuntimeError('; '.join(failures))
        finally:
            for _, _, handle, _ in active:
                handle.close()

    def archive_h1_v1(self) -> None:
        inventory = payload(c.RUN_ROOT / 'H1_resync/source_inventory.json')
        source_rows = inventory.get('sources', [])
        needs_revision = any(row.get('source') == 'qwen_r4' and row.get('schema_error') for row in source_rows)
        marker = ROOT / 'H1_resync_v2_archive.json'
        if not needs_revision or marker.is_file():
            return
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        archive = c.RUN_ROOT / 'archive' / f'H1_resync_pre_v2_{stamp}'
        source = c.RUN_ROOT / 'H1_resync'
        if source.is_dir():
            shutil.copytree(source, archive)
        mini_archive = c.MINI / 'archive' / f'H1_pre_v2_{stamp}'
        mini_archive.mkdir(parents=True, exist_ok=True)
        for path in (c.MINI / 'mini_stage3_H1_theory_handoff.md', c.MINI / 'stage3_H1_handoff_manifest.json'):
            if path.is_file():
                shutil.copy2(path, mini_archive / path.name)
        c.atomic_json(marker, {'status': 'complete', 'archive': str(archive), 'mini_archive': str(mini_archive), 'created_utc': c.utc_now()})

    def h1_is_corrected(self) -> bool:
        if not complete(H1):
            return False
        inventory = payload(c.RUN_ROOT / 'H1_resync/source_inventory.json')
        return bool(inventory.get('canonical_rows')) and not any(row.get('schema_error') for row in inventory.get('sources', []))

    def append_h1_revision(self) -> None:
        evolution = c.REPO / 'mypaper/code/code_evolution.md'
        marker = '<!-- cycle09-stage3-h1-resync-v2 -->'
        existing = evolution.read_text(encoding='utf-8') if evolution.is_file() else ''
        if marker in existing:
            return
        text = existing.rstrip() + (
            f'\n\n---\n\n{marker}\n\n'
            '## Cycle 09 Stage3 H1 corrected raw handoff\n\n'
            'Rebuilt H1 after correcting the retained Qwen R4 schema mapping '
            'and Llama source filename. The prior derived package is archived under '
            '`/root/autodl-tmp/cycle09_stage3_followup/archive/`; the canonical H1 '
            'paths now point to the corrected raw package.\n'
        )
        c.atomic_text(evolution, text + '\n')

    def core(self) -> None:
        # The active Q1 package owns GPU0.  It must complete before alpha=.5
        # step320 enters any cross-arm table.
        self.wait_external('H0_active_endpoint', H0_STATUS, H0_PID)

        self.archive_h1_v1()
        if not self.h1_is_corrected():
            self.run('H1_resync_v2', [PYTHON, str(SCRIPTS / 'cycle09_stage3_resync.py'), '--phase', 'finalize'])
        h1_handoff = c.MINI / 'stage3_H1_handoff_manifest.json'
        if not complete(h1_handoff):
            self.run('H1_handoff_v2', [PYTHON, str(SCRIPTS / 'cycle09_stage3_handoff.py'), '--gate', 'H1'])
        self.append_h1_revision()

        # T-PK is already complete on the retained native-Qwen BF16 objects.
        if not complete(TPK_Q, delta_mode='bf16_merged_minus_base'):
            self.run('H2_tpk_qwen_native', [
                PYTHON, str(SCRIPTS / 'cycle09_stage3_tpk.py'), '--phase', 'run',
                '--family', 'qwen3_4b', '--arms', 'opd,sft,offkd,seqkd,alpha05',
                '--steps', '5,20,40,80,160,320', '--layers', '18', '--modules', 'all',
                '--fractions', '0.01,0.025,0.05,0.10,0.25',
                '--delta-mode', 'bf16_merged_minus_base', '--device', 'cuda:0',
            ], gpu=1)
        if not complete(TPK_Q, delta_mode='bf16_merged_minus_base'):
            raise RuntimeError(f'Qwen native T-PK did not validate: {TPK_Q}')

        # The two measurements have no data dependency, so fill both GPUs now.
        jobs: list[tuple[str, list[str], int | None, str | None]] = []
        if not complete(WHITE_Q):
            jobs.append(('H2_white_qwen_native', [
                PYTHON, str(SCRIPTS / 'cycle09_stage3_twhite.py'), '--family', 'qwen3_4b',
                '--arms', 'opd,sft,offkd,seqkd,alpha05', '--steps', '0,5,20,40,80,160,320',
                '--layer', '18', '--device', 'cuda:0',
            ], 0, None))
        if not complete(SUB_Q, delta_mode='bf16_merged_minus_base'):
            jobs.append(('H2_sub_qwen_native', [
                PYTHON, str(SCRIPTS / 'cycle09_stage3_tsub.py'), '--family', 'qwen3_4b',
                '--arms', 'opd,offkd', '--steps', '5,20,40,80,160,320', '--layer', '18',
                '--probes', 'E_ood', '--rank-fraction', '0.05',
                '--delta-mode', 'bf16_merged_minus_base', '--device', 'cuda:0',
            ], 1, None))
        if jobs:
            self.run_parallel(jobs)

        if not complete(WHITE_Q):
            raise RuntimeError(f'Qwen native T-WHITE did not validate: {WHITE_Q}')
        white = payload(WHITE_Q)
        if int(white.get('schema_version', 0)) < 2:
            raise RuntimeError('Qwen T-WHITE was not produced by the explicit checkpoint-materialization implementation')
        if not complete(SUB_Q, delta_mode='bf16_merged_minus_base'):
            raise RuntimeError(f'Qwen native T-SUB did not validate: {SUB_Q}')

        if not complete(H2):
            self.run('H2_handoff', [PYTHON, str(SCRIPTS / 'cycle09_stage3_handoff.py'), '--gate', 'H2'])

        # The Llama partition is complete.  Build Qwen's counterpart in its
        # declared partition, then merge the two checked partitions locally.
        if not complete(PROBE):
            self.run('H2_probe_core_qwen_partition', [
                PYTHON, str(SCRIPTS / 'cycle09_stage3_probe_core.py'), '--families', 'qwen3_4b',
                '--phase', 'all', '--device', 'cuda:0',
            ], gpu=0, scope='partition_probe_qwen_20260723')
            self.run('H2_probe_core_merge', [
                PYTHON, str(SCRIPTS / 'cycle09_stage3_probe_core_merge.py'), '--phase', 'merge',
            ])
        if not complete(PROBE):
            raise RuntimeError(f'PROBE-CORE merge did not validate: {PROBE}')
        if not complete(H3):
            self.run('H3_handoff', [PYTHON, str(SCRIPTS / 'cycle09_stage3_handoff.py'), '--gate', 'H3'])

        tinc = c.RUN_ROOT / 'H4_increment/TINC_manifest.json'
        tbeh = c.RUN_ROOT / 'H4_increment/TBEH_manifest.json'
        if not (complete(tinc) and complete(tbeh)):
            self.run('H4_increment', [PYTHON, str(SCRIPTS / 'cycle09_stage3_increment.py'), '--phase', 'all', '--seed', '20260723'])
        m3 = c.RUN_ROOT / 'H4_increment/M3_audit_manifest.json'
        if not complete(m3):
            self.run('H4_m3_raw_audit', [PYTHON, str(SCRIPTS / 'cycle09_stage3_m3_audit.py'), '--phase', 'all'])
        if not complete(H4):
            self.run('H4_handoff', [PYTHON, str(SCRIPTS / 'cycle09_stage3_handoff.py'), '--gate', 'H4'])

        # Explicit user authorization: attach the frozen-self training/control
        # after the H4 package is frozen.  Its own launcher verifies all seven
        # native checkpoints and then produces behavior, geometry, and H5.
        if not complete(H5_FROZEN):
            self.run('H5_frozen_self_train_and_measure', [
                PYTHON, str(SCRIPTS / 'cycle09_stage3_frozen_self.py'), '--phase', 'all',
                '--gate-file', str(H4),
            ])
        if not complete(H5_FROZEN):
            raise RuntimeError(f'H5 frozen-self did not validate: {H5_FROZEN}')
        if not complete(H5):
            self.run('H5_handoff', [PYTHON, str(SCRIPTS / 'cycle09_stage3_handoff.py'), '--gate', 'H5'])

        self.state['state'] = 'complete_H5_awaiting_theory_go'
        self.state['completed_utc'] = c.utc_now()
        self.state['next_action'] = 'No H6 mediator or supplemental task is launched automatically.'
        self.write()


def detached() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    existing = pid_alive(PID_FILE)
    if existing:
        raise RuntimeError(f'full supervisor already running: pid={PID_FILE.read_text().strip()}')
    log = LOGS / 'supervisor.log'
    LOGS.mkdir(parents=True, exist_ok=True)
    with log.open('ab', buffering=0) as handle:
        process = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--run'], cwd=c.REPO, start_new_session=True, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
    c.atomic_text(PID_FILE, f'{process.pid}\n')
    c.atomic_json(STATUS, {'schema_version': 1, 'state': 'detached_waiting_for_H0', 'pid': process.pid, 'auto_shutdown': False, 'log': str(log), 'created_utc': c.utc_now()})
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--detach', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(payload(STATUS), indent=2)); return
    if args.detach:
        print(json.dumps({'pid': detached(), 'status': str(STATUS), 'auto_shutdown': False}, indent=2)); return
    c.atomic_text(PID_FILE, f'{os.getpid()}\n')
    supervisor = Supervisor()
    try:
        supervisor.core()
    except BaseException as error:
        if isinstance(error, SystemExit) and str(error).startswith('stop requested'):
            raise
        supervisor.state['state'] = 'failed'
        supervisor.state['failure'] = f'{type(error).__name__}: {error}'
        supervisor.write()
        raise


if __name__ == '__main__':
    main()
