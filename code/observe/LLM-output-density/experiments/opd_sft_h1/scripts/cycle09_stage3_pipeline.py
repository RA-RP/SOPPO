#!/usr/bin/env python3
"""Declarative, fail-closed execution DAG for Cycle09 Stage3 follow-up.

This file has no import-time side effects.  It only executes child processes
when called with both ``--execute`` and an explicit gate.  In particular, it
never observes Q1 completion and starts work on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cycle09_stage3_followup_common as c

PYTHON = '/root/miniconda3/envs/density/bin/python'
SCRIPTS = c.SCRIPT_DIR
PIPELINE_ROOT = c.RUN_ROOT / 'pipeline'
PLAN_PATH = PIPELINE_ROOT / 'stage3_pipeline_plan.json'
STATUS_PATH = PIPELINE_ROOT / 'stage3_pipeline_status.json'


@dataclass(frozen=True)
class Task:
    name: str
    gate: str
    deps: tuple[str, ...]
    resource: str
    lane: str
    script: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]
    description: str
    optional: bool = False


def command(script: str, *argv: str) -> tuple[str, ...]:
    return (PYTHON, str(SCRIPTS / script), *argv)


# Formal defaults are deliberately explicit.  They can be changed only by
# editing this contract or passing a separately recorded revision, not by an
# implicit shell environment.
TASKS: tuple[Task, ...] = (
    Task('S0_contracts', 'H0', (), 'cpu', 'cpu', 'cycle09_stage3_contracts.py',
         ('--phase', 'write'),
         (str(c.RUN_ROOT / 'contracts/contracts_manifest.json'),),
         'Write immutable input contracts only; no model, rollout, or smoke execution.'),
    Task('H0_q1_validate', 'H0', (), 'cpu', 'cpu', 'cycle09_q1_stageb_postprocess.py',
         ('--phase', 'validate'),
         (str(c.MINI / 'qwen_alpha05_stage_b_training_manifest.json'),),
         'Validate the current alpha=.5 checkpoint; does not train or recompute cells.'),
    Task('H0_q1_endpoint_package', 'H0', ('H0_q1_validate',), 'cpu', 'cpu',
         'cycle09_q1_stageb_handoff.py', (),
         (str(c.MINI / 'qwen_alpha05_stage_b_320_handoff_manifest.json'),),
         'Fail-closed validator for export, behavior, six-probe geometry, and support outputs.'),
    Task('H0_q1_handoff', 'H0', ('H0_q1_endpoint_package',), 'cpu', 'cpu',
         'cycle09_stage3_handoff.py',
         ('--gate', 'H0'),
         (str(c.MINI / 'mini_stage3_H0_theory_handoff.md'), str(c.MINI / 'stage3_H0_handoff_manifest.json')),
         'Freeze Q1 endpoint raw inventory and provenance.'),
    Task('H1_resync', 'H1', (), 'cpu', 'cpu', 'cycle09_stage3_resync.py',
         ('--phase', 'finalize'),
         (str(c.RUN_ROOT / 'H1_resync/H1_resync_manifest.json'),),
         'Canonical source inventory, naming audit inputs, and raw A/G DiD.'),
    Task('H1_support_inventory', 'H1', ('S0_contracts',), 'cpu', 'cpu',
         'cycle09_stage3_support_inventory.py', ('--phase', 'write'),
         (str(c.RUN_ROOT / 'contracts/support_inputs.json'),),
         'Freeze present training-support sources, shared-source groups, and explicit gaps.'),
    Task('H1_support', 'H1', ('H1_support_inventory',), 'cpu', 'cpu', 'cycle09_stage3_support.py',
         ('--phase', 'run', '--input-manifest', str(c.RUN_ROOT / 'contracts/support_inputs.json')),
         (str(c.RUN_ROOT / 'H1_support/T_SUPPORT_manifest.json'),),
         'Support and mediator statistics from frozen existing rollout manifests.'),
    Task('H1_handoff', 'H1', ('H1_resync', 'H1_support'), 'cpu', 'cpu', 'cycle09_stage3_handoff.py',
         ('--gate', 'H1'),
         (str(c.MINI / 'mini_stage3_H1_theory_handoff.md'), str(c.MINI / 'stage3_H1_handoff_manifest.json')),
         'Freeze reconciled cross-family baseline tables.'),
    Task('H2_tpk_qwen', 'H2', ('H0_q1_handoff',), 'gpu', 'gpu0', 'cycle09_stage3_tpk.py',
         ('--phase', 'run', '--family', 'qwen3_4b', '--arms', 'opd,sft,offkd,seqkd,alpha05',
          '--steps', '5,20,40,80,160,320', '--layers', '18', '--modules', 'all',
          '--fractions', '0.01,0.025,0.05,0.10,0.25', '--delta-mode', 'bf16_merged_minus_base', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_tpk/T_PK_qwen3_4b_manifest.json'),),
         'Joint source-principal p_k for the explicitly labelled final BF16 deployment object.'),
    Task('H2_tpk_llama', 'H2', (), 'gpu', 'gpu1', 'cycle09_stage3_tpk.py',
         ('--phase', 'run', '--family', 'llama3_2_3b', '--arms', 'opd,sft,offkd,seqkd',
          '--steps', '5,20,40,80,160,320', '--layers', '14', '--modules', 'all',
          '--fractions', '0.01,0.025,0.05,0.10,0.25', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_tpk/T_PK_llama3_2_3b_manifest.json'),),
         'Llama strict joint source-principal p_k.'),
    Task('H2_white_qwen', 'H2', ('H0_q1_handoff', 'S0_contracts'), 'gpu', 'gpu0', 'cycle09_stage3_twhite.py',
         ('--family', 'qwen3_4b', '--arms', 'opd,sft,offkd,seqkd,alpha05',
          '--steps', '0,5,20,40,80,160,320', '--layer', '18', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_white/T_WHITE_qwen3_4b_manifest.json'),),
         'Weight-only, fixed S_D0, and per-checkpoint S_Dt r_epsilon ablation.'),
    Task('H2_white_llama', 'H2', ('S0_contracts',), 'gpu', 'gpu1', 'cycle09_stage3_twhite.py',
         ('--family', 'llama3_2_3b', '--arms', 'opd,sft,offkd,seqkd',
          '--steps', '0,5,20,40,80,160,320', '--layer', '14', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_white/T_WHITE_llama3_2_3b_manifest.json'),),
         'Llama whitening-condition ablation.'),
    Task('H2_sub_qwen', 'H2', ('H0_q1_handoff',), 'gpu', 'gpu0', 'cycle09_stage3_tsub.py',
        ('--family', 'qwen3_4b', '--arms', 'opd,offkd', '--steps', '5,20,40,80,160,320',
          '--layer', '18', '--delta-mode', 'bf16_merged_minus_base', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_sub/T_SUB_qwen3_4b_manifest.json'),),
         'OPD/off-KD functional subspace: common output coordinates and fixed input whitening.'),
    Task('H2_sub_llama', 'H2', (), 'gpu', 'gpu1', 'cycle09_stage3_tsub.py',
         ('--family', 'llama3_2_3b', '--arms', 'opd,offkd', '--steps', '5,20,40,80,160,320',
          '--layer', '14', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_sub/T_SUB_llama3_2_3b_manifest.json'),),
         'Llama OPD/off-KD functional subspace.'),
    Task('H2_probe_core', 'H2', ('H0_q1_handoff',), 'gpu', 'gpu0', 'cycle09_stage3_probe_core.py',
         ('--families', 'qwen3_4b,llama3_2_3b', '--phase', 'all', '--device', 'cuda:0'),
         (str(c.RUN_ROOT / 'H2_probe_core/PROBE_CORE_manifest.json'),),
         'Exact MATH500-aligned E_math and alpha=.5 E_aime24 backfill; historical 32-item probe renamed E_mathHeld.'),
    Task('H2_handoff', 'H2', ('H2_tpk_qwen', 'H2_tpk_llama', 'H2_white_qwen', 'H2_white_llama'), 'cpu', 'cpu', 'cycle09_stage3_handoff.py',
         ('--gate', 'H2'),
         (str(c.MINI / 'mini_stage3_H2_theory_handoff.md'), str(c.MINI / 'stage3_H2_handoff_manifest.json')),
         'Freeze metric-validity package before geometry closure.'),
    Task('H3_handoff', 'H3', ('H2_sub_qwen', 'H2_sub_llama', 'H2_probe_core'), 'cpu', 'cpu', 'cycle09_stage3_handoff.py',
         ('--gate', 'H3'),
         (str(c.MINI / 'mini_stage3_H3_theory_handoff.md'), str(c.MINI / 'stage3_H3_handoff_manifest.json')),
         'Freeze geometry closure package.'),
    Task('H4_increment', 'H4', ('H2_handoff', 'H3_handoff', 'H1_handoff'), 'cpu', 'cpu', 'cycle09_stage3_increment.py',
         ('--phase', 'all', '--seed', '20260723'),
         (str(c.RUN_ROOT / 'H4_increment/TINC_manifest.json'), str(c.RUN_ROOT / 'H4_increment/TBEH_manifest.json')),
         'Frozen-fold T-INC/T-BEH under the corrected geometry schema.'),
    Task('H4_m3_raw_audit', 'H4', ('H4_increment',), 'cpu', 'cpu', 'cycle09_stage3_m3_audit.py',
         ('--phase', 'all'),
         (str(c.RUN_ROOT / 'H4_increment/M3_audit_manifest.json'),),
         'Epsilon/sample-count inventory and matched timing coverage; no imputation or timing fit.'),
    Task('H4_handoff', 'H4', ('H4_increment', 'H4_m3_raw_audit'), 'cpu', 'cpu', 'cycle09_stage3_handoff.py',
         ('--gate', 'H4'),
         (str(c.MINI / 'mini_stage3_H4_theory_handoff.md'), str(c.MINI / 'stage3_H4_handoff_manifest.json')),
         'Claim-ready core handoff; prerequisite for frozen-self.'),
    Task('H5_frozen_self', 'H5', ('H4_handoff',), 'gpu', 'two_gpu', 'cycle09_stage3_frozen_self.py',
         ('--phase', 'all', '--gate-file', str(c.MINI / 'stage3_H4_handoff_manifest.json')),
         (str(c.RUN_ROOT / 'H5_frozen_self/FROZEN_SELF_manifest.json'),),
         'GO-gated frozenSelf0-KD rollout, labels, training, landmarks, and total-effect tables.', optional=True),
    Task('H5_handoff', 'H5', ('H5_frozen_self',), 'cpu', 'cpu', 'cycle09_stage3_handoff.py',
         ('--gate', 'H5'),
         (str(c.MINI / 'mini_stage3_H5_theory_handoff.md'), str(c.MINI / 'stage3_H5_handoff_manifest.json')),
         'Frozen-self total-effect control.', optional=True),
    Task('H6_mediator', 'H6', ('H5_handoff',), 'cpu', 'cpu', 'cycle09_stage3_mediator.py',
         ('--phase', 'all', '--gate-file', str(c.MINI / 'stage3_H5_handoff_manifest.json')),
         (str(c.RUN_ROOT / 'H6_mediator/MEDIATOR_manifest.json'),),
         'GO-gated mediator association/reweighting; no total-effect replacement.', optional=True),
    Task('H6_handoff', 'H6', ('H6_mediator',), 'cpu', 'cpu', 'cycle09_stage3_handoff.py',
         ('--gate', 'H6'),
         (str(c.MINI / 'mini_stage3_H6_theory_handoff.md'), str(c.MINI / 'stage3_H6_handoff_manifest.json')),
         'Mechanism boundary package.', optional=True),
)


def task_index() -> dict[str, Task]:
    return {task.name: task for task in TASKS}


def closure(targets: list[str]) -> list[Task]:
    index = task_index(); selected: set[str] = set()
    def visit(name: str) -> None:
        if name in selected:
            return
        if name not in index:
            raise ValueError(f'unknown task {name}')
        for dependency in index[name].deps:
            visit(dependency)
        selected.add(name)
    for target in targets:
        visit(target)
    return [task for task in TASKS if task.name in selected]


def render(tasks: list[Task]) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'created_utc': c.utc_now(),
        'execution_policy': {
            'default': 'render_only',
            'requires_execute_flag': True,
            'requires_gate': True,
            'never_auto_starts_after_q1': True,
            'q1_current_job_is_untouched': True,
            'frozen_self_requires_H4': True,
        },
        'tasks': [asdict(task) | {'command': list(command(task.script, *task.argv))} for task in tasks],
    }


def output_complete(task: Task) -> bool:
    for value in task.outputs:
        path = Path(value)
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        if path.suffix == '.json':
            try:
                payload = c.read_json(path, {})
            except (OSError, ValueError, TypeError):
                return False
            status = str(payload.get('status', ''))
            if status and not (status.startswith('complete') or status.startswith('frozen')):
                return False
    return True


def execute(tasks: list[Task], gate: str) -> dict[str, Any]:
    """Run the selected DAG with one worker per declared GPU lane.

    This is deliberately dormant unless the caller supplied --execute.  It
    schedules only nodes whose declared artifact dependencies are complete and
    never multiplexes two processes onto the same physical GPU lane.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    permitted = [task for task in tasks if task.gate <= gate]
    pending = {task.name: task for task in permitted if not output_complete(task)}
    completed = {task.name for task in permitted if output_complete(task)}
    status: dict[str, Any] = {'started_utc': c.utc_now(), 'gate': gate, 'tasks': []}

    def launch(task: Task) -> dict[str, Any]:
        env = os.environ.copy()
        if task.lane == 'gpu0': env['CUDA_VISIBLE_DEVICES'] = '0'
        elif task.lane == 'gpu1': env['CUDA_VISIBLE_DEVICES'] = '1'
        elif task.lane == 'two_gpu': env['CUDA_VISIBLE_DEVICES'] = '0,1'
        process = subprocess.run(command(task.script, *task.argv), cwd=c.REPO, env=env)
        if process.returncode != 0:
            raise RuntimeError(f'{task.name} failed rc={process.returncode}')
        if not output_complete(task):
            raise RuntimeError(f'{task.name} returned without every declared output')
        return {'name': task.name, 'status': 'complete', 'lane': task.lane}

    while pending:
        ready = [task for task in pending.values() if set(task.deps).issubset(completed)]
        if not ready:
            blocked = {name: [dep for dep in task.deps if dep not in completed] for name, task in pending.items()}
            raise RuntimeError(f'pipeline dependency deadlock: {blocked}')
        # One task per resource lane per wave.  CPU tasks are independent and
        # are grouped under their own lane; two_gpu excludes every other GPU task.
        wave: list[Task] = []
        occupied: set[str] = set()
        for task in ready:
            claims = {'gpu0', 'gpu1'} if task.lane == 'two_gpu' else {task.lane}
            if claims.intersection(occupied):
                continue
            wave.append(task); occupied.update(claims)
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {pool.submit(launch, task): task for task in wave}
            for future in as_completed(futures):
                task = futures[future]
                row = future.result()
                completed.add(task.name); pending.pop(task.name); status['tasks'].append(row)
    status['status'] = 'complete'; status['finished_utc'] = c.utc_now()
    c.atomic_json(STATUS_PATH, status)
    return status

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', default='H4_handoff', help='comma-separated task names')
    parser.add_argument('--write-plan', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gate', choices=('H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'))
    args = parser.parse_args()
    tasks = closure([item.strip() for item in args.targets.split(',') if item.strip()])
    plan = render(tasks)
    if args.write_plan:
        c.atomic_json(PLAN_PATH, plan)
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if not args.gate:
        raise SystemExit('--execute requires --gate')
    print(json.dumps(execute(tasks, args.gate), indent=2))

if __name__ == '__main__':
    main()
