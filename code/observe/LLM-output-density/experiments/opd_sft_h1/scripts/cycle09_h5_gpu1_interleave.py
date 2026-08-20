#!/usr/bin/env python3
"""Fill long H5 GPU1 geometry waits with bounded, reusable MMLU-Pro cells.

This is deliberately conservative: it only runs while the overlap geometry
worker is waiting for step 160 (or has completed), never starts after step 220,
and shares the slot lock with geometry. It therefore cannot leave a long
generation job running into terminal two-GPU H5 postprocessing.
"""

from __future__ import annotations

import argparse
import fcntl
import time

import cycle09_stage3_followup_common as c
import cycle09_stage3_frozen_self_postprocess as post


ROOT = c.scoped_run('H5_frozen_self')
CHECKPOINTS = ROOT / 'checkpoints'
TRAINING = ROOT / 'training_manifest.json'
GEOMETRY_STATUS = ROOT / 'H5_geometry_overlap_status.json'
STATUS = ROOT / 'H5_gpu1_interleave_status.json'
SCHEDULER_LOCK = ROOT / 'H5_gpu1_interleave_scheduler.lock'
QUEUE = ((5, 'mmlu_pro'), (20, 'mmlu_pro'), (40, 'mmlu_pro'), (80, 'mmlu_pro'))
QUIESCE_AT_STEP = 220


def latest_checkpoint() -> int:
    try:
        return int((CHECKPOINTS / 'latest_checkpointed_iteration.txt').read_text(encoding='utf-8').strip())
    except (FileNotFoundError, ValueError):
        return 0


def complete(step: int, component: str) -> bool:
    path = ROOT / 'behavior' / 'formal' / 'frozen_self' / f'step_{step:03d}' / f'{component}_overlap_manifest.json'
    return c.read_json(path, {}).get('status') == 'complete'


def write_status(state: str, **extra: object) -> None:
    completed = [f'{step}:{component}' for step, component in QUEUE if complete(step, component)]
    c.atomic_json(STATUS, {
        'schema_version': 1,
        'status': state,
        'task': 'H5 GPU1 bounded overlap evaluation',
        'queue': [f'{step}:{component}' for step, component in QUEUE],
        'completed': completed,
        'latest_checkpoint': latest_checkpoint(),
        'quiesce_at_step': QUIESCE_AT_STEP,
        **extra,
        'updated_utc': c.utc_now(),
    })


def slot_open() -> bool:
    geometry = c.read_json(GEOMETRY_STATUS, {})
    if geometry.get('status') == 'complete':
        return True
    if geometry.get('status') != 'waiting_for_checkpoint':
        return False
    waiting = int(geometry.get('waiting_for_step', 0))
    return waiting == 160 and latest_checkpoint() < waiting


def run(poll_seconds: int) -> None:
    SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SCHEDULER_LOCK.open('w', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another H5 GPU1 interleave scheduler is already active') from error
        while True:
            if c.read_json(TRAINING, {}).get('status') == 'complete':
                write_status('complete', reason='training_complete')
                return
            latest = latest_checkpoint()
            pending = next(((step, component) for step, component in QUEUE if not complete(step, component)), None)
            if pending is None:
                write_status('complete', reason='queue_complete')
                return
            if latest >= QUIESCE_AT_STEP:
                write_status('complete', reason='terminal_quiesce', pending=f'{pending[0]}:{pending[1]}')
                return
            if not slot_open():
                write_status('waiting_for_long_geometry_gap', pending=f'{pending[0]}:{pending[1]}')
                time.sleep(poll_seconds)
                continue
            step, component = pending
            write_status('running', active=f'{step}:{component}')
            # The geometry watcher uses this lock only while measuring a durable
            # checkpoint. It waits if a bounded component is already active.
            with post.gpu1_work_slot():
                if not slot_open() and c.read_json(GEOMETRY_STATUS, {}).get('status') != 'complete':
                    continue
                post.behavior_component(step, component, 'cuda:0')
            write_status('between_cells', completed_cell=f'{step}:{component}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    try:
        run(args.poll_seconds)
    except BaseException as error:
        write_status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
