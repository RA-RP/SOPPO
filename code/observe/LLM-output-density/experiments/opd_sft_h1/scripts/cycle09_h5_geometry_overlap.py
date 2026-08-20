#!/usr/bin/env python3
from __future__ import annotations
"""Run H5 geometry as each nonterminal checkpoint becomes durable."""

import argparse
import fcntl
import json
import time
from pathlib import Path

import cycle09_stage3_followup_common as c
import cycle09_stage3_frozen_self_postprocess as post

ROOT = c.scoped_run('H5_frozen_self')
CHECKPOINTS = ROOT / 'checkpoints'
TRAINING = ROOT / 'training_manifest.json'
STATUS = ROOT / 'H5_geometry_overlap_status.json'
LOCK = ROOT / 'H5_geometry_overlap_gpu1.lock'
STEPS = (5, 20, 40, 80, 160)

def latest_checkpoint() -> int:
    path = CHECKPOINTS / 'latest_checkpointed_iteration.txt'
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except (FileNotFoundError, ValueError):
        return 0

def checkpoint_ready(step: int) -> bool:
    actor = CHECKPOINTS / f'global_step_{step}' / 'actor'
    return latest_checkpoint() >= step and len(list(actor.glob('model_world_size_*_rank_*.pt'))) == 1

def write_status(state: str, completed: list[int], **extra: object) -> None:
    c.atomic_json(STATUS, {
        'schema_version': 1,
        'status': state,
        'task': 'H5 checkpoint-triggered GPU1 geometry overlap',
        'device': 'cuda:0',
        'steps': list(STEPS),
        'completed_steps': completed,
        'latest_checkpoint': latest_checkpoint(),
        **extra,
        'updated_utc': c.utc_now(),
    })

def run(poll_seconds: int) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another H5 geometry overlap worker already holds the GPU1 lock') from error
        completed: list[int] = []
        write_status('running', completed, phase='base_reference')
        with post.gpu1_work_slot():
            post.geometry_reference('cuda:0')
        for step in STEPS:
            while not checkpoint_ready(step):
                if c.read_json(TRAINING, {}).get('status') == 'complete':
                    raise RuntimeError(f'H5 training completed before durable checkpoint {step}')
                write_status('waiting_for_checkpoint', completed, waiting_for_step=step)
                time.sleep(poll_seconds)
            write_status('running', completed, phase='geometry_cell', active_step=step)
            with post.gpu1_work_slot():
                post.geometry_cell(step, 'cuda:0')
            completed.append(step)
            write_status('running', completed, phase='checkpoint_wait')
        write_status('complete', completed)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    try:
        run(args.poll_seconds)
    except BaseException as error:
        existing = c.read_json(STATUS, {})
        write_status('failed', list(existing.get('completed_steps', [])), error=repr(error))
        raise

if __name__ == '__main__':
    main()
