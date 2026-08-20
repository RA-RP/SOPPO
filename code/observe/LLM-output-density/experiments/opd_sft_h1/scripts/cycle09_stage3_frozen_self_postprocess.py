#!/usr/bin/env python3
"""Export and measure the approved H5 frozenSelf0-KD trajectory.

The trainer keeps native VERL actor checkpoints.  This postprocessor exports
their LoRA adapters, creates auditable merged evaluation models, reuses the
established Llama behavior/geometry code under a distinct arm namespace, and
writes raw total-effect tables.  It contains no claim adjudication.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as b3
import cycle09_stage3_followup_common as c


ARM = 'frozen_self'
STEPS = (0, 5, 20, 40, 80, 160, 320)
ROOT = c.scoped_run('H5_frozen_self')
CHECKPOINTS = ROOT / 'checkpoints'
ADAPTERS = ROOT / 'adapters'
MERGED = ROOT / 'merged'
EXPORT_MANIFEST = ROOT / 'H5_export_manifest.json'
BEHAVIOR_MANIFEST = ROOT / 'H5_behavior_manifest.json'
GEOMETRY_MANIFEST = ROOT / 'H5_geometry_manifest.json'
TOTAL_EFFECT = ROOT / 'frozen_self_total_effect.csv'
MANIFEST = ROOT / 'FROZEN_SELF_manifest.json'
GPU1_WORK_SLOT = ROOT / 'H5_gpu1_work_slot.lock'


@contextmanager
def gpu1_work_slot():
    """Serialize transient GPU1 work without holding a lock while training runs."""
    GPU1_WORK_SLOT.parent.mkdir(parents=True, exist_ok=True)
    with GPU1_WORK_SLOT.open('w', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def behavior_device(device: str):
    """Bind behavior subprocesses to the worker's physical GPU."""
    if not device.startswith('cuda:'):
        raise ValueError(f'expected CUDA device, got {device!r}')
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':', 1)[1]
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous


def actor_path(step: int) -> Path:
    actor = CHECKPOINTS / f'global_step_{step}' / 'actor'
    shards = list(actor.glob('model_world_size_*_rank_*.pt'))
    if len(shards) != 1:
        raise FileNotFoundError(f'expected one H5 actor shard at step={step}: {actor}')
    return actor


def adapter_complete(path: Path) -> bool:
    return (path / 'adapter_config.json').is_file() and (path / 'adapter_model.safetensors').is_file() and (path / 'adapter_model.safetensors').stat().st_size > 0


def merged_complete(path: Path) -> bool:
    return b3.model_check(path)['complete']


def find_adapter(root: Path) -> Path:
    candidates = sorted(path.parent for path in root.rglob('adapter_config.json') if adapter_complete(path.parent))
    if len(candidates) != 1:
        raise RuntimeError(f'expected exactly one merger adapter under {root}; found={candidates}')
    return candidates[0]


def export_one(step: int) -> dict[str, Any]:
    target_adapter = ADAPTERS / f'checkpoint-{step:06d}'
    if not adapter_complete(target_adapter):
        work = ROOT / 'merger_work' / f'step_{step:03d}'
        if work.exists():
            shutil.rmtree(work)
        work.parent.mkdir(parents=True, exist_ok=True)
        command = [str(b3.VERL_PYTHON), '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp', '--local_dir', str(actor_path(step)), '--target_dir', str(work)]
        result = subprocess.run(command, cwd=c.REPO)
        if result.returncode:
            raise RuntimeError(f'H5 model_merger failed for step={step}: rc={result.returncode}')
        source = find_adapter(work)
        temporary = target_adapter.with_name(target_adapter.name + '.tmp')
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        target_adapter.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target_adapter)
        shutil.rmtree(work)
    config = json.loads((target_adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    if int(config.get('r', -1)) != 32 or int(config.get('lora_alpha', -1)) != 64:
        raise RuntimeError(f'H5 LoRA contract drift at step={step}: {config.get("r")}/{config.get("lora_alpha")}')
    target_model = MERGED / ARM / f'step_{step:03d}'
    if not merged_complete(target_model):
        from run_opd_minimal_closure import merge_lora_adapter
        merge_lora_adapter(b3.LLAMA_STUDENT, target_adapter, target_model)
        b3.install_llama_chat_template(target_model)
        gc.collect()
    check = b3.model_check(target_model)
    if not check['complete']:
        raise RuntimeError(f'H5 merged model incomplete at step={step}: {check["error"]}')
    return {
        'step': step,
        'actor': c.artifact(next(actor_path(step).glob('model_world_size_*_rank_*.pt'))),
        'adapter': c.artifact(target_adapter / 'adapter_model.safetensors'),
        'adapter_config': c.artifact(target_adapter / 'adapter_config.json'),
        'merged': check,
        'delta_w_source': 'PEFT adapter BA fp32; merged model used only for evaluation/activation geometry',
    }


def export() -> dict[str, Any]:
    rows = [export_one(step) for step in STEPS if step]
    payload = {'schema_version': 1, 'status': 'complete', 'task': 'H5 frozenSelf0-KD native checkpoint export', 'arm': ARM, 'steps': list(STEPS), 'rows': rows, 'created_utc': c.utc_now()}
    c.atomic_json(EXPORT_MANIFEST, payload)
    return payload


def patched_context():
    """Temporarily point the reusable Llama evaluators at this fifth arm."""
    import cycle09_llama_model_export as model_export
    original = {
        'arms': b3.ARMS,
        'steps': b3.MEASURED_CHECKPOINTS,
        'merged_root': model_export.MERGED_ROOT,
    }
    b3.ARMS = (*b3.ARMS, ARM) if ARM not in b3.ARMS else b3.ARMS
    b3.MEASURED_CHECKPOINTS = STEPS
    model_export.MERGED_ROOT = MERGED
    return original


def restore_context(original: dict[str, Any]) -> None:
    import cycle09_llama_model_export as model_export
    b3.ARMS = original['arms']; b3.MEASURED_CHECKPOINTS = original['steps']; model_export.MERGED_ROOT = original['merged_root']


def require_export() -> None:
    if not c.read_json(EXPORT_MANIFEST, {}).get('status') == 'complete':
        export()


def behavior_cell(step: int, device: str) -> dict[str, Any]:
    """Run one resumable behavior cell under the frozen-self arm namespace."""
    if step not in STEPS:
        raise ValueError(f'invalid frozen-self behavior step={step}')
    if step:
        export_one(step)
    import cycle09_llama_behavior as task
    original = patched_context()
    try:
        task.EVAL_ROOT = ROOT / 'behavior'
        arm = 'base' if step == 0 else ARM
        args = argparse.Namespace(arm=arm, step=step, smoke=False, gpu_mem=.85)
        with behavior_device(device):
            return task.run_cell(args)
    finally:
        restore_context(original)


def behavior_component(step: int, component: str, device: str, gpu_mem: float = .80) -> dict[str, Any]:
    """Run one reusable behavior component before the terminal H5 postprocess."""
    if step not in STEPS or step == 0:
        raise ValueError(f'invalid nonzero frozen-self behavior step={step}')
    if component not in {'mmlu_pro', 'ifeval', 'math500'}:
        raise ValueError(f'unsupported H5 behavior component={component}')
    export_one(step)
    import cycle09_llama_behavior as task
    original = patched_context()
    try:
        task.EVAL_ROOT = ROOT / 'behavior'
        arm = ARM
        root = task.cell_root(arm, step, False)
        model = task.model_path(arm, step)
        with behavior_device(device):
            if component == 'math500':
                output = task.run_math(arm, step, root, model, False, gpu_mem)
            else:
                output = task.run_lm_eval(component, step, root, model, False, gpu_mem)
        manifest = {
            'schema_version': 1,
            'status': 'complete',
            'task': 'H5 overlap behavior component',
            'arm': ARM,
            'step': step,
            'component': component,
            'device': device,
            'gpu_memory_utilization': gpu_mem,
            'output': c.artifact(output),
            'created_utc': c.utc_now(),
        }
        c.atomic_json(root / f'{component}_overlap_manifest.json', manifest)
        return manifest
    finally:
        restore_context(original)


def behavior_finalize(device: str) -> dict[str, Any]:
    require_export()
    import cycle09_llama_behavior as task
    original = patched_context()
    try:
        task.EVAL_ROOT = ROOT / 'behavior'
        payload = task.finalize((ARM,), STEPS, 'frozen_self')
        source = task.EVAL_ROOT / 'llama_frozen_self_behavior.csv'
        categories = task.EVAL_ROOT / 'llama_frozen_self_ifeval_categories.csv'
        target = ROOT / 'landmark_behavior.csv'; target_categories = ROOT / 'landmark_ifeval_categories.csv'
        shutil.copy2(source, target); shutil.copy2(categories, target_categories)
        result = {'schema_version': 1, 'status': 'complete', 'task': 'H5 frozenSelf landmark behavior', 'arm': ARM, 'steps': list(STEPS), 'source_manifest': payload, 'output': c.artifact(target), 'ifeval_categories': c.artifact(target_categories), 'device': device, 'created_utc': c.utc_now()}
        c.atomic_json(BEHAVIOR_MANIFEST, result)
        return result
    finally:
        restore_context(original)


def behavior(device: str) -> dict[str, Any]:
    for step in STEPS:
        behavior_cell(step, device)
    return behavior_finalize(device)


def geometry_reference(device: str) -> dict[str, Any]:
    """Create the shared base profiles once before any nonzero geometry cell."""
    import cycle09_llama_geometry as task
    original = patched_context()
    try:
        task.ROOT = ROOT / 'geometry'; task.CELL_ROOT = task.ROOT / 'cells'
        cached = c.read_json(task.base_cell_path(False), {})
        if cached.get('status') == 'complete':
            return cached
        args = argparse.Namespace(device=device, smoke=False, measurement_n=0, forward_batch_size=8, max_batch_tokens=16384, arm='base', step=0)
        return task.base_reference(args)
    finally:
        restore_context(original)


def geometry_cell(step: int, device: str) -> dict[str, Any]:
    if step not in STEPS or step == 0:
        raise ValueError(f'invalid frozen-self geometry step={step}')
    export_one(step)
    import cycle09_llama_geometry as task
    original = patched_context()
    try:
        task.ROOT = ROOT / 'geometry'; task.CELL_ROOT = task.ROOT / 'cells'
        if c.read_json(task.base_cell_path(False), {}).get('status') != 'complete':
            raise RuntimeError('H5 geometry base reference must complete before nonzero cells')
        args = argparse.Namespace(device=device, smoke=False, measurement_n=0, forward_batch_size=8, max_batch_tokens=16384, arm=ARM, step=step)
        return task.run_cell(args)
    finally:
        restore_context(original)


def geometry_finalize(device: str) -> dict[str, Any]:
    require_export()
    import cycle09_llama_geometry as task
    original = patched_context()
    try:
        task.ROOT = ROOT / 'geometry'; task.CELL_ROOT = task.ROOT / 'cells'
        payload = task.finalize((ARM,), STEPS, 'frozen_self')
        source = task.ROOT / 'llama_frozen_self_r_epsilon.csv'
        tails = task.ROOT / 'llama_frozen_self_tail_energy.csv'
        raw = task.ROOT / 'llama_frozen_self_raw_representation_suite.csv'
        target = ROOT / 'landmark_geometry.csv'; target_tails = ROOT / 'landmark_geometry_tail_energy.csv'; target_raw = ROOT / 'landmark_raw_representation.csv'
        shutil.copy2(source, target); shutil.copy2(tails, target_tails); shutil.copy2(raw, target_raw)
        result = {'schema_version': 1, 'status': 'complete', 'task': 'H5 frozenSelf landmark geometry', 'arm': ARM, 'steps': list(STEPS), 'source_manifest': payload, 'output': c.artifact(target), 'tails': c.artifact(target_tails), 'raw_representation': c.artifact(target_raw), 'device': device, 'created_utc': c.utc_now()}
        c.atomic_json(GEOMETRY_MANIFEST, result)
        return result
    finally:
        restore_context(original)


def geometry(device: str) -> dict[str, Any]:
    geometry_reference(device)
    for step in STEPS:
        if step:
            geometry_cell(step, device)
    return geometry_finalize(device)


def worker(plan: str, device: str) -> dict[str, Any]:
    """Execute a fixed, balanced sequence of disjoint H5 cells on one GPU."""
    actions = [item.strip() for item in plan.split(',') if item.strip()]
    if not actions:
        raise ValueError('H5 postprocess worker plan is empty')
    completed: list[dict[str, Any]] = []
    for action in actions:
        if action == 'geometry_reference':
            result = geometry_reference(device)
        else:
            try:
                kind, raw_step = action.split(':', 1)
                step = int(raw_step)
            except ValueError as error:
                raise ValueError(f'invalid H5 postprocess worker action={action!r}') from error
            if kind == 'behavior':
                result = behavior_cell(step, device)
            elif kind == 'geometry':
                result = geometry_cell(step, device)
            else:
                raise ValueError(f'unknown H5 postprocess worker action={action!r}')
        completed.append({'action': action, 'status': result.get('status')})
    return {'schema_version': 1, 'status': 'complete', 'task': 'H5 balanced postprocess worker', 'device': device, 'plan': actions, 'completed': completed, 'created_utc': c.utc_now()}


def total_effect() -> dict[str, Any]:
    behavior_path = ROOT / 'landmark_behavior.csv'; geometry_path = ROOT / 'landmark_geometry.csv'
    if not behavior_path.is_file() or not geometry_path.is_file():
        raise FileNotFoundError('H5 behavior and geometry must complete before total-effect summary')
    behavior = pd.read_csv(behavior_path)
    opd_behavior = pd.read_csv(c.MINI / 'llama_early_320_behavior.csv')
    columns = {'math500': ('accuracy',), 'mmlu_pro': ('strict_accuracy', 'flexible_accuracy', 'extract_failure_rate'), 'ifeval': ('prompt_strict_accuracy', 'instruction_strict_accuracy')}
    rows: list[dict[str, Any]] = []
    for task, values in columns.items():
        left = opd_behavior[(opd_behavior.arm == 'opd') & (opd_behavior.task == task)]
        right = behavior[(behavior.arm == ARM) & (behavior.task == task)]
        for column in values:
            if column not in left or column not in right:
                continue
            joined = left[['step', column]].merge(right[['step', column]], on='step', suffixes=('_opd', '_frozen_self'))
            for _, value in joined.iterrows():
                rows.append({'kind': 'behavior', 'task': task, 'probe': None, 'step': int(value.step), 'metric': column, 'opd': float(value[f'{column}_opd']), 'frozen_self': float(value[f'{column}_frozen_self']), 'opd_minus_frozen_self': float(value[f'{column}_opd'] - value[f'{column}_frozen_self'])})
    geometry = pd.read_csv(geometry_path)
    opd_geometry = pd.read_csv(c.MINI / 'llama_early_320_r_epsilon.csv')
    left = opd_geometry[(opd_geometry.arm == 'opd') & (opd_geometry.layer == 14) & (opd_geometry.epsilon == .05)].groupby(['step', 'probe'], as_index=False).agg(opd=('delta_from_base', 'mean'))
    right = geometry[(geometry.arm == ARM) & (geometry.layer == 14) & (geometry.epsilon == .05)].groupby(['step', 'probe'], as_index=False).agg(frozen_self=('delta_from_base', 'mean'))
    for _, value in left.merge(right, on=['step', 'probe']).iterrows():
        rows.append({'kind': 'geometry', 'task': None, 'probe': value.probe, 'step': int(value.step), 'metric': 'L14_epsilon05_seven_module_mean_delta_from_base', 'opd': float(value.opd), 'frozen_self': float(value.frozen_self), 'opd_minus_frozen_self': float(value.opd - value.frozen_self)})
    fields = ['kind', 'task', 'probe', 'step', 'metric', 'opd', 'frozen_self', 'opd_minus_frozen_self']
    c.atomic_csv(TOTAL_EFFECT, rows, fields)
    support = ROOT / 'frozen_support_manifest.json'
    result = {'schema_version': 1, 'status': 'complete', 'task': 'H5 frozenSelf0-KD total-effect raw comparison', 'arm': ARM, 'steps': list(STEPS), 'training_support': c.artifact(support), 'export': c.artifact(EXPORT_MANIFEST), 'behavior': c.artifact(BEHAVIOR_MANIFEST), 'geometry': c.artifact(GEOMETRY_MANIFEST), 'output': c.artifact(TOTAL_EFFECT), 'rows': len(rows), 'created_utc': c.utc_now()}
    c.atomic_json(MANIFEST, result)
    return result


def run(phase: str, device: str) -> dict[str, Any]:
    if phase == 'export': return export()
    if phase == 'behavior_cell': raise RuntimeError('behavior_cell requires --step; call the CLI dispatcher')
    if phase == 'behavior_finalize': return behavior_finalize(device)
    if phase == 'behavior': return behavior(device)
    if phase == 'geometry_reference': return geometry_reference(device)
    if phase == 'geometry_cell': raise RuntimeError('geometry_cell requires --step; call the CLI dispatcher')
    if phase == 'geometry_finalize': return geometry_finalize(device)
    if phase == 'geometry': return geometry(device)
    if phase == 'total_effect': return total_effect()
    export(); behavior(device); geometry(device); return total_effect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('export', 'behavior_cell', 'behavior_finalize', 'behavior', 'geometry_reference', 'geometry_cell', 'geometry_finalize', 'geometry', 'worker', 'total_effect', 'all'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--step', type=int)
    parser.add_argument('--plan', default='')
    args = parser.parse_args()
    if args.phase == 'behavior_cell':
        if args.step is None: parser.error('--phase behavior_cell requires --step')
        value = behavior_cell(args.step, args.device)
    elif args.phase == 'geometry_cell':
        if args.step is None: parser.error('--phase geometry_cell requires --step')
        value = geometry_cell(args.step, args.device)
    elif args.phase == 'worker':
        value = worker(args.plan, args.device)
    else:
        value = run(args.phase, args.device)
    print(json.dumps(value, indent=2))
