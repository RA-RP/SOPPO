#!/usr/bin/env python3
"""Restart-safe two-GPU DAG supervisor for the Cycle 09 Stage-4 A0-A9 work."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cycle09_block3_common as b3
import cycle09_stage4_state_displacement as s4

SCRIPT = b3.SCRIPT_DIR / "cycle09_stage4_state_displacement.py"
READOUT = b3.SCRIPT_DIR / "cycle09_stage4_readout.py"
CPU = b3.SCRIPT_DIR / "cycle09_stage4_cpu.py"
STATE = s4.ROOT / "supervisor_state.json"
PY = b3.DENSITY_PYTHON
CORE_ARMS = ("opd", "offkd", "sft", "seqkd")
P1_PROBES = ("E_general", "E_math", "E_if")


def write_state(phase: str, status: str, completed: list[str], error: str | None = None) -> None:
    s4.atomic_json(STATE, {
        "schema_version": "cycle09_stage4_supervisor_v2",
        "phase": phase,
        "status": status,
        "completed": completed,
        "error": error,
        "created_utc": b3.utc_now(),
        "no_auto_shutdown": True,
    })


def run(command: list[str]) -> None:
    print("[CMD] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=b3.REPO, check=True)


def stage4(*items: str) -> list[str]:
    return [str(PY), str(SCRIPT), *items]


def cpu(*items: str) -> list[str]:
    return [str(PY), str(CPU), *items]


def readout(*items: str) -> list[str]:
    return [str(PY), str(READOUT), *items]


def legacy_exception(model: str, command: list[str]) -> list[str]:
    """Only historical Qwen cells may use the user-approved fp32 weight-difference fallback."""
    return command + (["--allow-effective-weight-diff"] if model == "qwen" else [])


def available_cells(model: str) -> list[tuple[str, int]]:
    cells = [("base", 0)]
    for arm in CORE_ARMS:
        for step in s4.SPECS[model]["steps"]:
            if step == 0:
                continue
            try:
                s4.model_path(model, arm, step, materialize=False)
            except (OSError, RuntimeError, FileNotFoundError):
                if model != "llama":
                    continue
                try:
                    s4.lexport.validate_adapter(s4.lexport.adapter_target(arm, step), arm, step)
                except (OSError, RuntimeError, FileNotFoundError):
                    continue
            cells.append((arm, step))
    return cells


def selected_p1_steps(cells: list[tuple[str, int]], arm: str) -> tuple[int, ...]:
    present = sorted(step for found_arm, step in cells if found_arm == arm and step)
    if not present:
        return ()
    candidates = (5, 160, present[-1])
    return tuple(dict.fromkeys(step for step in candidates if step in present))


def encoded(commands: list[list[str]]) -> list[str]:
    return [json.dumps(command) for command in commands]


def execute_serial(commands: list[str]) -> list[str]:
    done = []
    for raw in commands:
        command = json.loads(raw)
        run(command)
        done.append(" ".join(command[2:6]))
    return done


def smoke_lane(model: str, device: str) -> list[str]:
    layer = str(s4.SPECS[model]["layer"])
    commands: list[list[str]] = []
    for arm, step in (("base", 0), ("opd", 20)):
        commands.append(stage4(
            "--phase", "profile", "--model", model, "--arm", arm, "--step", str(step),
            "--probe", "E_math", "--layers", layer, "--tag", "smoke", "--device", device,
            "--measurement-n", "2", "--forward-batch-size", "1", "--max-batch-tokens", "2048",
        ))
    for phase in ("metric", "centered"):
        commands.append(legacy_exception(model, stage4(
            "--phase", phase, "--model", model, "--arm", "opd", "--step", "20",
            "--probe", "E_math", "--layer", layer, "--tag", "smoke", "--device", device,
        )))
    for arm, step in (("base", 0), ("opd", 20)):
        commands.append(stage4(
            "--phase", "profile", "--model", model, "--arm", arm, "--step", str(step),
            "--probe", "E_math", "--layers", layer, "--tag", "smoke_retain", "--device", device,
            "--measurement-n", "2", "--retain-samples", "--forward-batch-size", "1",
            "--max-batch-tokens", "2048",
        ))
    commands.append(readout(
        "--phase", "local-output", "--model", model, "--arm", "opd", "--step", "20",
        "--probe", "E_math", "--layer", layer, "--tag", "smoke_retain", "--device", device,
    ))
    return encoded(commands)


def complete(path: Path) -> bool:
    return path.is_file() and s4.read_json(path, {}).get("status") == "complete"


def formal_a1_a3_lane(model: str, device: str) -> list[str]:
    """Schedule only incomplete dependencies so consumed profiles may be reclaimed safely."""
    layer = int(s4.SPECS[model]["layer"])
    cells = available_cells(model)
    commands: list[list[str]] = []
    scheduled_profiles: set[tuple[str, int, str]] = set()

    def add_profile(arm: str, step: int, probe: str) -> None:
        key = (arm, step, probe)
        profile_meta = s4.profile_meta(model, arm, step, probe, "main")
        if key in scheduled_profiles or complete(profile_meta):
            return
        commands.append(stage4(
            "--phase", "profile", "--model", model, "--arm", arm, "--step", str(step),
            "--probe", probe, "--layers", str(layer), "--tag", "main", "--device", device,
        ))
        scheduled_profiles.add(key)

    for arm, step in cells:
        for probe in s4.PROBES:
            metric = s4.cell_file(model, arm, step, probe, layer, "main")
            if complete(metric):
                continue
            add_profile(arm, step, probe)
            commands.append(legacy_exception(model, stage4(
                "--phase", "metric", "--model", model, "--arm", arm, "--step", str(step),
                "--probe", probe, "--layer", str(layer), "--tag", "main", "--device", device,
            )))

    for arm in ("opd", "offkd"):
        for step in selected_p1_steps(cells, arm):
            for probe in P1_PROBES:
                centered = s4.cell_file(model, arm, step, probe, layer, "main.centered")
                if complete(centered):
                    continue
                add_profile("base", 0, probe)
                add_profile(arm, step, probe)
                commands.append(legacy_exception(model, stage4(
                    "--phase", "centered", "--model", model, "--arm", arm, "--step", str(step),
                    "--probe", probe, "--layer", str(layer), "--tag", "main", "--device", device,
                )))
    return encoded(commands)


def formal_p1_lane(model: str, device: str) -> list[str]:
    """P1 follows its own model's P0 cells immediately; retained scope stays disk-bounded."""
    layer = str(s4.SPECS[model]["layer"])
    cells = available_cells(model)
    commands: list[list[str]] = []
    # Fixed token manifests: eight samples per retained cell.  Larger raw profiles are not retained.
    for probe in P1_PROBES:
        commands.append(stage4(
            "--phase", "profile", "--model", model, "--arm", "base", "--step", "0",
            "--probe", probe, "--layers", layer, "--tag", "p1", "--device", device,
            "--measurement-n", "8", "--retain-samples",
        ))
    for arm in CORE_ARMS:
        for step in selected_p1_steps(cells, arm):
            for probe in P1_PROBES:
                commands.append(stage4(
                    "--phase", "profile", "--model", model, "--arm", arm, "--step", str(step),
                    "--probe", probe, "--layers", layer, "--tag", "p1", "--device", device,
                    "--measurement-n", "8", "--retain-samples",
                ))
                commands.append(readout(
                    "--phase", "local-output", "--model", model, "--arm", arm, "--step", str(step),
                    "--probe", probe, "--layer", layer, "--tag", "p1", "--device", device,
                ))
    # Finite interventions are the minimum OPD/off-KD landmark suite; each cell remains resumable.
    for arm in ("opd", "offkd"):
        for step in selected_p1_steps(cells, arm):
            for probe in P1_PROBES:
                commands.append(readout(
                    "--phase", "zeroing", "--model", model, "--arm", arm, "--step", str(step),
                    "--probe", probe, "--layer", layer, "--device", device, "--measurement-n", "8",
                ))
    return encoded(commands)


def formal_lane(model: str, device: str) -> list[str]:
    return formal_a1_a3_lane(model, device) + formal_p1_lane(model, device)


def smoke() -> None:
    write_state("smoke", "running", [])
    run(stage4("--phase", "audit", "--models", "all"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(execute_serial, smoke_lane("qwen", "cuda:0")),
            pool.submit(execute_serial, smoke_lane("llama", "cuda:1")),
        ]
        completed = [item for future in futures for item in future.result()]
    run(readout(
        "--phase", "zeroing", "--model", "qwen", "--arm", "opd", "--step", "20",
        "--probe", "E_math", "--layer", "18", "--device", "cuda:0", "--measurement-n", "1",
    ))
    run(cpu("--phase", "a4", "--tag", "smoke"))
    run(cpu("--phase", "a7"))
    run(cpu("--phase", "a9"))
    write_state("smoke", "complete", completed)


def formal() -> None:
    write_state("formal", "running", [])
    run(stage4("--phase", "audit", "--models", "all"))
    # A7 is independent and runs while both model lanes are busy.
    with ThreadPoolExecutor(max_workers=3) as pool:
        audit_future = pool.submit(run, cpu("--phase", "a7"))
        futures = [
            pool.submit(execute_serial, formal_lane("qwen", "cuda:0")),
            pool.submit(execute_serial, formal_lane("llama", "cuda:1")),
        ]
        completed = [item for future in futures for item in future.result()]
        audit_future.result()
    run(stage4("--phase", "pairs", "--model", "qwen", "--layer", "18", "--tag", "main"))
    run(stage4("--phase", "pairs", "--model", "llama", "--layer", "14", "--tag", "main"))
    run(stage4("--phase", "finalize", "--tag", "main"))
    run(cpu("--phase", "bootstrap", "--tag", "p1", "--draws", "256"))
    run(cpu("--phase", "a4", "--tag", "main"))
    run(cpu("--phase", "a9"))
    write_state("formal", "complete", completed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", "formal", "dry-run"))
    args = parser.parse_args()
    if args.phase == "dry-run":
        print(json.dumps({
            "gpu0": "Qwen: A1/A3 full grid -> Qwen A5/A6 retained landmarks",
            "gpu1": "Llama: A1/A3 full grid -> Llama A5/A6 retained landmarks",
            "cpu": "A0/a7 concurrently; then A2/A3-bootstrap/A4/A9 after cell manifests",
            "restart": "atomic cell/profile manifests; completed work is skipped",
            "legacy_delta_policy": "Qwen only: explicit authorized effective-weight fallback; Llama: adapter BA fp32",
            "smoke_estimate_minutes": "15-20",
            "no_auto_shutdown": True,
        }, indent=2))
        return
    try:
        {"smoke": smoke, "formal": formal}[args.phase]()
    except Exception as error:
        write_state(args.phase, "failed", [], f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
