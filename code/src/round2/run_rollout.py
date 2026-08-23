"""Run the independent vLLM rollout worker for round2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .config import load_round2_config, validate_round2_config
from .rollout_backend import build_rollout_command, launch_spec_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_round2_config(args.config, args.override)
    validate_round2_config(config)
    spec = launch_spec_from_config(config)
    command = build_rollout_command(spec)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_ids
    env["SOPPO_ROUND2_ROLLOUT_ENGINE"] = "vllm"
    spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = spec.artifact_path.with_suffix(spec.artifact_path.suffix + ".launch.json")
    if resolved.exists():
        raise FileExistsError(f"Refuse to overwrite rollout launch record: {resolved}")
    resolved.write_text(
        json.dumps(
            {
                "engine": "vllm",
                "source": spec.source,
                "gpu_ids": spec.gpu_ids,
                "command": command,
                "cwd": str(spec.working_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("CUDA_VISIBLE_DEVICES=" + spec.gpu_ids)
    print(" ".join(command))
    if args.dry_run:
        return
    if not spec.entrypoint.is_file():
        raise FileNotFoundError(f"Rollout entrypoint is missing: {spec.entrypoint}")
    if not spec.working_dir.is_dir():
        raise FileNotFoundError(f"Rollout working_dir is missing: {spec.working_dir}")
    subprocess.run(command, cwd=spec.working_dir, env=env, check=True)


if __name__ == "__main__":
    main()
