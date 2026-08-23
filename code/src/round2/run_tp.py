"""Launch the project-owned Transformers TP=2 trainer."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from .config import load_round2_config, validate_round2_config
from .tp_backend import build_tp_command, launch_spec_from_config, write_launch_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_round2_config(args.config)
    validate_round2_config(config)
    spec = launch_spec_from_config(config, args.config, sys.executable)
    command = build_tp_command(spec)
    print("CUDA_VISIBLE_DEVICES=" + spec.gpu_ids)
    print(" ".join(shlex.quote(value) for value in command))
    if args.dry_run:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_ids
    env["SOPPO_ROUND2_BACKEND"] = "transformers-native-tp"
    write_launch_record(spec.output_dir / "tp_launch.resolved.json", spec, command)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
