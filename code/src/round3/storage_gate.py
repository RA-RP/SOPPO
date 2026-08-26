"""One-time formal free >= 2 * projected_peak storage gate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import load_round3_config, validate_round3_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    if config["execution"]["mode"] != "formal":
        raise ValueError("The one-time storage gate requires a formal config")
    projected = int(config["storage"]["projected_peak_bytes"])
    run_dir = Path(config["output"]["run_dir"]).resolve()
    probe = run_dir.parent
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    required = 2 * projected
    if usage.free < required:
        raise RuntimeError(
            f"Round3 storage gate failed: free={usage.free} < required={required}; no deletion attempted"
        )
    result = {
        "schema_version": "round3.formal_storage_gate.v1",
        "status": "passed",
        "experiment_id": config["provenance"]["experiment_id"],
        "filesystem_probe": str(probe),
        "projected_peak_bytes": projected,
        "required_free_bytes": required,
        "free_bytes_at_gate": usage.free,
        "automatic_deletion": False,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 formal storage gate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

