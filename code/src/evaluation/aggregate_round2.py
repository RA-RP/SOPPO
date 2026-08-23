"""Second-layer aggregation for round2.

The first-round aggregate is treated as an immutable input. This module only
combines summary-level JSON documents and never copies sample-level artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROUND2_METHODS = ("soppo_pe_sft_rollout_exp", "soppo_pe_rollout_only_exp")


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _method_name(summary: Dict[str, Any], fallback: str) -> str:
    method = summary.get("method") or summary.get("config", {}).get("method", {}).get("name")
    return str(method or fallback)


def _numeric_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    candidates = summary.get("metrics", summary)
    if not isinstance(candidates, dict):
        return {}
    result = {}
    for key, value in candidates.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[str(key)] = float(value)
    return result


def _delta(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(left) & set(right))
    return {key: left[key] - right[key] for key in keys}


def aggregate(
    frozen: Dict[str, Any],
    sft_rollout: Dict[str, Any],
    rollout_only: Dict[str, Any],
) -> Dict[str, Any]:
    frozen_metrics = _numeric_metrics(frozen)
    sft_metrics = _numeric_metrics(sft_rollout)
    only_metrics = _numeric_metrics(rollout_only)
    return {
        "schema_version": "round2.aggregate.v1",
        "round1_frozen": {
            "source": frozen.get("source") or frozen.get("experiment"),
            "method": _method_name(frozen, "round1_frozen"),
            "metrics": frozen_metrics,
        },
        "round2": {
            "soppo_pe_sft_rollout_exp": {
                "source": sft_rollout.get("source") or sft_rollout.get("experiment"),
                "method": _method_name(sft_rollout, ROUND2_METHODS[0]),
                "metrics": sft_metrics,
            },
            "soppo_pe_rollout_only_exp": {
                "source": rollout_only.get("source") or rollout_only.get("experiment"),
                "method": _method_name(rollout_only, ROUND2_METHODS[1]),
                "metrics": only_metrics,
            },
        },
        "deltas": {
            "sft_rollout_minus_rollout_only": _delta(sft_metrics, only_metrics),
            "sft_rollout_minus_round1_frozen": _delta(sft_metrics, frozen_metrics),
            "rollout_only_minus_round1_frozen": _delta(only_metrics, frozen_metrics),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1-frozen", required=True)
    parser.add_argument("--sft-rollout", required=True)
    parser.add_argument("--rollout-only", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite round2 aggregate: {output}")
    result = aggregate(
        _read_object(Path(args.round1_frozen).resolve(), "round1 frozen summary"),
        _read_object(Path(args.sft_rollout).resolve(), "sft-rollout summary"),
        _read_object(Path(args.rollout_only).resolve(), "rollout-only summary"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Round2 aggregate written: {output}")


if __name__ == "__main__":
    main()
