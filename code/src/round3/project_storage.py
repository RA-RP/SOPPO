"""Project retained Round3 storage from selected production-path strong smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "dpo_1k",
    "sspo_code_loss_stratified_ultrachat_2df9e9a",
    "dpo_8k",
    "dpo_pe_sft_rollout",
    "dpo_pe_rollout_only",
    "dpo_pe_dpo_reward_sft_rollout",
    "dpo_pe_dpo_reward_rollout_only",
)
DYNAMIC = {
    "dpo_pe_sft_rollout",
    "dpo_pe_rollout_only",
    "dpo_pe_dpo_reward_sft_rollout",
    "dpo_pe_dpo_reward_rollout_only",
}
SFT_ROLLOUT = {
    "dpo_pe_sft_rollout",
    "dpo_pe_dpo_reward_sft_rollout",
}


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def source_cache_size(data_dir: Path) -> int:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = set()
    for entries in manifest.get("source_parquet_files", {}).values():
        paths.update(Path(entry["path"]).resolve() for entry in entries)
    for entries in manifest.get("source_cache_files", {}).values():
        paths.update(Path(entry["path"]).resolve() for entry in entries)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("Round3 source parquet/Arrow-cache projection inputs are incomplete")
    return sum(path.stat().st_size for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--train-env", required=True)
    parser.add_argument("--rollout-env", required=True)
    parser.add_argument("--platform-log-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    args = parser.parse_args()
    smoke_root = Path(args.smoke_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    fixed = {
        "model": size(Path(args.model_dir).resolve()),
        "data": size(data_dir),
        "dataset_source_parquet_and_arrow_cache": source_cache_size(data_dir),
        "reference_cache": size(Path(args.reference_dir).resolve()),
        "train_environment": size(Path(args.train_env).resolve()),
        "rollout_environment": size(Path(args.rollout_env).resolve()),
        "retained_strong_smoke_artifacts": size(smoke_root),
        "existing_round3_platform_logs": size(Path(args.platform_log_root).resolve()),
    }
    selected_methods = tuple(args.methods or METHODS)
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Round3 storage projection methods must be unique")
    methods = {}
    for method in selected_methods:
        root = smoke_root / method
        checkpoint = size(root / "smoke_checkpoint" / "step_000001")
        if checkpoint <= 0:
            raise FileNotFoundError(f"Representative Round3 smoke checkpoint missing: {method}")
        staging = size(root / "rollouts" / "policy" / "step_000000") if method in DYNAMIC else 0
        jobs = 28 if method in SFT_ROLLOUT else 56 if method in DYNAMIC else 0
        # Bound retained JSON/text at 16 UTF-8/JSON bytes per generated token,
        # plus the measured request/response structure from the smoke.
        queue_measured = (
            size(root / "rollouts" / "requests") + size(root / "rollouts" / "responses")
            if method in DYNAMIC
            else 0
        )
        queue_worst_per_step = max(queue_measured, jobs * 1024 * 16)
        projected = checkpoint * 10 + staging * (251 if method in DYNAMIC else 0) + queue_worst_per_step * (250 if method in DYNAMIC else 0)
        methods[method] = {
            "representative_checkpoint_bytes": checkpoint,
            "representative_staging_adapter_bytes": staging,
            "queue_worst_per_step_bytes": queue_worst_per_step,
            "projected_method_bytes": projected,
        }
    fixed_total = sum(fixed.values())
    method_total = sum(value["projected_method_bytes"] for value in methods.values())
    miscellaneous_reserve = 1024**3
    projected_peak = fixed_total + method_total + miscellaneous_reserve
    result = {
        "schema_version": "round3.storage_projection.v2",
        "projected_methods": list(selected_methods),
        "fixed_bytes": fixed,
        "method_bytes": methods,
        "miscellaneous_reserve_bytes": miscellaneous_reserve,
        "projected_peak_bytes": projected_peak,
        "formal_required_free_bytes": 2 * projected_peak,
        "automatic_deletion": False,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 storage projection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
