from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.paths import resolve_repo_path
from opd_sft_h1.registry import load_jsonl


def _csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="/root/autodl-tmp/exp0609/trl_first_minimal")
    args = parser.parse_args()

    run_root = resolve_repo_path(args.run_root)
    assert run_root is not None
    summary = {
        "run_root": str(run_root),
        "run_registry_records": len(load_jsonl(run_root / "registry" / "run_registry.jsonl")),
        "checkpoint_records": len(load_jsonl(run_root / "registry" / "checkpoints.jsonl")),
        "eval_trajectory_rows": _csv_rows(run_root / "tables" / "eval_trajectory.csv"),
        "ood_penalty_rows": _csv_rows(run_root / "tables" / "ood_penalty.csv"),
        "geometry_long_rows": _csv_rows(run_root / "tables" / "geometry_long.csv"),
        "geometry_metrics_rows": _csv_rows(run_root / "tables" / "geometry_metrics.csv"),
        "matched_math500_rows": _csv_rows(run_root / "tables" / "matched_math500_pairs.csv"),
    }

    output_path = run_root / "summary_minimal_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
