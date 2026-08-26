"""Sample-free aggregation of frozen-base plus five Round3 final evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .data import VIEW_COUNTS


METHODS = (
    "frozen_base",
    "dpo_1k",
    "sspo_code_loss_stratified_ultrachat_2df9e9a",
    "dpo_8k",
    "dpo_pe_sft_rollout",
    "dpo_pe_rollout_only",
)
HEADS = (
    "dpo_reference_delta_beta_0.1",
    "raw_mean_logp_delta_beta_10",
)


def _load(root: Path, method: str) -> Dict[str, Any]:
    path = root / method / "metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "round3.final_metrics.v1" or value.get("method_id") != method:
        raise ValueError(f"Malformed Round3 final metrics for {method}")
    if set(value.get("heads", {})) != set(HEADS):
        raise ValueError(f"Round3 final metrics are missing score heads for {method}")
    if any(
        int(value["heads"][head].get("samples", 0)) != VIEW_COUNTS["test"]
        for head in HEADS
    ):
        raise ValueError(f"Round3 final metrics sample count mismatch for {method}")
    return value


def _delta(left: Dict[str, Any], right: Dict[str, Any], head: str) -> Dict[str, float]:
    metrics = ("accuracy_tie_half_credit", "nll_report_clamp_1e_12", "brier", "ece_15")
    return {
        name: float(left["heads"][head][name]) - float(right["heads"][head][name])
        for name in metrics
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.evaluations_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 aggregate: {output}")
    results = {method: _load(root, method) for method in METHODS}
    comparisons = {}
    for head in HEADS:
        comparisons[head] = {
            "dpo_1k_minus_frozen_base": _delta(results["dpo_1k"], results["frozen_base"], head),
            "sspo_minus_dpo_1k": _delta(results["sspo_code_loss_stratified_ultrachat_2df9e9a"], results["dpo_1k"], head),
            "dpo_8k_minus_dpo_1k_label_budget_gap": _delta(results["dpo_8k"], results["dpo_1k"], head),
            "pe_sft_rollout_minus_dpo_1k": _delta(results["dpo_pe_sft_rollout"], results["dpo_1k"], head),
            "pe_rollout_only_minus_dpo_1k": _delta(results["dpo_pe_rollout_only"], results["dpo_1k"], head),
            "pe_sft_rollout_minus_rollout_only": _delta(results["dpo_pe_sft_rollout"], results["dpo_pe_rollout_only"], head),
        }
    aggregate = {
        "schema_version": "round3.aggregate.v1",
        "experiment_contract": "round3-exp-v1.5",
        "single_seed_exploratory": True,
        "methods": {method: results[method] for method in METHODS},
        "same_head_only_comparisons": comparisons,
        "combined_score": None,
        "alpacaeval": "deferred_round4_not_run",
        "mt_bench": "deferred_round4_not_run",
        "pe_static": "deferred_round5_not_implemented",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
