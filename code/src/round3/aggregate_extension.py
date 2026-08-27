"""Fail-closed, sample-free aggregation across legacy and extension runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import jsonlines
import yaml

from .aggregate import HEADS, _delta
from .data import VIEW_COUNTS, file_sha256


BASELINE_METHODS = (
    "frozen_base",
    "dpo_1k",
    "sspo_code_loss_stratified_ultrachat_2df9e9a",
    "dpo_8k",
    "dpo_pe_sft_rollout",
    "dpo_pe_rollout_only",
)
EXTENSION_METHODS = (
    "dpo_pe_dpo_reward_sft_rollout",
    "dpo_pe_dpo_reward_rollout_only",
)
EVALUATOR_DEPENDENCIES = (
    "code/src/round3/final_evaluate.py",
    "code/src/round3/data.py",
    "code/src/model/dpo_loss.py",
    "code/src/model/model_utils.py",
)


def _load_metrics(root: Path, method: str) -> Dict[str, Any]:
    path = root / "evaluations" / method / "metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "round3.final_metrics.v1":
        raise ValueError(f"Malformed final metrics schema: {method}")
    if value.get("method_id") != method or set(value.get("heads", {})) != set(HEADS):
        raise ValueError(f"Malformed final metrics inventory: {method}")
    if any(
        int(value["heads"][head].get("samples", 0)) != VIEW_COUNTS["test"]
        for head in HEADS
    ):
        raise ValueError(f"Final metrics sample count mismatch: {method}")
    return value


def _prediction_identity(root: Path, method: str) -> List[Tuple[str, int]]:
    path = root / "evaluations" / method / "predictions.private.jsonl"
    rows: List[Tuple[str, int]] = []
    with jsonlines.open(path) as reader:
        for row in reader:
            if set(row) != {"sample_id", "label", *HEADS}:
                raise ValueError(f"Private prediction schema mismatch: {method}")
            label = int(row["label"])
            if label not in {0, 1}:
                raise ValueError(f"Private prediction label mismatch: {method}")
            rows.append((str(row["sample_id"]), label))
    if (
        len(rows) != VIEW_COUNTS["test"]
        or len({sample_id for sample_id, _ in rows}) != len(rows)
    ):
        raise ValueError(f"Private prediction identity mismatch: {method}")
    return rows


def _controller(root: Path, terminal_stage: str) -> Dict[str, Any]:
    value = json.loads((root / "controller.json").read_text(encoding="utf-8"))
    if value.get("state") != "completed" or value.get("stage") != terminal_stage:
        raise ValueError(f"Round3 controller is not terminal: {root}")
    return value


def _config(root: Path, method: str) -> Dict[str, Any]:
    return yaml.safe_load(
        (root / "resolved" / "formal" / f"{method}.yaml").read_text(encoding="utf-8")
    )


def _manifest_hashes(config: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = Path(config["data"]["data_dir"]).resolve()
    reference_dir = Path(config["data"]["reference_cache_dir"]).resolve()
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "model_repo": config["model"]["repo_id"],
        "model_revision": config["model"]["resolved_revision"],
        "model_manifest_sha256": file_sha256(config["model"]["manifest_path"]),
        "ultrafeedback_revision": config["data"]["ultrafeedback_revision"],
        "ultrachat_revision": config["data"]["ultrachat_revision"],
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "test_public_sha256": file_sha256(data_dir / "test.public.jsonl"),
        "test_private_labels_sha256": file_sha256(data_dir / "test.private_labels.jsonl"),
        "reference_outputs": {
            name: item["output_sha256"]
            for name, item in sorted(reference_manifest["files"].items())
        },
        "test_heads": list(config["evaluation"]["test_heads"]),
        "test_pairs": int(config["evaluation"]["test_pairs"]),
    }


def _git_blob(repo: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout


def _evaluator_equality(repo: Path, left_commit: str, right_commit: str) -> Dict[str, str]:
    hashes = {}
    for path in EVALUATOR_DEPENDENCIES:
        left = _git_blob(repo, left_commit, path)
        right = _git_blob(repo, right_commit, path)
        if left != right:
            raise ValueError(f"Cross-run final evaluator dependency changed: {path}")
        hashes[path] = hashlib.sha256(left).hexdigest()
    return hashes


def _comparisons(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    for head in HEADS:
        output[head] = {
            "dpo_1k_minus_frozen_base": _delta(results["dpo_1k"], results["frozen_base"], head),
            "sspo_minus_dpo_1k": _delta(results["sspo_code_loss_stratified_ultrachat_2df9e9a"], results["dpo_1k"], head),
            "dpo_8k_minus_dpo_1k_label_budget_gap": _delta(results["dpo_8k"], results["dpo_1k"], head),
            "simpo_reward_pe_sft_minus_dpo_1k": _delta(results["dpo_pe_sft_rollout"], results["dpo_1k"], head),
            "simpo_reward_pe_rollout_only_minus_dpo_1k": _delta(results["dpo_pe_rollout_only"], results["dpo_1k"], head),
            "dpo_reward_pe_sft_minus_dpo_1k": _delta(results["dpo_pe_dpo_reward_sft_rollout"], results["dpo_1k"], head),
            "dpo_reward_pe_rollout_only_minus_dpo_1k": _delta(results["dpo_pe_dpo_reward_rollout_only"], results["dpo_1k"], head),
            "dpo_reward_minus_simpo_reward_sft": _delta(results["dpo_pe_dpo_reward_sft_rollout"], results["dpo_pe_sft_rollout"], head),
            "dpo_reward_minus_simpo_reward_rollout_only": _delta(results["dpo_pe_dpo_reward_rollout_only"], results["dpo_pe_rollout_only"], head),
            "dpo_reward_sft_minus_rollout_only": _delta(results["dpo_pe_dpo_reward_sft_rollout"], results["dpo_pe_dpo_reward_rollout_only"], head),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--baseline-run-root", required=True)
    parser.add_argument("--extension-run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    baseline = Path(args.baseline_run_root).resolve()
    extension = Path(args.extension_run_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 extension aggregate: {output}")

    baseline_controller = _controller(baseline, "all_methods")
    extension_controller = json.loads((extension / "controller.json").read_text(encoding="utf-8"))
    if (
        extension_controller.get("state") != "running"
        or extension_controller.get("stage") != "cross_run_sample_free_aggregate"
        or extension_controller.get("baseline_experiment_id")
        != baseline_controller.get("experiment_id")
    ):
        raise ValueError("Extension controller/baseline linkage mismatch")
    link = json.loads((extension / "baseline_link.json").read_text(encoding="utf-8"))
    baseline_controller_path = baseline / "controller.json"
    baseline_sources_path = baseline / "source_revisions.json"
    baseline_config_path = baseline / "resolved" / "formal" / "dpo_1k.yaml"
    copied_sources_path = extension / "source_revisions.json"
    if (
        link.get("baseline_experiment_id") != baseline_controller.get("experiment_id")
        or link.get("baseline_git_commit") != baseline_controller.get("git_commit")
        or Path(str(link.get("baseline_controller", ""))).resolve()
        != baseline_controller_path.resolve()
        or Path(str(link.get("source_revisions", ""))).resolve()
        != baseline_sources_path.resolve()
        or Path(str(link.get("baseline_config", ""))).resolve()
        != baseline_config_path.resolve()
        or link.get("baseline_controller_sha256")
        != file_sha256(baseline_controller_path)
        or link.get("source_revisions_sha256") != file_sha256(baseline_sources_path)
        or link.get("baseline_config_sha256") != file_sha256(baseline_config_path)
        or link.get("copied_source_revisions_sha256")
        != file_sha256(copied_sources_path)
        or baseline_sources_path.read_bytes() != copied_sources_path.read_bytes()
    ):
        raise ValueError("Immutable extension baseline link mismatch")

    baseline_config = _config(baseline, "dpo_1k")
    extension_config = _config(extension, EXTENSION_METHODS[0])
    baseline_manifest = _manifest_hashes(baseline_config)
    extension_manifest = _manifest_hashes(extension_config)
    if baseline_manifest != extension_manifest:
        raise ValueError("Cross-run model/data/reference/test manifests differ")
    evaluator_hashes = _evaluator_equality(
        repo,
        str(baseline_controller["git_commit"]),
        str(extension_controller["git_commit"]),
    )

    results = {
        method: _load_metrics(baseline, method) for method in BASELINE_METHODS
    }
    results.update(
        {method: _load_metrics(extension, method) for method in EXTENSION_METHODS}
    )
    identity = _prediction_identity(baseline, "frozen_base")
    for method in (*BASELINE_METHODS[1:], *EXTENSION_METHODS):
        root = baseline if method in BASELINE_METHODS else extension
        if _prediction_identity(root, method) != identity:
            raise ValueError(f"Cross-run test sample order/labels differ: {method}")

    aggregate = {
        "schema_version": "round3.cross_run_aggregate.v1",
        "experiment_contract": "round3-exp-v1.6",
        "single_seed_exploratory": True,
        "runs": {
            "baseline": {
                "experiment_id": baseline_controller["experiment_id"],
                "git_commit": baseline_controller["git_commit"],
            },
            "extension": {
                "experiment_id": extension_controller["experiment_id"],
                "git_commit": extension_controller["git_commit"],
            },
        },
        "cross_run_contract": {
            **baseline_manifest,
            "baseline_link_sha256": file_sha256(extension / "baseline_link.json"),
            "sample_id_and_private_label_order_sha256": hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "evaluator_dependency_sha256": evaluator_hashes,
        },
        "methods": results,
        "same_head_only_comparisons": _comparisons(results),
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
