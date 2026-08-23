"""Aggregate the two round2 methods without copying sample-level artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from ..config import canonical_json


METHODS = (
    "soppo_pe_sft_rollout_exp",
    "soppo_pe_rollout_only_exp",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--export", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    export = Path(args.export).resolve()
    if export.exists():
        raise FileExistsError(f"Refuse to overwrite round2 export: {export}")

    test_gate = json.loads((experiment / "server_tests" / "complete.json").read_text())
    smoke_gate = json.loads((experiment / "strong_smoke" / "complete.json").read_text())
    if test_gate.get("status") != "succeeded" or smoke_gate.get("status") != "succeeded":
        raise RuntimeError("Round2 server-test/strong-smoke evidence is incomplete")

    metrics = {}
    commits = set()
    controlled_configs = set()
    for method in METHODS:
        run_dir = experiment / method
        completion = json.loads((run_dir / "complete.json").read_text())
        evaluation = json.loads((run_dir / "evaluation" / "complete.json").read_text())
        if completion.get("status") != "succeeded" or evaluation.get("status") != "succeeded":
            raise RuntimeError(f"Round2 method is incomplete: {method}")
        row = json.loads((run_dir / "evaluation" / "metrics.json").read_text())
        if row.get("method") != method or int(row.get("samples", 0)) != 3000:
            raise ValueError(f"Round2 evaluation contract mismatch: {method}")
        if row.get("score_type") != "simpo_mean_logp_delta_margin_free":
            raise ValueError(f"Round2 score type mismatch: {method}")
        metrics[method] = row
        config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
        config_sha256 = hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest()
        commits.add(config["provenance"]["git_commit"])
        if completion.get("git_commit") != config["provenance"]["git_commit"]:
            raise ValueError(f"Round2 completion/commit mismatch: {method}")
        if row.get("git_commit") != config["provenance"]["git_commit"]:
            raise ValueError(f"Round2 evaluation/commit mismatch: {method}")
        if completion.get("config_sha256") != config_sha256:
            raise ValueError(f"Round2 completion/config mismatch: {method}")
        if row.get("config_sha256") != config_sha256:
            raise ValueError(f"Round2 evaluation/config mismatch: {method}")
        tp_evidence = json.loads((run_dir / "tp_evidence.json").read_text())
        worker_evidence = json.loads(
            (run_dir / "rollouts" / "worker.ready.json").read_text()
        )
        for label, evidence in (
            ("TP", tp_evidence),
            ("rollout worker", worker_evidence),
        ):
            if evidence.get("git_commit") != config["provenance"]["git_commit"]:
                raise ValueError(f"Round2 {label}/commit mismatch: {method}")
            if evidence.get("config_sha256") != config_sha256:
                raise ValueError(f"Round2 {label}/config mismatch: {method}")
        for role in ("training", "rollout"):
            preflight = json.loads(
                (run_dir / "preflight" / f"{role}.json").read_text()
            )
            if preflight.get("git_commit") != config["provenance"]["git_commit"]:
                raise ValueError(f"Round2 {role} preflight/commit mismatch: {method}")
            if preflight.get("config_sha256") != config_sha256:
                raise ValueError(f"Round2 {role} preflight/config mismatch: {method}")
        controlled = copy.deepcopy(config)
        controlled["method"]["name"] = "<controlled-method>"
        controlled["rollout"]["source"] = "<controlled-source>"
        controlled["rollout"]["artifact_dir"] = "<method-output>"
        controlled["output"]["run_dir"] = "<method-output>"
        controlled_configs.add(canonical_json(controlled))
    if len(commits) != 1:
        raise ValueError("Round2 methods were produced by different Git commits")
    only_commit = next(iter(commits))
    if test_gate.get("git_commit") != only_commit or smoke_gate.get("git_commit") != only_commit:
        raise ValueError("Round2 gates and formal methods were produced by different commits")
    if len(controlled_configs) != 1:
        raise ValueError("Round2 methods differ outside candidate construction/output paths")

    export.mkdir(parents=True)
    delta = (
        metrics["soppo_pe_sft_rollout_exp"]["accuracy"]
        - metrics["soppo_pe_rollout_only_exp"]["accuracy"]
    )
    summary = {
        "schema_version": 1,
        "cycle_id": "cycle-20260818-01",
        "experiment_id": experiment.name,
        "git_commit": only_commit,
        "single_seed": 42,
        "methods": metrics,
        "comparisons": {
            "sft_rollout_minus_rollout_only_accuracy": delta,
        },
        "interpretation_limit": "single-seed round2 trend; no significance claim",
        "first_round_merge": "separate read-only downstream handoff",
    }
    (export / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (export / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "accuracy", "brier", "ece", "samples", "score_type"])
        for method in METHODS:
            row = metrics[method]
            writer.writerow(
                [
                    method,
                    row["accuracy"],
                    row["brier"],
                    row["ece"],
                    row["samples"],
                    row["score_type"],
                ]
            )
    lines = [
        "# SOPPO Round2 aggregate",
        "",
        "Single seed (`42`); exploratory trend, not a significance claim.",
        "",
        "| method | accuracy | Brier | ECE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = metrics[method]
        lines.append(
            f"| {method} | {row['accuracy']:.4f} | {row['brier']:.4f} | {row['ece']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- SFT+rollout minus rollout-only accuracy: `{delta:.4f}`",
            "- 第一轮冻结基线在后续结果交接中只读合并，本导出不回写第一轮产物。",
        ]
    )
    (export / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for method in METHODS:
        run_dir = experiment / method
        shutil.copy2(run_dir / "config.resolved.yaml", export / f"{method}.config.yaml")
        shutil.copy2(run_dir / "tp_evidence.json", export / f"{method}.tp_evidence.json")
        shutil.copy2(
            run_dir / "preflight" / "training.json",
            export / f"{method}.training_preflight.json",
        )
        shutil.copy2(
            run_dir / "preflight" / "rollout.json",
            export / f"{method}.rollout_preflight.json",
        )
        shutil.copy2(
            run_dir / "rollouts" / "worker.ready.json",
            export / f"{method}.rollout_worker.json",
        )
        shutil.copy2(
            run_dir / "evaluation" / "metrics.json",
            export / f"{method}.metrics.json",
        )
        shutil.copy2(
            run_dir / "evaluation" / "calibration.json",
            export / f"{method}.calibration.json",
        )
    (export / "EXPORT_COMPLETE").write_text("round2 export complete\n", encoding="utf-8")
    print(f"Round2 whitelist export complete: {export}")


if __name__ == "__main__":
    main()
