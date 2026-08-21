"""Fail-closed aggregation and no-sample local-transfer whitelist exporter."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml


METHODS = (
    "dpo10",
    "dpo100",
    "sspo_hard_exp",
    "soppo_pe_exp",
    "soppo_pe_static_lambda_0.1",
    "soppo_pe_static_lambda_0.3",
    "soppo_pe_static_lambda_0.5",
    "soppo_pe_static_lambda_1.0",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--environment-summary", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--task-registry", required=True)
    parser.add_argument("--final-config", required=True)
    parser.add_argument("--c-epsilon", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    export = Path(args.export).resolve()
    if export.exists():
        raise FileExistsError(f"Refuse to overwrite export: {export}")
    export.mkdir(parents=True)
    metrics = {}
    for method in METHODS:
        path = experiment / "evaluation" / method / "metrics.json"
        complete = path.parent / "complete.json"
        if not path.is_file() or not complete.is_file():
            raise FileNotFoundError(f"Evaluation is incomplete for {method}: {path.parent}")
        metrics[method] = json.loads(path.read_text(encoding="utf-8"))

    final = yaml.safe_load(Path(args.final_config).read_text(encoding="utf-8"))
    selected_static = f"soppo_pe_static_lambda_{float(final['selected_static_lambda']):.1f}"
    comparisons = {
        "dpo10_base_headroom_validation": float(final["dpo10_base_headroom"]),
        "dpo100_vs_dpo10_test_oracle_gap": metrics["dpo100"]["accuracy"] - metrics["dpo10"]["accuracy"],
        "pe_exp_vs_dpo10": metrics["soppo_pe_exp"]["accuracy"] - metrics["dpo10"]["accuracy"],
        "pe_exp_vs_hard_exp": metrics["soppo_pe_exp"]["accuracy"] - metrics["sspo_hard_exp"]["accuracy"],
        "pe_static_selected_vs_dpo10": metrics[selected_static]["accuracy"] - metrics["dpo10"]["accuracy"],
        "pe_static_selected_vs_hard_exp": metrics[selected_static]["accuracy"] - metrics["sspo_hard_exp"]["accuracy"],
    }
    summary = {
        "schema_version": 2,
        "cycle_id": "cycle-20260818-01",
        "experiment_id": experiment.name,
        "experiment_design": "v0.6-sspo-aligned-30k",
        "single_seed": 42,
        "methods": metrics,
        "selected_static_method_validation_only": selected_static,
        "comparisons": comparisons,
        "interpretation_limit": "single-seed MVP trend; no significance claim",
    }
    (export / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (export / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "accuracy", "brier", "ece", "samples", "score_type"])
        for method in METHODS:
            row = metrics[method]
            writer.writerow([
                method, row["accuracy"], row["brier"], row["ece"], row["samples"], row["score_type"]
            ])
    lines = [
        "# SOPPO v0.6 SSPO-aligned 30k MVP aggregate",
        "",
        "Single seed (`42`); values are exploratory trends, not significance claims.",
        "Static lambda was selected on validation before this independent test evaluation.",
        "",
        "| method | accuracy | Brier | ECE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = metrics[method]
        lines.append(f"| {method} | {row['accuracy']:.4f} | {row['brier']:.4f} | {row['ece']:.4f} |")
    lines.extend(["", f"- selected static arm: `{selected_static}`"])
    for name, value in comparisons.items():
        lines.append(f"- {name}: `{value:.4f}`")
    lines.append("- Mechanism interpretation still requires joint review with C_epsilon trajectories.")
    (export / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for source, name in (
        (args.data_manifest, "manifest_public.json"),
        (args.environment_summary, "environment_summary.json"),
        (args.data_audit, "data_audit.json"),
        (args.task_registry, "task_registry.json"),
        (args.final_config, "config_final.yaml"),
        (args.c_epsilon, "c_epsilon_trajectory.csv"),
    ):
        shutil.copy2(source, export / name)
    forbidden = {"prompt", "response_a", "response_b", "prediction", "label", "sample_id"}

    def nested_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from nested_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from nested_keys(child)

    for path in export.rglob("*"):
        if not path.is_file():
            continue
        found = set()
        if path.suffix == ".json":
            found = set(nested_keys(json.loads(path.read_text(encoding="utf-8")))) & forbidden
        elif path.suffix == ".yaml":
            found = set(nested_keys(yaml.safe_load(path.read_text(encoding="utf-8")))) & forbidden
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                found = {name.lower() for name in (csv.DictReader(handle).fieldnames or [])} & forbidden
        if found:
            raise ValueError(f"Whitelist scan found sample-level fields {sorted(found)} in {path}")
    print(f"Whitelist export complete: {export}")


if __name__ == "__main__":
    main()
