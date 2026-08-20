#!/usr/bin/env python3
"""D4.1 numerical parity audit for the Qwen merged-state rank track."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import sys

REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_qwen_d4_merged_state as d4  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402
import cycle09_r4_common as c4  # noqa: E402


FORMAL = d4.ROOT / "formal"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
OFFLINE_PARITY = (("sft", 160), ("offkd", 160), ("seqkd", 160))
OPD_BLOCKED = (5, 160, 624)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def saved_state(arm: str, step: int, probe: str) -> dict[str, Any]:
    path = d4.state_path("formal", arm, step, probe)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise RuntimeError(f"incomplete saved D4 cell: {path}")
    return value


def recompute(arm: str, step: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    report_arm = "base" if step == 0 else arm
    samples = {probe: d4.samples_for(probe, args.sample_limit) for probe in d4.CORE_PROBES}
    rows: list[dict[str, Any]] = []
    with d4.materialized_model(report_arm, step) as merged:
        model = d4.load_bf16(merged, args.device)
        try:
            for probe, fixed_samples in samples.items():
                expected = saved_state(report_arm, step, probe)
                if expected["sample_ids_sha256"] != d4.json_digest([item.sample_id for item in fixed_samples]):
                    raise RuntimeError(f"sample-ID mismatch for {report_arm}/{step}/{probe}")
                measured = d4.profile_state(model, fixed_samples, args.device, args.forward_batch_size, args.max_batch_tokens)
                stored = {(item["module"], float(item["epsilon"])): int(item["r_epsilon"]) for item in expected["state_rows"]}
                recomputed: dict[tuple[str, float], int] = {}
                for module in d4.MODULES:
                    group = c4.MODULE_TO_GROUP[module]
                    scale = d4.sqrt_gram(measured["grams"][d4.LAYER][group], args.device)
                    weight = campaign.module_at(model, d4.LAYER, module).weight.detach().to(args.device, torch.float32)
                    singular = torch.linalg.svdvals((weight @ scale.to(torch.float32)).to(torch.float64))
                    for epsilon, rank in d4.ranks(singular).items():
                        recomputed[(module, float(epsilon))] = int(rank)
                    del scale, weight, singular
                    torch.cuda.empty_cache()
                route = "bf16_cpu_save_pretrained_materialization_reload" if step == 0 else "bf16_cpu_peft_merge_and_unload_independent_reconstruction"
                for (module, epsilon), value in sorted(recomputed.items()):
                    reference = stored[(module, epsilon)]
                    rows.append({
                        "model": "qwen", "arm": report_arm, "checkpoint": step, "probe_name": probe,
                        "level": "module", "module": module, "epsilon": epsilon,
                        "saved_r_epsilon": reference, "recomputed_r_epsilon": value,
                        "difference": value - reference,
                        "status": "PASS_EXACT" if value == reference else "FAIL_DIFFERENT_RANK",
                        "parity_route": route,
                    })
                for epsilon in d4.EPSILONS:
                    reference_mean = sum(stored[(module, epsilon)] for module in d4.MODULES) / len(d4.MODULES)
                    value_mean = sum(recomputed[(module, epsilon)] for module in d4.MODULES) / len(d4.MODULES)
                    rows.append({
                        "model": "qwen", "arm": report_arm, "checkpoint": step, "probe_name": probe,
                        "level": "equal7_mean", "module": "equal7", "epsilon": epsilon,
                        "saved_r_epsilon": reference_mean, "recomputed_r_epsilon": value_mean,
                        "difference": value_mean - reference_mean,
                        "status": "PASS_EXACT" if value_mean == reference_mean else "FAIL_DIFFERENT_RANK",
                        "parity_route": route,
                    })
                expected_order = [epsilon for epsilon, _ in sorted(((epsilon, sum(stored[(module, epsilon)] for module in d4.MODULES)) for epsilon in d4.EPSILONS), key=lambda item: (item[1], item[0]))]
                actual_order = [epsilon for epsilon, _ in sorted(((epsilon, sum(recomputed[(module, epsilon)] for module in d4.MODULES)) for epsilon in d4.EPSILONS), key=lambda item: (item[1], item[0]))]
                rows.append({
                    "model": "qwen", "arm": report_arm, "checkpoint": step, "probe_name": probe,
                    "level": "epsilon_order", "module": "equal7", "epsilon": "all",
                    "saved_order": ";".join(map(str, expected_order)), "recomputed_order": ";".join(map(str, actual_order)),
                    "status": "PASS_EXACT" if expected_order == actual_order else "FAIL_DIFFERENT_ORDER",
                    "parity_route": route,
                })
        finally:
            d4.unload(model)
    return rows


def blocked_opd_rows() -> list[dict[str, Any]]:
    rows = []
    for step in OPD_BLOCKED:
        for probe in d4.CORE_PROBES:
            rows.append({
                "model": "qwen", "arm": "opd", "checkpoint": step, "probe_name": probe,
                "level": "reconstruction_availability", "module": "all", "epsilon": "all",
                "status": "BLOCKED_MISSING_OPD_ADAPTER",
                "parity_route": "unavailable_no_adapter_BA_path",
                "searched_paths": "D2 audit: all Qwen OPD formal checkpoints have existing merged models but no adapter path",
            })
    return rows


def formal(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_limit:
        raise ValueError("formal parity requires full frozen probe corpora")
    rows = recompute("base", 0, args)
    for arm, step in OFFLINE_PARITY:
        rows.extend(recompute(arm, step, args))
    rows.extend(blocked_opd_rows())
    target = MINI / "qwen_merged_state_parity_audit.csv"
    atomic_csv(target, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    manifest = {
        "schema_version": "cycle09_qwen_d4_parity_audit_v1", "status": "complete_with_declared_opd_block",
        "created_utc": now(), "rows": len(rows), "status_counts": counts,
        "recomputed": [{"arm": "base", "checkpoint": 0}, *[{"arm": arm, "checkpoint": step} for arm, step in OFFLINE_PARITY]],
        "opd_blocked_checkpoints": list(OPD_BLOCKED),
        "reason": "Qwen OPD serialized merged checkpoints are present, but no corresponding adapter B@A is available for W0+sBA reconstruction.",
        "artifact": str(target),
    }
    atomic_json(MINI / "qwen_merged_state_parity_manifest.json", manifest)
    protocol_path = MINI / "qwen_merged_state_numeric_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["parity_audit"] = manifest
    atomic_json(protocol_path, protocol)
    return manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("formal",), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = formal(args)
    print(json.dumps({"status": result["status"], "rows": result["rows"], "created_utc": now()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
