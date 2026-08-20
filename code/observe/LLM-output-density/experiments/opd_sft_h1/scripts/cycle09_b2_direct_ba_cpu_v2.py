#!/usr/bin/env python3
"""Strict-rank replacement for the CPU-only B2 direct-BA precompute.

This version uses numerical rank, rather than nominal adapter rank, for the
matched-top-k ``NA_RANK_LIMIT`` sentinel required by B1.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

import sys

SCRIPTS = Path("/root/LLM-output-density/experiments/opd_sft_h1/scripts")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_b_lora_proxy as b0  # noqa: E402
import cycle09_b2_direct_ba_cpu as v1  # noqa: E402


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for model, spec in b0.SPECS.items():
        for arm in spec["arms"]:
            for step in spec["steps"]:
                _, adapter_path = b0.source_paths(model, arm, step)
                for module in b0.MODULES:
                    adapter = b0.adapter_summary(adapter_path, spec["layer"], module)
                    identity = {
                        "model": model, "arm": arm, "checkpoint": step,
                        "layer": spec["layer"], "module": module,
                        "weight_object": "direct_BA_from_bf16_factors_fp32_matmul",
                    }
                    if adapter.get("status") != "AVAILABLE" or adapter.get("module_keys_status") != "AVAILABLE":
                        blocked.append({**identity, "status": "BLOCKED_DIRECT_BA_ARTIFACT_MISSING"})
                        continue
                    singular = v1.direct_spectrum(adapter)
                    energy = singular.square()
                    total = float(energy.sum())
                    tolerance = float(singular.max()) * max(singular.numel(), 1) * torch.finfo(torch.float64).eps
                    numerical_rank = int((singular > tolerance).sum())
                    row = {
                        **identity, "status": "complete", "lora_rank": adapter["rank"],
                        "lora_alpha": adapter["alpha"], "lora_scaling": adapter["scaling"],
                        "factor_storage_dtype": adapter["factor_storage_dtype"], "matmul_dtype": "fp32",
                        "svd_dtype": "fp64", "algebraic_rank": adapter["rank"],
                        "numerical_rank": numerical_rank,
                        "stable_rank": total / max(float(singular[0].square()), 1e-30),
                        "effective_rank": v1.effective_rank(singular), "frobenius_norm": math.sqrt(total),
                    }
                    for epsilon in v1.EPSILONS:
                        row[f"r_epsilon_{epsilon:g}"] = v1.rank_at_epsilon(singular, epsilon)
                    for k in v1.K_VALUES:
                        if numerical_rank < k:
                            row[f"tail_energy_k{k}"] = None
                            row[f"rank_limit_k{k}"] = "NA_RANK_LIMIT"
                        else:
                            row[f"tail_energy_k{k}"] = 0.5 * float(energy[k:].sum())
                            row[f"rank_limit_k{k}"] = "OK"
                    rows.append(row)
                    spectra.extend({**identity, "singular_index": index + 1, "singular_value": float(value)}
                                   for index, value in enumerate(singular.tolist()))
    output = b0.MINI / "lora_B2_direct_BA_cpu_precompute.csv"
    spectrum_output = b0.MINI / "lora_B2_direct_BA_cpu_spectra.csv"
    atomic_csv(output, rows)
    atomic_csv(spectrum_output, spectra)
    atomic_json(b0.MINI / "lora_B2_direct_BA_cpu_precompute_manifest.json", {
        "schema_version": "cycle09_b2_direct_ba_cpu_v2", "status": "complete_with_declared_blocks",
        "rows": len(rows), "spectrum_rows": len(spectra), "blocked": blocked,
        "rank_limit_policy": "numerical rank under documented fp64 tolerance; nominal adapter rank is never used for NA_RANK_LIMIT",
        "method": "thin QR(B), then fp64 SVD of scaling*(R@A); no dense Delta W or merged model loaded",
        "outputs": [str(output), str(spectrum_output)],
    })
    print(json.dumps({"status": "complete_with_declared_blocks", "rows": len(rows),
                      "spectrum_rows": len(spectra), "blocked": len(blocked)}, indent=2))


if __name__ == "__main__":
    main()
