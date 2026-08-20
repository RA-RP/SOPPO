#!/usr/bin/env python3
"""CPU-only B2 precompute for direct LoRA B@A spectra and fixed-k tails.

The nonzero singular spectrum of B@A is computed from a thin QR of B followed
by an SVD of R@A.  This avoids materialising full dense updates while M6 owns
both GPUs.  Merged-state comparisons and all G/S quantities remain for B2.
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
from safetensors.torch import load_file

import sys

SCRIPTS = Path("/root/LLM-output-density/experiments/opd_sft_h1/scripts")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_b_lora_proxy as b0  # noqa: E402


MINI = b0.MINI
EPSILONS = (0.01, 0.025, 0.05, 0.10)
K_VALUES = (4, 8, 16, 32)


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


def rank_at_epsilon(singular: torch.Tensor, epsilon: float) -> int:
    energy = singular.square()
    total = float(energy.sum())
    if total == 0:
        return 0
    return int(torch.searchsorted(energy.cumsum(0), (1.0 - epsilon) * energy.sum()).item() + 1)


def effective_rank(singular: torch.Tensor) -> float:
    energy = singular.square()
    total = float(energy.sum())
    if total == 0:
        return 0.0
    p = energy / total
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


def direct_spectrum(adapter: dict[str, Any]) -> torch.Tensor:
    weights = load_file(adapter["weights"], device="cpu")
    a = weights[adapter["a_key"]].float()
    b = weights[adapter["b_key"]].float()
    scale = float(adapter["scaling"])
    # B = QR gives B@A = Q@(R@A), so singular values are preserved by Q.
    _, r = torch.linalg.qr(b, mode="reduced")
    return torch.linalg.svdvals((r @ a) * scale).double()


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
                    singular = direct_spectrum(adapter)
                    energy = singular.square()
                    total = float(energy.sum())
                    numerical_tol = float(singular.max()) * max(adapter["rank"], 1) * torch.finfo(torch.float64).eps
                    row = {
                        **identity, "status": "complete", "lora_rank": adapter["rank"],
                        "lora_alpha": adapter["alpha"], "lora_scaling": adapter["scaling"],
                        "factor_storage_dtype": adapter["factor_storage_dtype"], "matmul_dtype": "fp32",
                        "svd_dtype": "fp64", "algebraic_rank": adapter["rank"],
                        "numerical_rank": int((singular > numerical_tol).sum()),
                        "stable_rank": total / max(float(singular[0].square()), 1e-30),
                        "effective_rank": effective_rank(singular), "frobenius_norm": math.sqrt(total),
                    }
                    for epsilon in EPSILONS:
                        row[f"r_epsilon_{epsilon:g}"] = rank_at_epsilon(singular, epsilon)
                    for k in K_VALUES:
                        row[f"tail_energy_k{k}"] = 0.5 * float(energy[k:].sum()) if singular.numel() >= k else None
                        row[f"rank_limit_k{k}"] = "OK" if singular.numel() >= k else "NA_RANK_LIMIT"
                    rows.append(row)
                    spectra.extend({**identity, "singular_index": index + 1, "singular_value": float(value)}
                                   for index, value in enumerate(singular.tolist()))
    output = MINI / "lora_B2_direct_BA_cpu_precompute.csv"
    spectrum_output = MINI / "lora_B2_direct_BA_cpu_spectra.csv"
    atomic_csv(output, rows)
    atomic_csv(spectrum_output, spectra)
    atomic_json(MINI / "lora_B2_direct_BA_cpu_precompute_manifest.json", {
        "schema_version": "cycle09_b2_direct_ba_cpu_v1", "status": "complete_with_declared_blocks",
        "rows": len(rows), "spectrum_rows": len(spectra), "blocked": blocked,
        "method": "thin QR(B), then fp64 SVD of scaling*(R@A); no dense Delta W or merged model loaded",
        "outputs": [str(output), str(spectrum_output)],
    })
    print(json.dumps({"status": "complete_with_declared_blocks", "rows": len(rows),
                      "spectrum_rows": len(spectra), "blocked": len(blocked)}, indent=2))


if __name__ == "__main__":
    main()
