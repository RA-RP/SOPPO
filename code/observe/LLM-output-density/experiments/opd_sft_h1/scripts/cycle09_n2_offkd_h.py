#!/usr/bin/env python3
"""N-2 item 4: off-KD H_bos/H_ood generation and seven-step geometry.

The generation protocol is copied from the R4 campaign: the same prompt banks,
temperature/top-p/token cap, and per-request stable seed.  Geometry is delegated
to the shared H worker so off-KD updates always use the fp32 adapter B@A track.
Every corpus, reference, and checkpoint cache is atomic and resumable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import cycle09_r4_campaign as camp
import cycle09_r4_common as c4
from cycle09_n2_build_ref import build as build_references
from cycle09_n2_build_ref import h_pairs


ARM = "offkd"
STEPS = (0, 5, 10, 20, 40, 160, 624)
DOMAINS = ("bos", "ood")
SEEDS = tuple(c4.GENERATION_SEEDS)
OFFKD_ROOT = Path("/root/autodl-tmp/cycle09_offkd")
MERGED_ROOT = OFFKD_ROOT / "_merged_models"
RUN_ROOT = Path("/root/autodl-tmp/cycle09_n2/offkd_h")
MINI = c4.MINI_ROOT
SCRIPTS = Path(__file__).resolve().parent
GEOMETRY_WORKER = SCRIPTS / "cycle09_n2_h80_measure.py"
REFERENCE_ROOT = c4.RUN_ROOT / "scratch/references"
STATUS_PATH = RUN_ROOT / "STATUS.json"


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def update_status(stage: str, detail: str, *, state: str = "running") -> None:
    write_json_atomic(
        STATUS_PATH,
        {
            "state": state,
            "stage": stage,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "shutdown": "disabled",
        },
    )


def model_path(step: int) -> Path:
    path = c4.BASE_MODEL if step == 0 else MERGED_ROOT / c4.step_label(step)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"missing off-KD merged model for step {step}: {path}")
    return path


def adapter_path(step: int) -> Path | None:
    if step == 0:
        return None
    direct = OFFKD_ROOT / "checkpoints" / f"checkpoint-{step:06d}"
    if (direct / "adapter_model.safetensors").exists():
        return direct
    for parent in sorted((OFFKD_ROOT / "checkpoint_backfill").glob("*/")):
        candidate = parent / f"checkpoint-{step:06d}"
        if (candidate / "adapter_model.safetensors").exists():
            return candidate
    raise FileNotFoundError(f"missing off-KD fp32 adapter for step {step}")


def corpus_path(step: int, domain: str, seed: int) -> Path:
    return c4.generated_corpus_path("H", domain, seed, ARM, step, c4.RUN_ROOT)


def pending_batches(step: int) -> list[tuple[Path, str, int]]:
    return [
        (corpus_path(step, domain, seed), domain, seed)
        for seed in SEEDS
        for domain in DOMAINS
        if not camp.complete_corpus(corpus_path(step, domain, seed), c4.N_GENERATED)
    ]


def generate_corpora(steps: tuple[int, ...]) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    banks = c4.prompt_banks(c4.N_GENERATED)
    for step in steps:
        batches = pending_batches(step)
        if not batches:
            print(f"[N2-H generate] {c4.step_label(step)} complete; skip", flush=True)
            continue
        update_status("generation", f"{c4.step_label(step)} batches={len(batches)}")
        print(
            f"[N2-H generate] {c4.step_label(step)} pending batches={len(batches)}",
            flush=True,
        )
        llm = LLM(
            model=str(model_path(step)),
            dtype="bfloat16",
            gpu_memory_utilization=0.82,
            max_model_len=4096,
            seed=c4.WINDOW_SEED,
        )
        try:
            for target, domain, seed in batches:
                prompts: list[str] = []
                params: list[Any] = []
                for item in banks[domain]:
                    prompts.append(
                        camp.formatted_prompt(
                            tokenizer, item["prompt"], item["instruction"], domain
                        )
                    )
                    params.append(
                        SamplingParams(
                            temperature=c4.TEMPERATURE,
                            top_p=c4.TOP_P,
                            max_tokens=c4.MAX_NEW_TOKENS,
                            seed=c4.stable_seed(seed, "H", domain, item["sample_id"]),
                        )
                    )
                outputs = llm.generate(prompts, params)
                rows = []
                for item, formatted, output in zip(banks[domain], prompts, outputs):
                    completion = output.outputs[0]
                    prompt_ids = list(output.prompt_token_ids)
                    generation_ids = list(completion.token_ids)
                    rows.append(
                        {
                            "sample_id": item["sample_id"],
                            "probe_type": "H",
                            "domain": domain,
                            "source_kind": "checkpoint_nontraining_generation",
                            "arm": ARM,
                            "step": step,
                            "generation_seed": seed,
                            "per_request_seed": c4.stable_seed(
                                seed, "H", domain, item["sample_id"]
                            ),
                            "prompt_text": item["prompt"] + item["instruction"],
                            "formatted_prompt": formatted,
                            "generation_text": completion.text,
                            "prompt_token_ids": prompt_ids,
                            "generation_token_ids": generation_ids,
                            "full_token_ids": prompt_ids + generation_ids,
                            "eligible_start": len(prompt_ids),
                            "eligible_end": len(prompt_ids) + len(generation_ids),
                            "finish_reason": completion.finish_reason,
                            "generation_config": {
                                "max_new_tokens": c4.MAX_NEW_TOKENS,
                                "temperature": c4.TEMPERATURE,
                                "top_p": c4.TOP_P,
                                "batch_seed": seed,
                            },
                        }
                    )
                c4.write_jsonl_atomic(target, rows)
                print(f"[N2-H generate] {target} n={len(rows)}", flush=True)
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()


def validate_inputs(steps: tuple[int, ...]) -> None:
    missing = []
    for step in steps:
        model_path(step)
        adapter_path(step)
        for domain in DOMAINS:
            for seed in SEEDS:
                path = corpus_path(step, domain, seed)
                if not camp.complete_corpus(path, c4.N_GENERATED):
                    missing.append(str(path))
                task_id = f"H_{ARM}_{domain}__{c4.step_label(step)}__g{seed}"
                if not (REFERENCE_ROOT / f"{task_id}.pt").exists():
                    missing.append(str(REFERENCE_ROOT / f"{task_id}.pt"))
    if missing:
        raise RuntimeError(f"off-KD H inputs incomplete ({len(missing)}): {missing[:6]}")


def run_geometry(steps: tuple[int, ...], python: str) -> None:
    for step in steps:
        work_root = RUN_ROOT / "geometry" / c4.step_label(step)
        update_status("geometry", c4.step_label(step))
        command = [
            python,
            str(GEOMETRY_WORKER),
            "--arm",
            ARM,
            "--probe-step",
            str(step),
            "--domains",
            ",".join(DOMAINS),
            "--steps",
            "0" if step == 0 else f"0,{step}",
            "--work-root",
            str(work_root),
        ]
        print(f"[N2-H geometry] START {c4.step_label(step)}", flush=True)
        subprocess.run(command, cwd=SCRIPTS, check=True)
        print(f"[N2-H geometry] DONE {c4.step_label(step)}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_rows(path: Path, key: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("arm") == ARM and row.get(key, "").startswith("H_offkd_")
    ]


def finalize(steps: tuple[int, ...]) -> None:
    files = {
        "spectra": MINI / "R4_v2_spectra_h_offkd.csv",
        "m1": MINI / "R4_m1_tail_ec.csv",
        "m2": MINI / "R4_m2_output_drift.csv",
        "theta": MINI / "R5_theta_reps.csv",
    }
    rows = {
        "spectra": relevant_rows(files["spectra"], "task_id"),
        "m1": relevant_rows(files["m1"], "task_id"),
        "m2": relevant_rows(files["m2"], "task_id"),
        "theta": relevant_rows(files["theta"], "probe"),
    }
    task_count = len(DOMAINS) * len(SEEDS)
    cell_count = task_count * len(c4.LAYERS) * len(c4.MODULES)
    expected = {
        "spectra": len(steps) * cell_count * 2,
        "m1": len(steps) * cell_count * 2 * 2,
        "m2": len(steps) * task_count * len(c4.LAYERS) * (len(c4.MODULES) * 2 + 1),
        "theta": (len(steps) - int(0 in steps)) * cell_count * 3,
    }
    actual = {name: len(values) for name, values in rows.items()}
    if actual != expected:
        raise RuntimeError(f"off-KD H row mismatch: expected={expected}, actual={actual}")
    expected_steps = {str(step) for step in steps}
    for name, values in rows.items():
        seen = {str(row["step"]) for row in values}
        if seen != expected_steps - ({"0"} if name == "theta" else set()):
            raise RuntimeError(f"off-KD H {name} step mismatch: {seen}")

    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": ARM,
        "steps": list(steps),
        "domains": list(DOMAINS),
        "generation_seeds": list(SEEDS),
        "n_generated_per_cell": c4.N_GENERATED,
        "generation_protocol": {
            "temperature": c4.TEMPERATURE,
            "top_p": c4.TOP_P,
            "max_new_tokens": c4.MAX_NEW_TOKENS,
            "seed_rule": "stable_seed(batch_seed, probe_type, domain, sample_id)",
        },
        "window_protocol": "R4 v2: generation-only windows; 512 tokens; k=3; sample-equal",
        "dW_track": "off-KD adapter B@A fp32 only",
        "theta_numerics": "fp64 SVD plus fp64 QR re-orthonormalization",
        "rows": actual,
        "files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in files.items()
        },
        "models": {str(step): str(model_path(step)) for step in steps},
        "adapters": {
            str(step): (str(adapter_path(step)) if step else "base") for step in steps
        },
        "shutdown": "disabled",
    }
    write_json_atomic(MINI / "offkd_h_geometry_manifest.json", manifest)
    update_status("complete", "all seven off-KD H checkpoints validated", state="complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.steps.split(",") if value)
    unknown = sorted(set(steps).difference(STEPS))
    if unknown:
        raise SystemExit(f"unsupported off-KD H steps: {unknown}")
    for step in steps:
        model_path(step)
        adapter_path(step)

    try:
        if not args.skip_generation:
            generate_corpora(steps)
        if not args.skip_references:
            update_status("references", f"steps={list(steps)}")
            build_references(h_pairs(ARM, steps, DOMAINS, SEEDS))
        validate_inputs(steps)
        if not args.skip_geometry:
            run_geometry(steps, args.python)
        finalize(steps)
    except BaseException as error:
        update_status("failed", repr(error), state="failed")
        raise


if __name__ == "__main__":
    main()
