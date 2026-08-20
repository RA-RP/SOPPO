#!/usr/bin/env python3
"""D4.1 Qwen full merged-state cells with ephemeral BF16 materialization.

The main state quantity is W_t S_{D,t}.  Adapter BA is deliberately absent
from this script except for PEFT's BF16 materialization step.  Every completed
cell has a separate numerical protocol record and its temporary merged model
is deleted in a finally block.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_actual_output_trajectory as actual  # noqa: E402
import cycle09_block3_common as b3  # noqa: E402
import cycle09_block3_qwen_probe_geometry as qprobe  # noqa: E402
import cycle09_offkd_eval as offkd_eval  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402
import cycle09_r4_common as c4  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402
from scripts.run_opd_minimal_closure import merge_lora_adapter  # noqa: E402


ROOT = b3.AUTODL / "cycle09_relative_functional_contraction/d4_merged_state"
TEMP_ROOT = b3.AUTODL / "cycle09_relative_functional_contraction/d4_materialized/qwen"
H2_CORPUS = (
    b3.AUTODL
    / "cycle09_stage3_followup/partitions/partition_probe_qwen_20260723/H2_probe_core/corpora/qwen3_4b/E_math.jsonl"
)
CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")
MODULES = tuple(qstage.MODULES)
EPSILONS = (0.01, 0.025, 0.05, 0.10)
LAYER = 18


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            value.update(part)
    return value.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@contextmanager
def lock(path: Path):
    import fcntl

    target = path.with_suffix(path.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def label(step: int) -> str:
    return f"step_{int(step):03d}"


def adapter_path(arm: str, step: int) -> Path:
    if arm == "sft":
        return b3.AUTODL / "cycle07_base_sft_trajectory/checkpoints" / label(step)
    if arm == "offkd":
        return offkd_eval.adapter_path(b3.AUTODL / "cycle09_offkd", step)
    if arm == "seqkd":
        return b3.AUTODL / "cycle09_seqkd/checkpoints" / f"checkpoint-{int(step):06d}"
    raise ValueError(f"adapter path unavailable for arm={arm}")


def assert_adapter(arm: str, step: int) -> Path:
    path = adapter_path(arm, step)
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    missing = [str(item) for item in required if not item.is_file()]
    if arm in ("offkd", "seqkd") and not (path / "complete.json").is_file():
        missing.append(str(path / "complete.json"))
    if missing:
        raise FileNotFoundError(f"incomplete adapter arm={arm} step={step}: {missing}")
    return path


def temporary_path(arm: str, step: int) -> Path:
    return TEMP_ROOT / arm / label(step)


def materialize_base(target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        str(qstage.BASE_MODEL), torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    try:
        model.save_pretrained(str(target), safe_serialization=True)
        AutoTokenizer.from_pretrained(str(qstage.BASE_MODEL), trust_remote_code=True).save_pretrained(str(target))
    finally:
        del model
        gc.collect()
    return target


@contextmanager
def materialized_model(arm: str, step: int):
    target = temporary_path(arm, step)
    try:
        # A killed process can leave an incomplete copy/merge behind. It is
        # strictly ephemeral and never a result artifact, so clear it first.
        if target.exists():
            shutil.rmtree(target)
        if step == 0:
            materialize_base(target)
        elif arm == "opd":
            source = qstage.model_path("opd", step)
            if not qstage.model_integrity(source).get("complete"):
                raise FileNotFoundError(f"OPD merged checkpoint missing: {source}")
            shutil.copytree(source, target)
        else:
            merge_lora_adapter(qstage.BASE_MODEL, assert_adapter(arm, step), target)
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


def load_bf16(path: Path, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval().to(device)
    return model


def unload(model: Any) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_path(probe: str) -> Path:
    if probe == "E_math":
        return H2_CORPUS
    return qprobe.corpus_path(probe)


def samples_for(probe: str, sample_limit: int) -> list[Any]:
    if probe not in CORE_PROBES:
        raise ValueError(f"unsupported probe={probe}")
    tokenizer = qprobe.load_qwen_tokenizer()
    source = sample_path(probe)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = c4.prepare_samples(
        source, tokenizer, corpus_id=f"qwen_d4_merged_state:{probe}",
        window_seed=c4.WINDOW_SEED, max_context_tokens=c4.MAX_CONTEXT_TOKENS,
    )
    if sample_limit:
        rows = rows[:sample_limit]
    if not rows:
        raise RuntimeError(f"empty frozen probe: {probe}")
    return rows


def root_for(tag: str) -> Path:
    return ROOT / tag


def state_path(tag: str, arm: str, step: int, probe: str) -> Path:
    return root_for(tag) / "state" / arm / label(step) / f"{probe}.json"


def output_path(tag: str, arm: str, step: int, probe: str) -> Path:
    return root_for(tag) / "outputs" / arm / label(step) / f"{probe}.json"


def base_cache_path(tag: str, probe: str) -> Path:
    return root_for(tag) / "scratch" / "base_logits" / f"{probe}.pt"


def base_cache_meta(tag: str, probe: str) -> Path:
    return base_cache_path(tag, probe).with_suffix(".json")


def sqrt_gram(gram: torch.Tensor, device: str) -> torch.Tensor:
    symmetric = ((gram + gram.T) / 2).to(device=device, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(symmetric)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T


def ranks(singular: torch.Tensor) -> dict[str, int]:
    energy = singular.to(torch.float64).square()
    total = energy.sum()
    if float(total) == 0:
        return {str(epsilon): 0 for epsilon in EPSILONS}
    cumulative = energy.cumsum(0)
    return {
        str(epsilon): int(torch.searchsorted(cumulative, (1.0 - epsilon) * total).item() + 1)
        for epsilon in EPSILONS
    }


def profile_state(model: Any, samples: list[Any], device: str, batch_size: int, max_batch_tokens: int) -> dict[str, Any]:
    return campaign.collect_profile(
        model, samples, [LAYER], device, keep_factors=False, keep_residual_samples=False,
        keep_input_sample_means=False, factor_layers=(LAYER,), forward_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens, early_stop=True,
    )


def build_base_cache(tag: str, probe: str, samples: list[Any], device: str) -> dict[str, Any]:
    target = base_cache_path(tag, probe)
    metadata = base_cache_meta(tag, probe)
    with lock(target):
        cached = read_json(metadata, {})
        if cached.get("status") == "complete" and target.is_file():
            return torch.load(target, map_location="cpu", weights_only=True)
        with materialized_model("base", 0) as merged:
            model = load_bf16(merged, device)
            try:
                records = []
                for sample in samples:
                    positions = actual.selected_positions(sample, 0)
                    records.append({
                        "sample_id": sample.sample_id,
                        "positions": positions.cpu(),
                        "token_weights": sample.token_weights[positions].float().cpu(),
                        "target_ids": sample.input_ids[0, positions].long().cpu(),
                        "logits": actual.selected_logits(model, sample, positions, device).cpu(),
                    })
            finally:
                unload(model)
        value = {
            "status": "complete", "schema_version": "cycle09_qwen_d4_base_logits_v1",
            "probe": probe, "sample_ids_sha256": json_digest([item.sample_id for item in samples]),
            "records": records, "logit_forward_dtype": "bf16", "logit_storage_dtype": "bf16",
            "created_utc": now(),
        }
        atomic_torch(target, value)
        atomic_json(metadata, {
            key: value[key] for key in ("status", "schema_version", "probe", "sample_ids_sha256", "logit_forward_dtype", "logit_storage_dtype", "created_utc")
        } | {"artifact": str(target), "bytes": target.stat().st_size, "sha256": digest(target)})
        return value


def output_rows(model: Any, samples: list[Any], baseline: dict[str, Any], device: str) -> list[dict[str, Any]]:
    source = {sample.sample_id: sample for sample in samples}
    rows = []
    for record in baseline["records"]:
        sample = source[record["sample_id"]]
        positions = record["positions"].long()
        base_logits = record["logits"].to(device=device, dtype=torch.float32)
        current_logits = actual.selected_logits(model, sample, positions, device).to(
            device=device, dtype=torch.float32
        )
        log_base = torch.log_softmax(base_logits, dim=-1)
        log_current = torch.log_softmax(current_logits, dim=-1)
        probability = log_base.exp()
        weights = record["token_weights"].to(device=device, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-30)
        token_ids = record["target_ids"].to(device=device, dtype=torch.long)
        kl = (probability * (log_base - log_current)).sum(dim=-1)
        nll_base = -log_base.gather(1, token_ids[:, None]).squeeze(1)
        nll_current = -log_current.gather(1, token_ids[:, None]).squeeze(1)
        delta = nll_current - nll_base
        rows.append({
            "sample_id": record["sample_id"], "token_count": int(len(positions)),
            "cumulative_kl_base_to_current": float((weights * kl).sum()),
            "nll_base": float((weights * nll_base).sum()), "nll_current": float((weights * nll_current).sum()),
            "delta_nll_cumulative": float((weights * delta).sum()),
            "absolute_delta_nll_cumulative": float((weights * delta.abs()).sum()),
        })
        del base_logits, current_logits, log_base, log_current, probability, kl, nll_base, nll_current, delta
        torch.cuda.empty_cache()
    return rows


def build_base_caches_memory(args: argparse.Namespace) -> dict[str, tuple[list[Any], dict[str, Any]]]:
    """Calculate all full-vocabulary base caches once, retaining them in host RAM."""
    prepared = {probe: samples_for(probe, args.sample_limit) for probe in CORE_PROBES}
    caches: dict[str, tuple[list[Any], dict[str, Any]]] = {}
    with materialized_model("base", 0) as merged:
        model = load_bf16(merged, args.device)
        try:
            for probe, samples in prepared.items():
                records = []
                for sample in samples:
                    positions = actual.selected_positions(sample, 0)
                    records.append({
                        "sample_id": sample.sample_id,
                        "positions": positions.cpu(),
                        "token_weights": sample.token_weights[positions].float().cpu(),
                        "target_ids": sample.input_ids[0, positions].long().cpu(),
                        "logits": actual.selected_logits(model, sample, positions, args.device).cpu(),
                    })
                caches[probe] = (samples, {
                    "status": "complete", "schema_version": "cycle09_qwen_d4_base_logits_memory_v1",
                    "probe": probe, "sample_ids_sha256": json_digest([item.sample_id for item in samples]),
                    "records": records, "logit_forward_dtype": "bf16", "logit_storage_dtype": "bf16",
                    "created_utc": now(),
                })
        finally:
            unload(model)
    return caches


def merge_route(arm: str, step: int) -> str:
    if step == 0:
        return "bf16_cpu_save_pretrained_materialization"
    if arm == "opd":
        return "bf16_existing_serialized_merged_copy"
    return "bf16_cpu_peft_merge_and_unload"


def run_loaded_cell(
    args: argparse.Namespace,
    model: Any,
    arm: str,
    probe: str,
    samples: list[Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    target = state_path(args.tag, arm, args.step, probe)
    output = output_path(args.tag, arm, args.step, probe)
    with lock(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete" and read_json(output, {}).get("status") == "complete":
            return cached | {"artifact": str(target), "output_artifact": str(output)}
        measured = profile_state(model, samples, args.device, args.forward_batch_size, args.max_batch_tokens)
        rows = output_rows(model, samples, baseline, args.device)
        spectra, state_rows = {}, []
        for module in MODULES:
            group = c4.MODULE_TO_GROUP[module]
            scale = sqrt_gram(measured["grams"][LAYER][group], args.device)
            weight = campaign.module_at(model, LAYER, module).weight.detach().to(args.device, torch.float32)
            singular = torch.linalg.svdvals((weight @ scale.to(torch.float32)).to(torch.float64))
            for epsilon, value in ranks(singular).items():
                state_rows.append({"module": module, "epsilon": float(epsilon), "r_epsilon": value})
            spectra[module] = singular.cpu().tolist()
            del scale, weight, singular
            torch.cuda.empty_cache()
        protocol = {
            "checkpoint_storage_dtype": "bf16_safetensors_on_disk",
            "merge_compute_dtype": merge_route(arm, args.step),
            "model_load_dtype": "bf16",
            "activation_dtype": "bf16_model_hidden_states",
            "gram_and_whitening_dtype": "fp64_eigh_clamp_nonnegative",
            "WS_matmul_dtype": "fp32",
            "svd_input_dtype": "fp64",
            "singular_value_accumulation_dtype": "fp64",
            "logit_forward_dtype": "bf16",
            "logit_storage_dtype": "bf16",
            "KL_NLL_compute_dtype": "fp32",
        }
        payload = {
            "schema_version": "cycle09_qwen_d4_merged_state_v1", "status": "complete",
            "model": "qwen", "arm": arm, "checkpoint": args.step, "probe_name": probe,
            "layer": LAYER, "sample_count": len(samples),
            "sample_ids_sha256": json_digest([item.sample_id for item in samples]),
            "state_rows": state_rows, "spectra": spectra, "numerical_protocol": protocol,
            "merged_model_lifecycle": "ephemeral_bf16_on_disk_deleted_after_atomic_outputs",
            "created_utc": now(),
        }
        output_payload = {
            "schema_version": "cycle09_qwen_d4_fixed_token_output_v1", "status": "complete",
            "model": "qwen", "arm": arm, "checkpoint": args.step, "probe_name": probe,
            "sample_count": len(samples), "sample_ids_sha256": payload["sample_ids_sha256"],
            "rows": rows, "numerical_protocol": protocol,
            "merged_model_lifecycle": payload["merged_model_lifecycle"], "created_utc": now(),
        }
        atomic_json(target, payload)
        atomic_json(output, output_payload)
        return payload | {"artifact": str(target), "output_artifact": str(output)}


def prepare_cell(args: argparse.Namespace, arm: str, probe: str) -> tuple[list[Any], dict[str, Any]] | None:
    target = state_path(args.tag, arm, args.step, probe)
    output = output_path(args.tag, arm, args.step, probe)
    if read_json(target, {}).get("status") == "complete" and read_json(output, {}).get("status") == "complete":
        return None
    samples = samples_for(probe, args.sample_limit)
    baseline = build_base_cache(args.tag, probe, samples, args.device)
    if baseline["sample_ids_sha256"] != json_digest([item.sample_id for item in samples]):
        raise RuntimeError("base-logit cache sample IDs do not match frozen probe")
    return samples, baseline


def run_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    arm = "base" if args.step == 0 else args.arm
    probes = CORE_PROBES if args.probes == "all" else tuple(args.probes.split(","))
    invalid = sorted(set(probes) - set(CORE_PROBES))
    if invalid:
        raise ValueError(f"unsupported probes: {invalid}")
    prepared = [(probe, value) for probe in probes if (value := prepare_cell(args, arm, probe)) is not None]
    if not prepared:
        return {"status": "complete", "arm": arm, "checkpoint": args.step, "cells": [], "cached": True}
    with materialized_model(arm, args.step) as merged:
        model = load_bf16(merged, args.device)
        try:
            results = [
                run_loaded_cell(args, model, arm, probe, samples, baseline)
                for probe, (samples, baseline) in prepared
            ]
        finally:
            unload(model)
    return {"status": "complete", "arm": arm, "checkpoint": args.step, "cells": results, "created_utc": now()}


def run_formal_memory(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full Qwen grid with base logits held in host RAM, never on data disk."""
    if args.sample_limit:
        raise ValueError("formal-memory requires the full frozen probe corpora")
    grid = [("base", 0)] + [
        (arm, step)
        for arm in ("opd", "sft", "offkd", "seqkd")
        for step in (5, 10, 20, 40, 80, 160, 320, 480, 624)
    ]

    def pending(arm: str, step: int) -> bool:
        return any(
            not (
                read_json(state_path(args.tag, arm, step, probe), {}).get("status") == "complete"
                and read_json(output_path(args.tag, arm, step, probe), {}).get("status") == "complete"
            )
            for probe in CORE_PROBES
        )

    todo = [(arm, step) for arm, step in grid if pending(arm, step)]
    progress_path = root_for(args.tag) / "formal_memory_progress.json"
    if not todo:
        atomic_json(progress_path, {
            "schema_version": "cycle09_qwen_d4_formal_memory_v1", "status": "complete",
            "base_logit_cache": "host_ram_only", "grid": grid, "completed_utc": now(),
        })
        return {"status": "complete", "cached": True, "progress": str(progress_path)}

    caches = build_base_caches_memory(args)
    cache_summary = {
        probe: {
            "sample_count": len(samples),
            "sample_ids_sha256": baseline["sample_ids_sha256"],
            "storage": "host_ram_only_bf16",
        }
        for probe, (samples, baseline) in caches.items()
    }
    completed = []
    for sequence, (arm, step) in enumerate(todo, start=1):
        active = [probe for probe in CORE_PROBES if pending(arm, step)]
        cell_args = argparse.Namespace(**vars(args))
        cell_args.arm = arm
        cell_args.step = step
        with materialized_model(arm, step) as merged:
            model = load_bf16(merged, args.device)
            try:
                cell_results = [
                    run_loaded_cell(cell_args, model, arm, probe, *caches[probe])
                    for probe in active
                ]
            finally:
                unload(model)
        completed.append({
            "sequence": sequence, "arm": arm, "checkpoint": step,
            "probes": active, "artifacts": [result["artifact"] for result in cell_results], "completed_utc": now(),
        })
        atomic_json(progress_path, {
            "schema_version": "cycle09_qwen_d4_formal_memory_v1", "status": "running",
            "base_logit_cache": "host_ram_only", "cache_summary": cache_summary,
            "grid": grid, "remaining_checkpoints": len(todo) - sequence,
            "completed_checkpoints": completed, "updated_utc": now(),
        })
    del caches
    gc.collect()
    atomic_json(progress_path, {
        "schema_version": "cycle09_qwen_d4_formal_memory_v1", "status": "complete",
        "base_logit_cache": "host_ram_only", "cache_summary": cache_summary,
        "grid": grid, "completed_checkpoints": completed, "completed_utc": now(),
    })
    return {"status": "complete", "progress": str(progress_path), "checkpoints": len(todo)}


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    args.probes = args.probe
    result = run_checkpoint(args)
    return result["cells"][0] if result["cells"] else {"status": "complete", "cached": True}


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.tag = "smoke"
    smoke_args.probe = "E_math"
    smoke_args.sample_limit = 2
    base_args = vars(smoke_args).copy()
    base_args.update(arm="base", step=0)
    offline_args = vars(smoke_args).copy()
    offline_args.update(arm="offkd", step=20)
    base = argparse.Namespace(**base_args)
    offline = argparse.Namespace(**offline_args)
    first, second = run_cell(base), run_cell(offline)
    return {"status": "complete", "base": first["artifact"], "offline": second["artifact"], "created_utc": now()}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", "cell", "checkpoint", "formal-memory"))
    parser.add_argument("--arm", default="offkd", choices=("base", "opd", "sft", "offkd", "seqkd"))
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--probe", default="E_math", choices=CORE_PROBES)
    parser.add_argument("--probes", default="all", help="comma-separated core probes, or all for checkpoint phase")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.phase == "smoke":
        result = smoke(args)
    elif args.phase == "checkpoint":
        result = run_checkpoint(args)
    elif args.phase == "formal-memory":
        result = run_formal_memory(args)
    else:
        result = run_cell(args)
    print(json.dumps({"status": result.get("status"), "artifact": result.get("artifact"), "created_utc": now()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
