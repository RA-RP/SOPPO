#!/usr/bin/env python3
"""Resumable two-GPU campaign for the remaining Cycle 09 numerical tracks.

The campaign deliberately has two serial GPU phases:

1. Llama OPD/off-KD at steps 5,20,40,80,160,320 receives two *labelled*
   weight objects on the same v2 fixed-support profile: direct LoRA B@A in
   fp32 and the effective difference of the persisted merged bf16 models.
2. The Qwen appendix closure (Math-CoT held-out, Numina, AIME25) runs from a
   dynamic two-worker queue.  Historical Qwen adapters are merged only into a
   per-worker scratch directory and are deleted after their cell completes.

No paper or Theory handoff file is written here.  Every completed cell has an
atomic JSON marker, so the detached supervisor is safe to resume after an
instance or terminal interruption.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
EVAL_COMPONENT = REPO / "Eval/component"
for _path in (SCRIPTS, EVAL_COMPONENT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_geometry as lgeom  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402
import cycle09_r4_common as c4  # noqa: E402
from scorer import extract_pred  # noqa: E402
from scorer_v2 import score  # noqa: E402


ROOT = Path("/root/autodl-tmp/cycle09_dual_m6")
DUAL = ROOT / "llama_dual_numeric"
M6 = ROOT / "qwen_m6"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

LLAMA_ARMS = ("opd", "offkd")
LLAMA_STEPS = (5, 20, 40, 80, 160, 320)
LLAMA_PROBES = ("E_general", "E_math", "E_ood", "E_if")
LLAMA_LAYER = 14
QWEN_ARMS = ("opd", "sft", "offkd", "seqkd")
MATHCOT_STEPS = (0, 20, 40, 160, 624)
NUMINA_STEPS = (40, 160, 624)
EPSILONS = (0.01, 0.025, 0.05, 0.10)
WINDOW_SEED = 42
M6_SEED = 20260725
MATHCOT_EVAL_N = 256
MATHCOT_PROBE_N = 32
NUMINA_EVAL_N = 200
NUMINA_PROBE_N = 128
AIME25_N = 30
AIME25_SEEDS = tuple(range(42, 52))

QWEN_BASE = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
QWEN_OPD = Path("/root/autodl-tmp/cycle08_opd_trajectory/_merged_models")
QWEN_SFT = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")
QWEN_OFFKD = Path("/root/autodl-tmp/cycle09_offkd")
QWEN_SEQKD = Path("/root/autodl-tmp/cycle09_seqkd")
MATHCOT_PARQUET = Path("/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet")
NUMINA_JSONL = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl")
AIME25_JSONL = Path("/root/autodl-tmp/dataset/aime25/test.jsonl")

QWEN_MODULES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
QWEN_GROUPS = c4.MODULE_TO_GROUP


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def label(step: int) -> str:
    return f"step_{int(step):03d}"


def dual_root() -> Path:
    """Keep smoke completion markers separate from the formal run."""
    namespace = os.environ.get("CYCLE09_DUAL_NAMESPACE", "formal")
    return DUAL if namespace == "formal" else DUAL / namespace


def m6_root() -> Path:
    namespace = os.environ.get("CYCLE09_M6_NAMESPACE", "formal")
    return M6 if namespace == "formal" else M6 / namespace


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def locked(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def disk_guard(required_gib: int = 50) -> None:
    free = shutil.disk_usage("/root/autodl-tmp").free
    required = required_gib << 30
    if free < required:
        raise RuntimeError(f"disk guard: free={free / (1 << 30):.1f}GiB < {required_gib}GiB")


def unload(model: Any | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_bf16(path: Path, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.eval().to(device)
    return model


def module_at(model: Any, layer: int, name: str) -> Any:
    node = model.model.layers[int(layer)]
    for part in name.split("."):
        node = getattr(node, part)
    return node


def group(module: str) -> str:
    return QWEN_GROUPS[module]


def sqrt_psd(matrix: torch.Tensor, device: str) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(((matrix + matrix.T) / 2).to(device, torch.float64))
    return ((vectors * values.clamp_min(0).sqrt()) @ vectors.T).float()


def functional_rank_values(singular: torch.Tensor, epsilon: float) -> int:
    """Rank lookup from one already-computed singular spectrum."""
    energy = singular.float().square()
    total = float(energy.sum())
    if total <= 0:
        return 0
    return int(torch.searchsorted(energy.cumsum(0), (1.0 - epsilon) * energy.sum()).item() + 1)


def effective_rank(matrix: torch.Tensor) -> float:
    energy = torch.linalg.svdvals(matrix.float()).square()
    total = energy.sum()
    if float(total) <= 0:
        return 0.0
    p = energy / total
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


def llama_adapter_delta(arm: str, step: int, layer: int, module: str, device: str) -> torch.Tensor:
    """The exact LoRA B@A update, separately from persisted merged bf16 weights."""
    from safetensors.torch import load_file

    adapter = lexport.adapter_target(arm, step)
    info = lexport.validate_adapter(adapter, arm, step)
    weights = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    a_suffix = f"layers.{layer}.{module}.lora_A.weight"
    b_suffix = f"layers.{layer}.{module}.lora_B.weight"
    a_keys = [key for key in weights if key.endswith(a_suffix)]
    b_keys = [key for key in weights if key.endswith(b_suffix)]
    if len(a_keys) != 1 or len(b_keys) != 1:
        raise RuntimeError(f"missing/ambiguous Llama adapter tensor {arm}/{step}/{layer}/{module}")
    return (weights[b_keys[0]].float() @ weights[a_keys[0]].float()).mul_(
        float(info["alpha"]) / float(info["rank"])
    ).to(device)


def llama_profile_path(arm: str, step: int, probe: str) -> Path:
    return dual_root() / "profiles" / arm / label(step) / f"{probe}.pt"


def llama_profile_meta(arm: str, step: int, probe: str) -> Path:
    return llama_profile_path(arm, step, probe).with_suffix(".json")


def llama_base_profile_path(probe: str) -> Path:
    return dual_root() / "profiles" / "base" / f"{probe}.pt"


def llama_base_profile_meta(probe: str) -> Path:
    return llama_base_profile_path(probe).with_suffix(".json")


def collect_llama_profile(model: Any, probe: str, device: str, measurement_n: int) -> tuple[dict[str, Any], list[Any]]:
    tokenizer = b3.load_llama_tokenizer()
    samples = lgeom.prepare_samples(tokenizer, probe, measurement_n)
    profile = campaign.collect_profile(
        model, samples, [LLAMA_LAYER], device,
        keep_factors=False, keep_residual_samples=False, keep_input_sample_means=False,
        factor_layers=(), forward_batch_size=4, max_batch_tokens=16384, early_stop=True,
    )
    return profile, samples


def save_profile(path: Path, meta_path: Path, profile: dict[str, Any], samples: list[Any], **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(profile, temporary)
    os.replace(temporary, path)
    atomic_json(meta_path, {
        "status": "complete", "profile": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "sample_count": len(samples),
        "sample_ids_sha256": sha256_json([sample.sample_id for sample in samples]),
        "window_protocol": "v2: window token mean -> sample window mean -> sample equal mean",
        "window_seed": WINDOW_SEED, "created_utc": utc_now(), **meta,
    })


def ensure_llama_base_profile(probe: str, device: str, measurement_n: int) -> dict[str, Any]:
    path, meta = llama_base_profile_path(probe), llama_base_profile_meta(probe)
    with locked(path):
        if read_json(meta, {}).get("status") == "complete" and path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)
        base = load_bf16(b3.LLAMA_STUDENT_RUNTIME, device)
        try:
            profile, samples = collect_llama_profile(base, probe, device, measurement_n)
            save_profile(path, meta, profile, samples, model="llama", arm="base", step=0, probe=probe,
                         storage_dtype="bfloat16")
            return profile
        finally:
            unload(base)


def dual_cell_path(arm: str, step: int, probe: str) -> Path:
    return dual_root() / "cells" / arm / label(step) / f"{probe}.json"


def run_dual_cell(arm: str, step: int, probe: str, device: str, measurement_n: int) -> dict[str, Any]:
    """Measure both numerical objects from one profile/current-model pair."""
    target = dual_cell_path(arm, step, probe)
    with locked(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        disk_guard()
        base_profile = ensure_llama_base_profile(probe, device, measurement_n)
        profile_path, profile_meta = llama_profile_path(arm, step, probe), llama_profile_meta(arm, step, probe)
        base, current = None, None
        try:
            base = load_bf16(b3.LLAMA_STUDENT_RUNTIME, device)
            current_path = lexport.merged_target(arm, step)
            if not b3.model_check(current_path)["complete"]:
                raise FileNotFoundError(f"missing persisted merged bf16 Llama model: {current_path}")
            current = load_bf16(current_path, device)
            if read_json(profile_meta, {}).get("status") == "complete" and profile_path.is_file():
                profile = torch.load(profile_path, map_location="cpu", weights_only=True)
            else:
                profile, samples = collect_llama_profile(current, probe, device, measurement_n)
                save_profile(profile_path, profile_meta, profile, samples, model="llama", arm=arm, step=step,
                             probe=probe, storage_dtype="bfloat16")
            rows: list[dict[str, Any]] = []
            spectra: list[dict[str, Any]] = []
            for module in b3.MODULES:
                key = group(module)
                s0 = sqrt_psd(base_profile["grams"][LLAMA_LAYER][key], device)
                st = sqrt_psd(profile["grams"][LLAMA_LAYER][key], device)
                w0 = module_at(base, LLAMA_LAYER, module).weight.detach().to(device, torch.float32)
                w_final = module_at(current, LLAMA_LAYER, module).weight.detach().to(device, torch.float32)
                dw_ba = llama_adapter_delta(arm, step, LLAMA_LAYER, module, device)
                dw_final = w_final - w0
                track_objects = (
                    ("adapter_BA_fp32", dw_ba, w0 + dw_ba),
                    ("final_merged_bf16_effective_difference", dw_final, w_final),
                )
                delta_gap = float(torch.linalg.norm(dw_final - dw_ba) / torch.linalg.norm(dw_ba).clamp_min(1e-30))
                for track, delta, wt in track_objects:
                    state = wt @ st
                    displacement = delta @ s0
                    state_singular = torch.linalg.svdvals(state)
                    displacement_singular = torch.linalg.svdvals(displacement)
                    singular = state_singular.double().cpu().tolist()
                    spectra.append({
                        "arm": arm, "checkpoint": step, "probe_name": probe, "layer": LLAMA_LAYER,
                        "module": module, "delta_track": track, "singular_values": singular,
                    })
                    for epsilon in EPSILONS:
                        rows.append({
                            "model": "llama", "arm": arm, "checkpoint": step, "probe_name": probe,
                            "layer": LLAMA_LAYER, "module": module, "epsilon": epsilon,
                            "delta_track": track, "weight_object": track,
                            "state_rank": functional_rank_values(state_singular, epsilon),
                            "displacement_rank": functional_rank_values(displacement_singular, epsilon),
                            "displacement_norm_raw": float(torch.linalg.norm(displacement)),
                            "displacement_norm_normalized": float(
                                torch.linalg.norm(displacement) / torch.linalg.norm(w0 @ s0).clamp_min(1e-30)
                            ),
                            "weight_norm_fro": float(torch.linalg.norm(delta)),
                            "weight_effective_rank": effective_rank(delta),
                            "final_minus_ba_relative_fro": delta_gap,
                            "profile_weight_storage": "persisted_merged_bfloat16",
                            "arithmetic": "fp32_after_bfloat16_load" if track.startswith("final_") else "adapter_BA_fp32",
                            "support": "fixed_base_v2", "sample_count": int(profile["n_samples"]),
                        })
                del s0, st, w0, w_final, dw_ba, dw_final, state_singular, displacement_singular
                torch.cuda.empty_cache()
            spectra_path = target.with_suffix(".spectra.json")
            atomic_json(spectra_path, {"status": "complete", "rows": spectra})
            payload = {
                "schema_version": "cycle09_llama_dual_numeric_v1", "status": "complete",
                "arm": arm, "checkpoint": step, "probe": probe, "layer": LLAMA_LAYER,
                "rows": rows, "spectra": str(spectra_path),
                "base_profile": str(llama_base_profile_path(probe)), "current_profile": str(profile_path),
                "weight_objects": {
                    "adapter_BA_fp32": "direct PEFT B@A, scale=alpha/rank, fp32",
                    "final_merged_bf16_effective_difference": "persisted merged bf16 checkpoint minus persisted base bf16, arithmetic fp32",
                },
                "created_utc": utc_now(),
            }
            atomic_json(target, payload)
            return payload
        finally:
            unload(current)
            unload(base)


def dual_worker(args: argparse.Namespace) -> None:
    arms = [item for item in args.arms.split(",") if item]
    steps = [int(item) for item in args.steps.split(",") if item]
    probes = [item for item in args.probes.split(",") if item]
    for arm in arms:
        for step in steps:
            for probe in probes:
                value = run_dual_cell(arm, step, probe, args.device, args.measurement_n)
                print(f"[dual] {arm}/{step}/{probe}: {value['status']}", flush=True)


def normalize_question(value: str) -> str:
    return " ".join(value.split())


def qwen_prompt(tokenizer: Any, problem: str) -> tuple[list[int], int, int]:
    prompt = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = [int(value) for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
    return ids, 0, len(ids)


def probe_row(
    tokenizer: Any, sample_id: str, problem: str, *, domain: str,
    answer: str | None = None, reference_response: str | None = None,
) -> dict[str, Any]:
    ids, start, end = qwen_prompt(tokenizer, problem)
    if reference_response is not None:
        response_ids = [int(value) for value in tokenizer(reference_response, add_special_tokens=False)["input_ids"]]
        if not response_ids:
            raise RuntimeError(f"empty reference response for {sample_id}")
        start, end = len(ids), len(ids) + len(response_ids)
        ids = ids + response_ids
    row = {
        "sample_id": sample_id, "domain": domain, "problem": problem, "answer": answer,
        "full_token_ids": ids, "eligible_start": start, "eligible_end": end,
        "problem_sha256": sha256_text(normalize_question(problem)),
    }
    return row


def m6_corpus_path(name: str) -> Path:
    return M6 / "corpora" / f"{name}.jsonl"


def m6_prepare() -> dict[str, Any]:
    """Freeze the exact 5k/held/Numina/AIME samples before any GPU work."""
    manifest_path = M6 / "m6_data_manifest.json"
    cached = read_json(manifest_path, {})
    if cached.get("status") == "complete":
        return cached
    from transformers import AutoTokenizer

    if not all(path.is_file() for path in (MATHCOT_PARQUET, NUMINA_JSONL, AIME25_JSONL)):
        raise FileNotFoundError("one or more M6 source datasets are missing")
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_BASE), local_files_only=True, trust_remote_code=True)
    table = pd.read_parquet(MATHCOT_PARQUET)
    train = table.sample(n=5000, random_state=42).reset_index(drop=False)
    train_indices = set(int(value) for value in train["index"])
    train_hashes = {sha256_text(normalize_question(str(row.question))) for row in train.itertuples()}
    # The holdout is deterministic, prompt-deduplicated, and never overlaps the actual seed=42 5k.
    candidates: list[tuple[int, str, str, str]] = []
    seen = set(train_hashes)
    for index, row in table.iterrows():
        if int(index) in train_indices:
            continue
        problem = str(row["question"])
        fingerprint = sha256_text(normalize_question(problem))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append((int(index), problem, str(row.get("answer", "")), str(row.get("response", ""))))
    rng = random.Random(M6_SEED)
    rng.shuffle(candidates)
    selected = candidates[: MATHCOT_EVAL_N + MATHCOT_PROBE_N]
    if len(selected) != MATHCOT_EVAL_N + MATHCOT_PROBE_N:
        raise RuntimeError("not enough deduplicated Math-CoT holdout rows")
    train_probe = [
        probe_row(tokenizer, f"mathCoTtrain_{idx:03d}", str(row.question), domain="E_mathCoTtrain",
                  answer=str(row.answer), reference_response=str(row.response))
        for idx, row in enumerate(train.iloc[:MATHCOT_PROBE_N].itertuples())
    ]
    held_eval = [
        probe_row(tokenizer, f"mathCoThold_eval_{idx:03d}", problem, domain="Eval_mathCoThold", answer=answer)
        for idx, (_original, problem, answer, _response) in enumerate(selected[:MATHCOT_EVAL_N])
    ]
    held_probe = [
        probe_row(tokenizer, f"mathCoThold_probe_{idx:03d}", problem, domain="E_mathCoThold",
                  answer=answer, reference_response=response)
        for idx, (_original, problem, answer, response) in enumerate(selected[MATHCOT_EVAL_N:])
    ]
    numina_rows = [
        {"problem": str(row["problem"]), "answer": str(row["answer"])} for row in read_jsonl(NUMINA_JSONL)
        if str(row.get("answer", "")).strip().lower() not in {"", "not found", "notfound", "none", "nan", "proof"}
    ]
    unique_numina: list[dict[str, str]] = []
    numina_seen = set(train_hashes)
    for row in numina_rows:
        fingerprint = sha256_text(normalize_question(row["problem"]))
        if fingerprint in numina_seen:
            continue
        numina_seen.add(fingerprint)
        unique_numina.append(row)
    random.Random(M6_SEED + 1).shuffle(unique_numina)
    if len(unique_numina) < NUMINA_EVAL_N + NUMINA_PROBE_N:
        raise RuntimeError("not enough deduplicated Numina rows")
    numina_eval = [
        probe_row(tokenizer, f"numina_eval_{idx:03d}", row["problem"], domain="Eval_Numina", answer=row["answer"])
        for idx, row in enumerate(unique_numina[:NUMINA_EVAL_N])
    ]
    numina_probe = [
        probe_row(tokenizer, f"numina_probe_{idx:03d}", row["problem"], domain="E_numina", answer=row["answer"])
        for idx, row in enumerate(unique_numina[NUMINA_EVAL_N:NUMINA_EVAL_N + NUMINA_PROBE_N])
    ]
    aime_rows = [
        probe_row(tokenizer, f"aime25_{idx:03d}", str(row["problem"]), domain="Eval_AIME25", answer=str(row["answer"]))
        for idx, row in enumerate(read_jsonl(AIME25_JSONL))
    ]
    if len(aime_rows) != AIME25_N:
        raise RuntimeError(f"unexpected AIME25 inventory: {len(aime_rows)}")
    corpora = {
        "E_mathCoTtrain": train_probe, "Eval_mathCoThold": held_eval,
        "E_mathCoThold": held_probe, "Eval_Numina": numina_eval,
        "E_numina": numina_probe, "Eval_AIME25": aime_rows,
    }
    for name, rows in corpora.items():
        atomic_jsonl(m6_corpus_path(name), rows)
    manifest = {
        "schema_version": "cycle09_m6_data_v1", "status": "complete", "seed": M6_SEED,
        "train_selection": "Math-CoT-20k dataframe sample(n=5000, random_state=42), matching Cycle07/Cycle08",
        "holdout_selection": "prompt-deduplicated rows outside actual train indices; deterministic shuffle",
        "numina_selection": "deduplicated against Math-CoT train fingerprints and split before evaluation/probe",
        "fewshot_rows": "none: M6 uses zero-shot prompts only; exclusion recorded explicitly",
        "counts": {name: len(rows) for name, rows in corpora.items()},
        "corpora": {name: {"path": str(m6_corpus_path(name)), "sha256": sha256_file(m6_corpus_path(name))}
                    for name in corpora},
        "disjointness": {
            "mathcot_eval_vs_probe": True, "mathcot_train_vs_hold": True,
            "numina_eval_vs_probe": True,
        }, "created_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def qwen_adapter_path(arm: str, step: int) -> Path:
    if step == 0:
        raise ValueError("base has no adapter")
    if arm == "sft":
        return QWEN_SFT / label(step)
    if arm == "offkd":
        if step == 80:
            return QWEN_OFFKD / "checkpoint_backfill/from_040/checkpoint-000080"
        if step in (320, 480):
            return QWEN_OFFKD / f"checkpoint_backfill/from_160/checkpoint-{step:06d}"
        return QWEN_OFFKD / f"checkpoints/checkpoint-{step:06d}"
    if arm == "seqkd":
        return QWEN_SEQKD / f"checkpoints/checkpoint-{step:06d}"
    raise ValueError(arm)


def qwen_direct_model(arm: str, step: int) -> Path | None:
    if step == 0:
        return QWEN_BASE
    if arm == "opd":
        candidate = QWEN_OPD / label(step)
        return candidate if (candidate / "config.json").is_file() else None
    return None


def complete_model(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def materialize_qwen(arm: str, step: int, scratch: Path) -> tuple[Path, bool]:
    """Return a model path and whether the caller must remove it after use."""
    direct = qwen_direct_model(arm, step)
    if direct is not None:
        if not complete_model(direct):
            raise FileNotFoundError(f"incomplete direct Qwen model: {direct}")
        return direct, False
    adapter = qwen_adapter_path(arm, step)
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"missing Qwen adapter: {adapter}")
    target = scratch / arm / label(step)
    with locked(target):
        if complete_model(target):
            return target, True
        temporary = target.with_name(target.name + ".tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            str(QWEN_BASE), dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True, trust_remote_code=True,
        )
        try:
            merged = PeftModel.from_pretrained(base, str(adapter), local_files_only=True).merge_and_unload()
            merged.save_pretrained(str(temporary), safe_serialization=True)
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(str(QWEN_BASE), local_files_only=True, trust_remote_code=True).save_pretrained(str(temporary))
            if not complete_model(temporary):
                raise RuntimeError(f"Qwen merge incomplete: {temporary}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
        finally:
            del base
            gc.collect()
    return target, True


def qwen_reference_path(probe: str) -> Path:
    return m6_root() / "geometry/references" / f"{probe}.pt"


def qwen_profile_path(arm: str, step: int, probe: str) -> Path:
    return m6_root() / "geometry/profiles" / arm / label(step) / f"{probe}.pt"


def prepared_qwen_samples(probe: str, limit: int | None = None) -> list[Any]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_BASE), local_files_only=True, trust_remote_code=True)
    samples = c4.prepare_samples(m6_corpus_path(probe), tokenizer, corpus_id=f"m6:{probe}", window_seed=WINDOW_SEED)
    return samples[:limit] if limit else samples


def collect_qwen_profile(model: Any, probe: str, device: str, limit: int | None = None) -> tuple[dict[str, Any], list[Any]]:
    samples = prepared_qwen_samples(probe, limit)
    profile = campaign.collect_profile(
        model, samples, [18], device,
        keep_factors=False, keep_residual_samples=False, keep_input_sample_means=False,
        factor_layers=(), forward_batch_size=4, max_batch_tokens=16384, early_stop=True,
    )
    return profile, samples


def ensure_qwen_reference(probe: str, device: str, limit: int | None = None) -> dict[str, Any]:
    target = qwen_reference_path(probe)
    with locked(target):
        if target.is_file():
            return torch.load(target, map_location="cpu", weights_only=True)
        model = load_bf16(QWEN_BASE, device)
        try:
            profile, samples = collect_qwen_profile(model, probe, device, limit)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            torch.save(profile, temporary)
            os.replace(temporary, target)
            atomic_json(target.with_suffix(".json"), {
                "status": "complete", "probe": probe, "sample_count": len(samples),
                "sample_ids_sha256": sha256_json([sample.sample_id for sample in samples]),
                "created_utc": utc_now(),
            })
            return profile
        finally:
            unload(model)


def qwen_geometry_cell(arm: str, step: int, probe: str, model_path: Path, device: str, limit: int | None = None) -> dict[str, Any]:
    target = m6_root() / "geometry/cells" / arm / label(step) / f"{probe}.json"
    with locked(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        reference = ensure_qwen_reference(probe, device, limit)
        profile_path = qwen_profile_path(arm, step, probe)
        base, model = None, None
        try:
            base = load_bf16(QWEN_BASE, device)
            model = load_bf16(model_path, device)
            profile, samples = collect_qwen_profile(model, probe, device, limit)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = profile_path.with_suffix(".tmp")
            torch.save(profile, temporary)
            os.replace(temporary, profile_path)
            scales = campaign.scaling_by_group(profile, [18], device)
            rows, spectra = [], []
            for module in QWEN_MODULES:
                weight = module_at(model, 18, module).weight.detach().to(device, torch.float32)
                transformed = weight @ scales[18][group(module)]
                singular = torch.linalg.svdvals(transformed).double().cpu().tolist()
                base_weight = module_at(base, 18, module).weight.detach().to(device, torch.float32)
                base_singular = torch.linalg.svdvals(
                    base_weight @ sqrt_psd(reference["grams"][18][group(module)], device)
                ).double().cpu().tolist()
                for epsilon in EPSILONS:
                    rows.append({
                        "model": "qwen", "arm": arm, "checkpoint": step, "probe_name": probe,
                        "layer": 18, "module": module, "epsilon": epsilon,
                        "r_epsilon": c4.functional_rank(singular, epsilon),
                        "base_r_epsilon": c4.functional_rank(base_singular, epsilon),
                        "track": "per_checkpoint", "sample_count": int(profile["n_samples"]),
                        "window_protocol": "v2_three_level_equal_sample",
                    })
                spectra.append({"module": module, "singular_values": singular})
                del weight, transformed, base_weight
            scales.clear()
            spectra_path = target.with_suffix(".spectra.json")
            atomic_json(spectra_path, {"status": "complete", "rows": spectra})
            payload = {
                "schema_version": "cycle09_m6_qwen_geometry_v1", "status": "complete",
                "arm": arm, "checkpoint": step, "probe": probe, "rows": rows,
                "profile": str(profile_path), "reference": str(qwen_reference_path(probe)),
                "spectra": str(spectra_path), "created_utc": utc_now(),
            }
            atomic_json(target, payload)
            return payload
        finally:
            unload(model)
            unload(base)


def behavior_cell_path(task: str, arm: str, step: int, seed: int) -> Path:
    return m6_root() / "behavior/cells" / task / arm / label(step) / f"seed_{seed}.json"


def evaluate_qwen(task: str, arm: str, step: int, model_path: Path, device: str, *, cap: int, seed: int, limit: int | None = None) -> dict[str, Any]:
    target = behavior_cell_path(task, arm, step, seed)
    with locked(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete":
            return cached
        rows = read_jsonl(m6_corpus_path(task))
        if limit:
            rows = rows[:limit]
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            tokenize=False, add_generation_prompt=True,
        ) for row in rows]
        llm = LLM(model=str(model_path), dtype="bfloat16", gpu_memory_utilization=0.88,
                  max_model_len=max(cap + 2048, 16384), trust_remote_code=True, disable_log_stats=True)
        try:
            outputs = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.9, max_tokens=cap, seed=seed))
            samples = []
            correct = trunc = boxed = 0
            for row, output in zip(rows, outputs):
                completion = output.outputs[0]
                text = completion.text
                ok = bool(score(text, row["answer"]))
                is_truncated = completion.finish_reason == "length"
                correct += int(ok)
                trunc += int(is_truncated)
                boxed += int("\\boxed" in text)
                samples.append({
                    "id": row["sample_id"], "gold": row["answer"], "pred": extract_pred(text), "ok": ok,
                    "finish_reason": completion.finish_reason, "response_tokens": len(completion.token_ids),
                    "generation_sha256": sha256_text(text),
                })
            sample_path = target.with_suffix(".samples.jsonl")
            atomic_jsonl(sample_path, samples)
            n = len(rows)
            payload = {
                "schema_version": "cycle09_m6_behavior_v1", "status": "complete", "task": task,
                "arm": arm, "checkpoint": step, "seed": seed, "n": n, "cap": cap,
                "accuracy": correct / n if n else 0.0, "truncation_rate": trunc / n if n else 0.0,
                "boxed_rate": boxed / n if n else 0.0,
                "mean_response_tokens": sum(item["response_tokens"] for item in samples) / n if n else 0.0,
                "model": str(model_path), "samples": str(sample_path), "created_utc": utc_now(),
            }
            atomic_json(target, payload)
            return payload
        finally:
            del llm, tokenizer
            gc.collect()
            torch.cuda.empty_cache()


def aime_steps(arm: str) -> tuple[int, ...]:
    """Same sparse AIME shape as the frozen AIME24 protocol, with seqKD final added."""
    return {"opd": (320, 624), "sft": (624,), "offkd": (480, 624), "seqkd": (624,)}[arm]


def m6_task_list() -> list[dict[str, Any]]:
    tasks = []
    # A model-group task keeps one temporary merge alive for behavior plus its probes.
    for arm in QWEN_ARMS:
        for step in MATHCOT_STEPS:
            if step == 0 and arm != "opd":
                continue
            tasks.append({"id": f"mathcot:{arm}:{step}", "arm": arm, "step": step,
                          "mathcot": True, "aime25": step in aime_steps(arm),
                          "numina": arm == "seqkd" and step in NUMINA_STEPS,
                          "cost": 100 if step else 60})
        for step in aime_steps(arm):
            key = f"mathcot:{arm}:{step}"
            if not any(task["id"] == key for task in tasks):
                tasks.append({"id": key, "arm": arm, "step": step, "mathcot": False,
                              "aime25": True, "numina": False, "cost": 50})
    for arm in QWEN_ARMS:
        for step in NUMINA_STEPS:
            key = f"numina:{arm}:{step}"
            tasks.append({"id": key, "arm": arm, "step": step, "mathcot": False,
                          "aime25": step in aime_steps(arm), "numina": True, "cost": 45})
    # Group identical arm/step flags so a checkpoint is never merged twice by the queue.
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for task in tasks:
        key = (task["arm"], task["step"])
        value = grouped.setdefault(key, {
            "id": f"group:{key[0]}:{key[1]}", "arm": key[0], "step": key[1],
            "mathcot": False, "aime25": False, "numina": False, "cost": 0,
        })
        for flag in ("mathcot", "aime25", "numina"):
            value[flag] = value[flag] or task[flag]
        value["cost"] += task["cost"]
    return sorted(grouped.values(), key=lambda item: (-item["cost"], item["id"]))


def queue_path() -> Path:
    override = os.environ.get("CYCLE09_M6_QUEUE")
    return Path(override) if override else m6_root() / "queue.json"


def initialise_m6_queue() -> None:
    path = queue_path()
    with locked(path):
        current = read_json(path, {})
        if current.get("schema_version") == "cycle09_m6_queue_v1":
            return
        atomic_json(path, {"schema_version": "cycle09_m6_queue_v1", "status": "ready",
                           "tasks": [{**task, "status": "pending"} for task in m6_task_list()],
                           "created_utc": utc_now()})


def claim_m6_task(worker: str) -> dict[str, Any] | None:
    path = queue_path()
    with locked(path):
        payload = read_json(path, {})
        for task in payload.get("tasks", []):
            if task.get("status") == "pending":
                task.update({
                    "status": "running", "worker": worker, "started_utc": utc_now(),
                    "attempts": int(task.get("attempts", 0)) + 1,
                })
                atomic_json(path, payload)
                return dict(task)
        return None


def finish_m6_task(task_id: str, worker: str, *, error: str | None = None) -> None:
    path = queue_path()
    with locked(path):
        payload = read_json(path, {})
        task = next(item for item in payload["tasks"] if item["id"] == task_id)
        task.update({"status": "failed" if error else "complete", "worker": worker,
                     "finished_utc": utc_now(), "error": error})
        atomic_json(path, payload)


def retry_failed_m6_tasks(max_attempts: int = 3) -> int:
    """Requeue transient failed cells without letting one cell idle both GPUs."""
    path = queue_path()
    with locked(path):
        payload = read_json(path, {})
        retried = 0
        for task in payload.get("tasks", []):
            if task.get("status") == "failed" and int(task.get("attempts", 0)) < max_attempts:
                task.update({"status": "pending", "error": None, "retried_utc": utc_now()})
                retried += 1
        if retried:
            atomic_json(path, payload)
        return retried


def run_m6_group(task: dict[str, Any], device: str, worker: str, smoke: bool) -> None:
    arm, step = str(task["arm"]), int(task["step"])
    report_arm = "base" if step == 0 else arm
    scratch = m6_root() / "runtime" / worker
    model_path, transient = materialize_qwen(arm, step, scratch)
    try:
        if task["mathcot"]:
            evaluate_qwen("Eval_mathCoThold", report_arm, step, model_path, device, cap=16384, seed=42,
                          limit=2 if smoke else None)
            qwen_geometry_cell(report_arm, step, "E_mathCoTtrain", model_path, device, limit=1 if smoke else MATHCOT_PROBE_N)
            qwen_geometry_cell(report_arm, step, "E_mathCoThold", model_path, device, limit=1 if smoke else MATHCOT_PROBE_N)
        if task["numina"]:
            if arm == "seqkd":
                evaluate_qwen("Eval_Numina", report_arm, step, model_path, device, cap=12288, seed=42,
                              limit=2 if smoke else None)
            qwen_geometry_cell(report_arm, step, "E_numina", model_path, device, limit=1 if smoke else NUMINA_PROBE_N)
        if task["aime25"]:
            for seed in ((42,) if smoke else AIME25_SEEDS):
                evaluate_qwen("Eval_AIME25", report_arm, step, model_path, device, cap=24576, seed=seed,
                              limit=2 if smoke else None)
    finally:
        if transient:
            shutil.rmtree(model_path, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()


def m6_worker(args: argparse.Namespace) -> None:
    worker = args.worker_id
    while True:
        task = claim_m6_task(worker)
        if task is None:
            return
        try:
            run_m6_group(task, args.device, worker, args.smoke)
        except Exception as error:
            finish_m6_task(task["id"], worker, error=repr(error))
            print(f"[m6] {worker}: {task['id']} failed: {error!r}; continuing queue", flush=True)
        else:
            finish_m6_task(task["id"], worker)
            print(f"[m6] {worker}: {task['id']} complete", flush=True)


def finalize_dual() -> dict[str, Any]:
    rows, spectra = [], []
    for path in sorted((DUAL / "cells").rglob("*.json")):
        if path.name.endswith(".spectra.json"):
            continue
        payload = read_json(path, {})
        if payload.get("status") == "complete":
            rows.extend(payload.get("rows", []))
            spectra_path = Path(payload["spectra"])
            if spectra_path.is_file():
                spectra.extend(read_json(spectra_path, {}).get("rows", []))
    expected = len(LLAMA_ARMS) * len(LLAMA_STEPS) * len(LLAMA_PROBES) * len(b3.MODULES) * len(EPSILONS) * 2
    if len(rows) != expected:
        raise RuntimeError(f"dual row count incomplete: {len(rows)} != {expected}")
    output = MINI / "Llama_dual_numeric_tracks.csv"
    atomic_csv(output, rows)
    atomic_json(MINI / "Llama_dual_numeric_tracks_manifest.json", {
        "status": "complete", "rows": len(rows), "spectra_rows": len(spectra),
        "output": str(output), "raw_root": str(DUAL), "created_utc": utc_now(),
    })
    return {"status": "complete", "rows": len(rows), "output": str(output)}


def finalize_m6() -> dict[str, Any]:
    behavior, geometry = [], []
    for path in sorted((M6 / "behavior/cells").rglob("*.json")):
        payload = read_json(path, {})
        if payload.get("status") == "complete":
            behavior.append({key: value for key, value in payload.items() if key not in {"samples"}})
    for path in sorted((M6 / "geometry/cells").rglob("*.json")):
        payload = read_json(path, {})
        if payload.get("status") == "complete":
            geometry.extend(payload.get("rows", []))
    atomic_csv(MINI / "M6_behavior.csv", behavior)
    atomic_csv(MINI / "M6_geometry_r_epsilon.csv", geometry)
    queue = read_json(queue_path(), {})
    if any(task.get("status") != "complete" for task in queue.get("tasks", [])):
        raise RuntimeError("M6 queue has incomplete or failed tasks")
    atomic_json(MINI / "M6_supplement_manifest.json", {
        "status": "complete", "behavior_rows": len(behavior), "geometry_rows": len(geometry),
        "queue": str(queue_path()), "data_manifest": str(M6 / "m6_data_manifest.json"),
        "created_utc": utc_now(),
    })
    return {"status": "complete", "behavior_rows": len(behavior), "geometry_rows": len(geometry)}


def preflight() -> dict[str, Any]:
    disk_guard()
    missing = []
    for arm in LLAMA_ARMS:
        for step in LLAMA_STEPS:
            if not lexport.adapter_complete(lexport.adapter_target(arm, step)):
                missing.append(str(lexport.adapter_target(arm, step)))
            if not b3.model_check(lexport.merged_target(arm, step))["complete"]:
                missing.append(str(lexport.merged_target(arm, step)))
    for arm in QWEN_ARMS:
        for step in set(MATHCOT_STEPS) | set(NUMINA_STEPS) | set(aime_steps(arm)):
            if step and qwen_direct_model(arm, step) is None:
                adapter = qwen_adapter_path(arm, step)
                if not (adapter / "adapter_config.json").is_file():
                    missing.append(str(adapter))
    if missing:
        raise FileNotFoundError("preflight missing artifacts:\n" + "\n".join(missing))
    payload = {"status": "complete", "free_gib": round(shutil.disk_usage("/root/autodl-tmp").free / (1 << 30), 2),
               "llama_cells": len(LLAMA_ARMS) * len(LLAMA_STEPS) * len(LLAMA_PROBES),
               "m6_groups": len(m6_task_list()), "created_utc": utc_now()}
    atomic_json(ROOT / "preflight.json", payload)
    return payload


def launch_children(mode: str, common: list[str]) -> int:
    commands = []
    if mode == "dual":
        commands = [
            ["CUDA_VISIBLE_DEVICES=0", sys.executable, __file__, "worker-dual", "--arms", "opd", *common],
            ["CUDA_VISIBLE_DEVICES=1", sys.executable, __file__, "worker-dual", "--arms", "offkd", *common],
        ]
    elif mode == "m6":
        commands = [
            ["CUDA_VISIBLE_DEVICES=0", sys.executable, __file__, "worker-m6", "--worker-id", "gpu0", *common],
            ["CUDA_VISIBLE_DEVICES=1", sys.executable, __file__, "worker-m6", "--worker-id", "gpu1", *common],
        ]
    processes = []
    for index, command in enumerate(commands):
        env = dict(os.environ)
        assignment, *rest = command
        key, value = assignment.split("=", 1)
        env[key] = value
        log = ROOT / "logs" / f"{mode}_gpu{index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        processes.append((subprocess.Popen(rest, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
    code = 0
    for process, handle in processes:
        return_code = process.wait()
        if return_code and not code:
            code = return_code
        handle.close()
    return code


def smoke() -> None:
    preflight()
    # Two small Llama cells exercise both tracks and both GPU workers.
    os.environ["CYCLE09_DUAL_NAMESPACE"] = "smoke"
    code = launch_children("dual", ["--steps", "20", "--probes", "E_math", "--measurement-n", "1", "--device", "cuda:0"])
    os.environ.pop("CYCLE09_DUAL_NAMESPACE", None)
    if code:
        raise RuntimeError(f"dual smoke worker exited {code}")
    m6_prepare()
    # M6 smoke keeps only two records and runs a single claimed group.  A separate queue avoids contaminating formal state.
    original = queue_path()
    smoke_queue = M6 / "queue.smoke.json"
    tasks = [{"id": "group:sft:20", "arm": "sft", "step": 20, "mathcot": True, "aime25": False,
              "numina": False, "cost": 1, "status": "pending"}]
    atomic_json(smoke_queue, {"schema_version": "cycle09_m6_queue_v1", "status": "smoke", "tasks": tasks})
    # queue_path is intentionally a function: temporarily point the process at the smoke file through an env variable.
    os.environ["CYCLE09_M6_QUEUE"] = str(smoke_queue)
    os.environ["CYCLE09_M6_NAMESPACE"] = "smoke"
    code = launch_children("m6", ["--device", "cuda:0", "--smoke"])
    os.environ.pop("CYCLE09_M6_QUEUE", None)
    os.environ.pop("CYCLE09_M6_NAMESPACE", None)
    if code:
        raise RuntimeError(f"M6 smoke worker exited {code}")
    atomic_json(ROOT / "smoke.json", {"status": "complete", "max_wall_seconds": 600, "created_utc": utc_now()})


def supervisor(args: argparse.Namespace) -> None:
    try:
        preflight()
        atomic_json(ROOT / "supervisor_state.json", {"status": "running", "phase": "dual", "started_utc": utc_now()})
        common = ["--steps", ",".join(map(str, LLAMA_STEPS)), "--probes", ",".join(LLAMA_PROBES),
                  "--measurement-n", "0", "--device", "cuda:0"]
        if launch_children("dual", common):
            raise RuntimeError("dual worker failed; resumable cell markers retained")
        finalize_dual()
        atomic_json(ROOT / "supervisor_state.json", {"status": "running", "phase": "m6", "updated_utc": utc_now()})
        m6_prepare()
        initialise_m6_queue()
        for _round in range(3):
            if launch_children("m6", ["--device", "cuda:0"]):
                raise RuntimeError("M6 worker launch failed; queue remains resumable")
            if not retry_failed_m6_tasks(max_attempts=3):
                break
        finalize_m6()
        atomic_json(ROOT / "supervisor_state.json", {"status": "complete", "phase": "complete", "completed_utc": utc_now()})
    except Exception as error:
        atomic_json(ROOT / "supervisor_state.json", {
            "status": "failed", "phase": "failed", "error": repr(error), "updated_utc": utc_now(),
        })
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("prepare-m6")
    sub.add_parser("smoke")
    sub.add_parser("supervisor")
    dual = sub.add_parser("worker-dual")
    dual.add_argument("--arms", required=True)
    dual.add_argument("--steps", default=",".join(map(str, LLAMA_STEPS)))
    dual.add_argument("--probes", default=",".join(LLAMA_PROBES))
    dual.add_argument("--measurement-n", type=int, default=0)
    dual.add_argument("--device", default="cuda:0")
    m6 = sub.add_parser("worker-m6")
    m6.add_argument("--worker-id", required=True)
    m6.add_argument("--device", default="cuda:0")
    m6.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2))
    elif args.command == "prepare-m6":
        print(json.dumps(m6_prepare(), indent=2))
    elif args.command == "smoke":
        smoke()
    elif args.command == "supervisor":
        supervisor(args)
    elif args.command == "worker-dual":
        dual_worker(args)
    elif args.command == "worker-m6":
        m6_worker(args)


if __name__ == "__main__":
    main()
