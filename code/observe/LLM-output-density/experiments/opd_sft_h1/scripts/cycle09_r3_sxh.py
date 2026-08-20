#!/usr/bin/env python3
"""Cycle 09 Round 3 S/X/H generation and three-layer probe runner (R3-5).

The new S/X/H taxonomy is intentionally isolated from the Round 2 legacy
S/X artifacts. Generation parameters and every corpus source are recorded in
a manifest; all generated corpus records retain prompt and checkpoint metadata.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path("/root/LLM-output-density")
GETSLICE = REPO / "GetSlice"
SIDE = REPO / "experiments/opd_sft_h1"
for item in (REPO, GETSLICE, SIDE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import cycle09_r2_unified_probe as r2  # noqa: E402
from opd_sft_h1.geometry_metrics import effective_rank  # noqa: E402

DEFAULT_RUN = Path("/root/autodl-tmp/cycle09_r3")
DEFAULT_MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
ARMS = ("opd", "sft")
STEPS = (0, 5, 10, 20, 40, 160, 624)
LAYERS = (9, 18, 27)
DOMAINS = ("math", "ood_knowledge", "general")
SEED = 3
LOADER_HEADROOM = 8
MATH_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
OOD_INSTRUCTION = "\nAnswer the question accurately and concisely."
GENERAL_INSTRUCTION = (
    "\nContinue with a coherent, factual passage that stays on the same topic."
)


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def parse_names(value: str, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str, default: tuple[int, ...]) -> list[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_path(
    run_root: Path,
    role: str,
    arm: str,
    step: int | None,
    domain: str,
) -> Path:
    arm_part = arm
    step_part = "fixed" if step is None else step_label(step)
    return run_root / "sxh" / "corpora" / role / arm_part / step_part / f"{domain}.jsonl"


def spectrum_path(run_root: Path, arm: str, step: int, task: str) -> Path:
    return run_root / "sxh" / "spectra" / arm / step_label(step) / f"{task}.json"


def prompt_sources() -> dict[str, Path]:
    paths = r2.probe_paths()
    math = r2.S_ROOT / "math_cot_probe" / "gamma_s.jsonl"
    required = {
        "math": math,
        "ood_knowledge": paths.get("X_ood_knowledge"),
        "general": paths.get("X_general"),
    }
    missing = [name for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen R3-5 prompt sources: {missing}")
    return {name: Path(path) for name, path in required.items()}


def build_prompt_bank(paths: dict[str, Path], n: int, max_prompt_chars: int) -> dict[str, list[dict[str, str]]]:
    math_rows = read_jsonl(paths["math"])
    ood_rows = read_jsonl(paths["ood_knowledge"])
    general_rows = read_jsonl(paths["general"])
    if min(len(math_rows), len(ood_rows), len(general_rows)) < n:
        raise ValueError("A frozen source has fewer records than --n-samples")

    banks = {
        "math": [
            {
                "id": str(index),
                "prompt": str(row["question"])[:max_prompt_chars],
                "instruction": MATH_INSTRUCTION,
            }
            for index, row in enumerate(math_rows[:n])
        ],
        "ood_knowledge": [
            {
                "id": str(index),
                "prompt": str(row["output"]["text"])[:max_prompt_chars],
                "instruction": OOD_INSTRUCTION,
            }
            for index, row in enumerate(ood_rows[:n])
        ],
        "general": [
            {
                "id": str(index),
                "prompt": str(row["output"]["text"])[:max_prompt_chars],
                "instruction": GENERAL_INSTRUCTION,
            }
            for index, row in enumerate(general_rows[:n])
        ],
    }
    return banks


def fixed_sft_x(path: Path, math_source: Path, n: int) -> None:
    if path.exists():
        return
    rows = read_jsonl(math_source)
    if len(rows) < n:
        raise ValueError("SFT source has fewer records than requested")
    records = []
    for index, row in enumerate(rows[:n]):
        text = f"{row['question']}\n{row['answer']}"
        records.append(
            {
                "id": str(index),
                "role": "X",
                "arm": "sft",
                "step": "fixed",
                "domain": "math",
                "source_kind": "fixed_dataset_cot",
                "output": {"text": text},
            }
        )
    write_jsonl(path, records)


def generate_corpora(
    model_path: Path,
    outputs: list[tuple[Path, str, str, int | None, list[dict[str, str]]]],
    args: argparse.Namespace,
) -> None:
    pending = [item for item in outputs if not item[0].exists()]
    if not pending:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    try:
        for target, role, domain, step, bank in pending:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["prompt"] + item["instruction"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in bank
            ]
            outputs_vllm = llm.generate(prompts, sampling)
            rows = []
            for item, output in zip(bank, outputs_vllm):
                response = output.outputs[0].text
                rows.append(
                    {
                        "id": item["id"],
                        "role": role,
                        "domain": domain,
                        "arm": "base" if role == "S" else target.parts[-3],
                        "step": "base" if step is None else int(step),
                        "source_kind": "base_generation" if role == "S" else "checkpoint_generation",
                        "prompt": item["prompt"],
                        "instruction": item["instruction"],
                        "generation": response,
                        "finish": output.outputs[0].finish_reason,
                        "generation_config": {
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "seed": args.seed,
                        },
                        "output": {"text": item["prompt"] + "\n" + response},
                    }
                )
            write_jsonl(target, rows)
            print(f"[Generate] {target} n={len(rows)}", flush=True)
    finally:
        del llm, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def generate_all(args: argparse.Namespace, banks: dict[str, list[dict[str, str]]], sources: dict[str, Path]) -> None:
    fixed_sft_x(corpus_path(args.run_root, "X", "sft", None, "math"), sources["math"], args.n_samples)

    base_outputs = [
        (corpus_path(args.run_root, "S", "base", None, domain), "S", domain, None, banks[domain])
        for domain in DOMAINS
    ]
    for arm in args.arms:
        if 0 not in args.steps:
            continue
        if arm == "opd":
            base_outputs.append(
                (corpus_path(args.run_root, "X", "opd", 0, "math"), "X", "math", 0, banks["math"])
            )
        for domain in ("ood_knowledge", "general"):
            base_outputs.append(
                (corpus_path(args.run_root, "H", arm, 0, domain), "H", domain, 0, banks[domain])
            )
        if arm == "sft":
            base_outputs.append(
                (corpus_path(args.run_root, "H", arm, 0, "math"), "H", "math", 0, banks["math"])
            )
    generate_corpora(r2.BASE_MODEL, base_outputs, args)

    for arm in args.arms:
        for step in args.steps:
            if step == 0:
                continue
            model_path = r2.model_path_for(arm, step)
            outputs = []
            if arm == "opd":
                outputs.append(
                    (corpus_path(args.run_root, "X", "opd", step, "math"), "X", "math", step, banks["math"])
                )
            for domain in ("ood_knowledge", "general"):
                outputs.append(
                    (corpus_path(args.run_root, "H", arm, step, domain), "H", domain, step, banks[domain])
                )
            if arm == "sft":
                outputs.append(
                    (corpus_path(args.run_root, "H", arm, step, "math"), "H", "math", step, banks["math"])
                )
            generate_corpora(model_path, outputs, args)


def sxh_tasks(run_root: Path, arm: str, step: int) -> dict[str, tuple[str, str, Path]]:
    tasks = {
        "S_math": ("S", "math", corpus_path(run_root, "S", "base", None, "math")),
        "S_ood_knowledge": (
            "S", "ood_knowledge", corpus_path(run_root, "S", "base", None, "ood_knowledge")
        ),
        "S_general": ("S", "general", corpus_path(run_root, "S", "base", None, "general")),
        "H_ood_knowledge": (
            "H", "ood_knowledge", corpus_path(run_root, "H", arm, step, "ood_knowledge")
        ),
        "H_general": ("H", "general", corpus_path(run_root, "H", arm, step, "general")),
    }
    if arm == "opd":
        tasks["X_math"] = ("X", "math", corpus_path(run_root, "X", "opd", step, "math"))
    else:
        tasks["X_math"] = ("X", "math", corpus_path(run_root, "X", "sft", None, "math"))
        tasks["H_math"] = ("H", "math", corpus_path(run_root, "H", "sft", step, "math"))
    missing = [name for name, (_, _, path) in tasks.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing R3-5 corpora for {arm}/{step}: {missing}")
    return tasks


def build_probe_loaders(tokenizer, tasks, args: argparse.Namespace, arm: str, step: int):
    from utils.data_utils import get_token_data_from_jsonl

    loaders = {}
    requested_windows = args.n_samples + LOADER_HEADROOM
    for name, (_, _, path) in tasks.items():
        cache = (
            args.run_root / "sxh" / "cache"
            / (
                f"{arm}__{step_label(step)}__{name}__n{args.n_samples}"
                f"__attempts{requested_windows}__seq{args.seqlen}.pt"
            )
        )
        candidate_windows = get_token_data_from_jsonl(
            jsonl_path=str(path),
            tokenizer=tokenizer,
            nsamples=requested_windows,
            seqlen=args.seqlen,
            seed=args.seed,
            batch_size=1,
            cache_file=str(cache),
            mode="x",
        )
        if len(candidate_windows) < args.n_samples:
            raise RuntimeError(
                f"{name} yielded {len(candidate_windows)} valid windows after "
                f"{requested_windows} attempts; expected at least {args.n_samples}"
            )
        loaders[name] = candidate_windows[:args.n_samples]
    return loaders


def spectrum_complete(path: Path, layers: list[int]) -> bool:
    data = read_json(path)
    return all(f"layer_{layer}" in data for layer in layers)


def probe_one_model(args: argparse.Namespace, arm: str, step: int) -> None:
    from utils.profiling_utils import profile_svdllm_single_layer_group, whitening

    tasks = sxh_tasks(args.run_root, arm, step)
    pending = {
        name: item for name, item in tasks.items()
        if not spectrum_complete(spectrum_path(args.run_root, arm, step, name), args.layers)
    }
    if not pending:
        print(f"[Probe skip] {arm}/{step_label(step)}", flush=True)
        return
    model_path = r2.model_path_for(arm, step)
    print(f"[Probe model] {arm}/{step_label(step)} tasks={list(tasks)}", flush=True)
    model, tokenizer = r2.load_model_for_custom(model_path, args.seqlen)
    try:
        loaders = build_probe_loaders(tokenizer, tasks, args, arm, step)
        for layer in args.layers:
            profiles = profile_svdllm_single_layer_group(
                model_name=str(model_path),
                model=model,
                calib_loaders_by_task=loaders,
                dev=args.device,
                target_layer=layer,
                layer_gpu_chunk_size=args.layer_gpu_chunk_size,
                singular_floor=0.0,
                activation_cache_device=args.activation_cache_device,
                cholesky_jitter=1e-5,
            )
            for task_name, profile in profiles.items():
                target = spectrum_path(args.run_root, arm, step, task_name)
                data = read_json(target)
                key = f"layer_{layer}"
                if key not in data:
                    sigma, _ = whitening(
                        model_name=str(model_path),
                        model=model,
                        profiling_mat=profile,
                        dev=args.device,
                        uv_dtype="float32",
                        return_uv=False,
                    )
                    data[key] = sigma[key]
                    write_json(target, data)
            del profiles
            gc.collect()
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for step in args.steps:
            for task, (role, domain, path) in sxh_tasks(args.run_root, arm, step).items():
                spectra = read_json(spectrum_path(args.run_root, arm, step, task))
                for layer in args.layers:
                    for module, values in spectra.get(f"layer_{layer}", {}).items():
                        rows.append(
                            {
                                "arm": arm,
                                "step": step,
                                "taxonomy_role": role,
                                "domain": domain,
                                "task": task,
                                "layer": layer,
                                "module": module,
                                "whitened_effective_rank": f"{effective_rank(values):.8f}",
                                "corpus_path": str(path),
                                "spectrum_path": str(spectrum_path(args.run_root, arm, step, task)),
                                "baseline_rule": "step_0=S within this taxonomy",
                            }
                        )
    fields = [
        "arm", "step", "taxonomy_role", "domain", "task", "layer", "module",
        "whitened_effective_rank", "corpus_path", "spectrum_path", "baseline_rule",
    ]
    with open(args.mini_root / "R3_sxh_er.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Summary] wrote R3_sxh_er.csv rows={len(rows)}", flush=True)


def write_manifest(args: argparse.Namespace, sources: dict[str, Path]) -> None:
    write_json(
        args.mini_root / "R3_sxh_generation_manifest.json",
        {
            "schema_version": 1,
            "taxonomy": "round3_S_X_H; separate from legacy_round2_S_X",
            "loader_valid_window_policy": {
                "attempts_per_task": args.n_samples + LOADER_HEADROOM,
                "kept_per_task": args.n_samples,
                "selection": "first n valid deterministic windows after strict tokenization",
            },
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "seqlen_for_getslice": args.seqlen,
            "prompt_sources": {key: str(value) for key, value in sources.items()},
            "domains": list(DOMAINS),
            "S": "base step_0 generation shared across arms",
            "X_opd": "per-checkpoint math self rollout",
            "X_sft": "fixed dataset CoT",
            "H_opd": "per-checkpoint nontraining ood_knowledge and general generations",
            "H_sft": "per-checkpoint math plus nontraining generations",
            "probe_text": "prompt plus generated assistant text; fixed SFT X is question plus dataset CoT",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation-cache-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--layer-gpu-chunk-size", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.run_root = args.run_root / "smoke_sxh"
        args.mini_root = args.mini_root / "smoke_sxh"
        args.generate = True
        args.probe = True
        args.summarize = True
        args.arms = "opd"
        args.steps = "0"
        args.layers = "18"
        args.n_samples = 1
        args.seqlen = 4
        args.max_new_tokens = 16
        args.max_model_len = 512
        args.max_prompt_chars = 600

    if args.all:
        args.generate = True
        args.probe = True
        args.summarize = True
    if not (args.generate or args.probe or args.summarize):
        parser.print_help()
        return

    args.arms = parse_names(args.arms, ARMS)
    args.steps = parse_ints(args.steps, STEPS)
    args.layers = parse_ints(args.layers, LAYERS)
    if args.n_samples <= 0 or args.seqlen <= 0:
        raise ValueError("--n-samples and --seqlen must be positive")
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.mini_root.mkdir(parents=True, exist_ok=True)
    r2.configure_roots(args.run_root, args.mini_root)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    sources = prompt_sources()
    banks = build_prompt_bank(sources, args.n_samples, args.max_prompt_chars)
    write_manifest(args, sources)
    print(
        f"[Plan] arms={args.arms} steps={args.steps} layers={args.layers} "
        f"n={args.n_samples} max_new={args.max_new_tokens}",
        flush=True,
    )
    if args.dry_run:
        return
    if args.generate:
        generate_all(args, banks, sources)
    if args.probe:
        for arm in args.arms:
            for step in args.steps:
                probe_one_model(args, arm, step)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()

