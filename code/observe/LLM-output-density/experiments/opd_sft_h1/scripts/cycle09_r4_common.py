#!/usr/bin/env python3
"""Shared definitions for Cycle 09 Round 4 window-v2 measurements."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO = Path("/root/LLM-output-density")
RUN_ROOT = Path("/root/autodl-tmp/cycle09_r4")
MINI_ROOT = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
OPD_MODELS = Path("/root/autodl-tmp/cycle08_opd_trajectory/_merged_models")
SFT_MODELS = Path("/root/autodl-tmp/cycle09_r3/sft_merged")
SFT_ADAPTERS = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints")
LEGACY_MATH = Path(
    "/root/autodl-tmp/cycle07_base_sft_trajectory/getslice/inputs/S/"
    "math_cot_probe/gamma_s.jsonl"
)
R2_INPUTS = Path("/root/autodl-tmp/cycle09_r2/getslice/inputs")
R3_SXH = Path("/root/autodl-tmp/cycle09_r3/sxh/corpora")
R3_FACTORS = Path("/root/autodl-tmp/cycle09_r3/factors")

ARMS = ("opd", "sft")
STEPS = (0, 5, 10, 20, 40, 160, 624)
LAYERS = (9, 18, 27)
GENERATION_SEEDS = (3, 17, 31)
WINDOW_SEED = 42
N_GENERATED = 32
WINDOW_TOKENS = 512
WINDOW_K = 3
MAX_NEW_TOKENS = 1024
# Profiling forward context cap. Generated corpora stay far below it; the fixed
# legacy dataset-CoT corpus reaches ~14.2k tokens, so keep the round-3 cap of
# 16384 to profile it whole (windows are drawn only from the generation region,
# so gram cost does not scale with this).
MAX_CONTEXT_TOKENS = 16384
TEMPERATURE = 0.6
TOP_P = 0.9
SCRATCH_LIMIT_GIB = 120

MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
GROUP_TO_MODULES = {
    "attn_qkv_input": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ),
    "attn_o_input": ("self_attn.o_proj",),
    "mlp_gate_up_input": ("mlp.gate_proj", "mlp.up_proj"),
    "mlp_down_input": ("mlp.down_proj",),
}
GROUP_CAPTURE_MODULE = {
    "attn_qkv_input": "self_attn.q_proj",
    "attn_o_input": "self_attn.o_proj",
    "mlp_gate_up_input": "mlp.gate_proj",
    "mlp_down_input": "mlp.down_proj",
}
MODULE_TO_GROUP = {
    module: group for group, modules in GROUP_TO_MODULES.items() for module in modules
}

DOMAIN_INSTRUCTIONS = {
    "math": "\nPlease reason step by step and put the final answer in \\boxed{}.",
    "ood": "\nAnswer the question accurately and concisely.",
    "general": "\nContinue naturally and coherently.",
    "bos": "",
}


@dataclass(frozen=True)
class ProbeTask:
    task_id: str
    probe_type: str
    domain: str
    corpus_path: str
    source_kind: str
    generation_seed: int | None
    generated: bool
    shared_across_arms: bool
    target_arm: str | None = None
    target_step: int | None = None
    alias_of: str | None = None


@dataclass
class WindowRecord:
    sample_id: str
    corpus_id: str
    window_index: int
    start: int
    end: int
    token_count: int
    relative_start: float
    relative_center: float
    relative_end: float
    position_bin: str


@dataclass
class PreparedSample:
    sample_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_weights: torch.Tensor
    eligible_start: int
    eligible_end: int
    windows: list[WindowRecord]
    source: dict[str, Any]


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def model_path(arm: str, step: int) -> Path:
    if int(step) == 0:
        return BASE_MODEL
    if arm == "opd":
        path = OPD_MODELS / step_label(step)
    elif arm == "sft":
        path = SFT_MODELS / step_label(step)
    else:
        raise ValueError(f"unknown arm: {arm}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"missing model: {path}")
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "::".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths() -> dict[str, Path]:
    paths = {
        "legacy_math": LEGACY_MATH,
        "E_ood": R2_INPUTS / "X_ood_knowledge/x_probe.jsonl",
        "E_general": R2_INPUTS / "X_general/x_probe.jsonl",
        "E_math_hard": R2_INPUTS / "X_math_hard/x_probe.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Round-4 source files: {missing}")
    return paths


def text_from_external_row(row: dict[str, Any]) -> str:
    output = row.get("output")
    if isinstance(output, dict) and output.get("text") is not None:
        return str(output["text"]).strip()
    for key in ("text", "question", "problem", "prompt"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    raise KeyError("external row has no supported text field")


def prompt_banks(n: int = N_GENERATED) -> dict[str, list[dict[str, str]]]:
    paths = source_paths()
    math_rows = read_jsonl(paths["legacy_math"])
    ood_rows = read_jsonl(paths["E_ood"])
    general_rows = read_jsonl(paths["E_general"])
    if min(len(math_rows), len(ood_rows), len(general_rows)) < n:
        raise ValueError("a Round-4 prompt source has fewer than n generated samples")
    return {
        "math": [
            {
                "sample_id": f"math_{idx:03d}",
                "prompt": str(row["question"]).strip(),
                "instruction": DOMAIN_INSTRUCTIONS["math"],
            }
            for idx, row in enumerate(math_rows[:n])
        ],
        "ood": [
            {
                "sample_id": f"ood_{idx:03d}",
                "prompt": text_from_external_row(row),
                "instruction": DOMAIN_INSTRUCTIONS["ood"],
            }
            for idx, row in enumerate(ood_rows[:n])
        ],
        "general": [
            {
                "sample_id": f"general_{idx:03d}",
                "prompt": text_from_external_row(row),
                "instruction": DOMAIN_INSTRUCTIONS["general"],
            }
            for idx, row in enumerate(general_rows[:n])
        ],
        "bos": [
            {
                "sample_id": f"bos_{idx:03d}",
                "prompt": "",
                "instruction": "",
            }
            for idx in range(n)
        ],
    }


def _tokenize_no_special(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def prepare_fixed_corpora(tokenizer, run_root: Path = RUN_ROOT) -> list[ProbeTask]:
    paths = source_paths()
    tasks: list[ProbeTask] = []

    legacy_target = run_root / "corpora/fixed/legacy_S_math.jsonl"
    if not legacy_target.exists():
        rows = []
        for idx, row in enumerate(read_jsonl(paths["legacy_math"])[:N_GENERATED]):
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            prompt_ids = _tokenize_no_special(tokenizer, question + "\n")
            answer_ids = _tokenize_no_special(tokenizer, answer)
            rows.append(
                {
                    "sample_id": f"legacy_math_{idx:03d}",
                    "probe_type": "legacy_S",
                    "domain": "math",
                    "source_kind": "fixed_dataset_cot_question_masked",
                    "prompt_text": question,
                    "generation_text": answer,
                    "prompt_token_ids": prompt_ids,
                    "generation_token_ids": answer_ids,
                    "full_token_ids": prompt_ids + answer_ids,
                    "eligible_start": len(prompt_ids),
                    "eligible_end": len(prompt_ids) + len(answer_ids),
                }
            )
        write_jsonl_atomic(legacy_target, rows)
    tasks.append(
        ProbeTask(
            task_id="legacy_S_math",
            probe_type="legacy_S",
            domain="math",
            corpus_path=str(legacy_target),
            source_kind="fixed_dataset_cot_question_masked",
            generation_seed=None,
            generated=False,
            shared_across_arms=True,
        )
    )

    caps = {"E_ood": 128, "E_general": 128, "E_math_hard": 30}
    domains = {"E_ood": "ood", "E_general": "general", "E_math_hard": "math_hard"}
    for probe, cap in caps.items():
        target = run_root / f"corpora/fixed/{probe}.jsonl"
        if not target.exists():
            rows = []
            for idx, row in enumerate(read_jsonl(paths[probe])[:cap]):
                text = text_from_external_row(row)
                token_ids = _tokenize_no_special(tokenizer, text)
                rows.append(
                    {
                        "sample_id": f"{probe}_{idx:03d}",
                        "probe_type": "E",
                        "domain": domains[probe],
                        "source_kind": f"external_fixed_{probe}",
                        "prompt_text": "",
                        "generation_text": text,
                        "prompt_token_ids": [],
                        "generation_token_ids": token_ids,
                        "full_token_ids": token_ids,
                        "eligible_start": 0,
                        "eligible_end": len(token_ids),
                    }
                )
            write_jsonl_atomic(target, rows)
        tasks.append(
            ProbeTask(
                task_id=probe,
                probe_type="E",
                domain=domains[probe],
                corpus_path=str(target),
                source_kind=f"external_fixed_{probe}",
                generation_seed=None,
                generated=False,
                shared_across_arms=True,
            )
        )
    return tasks


def generated_corpus_path(
    probe_type: str,
    domain: str,
    generation_seed: int,
    arm: str | None = None,
    step: int | None = None,
    run_root: Path = RUN_ROOT,
) -> Path:
    parts = [run_root, "corpora", "generated", probe_type]
    if arm is not None:
        parts.append(arm)
    if step is not None:
        parts.append(step_label(step))
    parts.extend([domain, f"gen_seed_{generation_seed}.jsonl"])
    path = Path(parts[0])
    for part in parts[1:]:
        path /= str(part)
    return path


def build_task_index(run_root: Path = RUN_ROOT) -> list[ProbeTask]:
    tasks = prepare_fixed_corpora_tokenizer_independent(run_root)
    for seed in GENERATION_SEEDS:
        for domain in ("math", "ood", "general", "bos"):
            path = generated_corpus_path("S", domain, seed, run_root=run_root)
            tasks.append(
                ProbeTask(
                    task_id=f"S_{domain}__g{seed}",
                    probe_type="S",
                    domain=domain,
                    corpus_path=str(path),
                    source_kind="base_generation",
                    generation_seed=seed,
                    generated=True,
                    shared_across_arms=True,
                )
            )
    for arm in ARMS:
        for step in STEPS:
            if arm == "opd":
                for seed in GENERATION_SEEDS:
                    path = generated_corpus_path("X", "math", seed, arm, step, run_root)
                    tasks.append(
                        ProbeTask(
                            task_id=f"X_opd_math__{step_label(step)}__g{seed}",
                            probe_type="X",
                            domain="math",
                            corpus_path=str(path),
                            source_kind="checkpoint_training_signal_rollout",
                            generation_seed=seed,
                            generated=True,
                            shared_across_arms=False,
                            target_arm=arm,
                            target_step=step,
                        )
                    )
            else:
                tasks.append(
                    ProbeTask(
                        task_id=f"X_sft_math__{step_label(step)}",
                        probe_type="X",
                        domain="math",
                        corpus_path=str(run_root / "corpora/fixed/legacy_S_math.jsonl"),
                        source_kind="fixed_dataset_cot_question_masked",
                        generation_seed=None,
                        generated=False,
                        shared_across_arms=False,
                        target_arm=arm,
                        target_step=step,
                        alias_of="legacy_S_math",
                    )
                )
            for domain in ("ood", "general", "bos"):
                for seed in GENERATION_SEEDS:
                    path = generated_corpus_path("H", domain, seed, arm, step, run_root)
                    tasks.append(
                        ProbeTask(
                            task_id=f"H_{arm}_{domain}__{step_label(step)}__g{seed}",
                            probe_type="H",
                            domain=domain,
                            corpus_path=str(path),
                            source_kind="checkpoint_nontraining_generation",
                            generation_seed=seed,
                            generated=True,
                            shared_across_arms=False,
                            target_arm=arm,
                            target_step=step,
                        )
                    )
    return tasks


def prepare_fixed_corpora_tokenizer_independent(run_root: Path) -> list[ProbeTask]:
    paths = {
        "legacy_S_math": run_root / "corpora/fixed/legacy_S_math.jsonl",
        "E_ood": run_root / "corpora/fixed/E_ood.jsonl",
        "E_general": run_root / "corpora/fixed/E_general.jsonl",
        "E_math_hard": run_root / "corpora/fixed/E_math_hard.jsonl",
    }
    metadata = {
        "legacy_S_math": ("legacy_S", "math", "fixed_dataset_cot_question_masked"),
        "E_ood": ("E", "ood", "external_fixed_E_ood"),
        "E_general": ("E", "general", "external_fixed_E_general"),
        "E_math_hard": ("E", "math_hard", "external_fixed_E_math_hard"),
    }
    return [
        ProbeTask(
            task_id=name,
            probe_type=metadata[name][0],
            domain=metadata[name][1],
            corpus_path=str(path),
            source_kind=metadata[name][2],
            generation_seed=None,
            generated=False,
            shared_across_arms=True,
        )
        for name, path in paths.items()
    ]


def tasks_for_model(all_tasks: list[ProbeTask], arm: str, step: int) -> list[ProbeTask]:
    selected = []
    for task in all_tasks:
        if task.alias_of is not None:
            if task.target_arm == arm and task.target_step == step:
                selected.append(task)
            continue
        if task.shared_across_arms:
            selected.append(task)
        elif task.target_arm == arm and task.target_step == step:
            selected.append(task)
    return selected


def position_bin(relative_center: float) -> str:
    if relative_center < 1.0 / 3.0:
        return "early"
    if relative_center < 2.0 / 3.0:
        return "mid"
    return "late"


def choose_windows(
    eligible_start: int,
    eligible_end: int,
    *,
    corpus_id: str,
    sample_id: str,
    window_seed: int = WINDOW_SEED,
    k: int = WINDOW_K,
    window_tokens: int = WINDOW_TOKENS,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    length = int(eligible_end) - int(eligible_start)
    if length <= 0:
        return [], np.zeros(0, dtype=np.float64)
    if length < window_tokens:
        windows = [(int(eligible_start), int(eligible_end))]
    else:
        n_offsets = length - window_tokens + 1
        rng = np.random.default_rng(stable_seed(window_seed, corpus_id, sample_id))
        if n_offsets >= k:
            offsets = rng.choice(n_offsets, size=k, replace=False)
        else:
            offsets = rng.choice(n_offsets, size=k, replace=True)
        windows = [
            (int(eligible_start + offset), int(eligible_start + offset + window_tokens))
            for offset in offsets.tolist()
        ]

    weights = np.zeros(length, dtype=np.float64)
    for start, end in windows:
        local_start = start - eligible_start
        local_end = end - eligible_start
        weights[local_start:local_end] += 1.0 / (len(windows) * max(end - start, 1))
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError(f"hierarchical window weights do not sum to one: {weights.sum()}")
    return windows, weights


def prepare_samples(
    corpus_path: Path,
    tokenizer,
    *,
    corpus_id: str,
    window_seed: int = WINDOW_SEED,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[PreparedSample]:
    prepared = []
    for row_idx, row in enumerate(read_jsonl(corpus_path)):
        sample_id = str(row.get("sample_id", row_idx))
        full_ids = [int(value) for value in row["full_token_ids"]]
        eligible_start = int(row["eligible_start"])
        eligible_end = int(row["eligible_end"])
        if len(full_ids) > max_context_tokens:
            overflow = len(full_ids) - max_context_tokens
            trim_left = min(overflow, eligible_start)
            if trim_left:
                full_ids = full_ids[trim_left:]
                eligible_start -= trim_left
                eligible_end -= trim_left
            if len(full_ids) > max_context_tokens:
                # Prompt-side trimming was not enough: keep the head of the
                # generation region and drop the tail, recording the deviation.
                dropped = len(full_ids) - max_context_tokens
                full_ids = full_ids[:max_context_tokens]
                eligible_end = min(eligible_end, max_context_tokens)
                row["tail_truncated_tokens"] = int(dropped)
                print(
                    f"[prepare_samples] tail-truncated {corpus_id}/{sample_id}: "
                    f"dropped={dropped} kept={max_context_tokens}",
                    flush=True,
                )

        windows, eligible_weights = choose_windows(
            eligible_start,
            eligible_end,
            corpus_id=corpus_id,
            sample_id=sample_id,
            window_seed=window_seed,
        )
        if not windows:
            continue
        token_weights = torch.zeros(len(full_ids), dtype=torch.float32)
        token_weights[eligible_start:eligible_end] = torch.from_numpy(
            eligible_weights.astype(np.float32)
        )
        length = max(eligible_end - eligible_start, 1)
        records = []
        for window_idx, (start, end) in enumerate(windows):
            rel_start = (start - eligible_start) / length
            rel_end = (end - eligible_start) / length
            rel_center = ((start + end) / 2.0 - eligible_start) / length
            records.append(
                WindowRecord(
                    sample_id=sample_id,
                    corpus_id=corpus_id,
                    window_index=window_idx,
                    start=start,
                    end=end,
                    token_count=end - start,
                    relative_start=float(rel_start),
                    relative_center=float(rel_center),
                    relative_end=float(rel_end),
                    position_bin=position_bin(float(rel_center)),
                )
            )
        prepared.append(
            PreparedSample(
                sample_id=sample_id,
                input_ids=torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
                attention_mask=torch.ones((1, len(full_ids)), dtype=torch.long),
                token_weights=token_weights,
                eligible_start=eligible_start,
                eligible_end=eligible_end,
                windows=records,
                source=row,
            )
        )
    if not prepared:
        raise ValueError(f"no usable window-v2 samples in {corpus_path}")
    return prepared


def effective_rank(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return 0.0
    probabilities = array / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(np.exp(entropy))


def tail_energy(values: Iterable[float], rank: int) -> float:
    sigma = np.asarray(list(values), dtype=np.float64)
    energy = np.square(sigma)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    return float(energy[int(rank):].sum() / total)


def functional_rank(values: Iterable[float], epsilon: float) -> int:
    sigma = np.asarray(list(values), dtype=np.float64)
    energy = np.square(sigma)
    total = float(energy.sum())
    if total <= 0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, 1.0 - float(epsilon), side="left") + 1)


def rms_log_core_drift(current: Iterable[float], base: Iterable[float], rank: int) -> float:
    current_values = np.asarray(list(current), dtype=np.float64)
    base_values = np.asarray(list(base), dtype=np.float64)
    width = min(int(rank), current_values.size, base_values.size)
    if width <= 0:
        return float("nan")
    current_norm = current_values / max(float(current_values.sum()), 1e-30)
    base_norm = base_values / max(float(base_values.sum()), 1e-30)
    delta = np.log(np.clip(current_norm[:width], 1e-30, None)) - np.log(
        np.clip(base_norm[:width], 1e-30, None)
    )
    return float(np.sqrt(np.mean(np.square(delta))))


def scratch_bytes(run_root: Path = RUN_ROOT) -> int:
    total = 0
    root = run_root / "scratch"
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def assert_scratch_budget(run_root: Path = RUN_ROOT, limit_gib: int = SCRATCH_LIMIT_GIB) -> None:
    used = scratch_bytes(run_root)
    limit = int(limit_gib) * 1024**3
    if used > limit:
        raise RuntimeError(
            f"Round-4 scratch budget exceeded: {used / 1024**3:.2f} GiB > {limit_gib} GiB"
        )


def task_to_dict(task: ProbeTask) -> dict[str, Any]:
    return asdict(task)

