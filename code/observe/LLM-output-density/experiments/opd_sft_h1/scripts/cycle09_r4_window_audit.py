#!/usr/bin/env python3
"""R4-0: audit prompt-token share in Cycle 09 v1 first windows."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

import cycle09_r4_common as c


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def summarize_cell(
    rows: list[dict[str, Any]],
    *,
    round_name: str,
    taxonomy: str,
    probe: str,
    domain: str,
    arm: str,
    step: str,
    source_path: Path,
    boundary_status: str,
) -> dict[str, Any]:
    shares = [float(row["share"]) for row in rows if row.get("share") is not None]
    prompt_tokens = [int(row["prompt_tokens_in_first_window"]) for row in rows if row.get("share") is not None]
    window_tokens = [int(row["first_window_tokens"]) for row in rows if row.get("share") is not None]
    return {
        "round": round_name,
        "taxonomy": taxonomy,
        "probe": probe,
        "domain": domain,
        "arm": arm,
        "step": step,
        "source_path": str(source_path),
        "boundary_status": boundary_status,
        "n_records": len(rows),
        "n_measurable": len(shares),
        "prompt_share_min": "" if not shares else f"{min(shares):.8f}",
        "prompt_share_mean": "" if not shares else f"{statistics.fmean(shares):.8f}",
        "prompt_share_max": "" if not shares else f"{max(shares):.8f}",
        "all_prompt_window_count": sum(share >= 1.0 - 1e-12 for share in shares),
        "mean_prompt_tokens_in_first_window": (
            "" if not prompt_tokens else f"{statistics.fmean(prompt_tokens):.4f}"
        ),
        "mean_first_window_tokens": (
            "" if not window_tokens else f"{statistics.fmean(window_tokens):.4f}"
        ),
        "v1_window_definition": "first_512_tokens_of_each_record",
        "audit_note": (
            "Record-level first-window audit; the legacy concat/random-character loader "
            "did not preserve a record-to-window map, so actual cached-window provenance "
            "cannot be reconstructed."
        ),
    }


def split_share(tokenizer, prompt: str, continuation: str) -> dict[str, Any]:
    prompt_n = token_count(tokenizer, prompt)
    continuation_n = token_count(tokenizer, continuation)
    first_n = min(c.WINDOW_TOKENS, prompt_n + continuation_n)
    prompt_first = min(prompt_n, first_n)
    return {
        "share": (prompt_first / first_n) if first_n else None,
        "prompt_tokens_in_first_window": prompt_first,
        "first_window_tokens": first_n,
    }


def no_prompt_share(tokenizer, text: str) -> dict[str, Any]:
    first_n = min(c.WINDOW_TOKENS, token_count(tokenizer, text))
    return {
        "share": 0.0 if first_n else None,
        "prompt_tokens_in_first_window": 0,
        "first_window_tokens": first_n,
    }


def legacy_math_rows(tokenizer, path: Path) -> list[dict[str, Any]]:
    output = []
    for row in c.read_jsonl(path)[: c.N_GENERATED]:
        output.append(
            split_share(
                tokenizer,
                str(row.get("question", "")).strip() + "\n",
                str(row.get("answer", "")).strip(),
            )
        )
    return output


def external_rows(tokenizer, path: Path, cap: int | None = None) -> list[dict[str, Any]]:
    source = c.read_jsonl(path)
    if cap is not None:
        source = source[:cap]
    return [no_prompt_share(tokenizer, c.text_from_external_row(row)) for row in source]


def unknown_rows(path: Path, cap: int | None = None) -> list[dict[str, Any]]:
    rows = c.read_jsonl(path)
    if cap is not None:
        rows = rows[:cap]
    return [
        {
            "share": None,
            "prompt_tokens_in_first_window": 0,
            "first_window_tokens": 0,
        }
        for _ in rows
    ]


def r3_generated_rows(tokenizer, path: Path) -> tuple[list[dict[str, Any]], str]:
    values = []
    status = "measured_from_prompt_instruction_generation_fields"
    for row in c.read_jsonl(path):
        prompt = str(row.get("prompt", ""))
        instruction = str(row.get("instruction", ""))
        generation = str(row.get("generation", ""))
        if not prompt and not instruction and row.get("role") == "S":
            status = "missing_boundary"
            values.append(
                {
                    "share": None,
                    "prompt_tokens_in_first_window": 0,
                    "first_window_tokens": 0,
                }
            )
            continue
        values.append(split_share(tokenizer, prompt + instruction + "\n", generation))
    return values, status


def audit_round2(tokenizer) -> list[dict[str, Any]]:
    rows = []
    legacy = legacy_math_rows(tokenizer, c.LEGACY_MATH)
    for arm in c.ARMS:
        rows.append(
            summarize_cell(
                legacy,
                round_name="round2",
                taxonomy="legacy_S_X",
                probe="S",
                domain="math",
                arm=arm,
                step="all",
                source_path=c.LEGACY_MATH,
                boundary_status="measured_question_vs_answer",
            )
        )

    probes = {
        "X_math": Path(
            "/root/autodl-tmp/cycle07_base_sft_trajectory/getslice/inputs/"
            "X_base/x_probe.jsonl"
        ),
        "X_ood_knowledge": c.R2_INPUTS / "X_ood_knowledge/x_probe.jsonl",
        "X_general": c.R2_INPUTS / "X_general/x_probe.jsonl",
        "X_math_hard": c.R2_INPUTS / "X_math_hard/x_probe.jsonl",
    }
    bos_candidates = [
        c.R2_INPUTS / "X_bos/x_probe.jsonl",
        Path("/root/autodl-tmp/cycle04_smoke/getslice/inputs/X_bos/x_probe.jsonl"),
    ]
    for candidate in bos_candidates:
        if candidate.exists():
            probes["X_bos"] = candidate
            break

    for probe, path in probes.items():
        if not path.exists():
            continue
        if probe in {"X_ood_knowledge", "X_general", "X_math_hard"}:
            values = external_rows(tokenizer, path)
            status = "not_applicable_external_fixed_text"
        elif probe == "X_bos":
            values = external_rows(tokenizer, path)
            status = "no_prompt_bos_generation"
        else:
            values = unknown_rows(path, c.N_GENERATED)
            status = "boundary_not_stored_in_v1_file"
        domain = {
            "X_math": "math",
            "X_ood_knowledge": "ood",
            "X_general": "general",
            "X_math_hard": "math_hard",
            "X_bos": "bos",
        }[probe]
        for arm in c.ARMS:
            rows.append(
                summarize_cell(
                    values,
                    round_name="round2",
                    taxonomy="legacy_S_X",
                    probe=probe,
                    domain=domain,
                    arm=arm,
                    step="all",
                    source_path=path,
                    boundary_status=status,
                )
            )
    return rows


def parse_r3_corpus_path(path: Path) -> tuple[str, str, str, str]:
    relative = path.relative_to(c.R3_SXH)
    role, arm, step = relative.parts[:3]
    domain = path.stem
    return role, arm, step, domain


def audit_round3(tokenizer) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(c.R3_SXH.rglob("*.jsonl")):
        role, source_arm, step, domain = parse_r3_corpus_path(path)
        if role == "X" and source_arm == "sft":
            values = legacy_math_rows(tokenizer, c.LEGACY_MATH)
            status = "measured_question_vs_answer"
        else:
            values, status = r3_generated_rows(tokenizer, path)
        arms = c.ARMS if role == "S" else (source_arm,)
        for arm in arms:
            rows.append(
                summarize_cell(
                    values,
                    round_name="round3",
                    taxonomy="round3_S_X_H",
                    probe=role,
                    domain=domain,
                    arm=arm,
                    step=step,
                    source_path=path,
                    boundary_status=status,
                )
            )

    legacy = legacy_math_rows(tokenizer, c.LEGACY_MATH)
    for arm in c.ARMS:
        rows.append(
            summarize_cell(
                legacy,
                round_name="round3",
                taxonomy="legacy_R3_getslice",
                probe="S",
                domain="math",
                arm=arm,
                step="all",
                source_path=c.LEGACY_MATH,
                boundary_status="measured_question_vs_answer",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=c.MINI_ROOT / "R4_window_audit.csv")
    parser.add_argument("--base-model", type=Path, default=c.BASE_MODEL)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    rows = audit_round2(tokenizer) + audit_round3(tokenizer)
    fields = [
        "round",
        "taxonomy",
        "probe",
        "domain",
        "arm",
        "step",
        "source_path",
        "boundary_status",
        "n_records",
        "n_measurable",
        "prompt_share_min",
        "prompt_share_mean",
        "prompt_share_max",
        "all_prompt_window_count",
        "mean_prompt_tokens_in_first_window",
        "mean_first_window_tokens",
        "v1_window_definition",
        "audit_note",
    ]
    c.write_csv_atomic(args.output, rows, fields)
    print(f"[R4-0] wrote {args.output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()

