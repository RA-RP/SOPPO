#!/usr/bin/env python3
"""S1-1 extraction audit and S1-2 frozen flexible-chain rescoring."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer

import cycle09_r4_common as c4


MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
LOG_MANIFEST = MINI / "S1_mmlupro_log_manifest.json"
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ARMS = ("opd", "sft", "offkd")
LETTERS = set("ABCDEFGHIJ")

# Tier 1 exactly mirrors the installed lm-eval task's current regex.
STRICT_RE = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")
ANSWER_RE = re.compile(r"Answer\s*[:：]\s*\(?([A-J])\)?")
ZH_ANSWER_RE = re.compile(r"答案\s*(?:是|为|[:：])?\s*\(?([A-J])\)?")
BOXED_RE = re.compile(r"\\boxed\s*\{\s*\(?([A-J])\)?\s*\}")
STANDALONE_RE = re.compile(r"\b([A-J])\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def response_text(record: dict) -> str:
    response = record.get("resps")
    while isinstance(response, list) and response:
        response = response[0]
    if response is None:
        return ""
    return str(response)


def strict_prediction(record: dict) -> str | None:
    value = record.get("filtered_resps")
    while isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str) and value.upper() in LETTERS and len(value) == 1:
        return value.upper()
    return None


def max_generation_tokens(record: dict) -> int:
    try:
        return int(record["arguments"]["gen_args_0"]["arg_1"]["max_gen_toks"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("sample record lacks max_gen_toks provenance")


def flexible_extract(text: str) -> tuple[str | None, int]:
    match = STRICT_RE.search(text)
    if match:
        return match.group(1), 1
    for regex in (ANSWER_RE, ZH_ANSWER_RE):
        match = regex.search(text)
        if match:
            return match.group(1), 2
    match = BOXED_RE.search(text)
    if match:
        return match.group(1), 3
    matches = STANDALONE_RE.findall(text)
    if matches:
        return matches[-1], 4
    return None, 0


def failure_shape(
    text: str, strict_success: bool, response_tokens: int, max_tokens: int
) -> str:
    if strict_success:
        return "not_failure"
    if not text.strip():
        return "empty"
    if response_tokens >= max_tokens:
        return "truncated"
    if STANDALONE_RE.search(text) is None:
        return "no_uppercase_standalone_A_to_J"
    return "letter_bad_format"


def load_rows(tokenizer) -> tuple[pd.DataFrame, dict]:
    if not LOG_MANIFEST.is_file():
        raise FileNotFoundError(
            f"run cycle09_s1_mmlupro_logs.py first: {LOG_MANIFEST}"
        )
    manifest = json.loads(LOG_MANIFEST.read_text(encoding="utf-8"))
    cells = manifest.get("cells", [])
    if len(cells) != 30:
        raise RuntimeError(f"incomplete log manifest: cells={len(cells)}")

    rows = []
    for cell in cells:
        arm, step = cell["arm"], int(cell["step"])
        cell_rows = []
        for file_info in cell["sample_files"]:
            subject = file_info["subject"]
            path = Path(file_info["path"])
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    text = response_text(record)
                    prediction = strict_prediction(record)
                    flexible_prediction, tier = flexible_extract(text)
                    cell_rows.append(
                        {
                            "arm": arm,
                            "step": step,
                            "subject": subject,
                            "target": str(record["target"]).upper(),
                            "strict_prediction": prediction,
                            "strict_extract_success": prediction is not None,
                            "strict_exact_match": float(record.get("exact_match", 0.0)),
                            "flexible_prediction": flexible_prediction,
                            "flexible_tier": tier,
                            "flexible_exact_match": float(
                                flexible_prediction == str(record["target"]).upper()
                            ),
                            "response_text": text,
                            "response_chars": len(text),
                            "max_generation_tokens": max_generation_tokens(record),
                        }
                    )
        if len(cell_rows) != 1400:
            raise RuntimeError(f"{arm}/{step} has {len(cell_rows)} rows")

        for start in range(0, len(cell_rows), 64):
            batch = cell_rows[start : start + 64]
            encoded = tokenizer(
                [row["response_text"] for row in batch],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
            for row, length in zip(batch, lengths):
                row["response_tokens"] = int(length)
                row["failure_shape"] = failure_shape(
                    row["response_text"],
                    row["strict_extract_success"],
                    row["response_tokens"],
                    row["max_generation_tokens"],
                )
                del row["response_text"]
        rows.extend(cell_rows)

    frame = pd.DataFrame(rows)
    expected = set((arm, step) for arm in ARMS for step in STEPS)
    observed = set(zip(frame["arm"], frame["step"]))
    if observed != expected or len(frame) != 30 * 1400:
        raise RuntimeError(
            f"incomplete audit grid rows={len(frame)}, missing={sorted(expected-observed)}"
        )
    return frame, manifest


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def safe_median(values: pd.Series) -> float:
    return float(values.median()) if len(values) else float("nan")


def extraction_audit(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    shapes = (
        "empty",
        "no_uppercase_standalone_A_to_J",
        "letter_bad_format",
        "truncated",
    )
    for (arm, step), group in frame.groupby(["arm", "step"], sort=True):
        failures = group[~group["strict_extract_success"]]
        row = {
            "arm": arm,
            "step": int(step),
            "n": len(group),
            "n_extract_fail": len(failures),
            "extract_fail_rate": len(failures) / len(group),
            "response_tokens_mean": float(group["response_tokens"].mean()),
            "response_tokens_median": float(group["response_tokens"].median()),
            "response_chars_mean": float(group["response_chars"].mean()),
            "response_chars_median": float(group["response_chars"].median()),
            "failed_response_tokens_mean": safe_mean(failures["response_tokens"]),
            "failed_response_tokens_median": safe_median(failures["response_tokens"]),
        }
        for shape in shapes:
            count = int((failures["failure_shape"] == shape).sum())
            row[f"failure_{shape}_n"] = count
            row[f"failure_{shape}_fraction_of_failures"] = (
                count / len(failures) if len(failures) else np.nan
            )
            row[f"failure_{shape}_rate_all_samples"] = count / len(group)
        if sum(row[f"failure_{shape}_n"] for shape in shapes) != len(failures):
            raise RuntimeError(f"failure shapes do not partition {arm}/{step}")
        records.append(row)
    return pd.DataFrame(records)


def flexible_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (arm, step), group in frame.groupby(["arm", "step"], sort=True):
        failures = group[~group["strict_extract_success"]]
        recovered = failures[failures["flexible_tier"] > 0]
        row = {
            "arm": arm,
            "step": int(step),
            "n": len(group),
            "exact_match": float(group["strict_exact_match"].mean()),
            "mmlu_pro_flexible": float(group["flexible_exact_match"].mean()),
            "format_component_flexible_minus_exact": float(
                group["flexible_exact_match"].mean()
                - group["strict_exact_match"].mean()
            ),
            "strict_extract_fail_rate": float(
                1.0 - group["strict_extract_success"].mean()
            ),
            "flexible_no_extract_rate": float((group["flexible_tier"] == 0).mean()),
            "n_strict_fail": len(failures),
            "n_strict_fail_recovered": len(recovered),
            "strict_fail_recovery_rate": (
                len(recovered) / len(failures) if len(failures) else np.nan
            ),
        }
        for tier in range(1, 5):
            row[f"tier{tier}_rate_all_samples"] = float(
                (group["flexible_tier"] == tier).mean()
            )
            row[f"tier{tier}_fraction_of_strict_failures"] = (
                float((failures["flexible_tier"] == tier).mean())
                if len(failures)
                else np.nan
            )
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    frame, log_manifest = load_rows(tokenizer)
    audit = extraction_audit(frame)
    flexible = flexible_table(frame)
    if len(audit) != 30 or len(flexible) != 30:
        raise RuntimeError("S1-1/2 summaries are not three arms x ten steps")
    atomic_csv(audit, MINI / "S1_mmlupro_extract_audit.csv")
    atomic_csv(flexible, MINI / "S1_mmlupro_flexible.csv")

    tokenizer_json = c4.BASE_MODEL / "tokenizer.json"
    atomic_json(
        {
            "schema_version": 1,
            "tasks": ["S1-1", "S1-2"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arms": list(ARMS),
            "steps": list(STEPS),
            "n_rows_with_arm_aliases": len(frame),
            "n_unique_log_cells": log_manifest["n_unique_log_cells"],
            "n_samples_per_cell": 1400,
            "strict_extraction": {
                "source": (
                    "/root/miniconda3/envs/density/lib/python3.12/site-packages/"
                    "lm_eval/tasks/mmlu_pro/_default_template_yaml"
                ),
                "regex": STRICT_RE.pattern,
                "prediction": "stored filtered_resps; regex shown for provenance",
            },
            "flexible_chain": [
                {"tier": 1, "regex": STRICT_RE.pattern},
                {
                    "tier": 2,
                    "regexes": [ANSWER_RE.pattern, ZH_ANSWER_RE.pattern],
                },
                {"tier": 3, "regex": BOXED_RE.pattern},
                {
                    "tier": 4,
                    "regex": STANDALONE_RE.pattern,
                    "selection": "last match in full response; uppercase only",
                },
            ],
            "failure_shape_partition_order": [
                "empty",
                "truncated (token length >= stored max_gen_toks)",
                "no_uppercase_standalone_A_to_J",
                "letter_bad_format",
            ],
            "response_length_tokenizer": {
                "path": str(c4.BASE_MODEL),
                "tokenizer_json_sha256": sha256_file(tokenizer_json),
                "add_special_tokens": False,
            },
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "log_manifest": str(LOG_MANIFEST),
            "log_manifest_sha256": sha256_file(LOG_MANIFEST),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MINI / "S1_mmlupro_audit_manifest.json",
    )
    print("=== S1-1 ===")
    print(audit.to_string(index=False))
    print("=== S1-2 ===")
    print(flexible.to_string(index=False))


if __name__ == "__main__":
    main()
