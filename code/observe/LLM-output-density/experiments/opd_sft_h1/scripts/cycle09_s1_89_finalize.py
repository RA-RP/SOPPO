#!/usr/bin/env python3
"""Validate and hand back the raw S1-8/S1-9 emergency artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/root/LLM-output-density")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
S1_8 = MINI / "S1_mmlupro_loglik.csv"
S1_8_MANIFEST = MINI / "S1_mmlupro_loglik_manifest.json"
S1_9 = MINI / "S1_ifeval_breakdown.csv"
S1_9_MANIFEST = MINI / "S1_ifeval_breakdown_manifest.json"
PROMPT = MINI / "S1_mmlupro_loglik_prompt_template.txt"
HANDOFF = MINI / "mini_stage1_emergency_s1_89_handoff.md"
MANIFEST = MINI / "S1_89_handoff_manifest.json"
CODE_EVOLUTION = REPO / "mypaper/code/code_evolution.md"
MARKER = "## Cycle 09 Stage 1 Emergency S1-8/S1-9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    atomic_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        path,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row[field]) for field in fields) + " |")
    return lines


def validate() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    required = (S1_8, S1_8_MANIFEST, S1_9, S1_9_MANIFEST, PROMPT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing emergency artifacts: {missing}")
    rows8 = read_csv(S1_8)
    rows9 = read_csv(S1_9)
    if len(rows8) != 30:
        raise RuntimeError(f"S1-8 row count {len(rows8)} != 30")
    if len(rows9) != 270:
        raise RuntimeError(f"S1-9 row count {len(rows9)} != 270")
    if len({(row["arm"], row["step"]) for row in rows8}) != 30:
        raise RuntimeError("S1-8 arm-step grid is not unique")
    if len(
        {
            (row["arm"], row["step"], row["instruction_category"])
            for row in rows9
        }
    ) != 270:
        raise RuntimeError("S1-9 arm-step-category grid is not unique")
    for path in (S1_8_MANIFEST, S1_9_MANIFEST):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"manifest is not complete: {path}")
    return rows8, rows9


def write_handoff(rows8: list[dict[str, str]], rows9: list[dict[str, str]]) -> None:
    fields8 = [
        "arm",
        "step",
        "n",
        "strict_exact_match",
        "flexible_exact_match",
        "acc_ll",
        "acc_ll_norm",
    ]
    fields9 = [
        "arm",
        "step",
        "instruction_category",
        "n_prompts",
        "n_pass",
        "pass_rate",
        "resp_len",
        "resp_len_median",
        "resp_words_mean",
        "resp_words_median",
    ]
    lines = [
        "# Stage 1 Emergency S1-8 / S1-9 Raw Handoff",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## S1-8 MMLU-Pro Loglikelihood",
        "",
        *markdown_table(rows8, fields8),
        "",
        "Provenance:",
        f"- data: `{S1_8}`; sha256 `{sha256_file(S1_8)}`",
        f"- manifest: `{S1_8_MANIFEST}`; sha256 `{sha256_file(S1_8_MANIFEST)}`",
        f"- full prompt: `{PROMPT}`; sha256 `{sha256_file(PROMPT)}`",
        "",
        "## S1-9 IFEval Instruction Categories",
        "",
        *markdown_table(rows9, fields9),
        "",
        "Provenance:",
        f"- data: `{S1_9}`; sha256 `{sha256_file(S1_9)}`",
        f"- manifest: `{S1_9_MANIFEST}`; sha256 `{sha256_file(S1_9_MANIFEST)}`",
        "- generation rerun: `false`",
        "",
        "Raw readings only.",
        "",
    ]
    atomic_text("\n".join(lines), HANDOFF)


def append_code_evolution() -> None:
    existing = CODE_EVOLUTION.read_text(encoding="utf-8")
    if MARKER in existing:
        return
    block = "\n".join(
        [
            "",
            MARKER,
            "",
            "紧急追加 S1-8/S1-9 已按 `stage_plan_handoff.md` 的冻结规格执行；本节只登记产物与口径，不解释、不裁决。",
            "",
            "| task | artifact | rows | sha256 |",
            "|---|---|---:|---|",
            f"| S1-8 MMLU-Pro conditional LL | {S1_8} | 30 | {sha256_file(S1_8)} |",
            f"| S1-9 IFEval native-category audit | {S1_9} | 270 | {sha256_file(S1_9)} |",
            f"| raw Theory handoff | {HANDOFF} | - | {sha256_file(HANDOFF)} |",
            "",
            "**S1-8 provenance**：既有 `--limit 100/class, seed=42` 日志中锁定同一 1400 个 question_id；0-shot、非 CoT、full-option continuation；lm-eval `acc`=raw conditional-LL argmax，`acc_norm`=LL/choice-character-length argmax；base 只测一次并映射到三臂 step 0；逐样本 LL 日志保留。",
            "",
            "**S1-9 provenance**：只读取三臂十点现存 IFEval 逐样本 JSONL，零生成重跑；按 `instruction_id_list` 冒号前原生前缀分组，使用已存 `prompt_level_strict_acc`；`resp_len` 为 raw response Unicode 字符数均值。",
            "",
        ]
    )
    atomic_text(existing.rstrip() + "\n" + block, CODE_EVOLUTION)


def main() -> None:
    rows8, rows9 = validate()
    write_handoff(rows8, rows9)
    append_code_evolution()
    artifacts = []
    for path in (S1_8, S1_8_MANIFEST, S1_9, S1_9_MANIFEST, PROMPT, HANDOFF):
        artifacts.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_json(
        {
            "schema_version": 1,
            "task": "Stage 1 emergency S1-8/S1-9 Theory handoff",
            "status": "COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation": False,
            "adjudication": False,
            "artifacts": artifacts,
            "code_evolution": str(CODE_EVOLUTION),
            "code_evolution_sha256": sha256_file(CODE_EVOLUTION),
        },
        MANIFEST,
    )
    print(f"[S1-8/9 handoff] complete {HANDOFF}")


if __name__ == "__main__":
    main()
