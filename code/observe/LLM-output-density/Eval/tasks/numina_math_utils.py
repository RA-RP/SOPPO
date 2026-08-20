"""Robust answer extraction for the local NuminaMath lm-eval task."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


INVALID_ANSWER = "[invalid]"


def extract_last_boxed(text: Any) -> Optional[str]:
    """Return the content of the last \\boxed{...}/\\fbox{...}, supporting nested braces."""
    if text is None:
        return None
    raw = str(text)
    idx = max(raw.rfind("\\boxed"), raw.rfind("\\fbox"))
    if idx < 0:
        return None

    brace_start = raw.find("{", idx)
    if brace_start < 0:
        # Supports the uncommon "\boxed 42" form.
        after = raw[idx:].split(maxsplit=1)
        if len(after) == 2:
            return after[1].splitlines()[0].strip()
        return None

    depth = 0
    for pos in range(brace_start, len(raw)):
        char = raw[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[brace_start + 1 : pos].strip()
    return None


def extract_fallback_answer(text: Any) -> str:
    if text is None:
        return INVALID_ANSWER
    raw = str(text).strip()
    if not raw:
        return INVALID_ANSWER

    boxed = extract_last_boxed(raw)
    if boxed:
        return boxed

    patterns = [
        r"Final Answer\s*:\s*(.+)",
        r"The final answer is\s*(.+)",
        r"Answer\s*:\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else INVALID_ANSWER


def normalize_answer(value: Any) -> str:
    answer = extract_last_boxed(value)
    if answer is None:
        answer = "" if value is None else str(value)
    answer = answer.strip().strip(" \t\n\r.。;；")

    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2]
    if answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2]

    while len(answer) >= 2 and answer[0] == "$" and answer[-1] == "$":
        answer = answer[1:-1].strip()

    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\$": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
    }
    for before, after in replacements.items():
        answer = answer.replace(before, after)

    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", answer)
    answer = answer.replace(",", "")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.strip(" \t\n\r.。;；")
    while len(answer) >= 2 and answer[0] == "$" and answer[-1] == "$":
        answer = answer[1:-1].strip()
    answer = answer.strip(" \t\n\r.。;；")
    return answer


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, int]:
    prediction_text = results[0] if results else ""
    pred = normalize_answer(extract_fallback_answer(prediction_text))
    gold = normalize_answer(doc.get("answer", ""))
    return {"exact_match": int(pred == gold and pred != "")}
