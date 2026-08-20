"""Prompt and target adapters for the Stage-1 MMLU-Pro LL audit."""

from __future__ import annotations


PROMPT_TEMPLATE = (
    "The following is a multiple choice question about {category}.\n\n"
    "Question: {question}\n"
    "Answer:"
)
CHOICE_TEMPLATE = "{option}"


def doc_to_text(doc: dict) -> str:
    return PROMPT_TEMPLATE.format(
        category=str(doc["category"]).replace("_", " ").strip(),
        question=str(doc["question"]).strip(),
    )


def doc_to_choice(doc: dict) -> list[str]:
    return [CHOICE_TEMPLATE.format(option=str(option).strip()) for option in doc["options"]]


def doc_to_target(doc: dict) -> int:
    return int(doc["answer_index"])
