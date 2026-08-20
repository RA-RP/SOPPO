from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _records(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        return list(rows.to_dict(orient="records"))
    return [dict(row) for row in rows]


def _value(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _id(row: Mapping[str, Any], prefix: str) -> tuple[Any, Any]:
    run_id = _value(row, (f"{prefix}_run_id", "run_id", "RunID", "Source"))
    checkpoint_id = _value(row, (f"{prefix}_checkpoint_id", "checkpoint_id", "CheckpointID", "DataSize"))
    return run_id, checkpoint_id


def _gain(row: Mapping[str, Any], prefix: str) -> float | None:
    return _float_or_none(
        _value(
            row,
            (
                f"{prefix}_math500_gain",
                "math500_gain",
                "Math500Gain",
                "MATH500_gain",
                "MATH500 Gain",
            ),
        )
    )


def _metric(row: Mapping[str, Any], prefix: str, name: str) -> float | None:
    return _float_or_none(_value(row, (f"{prefix}_{name}", name, name.title().replace("_", ""))))


def match_opd_to_sft(opd_rows: Any, sft_rows: Any, max_gap: float = 2.0) -> list[dict[str, Any]]:
    opd_records = _records(opd_rows)
    sft_records = _records(sft_rows)
    matches: list[dict[str, Any]] = []

    for opd in opd_records:
        opd_run_id, opd_checkpoint_id = _id(opd, "opd")
        opd_gain = _gain(opd, "opd")
        best_sft: dict[str, Any] | None = None
        best_gap: float | None = None

        if opd_gain is not None:
            for sft in sft_records:
                sft_gain = _gain(sft, "sft")
                if sft_gain is None:
                    continue
                gap = abs(opd_gain - sft_gain)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_sft = sft

        row: dict[str, Any] = {
            "opd_run_id": opd_run_id,
            "opd_checkpoint_id": opd_checkpoint_id,
            "opd_math500_gain": opd_gain,
            "sft_run_id": None,
            "sft_checkpoint_id": None,
            "sft_math500_gain": None,
            "math500_gain_gap": best_gap,
            "match_status": "no_sft_candidate",
            "general_ood_penalty_delta": None,
            "worst_ood_drop_delta": None,
        }

        if best_sft is not None:
            sft_run_id, sft_checkpoint_id = _id(best_sft, "sft")
            sft_gain = _gain(best_sft, "sft")
            row.update(
                {
                    "sft_run_id": sft_run_id,
                    "sft_checkpoint_id": sft_checkpoint_id,
                    "sft_math500_gain": sft_gain,
                    "match_status": "matched" if best_gap is not None and best_gap <= max_gap else "unmatched_nearest",
                }
            )

            opd_penalty = _metric(opd, "opd", "general_ood_penalty")
            sft_penalty = _metric(best_sft, "sft", "general_ood_penalty")
            if opd_penalty is not None and sft_penalty is not None:
                row["general_ood_penalty_delta"] = opd_penalty - sft_penalty

            opd_worst = _metric(opd, "opd", "worst_ood_drop")
            sft_worst = _metric(best_sft, "sft", "worst_ood_drop")
            if opd_worst is not None and sft_worst is not None:
                row["worst_ood_drop_delta"] = opd_worst - sft_worst

        matches.append(row)

    return matches
