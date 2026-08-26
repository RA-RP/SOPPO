"""Independently verify finite-only, earlier-step Round3 checkpoint selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jsonlines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    selections = []
    with jsonlines.open(run_dir / "logs" / "metrics.jsonl") as reader:
        for row in reader:
            if row.get("record_type") == "selection":
                selections.append(row)
    if len(selections) != 10 or [row["checkpoint_step"] for row in selections] != list(range(25, 251, 25)):
        raise ValueError("Round3 selection verifier requires all ten registered checkpoints")
    finite = [
        row
        for row in selections
        if row.get("valid") and row.get("eval_selection_loss") is not None and math.isfinite(float(row["eval_selection_loss"]))
    ]
    if not finite:
        raise RuntimeError("All Round3 checkpoint selection losses are invalid")
    selected = min(finite, key=lambda row: (float(row["eval_selection_loss"]), int(row["checkpoint_step"])))
    best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
    expected = (
        float(selected["eval_selection_loss"]),
        int(selected["checkpoint_step"]),
        str(Path(selected["checkpoint"]).resolve()),
    )
    actual = (
        float(best["eval_selection_loss"]),
        int(best["checkpoint_step"]),
        str(Path(best["checkpoint"]).resolve()),
    )
    if actual != expected:
        raise ValueError("Round3 best.json differs from finite lexicographic selection")
    output = {
        "status": "verified",
        "method_id": best["method_id"],
        "candidate_checkpoints": 10,
        "finite_checkpoints": len(finite),
        "invalid_checkpoints": 10 - len(finite),
        "checkpoint_step": expected[1],
        "checkpoint": expected[2],
        "eval_selection_loss": expected[0],
        "tie_break": "lexicographic_raw_loss_then_earlier_step",
    }
    path = run_dir / "selection_verified.json"
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 selection verification: {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

