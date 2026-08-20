#!/usr/bin/env python3
"""Cycle 08 data prep — build the verl OPD prompt parquet.

OPD is on-policy: the student generates its own rollouts, the teacher scores
them. So training data = PROMPTS ONLY (unlike Cycle 07 SFT, which needed
question+response pairs). We reuse the EXACT same seed=42 5000-row sample as
Cycle 07 so the OPD vs SFT trajectories are comparable prompt-for-prompt.

Output: a verl-format parquet (schema matching examples/data_preprocess/gsm8k.py):
  data_source, prompt(messages), ability, reward_model{style,ground_truth}, extra_info{index,answer}

The prompt is the raw question + the Cycle 07 boxed instruction (single newline,
identical to runner_think.INSTR and the Math-CoT-20k `message` field), wrapped as
a single user-turn messages list; verl applies the chat template at load time.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SRC_PARQUET = Path("/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet")
PROBE_ROWS = Path("/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/probe_rows.jsonl")
RUN_ROOT = Path("/root/autodl-tmp/cycle08_opd_trajectory")
N_TRAIN = 5000
SEED = 42
# Identical to Cycle 07 (runner_think.INSTR + the parquet `message` field): single newline.
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DATA_SOURCE = "Math-CoT-20k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUN_ROOT / "data" / "opd_prompts_5k.parquet"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Reproduce the EXACT Cycle 07 seed=42 sample (df.sample(5000, random_state=42)).
    df = pd.read_parquet(SRC_PARQUET)
    samp = df.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    print(f"[data] source rows={len(df)} -> sampled {len(samp)} (seed={SEED})")

    # Sanity check: first questions must match probe_rows.jsonl (the Cycle 07 sample).
    if PROBE_ROWS.exists():
        probe0 = json.loads(PROBE_ROWS.open().readline())["question"].strip()
        samp0 = str(samp.iloc[0]["question"]).strip()
        match = probe0[:80] == samp0[:80]
        print(f"[check] sample[0] matches cycle07 probe_rows[0]: {match}")
        if not match:
            print(f"  probe0: {probe0[:80]!r}\n  samp0 : {samp0[:80]!r}")
            raise SystemExit("Seed=42 sample does NOT match Cycle 07 — refusing to write mismatched prompts.")

    records = []
    for i, row in samp.iterrows():
        question = str(row["question"])
        gold = str(row.get("answer", ""))  # gold (short) answer; OPD ignores it (use_task_rewards=False)
        records.append({
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": question + INSTR}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gold},
            "extra_info": {"index": int(i), "split": "train"},
        })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(out, index=False)
    meta = {
        "n_train": N_TRAIN, "seed": SEED, "data_source": DATA_SOURCE,
        "instr": INSTR, "src_parquet": str(SRC_PARQUET),
        "prompt_key": "prompt", "note": "OPD prompts only; same seed=42 sample as Cycle 07 SFT",
        "out_parquet": str(out),
    }
    (out.parent / "data_meta_opd.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[data] wrote {len(out_df)} prompt rows -> {out}")
    print(f"[data] schema: {list(out_df.columns)}")
    print(f"[data] sample prompt[0]: {out_df.iloc[0]['prompt']}")


if __name__ == "__main__":
    main()
