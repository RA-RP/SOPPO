#!/usr/bin/env python3
"""Cycle 09 block 2 G5 via the validated two-pass RAW top-32 pipeline."""

from pathlib import Path

import cycle09_offkd_rollout as rollout


REPO = Path("/root/LLM-output-density")
ROOT = Path("/root/autodl-tmp/cycle09_block2/model2_llama")
rollout.EXP_ROOT = ROOT
rollout.COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/model2_llama"
)
rollout.TEACHER = Path(
    "/root/autodl-tmp/model/Meta/modelscope/Meta-Llama-3.1-8B-Instruct"
)
rollout.MAX_MODEL_LEN = 12288
rollout.RUN_LABEL = "model2_llama"


if __name__ == "__main__":
    rollout.main()
