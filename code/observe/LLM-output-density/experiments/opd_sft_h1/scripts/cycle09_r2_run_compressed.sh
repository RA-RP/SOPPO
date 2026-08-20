#!/usr/bin/env bash
set -euo pipefail

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export TMPDIR=/root/autodl-tmp/pip-tmp

PY=${PY:-/root/miniconda3/envs/density/bin/python}
MODEL_OUTER=${MODEL_OUTER:-/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_r2_model_outer.py}
UNIFIED=${UNIFIED:-/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_r2_unified_probe.py}
RAW_ER=${RAW_ER:-/root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_r2_raw_er.py}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/cycle09_r2}
MINI_ROOT=${MINI_ROOT:-/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini}

STEPS=${STEPS:-0,5,10,20,40,160,624}
LAYERS=${LAYERS:-9,18,27}
PROBES=${PROBES:-X_math,X_ood_knowledge,X_general,X_math_hard,X_bos}
RAW_PROBES=${RAW_PROBES:-X_math,X_ood_knowledge}
N_SAMPLES=${N_SAMPLES:-32}
RAW_N_SAMPLES=${RAW_N_SAMPLES:-32}
SEQLEN=${SEQLEN:-512}
UV_RANK=${UV_RANK:-128}

MODEL_OUTER_ARGS=(
  --run-root "$RUN_ROOT"
  --mini-root "$MINI_ROOT"
  --arms opd,sft
  --steps "$STEPS"
  --layers "$LAYERS"
  --probes "$PROBES"
  --raw-probes "$RAW_PROBES"
  --n-samples "$N_SAMPLES"
  --raw-n-samples "$RAW_N_SAMPLES"
  --seqlen "$SEQLEN"
  --uv-rank "$UV_RANK"
  --check-inputs
)

SUMMARY_ARGS=(
  --run-root "$RUN_ROOT"
  --mini-root "$MINI_ROOT"
  --arms opd,sft
  --steps "$STEPS"
  --uv-layers "$LAYERS"
  --probes "$PROBES"
  --n-samples "$N_SAMPLES"
  --seqlen "$SEQLEN"
  --uv-rank "$UV_RANK"
)

RAW_SUMMARY_ARGS=(
  --run-root "$RUN_ROOT"
  --mini-root "$MINI_ROOT"
  --arms opd,sft
  --steps "$STEPS"
  --layers "$LAYERS"
  --probes "$RAW_PROBES"
)

stamp() {
  date '+[%F %T]'
}

phase() {
  echo
  echo "$(stamp) $*"
}

phase "Round2 compressed model-outer run"
"$PY" "$MODEL_OUTER" "${MODEL_OUTER_ARGS[@]}"

phase "Summarize T5/T6/T7"
"$PY" "$UNIFIED" --summarize --theta-ranks 8,16,32,64,128 "${SUMMARY_ARGS[@]}"

phase "Summarize T8 dual ER"
"$PY" "$RAW_ER" --summarize "${RAW_SUMMARY_ARGS[@]}"

phase "Final status"
"$PY" "$UNIFIED" --status "${SUMMARY_ARGS[@]}"
"$PY" "$RAW_ER" --status "${RAW_SUMMARY_ARGS[@]}"

phase "Round2 compressed model-outer complete"
