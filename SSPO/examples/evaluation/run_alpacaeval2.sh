#!/usr/bin/env bash
set -euo pipefail

MODEL_OUTPUTS=${1:?"usage: run_alpacaeval2.sh MODEL_OUTPUTS [OUTPUT_DIR]"}
OUTPUT_DIR=${2:-"$(dirname "${MODEL_OUTPUTS}")/alpacaeval2"}
ANNOTATORS_CONFIG=${ALPACAEVAL_ANNOTATORS_CONFIG:-weighted_alpaca_eval_gpt4_turbo}

if [[ ! -f "${MODEL_OUTPUTS}" ]]; then
  echo "model outputs not found: ${MODEL_OUTPUTS}" >&2
  exit 1
fi

if [[ "${ANNOTATORS_CONFIG}" == "weighted_alpaca_eval_gpt4_turbo" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required by ${ANNOTATORS_CONFIG}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
alpaca_eval \
  --model_outputs "${MODEL_OUTPUTS}" \
  --annotators_config "${ANNOTATORS_CONFIG}" \
  --fn_metric get_length_controlled_winrate \
  --sort_by length_controlled_winrate \
  --output_path "${OUTPUT_DIR}"
