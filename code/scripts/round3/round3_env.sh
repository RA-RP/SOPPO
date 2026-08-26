#!/usr/bin/env bash
# Shared path contract for the authorized standalone 3x4090 Round3 server.
set -euo pipefail

ROUND3_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$ROUND3_SCRIPT_DIR/../.." && pwd)"
SOPPO_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
SERVER_BASE="$(cd "$SOPPO_ROOT/.." && pwd)"

ENV_ROOT="${SOPPO_ENV_ROOT:-$SERVER_BASE/envs}"
CACHE_ROOT="${SOPPO_CACHE_ROOT:-$SERVER_BASE/cache}"
DATA_ROOT="${SOPPO_DATA_ROOT:-$SERVER_BASE/data}"
MODEL_ROOT="${SOPPO_MODEL_ROOT:-$SERVER_BASE/models}"
RUN_ROOT="${SOPPO_RUN_ROOT:-$SERVER_BASE/runs}"
EXPORT_ROOT="${SOPPO_EXPORT_ROOT:-$SERVER_BASE/exports}"
PLATFORM_LOG_ROOT="${SOPPO_PLATFORM_LOG_ROOT:-$SERVER_BASE/platform_logs}"

round3_require_external_absolute_path() {
    local path="$1"
    local name="$2"
    [[ "$path" == /* ]] || { echo "ERROR: $name must be an absolute server path" >&2; return 1; }
    local resolved
    resolved="$(realpath -m -- "$path")"
    [[ "$resolved" != "/" ]] || { echo "ERROR: $name must not target filesystem root" >&2; return 1; }
    case "$resolved" in
        "$SOPPO_ROOT"|"$SOPPO_ROOT"/*)
            echo "ERROR: $name must remain outside the SOPPO Git checkout" >&2
            return 1
            ;;
    esac
}

for round3_path_contract in \
    "ENV_ROOT:$ENV_ROOT" \
    "CACHE_ROOT:$CACHE_ROOT" \
    "DATA_ROOT:$DATA_ROOT" \
    "MODEL_ROOT:$MODEL_ROOT" \
    "RUN_ROOT:$RUN_ROOT" \
    "EXPORT_ROOT:$EXPORT_ROOT" \
    "PLATFORM_LOG_ROOT:$PLATFORM_LOG_ROOT"; do
    round3_require_external_absolute_path \
        "${round3_path_contract#*:}" "${round3_path_contract%%:*}"
done
unset round3_path_contract

export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
unset TRANSFORMERS_CACHE

ROUND3_TRAIN_ENV="${SOPPO_ROUND3_TRAIN_ENV:-$ENV_ROOT/round3-train}"
ROUND3_ROLLOUT_ENV="${SOPPO_ROUND3_ROLLOUT_ENV:-$ENV_ROOT/round3-rollout}"
ROUND3_TRAIN_PYTHON="$ROUND3_TRAIN_ENV/bin/python"
ROUND3_ROLLOUT_PYTHON="$ROUND3_ROLLOUT_ENV/bin/python"
ROUND3_CONFIG_DIR="$CODE_ROOT/configs/round3"
ROUND3_MODEL_DIR="${SOPPO_ROUND3_MODEL_DIR:-$MODEL_ROOT/Qwen3-1.7B}"
ROUND3_DATA_DIR="${SOPPO_ROUND3_DATA_DIR:-$DATA_ROOT/round3/dual_source_v1}"
ROUND3_REFERENCE_DIR="${SOPPO_ROUND3_REFERENCE_DIR:-$CACHE_ROOT/round3/reference_qwen3_1.7b_dual_source_v1}"

for round3_path_contract in \
    "ROUND3_TRAIN_ENV:$ROUND3_TRAIN_ENV" \
    "ROUND3_ROLLOUT_ENV:$ROUND3_ROLLOUT_ENV" \
    "ROUND3_MODEL_DIR:$ROUND3_MODEL_DIR" \
    "ROUND3_DATA_DIR:$ROUND3_DATA_DIR" \
    "ROUND3_REFERENCE_DIR:$ROUND3_REFERENCE_DIR"; do
    round3_require_external_absolute_path \
        "${round3_path_contract#*:}" "${round3_path_contract%%:*}"
done
unset round3_path_contract

round3_require_experiment_id() {
    if [[ ! "${SOPPO_ROUND3_EXPERIMENT_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "ERROR: set an explicit path-safe SOPPO_ROUND3_EXPERIMENT_ID; it must not be guessed" >&2
        return 1
    fi
    ROUND3_RUN_ROOT="$RUN_ROOT/$SOPPO_ROUND3_EXPERIMENT_ID"
    ROUND3_LOG_ROOT="$PLATFORM_LOG_ROOT/$SOPPO_ROUND3_EXPERIMENT_ID"
    export ROUND3_RUN_ROOT ROUND3_LOG_ROOT
}

round3_require_full_sha() {
    local value="$1"
    local name="$2"
    if [[ ! "$value" =~ ^[0-9a-f]{40}$ ]]; then
        echo "ERROR: $name must be an explicitly resolved lowercase 40-character SHA" >&2
        return 1
    fi
}

round3_resolved_source_sha() {
    local key="$1"
    local evidence="$ROUND3_RUN_ROOT/source_revisions.json"
    [[ -f "$evidence" ]] || {
        echo "ERROR: resolve immutable source revisions first: $evidence" >&2
        return 1
    }
    "$ROUND3_TRAIN_PYTHON" - "$evidence" "$key" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["sources"][sys.argv[2]]["resolved_sha"]
if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
    raise SystemExit("malformed resolved source SHA evidence")
print(value)
PY
}

round3_resolved_value() {
    local config_path="$1"
    local dotted_key="$2"
    "$ROUND3_TRAIN_PYTHON" - "$config_path" "$dotted_key" <<'PY'
import sys
import yaml

value = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for piece in sys.argv[2].split("."):
    value = value[piece]
if isinstance(value, bool):
    print(str(value).lower())
elif isinstance(value, list):
    print(",".join(str(item) for item in value))
else:
    print(value)
PY
}
