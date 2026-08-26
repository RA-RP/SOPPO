#!/usr/bin/env bash
# Create isolated candidate train/rollout environments; run only after server authorization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
source "$SCRIPT_DIR/../standalone/runtime_env.sh"

CONDA_EXE="$(soppo_find_conda || true)"
PYTHON_BIN="${SOPPO_PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3.10 || command -v python3 || true)"
fi
if [[ -z "$CONDA_EXE" ]]; then
    [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
        echo "ERROR: set SOPPO_PYTHON_BIN to an executable Python 3.10 when conda is unavailable" >&2
        exit 1
    }
    [[ "$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]] || {
        echo "ERROR: Round3 venv fallback requires Python 3.10" >&2
        exit 1
    }
fi
mkdir -p "$ENV_ROOT" "$CACHE_ROOT/pip" "$CACHE_ROOT/conda/pkgs"
soppo_set_cache_env

create_env() {
    local destination="$1"
    local requirements="$2"
    if [[ ! -x "$destination/bin/python" ]]; then
        if [[ -n "$CONDA_EXE" ]]; then
            "$CONDA_EXE" create --prefix "$destination" python=3.10 pip -y
        else
            "$PYTHON_BIN" -m venv "$destination"
        fi
    fi
    local environment_python="$destination/bin/python"
    [[ "$("$environment_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]] || {
        echo "ERROR: existing Round3 environment is not Python 3.10: $destination" >&2
        return 1
    }
    "$environment_python" -m pip install --upgrade pip
    "$environment_python" -m pip install -r "$requirements"
    "$environment_python" -m pip check
    "$environment_python" -m pip freeze \
        > "$destination/environment.freeze.txt"
}

create_env "$ROUND3_TRAIN_ENV" "$CODE_ROOT/requirements-round3-train.in"
create_env "$ROUND3_ROLLOUT_ENV" "$CODE_ROOT/requirements-round3-rollout.in"
echo "Round3 candidate environments created outside Git."
