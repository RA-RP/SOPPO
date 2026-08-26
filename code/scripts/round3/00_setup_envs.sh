#!/usr/bin/env bash
# Create isolated candidate train/rollout environments; run only after server authorization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
source "$SCRIPT_DIR/../standalone/runtime_env.sh"

CONDA_EXE="${SOPPO_CONDA_EXE:-}"
if [[ -z "$CONDA_EXE" ]]; then
    CONDA_EXE="$(soppo_find_conda || true)"
fi
[[ -n "$CONDA_EXE" && -x "$CONDA_EXE" ]] || {
    echo "ERROR: conda is required; set SOPPO_CONDA_EXE to an absolute path" >&2
    exit 1
}
mkdir -p "$ENV_ROOT" "$CACHE_ROOT/pip" "$CACHE_ROOT/conda/pkgs"
soppo_set_cache_env

create_env() {
    local destination="$1"
    local requirements="$2"
    if [[ ! -x "$destination/bin/python" ]]; then
        "$CONDA_EXE" create --prefix "$destination" python=3.11 pip -y
    fi
    "$CONDA_EXE" run --prefix "$destination" python -m pip install --upgrade pip
    "$CONDA_EXE" run --prefix "$destination" python -m pip install -r "$requirements"
    "$CONDA_EXE" run --prefix "$destination" python -m pip check
    "$CONDA_EXE" run --prefix "$destination" python -m pip freeze \
        > "$destination/environment.freeze.txt"
}

create_env "$ROUND3_TRAIN_ENV" "$CODE_ROOT/requirements-round3-train.in"
create_env "$ROUND3_ROLLOUT_ENV" "$CODE_ROOT/requirements-round3-rollout.in"
echo "Round3 candidate environments created outside Git."

