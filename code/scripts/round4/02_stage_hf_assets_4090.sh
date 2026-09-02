#!/usr/bin/env bash
# Server task: stage immutable Round4 model/dataset snapshots on the networked 4090 host.
set -euo pipefail
umask 077

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

SERVER_BASE="${ROUND4_SERVER_BASE:?set ROUND4_SERVER_BASE to the external server base directory}"
REPO_ROOT="${ROUND4_REPO_ROOT:?set ROUND4_REPO_ROOT to the clean SOPPO checkout}"
BUILDER_PYTHON="${ROUND4_BUILDER_PYTHON:?set ROUND4_BUILDER_PYTHON to the Python 3.12 builder executable}"

[[ "$SERVER_BASE" == /* ]] || fail "ROUND4_SERVER_BASE must be absolute"
[[ "$REPO_ROOT" == /* && -d "$REPO_ROOT/.git" ]] || fail "ROUND4_REPO_ROOT is not the SOPPO Git checkout"
[[ "$BUILDER_PYTHON" == /* && -x "$BUILDER_PYTHON" ]] || fail "builder Python is not executable"

CANON_SERVER_BASE="$(realpath -e "$SERVER_BASE")"
CANON_REPO_ROOT="$(realpath -e "$REPO_ROOT")"
[[ "$CANON_REPO_ROOT" == "$REPO_ROOT" ]] || fail "ROUND4_REPO_ROOT must already be canonical"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || fail "checkout must be clean before asset staging"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

PACKAGE_ROOT="${ROUND4_PACKAGE_ROOT:-$SERVER_BASE/exports/round4-offline-package/$COMMIT}"
PACKAGE_ROOT="$(realpath -e "$PACKAGE_ROOT")"
case "$PACKAGE_ROOT" in
    "$CANON_SERVER_BASE"/*) ;;
    *) fail "ROUND4_PACKAGE_ROOT must stay below ROUND4_SERVER_BASE" ;;
esac
[[ -f "$PACKAGE_ROOT/SHA256SUMS" && -d "$PACKAGE_ROOT/wheelhouse" ]] \
    || fail "verified Round4 wheelhouse package is missing"
(
    cd "$PACKAGE_ROOT"
    sha256sum --check SHA256SUMS
)

STAGER_ENV="${ROUND4_ASSET_STAGER_ENV:-$SERVER_BASE/envs/round4-asset-stager-py312}"
STAGER_ENV="$(realpath -m "$STAGER_ENV")"
case "$STAGER_ENV" in
    "$CANON_SERVER_BASE"/*) ;;
    *) fail "ROUND4_ASSET_STAGER_ENV must stay below ROUND4_SERVER_BASE" ;;
esac
if [[ ! -e "$STAGER_ENV" ]]; then
    "$BUILDER_PYTHON" -m venv "$STAGER_ENV"
elif [[ ! -x "$STAGER_ENV/bin/python" ]]; then
    fail "existing asset stager path is not a valid venv"
fi
[[ "$($STAGER_ENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.12" ]] \
    || fail "asset stager must use Python 3.12"

"$STAGER_ENV/bin/python" -m pip install \
    --no-index \
    --find-links "$PACKAGE_ROOT/wheelhouse" \
    'huggingface-hub==0.30.2'
"$STAGER_ENV/bin/python" -m pip check

AVAILABLE_KIB="$(df -Pk "$SERVER_BASE" | awk 'NR == 2 {print $4}')"
MIN_FREE_GIB="${ROUND4_MIN_FREE_GIB:-20}"
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || fail "ROUND4_MIN_FREE_GIB must be an integer"
(( AVAILABLE_KIB >= MIN_FREE_GIB * 1024 * 1024 )) \
    || fail "less than ${MIN_FREE_GIB} GiB is available for Round4 assets"

DATA_ROOT="${ROUND4_RAW_DATA_ROOT:-$SERVER_BASE/data/round4-v1/raw}"
MODEL_TARGET="${ROUND4_MODEL_TARGET:-$SERVER_BASE/models/Qwen3-1.7B-b9352fbb}"
HF_CACHE="${ROUND4_HF_CACHE:-$SERVER_BASE/cache/huggingface-round4}"
INDEX_OUTPUT="${ROUND4_ASSET_INDEX:-$SERVER_BASE/exports/round4-assets/$COMMIT/ROUND4_ASSET_INDEX.json}"

"$STAGER_ENV/bin/python" "$REPO_ROOT/code/scripts/round4/02_stage_hf_assets.py" \
    --server-base "$SERVER_BASE" \
    --data-root "$DATA_ROOT" \
    --model-target "$MODEL_TARGET" \
    --cache-dir "$HF_CACHE" \
    --index-output "$INDEX_OUTPUT" \
    --model-revision 'b9352fbb8ce704292730cf54b3b1dceb2a808738' \
    --ultrafeedback-revision "${ROUND4_ULTRAFEEDBACK_REVISION:-3949bf5f8c17c394422ccfab0c31ea9c20bdeb85}" \
    --ultrachat-revision "${ROUND4_ULTRACHAT_REVISION:-8049631c405ae6576f93f445c6b8166f76f5505a}" \
    --alpaca-eval-revision "${ROUND4_ALPACA_EVAL_REVISION:-2edc6fad8be6b14ea7230aabfd08188da6b8b814}" \
    --max-workers "${ROUND4_HF_MAX_WORKERS:-8}"

sha256sum "$INDEX_OUTPUT" > "$INDEX_OUTPUT.sha256"
sha256sum --check "$INDEX_OUTPUT.sha256"
printf 'Round4 assets staged for commit: %s\n' "$COMMIT"
du -sh "$DATA_ROOT" "$MODEL_TARGET" "$STAGER_ENV"
