#!/usr/bin/env bash
# Server task: create a fresh Round4 venv inside the existing FusionOne py312/cu124 image.
set -euo pipefail
umask 077

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

PACKAGE_ROOT="${ROUND4_PACKAGE_ROOT:?set ROUND4_PACKAGE_ROOT to the transferred offline package}"
ENV_ROOT="${ROUND4_ENV_ROOT:?set ROUND4_ENV_ROOT to a new external A100 environment directory}"
A100_BASE="${ROUND4_A100_BASE:?set ROUND4_A100_BASE to the external A100 base directory}"
PYTHON_BIN="${ROUND4_PYTHON_BIN:-$(command -v python3.12 || true)}"
EXPECTED_GPUS="${ROUND4_EXPECTED_GPUS:-2}"
EXPECTED_COMMIT="${ROUND4_EXPECTED_COMMIT:?set ROUND4_EXPECTED_COMMIT to the full approved Git SHA}"

[[ "$PACKAGE_ROOT" == /* && -d "$PACKAGE_ROOT/wheelhouse" ]] || fail "offline package is missing"
[[ "$ENV_ROOT" == /* ]] || fail "ROUND4_ENV_ROOT must be absolute"
[[ "$A100_BASE" == /* && -d "$A100_BASE" ]] || fail "ROUND4_A100_BASE must be an existing absolute directory"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || fail "python3.12 is unavailable in the image"
[[ "$EXPECTED_GPUS" =~ ^[1-9][0-9]*$ ]] || fail "ROUND4_EXPECTED_GPUS must be positive"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "ROUND4_EXPECTED_COMMIT must be a full lowercase Git SHA"
[[ ! -e "$ENV_ROOT" ]] || fail "refusing to overwrite existing environment: $ENV_ROOT"

CANON_A100_BASE="$(realpath -e "$A100_BASE")"
PACKAGE_ROOT="$(realpath -e "$PACKAGE_ROOT")"
CANON_ENV_ROOT="$(realpath -m "$ENV_ROOT")"
case "$PACKAGE_ROOT" in
    "$CANON_A100_BASE"/*) ;;
    *) fail "ROUND4_PACKAGE_ROOT must stay below ROUND4_A100_BASE" ;;
esac
case "$CANON_ENV_ROOT" in
    "$CANON_A100_BASE"/*) ;;
    *) fail "ROUND4_ENV_ROOT must stay below ROUND4_A100_BASE" ;;
esac
ENV_ROOT="$CANON_ENV_ROOT"

PYTHON_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_MINOR" == "3.12" ]] || fail "Round4 requires Python 3.12, got $PYTHON_MINOR"

(
    cd "$PACKAGE_ROOT"
    sha256sum --check SHA256SUMS
    diff -u \
        <(awk '$2 ~ /^wheelhouse\// {print $2}' SHA256SUMS | sort) \
        <(find wheelhouse -maxdepth 1 -type f -name '*.whl' -print | sort)
)
grep -Fx "commit=$EXPECTED_COMMIT" "$PACKAGE_ROOT/BUILD_INFO.txt" >/dev/null \
    || fail "offline package BUILD_INFO is not bound to ROUND4_EXPECTED_COMMIT"

mkdir -p "$(dirname "$ENV_ROOT")"
"$PYTHON_BIN" -m venv "$ENV_ROOT"
ENV_PYTHON="$ENV_ROOT/bin/python"
[[ -x "$ENV_PYTHON" ]] || fail "venv creation failed"

export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$ENV_PYTHON" -m pip install \
    --find-links "$PACKAGE_ROOT/wheelhouse" \
    --constraint "$PACKAGE_ROOT/constraints.round4.txt" \
    --requirement "$PACKAGE_ROOT/requirements.runtime.txt"

mapfile -t PROJECT_WHEELS < <(find "$PACKAGE_ROOT/wheelhouse" -maxdepth 1 -type f -name 'llamafactory-0.9.2.dev0-*.whl' | sort)
[[ "${#PROJECT_WHEELS[@]}" -eq 1 ]] || fail "expected exactly one llamafactory project wheel"
"$ENV_PYTHON" -m pip install --no-deps "${PROJECT_WHEELS[0]}"
"$ENV_PYTHON" -m pip check

ROUND4_EXPECTED_GPUS="$EXPECTED_GPUS" "$ENV_PYTHON" - <<'PY'
import importlib
import importlib.metadata
import os
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"unexpected Python: {sys.version}")

for module in [
    "accelerate",
    "alpaca_eval",
    "bitsandbytes",
    "datasets",
    "llamafactory",
    "peft",
    "tokenizers",
    "transformers",
    "trl",
]:
    importlib.import_module(module)

import torch

expected_gpus = int(os.environ["ROUND4_EXPECTED_GPUS"])
if importlib.metadata.version("setuptools") != "78.1.0":
    raise SystemExit(
        "unexpected setuptools: " + importlib.metadata.version("setuptools")
    )
if not torch.__version__.startswith("2.5.1+cu124"):
    raise SystemExit(f"unexpected torch build: {torch.__version__}")
if torch.version.cuda != "12.4":
    raise SystemExit(f"unexpected torch CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected_gpus:
    raise SystemExit(f"expected {expected_gpus} visible GPUs, got {torch.cuda.device_count()}")

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()}")
for distribution in [
    "transformers",
    "datasets",
    "peft",
    "trl",
    "alpaca-eval",
    "bitsandbytes",
    "setuptools",
]:
    print(f"{distribution}={importlib.metadata.version(distribution)}")
PY

"$ENV_PYTHON" -m pip freeze > "$ENV_ROOT/environment.freeze.txt"
printf '%s\n' "$EXPECTED_COMMIT" > "$ENV_ROOT/ROUND4_CODE_COMMIT"
sha256sum "$PACKAGE_ROOT/SHA256SUMS" > "$ENV_ROOT/offline-package-manifest.sha256"
printf 'Round4 A100 environment ready: %s\n' "$ENV_ROOT"
