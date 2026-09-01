#!/usr/bin/env bash
# Server task: build a CPython 3.12 / Linux x86_64 offline wheelhouse on the connected staging server.
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
[[ "$REPO_ROOT" == /* ]] || fail "ROUND4_REPO_ROOT must be absolute"
[[ "$BUILDER_PYTHON" == /* && -x "$BUILDER_PYTHON" ]] || fail "builder Python is not executable"
[[ -d "$REPO_ROOT/.git" ]] || fail "ROUND4_REPO_ROOT is not the SOPPO Git checkout"

CANON_SERVER_BASE="$(realpath -e "$SERVER_BASE")"
CANON_REPO_ROOT="$(realpath -e "$REPO_ROOT")"
[[ "$CANON_REPO_ROOT" == "$REPO_ROOT" ]] || fail "ROUND4_REPO_ROOT must already be canonical"

PROJECT_ROOT="$REPO_ROOT/SSPO"
REQUIREMENTS_SOURCE="$PROJECT_ROOT/requirements.txt"
[[ -f "$REQUIREMENTS_SOURCE" && -f "$PROJECT_ROOT/pyproject.toml" ]] || fail "SSPO package files are missing"

PYTHON_MINOR="$($BUILDER_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_MINOR" == "3.12" ]] || fail "Round4 wheel builder requires Python 3.12, got $PYTHON_MINOR"

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || fail "checkout must be clean before packaging"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel)" == "$REPO_ROOT" ]] || fail "unexpected Git root"

PACKAGE_ROOT="${ROUND4_PACKAGE_ROOT:-$SERVER_BASE/exports/round4-offline-package/$COMMIT}"
CANON_PACKAGE_ROOT="$(realpath -m "$PACKAGE_ROOT")"
case "$CANON_PACKAGE_ROOT" in
    "$CANON_SERVER_BASE"/*) ;;
    *) fail "ROUND4_PACKAGE_ROOT must stay below ROUND4_SERVER_BASE" ;;
esac
case "$CANON_PACKAGE_ROOT" in
    "$CANON_REPO_ROOT"|"$CANON_REPO_ROOT"/*) fail "offline package must stay outside the Git checkout" ;;
esac
[[ ! -L "$PACKAGE_ROOT" ]] || fail "ROUND4_PACKAGE_ROOT must not be a symlink"
PACKAGE_ROOT="$CANON_PACKAGE_ROOT"

WHEELHOUSE="$PACKAGE_ROOT/wheelhouse"
PIP_CACHE_ROOT="${ROUND4_PIP_CACHE:-$SERVER_BASE/cache/pip-round4-py312}"
PIP_CACHE_ROOT="$(realpath -m "$PIP_CACHE_ROOT")"
case "$PIP_CACHE_ROOT" in
    "$CANON_SERVER_BASE"/*) ;;
    *) fail "ROUND4_PIP_CACHE must stay below ROUND4_SERVER_BASE" ;;
esac
mkdir -p "$WHEELHOUSE" "$PIP_CACHE_ROOT"
export PIP_CACHE_DIR="$PIP_CACHE_ROOT"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

RUNTIME_REQUIREMENTS="$PACKAGE_ROOT/requirements.runtime.txt"
CONSTRAINTS="$PACKAGE_ROOT/constraints.round4.txt"

# The editable local package is built as its own wheel below. All runtime pins remain unchanged.
awk '!/^[[:space:]]*-e[[:space:]]+\.[[:space:]]*($|#)/' "$REQUIREMENTS_SOURCE" > "$RUNTIME_REQUIREMENTS"
printf '%s\n' 'torch==2.5.1+cu124' > "$CONSTRAINTS"

# Download the CUDA build and its NVIDIA runtime dependencies only from PyTorch's official cu124 index.
"$BUILDER_PYTHON" -m pip download \
    --dest "$WHEELHOUSE" \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.5.1'

compgen -G "$WHEELHOUSE/torch-2.5.1+cu124-cp312-cp312-linux_x86_64.whl" >/dev/null \
    || fail "the expected CPython 3.12 CUDA 12.4 torch wheel was not downloaded"

# Build source-only dependencies such as fire and jieba now; the A100 installation stays compiler-free and offline.
"$BUILDER_PYTHON" -m pip wheel \
    --wheel-dir "$WHEELHOUSE" \
    --find-links "$WHEELHOUSE" \
    --constraint "$CONSTRAINTS" \
    --requirement "$RUNTIME_REQUIREMENTS"

"$BUILDER_PYTHON" -m pip wheel \
    --wheel-dir "$WHEELHOUSE" \
    --no-deps \
    "$PROJECT_ROOT"

compgen -G "$WHEELHOUSE/fire-0.7.0-*.whl" >/dev/null || fail "fire wheel is missing"
compgen -G "$WHEELHOUSE/jieba-0.42.1-*.whl" >/dev/null || fail "jieba wheel is missing"
compgen -G "$WHEELHOUSE/llamafactory-0.9.2.dev0-*.whl" >/dev/null || fail "local llamafactory wheel is missing"

BUILD_INFO_TMP="$PACKAGE_ROOT/BUILD_INFO.txt.tmp"
{
    printf 'schema=round4-offline-wheelhouse-v1\n'
    printf 'commit=%s\n' "$COMMIT"
    printf 'builder_python=%s\n' "$($BUILDER_PYTHON --version 2>&1)"
    printf 'builder_pip=%s\n' "$($BUILDER_PYTHON -m pip --version)"
    printf 'target_python=CPython 3.12\n'
    printf 'target_platform=Linux x86_64\n'
    printf 'torch=2.5.1+cu124\n'
    printf 'source_requirements_sha256=%s\n' "$(sha256sum "$REQUIREMENTS_SOURCE" | awk '{print $1}')"
} > "$BUILD_INFO_TMP"
mv "$BUILD_INFO_TMP" "$PACKAGE_ROOT/BUILD_INFO.txt"

MANIFEST_TMP="$PACKAGE_ROOT/SHA256SUMS.tmp"
(
    cd "$PACKAGE_ROOT"
    find wheelhouse -maxdepth 1 -type f -name '*.whl' -print0 \
        | sort -z \
        | xargs -0 sha256sum
    sha256sum BUILD_INFO.txt constraints.round4.txt requirements.runtime.txt
) > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$PACKAGE_ROOT/SHA256SUMS"

(
    cd "$PACKAGE_ROOT"
    sha256sum --check SHA256SUMS
)

printf 'Offline package ready: %s\n' "$PACKAGE_ROOT"
printf 'Git commit: %s\n' "$COMMIT"
printf 'Wheel count: %s\n' "$(find "$WHEELHOUSE" -maxdepth 1 -type f -name '*.whl' | wc -l)"
du -sh "$PACKAGE_ROOT"
