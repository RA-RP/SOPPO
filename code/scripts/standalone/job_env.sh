#!/bin/bash
# Fail-closed runtime shared by standalone stages.

soppo_job_init() {
    if [[ "${RUN_CONTEXT:-}" != "standalone" ]]; then
        echo "ERROR: RUN_CONTEXT=standalone is required" >&2
        return 1
    fi
    local script_dir
    script_dir="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    if [[ ! -f "$script_dir/server_paths.sh" || ! -f "$script_dir/runtime_env.sh" ]]; then
        echo "ERROR: Invalid SOPPO standalone script directory: $script_dir" >&2
        return 1
    fi
    source "$script_dir/server_paths.sh"
    source "$script_dir/runtime_env.sh"

    if [[ -n "${SOPPO_EXPECTED_GIT_COMMIT:-}" ]]; then
        if [[ ! "$SOPPO_EXPECTED_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
            echo "ERROR: Invalid SOPPO_EXPECTED_GIT_COMMIT" >&2
            return 1
        fi
        local actual_commit
        actual_commit="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
        if [[ "$actual_commit" != "$SOPPO_EXPECTED_GIT_COMMIT" ]]; then
            echo "ERROR: Server checkout changed after standalone pipeline start" >&2
            echo "  expected: $SOPPO_EXPECTED_GIT_COMMIT" >&2
            echo "  actual:   $actual_commit" >&2
            return 1
        fi
        if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
            echo "ERROR: Server checkout became dirty after standalone pipeline start" >&2
            return 1
        fi
    fi

    soppo_activate_env "$ENV_ROOT/youc"
    soppo_set_cache_env
    export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

soppo_hardware_gate() {
    local output_file="${1:?hardware output path required}"
    local required_count="${SOPPO_NPROC_PER_NODE:-2}"
    local minimum_mib="${SOPPO_MIN_GPU_MEMORY_MIB:-79000}"
    local required_name="${SOPPO_REQUIRED_GPU_NAME:-}"
    mkdir -p "$(dirname "$output_file")"
    python - "$output_file" "$required_count" "$minimum_mib" "$required_name" <<'PY'
import csv
import sys
from pathlib import Path

import torch

output, required_text, minimum_text, required_name = sys.argv[1:]
required = int(required_text)
minimum = int(minimum_text)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch")
count = torch.cuda.device_count()
if count != required:
    raise SystemExit(f"Expected exactly {required} visible GPUs, got {count}")
rows = []
for index in range(count):
    properties = torch.cuda.get_device_properties(index)
    total_mib = properties.total_memory // (1024 * 1024)
    if total_mib < minimum:
        raise SystemExit(
            f"GPU {index} has {total_mib} MiB; standalone contract requires >= {minimum} MiB"
        )
    if required_name and required_name.lower() not in properties.name.lower():
        raise SystemExit(
            f"GPU {index} is {properties.name!r}; required name contains {required_name!r}"
        )
    rows.append((index, properties.name, total_mib, torch.version.cuda))
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("logical_index", "name", "memory_total_mib", "torch_cuda"))
    writer.writerows(rows)
temporary.replace(path)
for row in rows:
    print(f"GPU {row[0]}: {row[1]}, {row[2]} MiB, torch CUDA {row[3]}")
PY
}

soppo_torchrun() {
    torchrun --standalone --nnodes=1 --nproc_per_node="${SOPPO_NPROC_PER_NODE:-2}" "$@"
}
