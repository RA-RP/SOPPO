#!/bin/bash
# Slurm array worker: 0=base, 1..8=one final trajectory.
set -euo pipefail
SCRIPT_DIR="${SOPPO_CLUSTER_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$INDEX" =~ ^[0-8]$ ]] || { echo "ERROR: C_epsilon array index must be 0..8" >&2; exit 2; }
MODEL_DIR="${SOPPO_MODEL_DIR:-$MODEL_ROOT/Qwen3-4B}"
MODEL_MANIFEST="$MODEL_DIR/model_manifest.json"
ROOT="$RUN_ROOT/$EXPERIMENT_ID/c_epsilon"
INPUTS="$ROOT/inputs"
GETSLICE="$OBSERVE_ROOT/GetSlice/slice.py"
soppo_hardware_gate "$RUN_ROOT/$EXPERIMENT_ID/pipeline/hardware/c_epsilon_${INDEX}.csv"
python -m src.model.model_manifest --model-dir "$MODEL_DIR" --verify

if [[ "$INDEX" == "0" ]]; then
    CELL="$ROOT/raw/base"
    CONFIG="$ROOT/configs/base.json"
    python -m src.evaluation.c_epsilon config \
        --model "$MODEL_DIR" --inputs "$INPUTS" --output "$CELL" --config "$CONFIG"
    python "$GETSLICE" --config "$CONFIG"
    test -s "$CELL/sMat_S.json" && test -s "$CELL/xMat_X_S.json"
    exit 0
fi

METHODS=(
    dpo10
    dpo100
    sspo_hard_exp
    soppo_pe_exp
    soppo_pe_static_lambda_0.1
    soppo_pe_static_lambda_0.3
    soppo_pe_static_lambda_0.5
    soppo_pe_static_lambda_1.0
)
METHOD="${METHODS[$((INDEX - 1))]}"
RUN_DIR="$RUN_ROOT/$EXPERIMENT_ID/main/$METHOD"
METHOD_ROOT="$ROOT/raw/methods/$METHOD"
if [[ -e "$METHOD_ROOT" ]]; then
    echo "ERROR: Refuse to overwrite C_epsilon method output: $METHOD_ROOT" >&2
    exit 1
fi
mkdir -p "$METHOD_ROOT"

mapfile -t CHECKPOINTS < <(python - "$RUN_DIR/checkpoints" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
values = sorted(
    (path for path in root.glob("step_*") if path.is_dir()),
    key=lambda path: int(re.search(r"(\d+)$", path.name).group(1)),
)
if len(values) < 3:
    raise SystemExit("C_epsilon requires at least three checkpoints")
if len(values) <= 10:
    selected = values
else:
    tail = values[3:]
    indexes = sorted({round(i * (len(tail) - 1) / 6) for i in range(7)})
    selected = values[:3] + [tail[index] for index in indexes]
for path in selected[:10]:
    print(path)
PY
)

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    STEP="$(basename "$CHECKPOINT")"
    CELL="$METHOD_ROOT/$STEP"
    CONFIG="$ROOT/configs/${METHOD}_${STEP}.json"
    test -f "$CHECKPOINT/adapter_config.json"
    test -f "$CHECKPOINT/checkpoint_meta.json"
    python - "$CHECKPOINT" "$MODEL_DIR" "$MODEL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, expected_base, expected_manifest = map(Path, sys.argv[1:])
adapter = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
metadata = json.loads((checkpoint / "checkpoint_meta.json").read_text(encoding="utf-8"))
actual = {
    Path(adapter["base_model_name_or_path"]).resolve(),
    Path(metadata["base_model"]).resolve(),
}
if actual != {expected_base.resolve()}:
    raise SystemExit(f"Adapter/base mismatch for {checkpoint}: {sorted(map(str, actual))}")
if Path(metadata["model_manifest"]).resolve() != expected_manifest.resolve():
    raise SystemExit(f"Adapter/manifest mismatch for {checkpoint}")
PY
    python -m src.evaluation.c_epsilon config \
        --model "$CHECKPOINT" --inputs "$INPUTS" --output "$CELL" --config "$CONFIG"
    python "$GETSLICE" --config "$CONFIG"
    test -s "$CELL/sMat_S.json" && test -s "$CELL/xMat_X_S.json"
done
touch "$METHOD_ROOT/RAW_COMPLETE"
