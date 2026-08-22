#!/bin/bash
# Cancel only live Slurm jobs named in one SOPPO task registry.
set -Eeuo pipefail

if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: RUN_CONTEXT=cluster is required" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/job_env.sh"
soppo_job_init

REGISTRY="$RUN_ROOT/$EXPERIMENT_ID/pipeline/task_registry.json"
EXECUTE=0
while (( $# > 0 )); do
    case "$1" in
        --registry)
            REGISTRY="${2:?--registry requires a path}"
            shift 2
            ;;
        --execute)
            EXECUTE=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

test -f "$REGISTRY" || { echo "ERROR: registry not found: $REGISTRY" >&2; exit 1; }
command -v squeue >/dev/null || { echo "ERROR: squeue is unavailable" >&2; exit 1; }
command -v scancel >/dev/null || { echo "ERROR: scancel is unavailable" >&2; exit 1; }
mapfile -t RECORDS < <(python - "$REGISTRY" "$EXPERIMENT_ID" <<'PY'
import json
import re
import sys
from pathlib import Path

path, expected_experiment = sys.argv[1:]
registry = json.loads(Path(path).read_text(encoding="utf-8"))
if registry.get("experiment_id") != expected_experiment:
    raise SystemExit(
        f"registry experiment mismatch: {registry.get('experiment_id')} != {expected_experiment}"
    )
jobs = registry.get("jobs")
if not isinstance(jobs, dict) or not jobs:
    raise SystemExit("registry contains no jobs")
seen = set()
for stage, raw_job_id in jobs.items():
    job_id = str(raw_job_id)
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise SystemExit(f"invalid job id for {stage}: {job_id}")
    if job_id in seen:
        raise SystemExit(f"duplicate job id in registry: {job_id}")
    seen.add(job_id)
    print(f"{stage}\t{job_id}")
PY
)
(( ${#RECORDS[@]} > 0 )) || { echo "ERROR: registry yielded no jobs" >&2; exit 1; }

declare -a ACTIVE_IDS=()
declare -a ACTIVE_RECORDS=()
echo "Registry-scoped job preview:"
for record in "${RECORDS[@]}"; do
    stage="${record%%$'\t'*}"
    job_id="${record##*$'\t'}"
    queue_line="$(squeue -h -j "$job_id" -o '%A|%T|%j|%R' 2>/dev/null || true)"
    if [[ -n "$queue_line" ]]; then
        ACTIVE_IDS+=("$job_id")
        ACTIVE_RECORDS+=("$record")
        printf '  %-24s %s\n' "$stage" "$queue_line"
    else
        printf '  %-24s %s\n' "$stage" "$job_id|not live"
    fi
done

if (( ${#ACTIVE_IDS[@]} == 0 )); then
    echo "No live jobs from this registry; nothing to cancel."
    exit 0
fi
if (( EXECUTE == 0 )); then
    echo "Dry run only. Re-run with --execute to cancel exactly the live jobs above."
    exit 0
fi

CONFIRMATION="cancel-$EXPERIMENT_ID"
if [[ -t 0 ]]; then
    read -r -p "Type '$CONFIRMATION' to continue: " answer
else
    answer="${SOPPO_CONFIRM_CANCEL:-}"
fi
if [[ "$answer" != "$CONFIRMATION" ]]; then
    echo "ERROR: cancellation confirmation did not match" >&2
    exit 1
fi

for job_id in "${ACTIVE_IDS[@]}"; do
    scancel "$job_id"
done
echo "Cancelled ${#ACTIVE_IDS[@]} live jobs from registry: $REGISTRY"
