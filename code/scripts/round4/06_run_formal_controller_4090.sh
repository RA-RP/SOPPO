#!/usr/bin/env bash
# Server task: recoverably coordinate immutable A100 formal outputs with 4090 API judging.
set -euo pipefail
umask 077

RUN_ID="${1:?usage: 06_run_formal_controller_4090.sh RUN_ID}"
BASE="${ROUND4_4090_BASE:-/data/youchang/youchang}"
REPO="${ROUND4_4090_REPO:-$BASE/SOPPO}"
PYTHON="${ROUND4_JUDGE_PYTHON:?set ROUND4_JUDGE_PYTHON to the 4090 judge Python}"
JUDGE="$REPO/code/scripts/round4/04_run_api_judge_4090.sh"
A100_CONFIG="${ROUND4_A100_SSH_CONFIG:-$HOME/.ssh/round4_a100_2_config}"
A100_HOST="${ROUND4_A100_SSH_HOST:-round4-a100-2}"
A100_BASE="${ROUND4_A100_BASE:-/root}"
CODE_COMMIT="${ROUND4_CODE_COMMIT:?set ROUND4_CODE_COMMIT to the formal exact commit}"
LOCAL_EXPORT="$BASE/exports/round4/formal/$RUN_ID"
REMOTE_EXPORT="$A100_BASE/exports/round4/formal/$RUN_ID"
REFERENCE="${ROUND4_ALPACA_REFERENCE:-$BASE/data/round4-v1/raw/alpaca_eval-2edc6fad8be6/alpaca_eval_gpt4_baseline.json}"
MAX_WAIT_SECONDS="${ROUND4_CONTROLLER_MAX_WAIT_SECONDS:-259200}"

fail() { echo "ERROR: $*" >&2; exit 1; }
remote() { ssh -F "$A100_CONFIG" -o BatchMode=yes "$A100_HOST" "$@"; }

[[ -x "$PYTHON" && -f "$JUDGE" ]] || fail "judge environment or script is unavailable"
[[ -r "$REFERENCE" ]] || fail "frozen AlpacaEval reference outputs are unavailable"
[[ -r "$A100_CONFIG" ]] || fail "A100 SSH config is unavailable"
[[ "$(remote "git -C '$A100_BASE/SOPPO' rev-parse HEAD")" == "$CODE_COMMIT" ]] || fail "A100 exact commit mismatch"
[[ -z "$(remote "git -C '$A100_BASE/SOPPO' status --porcelain")" ]] || fail "A100 worktree is not clean"
[[ ! -e "$LOCAL_EXPORT" ]] || fail "refusing to reuse local formal controller output: $LOCAL_EXPORT"
mkdir -p "$LOCAL_EXPORT"

copy_and_verify() {
    local method="$1"
    local target="$LOCAL_EXPORT/$method"
    local output="$target/alpacaeval_outputs.json"
    local request="$target/JUDGE_REQUEST_${method}.json"
    mkdir -p "$target"
    scp -F "$A100_CONFIG" -o BatchMode=yes "$A100_HOST:$REMOTE_EXPORT/$method/alpacaeval_outputs.json" "$output"
    scp -F "$A100_CONFIG" -o BatchMode=yes "$A100_HOST:$REMOTE_EXPORT/$method/alpacaeval_outputs.json.manifest.json" "$output.manifest.json"
    scp -F "$A100_CONFIG" -o BatchMode=yes "$A100_HOST:$REMOTE_EXPORT/JUDGE_REQUEST_${method}.json" "$request"
    "$PYTHON" - "$request" "$method" "$output" <<'PY'
import hashlib, json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = request.get("methods", {}).get(sys.argv[2])
output = Path(sys.argv[3])
if not isinstance(entry, dict):
    raise SystemExit("judge request lacks requested method")
digest = hashlib.sha256(output.read_bytes()).hexdigest()
rows = json.loads(output.read_text(encoding="utf-8"))
if digest != entry.get("output_sha256") or len(rows) != entry.get("row_count"):
    raise SystemExit("copied output does not match A100 judge request")
print(f"verified method={sys.argv[2]} rows={len(rows)} output_sha256={digest}")
PY
}

judge_method() {
    local method="$1"
    local target="$LOCAL_EXPORT/$method"
    copy_and_verify "$method"
    bash "$JUDGE" primary "$target/alpacaeval_outputs.json" "$REFERENCE" "$target/judge-primary"
    "$PYTHON" - "$target/judge-primary/JUDGE_RESULT.json" "$method" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = payload.get("metrics", {})
print("judge_complete method={} win_rate={} length_controlled_winrate={}".format(sys.argv[2], metrics.get("win_rate"), metrics.get("length_controlled_winrate")))
PY
}

wait_for_method() {
    local method="$1" elapsed=0
    while true; do
        if remote "test -f '$REMOTE_EXPORT/$method/alpacaeval_outputs.json' && test -f '$REMOTE_EXPORT/JUDGE_REQUEST_${method}.json'"; then return 0; fi
        if ! remote "pgrep -f '[0]5_run_formal_a100.sh $method $RUN_ID' >/dev/null"; then fail "A100 action '$method' stopped without immutable output/request"; fi
        if (( elapsed >= MAX_WAIT_SECONDS )); then fail "timed out waiting for A100 action '$method'"; fi
        echo "waiting method=$method elapsed_seconds=$elapsed"
        sleep 30
        elapsed=$((elapsed + 30))
    done
}

start_and_wait() {
    local method="$1"
    remote "export ROUND4_CODE_COMMIT='$CODE_COMMIT'; bash '$A100_BASE/SOPPO/code/scripts/round4/05_run_formal_a100.sh' '$method' '$RUN_ID'" &
    local remote_pid=$!
    wait_for_method "$method"
    wait "$remote_pid"
}

# Base was launched separately. Each later arm begins only after its predecessor
# has a completed, immutable 4090 judge result.
wait_for_method base
judge_method base
for method in dpo sspo staticpe frozenpe; do
    start_and_wait "$method"
    judge_method "$method"
done

"$PYTHON" - "$LOCAL_EXPORT" "$RUN_ID" "$CODE_COMMIT" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
methods = {}
for method in ("base", "dpo", "sspo", "staticpe", "frozenpe"):
    methods[method] = json.loads((root / method / "judge-primary" / "JUDGE_RESULT.json").read_text(encoding="utf-8"))["metrics"]
(root / "FORMAL_CONTROLLER_SUMMARY.json").write_text(json.dumps({"schema": "round4-formal-controller-summary-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": sys.argv[2], "code_commit": sys.argv[3], "methods": methods}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(root / "FORMAL_CONTROLLER_SUMMARY.json")
PY
