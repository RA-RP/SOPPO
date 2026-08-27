#!/usr/bin/env bash
# Reuse immutable source revisions only after the old exact-commit controller is terminal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
round3_require_baseline_experiment_id
SOURCE="$ROUND3_BASELINE_RUN_ROOT/source_revisions.json"
CONTROLLER="$ROUND3_BASELINE_RUN_ROOT/controller.json"
BASELINE_CONFIG="$ROUND3_BASELINE_RUN_ROOT/resolved/formal/dpo_1k.yaml"
OUTPUT="$ROUND3_RUN_ROOT/source_revisions.json"
LINK="$ROUND3_RUN_ROOT/baseline_link.json"
[[ -f "$SOURCE" && -f "$CONTROLLER" && -f "$BASELINE_CONFIG" ]] || {
    echo "ERROR: baseline source/controller/config evidence is incomplete" >&2
    exit 1
}
[[ ! -e "$OUTPUT" && ! -e "$LINK" ]] || {
    echo "ERROR: refuse to overwrite Round3 extension baseline evidence" >&2
    exit 1
}
mkdir -p "$ROUND3_RUN_ROOT"
"$ROUND3_TRAIN_PYTHON" - "$SOURCE" "$CONTROLLER" "$BASELINE_CONFIG" "$OUTPUT" "$LINK" \
    "$SOPPO_ROUND3_BASELINE_EXPERIMENT_ID" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

source, controller, baseline_config, output, link = map(Path, sys.argv[1:6])
baseline_id = sys.argv[6]
revisions = json.loads(source.read_text(encoding="utf-8"))
state = json.loads(controller.read_text(encoding="utf-8"))
config = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))
if revisions.get("schema_version") != "round3.source_revisions.v1":
    raise SystemExit("malformed baseline source revisions")
if set(revisions.get("sources", {})) != {"model", "ultrafeedback", "ultrachat"}:
    raise SystemExit("baseline source inventory changed")
if state.get("state") != "completed" or state.get("stage") != "all_methods":
    raise SystemExit("baseline Round3 controller has not completed all methods")
if state.get("experiment_id") != baseline_id:
    raise SystemExit("baseline controller experiment ID mismatch")
baseline_commit = state.get("git_commit", "")
if len(baseline_commit) != 40 or any(c not in "0123456789abcdef" for c in baseline_commit):
    raise SystemExit("baseline controller Git commit is malformed")
if (
    config.get("contract") != {
        "theory": "r3-theory-v1.0",
        "experiment": "round3-exp-v1.5",
    }
    or config.get("provenance", {}).get("experiment_id") != baseline_id
    or config.get("provenance", {}).get("git_commit") != baseline_commit
    or config.get("method", {}).get("name") != "dpo_1k"
):
    raise SystemExit("baseline resolved config does not identify the legacy formal")
for item in revisions["sources"].values():
    sha = item.get("resolved_sha", "")
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit("baseline source SHA is malformed")

raw = source.read_bytes()
source_partial = output.with_suffix(".json.partial")
source_partial.write_bytes(raw)
source_partial.replace(output)
payload = {
    "schema_version": "round3.extension_baseline_link.v1",
    "baseline_experiment_id": baseline_id,
    "baseline_git_commit": baseline_commit,
    "baseline_controller": str(controller.resolve()),
    "baseline_controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
    "baseline_config": str(baseline_config.resolve()),
    "baseline_config_sha256": hashlib.sha256(baseline_config.read_bytes()).hexdigest(),
    "baseline_reference_cache_dir": str(Path(config["data"]["reference_cache_dir"]).resolve()),
    "source_revisions": str(source.resolve()),
    "source_revisions_sha256": hashlib.sha256(raw).hexdigest(),
    "copied_source_revisions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "linked_at": datetime.now(timezone.utc).isoformat(),
}
partial = link.with_suffix(".json.partial")
partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
partial.replace(link)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
