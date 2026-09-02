#!/usr/bin/env bash
# Server task: run one Round4 API judge profile on 4090-3 only.
set -euo pipefail
umask 077

fail() { echo "ERROR: $*" >&2; exit 1; }

PROFILE_NAME="${1:?usage: 04_run_api_judge_4090.sh PROFILE_NAME MODEL_OUTPUTS REFERENCE_OUTPUTS OUTPUT_DIR}"
MODEL_OUTPUTS="${2:?missing MODEL_OUTPUTS}"
REFERENCE_OUTPUTS="${3:?missing REFERENCE_OUTPUTS}"
OUTPUT_DIR="${4:?missing OUTPUT_DIR}"
PROFILE_STORE="${ROUND4_JUDGE_PROFILE_STORE:-$HOME/.config/soppo/judge_profiles.json}"
CREDENTIALS_STORE="${ROUND4_JUDGE_CREDENTIALS_STORE:-$HOME/.config/soppo/judge_credentials.json}"
PYTHON_BIN="${ROUND4_JUDGE_PYTHON:?set ROUND4_JUDGE_PYTHON to the 4090 judge-environment Python}"
ALPACA_CLI="${ROUND4_ALPACA_EVAL_CLI:-${PYTHON_BIN%/python}/alpaca_eval}"

[[ -f "$MODEL_OUTPUTS" && -f "$REFERENCE_OUTPUTS" ]] || fail "model/reference outputs are required"
[[ -f "$PROFILE_STORE" ]] || fail "judge profile store does not exist"
[[ -x "$PYTHON_BIN" && -x "$ALPACA_CLI" ]] || fail "judge Python or alpaca_eval executable is unavailable"
[[ "$(stat -c '%a' "$PROFILE_STORE")" =~ ^[46][0-7][0-7]$ ]] || fail "profile store must not be group/world readable"
if [[ -e "$CREDENTIALS_STORE" ]]; then
    [[ -f "$CREDENTIALS_STORE" ]] || fail "judge credentials store must be a regular file"
    [[ "$(stat -c '%a' "$CREDENTIALS_STORE")" =~ ^[46][0-7][0-7]$ ]] || fail "credentials store must not be group/world readable"
fi
[[ ! -e "$OUTPUT_DIR" ]] || fail "refusing to overwrite judge output: $OUTPUT_DIR"

CREDENTIAL_EXPORTS="$("$PYTHON_BIN" - "$PROFILE_STORE" "$CREDENTIALS_STORE" "$PROFILE_NAME" <<'PY'
import json, os, shlex, sys
import re
from pathlib import Path

profiles = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("profiles", {})
profile = profiles.get(sys.argv[3])
if not isinstance(profile, dict):
    raise SystemExit("unknown judge profile")

credentials = {}
credentials_path = Path(sys.argv[2])
if credentials_path.exists():
    credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
if not isinstance(credentials, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in credentials.items()):
    raise SystemExit("credentials store must be a string-to-string JSON object")

for name in (profile.get("api_key_env"), profile.get("base_url_env")):
    if not isinstance(name, str) or not name:
        raise SystemExit("profile lacks credential environment variable names")
    value = os.environ.get(name) or credentials.get(name)
    if not value:
        raise SystemExit("credential is absent: " + name)
    print("export " + name + "=" + shlex.quote(value))
PY
)" || fail "judge credential resolution failed"
eval "$CREDENTIAL_EXPORTS"

PROFILE_JSON="$($PYTHON_BIN - "$PROFILE_STORE" "$PROFILE_NAME" <<'PY'
import json, os, sys
from pathlib import Path

store = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
profile = store.get("profiles", {}).get(sys.argv[2])
if not isinstance(profile, dict):
    raise SystemExit("unknown judge profile")
for key in ("api_key_env", "base_url_env", "model_name"):
    if not isinstance(profile.get(key), str) or not profile[key]:
        raise SystemExit("profile lacks " + key)
if not os.environ.get(profile["api_key_env"]):
    raise SystemExit("API key environment variable is absent")
if not os.environ.get(profile["base_url_env"]):
    raise SystemExit("base URL environment variable is absent")
print(json.dumps(profile, sort_keys=True))
PY
)" || fail "judge profile validation failed"

export OPENAI_API_KEY
export OPENAI_BASE_URL
export OPENAI_API_BASE
eval "$($PYTHON_BIN - "$PROFILE_JSON" <<'PY'
import json, os, shlex, sys
p = json.loads(sys.argv[1])
print("OPENAI_API_KEY=" + shlex.quote(os.environ[p["api_key_env"]]))
print("OPENAI_BASE_URL=" + shlex.quote(os.environ[p["base_url_env"]]))
print("OPENAI_API_BASE=" + shlex.quote(os.environ[p["base_url_env"]]))
PY
)"

mkdir -p "$OUTPUT_DIR"
CONFIG_DIR="$OUTPUT_DIR/annotator_config"
mkdir -p "$CONFIG_DIR"
"$PYTHON_BIN" - "$PROFILE_JSON" "$CONFIG_DIR" <<'PY'
import json, re, shutil, sys
from pathlib import Path
import alpaca_eval

profile = json.loads(sys.argv[1])
output = Path(sys.argv[2])
root = Path(alpaca_eval.__file__).resolve().parent / "evaluators_configs" / profile.get("annotator_template", "weighted_alpaca_eval_gpt4_turbo")
if not (root / "configs.yaml").is_file():
    raise SystemExit("annotator template is unavailable in installed alpaca_eval")
shutil.copytree(root, output, dirs_exist_ok=True)
config = output / "configs.yaml"
text = config.read_text(encoding="utf-8")
settings = {
    "model_name": profile["model_name"],
    "max_tokens": profile.get("max_tokens", 1),
    "temperature": profile.get("temperature", 1),
    "logprobs": profile.get("logprobs", True),
    "top_logprobs": profile.get("top_logprobs", 5),
}
if not isinstance(settings["model_name"], str) or not settings["model_name"]:
    raise SystemExit("profile model_name must be a non-empty string")
if not isinstance(settings["max_tokens"], int) or settings["max_tokens"] < 1:
    raise SystemExit("profile max_tokens must be a positive integer")
if not isinstance(settings["temperature"], (int, float)):
    raise SystemExit("profile temperature must be numeric")
if not isinstance(settings["logprobs"], bool):
    raise SystemExit("profile logprobs must be boolean")
if not isinstance(settings["top_logprobs"], int) or settings["top_logprobs"] < 0:
    raise SystemExit("profile top_logprobs must be a non-negative integer")

for key, value in settings.items():
    replacement = "    {}: {}".format(key, json.dumps(value))
    text, count = re.subn(r"(?m)^    " + re.escape(key) + r":.*$", replacement, text)
    if count != 1:
        raise SystemExit("installed evaluator template lacks exactly one completions_kwargs." + key)
config.write_text(text, encoding="utf-8")
PY

"$ALPACA_CLI" \
  --model_outputs "$MODEL_OUTPUTS" \
  --reference_outputs "$REFERENCE_OUTPUTS" \
  --annotators_config "$CONFIG_DIR" \
  --fn_metric get_length_controlled_winrate \
  --sort_by length_controlled_winrate \
  --output_path "$OUTPUT_DIR"

"$PYTHON_BIN" - "$PROFILE_JSON" "$OUTPUT_DIR" <<'PY'
import csv, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

profile = json.loads(sys.argv[1])
out = Path(sys.argv[2])
leaderboards = list(out.rglob("leaderboard.csv"))
if len(leaderboards) != 1:
    raise SystemExit("expected one leaderboard.csv")
with leaderboards[0].open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if len(rows) != 1:
    raise SystemExit("expected exactly one leaderboard row")
row = rows[0]
metrics = {key: float(row[key]) for key in ("win_rate", "length_controlled_winrate")}
if not all(math.isfinite(value) for value in metrics.values()):
    raise SystemExit("non-finite WR/LC")
fingerprint = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
(out / "JUDGE_RESULT.json").write_text(json.dumps({
    "schema": "round4-judge-result-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "profile_fingerprint": fingerprint,
    "judge_model": profile["model_name"],
    "metrics": metrics,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
