#!/usr/bin/env bash
# Resolve explicit public Git refs to immutable commits; network/server only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
[[ -x "$ROUND3_TRAIN_PYTHON" ]] || { echo "ERROR: Round3 train environment is missing" >&2; exit 1; }
OUTPUT="$ROUND3_RUN_ROOT/source_revisions.json"
[[ ! -e "$OUTPUT" ]] || { echo "ERROR: refuse to overwrite source revision evidence: $OUTPUT" >&2; exit 1; }

MODEL_REF="${SOPPO_ROUND3_MODEL_REF:-}"
UF_REF="${SOPPO_ROUND3_ULTRAFEEDBACK_REF:-}"
UC_REF="${SOPPO_ROUND3_ULTRACHAT_REF:-}"
for value in "$MODEL_REF" "$UF_REF" "$UC_REF"; do
    [[ "$value" =~ ^refs/(heads|tags)/[A-Za-z0-9._/-]+$ ]] || {
        echo "ERROR: every Round3 source ref must be an explicit refs/heads/... or refs/tags/... value" >&2
        exit 1
    }
done

resolve_ref() {
    local url="$1"
    local ref="$2"
    local listing
    listing="$(git ls-remote --exit-code "$url" "$ref" "${ref}^{}")" || {
        echo "ERROR: cannot resolve $url $ref" >&2
        return 1
    }
    local sha
    sha="$(printf '%s\n' "$listing" | awk -v peeled="${ref}^{}" '$2 == peeled {print $1; found=1} END {if (!found) exit 1}')" || \
        sha="$(printf '%s\n' "$listing" | awk -v exact="$ref" '$2 == exact {print $1}')"
    [[ "$(printf '%s\n' "$sha" | wc -l | tr -d '[:space:]')" == "1" ]] || {
        echo "ERROR: ambiguous source revision: $url $ref" >&2
        return 1
    }
    round3_require_full_sha "$sha" "resolved SHA for $url $ref"
    printf '%s\n' "$sha"
}

MODEL_SHA="$(resolve_ref 'https://www.modelscope.cn/Qwen/Qwen3-1.7B.git' "$MODEL_REF")"
UF_SHA="$(resolve_ref 'https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized.git' "$UF_REF")"
UC_SHA="$(resolve_ref 'https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k.git' "$UC_REF")"
mkdir -p "$(dirname "$OUTPUT")"
"$ROUND3_TRAIN_PYTHON" - "$OUTPUT" "$MODEL_REF" "$MODEL_SHA" "$UF_REF" "$UF_SHA" "$UC_REF" "$UC_SHA" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": "round3.source_revisions.v1",
    "resolved_at": datetime.now(timezone.utc).isoformat(),
    "sources": {
        "model": {
            "repo_id": "Qwen/Qwen3-1.7B", "requested_ref": sys.argv[2],
            "resolved_sha": sys.argv[3], "transport": "public_git_ls_remote",
        },
        "ultrafeedback": {
            "repo_id": "HuggingFaceH4/ultrafeedback_binarized", "requested_ref": sys.argv[4],
            "resolved_sha": sys.argv[5], "transport": "public_git_ls_remote",
        },
        "ultrachat": {
            "repo_id": "HuggingFaceH4/ultrachat_200k", "requested_ref": sys.argv[6],
            "resolved_sha": sys.argv[7], "transport": "public_git_ls_remote",
        },
    },
}
partial = output.with_suffix(".json.partial")
partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
partial.replace(output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
