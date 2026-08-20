#!/usr/bin/env bash
# Detached, smoke-only validation for the Cycle09 Stage3 follow-up queue.
set -uo pipefail

SCOPE="${1:-smoke_$(date -u +%Y%m%d_%H%M%S)}"
REPO=/root/LLM-output-density
SCRIPTS="$REPO/experiments/opd_sft_h1/scripts"
PYTHON=/root/miniconda3/envs/density/bin/python
ROOT="/root/autodl-tmp/cycle09_stage3_followup/smoke/$SCOPE/runner"
STATUS="$ROOT/status.json"
RETURNS="$ROOT/returncodes.tsv"

mkdir -p "$ROOT"
export CYCLE09_STAGE3_SCOPE="$SCOPE"
printf '{"status":"running","scope":"%s","formal_pipeline_started":false}\n' "$SCOPE" > "$STATUS"
: > "$RETURNS"

run_logged() {
  local name="$1"
  shift
  "$@" > "$ROOT/$name.log" 2>&1
  local rc=$?
  printf '%s\t%s\n' "$name" "$rc" >> "$RETURNS"
  return "$rc"
}

failed=0
run_logged static_compile "$PYTHON" -m py_compile \
  "$SCRIPTS/cycle09_stage3_followup_common.py" \
  "$SCRIPTS/cycle09_stage3_resync.py" \
  "$SCRIPTS/cycle09_stage3_support.py" \
  "$SCRIPTS/cycle09_stage3_tpk.py" \
  "$SCRIPTS/cycle09_stage3_twhite.py" \
  "$SCRIPTS/cycle09_stage3_tsub.py" \
  "$SCRIPTS/cycle09_stage3_probe_core.py" \
  "$SCRIPTS/cycle09_stage3_increment.py" \
  "$SCRIPTS/cycle09_stage3_frozen_self.py" \
  "$SCRIPTS/cycle09_frozen_self_materialize.py" \
  "$SCRIPTS/cycle09_stage3_mediator.py" \
  "$SCRIPTS/cycle09_stage3_smoke_cpu.py" || failed=1

run_logged cpu_matrix "$PYTHON" "$SCRIPTS/cycle09_stage3_smoke_cpu.py" || failed=1

run_logged probe_core env CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$SCRIPTS/cycle09_stage3_probe_core.py" \
  --families qwen3_4b,llama3_2_3b --phase all --arms opd --steps 20 \
  --probes E_math --sample-limit 2 --device cuda:0 &
probe_pid=$!

run_logged frozen_materializer env CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
  "$SCRIPTS/cycle09_frozen_self_materialize.py" --phase all --smoke &
frozen_pid=$!

wait "$probe_pid" || failed=1
wait "$frozen_pid" || failed=1

if [[ "$failed" -eq 0 ]]; then
  final_status=complete
else
  final_status=failed
fi

"$PYTHON" -c "import csv,json,pathlib,datetime; root=pathlib.Path('$ROOT'); rows=[{'unit':r[0],'returncode':int(r[1])} for r in csv.reader((root/'returncodes.tsv').open(),delimiter='\\t') if r]; payload={'schema_version':1,'status':'$final_status','scope':'$SCOPE','formal_pipeline_started':False,'auto_shutdown':False,'units':rows,'outputs':{'cpu_manifest':str(root/'cpu_smoke_manifest.json'),'probe_manifest':str(root.parent/'H2_probe_core/PROBE_CORE_manifest.json'),'frozen_manifest':str(root.parent/'H5_frozen_self/frozen_support_manifest.json')},'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}; (root/'stage3_smoke_result.json').write_text(json.dumps(payload,indent=2)+'\\n'); (root/'status.json').write_text(json.dumps({'status':'$final_status','scope':'$SCOPE'})+'\\n')"

exit "$failed"
