#!/usr/bin/env bash
# Stage 1 GPU chain: wait for H/B1 backfill -> recompute S1-7 (ten points) -> S1-3 (main).
# No auto-shutdown (user asked to stay up).
set -uo pipefail
R=/root/autodl-tmp/cycle09_s1; PY=/root/miniconda3/envs/density/bin/python
S=/root/LLM-output-density/experiments/opd_sft_h1/scripts
mkdir -p $R/logs
log(){ printf '[s1 %s] %s\n' "$(date '+%F %T')" "$*" | tee -a $R/logs/chain.log; }
st(){ printf 'stage1 chain: %s\ndetail: %s\nupdated: %s\n' "$1" "$2" "$(date '+%F %T')" > $R/S1_STATUS.txt; }

st RUNNING "waiting for H/B1 backfill"
while pgrep -f cycle09_s1_backfill_hgen > /dev/null; do sleep 30; done
log "H/B1 backfill finished"

st RUNNING "S1-7 recompute (ten points)"
$PY $S/cycle09_s1_7_htext.py >> $R/logs/s1_7.log 2>&1 && log "S1-7 done" || log "S1-7 FAILED"

st RUNNING "S1-3 collect+bootstrap (main)"
log "starting S1-3"
$PY $S/cycle09_s1_3_transient_ci.py --all >> $R/logs/s1_3.log 2>&1
rc=$?
[ $rc -eq 0 ] && { log "S1-3 COMPLETED"; st DONE "S1-3 complete"; } || { log "S1-3 FAILED rc=$rc"; st FAILED "S1-3 rc=$rc"; }
