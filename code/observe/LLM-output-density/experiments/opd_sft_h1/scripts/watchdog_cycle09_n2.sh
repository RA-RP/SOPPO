#!/usr/bin/env bash
# N-2 (pulled forward from the night block), single 96G GPU, strictly serial.
#   1  X_offkd_math probe: prepare (corpus + base reference) -> off-KD ten points
#   2  OPD H_bos/H_ood @step_080 measurement (generation already backfilled this afternoon)
#   3  T5: off-KD raw ER, ten points (R5-A6 / 2605.30524 replica)
#   4  off-KD H_bos/H_ood seven steps incl. generation (runs as far as time allows)
# Each item is atomic/resumable. No shutdown (user asked to stay up).
set -uo pipefail
R=/root/autodl-tmp/cycle09_n2; PY=/root/miniconda3/envs/density/bin/python
S=/root/LLM-output-density/experiments/opd_sft_h1/scripts
mkdir -p $R/logs
log(){ printf '[n2 %s] %s\n' "$(date '+%F %T')" "$*" | tee -a $R/logs/chain.log; }
st(){ printf 'N2 chain: %s\ndetail: %s\nupdated: %s\n' "$1" "$2" "$(date '+%F %T')" > $R/N2_STATUS.txt; }

# wait for the prepare job started outside the chain
while pgrep -f cycle09_n2_prepare_xoffkd > /dev/null; do sleep 20; done
log "X_offkd prepare finished"

st RUNNING "1: X_offkd_math geometry (off-KD ten points)"
log "START item1 X_offkd geometry"
$PY $S/cycle09_n2_xoffkd_geometry.py >> $R/logs/item1_xoffkd.log 2>&1 \
  && log "DONE item1" || { log "FAILED item1"; st FAILED item1; }

st RUNNING "2a: base references for OPD H@80"
log "START item2a build base refs"
$PY $S/cycle09_n2_build_ref.py --arm opd --steps 80 --domains bos,ood >> $R/logs/item2_h80.log 2>&1 \
  && log "DONE item2a" || { log "FAILED item2a"; st FAILED item2a; }

st RUNNING "2: OPD H_bos/H_ood @80 measurement"
log "START item2 OPD H@80"
$PY $S/cycle09_n2_h80_measure.py >> $R/logs/item2_h80.log 2>&1 \
  && log "DONE item2" || { log "FAILED item2"; st FAILED item2; }

st RUNNING "3: T5 off-KD raw ER ten points"
log "START item3 T5 raw ER"
$PY $S/cycle09_n2_t5_rawer.py >> $R/logs/item3_t5.log 2>&1 \
  && log "DONE item3" || { log "FAILED item3"; st FAILED item3; }

st RUNNING "4: off-KD H_bos/H_ood seven steps (incl. generation)"
log "START item4 offkd H"
$PY $S/cycle09_n2_offkd_h.py >> $R/logs/item4_offkd_h.log 2>&1 \
  && log "DONE item4" || { log "FAILED/PARTIAL item4 (resumable)"; }

st DONE "N-2 queue finished (item4 may be partial)"
log "N2 chain finished"
