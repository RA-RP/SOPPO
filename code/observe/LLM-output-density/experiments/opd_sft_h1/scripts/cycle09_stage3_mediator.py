#!/usr/bin/env python3
"""H6 descriptive mediator associations and one-family-at-a-time reweighting."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import cycle09_stage3_followup_common as c
ROOT=c.scoped_run('H6_mediator')

def run(gate:Path)->dict:
 if c.read_json(gate,{}).get('status')!='complete': raise RuntimeError('H5 gate is not complete')
 support=c.scope_root()/'H1_support/T_SUPPORT_stats.csv'; frozen=c.scope_root()/'H5_frozen_self/frozen_self_total_effect.csv'
 if not support.is_file() or not frozen.is_file(): raise FileNotFoundError('requires frozen support and H5 total-effect tables')
 left=pd.read_csv(support); right=pd.read_csv(frozen); keys=[key for key in ('family','arm','step') if key in left and key in right]
 joined=right.merge(left,on=keys,how='inner'); metrics=[key for key in ('response_tokens_mean','eos_rate','truncation_rate','distinct_2','distinct_4','token_entropy') if key in joined]
 rows=[]
 for metric in metrics:
  for outcome in [key for key in ('A_D','G_D','total_effect') if key in joined]: rows.append({'mediator':metric,'outcome':outcome,'n':len(joined),'pearson':joined[[metric,outcome]].corr().iloc[0,1],'analysis':'descriptive_post_treatment_association'})
 output=ROOT/'MEDIATOR_associations.csv';c.atomic_csv(output,rows)
 result={'schema_version':1,'status':'complete','task':'H6 mediator descriptive association only','gate':c.artifact(gate),'output':c.artifact(output),'created_utc':c.utc_now()};c.atomic_json(ROOT/'MEDIATOR_manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('all',),required=True);p.add_argument('--gate-file',type=Path,required=True);print(json.dumps(run(p.parse_args().gate_file),indent=2))
