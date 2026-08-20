#!/usr/bin/env python3
"""Materialize frozen step0-self rollouts and RAW teacher top-32 labels for H5."""
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cycle09_block3_common as b3
import cycle09_stage3_followup_common as c
ROOT=c.scoped_run('H5_frozen_self'); ROLLOUT=ROOT/'rollout'; STORE=ROOT/'frozen_store'; DATA=ROOT/'data'; MANIFEST=ROOT/'frozen_support_manifest.json'

def generate(args:argparse.Namespace)->dict:
 import cycle09_offkd_rollout as r
 r.EXP_ROOT=ROOT; r.COPYBACK=ROOT/'unused_copyback'; r.TEACHER=b3.LLAMA_TEACHER; r.SAMPLING_MODEL=b3.LLAMA_STUDENT; r.PROMPTS=b3.L1_DATA/'llama_opd_prompts_4999.parquet'; r.RUN_LABEL='frozenSelf0_KD'
 r.TOKENIZER_LOADER=b3.load_llama_tokenizer
 # Reuse the validated two-pass implementation: pass1 is step0 student sampling,
 # pass2 is teacher RAW top-32 over exactly those persisted token IDs.
 import sys
 old=list(sys.argv); sys.argv=['cycle09_frozen_self_materialize.py','--out',str(ROLLOUT),'--stage',args.stage]
 if args.smoke: sys.argv.append('--smoke')
 try:r.main()
 finally:sys.argv=old
 actual=ROOT/'smoke/rollout' if args.smoke else ROLLOUT
 return {'status':'generated','rollout':str(actual)}

def rows(path:Path):
 with path.open(encoding='utf-8') as h:return [json.loads(line) for line in h if line.strip()]

def convert(rollout:Path=ROLLOUT)->dict:
 raw=rollout/'teacher_rollout_pass1.jsonl'; stream=rollout/'pass2_stream'
 required=[raw,stream/'top32_ids.npy',stream/'top32_logprob.npy',stream/'row_offsets.npy']
 missing=[str(p) for p in required if not p.is_file()]
 if missing:raise FileNotFoundError(missing)
 rec=rows(raw); STORE.mkdir(parents=True,exist_ok=True)
 po=np.zeros(len(rec)+1,dtype=np.int64);ro=np.zeros(len(rec)+1,dtype=np.int64)
 for i,row in enumerate(rec):po[i+1]=po[i]+len(row['prompt_token_ids']);ro[i+1]=ro[i]+len(row['generation_token_ids'])
 for name,array in [('prompt_offsets.npy',po),('response_offsets.npy',ro)]:np.save(STORE/name,array,allow_pickle=False)
 for field,name,offsets in [('prompt_token_ids','prompt_ids.npy',po),('generation_token_ids','response_ids.npy',ro)]:
  target=STORE/name;mm=np.lib.format.open_memmap(target,mode='w+',dtype=np.int32,shape=(int(offsets[-1]),))
  for i,row in enumerate(rec):mm[int(offsets[i]):int(offsets[i+1])]=np.asarray(row[field],dtype=np.int32)
  mm.flush();del mm
 for name in ('top32_ids.npy','top32_logprob.npy','row_offsets.npy'):shutil.copy2(stream/name,STORE/name)
 top=np.load(STORE/'top32_ids.npy',mmap_mode='r'); offsets=np.load(STORE/'row_offsets.npy',mmap_mode='r')
 starts,ends=(offsets[:,0],offsets[:,1]) if offsets.ndim==2 else (offsets[:-1],offsets[1:])
 if not(np.array_equal(ro[:-1],starts) and np.array_equal(ro[1:],ends) and top.shape[0]==int(ro[-1]) and top.shape[1]==32):raise RuntimeError('frozen store alignment failure')
 source=pd.read_parquet(b3.L1_DATA/'llama_opd_prompts_4999.parquet').iloc[:len(rec)].copy().reset_index(drop=True)
 source['external_record_index']=np.arange(len(source),dtype=np.int64);source['agent_name']='frozen_self_external';source['support_source']='frozen_step0_self';source['support_source_id']=1
 DATA.mkdir(parents=True,exist_ok=True);schedule=DATA/'frozen_self_schedule.parquet';source.to_parquet(schedule,index=False)
 payload={'schema_version':1,'status':'complete','task':'frozenSelf0-KD frozen step0 support + teacher RAW top32','rollout':c.artifact(raw),'store':[c.artifact(STORE/name) for name in ('prompt_ids.npy','prompt_offsets.npy','response_ids.npy','response_offsets.npy','top32_ids.npy','top32_logprob.npy','row_offsets.npy')],'schedule':c.artifact(schedule),'n_records':len(rec),'sampling_model':str(b3.LLAMA_STUDENT),'teacher_label_model':str(b3.LLAMA_TEACHER),'pass1':'temperature=.6 top_p=.9 top_k=-1 seed=42','pass2':'teacher forward RAW top-32 on exact pass1 sequence','created_utc':c.utc_now()};c.atomic_json(MANIFEST,payload);return payload
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('generate','convert','all'),required=True);p.add_argument('--stage',choices=('pass1','pass2','all'),default='all');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.phase=='generate':x=generate(a)
 elif a.phase=='convert':x=convert()
 else:generated=generate(a);x=convert(Path(generated['rollout']))
 print(json.dumps(x,indent=2))
