#!/usr/bin/env python3
"""PROBE-CORE: materialize exact MATH500/AIME24 probes and measure landmarks."""
from __future__ import annotations
import argparse,gc,hashlib,json
from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_followup_common as c
ROOT=c.scoped_run('H2_probe_core'); CORPORA=ROOT/'corpora'; CONTRACT=ROOT/'probe_core_contract.json'; OUTPUT=ROOT/'PROBE_CORE_r_epsilon.csv'; MANIFEST=ROOT/'PROBE_CORE_manifest.json'; EPS=(.01,.025,.05,.10)


def records(path:Path)->list[dict]:
 with path.open(encoding='utf-8') as h:return [json.loads(x) for x in h if x.strip()]
def problem(row:dict)->str:
 lower={str(key).lower():value for key,value in row.items()}
 for key in ('problem','question','prompt','text'):
  value=lower.get(key)
  if isinstance(value,str) and value.strip():return value.strip()
 raise KeyError('no problem field')
def model_root(family:str)->Path:return c.AUTODL/'model/Qwen/Qwen3-4B-Base' if family=='qwen3_4b' else c.AUTODL/'model/Meta/modelscope/Llama-3.2-3B'
def fixed_rows(tokenizer,rows:list[dict],probe:str)->list[dict]:
 out=[]
 for index,row in enumerate(rows):
  text=problem(row); ids=tokenizer(text,add_special_tokens=False)['input_ids'];
  if not ids:raise RuntimeError(f'empty tokenization {probe}/{index}')
  out.append({'sample_id':f'{probe}_{index:04d}','probe_type':'E','domain':probe,'source_kind':'exact_eval_prompt','prompt_text':'','generation_text':text,'prompt_token_ids':[],'generation_token_ids':list(map(int,ids)),'full_token_ids':list(map(int,ids)),'eligible_start':0,'eligible_end':len(ids),'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
 return out

def prepare(families:list[str])->dict:
 math=records(c.REPO/'Eval/tasks/data/hendrycks_math500/test.jsonl');aime=records(c.REPO/'Eval/tasks/data/aime24/train.jsonl')
 if len(math)!=500:raise RuntimeError(f'MATH500 count drift {len(math)}')
 if not aime:raise RuntimeError('AIME24 is empty')
 outputs=[]
 for family in families:
  tokenizer=AutoTokenizer.from_pretrained(str(model_root(family)),local_files_only=True,trust_remote_code=True)
  for probe,source in [('E_math',math),('E_aime24',aime)]:
   target=CORPORA/family/f'{probe}.jsonl';c.atomic_jsonl(target,fixed_rows(tokenizer,source,probe));outputs.append(c.artifact(target))
 contract={'schema_version':1,'status':'complete','task':'PROBE-CORE exact v2 corpus materialization','families':families,'sources':{'E_math':str(c.REPO/'Eval/tasks/data/hendrycks_math500/test.jsonl'),'E_aime24':str(c.REPO/'Eval/tasks/data/aime24/train.jsonl'),'E_mathHeld':'historical 32-item artifact, never relabelled'},'outputs':outputs,'E_math_label':'E_math','historical_label':'E_mathHeld','created_utc':c.utc_now()};c.atomic_json(CONTRACT,contract);return contract

def model_path(family:str,arm:str,step:int)->Path:
 if family=='qwen3_4b':
  import cycle09_q1_geometry as q1;import cycle09_stage3_common as s3;q1.configure();return s3.model_path(arm,step)
 import cycle09_llama_model_export as export; import cycle09_block3_common as b3
 return b3.LLAMA_STUDENT if step==0 else export.merged_target(arm,step)
def load_checkpoint(family:str,arm:str,step:int,device:str):
 if family=='llama3_2_3b':return c.load_llama_checkpoint(arm,step,device)
 import cycle09_stage3_tpk as tpk
 return tpk.load_qwen_checkpoint(arm,step,device)[0]

def contexts(family:str):
 if family=='qwen3_4b':return [('opd',(0,20,40,160,320)),('sft',(0,20,40,160,320)),('offkd',(0,20,40,160,320)),('seqkd',(0,20,40,160,320)),('alpha05',(0,20,40,80,160,320))]
 return [('opd',(0,20,40,80,160,320)),('sft',(0,20,40,80,160,320)),('offkd',(0,20,40,80,160,320)),('seqkd',(0,20,40,80,160,320))]
def samples(path:Path,tokenizer):return c4.prepare_samples(path,tokenizer,corpus_id=str(path),window_seed=c4.WINDOW_SEED,max_context_tokens=c4.MAX_CONTEXT_TOKENS)
def ranks(x):
 sigma=torch.linalg.svdvals(x).double().cpu().tolist();return {eps:c4.functional_rank(sigma,eps) for eps in EPS}
def measure(families:list[str],device:str,arms_filter:list[str],steps_filter:list[int],probes_filter:list[str],sample_limit:int)->dict:
 contract=c.read_json(CONTRACT,{})
 if contract.get('status')!='complete':raise RuntimeError('exact probe contract is not materialized')
 rows=[]
 for family in families:
  layer=18 if family=='qwen3_4b' else 14;tok=AutoTokenizer.from_pretrained(str(model_root(family)),local_files_only=True,trust_remote_code=True);base=load_checkpoint(family,'opd',0,device)
  try:
   cache={}
   for probe in probes_filter or ('E_math','E_aime24'):
    ps=samples(CORPORA/family/f'{probe}.jsonl',tok)
    if sample_limit:ps=ps[:sample_limit]
    if not ps:raise RuntimeError(f'no samples for {family}/{probe}')
    profile=campaign.collect_profile(base,ps,[layer],device,keep_factors=False,keep_residual_samples=False,factor_layers=(),forward_batch_size=8,max_batch_tokens=16384,early_stop=True);cache[probe]=(ps,profile,campaign.scaling_by_group(profile,[layer],device))
   for arm,steps in contexts(family):
    for step in steps:
     if arms_filter and arm not in arms_filter:continue
     if step==0 and arm!='opd':continue
     if steps_filter and step not in steps_filter:continue
     current=base if step==0 else load_checkpoint(family,arm,step,device)
     try:
      for probe,(ps,base_profile,base_scales) in cache.items():
       current_profile=base_profile if step==0 else campaign.collect_profile(current,ps,[layer],device,keep_factors=False,keep_residual_samples=False,factor_layers=(),forward_batch_size=8,max_batch_tokens=16384,early_stop=True);current_scales=base_scales if step==0 else campaign.scaling_by_group(current_profile,[layer],device)
       for module in c.MODULES:
        group=c4.MODULE_TO_GROUP[module];weight=campaign.module_at(current,layer,module).weight.detach().to(device=device,dtype=torch.float32);base_weight=campaign.module_at(base,layer,module).weight.detach().to(device=device,dtype=torch.float32);now=ranks(weight@current_scales[layer][group]);before=ranks(base_weight@base_scales[layer][group])
        for eps in EPS:rows.append({'family':family,'arm':arm if step else 'base','step':step,'probe':probe,'layer':layer,'module':module,'epsilon':eps,'r_epsilon':now[eps],'base_r_epsilon':before[eps],'delta_from_base':now[eps]-before[eps],'track':'per_checkpoint_S_Dt','normalization':'window token mean -> sample window mean -> sample equal mean'})
        del weight,base_weight
       if step:current_scales.clear();del current_profile
       torch.cuda.empty_cache()
     finally:
      if current is not base:campaign.unload_model(current)
   for _,profile,scales in cache.values():scales.clear();del profile
  finally:campaign.unload_model(base);gc.collect();torch.cuda.empty_cache()
 c.atomic_csv(OUTPUT,rows);result={'schema_version':1,'status':'complete','task':'PROBE-CORE exact landmark geometry','contract':c.artifact(CONTRACT),'families':families,'arms':arms_filter or 'formal_default','steps':steps_filter or 'formal_default','probes':probes_filter or ['E_math','E_aime24'],'sample_limit':sample_limit or None,'llama_runtime_policy':'PEFT adapter merged in memory; no persistent merged checkpoint required','output':c.artifact(OUTPUT),'rows':len(rows),'created_utc':c.utc_now()};c.atomic_json(MANIFEST,result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--families',default='qwen3_4b,llama3_2_3b');p.add_argument('--phase',choices=('prepare','measure','all'),required=True);p.add_argument('--arms',default='');p.add_argument('--steps',default='');p.add_argument('--probes',default='');p.add_argument('--sample-limit',type=int,default=0);p.add_argument('--device',default='cuda:0');a=p.parse_args();f=[x for x in a.families.split(',') if x]
 arms=[x for x in a.arms.split(',') if x];steps=[int(x) for x in a.steps.split(',') if x];probes=[x for x in a.probes.split(',') if x]
 if a.phase=='prepare':x=prepare(f)
 elif a.phase=='measure':x=measure(f,a.device,arms,steps,probes,a.sample_limit)
 else:prepare(f);x=measure(f,a.device,arms,steps,probes,a.sample_limit)
 print(json.dumps(x,indent=2))
