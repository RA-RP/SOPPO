#!/usr/bin/env python3
"""T-SUB functional-subspace comparison with fixed input whitening.

The update object is explicit: adapter BA for process-level comparisons, or
BF16 merged-minus-base for final deployed-weight comparisons.
"""
from __future__ import annotations
import argparse, gc, json, math
from pathlib import Path
from typing import Any
import torch
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_followup_common as common
import cycle09_stage3_tpk as tpk

ROOT=common.scoped_run('H2_sub')


def preflight_path(family:str)->Path:
 return ROOT/f'T_SUB_{family}_adapter_preflight.json'


def preflight(args:argparse.Namespace)->dict[str,Any]:
 arms=[x.strip() for x in args.arms.split(',') if x.strip()]
 if len(arms)!=2:raise ValueError('T-SUB requires exactly two arms')
 steps=[int(x) for x in args.steps.split(',') if x.strip()]
 cells=[]
 for step in steps:
  cells.extend(tpk.delta_source(args.family,arm,step,args.delta_mode) for arm in arms)
 payload={'schema_version':1,'task':'T-SUB update-source preflight','family':args.family,
          'arms':arms,'steps':steps,'cells':cells,
          'delta_mode':args.delta_mode,
          'complete':all(item.get('complete') for item in cells),
          'created_utc':common.utc_now()}
 common.atomic_json(preflight_path(args.family),payload);return payload


def context(family:str):
 if family=='qwen3_4b':
  import cycle09_q1_geometry as q1; import cycle09_block3_qwen_probe_geometry as geom
  q1.configure()
  from transformers import AutoTokenizer
  return (common.AUTODL/'model/Qwen/Qwen3-4B-Base', lambda:AutoTokenizer.from_pretrained(str(common.AUTODL/'model/Qwen/Qwen3-4B-Base'),local_files_only=True,trust_remote_code=True), lambda p,t:geom.samples_for(p,t,factor_only=False), tuple(geom.ALL_PROBES))
 import cycle09_block3_common as b3; import cycle09_llama_geometry as geom
 return (b3.LLAMA_STUDENT,b3.load_llama_tokenizer,lambda p,t:geom.prepare_samples(t,p,0),tuple(geom.PROBE_NAMES))


def profile(family:str,model,samples,layer:int,device:str):
 if family=='qwen3_4b': return campaign.collect_profile(model,samples,[layer],device,keep_factors=False,keep_residual_samples=False,factor_layers=(),forward_batch_size=8,max_batch_tokens=16384,early_stop=True)
 import cycle09_llama_geometry as geom
 return geom.profile_model(model,samples,[layer],device,keep_sample_means=False,forward_batch_size=8,max_batch_tokens=16384)


def metric(left:torch.Tensor,right:torch.Tensor,k:int)->dict[str,float]:
 ul,_,vhl=torch.linalg.svd(left,full_matrices=False); ur,_,vhr=torch.linalg.svd(right,full_matrices=False)
 k=min(k,ul.shape[1],ur.shape[1]); u=ul[:,:k].T@ur[:,:k]; v=vhl[:k,:]@vhr[:k,:].T
 su=torch.linalg.svdvals(u).clamp(0,1); sv=torch.linalg.svdvals(v).clamp(0,1)
 angle=lambda s: torch.rad2deg(torch.acos(s.clamp(-1,1)))
 return {'k':k,'output_projector_overlap':float(torch.sum(u.square())/k),'input_projector_overlap_fixed':float(torch.sum(v.square())/k),'output_angle_mean_deg':float(angle(su).mean()),'output_angle_max_deg':float(angle(su).max()),'input_angle_mean_deg_fixed':float(angle(sv).mean()),'input_angle_max_deg_fixed':float(angle(sv).max())}


def run(args:argparse.Namespace)->dict[str,Any]:
 arms=[x.strip() for x in args.arms.split(',') if x.strip()];
 if len(arms)!=2: raise ValueError('T-SUB requires exactly two arms')
 steps=[int(x) for x in args.steps.split(',') if x.strip()]
 check=preflight(args)
 missing=[item for item in check['cells'] if not item.get('complete')]
 if missing:raise RuntimeError(f'update-source preflight failed for {args.delta_mode}: {missing}')
 base_path,tokenizer_fn,samples_fn,available_probes=context(args.family)
 probes=[x.strip() for x in args.probes.split(',') if x.strip()] or list(available_probes)
 unknown=set(probes).difference(available_probes)
 if unknown: raise ValueError(f'unknown probes {sorted(unknown)}')
 output=ROOT/f'T_SUB_{args.family}.csv'; manifest=ROOT/f'T_SUB_{args.family}_manifest.json'
 tokenizer=tokenizer_fn(); base=campaign.load_model(base_path,args.device); rows=[]
 try:
  for probe in probes:
   samples=samples_fn(probe,tokenizer)
   if args.sample_limit:samples=samples[:args.sample_limit]
   if not samples:raise RuntimeError(f'no samples for {probe}')
   base_profile=profile(args.family,base,samples,args.layer,args.device); scales=campaign.scaling_by_group(base_profile,[args.layer],args.device)
   try:
    for step in steps:
     sources=[tpk.delta_source(args.family,arm,step,args.delta_mode) for arm in arms]
     if not all(item.get('complete') for item in sources): raise RuntimeError(f'fixed-whitening subspace requires resolved update sources: {sources}')
     for module in common.MODULES:
      group=c4.MODULE_TO_GROUP[module]
      base_weight_bf16=tpk.source_weight_bf16(base,args.layer,module,args.device)
      deltas=[tpk.load_delta(source,args.layer,module,args.device,base_weight_bf16) for source in sources]
      if any(delta is None for delta in deltas): continue
      try:
       left=deltas[0]@scales[args.layer][group]; right=deltas[1]@scales[args.layer][group]
       k=max(1,int(round(args.rank_fraction*min(left.shape))))
       rows.append({'family':args.family,'arm_left':arms[0],'arm_right':arms[1],'step':step,'probe':probe,'layer':args.layer,'module':module,'rank_fraction':args.rank_fraction,**metric(left,right,k),'delta_construction':args.delta_mode,'left_native_source':sources[0].get('native_source','adapter_BA'),'right_native_source':sources[1].get('native_source','adapter_BA'),'input_coordinates':'fixed_base_S_D0'})
      finally:
       for delta in deltas: del delta
       torch.cuda.empty_cache()
   finally:
    scales.clear(); del base_profile
 finally:
  campaign.unload_model(base);gc.collect();torch.cuda.empty_cache()
 common.atomic_csv(output,rows)
 result={'schema_version':1,'status':'complete','task':'T-SUB direct functional-subspace comparison','family':args.family,'delta_mode':args.delta_mode,'arms':arms,'steps':steps,'layer':args.layer,'probes':probes,'sample_limit':args.sample_limit or None,'rank_fraction':args.rank_fraction,'preflight':common.artifact(preflight_path(args.family)),'output':common.artifact(output),'rows':len(rows),'created_utc':common.utc_now()}
 common.atomic_json(manifest,result);return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('preflight','run'),default='run');parser.add_argument('--family',choices=('qwen3_4b','llama3_2_3b'),required=True);parser.add_argument('--arms',default='opd,offkd');parser.add_argument('--steps',required=True);parser.add_argument('--layer',type=int,required=True);parser.add_argument('--probes',default='E_ood');parser.add_argument('--sample-limit',type=int,default=0);parser.add_argument('--rank-fraction',type=float,default=.05);parser.add_argument('--delta-mode',choices=('adapter_ba','bf16_merged_minus_base'),default='adapter_ba');parser.add_argument('--device',default='cuda:0')
 args=parser.parse_args();print(json.dumps(preflight(args) if args.phase=='preflight' else run(args),indent=2))
