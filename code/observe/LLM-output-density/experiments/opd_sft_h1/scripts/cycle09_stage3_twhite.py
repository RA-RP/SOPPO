#!/usr/bin/env python3
"""T-WHITE: weight-only, fixed-base, and current-conditioning r_epsilon.

The three tracks use the same probe samples, layer, module aggregation and
functional-rank implementation.  This is intentionally a remeasurement, not a
post-hoc transformation of previously rounded rank counts.
"""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
from typing import Any
import torch
import cycle09_r4_campaign as campaign
import cycle09_r4_common as c4
import cycle09_stage3_followup_common as common

ROOT=common.scoped_run('H2_white')
EPS=(0.01,0.025,0.05,0.10)


def qwen_context():
    import cycle09_q1_geometry as q1
    import cycle09_block3_qwen_probe_geometry as geom
    import cycle09_stage3_tpk as tpk
    q1.configure()
    def tokenizer():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(common.AUTODL/'model/Qwen/Qwen3-4B-Base'), local_files_only=True, trust_remote_code=True)
    return {'probes':tuple(geom.ALL_PROBES),
            'load':lambda arm,step,device: tpk.load_qwen_checkpoint(arm,step,device),'tokenizer':tokenizer,
            'samples':lambda probe,tok: geom.samples_for(probe,tok,factor_only=False),
            'profile':lambda mod,samples,layer,device: campaign.collect_profile(mod,samples,[layer],device,keep_factors=False,keep_residual_samples=False,factor_layers=(),forward_batch_size=8,max_batch_tokens=16384,early_stop=True),
            'base':common.AUTODL/'model/Qwen/Qwen3-4B-Base'}


def llama_context():
    import cycle09_block3_common as b3
    import cycle09_llama_geometry as geom
    import cycle09_llama_model_export as export
    return {'probes':tuple(geom.PROBE_NAMES),
            'load':lambda arm,step,device: (common.load_llama_checkpoint(arm,step,device), {'checkpoint_materialization':'peft_adapter_merged_in_memory'}),
            'tokenizer':b3.load_llama_tokenizer,'samples':lambda probe,tok: geom.prepare_samples(tok,probe,0),
            'profile':lambda mod,samples,layer,device: geom.profile_model(mod,samples,[layer],device,keep_sample_means=False,forward_batch_size=8,max_batch_tokens=16384),
            'base':b3.LLAMA_STUDENT}


def ranks(matrix: torch.Tensor)->dict[float,int]:
    sigma=torch.linalg.svdvals(matrix).double().cpu().tolist()
    return {eps:c4.functional_rank(sigma,eps) for eps in EPS}


def run(args:argparse.Namespace)->dict[str,Any]:
    if args.family=='qwen3_4b': context=qwen_context()
    else: context=llama_context()
    arms=[item.strip() for item in args.arms.split(',') if item.strip()]
    steps=[int(item) for item in args.steps.split(',') if item.strip()]
    probes=[item.strip() for item in args.probes.split(',') if item.strip()] if args.probes else list(context['probes'])
    unknown=set(probes).difference(context['probes'])
    if unknown: raise ValueError(f'unknown probes {unknown}')
    root=ROOT; output=root/f'T_WHITE_{args.family}.csv'; manifest=root/f'T_WHITE_{args.family}_manifest.json'
    tok=context['tokenizer'](); base_model,base_provenance=context['load']('opd',0,args.device)
    rows=[]; sources=[]
    try:
      base_cache={}
      for probe in probes:
        samples=context['samples'](probe,tok)
        if args.sample_limit:
          samples=samples[:args.sample_limit]
        if not samples:
          raise RuntimeError(f'no samples for {probe}')
        base_profile=context['profile'](base_model,samples,args.layer,args.device)
        base_scales=campaign.scaling_by_group(base_profile,[args.layer],args.device)
        base_cache[probe]=(samples,base_profile,base_scales)
      baseline_ranks={}
      for probe,(_,_,base_scales) in base_cache.items():
       for module in common.MODULES:
        group=c4.MODULE_TO_GROUP[module]
        base_weight=campaign.module_at(base_model,args.layer,module).weight.detach().to(device=args.device,dtype=torch.float32)
        baseline_ranks[(probe,module,'weight_only')]=ranks(base_weight)
        baseline_ranks[(probe,module,'fixed_S_D0')]=ranks(base_weight@base_scales[args.layer][group])
        del base_weight
      for arm in arms:
       for step in steps:
        if step==0 and arm != arms[0]: continue
        current,current_provenance=(base_model,base_provenance) if step==0 else context['load'](arm,step,args.device)
        try:
         weight_rank_cache={}
         for probe in probes:
          samples,base_profile,base_scales=base_cache[probe]
          current_profile=base_profile if step==0 else context['profile'](current,samples,args.layer,args.device)
          current_scales=base_scales if step==0 else campaign.scaling_by_group(current_profile,[args.layer],args.device)
          for module in common.MODULES:
           group=c4.MODULE_TO_GROUP[module]
           weight=campaign.module_at(current,args.layer,module).weight.detach().to(device=args.device,dtype=torch.float32)
           if module not in weight_rank_cache:
            weight_rank_cache[module]=ranks(weight)
           matrices={'weight_only':weight_rank_cache[module],
                     'fixed_S_D0':ranks(weight@base_scales[args.layer][group]),
                     'per_checkpoint_S_Dt':ranks(weight@current_scales[args.layer][group])}
           for track,current_r in matrices.items():
            base_key='weight_only' if track=='weight_only' else 'fixed_S_D0'
            base_r=baseline_ranks[(probe,module,base_key)]
            for eps in EPS: rows.append({'family':args.family,'arm':arm if step else 'base','step':step,'probe':probe,'layer':args.layer,'module':module,'epsilon':eps,'track':track,'r_epsilon':current_r[eps],'base_r_epsilon':base_r[eps],'delta_from_base':current_r[eps]-base_r[eps],'normalization':'window token mean -> sample window mean -> sample equal mean','checkpoint_materialization':current_provenance.get('checkpoint_materialization'),'native_source':current_provenance.get('native_source','adapter_BA')})
           del weight
          if step: current_scales.clear(); del current_profile
          torch.cuda.empty_cache()
        finally:
         if current is not base_model: campaign.unload_model(current)
      baseline_ranks.clear()
      for _,profile,scales in base_cache.values(): scales.clear(); del profile
    finally:
      campaign.unload_model(base_model); gc.collect(); torch.cuda.empty_cache()
    common.atomic_csv(output,rows)
    result={'schema_version':2,'status':'complete','task':'T-WHITE activation-conditioning ablation','family':args.family,'arms':arms,'steps':steps,'layer':args.layer,'probes':probes,'sample_limit':args.sample_limit or None,'tracks':['weight_only','fixed_S_D0','per_checkpoint_S_Dt'],'qwen_checkpoint_policy':'saved_merged_bf16 where retained; otherwise adapter merge quantized to the retained BF16 base','llama_runtime_policy':'PEFT adapter merged in memory; no persistent merged checkpoint required','output':common.artifact(output),'rows':len(rows),'created_utc':common.utc_now()}
    common.atomic_json(manifest,result); return result

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--family',choices=('qwen3_4b','llama3_2_3b'),required=True); parser.add_argument('--arms',required=True); parser.add_argument('--steps',required=True); parser.add_argument('--layer',type=int,required=True); parser.add_argument('--probes',default=''); parser.add_argument('--sample-limit',type=int,default=0); parser.add_argument('--device',default='cuda:0')
 print(json.dumps(run(parser.parse_args()),indent=2))
