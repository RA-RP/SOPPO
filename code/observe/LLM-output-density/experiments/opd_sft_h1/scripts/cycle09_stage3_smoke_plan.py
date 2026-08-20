#!/usr/bin/env python3
"""Render or explicitly execute the minimal Stage3 smoke matrix.

No waiting, GPU allocation, or subprocess launch happens unless --execute is
provided.  This is intentionally separate from the formal pipeline.
"""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
import cycle09_stage3_followup_common as c
PYTHON=str(c.DENSITY_PYTHON); S=c.SCRIPT_DIR; ROOT=c.RUN_ROOT/'smoke_plan'; PLAN=ROOT/'stage3_smoke_plan.json'; RESULT=ROOT/'stage3_smoke_result.json'

def unit(name:str,gpu:str,script:str,*args:str)->dict:
 return {'name':name,'gpu':gpu,'command':[PYTHON,str(S/script),*args]}

def plan()->dict:
 return {'schema_version':1,'status':'planned','created_utc':c.utc_now(),'policy':{'requires_user_execute':True,'requires_q1_handoff_complete':True,'formal_pipeline_not_started':True},'units':[
  unit('static_tpk','cpu','cycle09_stage3_tpk.py','--phase','preflight','--family','llama3_2_3b','--arms','opd,offkd','--steps','20'),
  unit('tpk_one_module','gpu0','cycle09_stage3_tpk.py','--phase','run','--family','llama3_2_3b','--arms','opd,offkd','--steps','20','--layers','14','--modules','self_attn.q_proj','--fractions','0.05','--device','cuda:0'),
  unit('white_one_probe','gpu1','cycle09_stage3_twhite.py','--family','llama3_2_3b','--arms','opd,offkd','--steps','20','--layer','14','--probes','E_ood','--device','cuda:0'),
  unit('sub_one_probe','gpu0','cycle09_stage3_tsub.py','--family','llama3_2_3b','--arms','opd,offkd','--steps','20','--layer','14','--probes','E_ood','--rank-fraction','0.05','--device','cuda:0'),
  unit('probe_contract','cpu','cycle09_stage3_probe_core.py','--families','qwen3_4b,llama3_2_3b','--phase','prepare'),
 ],'expected_wall_minutes':{'static':10,'core_gpu':45,'optional_frozen_self':40}}

def q1_ready()->bool:
 return c.read_json(c.MINI/'qwen_alpha05_stage_b_320_handoff_manifest.json',{}).get('status')=='complete'

def execute(payload:dict)->dict:
 if not q1_ready(): raise RuntimeError('Q1 H0 handoff is not complete; smoke is intentionally blocked')
 rows=[]
 for item in payload['units']:
  env=os.environ.copy()
  if item['gpu']=='gpu0':env['CUDA_VISIBLE_DEVICES']='0'
  if item['gpu']=='gpu1':env['CUDA_VISIBLE_DEVICES']='1'
  result=subprocess.run(item['command'],cwd=c.REPO,env=env)
  rows.append({'name':item['name'],'returncode':result.returncode})
  if result.returncode:break
 out={'schema_version':1,'status':'complete' if all(x['returncode']==0 for x in rows) else 'failed','units':rows,'created_utc':c.utc_now()};c.atomic_json(RESULT,out);return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-plan',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args();x=plan();
 if a.write_plan:c.atomic_json(PLAN,x)
 print(json.dumps(execute(x) if a.execute else x,indent=2))
