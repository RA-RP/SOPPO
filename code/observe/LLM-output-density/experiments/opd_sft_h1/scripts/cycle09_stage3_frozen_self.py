#!/usr/bin/env python3
"""H5 frozenSelf0-KD immutable training contract and approved launcher."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
import cycle09_stage3_followup_common as c
ROOT=c.scoped_run('H5_frozen_self'); CONTRACT=ROOT/'FROZEN_SELF_contract.json'; MANIFEST=ROOT/'FROZEN_SELF_manifest.json'
TRAIN_SMOKE=ROOT/'single_gpu_train_smoke_manifest.json'

def require_gate(path:Path)->dict:
 payload=c.read_json(path,{})
 if payload.get('status')!='complete' or payload.get('gate')!='H4': raise RuntimeError('frozenSelf0-KD requires completed H4 handoff')
 return payload

def prepare(gate:Path)->dict:
 require_gate(gate)
 contract={'schema_version':1,'status':'prepared','task':'frozenSelf0-KD total-effect control','created_utc':c.utc_now(),'h4_gate':c.artifact(gate),
 'design':{'student_step0':str(c.AUTODL/'model/Meta/modelscope/Llama-3.2-3B'),'teacher':str(c.AUTODL/'model/Meta/modelscope/Meta-Llama-3.1-8B-Instruct'),'prompt_pool':str(c.AUTODL/'cycle08_opd_trajectory/data/opd_prompts_5k.parquet'),'support':'one step0 student rollout pass, frozen thereafter','targets':'same fixed-teacher top-32 dense KL as OPD','optimizer_schedule':'same as Llama OPD','lora':'r32 alpha64 all-linear','checkpoint_grid':[0,5,20,40,80,160,320],'primary_estimand':'natural total effect OPD minus frozenSelf0-KD; do not length/EOS/repetition match'},
 'required_outputs':['frozen_step0_rollouts.jsonl','teacher_top32_targets','training_manifest.json','landmark_behavior.csv','landmark_geometry.csv','frozen_self_total_effect.csv'],
 'execution_schedule':{
   'rollout':'two fixed contiguous data-parallel shards: GPU0 orders 0-2498, GPU1 orders 2499-4998; applies to pass1 and teacher RAW top32 pass2',
   'training':'GPU0 only; frozen per-token RAW top32 labels keep forward_kl_topk enabled while suppressing the unused live-teacher server',
   'post_training':'export, then two balanced GPU workers: each runs three behavior checkpoint cells and three geometry checkpoint cells; each mix has two long-cap behavior cells, one short-cap cell, two three-layer geometry cells, and one headline-only geometry cell',
   'auto_shutdown':False,
 },
 'materializer':str(c.SCRIPT_DIR/'cycle09_frozen_self_parallel_materialize.py'),'runner':str(c.SCRIPT_DIR/'run_cycle09_frozen_self.sh')}
 c.atomic_json(CONTRACT,contract);return contract

def materialize() -> dict:
 result=subprocess.run([str(c.DENSITY_PYTHON),str(c.SCRIPT_DIR/'cycle09_frozen_self_parallel_materialize.py'),'--phase','all'],cwd=c.REPO)
 if result.returncode: raise RuntimeError(f'frozen-self materializer failed rc={result.returncode}')
 payload=c.read_json(ROOT/'frozen_support_manifest.json',{})
 if payload.get('status')!='complete': raise RuntimeError('materializer returned without complete manifest')
 return payload

def train_smoke() -> dict:
 existing=c.read_json(TRAIN_SMOKE,{})
 if existing.get('status')=='complete': return existing
 smoke_root=ROOT/'single_gpu_training_smoke'
 if smoke_root.exists():
  # Preserve every failed smoke for audit, then select a fresh immutable retry root.
  retry=1
  while (ROOT/f'single_gpu_training_smoke_retry{retry:02d}').exists(): retry+=1
  smoke_root=ROOT/f'single_gpu_training_smoke_retry{retry:02d}'
 smoke_root.mkdir(parents=True)
 for name in ('data','frozen_store'):
  (smoke_root/name).symlink_to(ROOT/name,target_is_directory=True)
 env=os.environ.copy();env.update({
  'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1','FROZEN_SELF_ROOT':str(smoke_root),
  'FROZEN_SELF_TRAIN_STEPS':'1','FROZEN_SELF_TRAIN_EPOCHS':'1','FROZEN_SELF_TRAIN_BATCH':'16',
  'FROZEN_SELF_SAVE_FREQ':'1','FROZEN_SELF_RESUME_MODE':'disable',
  'FROZEN_SELF_ENABLE_RETENTION_PRUNER':'0',
  'FROZEN_SELF_EXPERIMENT_NAME':'frozenSelf0_KD_single_gpu_smoke',
 })
 log=ROOT/'logs'/'single_gpu_train_smoke.log';log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a',encoding='utf-8') as handle:
  result=subprocess.run(['bash',str(c.SCRIPT_DIR/'run_cycle09_frozen_self.sh'),'--contract',str(CONTRACT)],cwd=c.REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
 if result.returncode: raise RuntimeError(f'H5 single-GPU train smoke failed rc={result.returncode}; log={log}')
 actor=smoke_root/'checkpoints'/'global_step_1'/'actor'
 shards=sorted(actor.glob('model_world_size_*_rank_*.pt'))
 if len(shards)!=1: raise RuntimeError(f'H5 single-GPU smoke did not retain step1 actor: {actor}')
 payload={'schema_version':1,'status':'complete','task':'H5 frozen-self precomputed-target single-GPU train smoke','gpu_visible_devices':'0','steps':1,'batch_size':16,'smoke_root':str(smoke_root),'model_runtime':'/root/autodl-tmp/cycle09_block3/llama_opd/model/student_runtime (same base weights; tokenizer chat template installed)','checkpoint':c.artifact(shards[0]),'log':c.artifact(log),'created_utc':c.utc_now()}
 c.atomic_json(TRAIN_SMOKE,payload);return payload

def train(gate:Path)->dict:
 contract=prepare(gate); runner=Path(contract['runner'])
 existing=c.read_json(ROOT/'training_manifest.json',{})
 if existing.get('status')=='complete': return existing
 materializer=ROOT/'frozen_support_manifest.json'
 if not materializer.is_file() or c.read_json(materializer,{}).get('status')!='complete': materialize()
 if not runner.is_file(): raise FileNotFoundError(f'frozen-self runner is not installed: {runner}')
 train_smoke()
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONUNBUFFERED']='1'
 result=subprocess.run(['bash',str(runner),'--contract',str(CONTRACT)],cwd=c.REPO,env=env)
 if result.returncode: raise RuntimeError(f'frozen-self runner failed rc={result.returncode}')
 checkpoints=[]
 for step in contract['design']['checkpoint_grid'][1:]:
  actor=ROOT/'checkpoints'/f'global_step_{step}'/'actor'
  shards=sorted(actor.glob('model_world_size_*_rank_*.pt'))
  if len(shards)!=1: raise RuntimeError(f'frozen-self checkpoint missing at step={step}: {actor}')
  checkpoints.append(c.artifact(shards[0]))
 payload={'schema_version':1,'status':'complete','task':'frozenSelf0-KD training','contract':c.artifact(CONTRACT),'support':c.artifact(materializer),'checkpoints':checkpoints,'created_utc':c.utc_now()}
 c.atomic_json(ROOT/'training_manifest.json',payload);return payload

def postprocess(gate:Path)->dict:
 require_gate(gate)
 training=c.read_json(ROOT/'training_manifest.json',{})
 if training.get('status')!='complete': raise RuntimeError('frozen-self postprocess requires complete training_manifest.json')
 script=c.SCRIPT_DIR/'cycle09_stage3_frozen_self_postprocess.py'
 # Export is a shared prerequisite.  The later cells write disjoint arm/step
 # paths, so we mix behavior and geometry on both GPUs by their real cost rather
 # than pinning an entire measurement family to one card.
 result=subprocess.run([str(c.DENSITY_PYTHON),str(script),'--phase','export','--device','cpu'],cwd=c.REPO)
 if result.returncode: raise RuntimeError(f'frozen-self export failed rc={result.returncode}')
 plans={
  '0':'behavior:0,behavior:5,geometry:5,behavior:40,geometry:40,behavior:160,geometry:80',
  '1':'geometry_reference,behavior:20,geometry:20,behavior:80,geometry:160,behavior:320,geometry:320',
 }
 schedule=ROOT/'H5_postprocess_schedule.json'
 c.atomic_json(schedule,{'schema_version':1,'status':'running','task':'H5 balanced postprocess schedule','plans':plans,'rationale':'Each GPU receives three behavior and three nonzero geometry cells; both receive two 16k-cap behavior cells, one 4k-cap behavior cell, two 3-layer geometry cells, and one L14-only geometry cell.','created_utc':c.utc_now()})
 jobs=[]
 for gpu,plan in plans.items():
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;env['PYTHONUNBUFFERED']='1'
  log=ROOT/'logs'/f'postprocess_balanced_gpu{gpu}.log';log.parent.mkdir(parents=True,exist_ok=True)
  handle=log.open('a',encoding='utf-8')
  child=subprocess.Popen([str(c.DENSITY_PYTHON),str(script),'--phase','worker','--device','cuda:0','--plan',plan],cwd=c.REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
  jobs.append((gpu,plan,child,handle,log))
 failures=[]
 for gpu,plan,child,handle,log in jobs:
  rc=child.wait();handle.close()
  if rc: failures.append(f'gpu={gpu} plan={plan} rc={rc}; log={log}')
 if failures:
  c.atomic_json(schedule,{'schema_version':1,'status':'failed','task':'H5 balanced postprocess schedule','plans':plans,'failures':failures,'created_utc':c.utc_now()})
  raise RuntimeError('; '.join(failures))
 for phase in ('behavior_finalize','geometry_finalize'):
  result=subprocess.run([str(c.DENSITY_PYTHON),str(script),'--phase',phase,'--device','cpu'],cwd=c.REPO)
  if result.returncode: raise RuntimeError(f'frozen-self {phase} failed rc={result.returncode}')
 c.atomic_json(schedule,{'schema_version':1,'status':'complete','task':'H5 balanced postprocess schedule','plans':plans,'rationale':'Each GPU receives three behavior and three nonzero geometry cells; both receive two 16k-cap behavior cells, one 4k-cap behavior cell, two 3-layer geometry cells, and one L14-only geometry cell.','created_utc':c.utc_now()})
 result=subprocess.run([str(c.DENSITY_PYTHON),str(script),'--phase','total_effect','--device','cpu'],cwd=c.REPO)
 if result.returncode: raise RuntimeError(f'frozen-self total-effect summary failed rc={result.returncode}')
 payload=c.read_json(MANIFEST,{})
 if payload.get('status')!='complete': raise RuntimeError('postprocess exited without complete frozen-self manifest')
 return payload

def launch(gate:Path)->dict:
 train(gate);return postprocess(gate)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('prepare','materialize','smoke','train','postprocess','all'),required=True);p.add_argument('--gate-file',type=Path,required=True);a=p.parse_args();
 if a.phase=='prepare': value=prepare(a.gate_file)
 elif a.phase=='materialize': require_gate(a.gate_file); value=materialize()
 elif a.phase=='smoke': require_gate(a.gate_file); value=train_smoke()
 elif a.phase=='train': value=train(a.gate_file)
 elif a.phase=='postprocess': value=postprocess(a.gate_file)
 else: value=launch(a.gate_file)
 print(json.dumps(value,indent=2))
