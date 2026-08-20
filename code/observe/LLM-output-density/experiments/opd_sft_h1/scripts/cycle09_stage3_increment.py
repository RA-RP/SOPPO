#!/usr/bin/env python3
"""Frozen-split T-INC/T-BEH analysis after H2/H3 inputs are complete."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_absolute_error, r2_score, roc_auc_score
import cycle09_stage3_followup_common as c

ROOT=c.scoped_run('H4_increment')
SCOPE_ROOT=c.scope_root()
CANONICAL=SCOPE_ROOT/'H1_resync/canonical_headline_geometry.csv'
DID=SCOPE_ROOT/'H1_resync/T_DID_geometry.csv'
TPK_Q=SCOPE_ROOT/'H2_tpk/T_PK_qwen3_4b.csv'
WHITE_Q=SCOPE_ROOT/'H2_white/T_WHITE_qwen3_4b.csv'
PROBE_CORE=SCOPE_ROOT/'H2_probe_core/PROBE_CORE_r_epsilon.csv'
TINC_SPLITS=ROOT/'TINC_frozen_splits.json'; TINC_OOF=ROOT/'TINC_trackA_oof.csv'; TINC_CMP=ROOT/'TINC_trackA_nested_comparison.csv'; TBEH_CMP=ROOT/'TBEH_model_comparison.csv'; TBEH_OOF=ROOT/'TBEH_oof_predictions.csv'; TINC_WHITE=ROOT/'TINC_whitening_ablation.csv'; TINC_REPORT=ROOT/'TINC_report.md'; TINC_MANIFEST=ROOT/'TINC_manifest.json'; TBEH_MANIFEST=ROOT/'TBEH_manifest.json'


def require(path:Path)->Path:
 if not path.is_file(): raise FileNotFoundError(path)
 return path

def frozen_splits(steps:list[int],seed:int)->list[dict]:
 # Checkpoint-blocked deterministic folds, no pseudo-replication over modules/probes.
 return [{'fold_id':f'leave_step_{step}','heldout_steps':[step],'train_steps':[x for x in steps if x!=step],'seed':seed} for step in sorted(steps)]

def ztrain(train:pd.Series,test:pd.Series)->tuple[np.ndarray,np.ndarray]:
 mean=float(train.mean()); scale=max(float(train.std(ddof=0)),1e-12); return ((train-mean)/scale).to_numpy(),((test-mean)/scale).to_numpy()

def track_a(frame:pd.DataFrame,splits:list[dict])->tuple[pd.DataFrame,pd.DataFrame]:
 data=frame[(frame.arm.isin(['opd','offkd']))&(frame.step>0)].copy(); data['target']=(data.arm=='opd').astype(int); rows=[]; summary=[]
 for probe,part in data.groupby('probe',sort=True):
  # One genuine arm x checkpoint observation: seven-module equal mean already occurs here.
  part=part.groupby(['arm','step'],as_index=False).agg(A_D=('A_D','mean'),p_k=('p_k','mean'),target=('target','first'))
  for fold in splits:
   train=part[part.step.isin(fold['train_steps'])]; test=part[part.step.isin(fold['heldout_steps'])]
   if train.target.nunique()<2 or test.empty: continue
   xp_tr,xp_te=ztrain(train.p_k,test.p_k); xr_tr,xr_te=ztrain(train.A_D,test.A_D)
   base=LogisticRegression(C=1.0,solver='liblinear',random_state=fold['seed']).fit(xp_tr[:,None],train.target)
   aug=LogisticRegression(C=1.0,solver='liblinear',random_state=fold['seed']).fit(np.c_[xp_tr,xr_tr],train.target)
   for model,name,x in ((base,'P',xp_te[:,None]),(aug,'P_plus_r',np.c_[xp_te,xr_te])):
    prob=model.predict_proba(x)[:,1]; pred=(prob>=.5).astype(int)
    for idx,row in test.reset_index(drop=True).iterrows(): rows.append({'probe':probe,'fold_id':fold['fold_id'],'step':int(row.step),'arm':row.arm,'target':int(row.target),'model':name,'probability':float(prob[idx]),'prediction':int(pred[idx])})
  pred=pd.DataFrame([row for row in rows if row['probe']==probe])
  for model,cell in pred.groupby('model'):
   summary.append({'probe':probe,'model':model,'n':len(cell),'log_loss':log_loss(cell.target,cell.probability,labels=[0,1]),'balanced_accuracy':balanced_accuracy_score(cell.target,cell.prediction),'roc_auc':roc_auc_score(cell.target,cell.probability) if cell.target.nunique()==2 else np.nan})
 return pd.DataFrame(rows),pd.DataFrame(summary)

def behavior_frame()->pd.DataFrame:
 # Inputs are frozen to existing formal behavior summaries; the exact mapping is
 # reported in the output manifest and missing mappings fail rather than impute.
 candidates=[c.MINI/'three_arm_full_trajectory.csv',c.MINI/'qwen_alpha05_behavior_keypoints.csv']
 frames=[pd.read_csv(path) for path in candidates if path.is_file()]
 if not frames: raise FileNotFoundError('no frozen behavior summary available')
 raw=pd.concat(frames,ignore_index=True,sort=False)
 mappings=[]
 for _,row in raw.iterrows():
  if 'math500_acc' in raw.columns and pd.notna(row.get('math500_acc')): mappings.append({'arm':row.get('arm'),'step':row.get('step'),'outcome':'math500_accuracy','value':row.get('math500_acc'),'probe':'E_math'})
  if 'ifeval_prompt_strict' in raw.columns and pd.notna(row.get('ifeval_prompt_strict')): mappings.append({'arm':row.get('arm'),'step':row.get('step'),'outcome':'ifeval_prompt_strict','value':row.get('ifeval_prompt_strict'),'probe':'E_if'})
 return pd.DataFrame(mappings)


def geometry_frame() -> pd.DataFrame:
 """Use H1 headline geometry plus the exact H2 landmarks without relabelling.

 H2's exact MATH500 probe supersedes a historical same-named 32-item probe
 wherever it is present.  No absent E_if/E_mmluPro cell is imputed.
 """
 historical=pd.read_csv(require(CANONICAL))
 historical=historical[(historical.family=='qwen3_4b')&(historical.layer==18)].copy()
 core=pd.read_csv(require(PROBE_CORE))
 core=core[(core.family=='qwen3_4b')&(core.layer==18)&(core.epsilon==.05)&(core.track=='per_checkpoint_S_Dt')].copy()
 core=core.rename(columns={'delta_from_base':'A_D'})
 if not core.empty:
  historical=historical[historical.probe!='E_math']
  core=core[['arm','step','probe','layer','module','A_D']]
  historical=historical.rename(columns={'delta_from_base':'A_D'})
  return pd.concat([historical[['arm','step','probe','layer','module','A_D']],core],ignore_index=True)
 return historical.rename(columns={'delta_from_base':'A_D'})[['arm','step','probe','layer','module','A_D']]

def track_b(geometry:pd.DataFrame,splits:list[dict],behavior:pd.DataFrame|None=None)->tuple[pd.DataFrame,pd.DataFrame]:
 behavior=behavior_frame() if behavior is None else behavior; joined=behavior.merge(geometry[['arm','step','probe','A_D','G_D','p_k']],on=['arm','step','probe'],how='inner'); rows=[]; summary=[]
 for (outcome,probe),part in joined.groupby(['outcome','probe'],sort=True):
  part=part[(part.step>0)&part.arm.isin(['opd','offkd','seqkd','sft'])].copy()
  for fold in splits:
   train=part[part.step.isin(fold['train_steps'])]; test=part[part.step.isin(fold['heldout_steps'])]
   if len(train)<8 or test.empty: continue
   # nuisance: progress plus arm dummies; objective is intentionally absent.
   nuisance=lambda x: np.c_[np.ones(len(x)),x.step/max(1,x.step.max()),pd.get_dummies(x.arm).reindex(columns=['offkd','opd','seqkd','sft'],fill_value=0).to_numpy()]
   ntr,nte=nuisance(train),nuisance(test); pk_tr,pk_te=ztrain(train.p_k,test.p_k); ad_tr,ad_te=ztrain(train.A_D,test.A_D)
   base=Ridge(alpha=1.0).fit(np.c_[ntr,pk_tr],train.value); aug=Ridge(alpha=1.0).fit(np.c_[ntr,pk_tr,ad_tr],train.value)
   for model,fit,x in (('N_plus_P',base,np.c_[nte,pk_te]),('N_plus_P_plus_A',aug,np.c_[nte,pk_te,ad_te])):
    predicted=fit.predict(x)
    for idx,row in test.reset_index(drop=True).iterrows(): rows.append({'outcome':outcome,'probe':probe,'fold_id':fold['fold_id'],'arm':row.arm,'step':int(row.step),'model':model,'observed':float(row.value),'predicted':float(predicted[idx]),'coverage_status':'eligible_oof'})
  out=pd.DataFrame([row for row in rows if row['outcome']==outcome and row['probe']==probe])
  if out.empty:
   # No prompt-level behavior cell is manufactured when its frozen geometry
   # grid has no matching checkpoint.  Keep both planned model labels so the
   # raw comparison table records the coverage boundary explicitly.
   for model in ('N_plus_P','N_plus_P_plus_A'):
    summary.append({'outcome':outcome,'probe':probe,'model':model,'n':0,'r2_oof':np.nan,'mae_oof':np.nan,'coverage_status':'no_eligible_oof_fold'})
   continue
  for model,cell in out.groupby('model'):
   summary.append({'outcome':outcome,'probe':probe,'model':model,'n':len(cell),'r2_oof':r2_score(cell.observed,cell.predicted),'mae_oof':mean_absolute_error(cell.observed,cell.predicted),'coverage_status':'eligible_oof'})
 return pd.DataFrame(rows),pd.DataFrame(summary)

def run(seed:int)->dict:
 geom=geometry_frame().groupby(['arm','step','probe'],as_index=False).agg(A_D=('A_D','mean')); did=pd.read_csv(require(DID)); pk=pd.read_csv(require(TPK_Q)); require(WHITE_Q)
 p=pk[(pk.layer==18)&(pk.rank_fraction==.05)].groupby(['arm','step'],as_index=False).agg(p_k=('p_k','mean'))
 merged=geom.merge(p,on=['arm','step'],how='inner'); merged=merged.merge(did[['family','row_type','arm','step','probe','G_D']].query("family=='qwen3_4b' and row_type=='arm_trajectory'")[['arm','step','probe','G_D']],on=['arm','step','probe'],how='left')
 steps=sorted(set(merged.step).difference({0})); splits=frozen_splits(steps,seed); c.atomic_json(TINC_SPLITS,{'schema_version':1,'seed':seed,'splits':splits})
 oof,cmp=track_a(merged,splits); beh_oof,beh_cmp=track_b(merged,splits)
 c.atomic_csv(TINC_OOF,oof.to_dict('records'));c.atomic_csv(TINC_CMP,cmp.to_dict('records'));c.atomic_csv(TBEH_OOF,beh_oof.to_dict('records'));c.atomic_csv(TBEH_CMP,beh_cmp.to_dict('records'))
 white=pd.read_csv(WHITE_Q); c.atomic_csv(TINC_WHITE,white.to_dict('records'))
 c.atomic_text(TINC_REPORT,'# T-INC/T-BEH Raw Report\n\nFrozen folds and raw OOF artifacts only; no interpretation.\n')
 manifest={'schema_version':2,'status':'complete','seed':seed,'inputs':[c.artifact(path) for path in (CANONICAL,DID,TPK_Q,WHITE_Q,PROBE_CORE)],'outputs':[c.artifact(path) for path in (TINC_SPLITS,TINC_OOF,TINC_CMP,TBEH_OOF,TBEH_CMP,TINC_WHITE,TINC_REPORT)],'geometry_policy':'H2 exact E_math replaces historical E_math where present; no missing probe cell is imputed','created_utc':c.utc_now()}
 c.atomic_json(TINC_MANIFEST,manifest);c.atomic_json(TBEH_MANIFEST,manifest|{'task':'T-BEH nested OOF'});return manifest

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('all',),required=True);parser.add_argument('--seed',type=int,required=True)
 print(json.dumps(run(parser.parse_args().seed),indent=2))
