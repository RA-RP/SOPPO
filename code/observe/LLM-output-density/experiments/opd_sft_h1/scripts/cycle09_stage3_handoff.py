#!/usr/bin/env python3
"""Immutable raw-readout handoff writer for Stage3 gates H0--H6."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import cycle09_stage3_followup_common as c

SCOPE_ROOT = c.scope_root()
EVOLUTION = c.REPO/'mypaper/code/code_evolution.md'
GATE_INPUTS = {
 'H0': [c.MINI/'qwen_alpha05_stage_b_320_handoff_manifest.json'],
 'H1': [SCOPE_ROOT/'H1_resync/H1_resync_manifest.json', SCOPE_ROOT/'H1_support/T_SUPPORT_manifest.json'],
 'H2': [SCOPE_ROOT/'H2_tpk/T_PK_qwen3_4b_manifest.json', SCOPE_ROOT/'H2_tpk/T_PK_llama3_2_3b_manifest.json', SCOPE_ROOT/'H2_white/T_WHITE_qwen3_4b_manifest.json', SCOPE_ROOT/'H2_white/T_WHITE_llama3_2_3b_manifest.json'],
 'H3': [SCOPE_ROOT/'H2_sub/T_SUB_qwen3_4b_manifest.json', SCOPE_ROOT/'H2_sub/T_SUB_llama3_2_3b_manifest.json', SCOPE_ROOT/'H2_probe_core/PROBE_CORE_manifest.json'],
 'H4': [SCOPE_ROOT/'H4_increment/TINC_manifest.json', SCOPE_ROOT/'H4_increment/TBEH_manifest.json', SCOPE_ROOT/'H4_increment/M3_audit_manifest.json'],
 'H5': [SCOPE_ROOT/'H5_frozen_self/FROZEN_SELF_manifest.json'],
 'H6': [SCOPE_ROOT/'H6_mediator/MEDIATOR_manifest.json'],
}


def table(path: Path, max_rows: int=40) -> str:
    if path.suffix != '.csv' or not path.is_file(): return ''
    frame=pd.read_csv(path).head(max_rows)
    return '\n\n' + c.markdown_table(frame, list(frame.columns)) + '\n'


def declared_artifacts(value):
    """Yield manifest-declared files without rehashing large nested artifacts."""
    if isinstance(value, dict):
        declared=value.get('path')
        if declared and Path(str(declared)).is_file():
            path=Path(str(declared))
            yield {
                'path':str(path),
                'bytes':int(value.get('bytes',path.stat().st_size)),
                'sha256':value.get('sha256'),
                'complete':bool(value.get('complete',path.stat().st_size>0)),
            }
            return
        for child in value.values():
            yield from declared_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from declared_artifacts(child)


def run(gate: str) -> dict:
    inputs=GATE_INPUTS[gate]; missing=[str(path) for path in inputs if not path.is_file()]
    if missing: raise FileNotFoundError(f'{gate} incomplete inputs: {missing}')
    inventory=[]; readouts=[]; declared_missing=[]
    for manifest in inputs:
        payload=c.read_json(manifest,{})
        if not str(payload.get('status', '')).startswith('complete'):
            raise RuntimeError(f'{manifest} not complete')
        for cell in payload.get('missing_cells', []):
            declared_missing.append({'source_manifest':str(manifest), **cell})
        inventory.append(c.artifact(manifest))
        for artifact in declared_artifacts(payload):
            inventory.append(artifact)
            artifact_path=Path(artifact['path'])
            if artifact_path.suffix=='.csv':
                readouts.append(artifact_path)
    # Deduplicate while retaining source order.
    dedup={item['path']: item for item in inventory}; inventory=list(dedup.values())
    readouts=list(dict.fromkeys(readouts))
    markdown=[f'# Stage3 {gate} Raw Theory Handoff','',f'Generated UTC: {c.utc_now()}','', '## Artifact Inventory','', c.markdown_table(pd.DataFrame(inventory), ['path','bytes','sha256','complete']), '', '## Raw Readouts (first 40 rows per CSV)','']
    for csv_path in readouts:
        markdown += [f'### {csv_path.name}', table(csv_path)]
    markdown += ['## Declared Missing Cells','']
    if declared_missing:
        markdown += [c.markdown_table(pd.DataFrame(declared_missing), list(pd.DataFrame(declared_missing).columns)), '']
    else:
        markdown += ['None.', '']
    markdown += ['## Provenance','', '- This package records raw artifacts and coverage only; it contains no interpretation or adjudication.', '- Input hashes are in the JSON manifest.']
    package_root = c.MINI if c.scope_root() == c.RUN_ROOT else c.scope_root() / 'handoffs'
    output=package_root/f'mini_stage3_{gate}_theory_handoff.md'; manifest=package_root/f'stage3_{gate}_handoff_manifest.json'
    c.atomic_text(output,'\n'.join(markdown)+'\n')
    result={'schema_version':1,'status':'complete','gate':gate,'inputs':[c.artifact(path) for path in inputs], 'inventory':inventory,'missing_cells':declared_missing,'handoff':c.artifact(output),'created_utc':c.utc_now()}
    c.atomic_json(manifest,result)
    marker=f'<!-- cycle09-stage3-{gate.lower()}-handoff -->'
    existing=EVOLUTION.read_text(encoding='utf-8') if EVOLUTION.is_file() else ''
    if marker not in existing:
        entry=(
            f'\n---\n\n{marker}\n\n'
            f'## Cycle 09 Stage3 {gate} raw handoff\n\n'
            f'Completed raw-artifact package: `{output}`. Machine-readable manifest: '
            f'`{manifest}`. The package records coverage, declared missing cells, and '
            'provenance without interpretation.\n'
        )
        c.atomic_text(EVOLUTION,existing.rstrip()+entry+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--gate', choices=tuple(GATE_INPUTS), required=True)
    print(json.dumps(run(parser.parse_args().gate),indent=2))
