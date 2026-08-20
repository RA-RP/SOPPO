#!/usr/bin/env python3
"""H1 CPU reconciliation and checkpoint-wise A/G DiD tables for Cycle09."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import cycle09_stage3_followup_common as c

ROOT = c.scoped_run('H1_resync')
CANONICAL = ROOT / 'canonical_headline_geometry.csv'
INVENTORY = ROOT / 'source_inventory.json'
DID = ROOT / 'T_DID_geometry.csv'
MANIFEST = ROOT / 'H1_resync_manifest.json'


def inventory() -> dict:
    frame, sources = c.canonical_geometry()
    c.atomic_json(INVENTORY, {'created_utc': c.utc_now(), 'sources': sources,
                              'canonical_rows': int(len(frame)),
                              'headline': {'epsilon': .05, 'track': 'per_checkpoint',
                                           'layers': c.HEADLINE_LAYER, 'modules': list(c.MODULES)}})
    if frame.empty:
        raise RuntimeError('no readable headline geometry source files')
    c.atomic_csv(CANONICAL, frame.to_dict('records'), list(frame.columns))
    return {'status': 'complete', 'canonical': c.artifact(CANONICAL),
            'inventory': c.artifact(INVENTORY), 'rows': len(frame)}


def did() -> dict:
    if not CANONICAL.is_file():
        inventory()
    frame = pd.read_csv(CANONICAL)
    frame = frame[frame.apply(lambda row: int(row['layer']) == c.HEADLINE_LAYER[row['family']], axis=1)]
    grouped = frame.groupby(['family', 'arm', 'step', 'probe'], as_index=False).agg(
        r_epsilon=('r_epsilon', 'mean'), A_D=('delta_from_base', 'mean'),
        n_modules=('module', 'nunique'), source_count=('source', 'nunique'))
    general = grouped[grouped['probe'].isin(('E_general', 'general'))][
        ['family', 'arm', 'step', 'A_D']].rename(columns={'A_D': 'A_general'})
    result = grouped.merge(general, how='left', on=['family', 'arm', 'step'])
    result['G_D'] = result['A_D'] - result['A_general']
    contrast_rows = []
    for family, part in result.groupby('family', sort=True):
        opd = part[part['arm'] == 'opd'].set_index(['step', 'probe'])
        offkd = part[part['arm'] == 'offkd'].set_index(['step', 'probe'])
        common = opd.index.intersection(offkd.index)
        for step, probe in common:
            left, right = opd.loc[(step, probe)], offkd.loc[(step, probe)]
            contrast_rows.append({'family': family, 'row_type': 'opd_minus_offkd', 'arm': 'opd-offkd',
                                  'step': int(step), 'probe': probe,
                                  'A_D': float(left['A_D'] - right['A_D']),
                                  'G_D': float(left['G_D'] - right['G_D']) if pd.notna(left['G_D']) and pd.notna(right['G_D']) else None,
                                  'r_epsilon': None, 'n_modules': int(min(left['n_modules'], right['n_modules'])),
                                  'source_count': int(min(left['source_count'], right['source_count']))})
    result['row_type'] = 'arm_trajectory'
    result = pd.concat([result, pd.DataFrame(contrast_rows)], ignore_index=True, sort=False)
    result = result.sort_values(['family', 'row_type', 'probe', 'step', 'arm'], kind='stable')
    c.atomic_csv(DID, result.to_dict('records'), list(result.columns))
    return {'status': 'complete', 'output': c.artifact(DID), 'rows': len(result),
            'note': 'A_D primary and G_D secondary; no interpolation or endpoint-only substitution'}


def finalize() -> dict:
    if not DID.is_file():
        did()
    payload = {'schema_version': 1, 'status': 'complete', 'task': 'H1 RESYNC + T-DID raw reconciliation',
               'canonical': c.artifact(CANONICAL), 'inventory': c.artifact(INVENTORY),
               'did': c.artifact(DID), 'created_utc': c.utc_now()}
    c.atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('inventory', 'did', 'finalize'), required=True)
    args = parser.parse_args()
    result = inventory() if args.phase == 'inventory' else did() if args.phase == 'did' else finalize()
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
