#!/usr/bin/env python3
"""Frozen-self dataset and agent loop; compatible with the established Q1 store ABI."""
from __future__ import annotations
import torch
from cycle09_q1_mixture_runtime import FrozenExternalAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
class FrozenSelfDataset(RLHFDataset):
 def __getitem__(self,item):
  row=super().__getitem__(item)
  # The schedule provenance label belongs to Q1 only.  Do not forward it into
  # the global Q1 source-balanced loss patch for this single-source H5 arm.
  row.pop('support_source_id',None)
  row.pop('support_source',None)
  row['external_record_index']=torch.tensor(int(row.pop('external_record_index')),dtype=torch.int64)
  return row
