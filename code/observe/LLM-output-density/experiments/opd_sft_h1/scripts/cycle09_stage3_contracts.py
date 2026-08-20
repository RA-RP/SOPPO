#!/usr/bin/env python3
"""Write immutable input contracts for Stage3 follow-up; never runs a model."""
from __future__ import annotations

import argparse
import json

import cycle09_stage3_followup_common as c


ROOT = c.scoped_run("contracts")


def write() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    support = {
        "schema_version": 2,
        "status": "template",
        "instruction": (
            "Populate only with pre-existing training-support artifacts; "
            "behavior/evaluation generations are forbidden substitutes."
        ),
        "required_cell_fields": ["family", "arm", "step", "path"],
        "cells": [],
        "missing_cells": [],
        "shared_source_groups": {},
        "metrics": [
            "response_tokens_mean",
            "eos_rate",
            "truncation_rate",
            "exact_duplicate_rate",
            "normalized_duplicate_rate",
            "near_duplicate_cluster_rate",
            "distinct_2",
            "distinct_4",
            "token_entropy",
            "sequence_entropy",
            "source_kl_mean",
            "source_loss_mean",
            "training_objective_at_step",
        ],
    }
    frozen = {
        "schema_version": 1,
        "status": "template",
        "instruction": (
            "Materialize exactly one step0 student rollout and fixed teacher "
            "top-32 targets before an authorized H5 launch."
        ),
        "prompt_order_source": str(
            c.AUTODL / "cycle08_opd_trajectory/data/opd_prompts_5k.parquet"
        ),
        "same_pre_treatment_controls": [
            "initial_student",
            "prompt_pool_order",
            "teacher",
            "top32_KL",
            "optimizer",
            "LoRA",
            "nominal_batch_size",
        ],
        "forbidden_primary_adjustments": [
            "length_match",
            "EOS_match",
            "repetition_match",
            "truncate_to_common_realized_distribution",
            "joint_post_treatment_reweight",
        ],
    }
    white = {
        "schema_version": 1,
        "status": "frozen",
        "tracks": ["weight_only", "fixed_S_D0", "per_checkpoint_S_Dt"],
        "headline": {
            "epsilon": 0.05,
            "module_aggregation": "seven_module_equal_mean",
            "normalization": "window token mean -> sample window mean -> sample equal mean",
        },
        "families": {
            "qwen3_4b": {"layer": 18},
            "llama3_2_3b": {"layer": 14},
        },
    }
    payloads = {
        "support_inputs.json": support,
        "frozen_self_inputs.json": frozen,
        "twhite_contract.json": white,
    }
    for name, payload in payloads.items():
        path = ROOT / name
        existing = c.read_json(path, {})
        populated_support = (
            name == "support_inputs.json"
            and isinstance(existing.get("cells"), list)
            and bool(existing["cells"])
            and str(existing.get("status", "")).startswith("frozen")
        )
        if not populated_support:
            c.atomic_json(path, payload)
    result = {
        "schema_version": 1,
        "status": "complete",
        "outputs": [c.artifact(ROOT / name) for name in payloads],
        "created_utc": c.utc_now(),
    }
    c.atomic_json(ROOT / "contracts_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write",), required=True)
    parser.parse_args()
    print(json.dumps(write(), indent=2))
