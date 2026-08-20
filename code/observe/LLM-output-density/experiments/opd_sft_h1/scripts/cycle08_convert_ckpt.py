#!/usr/bin/env python3
"""Cycle 08 — convert a verl OPD checkpoint into a merged HF model for eval/geometry.

verl (FSDP backend) saves each step as:
    {default_local_dir}/global_step_{N}/actor/
        model_world_size_{W}_rank_{r}.pt   (sharded FSDP weights: base + LoRA)
        optim_*.pt, extra_state_*.pt
        huggingface/                       (config + tokenizer)
        fsdp_config.json
        lora_train_meta.json               (LoRA r/alpha/task_type, written by verl)

Cycle 07's eval (runner_think) and geometry (GetSlice / export_weights) consume a
*full merged HF model directory* (base+LoRA merged), exactly like cycle07's
`_merge_checkpoint`. This script produces that, per step, by wrapping
`verl.model_merger merge --backend fsdp`, then normalizing the output:

  - if model_merger emits a PEFT adapter (adapter_config.json) -> merge onto BASE
    via peft.merge_and_unload() to get a full model;
  - if it already emits a full HF model -> use as-is.

IMPORTANT (run environment): model_merger needs the `verl` env (torch2.9 + verl).
Run this with the verl python:
    /root/autodl-tmp/envs/verl/bin/python cycle08_convert_ckpt.py --step 40 ...
The orchestrator (density env) shells out to that interpreter.

NOTE: the exact model_merger LoRA output shape (adapter vs merged-full) is confirmed
empirically at the Stage 3 (2x96G) execution smoke — the first real saved checkpoint.
Both branches are handled here so whichever verl emits, the converter normalizes it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE_MODEL = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
VERL_PY = Path("/root/autodl-tmp/envs/verl/bin/python")


def verl_step_actor_dir(ckpt_root: Path, step: int) -> Path:
    """Locate the verl actor checkpoint dir for a global step."""
    d = ckpt_root / f"global_step_{step}" / "actor"
    if not d.exists():
        raise FileNotFoundError(
            f"verl checkpoint not found: {d}\n"
            f"  (expected layout: {ckpt_root}/global_step_<N>/actor/). "
            f"Existing: {sorted(p.name for p in ckpt_root.glob('global_step_*'))}")
    return d


def _looks_like_full_model(d: Path) -> bool:
    has_cfg = (d / "config.json").exists()
    has_weights = any(d.glob("*.safetensors")) or any(d.glob("pytorch_model*.bin"))
    return has_cfg and has_weights


def _find_adapter_dir(d: Path) -> Path | None:
    for cand in [d, *[p for p in d.iterdir() if p.is_dir()]] if d.exists() else []:
        if (cand / "adapter_config.json").exists():
            return cand
    return None


def _merge_adapter_onto_base(base: Path, adapter: Path, out: Path) -> Path:
    """base + LoRA adapter -> full merged HF model at `out` (verl env: transformers+peft)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    # tokenizer: prefer the checkpoint's, fall back to base
    tok_src = adapter if (adapter / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=True).save_pretrained(str(out))
    return out


def convert_step(ckpt_root: Path, step: int, out_dir: Path, *, base: Path = BASE_MODEL,
                 keep_intermediate: bool = False) -> Path:
    """Convert one verl step -> a full merged HF model dir at `out_dir`. Idempotent."""
    if _looks_like_full_model(out_dir):
        print(f"[convert] step {step}: merged model already present -> {out_dir}", flush=True)
        return out_dir

    actor = verl_step_actor_dir(ckpt_root, step)
    raw = out_dir.parent / f"{out_dir.name}__merger_raw"
    raw.mkdir(parents=True, exist_ok=True)

    cmd = [str(VERL_PY), "-m", "verl.model_merger", "merge",
           "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(raw)]
    print(f"[convert] step {step}: model_merger -> {raw}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"model_merger failed rc={r.returncode} for step {step} ({actor})")

    adapter = _find_adapter_dir(raw)
    if adapter is not None:
        print(f"[convert] step {step}: found PEFT adapter at {adapter}; merging onto base", flush=True)
        _merge_adapter_onto_base(base, adapter, out_dir)
    elif _looks_like_full_model(raw):
        print(f"[convert] step {step}: model_merger produced a full model; moving into place", flush=True)
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        shutil.move(str(raw), str(out_dir))
        raw = None  # moved
    else:
        raise RuntimeError(
            f"model_merger output at {raw} is neither a PEFT adapter nor a full HF model; "
            f"contents={sorted(p.name for p in raw.iterdir())}")

    if raw is not None and not keep_intermediate:
        shutil.rmtree(raw, ignore_errors=True)
    if not _looks_like_full_model(out_dir):
        raise RuntimeError(f"conversion finished but {out_dir} is not a loadable HF model")
    print(f"[convert] step {step}: done -> {out_dir}", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Convert verl OPD checkpoint -> merged HF model")
    ap.add_argument("--ckpt-root", type=Path, required=True,
                    help="verl trainer.default_local_dir (holds global_step_*/actor)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="output merged HF model dir")
    ap.add_argument("--base", type=Path, default=BASE_MODEL)
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()
    convert_step(args.ckpt_root, args.step, args.out_dir,
                 base=args.base, keep_intermediate=args.keep_intermediate)


if __name__ == "__main__":
    main()
