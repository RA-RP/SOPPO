#!/usr/bin/env python3
"""cycle07_gpu_worker.py — isolated GPU work for Cycle 07, run as a subprocess.

The orchestrator (run_cycle07.py) must hold ZERO GPU memory so its eval/geometry
subprocesses can claim the full card. Any heavy in-process GPU op (training,
probe rollout) therefore runs here as a separate process; when it exits the OS
reclaims all of its GPU memory. Driven by a JSON config file.

Subcommands:
  train      — one SFTTrainer run on the base model, saving LoRA adapters at a step grid
  rollout-x  — generate chat-template completions for the GetSlice X probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _step_label(step: int) -> str:
    return f"step_{step:03d}"


def cmd_train(cfg: dict) -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    import inspect

    def _filter(cls, kwargs):
        sig = inspect.signature(cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in set(sig.parameters)}

    base_model = cfg["base_model"]
    ckpt_root = Path(cfg["ckpt_root"])
    grid = set(cfg["grid"])

    class GridSaveCallback(TrainerCallback):
        def _save(self, step, model):
            d = ckpt_root / _step_label(step)
            if (d / "adapter_config.json").exists():
                return
            model.save_pretrained(str(d))
            print(f"[CKPT] saved adapter at step {step} -> {d}", flush=True)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in grid:
                self._save(state.global_step, kwargs["model"])

        def on_train_end(self, args, state, control, **kwargs):
            self._save(state.global_step, kwargs["model"])

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = load_dataset("json", data_files=cfg["train_jsonl"], split="train")

    max_len = cfg["max_len"]
    sft_kwargs = {
        "output_dir": str(ckpt_root / "_trainer"),
        "do_train": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": cfg["eff_batch"],
        "learning_rate": cfg["lr"],
        "num_train_epochs": cfg["epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "bf16": True,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "seed": cfg["seed"],
        "max_seq_length": max_len,
        "max_length": max_len,
        "dataset_text_field": "text",
        "packing": False,
        "remove_unused_columns": True,
    }
    # Fixed step count overrides epochs when max_steps > 0 (HF Trainer cycles the dataloader
    # past one epoch as needed). Used for the eff_batch-16 / 632-step trajectory.
    if int(cfg.get("max_steps", -1)) > 0:
        sft_kwargs["max_steps"] = int(cfg["max_steps"])
    args = SFTConfig(**_filter(SFTConfig, sft_kwargs))
    peft_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(model=base_model, args=args, train_dataset=train_ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.add_callback(GridSaveCallback())
    print(f"[TRAIN] starting; grid={sorted(grid)}", flush=True)
    trainer.train()
    print("[TRAIN] done", flush=True)


def cmd_rollout_x(cfg: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = cfg["base_model"]
    instr = cfg["instr"]
    rows = [json.loads(l) for l in open(cfg["prompts_jsonl"]) if l.strip()][: cfg["n_probe"]]
    prompts = [r["question"] for r in rows]

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()

    out_rows = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p + instr}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        comp = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out_rows.append({"output": {"text": f"{p}\n{comp}"}})

    out_path = Path(cfg["out_jsonl"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[X PROBE] {len(out_rows)} -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subcommand", choices=["train", "rollout-x"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.subcommand == "train":
        cmd_train(cfg)
    else:
        cmd_rollout_x(cfg)


if __name__ == "__main__":
    main()
