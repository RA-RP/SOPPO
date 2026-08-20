"""Weight export utility for principalEvidence.py analysis.

Loads a HuggingFace merged model checkpoint and exports per-layer weight
matrices as .npy files in the format expected by AnalyseMat/principalEvidence.py.

Output filename format: model_layers_{layer}_{module}_weight.npy
  e.g. model_layers_6_self_attn.q_proj_weight.npy

Two modes:
  --mode flat   → write all files directly into --output_dir  (for base reference model)
  --mode nested → write into --output_dir/{data_size}/  (for finetuned models in npy_output_root)

Usage:
  python export_weights.py --model_path /path/to/model --output_dir /out/dir --mode flat
  python export_weights.py --model_path /path/to/model --output_dir /out/dir \\
      --mode nested --data_size 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

LAYERS = [6, 14, 22]

# Module attribute paths within a single transformer layer (Qwen3 / standard LLaMA-like arch).
# Each entry: (module_short_name, attribute_path_from_layer)
# attribute_path is joined by "." and resolved via getattr chain.
MODULES = [
    ("self_attn.q_proj", ["self_attn", "q_proj"]),
    ("self_attn.k_proj", ["self_attn", "k_proj"]),
    ("self_attn.v_proj", ["self_attn", "v_proj"]),
    ("self_attn.o_proj", ["self_attn", "o_proj"]),
    ("mlp.gate_proj",    ["mlp", "gate_proj"]),
    ("mlp.up_proj",      ["mlp", "up_proj"]),
    ("mlp.down_proj",    ["mlp", "down_proj"]),
]


def _get_layer(model, layer_idx: int):
    """Return the transformer layer object at index layer_idx."""
    # Standard HuggingFace Qwen3/LLaMA layout: model.model.layers[N]
    try:
        return model.model.layers[layer_idx]
    except AttributeError:
        pass
    # Fallback: model.layers[N]
    try:
        return model.layers[layer_idx]
    except AttributeError:
        raise AttributeError(
            f"Cannot locate layers list in model. "
            f"Tried model.model.layers[{layer_idx}] and model.layers[{layer_idx}]."
        )


def _get_weight(layer, attr_path: list[str]) -> torch.Tensor:
    """Resolve attr_path chain from layer and return .weight tensor."""
    obj = layer
    for attr in attr_path:
        obj = getattr(obj, attr)
    return obj.weight


def export_model_weights(
    model_path: str,
    output_dir: Path,
    *,
    layers: list[int] = LAYERS,
    modules: list[tuple[str, list[str]]] = MODULES,
) -> dict[str, str]:
    """Load model and export weight matrices as .npy files.

    Returns a dict {filename: str(path)} recording what was exported.
    Skips files that already exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    already_done = set()
    for layer in layers:
        for mod_name, _ in modules:
            fname = f"model_layers_{layer}_{mod_name}_weight.npy"
            if (output_dir / fname).exists():
                already_done.add(fname)

    total = len(layers) * len(modules)
    if len(already_done) == total:
        print(f"[SKIP] all {total} weight files exist in {output_dir}", flush=True)
        return {f: str(output_dir / f) for f in already_done}

    print(f"[EXPORT] loading model from {model_path} ...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    exported = {}
    for layer_idx in layers:
        layer = _get_layer(model, layer_idx)
        for mod_name, attr_path in modules:
            fname = f"model_layers_{layer_idx}_{mod_name}_weight.npy"
            out_path = output_dir / fname
            if out_path.exists():
                print(f"  [skip] {fname}", flush=True)
                exported[fname] = str(out_path)
                continue
            try:
                w = _get_weight(layer, attr_path)
                arr = w.detach().to(torch.float32).cpu().numpy()
                np.save(str(out_path), arr)
                print(f"  [save] {fname}  shape={arr.shape}", flush=True)
                exported[fname] = str(out_path)
            except AttributeError as exc:
                print(f"  [warn] {fname}: {exc}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[EXPORT] done — {len(exported)}/{total} files in {output_dir}", flush=True)
    return exported


def main():
    ap = argparse.ArgumentParser(description="Export layer weight matrices as .npy for principalEvidence")
    ap.add_argument("--model_path", required=True, help="Path to merged HuggingFace model checkpoint")
    ap.add_argument("--output_dir", required=True, help="Directory to write .npy files (base dir)")
    ap.add_argument("--mode", choices=["flat", "nested"], default="flat",
                    help="flat: write directly into output_dir; "
                         "nested: write into output_dir/{data_size}/")
    ap.add_argument("--data_size", type=str, default=None,
                    help="Required when mode=nested. The integer data-size label (e.g. '512').")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS,
                    help="Layer indices to export")
    args = ap.parse_args()

    out_base = Path(args.output_dir)
    if args.mode == "nested":
        if not args.data_size:
            ap.error("--data_size is required when --mode nested")
        out_dir = out_base / args.data_size
    else:
        out_dir = out_base

    modules = MODULES
    export_model_weights(args.model_path, out_dir, layers=args.layers, modules=modules)


if __name__ == "__main__":
    main()
