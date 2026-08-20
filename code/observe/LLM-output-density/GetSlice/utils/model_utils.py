# coding: utf8
import torch
import torch.nn as nn


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def get_model_from_huggingface(
    model_id: str,
    torch_dtype: str = "float16",
    trust_remote_code: bool = True,
    cache_dir: str = None,
):
    """Step1-style model loader (kept close to SVD-LLM-Fix logic)."""
    print("model_id:", model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype(torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    model.seqlen = 2048
    return model, tokenizer


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    """Recursively find all target layers in one transformer block."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name != "" else name1
        res.update(find_layers(child, layers=layers, name=child_name))
    return res
