# coding: utf8
import torch
import torch.nn as nn
from pathlib import Path


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

    model_path = Path(model_id)
    is_adapter = (model_path / "adapter_config.json").is_file()
    tokenizer_source = model_id
    if is_adapter:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(model_id, local_files_only=True)
        tokenizer_source = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_source,
        device_map="cpu",
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if is_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            model_id,
            is_trainable=False,
            local_files_only=True,
        ).merge_and_unload(safe_merge=True)
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
