"""
Model loading and utility functions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Tuple, Optional


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.float16,
    use_flash_attention: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load language model and tokenizer.

    Args:
        model_name_or_path: Path or HF model name (e.g., "Qwen/Qwen3-4B")
        device: Device to load model ('cuda' or 'cpu')
        torch_dtype: Model dtype (default fp16)
        use_flash_attention: Whether to use flash attention

    Returns:
        model: Loaded language model
        tokenizer: Loaded tokenizer
    """
    print(f"Loading model from: {model_name_or_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch_dtype,
        'device_map': device if device == 'auto' else None,
    }

    if use_flash_attention:
        model_kwargs['attn_implementation'] = 'flash_attention_2'

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    if device != 'auto' and device != 'cpu':
        model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, tokenizer


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Freeze all parameters of a model.

    Args:
        model: Model to freeze

    Returns:
        Frozen model
    """
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    print("Model frozen (all parameters set to requires_grad=False)")

    return model


def unfreeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unfreeze all parameters of a model.

    Args:
        model: Model to unfreeze

    Returns:
        Unfrozen model
    """
    for param in model.parameters():
        param.requires_grad = True

    model.train()
    print("Model unfrozen (all parameters set to requires_grad=True)")

    return model


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional = None,
    step: Optional[int] = None,
    config: Optional[dict] = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save checkpoint
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        step: Optional training step number
        config: Optional config to save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save optimizer and scheduler states
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_path / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), output_path / "scheduler.pt")

    # Save metadata
    metadata = {
        'step': step,
    }
    if config is not None:
        metadata['config'] = config

    torch.save(metadata, output_path / "metadata.pt")

    print(f"Checkpoint saved to {output_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.float16
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """
    Load model checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint
        device: Device to load model
        torch_dtype: Model dtype

    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        metadata: Metadata dictionary
    """
    checkpoint_path = Path(checkpoint_dir)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        str(checkpoint_path),
        device=device,
        torch_dtype=torch_dtype
    )

    # Load metadata if exists
    metadata_path = checkpoint_path / "metadata.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path, map_location='cpu')
    else:
        metadata = {}

    print(f"Checkpoint loaded from {checkpoint_dir}")
    if 'step' in metadata:
        print(f"Training step: {metadata['step']}")

    return model, tokenizer, metadata


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_billions': total / 1e9,
        'trainable_billions': trainable / 1e9
    }
