# coding: utf8
import torch
import torch.nn as nn
from tqdm import tqdm

from .model_utils import find_layers


class _LayerWrapper(nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            module = self._modules.get("module")
            if module is not None:
                try:
                    return getattr(module, name)
                except AttributeError:
                    pass
            raise exc


def _is_llama3_style(model_name: str) -> bool:
    name = model_name.lower()
    return ("llama-3" in name) or ("llama-2" in name) or ("qwen" in name)


def convert_llama3_kwargs_to_llama1(kwargs_llama3):
    """
    Keep close to original SVDLLM.py behavior, but with safer fallbacks.
    """
    position_ids = kwargs_llama3.get("position_ids", None)

    if kwargs_llama3.get("attention_mask", None) is not None:
        attention_mask = kwargs_llama3["attention_mask"]
    else:
        if position_ids is None:
            raise KeyError("position_ids is required to build causal mask")
        seq_len = position_ids.shape[1]
        device = position_ids.device
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float16, device=device))
        attention_mask = (1.0 - mask) * -65504.0
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    kwargs_llama1 = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_value": None,
        "use_cache": kwargs_llama3.get("use_cache", True),
        "output_attentions": kwargs_llama3.get("output_attentions", False),
        "position_embeddings": kwargs_llama3.get("position_embeddings", None),
    }
    return kwargs_llama1


def _as_profile_mat(profile):
    if isinstance(profile, dict) and "profiling_mat" in profile:
        return profile["profiling_mat"]
    if isinstance(profile, dict) and "sqrt_mat" in profile:
        return profile["sqrt_mat"]
    return profile


def _gram_to_svd_sqrt(gram: torch.Tensor, singular_floor: float = 0.0) -> torch.Tensor:
    """Return R = U * sqrt(S), where sym(gram) ~= U S Vh."""
    gram = gram.to(dtype=torch.float64)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    u, s, _ = torch.linalg.svd(gram, full_matrices=False)
    s = torch.clamp(s, min=float(singular_floor))
    return u * torch.sqrt(s).unsqueeze(0)


def _gram_to_svdllm_scaling_diag_matrix(
    gram: torch.Tensor,
    cholesky_jitter: float = 1e-4,
    singular_floor: float = 0.0,
) -> torch.Tensor:
    """
    Match SVDLLM's profiling factorization:
    try Cholesky(G), then add (-min_eig + jitter) I and retry.
    If that still fails, use eigh as a final stability fallback.
    """
    raw_scaling_diag_matrix = gram.to(dtype=torch.float64)
    try:
        return torch.linalg.cholesky(raw_scaling_diag_matrix)
    except Exception:
        eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
        min_eig = eigenvalues[0]
        eye = torch.eye(
            raw_scaling_diag_matrix.shape[0],
            dtype=raw_scaling_diag_matrix.dtype,
            device=raw_scaling_diag_matrix.device,
        )
        raw_scaling_diag_matrix = raw_scaling_diag_matrix + (-min_eig + float(cholesky_jitter)) * eye

    try:
        return torch.linalg.cholesky(raw_scaling_diag_matrix)
    except Exception:
        sym_gram = 0.5 * (raw_scaling_diag_matrix + raw_scaling_diag_matrix.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(sym_gram)
        floor = max(float(singular_floor), float(cholesky_jitter))
        eigenvalues = torch.clamp(eigenvalues, min=floor)
        return eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)


def _layer_index(layer_key) -> int:
    if isinstance(layer_key, str) and layer_key.startswith("layer_"):
        return int(layer_key.split("_", 1)[1])
    return int(layer_key)


def _ensure_svd_sqrt_profile_legacy(profile, singular_floor: float = 0.0) -> dict:
    """
    Accept a new SVD-sqrt profile, a raw profiling_mat, or a legacy gram_mat
    profile and return the SVD-sqrt wrapper used by this module.
    """
    if isinstance(profile, dict) and "profiling_mat" in profile:
        return profile

    if isinstance(profile, dict) and "sqrt_mat" in profile:
        return {
            "profiling_mat": profile["sqrt_mat"],
            "meta": dict(profile.get("meta", {}), factorization="svd_sqrt"),
        }

    if isinstance(profile, dict) and "gram_mat" in profile:
        profiling_mat = {}
        for layer_idx, layer_profile in profile["gram_mat"].items():
            profiling_mat[layer_idx] = {}
            for linear_name, gram in layer_profile.items():
                profiling_mat[layer_idx][linear_name] = _gram_to_svd_sqrt(
                    gram,
                    singular_floor=singular_floor,
                ).cpu()
        return {
            "profiling_mat": profiling_mat,
            "meta": dict(
                profile.get("meta", {}),
                source_profile="gram_mat",
                factorization="svd_sqrt",
            ),
        }

    return {
        "profiling_mat": profile,
        "meta": {"factorization": "svd_sqrt"},
    }


@torch.no_grad()
def profile_low_resource_gram(
    model_name,
    model,
    calib_loader,
    dev,
    output_kind="gram",
    singular_floor: float = 0.0,
):
    """
    Step1-style low-resource profiling.

    Output format:
    {
      "gram_mat" or "profiling_mat": {layer_idx: {linear_name: Tensor[d, d]}},
      "meta": {...}
    }
    """
    if len(calib_loader) == 0:
        raise ValueError("calib_loader is empty")
    if output_kind not in ("gram", "svd_sqrt"):
        raise ValueError(f"Unsupported output_kind: {output_kind}")

    dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = torch.device("cpu")

    # 1) Step1-style module placement
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    # Use CPU buffers to keep GPU memory low during replay.
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device="cpu",
    )
    outs = torch.zeros_like(inps)

    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings_cos": None,
        "position_embeddings_sin": None,
    }

    class Catcher(_LayerWrapper):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            cached_inp = inp.detach()
            if cached_inp.dim() == 3 and cached_inp.shape[0] == 1:
                cached_inp = cached_inp[0]
            inps[cache["i"]] = cached_inp.cpu()
            cache["i"] += 1

            if _is_llama3_style(model_name):
                kwargs = convert_llama3_kwargs_to_llama1(kwargs)
            _append_position_embeddings(cache, kwargs.get("position_embeddings", None))

            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.detach().cpu()
                if cache["attention_mask"] is None:
                    cache["attention_mask"] = attn_mask
                else:
                    cache["attention_mask"] = torch.cat((cache["attention_mask"], attn_mask), dim=0)

            if "opt" not in model_name:
                position_ids = kwargs.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids.detach().cpu()
                    if cache["position_ids"] is None:
                        cache["position_ids"] = position_ids
                    else:
                        cache["position_ids"] = torch.cat((cache["position_ids"], position_ids), dim=0)

            # Stop full forward after collecting first-layer input.
            raise ValueError

    # 2) Step1 profiling-A: capture first-layer input states.
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    # Restore modules and free memory.
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attention_masks = cache["attention_mask"]
    position_ids = cache["position_ids"] if "opt" not in model_name else None
    pos_cos = cache["position_embeddings_cos"]
    pos_sin = cache["position_embeddings_sin"]

    profile_mat = {}

    # 3) Step1 profiling-B: replay layer-by-layer and collect per-linear gram.
    for i in tqdm(range(len(layers)), desc="Profiling layers"):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        layer_profile = {}

        def hook(module, inputs, outputs):
            inp = inputs[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp).sum(dim=0).to(dtype=torch.float64)
            #多个batch的情况，batch内部直接累加求和，batch就是x_nsamples这类的
            module.scaling_gram += adds

        handles = []
        for linear_name in subset:
            in_features = subset[linear_name].weight.shape[1]
            subset[linear_name].scaling_gram = torch.zeros(
                (in_features, in_features),
                dtype=torch.float64,
                device=dev,
            )
            handles.append(subset[linear_name].register_forward_hook(hook))

        outs.zero_()
        for j in range(inps.shape[0]):
            hidden_in = inps[j].unsqueeze(0).to(dev)

            if pos_cos is not None and pos_sin is not None:
                layer_out = layer(
                    hidden_in,
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev),
                    position_ids=position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None,
                    position_embeddings=(
                        pos_cos[j].unsqueeze(0).to(dev),
                        pos_sin[j].unsqueeze(0).to(dev),
                    ),
                )[0]
            elif "opt" not in model_name:
                layer_out = layer(
                    hidden_in,
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None,
                    position_ids=position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None,
                )[0]
            else:
                layer_out = layer(
                    hidden_in,
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None,
                )[0]

            out_j = layer_out.detach()
            if out_j.dim() == 3 and out_j.shape[0] == 1:
                out_j = out_j[0]
            outs[j] = out_j.cpu()

        for h in handles:
            h.remove()

        for linear_name in subset:
            gram = subset[linear_name].scaling_gram
            if output_kind == "svd_sqrt":
                layer_profile[linear_name] = _gram_to_svd_sqrt(
                    gram,
                    singular_floor=singular_floor,
                ).cpu()
            else:
                layer_profile[linear_name] = gram.cpu()
            del subset[linear_name].scaling_gram

        profile_mat[i] = layer_profile
        layers[i] = layer.cpu()

        # next layer input = current layer output
        inps, outs = outs, inps

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matrix_key = "gram_mat" if output_kind == "gram" else "profiling_mat"
    return {
        matrix_key: profile_mat,
        "meta": {
            "model_name": model_name,
            "num_batches": int(len(calib_loader)),
            "seqlen": int(model.seqlen),
            "token_count": int(len(calib_loader) * model.seqlen),
            "device_used": str(dev),
            "profile_output": output_kind,
            "factorization": "svd_sqrt" if output_kind == "svd_sqrt" else "none",
        },
    }


@torch.no_grad()
def profile_low_resource_svd_sqrt(
    model_name,
    model,
    calib_loader,
    dev,
    singular_floor: float = 0.0,
):
    return profile_low_resource_gram(
        model_name=model_name,
        model=model,
        calib_loader=calib_loader,
        dev=dev,
        output_kind="svd_sqrt",
        singular_floor=singular_floor,
    )


@torch.no_grad()
def compute_whitened_svd_values(model_name, model, profile, dev) -> dict:
    """
    Compute the SVDLLM step1-style singular values:
      svdvals(W @ R), where R R^T approximates the activation Gram matrix.
    """
    dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = torch.device("cpu")

    if "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    profiling_mat = _as_profile_mat(profile)
    sigma_dict = {}

    for layer_idx in tqdm(sorted(profiling_mat.keys(), key=_layer_index), desc="Computing whitened SVD"):
        layer_number = _layer_index(layer_idx)
        layer = layers[layer_number]
        subset = find_layers(layer)
        layer_key = f"layer_{layer_number}"
        sigma_dict[layer_key] = {}

        for linear_name in sorted(set(subset.keys()) & set(profiling_mat[layer_idx].keys())):
            weight = subset[linear_name].weight.detach().to(dtype=torch.float64, device=dev)
            sqrt_mat = profiling_mat[layer_idx][linear_name].to(dtype=torch.float64, device=dev)
            whitened_weight = torch.matmul(weight, sqrt_mat)
            if torch.isnan(whitened_weight).any() or torch.isinf(whitened_weight).any():
                print(f"[Warn] Skipping {layer_key}.{linear_name} due to NaN/Inf.")
                del weight, sqrt_mat, whitened_weight
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            singular_values = torch.linalg.svdvals(whitened_weight)
            sigma_dict[layer_key][linear_name] = singular_values.cpu().tolist()

            del weight, sqrt_mat, whitened_weight, singular_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return sigma_dict


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


def _resolve_cache_device(dev: torch.device, activation_cache_device: str) -> torch.device:
    requested = str(activation_cache_device).strip().lower()
    if requested not in ("cuda", "cpu"):
        raise ValueError("activation_cache_device must be 'cuda' or 'cpu'")
    if requested == "cuda":
        if dev.type == "cuda" and torch.cuda.is_available():
            return dev
        print("[Warn] activation_cache_device='cuda' requested but CUDA is unavailable. Use CPU cache.")
    return torch.device("cpu")


def _profiling_mat_from_profile(profile):
    if isinstance(profile, dict) and "profiling_mat" in profile:
        return profile["profiling_mat"]
    if isinstance(profile, dict) and "sqrt_mat" in profile:
        return profile["sqrt_mat"]
    return profile


def _get_transformer_layers(model_name, model):
    if "opt" in model_name:
        return model.model.decoder.layers
    return model.model.layers


def _move_input_modules(model_name, model, device):
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    else:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)


def _append_cache_tensor(cache: dict, key: str, tensor: torch.Tensor):
    tensor = tensor.detach().cpu()
    if cache[key] is None:
        cache[key] = tensor
    else:
        cache[key] = torch.cat((cache[key], tensor), dim=0)


def _append_position_embeddings(cache: dict, position_embeddings):
    if position_embeddings is None:
        return
    cos, sin = position_embeddings
    _append_cache_tensor(cache, "position_embeddings_cos", cos)
    _append_cache_tensor(cache, "position_embeddings_sin", sin)


def _forward_transformer_layer(
    model_name,
    layer,
    hidden_in,
    state: dict,
    sample_idx: int,
    dev: torch.device,
):
    attention_masks = state["attention_mask"]
    position_ids = state["position_ids"] if "opt" not in model_name else None
    position_embeddings_coss = state["position_embeddings_cos"]
    position_embeddings_sins = state["position_embeddings_sin"]

    if position_embeddings_coss is not None and position_embeddings_sins is not None:
        return layer(
            hidden_in,
            attention_mask=attention_masks[sample_idx].unsqueeze(0).to(dev) if attention_masks is not None else None,
            position_ids=position_ids[sample_idx].unsqueeze(0).to(dev) if position_ids is not None else None,
            position_embeddings=(
                position_embeddings_coss[sample_idx].unsqueeze(0).to(dev),
                position_embeddings_sins[sample_idx].unsqueeze(0).to(dev),
            ),
        )[0]

    if "opt" not in model_name:
        return layer(
            hidden_in,
            attention_mask=attention_masks[sample_idx].unsqueeze(0).to(dev) if attention_masks is not None else None,
            position_ids=position_ids[sample_idx].unsqueeze(0).to(dev) if position_ids is not None else None,
        )[0]

    return layer(
        hidden_in,
        attention_mask=attention_masks[sample_idx].unsqueeze(0).to(dev) if attention_masks is not None else None,
    )[0]


def _capture_first_layer_inputs_for_group(
    model_name,
    model,
    calib_loaders_by_task: dict,
    dev: torch.device,
    cache_device: torch.device,
) -> dict:
    layers = _get_transformer_layers(model_name, model)
    _move_input_modules(model_name, model, dev)
    layers[0] = layers[0].to(dev)
    base_layer0 = layers[0]

    dtype = next(iter(model.parameters())).dtype
    states = {}

    class Catcher(_LayerWrapper):
        def __init__(self, module, inps, cache):
            super().__init__()
            self.module = module
            self.inps = inps
            self.cache = cache

        def forward(self, inp, **kwargs):
            cached_inp = inp.detach()
            if cached_inp.dim() == 3 and cached_inp.shape[0] == 1:
                cached_inp = cached_inp[0]
            self.inps[self.cache["i"]] = cached_inp.to(cache_device)
            self.cache["i"] += 1

            if _is_llama3_style(model_name):
                kwargs = convert_llama3_kwargs_to_llama1(kwargs)
            _append_position_embeddings(self.cache, kwargs.get("position_embeddings", None))

            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None:
                _append_cache_tensor(self.cache, "attention_mask", attention_mask)

            if "opt" not in model_name:
                position_ids = kwargs.get("position_ids", None)
                if position_ids is not None:
                    _append_cache_tensor(self.cache, "position_ids", position_ids)

            raise ValueError

    try:
        for task_name, calib_loader in calib_loaders_by_task.items():
            if len(calib_loader) == 0:
                raise ValueError(f"calib_loader is empty for task: {task_name}")

            inps = torch.zeros(
                (len(calib_loader), model.seqlen, model.config.hidden_size),
                dtype=dtype,
                device=cache_device,
            )
            outs = torch.zeros_like(inps)
            cache = {
                "i": 0,
                "attention_mask": None,
                "position_ids": None,
                "position_embeddings_cos": None,
                "position_embeddings_sin": None,
            }

            layers[0] = Catcher(base_layer0, inps, cache)
            for batch in tqdm(calib_loader, desc=f"Capture layer0 inputs [{task_name}]", leave=False):
                try:
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    model(**batch)
                except ValueError:
                    pass

            layers[0] = layers[0].module
            states[task_name] = {
                "inps": inps,
                "outs": outs,
                "attention_mask": cache["attention_mask"],
                "position_ids": cache["position_ids"] if "opt" not in model_name else None,
                "position_embeddings_cos": cache["position_embeddings_cos"],
                "position_embeddings_sin": cache["position_embeddings_sin"],
            }
            base_layer0 = layers[0]
    finally:
        if isinstance(layers[0], Catcher):
            layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        _move_input_modules(model_name, model, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return states


def _replay_layer_for_state(
    model_name,
    layer,
    state: dict,
    dev: torch.device,
    cache_device: torch.device,
    after_sample=None,
):
    inps = state["inps"]
    outs = state["outs"]
    outs.zero_()

    for sample_idx in range(inps.shape[0]):
        hidden_in = inps[sample_idx].unsqueeze(0).to(dev)
        layer_out = _forward_transformer_layer(
            model_name=model_name,
            layer=layer,
            hidden_in=hidden_in,
            state=state,
            sample_idx=sample_idx,
            dev=dev,
        )
        out_j = layer_out.detach()
        if out_j.dim() == 3 and out_j.shape[0] == 1:
            out_j = out_j[0]
        outs[sample_idx] = out_j.to(cache_device)
        if after_sample is not None:
            after_sample(sample_idx)

    state["inps"], state["outs"] = outs, inps


def _profile_layer_for_state(
    model_name,
    layer,
    state: dict,
    dev: torch.device,
    cache_device: torch.device,
    cholesky_jitter: float,
    singular_floor: float,
    sample_factor_callback=None,
):
    subset = find_layers(layer)
    # Qwen/Llama decoder blocks expose four distinct linear-input streams.
    # q/k/v and gate/up share an input, while o/down each consume a later stream.
    factor_group_by_module = {
        "self_attn.q_proj": "attn_qkv_input",
        "self_attn.k_proj": "attn_qkv_input",
        "self_attn.v_proj": "attn_qkv_input",
        "self_attn.o_proj": "attn_o_input",
        "mlp.gate_proj": "mlp_gate_up_input",
        "mlp.up_proj": "mlp_gate_up_input",
        "mlp.down_proj": "mlp_down_input",
    }
    expected_factor_groups = {
        factor_group_by_module[name]
        for name in subset
        if name in factor_group_by_module
    }
    module_name_by_id = {id(module): name for name, module in subset.items()}
    sample_factors = {}

    def hook(module, inputs, outputs):
        raw_inp = inputs[0].detach()
        inp = raw_inp.float()
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1, 2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum

        if sample_factor_callback is not None:
            module_name = module_name_by_id[id(module)]
            group = factor_group_by_module.get(module_name)
            if group is not None and group not in sample_factors:
                stored = raw_inp
                if stored.dim() == 3 and stored.shape[0] == 1:
                    stored = stored[0]
                elif stored.dim() != 2:
                    raise ValueError(
                        "sample factor capture expects a [1, tokens, hidden] or [tokens, hidden] input; "
                        f"got shape={tuple(stored.shape)} for {module_name}"
                    )
                # Store the model's native activation values. Bootstrap reconstruction
                # casts them to fp32 before forming each sample Gram, matching the path above.
                sample_factors[group] = stored.cpu().contiguous()

    def emit_sample_factors(sample_idx: int) -> None:
        if sample_factor_callback is None:
            return
        missing = expected_factor_groups.difference(sample_factors)
        if missing:
            raise RuntimeError(
                "sample factor capture missed groups "
                f"{sorted(missing)} at sample {sample_idx}"
            )
        sample_factor_callback(sample_idx, dict(sample_factors))
        sample_factors.clear()

    handles = []
    try:
        for name in subset:
            in_features = subset[name].weight.shape[1]
            subset[name].raw_scaling_diag_matrix = torch.zeros(
                (in_features, in_features),
                dtype=torch.float32,
                device=dev,
            )
            handles.append(subset[name].register_forward_hook(hook))

        _replay_layer_for_state(
            model_name=model_name,
            layer=layer,
            state=state,
            dev=dev,
            cache_device=cache_device,
            after_sample=emit_sample_factors if sample_factor_callback is not None else None,
        )
    finally:
        for handle in handles:
            handle.remove()

    layer_profile = {}
    for name in subset:
        raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix
        scaling_diag_matrix = _gram_to_svdllm_scaling_diag_matrix(
            raw_scaling_diag_matrix,
            cholesky_jitter=cholesky_jitter,
            singular_floor=singular_floor,
        )
        layer_profile[name] = scaling_diag_matrix.cpu()
        del subset[name].raw_scaling_diag_matrix

    return layer_profile


def ensure_svd_sqrt_profile(
    profile,
    singular_floor: float = 0.0,
    cholesky_jitter: float = 1e-4,
) -> dict:
    """
    Return a bare scaling_diag_matrix profiling_mat dict.

    New files are saved as {layer_idx: {linear_name: scaling_diag_matrix}}.
    Historical wrappers and legacy raw gram profiles are accepted for loading.
    """
    if isinstance(profile, dict) and "gram_mat" in profile:
        profiling_mat = {}
        for layer_idx, layer_profile in profile["gram_mat"].items():
            profiling_mat[layer_idx] = {}
            for linear_name, raw_scaling_diag_matrix in layer_profile.items():
                profiling_mat[layer_idx][linear_name] = _gram_to_svdllm_scaling_diag_matrix(
                    raw_scaling_diag_matrix,
                    cholesky_jitter=cholesky_jitter,
                    singular_floor=singular_floor,
                ).cpu()
        return profiling_mat
    return _profiling_mat_from_profile(profile)


@torch.no_grad()
def profile_svdllm_low_resource(
    model_name,
    model,
    calib_loader,
    dev,
    singular_floor: float = 0.0,
    activation_cache_device: str = "cuda",
    cholesky_jitter: float = 1e-4,
):
    """
    SVDLLM step1-style low-resource profiling.

    Returns a bare profiling_mat:
      {layer_idx: {linear_name: scaling_diag_matrix}}
    where scaling_diag_matrix follows SVDLLM's Cholesky(+jitter) path.
    """
    if len(calib_loader) == 0:
        raise ValueError("calib_loader is empty")

    dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = torch.device("cpu")
    cache_device = _resolve_cache_device(dev, activation_cache_device)

    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=cache_device,
    )
    outs = torch.zeros_like(inps)

    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings_cos": None,
        "position_embeddings_sin": None,
    }

    class Catcher(_LayerWrapper):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            cached_inp = inp.detach()
            if cached_inp.dim() == 3 and cached_inp.shape[0] == 1:
                cached_inp = cached_inp[0]
            inps[cache["i"]] = cached_inp.to(cache_device)
            cache["i"] += 1

            if _is_llama3_style(model_name):
                kwargs = convert_llama3_kwargs_to_llama1(kwargs)
            _append_position_embeddings(cache, kwargs.get("position_embeddings", None))

            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.detach().cpu()
                if cache["attention_mask"] is None:
                    cache["attention_mask"] = attention_mask
                else:
                    cache["attention_mask"] = torch.cat((cache["attention_mask"], attention_mask), dim=0)

            if "opt" not in model_name:
                position_ids = kwargs.get("position_ids", None)
                if position_ids is not None:
                    position_ids = position_ids.detach().cpu()
                    if cache["position_ids"] is None:
                        cache["position_ids"] = position_ids
                    else:
                        cache["position_ids"] = torch.cat((cache["position_ids"], position_ids), dim=0)

            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attention_masks = cache["attention_mask"]
    position_ids = cache["position_ids"] if "opt" not in model_name else None
    position_embeddings_coss = cache["position_embeddings_cos"]
    position_embeddings_sins = cache["position_embeddings_sin"]

    profiling_mat = {}
    for i in tqdm(range(len(layers)), desc="Profiling layers"):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        def hook(module, inputs, outputs):
            inp = inputs[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.raw_scaling_diag_matrix += adds_sum

        handles = []
        for name in subset:
            in_features = subset[name].weight.shape[1]
            subset[name].raw_scaling_diag_matrix = torch.zeros(
                (in_features, in_features),
                dtype=torch.float32,
                device=dev,
            )
            handles.append(subset[name].register_forward_hook(hook))

        outs.zero_()
        for j in range(inps.shape[0]):
            hidden_in = inps[j].unsqueeze(0).to(dev)
            if position_embeddings_coss is not None and position_embeddings_sins is not None:
                layer_out = layer(
                    hidden_in,
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None,
                    position_ids=position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None,
                    position_embeddings=(
                        position_embeddings_coss[j].unsqueeze(0).to(dev),
                        position_embeddings_sins[j].unsqueeze(0).to(dev),
                    ),
                )[0]
            elif "opt" not in model_name:
                layer_out = layer(
                    hidden_in,
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None,
                    position_ids=position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None,
                )[0]
            else:
                layer_out = layer(
                    hidden_in,
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None,
                )[0]

            out_j = layer_out.detach()
            if out_j.dim() == 3 and out_j.shape[0] == 1:
                out_j = out_j[0]
            outs[j] = out_j.to(cache_device)

        for handle in handles:
            handle.remove()

        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix
            scaling_diag_matrix = _gram_to_svdllm_scaling_diag_matrix(
                raw_scaling_diag_matrix,
                cholesky_jitter=cholesky_jitter,
                singular_floor=singular_floor,
            )
            layer_profile[name] = scaling_diag_matrix.cpu()
            del subset[name].raw_scaling_diag_matrix

        profiling_mat[i] = layer_profile
        layers[i] = layer.cpu()
        inps, outs = outs, inps

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return profiling_mat


@torch.no_grad()
def profile_svdllm_single_layer_group(
    model_name,
    model,
    calib_loaders_by_task: dict,
    dev,
    target_layer: int,
    layer_gpu_chunk_size: int = 1,
    singular_floor: float = 0.0,
    activation_cache_device: str = "cuda",
    cholesky_jitter: float = 1e-4,
    sample_factor_callbacks_by_task: dict | None = None,
) -> dict:
    """
    Profile one transformer layer for one or more datasets.

    Returns:
      {task_name: {target_layer: {linear_name: scaling_diag_matrix}}}

    The replay schedule is layer-chunk outer, task inner: a chunk of
    transformer layers stays on the target device while every task in the
    group advances its rolling inps/outs through that chunk.
    """
    if not calib_loaders_by_task:
        raise ValueError("calib_loaders_by_task is empty")

    dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = torch.device("cpu")
    cache_device = _resolve_cache_device(dev, activation_cache_device)

    layers = _get_transformer_layers(model_name, model)
    target_layer = int(target_layer)
    if target_layer < 0 or target_layer >= len(layers):
        raise ValueError(
            f"target_layer={target_layer} out of range. "
            f"Valid range: 0..{len(layers) - 1}"
        )

    layer_gpu_chunk_size = int(layer_gpu_chunk_size)
    if layer_gpu_chunk_size <= 0:
        raise ValueError("layer_gpu_chunk_size must be > 0")

    states = _capture_first_layer_inputs_for_group(
        model_name=model_name,
        model=model,
        calib_loaders_by_task=calib_loaders_by_task,
        dev=dev,
        cache_device=cache_device,
    )

    profiling_by_task = {}
    layer_ranges = range(0, target_layer + 1, layer_gpu_chunk_size)
    for chunk_start in tqdm(layer_ranges, desc=f"Replay to layer_{target_layer}"):
        chunk_end = min(target_layer, chunk_start + layer_gpu_chunk_size - 1)
        loaded_layer_indices = []
        try:
            for layer_idx in range(chunk_start, chunk_end + 1):
                layers[layer_idx] = layers[layer_idx].to(dev)
                loaded_layer_indices.append(layer_idx)

            for layer_idx in range(chunk_start, chunk_end + 1):
                layer = layers[layer_idx]
                if layer_idx == target_layer:
                    for task_name, state in states.items():
                        print(f"[SingleLayer] Profiling {task_name} layer_{target_layer}")
                        profiling_by_task[task_name] = {
                            target_layer: _profile_layer_for_state(
                                model_name=model_name,
                                layer=layer,
                                state=state,
                                dev=dev,
                                cache_device=cache_device,
                                cholesky_jitter=cholesky_jitter,
                                singular_floor=singular_floor,
                                sample_factor_callback=(sample_factor_callbacks_by_task or {}).get(task_name),
                            )
                        }
                    break

                for state in states.values():
                    _replay_layer_for_state(
                        model_name=model_name,
                        layer=layer,
                        state=state,
                        dev=dev,
                        cache_device=cache_device,
                    )
        finally:
            for layer_idx in loaded_layer_indices:
                layers[layer_idx] = layers[layer_idx].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return profiling_by_task


@torch.no_grad()
def profile_svdllm_all_layers_group(
    model_name,
    model,
    calib_loaders_by_task: dict,
    dev,
    singular_floor: float = 0.0,
    activation_cache_device: str = "cuda",
    cholesky_jitter: float = 1e-4,
    after_profile=None,
    sample_factor_callbacks_by_task_and_layer: dict | None = None,
) -> dict:
    """Stream all layer/task profiles without retaining all Gram matrices."""
    if not calib_loaders_by_task:
        raise ValueError("calib_loaders_by_task is empty")

    dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = torch.device("cpu")
    cache_device = _resolve_cache_device(dev, activation_cache_device)
    layers = _get_transformer_layers(model_name, model)
    states = _capture_first_layer_inputs_for_group(
        model_name=model_name,
        model=model,
        calib_loaders_by_task=calib_loaders_by_task,
        dev=dev,
        cache_device=cache_device,
    )

    profiling_by_task = {}
    callbacks = sample_factor_callbacks_by_task_and_layer or {}
    for layer_idx in tqdm(range(len(layers)), desc="Profiling grouped layers"):
        layer = layers[layer_idx].to(dev)
        try:
            for task_name, state in states.items():
                callback = callbacks.get(task_name, {}).get(layer_idx)
                profile = _profile_layer_for_state(
                    model_name=model_name,
                    layer=layer,
                    state=state,
                    dev=dev,
                    cache_device=cache_device,
                    cholesky_jitter=cholesky_jitter,
                    singular_floor=singular_floor,
                    sample_factor_callback=callback,
                )
                if after_profile is None:
                    profiling_by_task.setdefault(task_name, {})[layer_idx] = profile
                else:
                    after_profile(task_name, layer_idx, {layer_idx: profile})
                    del profile
        finally:
            layers[layer_idx] = layer.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return profiling_by_task

@torch.no_grad()
def whitening(model_name, model, profiling_mat, dev, uv_dtype: str = "float32", return_uv: bool = True):
    """
    SVDLLM step1-style whitening SVD.

    Returns:
      sigma_dict: JSON-friendly singular values only.
      uv_dict: torch tensors for raw U/S/VT of W_scale = W @ profiling_mat.
    """
    dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA unavailable. Fallback to CPU.")
        dev = torch.device("cpu")
    uv_torch_dtype = _resolve_dtype(uv_dtype)
    profiling_mat = ensure_svd_sqrt_profile(profiling_mat)

    if "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    sigma_dict = {}
    uv_dict = {}
    print("Start SVD decomposition after whitening...")
    for layer_idx in tqdm(sorted(profiling_mat.keys(), key=_layer_index), desc="Whitening layers"):
        layer_number = _layer_index(layer_idx)
        layer = layers[layer_number]
        subset = find_layers(layer)
        layer_key = f"layer_{layer_number}"
        sigma_dict[layer_key] = {}
        uv_dict[layer_key] = {}

        for name in sorted(set(subset.keys()) & set(profiling_mat[layer_idx].keys())):
            print(f"starting whitening {name}")
            calc_dtype = torch.float64 if return_uv else torch.float32
            W = subset[name].weight.detach().to(dtype=calc_dtype, device=dev)
            scaling_diag_matrix = profiling_mat[layer_idx][name].to(dtype=calc_dtype, device=dev)
            W_scale = torch.matmul(W, scaling_diag_matrix)
            if torch.isnan(W_scale).any() or torch.isinf(W_scale).any():
                print(f"[Warn] Skipping {layer_key}.{name} due to NaN/Inf.")
                del W, scaling_diag_matrix, W_scale
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if return_uv:
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                sigma_dict[layer_key][name] = S.cpu().tolist()
                uv_dict[layer_key][name] = {
                    "U": U.cpu().to(dtype=uv_torch_dtype),
                    "S": S.cpu(),
                    "VT": VT.cpu().to(dtype=uv_torch_dtype),
                }
                del U, VT
            else:
                S = torch.linalg.svdvals(W_scale)
                sigma_dict[layer_key][name] = S.cpu().tolist()

            del W, scaling_diag_matrix, W_scale, S
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return sigma_dict, uv_dict
