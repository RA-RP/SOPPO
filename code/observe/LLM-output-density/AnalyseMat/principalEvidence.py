import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


CSV_COLUMNS = [
    "Source",
    "DataSize",
    "Layer",
    "Module",
    "RankK",
    "PrincipalTopRatio",
    "PrincipalTopCount",
    "UpdateCount",
    "PrincipalCount",
    "OverlapCount",
    "UpdateDensity",
    "PrincipalDensity",
    "OverlapOverUpdate",
    "OverlapOverPrincipal",
    "RandomExpectedOverlap",
    "OverlapLift",
    "Jaccard",
    "UAngleMeanDeg",
    "UAngleMaxDeg",
    "UAngleMinDeg",
    "VAngleMeanDeg",
    "VAngleMaxDeg",
    "VAngleMinDeg",
]


def _task_filter(config):
    tasks = (config.get("analyse") or {}).get("tasks") or (config.get("analyseMat") or {}).get("tasks") or []
    allowed = set()
    for task in tasks:
        if isinstance(task, (list, tuple)) and len(task) >= 2:
            allowed.add((str(task[0]), str(task[1])))
        elif isinstance(task, dict):
            source = task.get("source") or task.get("dataset") or task.get("name")
            size = task.get("size") or task.get("max_samples") or task.get("DataSize")
            if source is not None and size is not None:
                allowed.add((str(source), str(size)))
    return allowed


def run_principal_evidence(config):
    analyse_cfg = config.get("analyse", {})
    related_cfg = analyse_cfg.get("related_work", {})
    if not related_cfg.get("enable", False):
        return None

    base_model_dir = Path(analyse_cfg["base_model_npy_dir"]).expanduser().resolve()
    finetuned_root = Path(analyse_cfg["npy_output_root"]).expanduser().resolve()
    output_root = _resolve_output_root(analyse_cfg, related_cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "details").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)

    target_layer = _parse_target_layer(related_cfg.get("target_layer"))
    target_modules = _normalize_modules(related_cfg.get("target_modules"))
    rank_k_default = int(related_cfg.get("principal_rank_k") or analyse_cfg.get("svd_topk", 50))
    top_ratio = float(related_cfg.get("principal_top_ratio", 0.01))
    top_count = related_cfg.get("principal_top_count")
    top_count = int(top_count) if top_count is not None else None
    update_mask_rule = str(related_cfg.get("update_mask_rule", "bf16")).lower()
    save_png = bool(related_cfg.get("save_png", True))
    plot_max_side = int(related_cfg.get("plot_max_side", 768))
    allowed_tasks = _task_filter(config)

    if update_mask_rule != "bf16":
        raise ValueError("AnalyseMat related_work currently supports update_mask_rule='bf16' only")
    if not base_model_dir.exists():
        raise FileNotFoundError(f"base_model_npy_dir does not exist: {base_model_dir}")
    if not finetuned_root.exists():
        raise FileNotFoundError(f"npy_output_root does not exist: {finetuned_root}")

    print("[AnalyseMat] running related-work principal evidence")
    print(f"[AnalyseMat] target_layer={target_layer}, output_root={output_root}")

    rows = []
    base_cache = {}
    for source_dir in sorted(path for path in finetuned_root.iterdir() if path.is_dir()):
        for size_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            if allowed_tasks and (source_dir.name, size_dir.name) not in allowed_tasks:
                continue
            data_size = _parse_data_size(size_dir.name)
            if data_size is None:
                continue
            layer_files = _target_layer_files(size_dir, target_layer, target_modules)
            for ft_path in tqdm(layer_files, desc=f"{source_dir.name}/{size_dir.name}/layer_{target_layer}"):
                base_path = base_model_dir / ft_path.name
                if not base_path.exists():
                    print(f"[AnalyseMat] warning: missing base weight: {base_path}")
                    continue

                module = _module_from_filename(ft_path.name, target_layer)
                try:
                    row = _analyse_one_weight(
                        source=source_dir.name,
                        data_size=data_size,
                        layer=target_layer,
                        module=module,
                        base_path=base_path,
                        finetuned_path=ft_path,
                        rank_k_default=rank_k_default,
                        top_ratio=top_ratio,
                        top_count=top_count,
                        output_root=output_root,
                        save_png=save_png,
                        plot_max_side=plot_max_side,
                        base_cache=base_cache,
                    )
                except RuntimeError as exc:
                    print(f"[AnalyseMat] warning: failed {source_dir.name}/{size_dir.name}/{ft_path.name}: {exc}")
                    continue
                if row is not None:
                    rows.append(row)

    csv_path = output_root / "principal_evidence.csv"
    _write_csv(csv_path, rows, replace_tasks=allowed_tasks)
    print(f"[AnalyseMat] saved related-work principal evidence rows={len(rows)} -> {csv_path}")
    return csv_path


def _analyse_one_weight(
    source,
    data_size,
    layer,
    module,
    base_path,
    finetuned_path,
    rank_k_default,
    top_ratio,
    top_count,
    output_root,
    save_png,
    plot_max_side,
    base_cache,
):
    W0 = torch.from_numpy(np.load(base_path)).to(dtype=torch.float32, device="cpu")
    Wp = torch.from_numpy(np.load(finetuned_path)).to(dtype=torch.float32, device="cpu")
    if W0.ndim != 2 or Wp.ndim != 2:
        return None
    if tuple(W0.shape) != tuple(Wp.shape):
        raise RuntimeError(f"shape mismatch: base={tuple(W0.shape)}, finetuned={tuple(Wp.shape)}")

    cache_key = str(base_path)
    if cache_key not in base_cache:
        rank_k = min(int(rank_k_default), int(min(W0.shape)))
        if rank_k <= 0:
            raise RuntimeError(f"invalid rank_k={rank_k}")
        U0, S0, Vh0 = torch.linalg.svd(W0, full_matrices=False)
        U0k = U0[:, :rank_k].contiguous()
        S0k = S0[:rank_k].contiguous()
        Vh0k = Vh0[:rank_k, :].contiguous()
        principal_mask, principal_count = _principal_mask_from_svd(U0k, S0k, Vh0k, top_ratio, top_count)
        base_cache[cache_key] = {
            "rank_k": rank_k,
            "U0k": U0k,
            "Vh0k": Vh0k,
            "principal_mask": principal_mask,
            "principal_count": principal_count,
        }
    cached = base_cache[cache_key]
    rank_k = cached["rank_k"]
    principal_mask = cached["principal_mask"]
    principal_count = cached["principal_count"]

    update_mask = W0.to(torch.bfloat16).ne(Wp.to(torch.bfloat16))
    overlap_mask = update_mask & principal_mask

    Up, _, Vhp = torch.linalg.svd(Wp, full_matrices=False)
    Upk = Up[:, :rank_k].contiguous()
    Vhpk = Vhp[:rank_k, :].contiguous()
    u_angles = _principal_angles_deg(cached["U0k"].T @ Upk)
    v_angles = _principal_angles_deg(cached["Vh0k"] @ Vhpk.T)

    total = int(update_mask.numel())
    update_count = int(update_mask.sum().item())
    overlap_count = int(overlap_mask.sum().item())
    union_count = update_count + principal_count - overlap_count
    expected_overlap = (update_count * principal_count / total) if total else 0.0
    row = {
        "Source": source,
        "DataSize": data_size,
        "Layer": int(layer),
        "Module": module,
        "RankK": int(rank_k),
        "PrincipalTopRatio": float(top_ratio),
        "PrincipalTopCount": int(principal_count),
        "UpdateCount": update_count,
        "PrincipalCount": int(principal_count),
        "OverlapCount": overlap_count,
        "UpdateDensity": _safe_div(update_count, total),
        "PrincipalDensity": _safe_div(principal_count, total),
        "OverlapOverUpdate": _safe_div(overlap_count, update_count),
        "OverlapOverPrincipal": _safe_div(overlap_count, principal_count),
        "RandomExpectedOverlap": expected_overlap,
        "OverlapLift": _safe_div(overlap_count, expected_overlap),
        "Jaccard": _safe_div(overlap_count, union_count),
        "UAngleMeanDeg": _mean(u_angles),
        "UAngleMaxDeg": _max(u_angles),
        "UAngleMinDeg": _min(u_angles),
        "VAngleMeanDeg": _mean(v_angles),
        "VAngleMaxDeg": _max(v_angles),
        "VAngleMinDeg": _min(v_angles),
    }

    details_dir = output_root / "details" / source / str(data_size)
    details_dir.mkdir(parents=True, exist_ok=True)
    detail_path = details_dir / f"layer_{layer}_{_safe_token(module)}.json"
    figure_path = None
    if save_png:
        figure_dir = output_root / "figures" / source / str(data_size)
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figure_dir / f"layer_{layer}_{_safe_token(module)}.png"
        _save_mask_plot(update_mask, principal_mask, overlap_mask, figure_path, plot_max_side, source, data_size, layer, module)

    detail = dict(row)
    detail.update({
        "Shape": list(W0.shape),
        "UpdateMaskRule": "bf16",
        "UAnglesDeg": _to_float_list(u_angles),
        "VAnglesDeg": _to_float_list(v_angles),
        "BasePath": str(base_path),
        "FinetunedPath": str(finetuned_path),
        "FigurePath": str(figure_path) if figure_path is not None else None,
    })
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(detail), f, indent=2)
    return row


def _principal_mask_from_svd(Uk, Sk, Vhk, top_ratio, top_count):
    score = torch.abs((Uk * Sk.unsqueeze(0)) @ Vhk)
    total = int(score.numel())
    if top_count is None:
        count = int(math.ceil(total * float(top_ratio)))
    else:
        count = int(top_count)
    count = max(1, min(count, total))
    flat = score.flatten()
    _, indices = torch.topk(flat, k=count, largest=True, sorted=False)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[indices.cpu()] = True
    return mask.reshape(score.shape), count


def _principal_angles_deg(cross):
    cosines = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosines)).cpu().numpy()


def _target_layer_files(size_dir, target_layer, target_modules):
    prefix = f"model_layers_{target_layer}_"
    files = []
    for path in sorted(size_dir.glob(f"{prefix}*_weight.npy")):
        module = _module_from_filename(path.name, target_layer)
        if _module_allowed(module, target_modules):
            arr = np.load(path, mmap_mode="r")
            if arr.ndim == 2:
                files.append(path)
    return files


def _module_allowed(module, target_modules):
    if not target_modules:
        return True
    module_base = module[:-len("_weight")] if module.endswith("_weight") else module
    for item in target_modules:
        if item in {module, module_base} or item in module:
            return True
    return False


def _module_from_filename(filename, target_layer):
    prefix = f"model_layers_{target_layer}_"
    if filename.startswith(prefix):
        filename = filename[len(prefix):]
    if filename.endswith(".npy"):
        filename = filename[:-4]
    return filename


def _resolve_output_root(analyse_cfg, related_cfg):
    configured = related_cfg.get("output_root")
    if configured:
        return Path(configured).expanduser().resolve()
    difference_root = Path(analyse_cfg.get("output_root", "./analyseMat/differenceMat")).expanduser().resolve()
    return difference_root.parent / "principalEvidence"


def _parse_target_layer(value):
    if value is None:
        raise ValueError("analyse.related_work.target_layer is required")
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("layer_"):
            value = value.split("_", 1)[1]
    layer = int(value)
    if layer < 0:
        raise ValueError("analyse.related_work.target_layer must be >= 0")
    return layer


def _normalize_modules(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    modules = []
    for item in value:
        text = str(item).strip()
        if text:
            modules.append(text)
    return modules or None


def _parse_data_size(value):
    try:
        return int(value)
    except ValueError:
        return None


def _safe_div(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0:
        return 0.0
    return float(numerator) / denominator


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _max(values):
    return float(np.max(values)) if len(values) else 0.0


def _min(values):
    return float(np.min(values)) if len(values) else 0.0


def _to_float_list(values):
    return [float(value) for value in values]


def _write_csv(path, rows, replace_tasks=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    merged_rows = []
    if replace_tasks and path.exists():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    key = (str(row.get("Source", "")), str(row.get("DataSize", "")))
                    if key not in replace_tasks:
                        merged_rows.append(row)
        except Exception as exc:
            print(f"[AnalyseMat] warning: failed to merge existing CSV {path}: {exc}")
    merged_rows.extend(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _save_mask_plot(update_mask, principal_mask, overlap_mask, path, max_side, source, data_size, layer, module):
    panels = [
        ("Update mask", update_mask),
        ("Principal mask", principal_mask),
        ("Overlap", overlap_mask),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, mask) in zip(axes, panels):
        image = _downsample_mask(mask, max_side)
        ax.imshow(image, cmap="gray_r", interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{source}/{data_size} layer_{layer} {module}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _downsample_mask(mask, max_side):
    mask = mask.to(dtype=torch.float32, device="cpu")
    rows, cols = mask.shape
    stride = max(1, int(math.ceil(rows / max_side)), int(math.ceil(cols / max_side)))
    if stride == 1:
        return mask.numpy()
    pooled = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=stride, stride=stride, ceil_mode=True)
    return pooled.squeeze(0).squeeze(0).numpy()


def _safe_token(value):
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return token or "module"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
