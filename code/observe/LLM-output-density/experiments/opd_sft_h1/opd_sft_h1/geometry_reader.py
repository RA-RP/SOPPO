from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SINGULAR_PATTERNS = ("sMat_*.json", "xMat_*.json")


def _is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def iter_singular_json_paths(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    for pattern in SINGULAR_PATTERNS:
        for path in sorted(root_path.rglob(pattern)):
            if _is_hidden_path(path):
                continue
            yield path


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def _singular_values(value: Any) -> list[float] | None:
    if _is_number_list(value):
        return [float(item) for item in value]
    if isinstance(value, dict):
        for key in ("singular_values", "sigma", "values", "S"):
            if key in value:
                return _singular_values(value[key])
    return None


def _iter_entries(data: Any) -> Iterable[tuple[str, str, list[float]]]:
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            values = _singular_values(item)
            if values is None:
                continue
            layer = str(item.get("layer") or item.get("layer_name") or "")
            module = str(item.get("module") or item.get("module_name") or "")
            yield layer, module, values
        return

    if not isinstance(data, dict):
        return

    for layer, modules in data.items():
        values = _singular_values(modules)
        if values is not None:
            yield "", str(layer), values
            continue

        if not isinstance(modules, dict):
            continue

        for module, payload in modules.items():
            module_values = _singular_values(payload)
            if module_values is not None:
                yield str(layer), str(module), module_values


def _probe_distribution(path: Path) -> str:
    if path.name.startswith("sMat_"):
        return "S"
    if path.name.startswith("xMat_"):
        return "X"
    return "unknown"


def _probe_source(path: Path) -> str:
    stem = path.stem
    for prefix in ("sMat_", "xMat_"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return stem


def _infer_step(path: Path) -> int | None:
    text = "/".join(path.parts)
    patterns = (
        r"(?:checkpoint|ckpt|step|global_step|iter)[-_]?(\d+)",
        r"(\d+)(?:st|nd|rd|th)?[_-]?step",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _source_label(path: Path, root: Path) -> str:
    try:
        rel_parent = path.relative_to(root).parent
    except ValueError:
        rel_parent = path.parent
    if str(rel_parent) in {"", "."}:
        return path.stem
    return str(rel_parent).replace("\\", "/")


def load_singular_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_geometry_rows(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    for path in iter_singular_json_paths(root_path):
        data = load_singular_json(path)
        for layer, module, singular_values in _iter_entries(data):
            rows.append(
                {
                    "source": _source_label(path, root_path),
                    "step": _infer_step(path),
                    "probe_distribution": _probe_distribution(path),
                    "probe_source": _probe_source(path),
                    "layer": layer,
                    "module": module,
                    "singular_values": singular_values,
                    "singular_json_path": str(path),
                }
            )
    return rows
