from pathlib import Path


def iter_visible_json_files(path: str):
    root = Path(path)
    for file_path in root.rglob("*.json"):
        if any(part.startswith(".") for part in file_path.parts):
            continue
        yield file_path


def split_model_size(stem: str):
    """Return model, data size, suffix from names like GLM_5_200_2026..."""
    parts = stem.split("_")
    for idx in range(len(parts) - 1, 0, -1):
        if parts[idx].isdigit() and int(parts[idx]) >= 100:
            model = "_".join(parts[:idx])
            size = parts[idx]
            suffix = "_".join(parts[idx + 1 :])
            return model, size, suffix

    for idx in range(len(parts) - 1, 0, -1):
        if parts[idx].isdigit():
            model = "_".join(parts[:idx])
            size = parts[idx]
            suffix = "_".join(parts[idx + 1 :])
            return model, size, suffix
    return stem, "", ""
