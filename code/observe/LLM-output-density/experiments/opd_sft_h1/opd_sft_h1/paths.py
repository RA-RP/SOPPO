from __future__ import annotations

from pathlib import Path


SIDECAR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SIDECAR_ROOT.parents[1]
DEFAULT_REGISTRY_DIR = SIDECAR_ROOT / "registry"


def resolve_repo_path(path: str | Path | None, repo_root: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return (root / path_obj).resolve()


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
