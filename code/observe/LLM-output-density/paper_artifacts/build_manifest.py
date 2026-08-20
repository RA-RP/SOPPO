from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.csv", "SOURCE_MAP.csv"}
ORIGINAL_PREFIX = (
    "mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and path.name != Path(__file__).name
    )


def main() -> None:
    paths = selected_files()
    with (ROOT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in paths:
            writer.writerow([path.relative_to(ROOT), path.stat().st_size, sha256(path)])

    with (ROOT / "SOURCE_MAP.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact_path", "original_repository_path"])
        for path in paths:
            relative = path.relative_to(ROOT)
            if relative.parts[:2] == ("cycle09", "mini"):
                original = Path(ORIGINAL_PREFIX).joinpath(*relative.parts[2:])
            else:
                original = Path("[clean-repository-documentation]") / relative
            writer.writerow([relative, original])


if __name__ == "__main__":
    main()
