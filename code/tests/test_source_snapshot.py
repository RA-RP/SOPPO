import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "cluster" / "source_snapshot.py"


def run_snapshot(*args, check=True):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        check=check,
        capture_output=True,
        text=True,
    )


def test_source_snapshot_detects_content_changes(tmp_path):
    source = tmp_path / "source" / "SOPPO"
    source.mkdir(parents=True)
    regular = source / "code.py"
    regular.write_text("value = 1\n", encoding="utf-8")
    executable = source / "run.sh"
    executable.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    os.symlink("code.py", source / "code-link.py")
    manifest = tmp_path / "source_manifest.json"
    commit = "a" * 40

    created = run_snapshot(
        "create", "--root", source, "--manifest", manifest, "--commit", commit
    )
    manifest_sha256 = created.stdout.strip()
    assert len(manifest_sha256) == 64
    verified = run_snapshot(
        "verify",
        "--root",
        source,
        "--manifest",
        manifest,
        "--manifest-sha256",
        manifest_sha256,
        "--commit",
        commit,
    )
    assert "Verified source snapshot" in verified.stdout

    regular.write_text("value = 2\n", encoding="utf-8")
    rejected = run_snapshot(
        "verify",
        "--root",
        source,
        "--manifest",
        manifest,
        "--manifest-sha256",
        manifest_sha256,
        "--commit",
        commit,
        check=False,
    )
    assert rejected.returncode != 0
    assert "content mismatch" in rejected.stderr
