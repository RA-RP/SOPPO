from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cycle08_ckpt_pruner as pruner


def create_steps(root: Path, *steps: int) -> None:
    for step in steps:
        (root / f"global_step_{step}").mkdir(parents=True)


def test_pruner_protects_last_complete_checkpoint_during_next_save(tmp_path: Path):
    create_steps(tmp_path, 5, 10, 15, 20)
    (tmp_path / pruner.TRACKER).write_text("15", encoding="utf-8")

    deletable, state = pruner._deletable_steps(tmp_path, {5, 10})

    assert deletable == []
    assert state["tracked"] == 15
    assert state["highest"] == 20


def test_pruner_reclaims_old_rolling_checkpoint_after_commit(tmp_path: Path):
    create_steps(tmp_path, 5, 10, 15, 20)
    (tmp_path / pruner.TRACKER).write_text("20", encoding="utf-8")

    deletable, state = pruner._deletable_steps(tmp_path, {5, 10})

    assert deletable == [15]
    assert state["tracked"] == 20


def test_pruner_fails_closed_on_corrupt_tracker(tmp_path: Path):
    create_steps(tmp_path, 5, 10, 15)
    (tmp_path / pruner.TRACKER).write_text("", encoding="utf-8")

    deletable, state = pruner._deletable_steps(tmp_path, {5, 10})

    assert deletable == []
    assert "guarded_reason" in state
