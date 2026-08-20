from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.registry import append_jsonl, load_jsonl, update_status, validate_checkpoint_record, validate_run_record


def _run_record() -> dict:
    return {
        "run_id": "run-1",
        "trajectory_group_id": "group-1",
        "method": "trl_opd_like",
        "role_label": "TRL-OPD-lmbda-1.0",
        "parent_run_id": None,
        "start_checkpoint": "theta0",
        "seed": 42,
        "model": {"student": "theta0"},
        "data": {"prompt_jsonl": "prompts.jsonl"},
        "training": {"max_steps": 2},
        "artifacts": {},
        "status": "planned",
        "teacher_model": "teacher",
        "teacher_mode": "local",
        "lmbda": 1.0,
        "beta": 0.5,
        "loss_top_k": 1,
        "use_vllm": False,
        "use_teacher_server": False,
        "teacher_model_server_url": None,
    }


class RegistryTest(unittest.TestCase):
    def test_registry_append_load_update_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run_registry.jsonl"
            record = _run_record()

            self.assertTrue(validate_run_record(record))
            append_jsonl(path, record)
            self.assertEqual(load_jsonl(path)[0]["run_id"], "run-1")

            updated = update_status("run-1", "completed", path)
            self.assertEqual(updated["status"], "completed")
            self.assertEqual(load_jsonl(path)[0]["status"], "completed")

    def test_registry_rejects_ambiguous_opd_method(self) -> None:
        record = _run_record()
        record["method"] = "opd"
        with self.assertRaises(ValueError):
            validate_run_record(record)

    def test_checkpoint_record_requires_checkpoint_id(self) -> None:
        record = _run_record()
        record["checkpoint_id"] = "run-1-final"
        record["checkpoint_path"] = "out"
        self.assertTrue(validate_checkpoint_record(record))


if __name__ == "__main__":
    unittest.main()
