from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.geometry_metrics import effective_rank, principal_angle_unavailable, spectral_gap
from opd_sft_h1.geometry_reader import read_geometry_rows


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class GeometryReaderTest(unittest.TestCase):
    def test_geometry_reader_reads_smat_xmat_and_skips_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shutil.copy(FIXTURES / "tiny_smat.json", tmp_path / "sMat_tiny.json")
            shutil.copy(FIXTURES / "tiny_xmat.json", tmp_path / "xMat_tiny.json")
            hidden = tmp_path / ".ipynb_checkpoints"
            hidden.mkdir()
            (hidden / "sMat_hidden.json").write_text(json.dumps({"layer": {"q": [99.0]}}), encoding="utf-8")

            rows = read_geometry_rows(tmp_path)

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["probe_distribution"] for row in rows}, {"S", "X"})
        self.assertTrue(all(".ipynb_checkpoints" not in row["singular_json_path"] for row in rows))
        self.assertTrue(rows[0]["singular_values"])

    def test_geometry_metrics_handle_no_uv(self) -> None:
        self.assertEqual(effective_rank([1.0, 1.0]), 2.0)
        self.assertEqual(spectral_gap([3.0, 1.0, 0.5], 1), 2.0)
        self.assertEqual(
            principal_angle_unavailable(),
            {
                "principal_angle": None,
                "principal_angle_status": "unavailable_no_uv",
            },
        )


if __name__ == "__main__":
    unittest.main()
