from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.ood_metrics import (
    general_ood_avg,
    general_ood_penalty,
    math500_gain,
    per_benchmark_drop,
    worst_ood_drop,
)


class OodMetricsTest(unittest.TestCase):
    def test_ood_drop_only_penalizes_decreases(self) -> None:
        baseline = {"MMLU": 0.60, "GSM8K": 0.70, "WinoGrande": 0.50}
        row = {"MMLU": 0.55, "GSM8K": 0.75, "WinoGrande": 0.49}

        drops = per_benchmark_drop(row, baseline, ["MMLU", "GSM8K", "WinoGrande"])

        self.assertAlmostEqual(drops["MMLU"], 0.05)
        self.assertEqual(drops["GSM8K"], 0.0)
        self.assertAlmostEqual(drops["WinoGrande"], 0.01)
        self.assertAlmostEqual(worst_ood_drop(row, baseline, ["MMLU", "GSM8K", "WinoGrande"]), 0.05)

    def test_p2_p3_penalties_are_p_norms(self) -> None:
        baseline = {"A": 1.0, "B": 1.0}
        row = {"A": 0.7, "B": 0.6}

        p2 = general_ood_penalty(row, baseline, ["A", "B"], p=2)
        p3 = general_ood_penalty(row, baseline, ["A", "B"], p=3)

        self.assertTrue(math.isclose(p2, (0.3**2 + 0.4**2) ** 0.5))
        self.assertTrue(math.isclose(p3, (0.3**3 + 0.4**3) ** (1 / 3)))

    def test_math500_gain_and_avg(self) -> None:
        self.assertEqual(math500_gain(0.56, 0.50), 0.06000000000000005)
        self.assertEqual(general_ood_avg({"A": 0.2, "B": None, "C": 0.4}, ["A", "B", "C"]), 0.30000000000000004)


if __name__ == "__main__":
    unittest.main()
