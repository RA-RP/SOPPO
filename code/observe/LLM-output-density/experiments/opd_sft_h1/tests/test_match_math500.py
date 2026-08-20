from __future__ import annotations

import sys
import unittest
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.match_math500 import match_opd_to_sft


class MatchMath500Test(unittest.TestCase):
    def test_match_opd_to_sft_within_gap(self) -> None:
        opd = [
            {
                "run_id": "opd-run",
                "checkpoint_id": "opd-ckpt",
                "math500_gain": 3.0,
                "general_ood_penalty": 0.04,
                "worst_ood_drop": 0.03,
            }
        ]
        sft = [
            {
                "run_id": "sft-run",
                "checkpoint_id": "sft-ckpt",
                "math500_gain": 2.5,
                "general_ood_penalty": 0.02,
                "worst_ood_drop": 0.01,
            }
        ]

        rows = match_opd_to_sft(opd, sft, max_gap=1.0)

        self.assertEqual(rows[0]["match_status"], "matched")
        self.assertEqual(rows[0]["sft_checkpoint_id"], "sft-ckpt")
        self.assertEqual(rows[0]["math500_gain_gap"], 0.5)
        self.assertEqual(rows[0]["general_ood_penalty_delta"], 0.02)
        self.assertEqual(rows[0]["worst_ood_drop_delta"], 0.019999999999999997)

    def test_unmatched_nearest_when_gap_too_large(self) -> None:
        rows = match_opd_to_sft(
            [{"run_id": "opd", "checkpoint_id": "opd1", "math500_gain": 10.0}],
            [{"run_id": "sft", "checkpoint_id": "sft1", "math500_gain": 4.0}],
            max_gap=2.0,
        )

        self.assertEqual(rows[0]["match_status"], "unmatched_nearest")
        self.assertEqual(rows[0]["sft_checkpoint_id"], "sft1")


if __name__ == "__main__":
    unittest.main()
