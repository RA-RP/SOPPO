#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small unit tests for baseline selection helpers."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Baseline import select_baseline as baseline


def _toy_frame() -> pd.DataFrame:
    rows = []
    for idx, source in enumerate(["a"] * 6 + ["b"] * 3 + ["c"] * 1):
        rows.append(
            {
                "row_id": f"r{idx}",
                "source": source,
                "problem": f"p{idx}",
                "solution": f"s{idx}",
                "ppl_cond": idx,
                "ifd_ratio": (idx % 6) + 0.1,
            }
        )
    return pd.DataFrame(rows)


class BaselineSelectionTests(unittest.TestCase):
    def test_stratified_allocation_sums_to_requested_total(self) -> None:
        frame = _toy_frame()
        allocation = baseline.allocate_stratified_counts(frame["source"].value_counts(), 5)
        self.assertEqual(sum(allocation.values()), 5)
        self.assertEqual(allocation["a"], 3)
        self.assertEqual(allocation["b"], 2)
        self.assertNotIn("c", allocation)

    def test_random_selection_is_reproducible(self) -> None:
        frame = _toy_frame()
        allocation = baseline.allocate_stratified_counts(frame["source"].value_counts(), 5)
        first = baseline.select_from_sorted_frame(
            baseline.add_random_key(frame, 42),
            allocation,
            ["_baseline_random_key", "row_id"],
        )
        second = baseline.select_from_sorted_frame(
            baseline.add_random_key(frame, 42),
            allocation,
            ["_baseline_random_key", "row_id"],
        )
        self.assertEqual(first["row_id"].tolist(), second["row_id"].tolist())

    def test_ranked_selection_uses_score_inside_each_source(self) -> None:
        frame = _toy_frame()
        allocation = {"a": 2, "b": 2, "c": 1}
        selected = baseline.select_from_sorted_frame(frame, allocation, ["ppl_cond", "row_id"])
        by_source = {source: group["row_id"].tolist() for source, group in selected.groupby("source")}
        self.assertEqual(by_source["a"], ["r0", "r1"])
        self.assertEqual(by_source["b"], ["r6", "r7"])
        self.assertEqual(by_source["c"], ["r9"])

    def test_ppl_middle_uses_middle_fraction_per_source(self) -> None:
        frame = _toy_frame()
        allocation = {"a": 2}
        selected = baseline.select_ppl_middle_frame(
            frame,
            allocation,
            seed=7,
            middle_fraction=0.5,
        )
        by_source = {source: set(group["row_id"].tolist()) for source, group in selected.groupby("source")}
        self.assertTrue(by_source["a"].issubset({"r1", "r2", "r3"}))

    def test_ppl_middle_epoch_scales_by_nominal_over_actual(self) -> None:
        overrides, meta = baseline._train_epoch_override_scaled(
            {
                "train": {
                    "method": "lora",
                    "config_paths": {"lora": "Train/Lora.yaml"},
                }
            },
            size=200,
            numerator_count=200,
            denominator_count=140,
            scale_rule="round(base_epochs * nominal_size / actual_base_count)",
        )
        self.assertEqual(overrides["num_train_epochs"], 36)
        self.assertEqual(meta["base_num_train_epochs"], 25)

    def test_ifd_is_not_a_default_method(self) -> None:
        self.assertNotIn("ifd", baseline.DEFAULT_METHODS)
        with self.assertRaises(ValueError):
            baseline.normalize_methods(["ifd"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_diversity_selection_respects_source_quota(self) -> None:
        frame = _toy_frame()
        embeddings = np.array(
            [[idx, 0.0] if idx < 6 else [0.0, idx] for idx in range(len(frame))],
            dtype=np.float32,
        )
        embeddings = baseline._normalize_rows(embeddings + 0.01)
        selected = baseline.select_diversity_frame(
            frame,
            {"a": 3, "b": 2},
            embeddings=embeddings,
            seed=11,
            diversity_cfg={"max_iter": 5, "n_init": 1, "kmeans_batch_size": 4},
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected["source"].value_counts().to_dict(), {"a": 3, "b": 2})

    def test_cfs_free_text_converts_to_sharegpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "cfs.jsonl"
            source_path.write_text("", encoding="utf-8")
            record = baseline.cfs_record_to_sharegpt(
                {"id": "x0", "text": "synthetic continuation"},
                index=0,
                source_path=source_path,
                user_prompt="Continue.",
            )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["messages"][-1]["role"], "assistant")
        self.assertEqual(record["source"], "cfs")


if __name__ == "__main__":
    unittest.main()
