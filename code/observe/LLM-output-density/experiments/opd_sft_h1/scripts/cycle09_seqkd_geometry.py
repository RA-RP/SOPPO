#!/usr/bin/env python3
"""Cycle 09 block 2 G3: seqKD ten-point geometry via the frozen off-KD pipeline."""

from pathlib import Path

import cycle09_offkd_geometry as geometry


ROOT = Path("/root/autodl-tmp/cycle09_seqkd")
geometry.ARM = "seqkd"
geometry.SCHEMA_VERSION = "cycle09_seqkd_geometry_v2_20260718"
geometry.OFFKD_ROOT = ROOT
geometry.OFFKD_MERGED = ROOT / "_merged_models"
geometry.OFFKD_CKPTS = ROOT / "checkpoints"
geometry.OFFKD_BACKFILL = ROOT / "no_backfill_native_grid"
geometry.RUN_ROOT = ROOT / "geometry"
geometry.MAIN_GRID = geometry.EXTENDED_GRID
geometry.NUMERICAL_BACKFILL_STEPS = set()


if __name__ == "__main__":
    geometry.main()
