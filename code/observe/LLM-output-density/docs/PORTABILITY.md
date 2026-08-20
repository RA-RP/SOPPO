# Portability Notes

The repository preserves the scripts used during the original experiments. Many Cycle
09 orchestration scripts record machine-local paths under `/root/autodl-tmp`; changing
all of them mechanically would make the historical execution record less trustworthy.

For a new environment:

1. Define `PROJECT_ROOT`, `ARTIFACT_ROOT`, `MODEL_ROOT`, and `DATA_ROOT` from
   `configs/paths.env.example`.
2. Use the compact, repository-relative paper pipeline first:
   `python paper/scripts/plot_aaai27_figures.py`.
3. Treat scripts with hard-coded original paths as provenance-preserving historical
   entry points until their path configuration has been explicitly migrated and tested.
4. Never rewrite paths inside frozen numeric manifests; they describe the original run.

The current clean-up changes only the copied figure-generation script. Training and
measurement scripts retain their original logic.

