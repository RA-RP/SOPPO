# LLM Output Density

Research code, paper sources, and compact evidence artifacts for studying
domain-conditioned functional spectra during post-training. The project compares
on-policy distillation (OPD), supervised fine-tuning (SFT), and matched offline
distillation controls across Qwen and Llama trajectories.

## Repository Layout

- `AnalyseMat/`, `Baseline/`, `Eval/`, `GetSlice/`: reusable analysis and evaluation tools.
- `experiments/opd_sft_h1/`: training, evaluation, geometry, and robustness pipelines.
- `mypaper/`: the research workspace for facts, protocols, claims, related work, and cycle history.
- `paper/`: Chinese and English AAAI LaTeX sources, supplementary material, and writing workflow.
- `paper_artifacts/`: compact numeric artifacts needed to rebuild the paper figures and audit key tables.

Large model checkpoints, rollout caches, raw activation caches, per-token spectra,
and copyrighted literature PDFs are intentionally excluded.

## Paper Workflow

The paper has three authoritative layers:

1. `mypaper/theory/human_read-ch.md`: theory, experimental facts, and claim boundaries.
2. `paper/zh/main_zh.tex`: paper structure and Chinese-language argumentation.
3. `paper/en/main_en.tex`: final English manuscript.

Changes should flow in that order. See [paper/README.md](paper/README.md) and
[paper/workflow/WORKFLOW.md](paper/workflow/WORKFLOW.md).

## Environment

Python 3.10 was used for the original experiments.

```bash
conda create -n density python=3.10 -y
conda activate density
pip install -r requirements.txt
```

The original experiments used machine-local model and artifact roots. Copy
`configs/paths.env.example` to an untracked `.env` file or export the variables
in your shell. Historical scripts that predate this clean repository may still
contain recorded absolute paths; see [docs/PORTABILITY.md](docs/PORTABILITY.md).

Never place access tokens in tracked files. Authenticate interactively or use an
environment variable supplied by your secret manager.

## Rebuilding Paper Assets

```bash
# Rebuild the five vector figures from compact CSV artifacts.
python paper/scripts/plot_aaai27_figures.py

# Build English main paper and supplement.
make paper-en
make paper-supplement

# Build the Chinese working paper.
make paper-zh
```

The paper compiles from committed vector figures even when figure-generation
dependencies are not installed.

## Verification

```bash
# Check repository size boundaries, credential patterns, and artifact hashes.
make verify

# Run the Python test suite after installing the declared dependencies.
make test-fast
```

In the lightweight preparation environment, 10 dependency-free tests pass. Two
additional tests require `transformers` and `tensordict`, which are declared in
`requirements.txt` and `requirements-verl.txt` respectively.

## Artifact Boundary

`paper_artifacts/` contains derived numeric tables and manifests selected from the
Cycle 09 analysis. It does not contain raw prompts, generated completions, model
weights, activation caches, or full training outputs. Every included file is listed
with its byte size and SHA-256 hash in `paper_artifacts/MANIFEST.csv`.

## Publication Status

This repository is prepared as a clean private working repository. Before making it
public, replace the anonymous citation metadata, choose an explicit open-source
license if desired, run the security checks in `docs/PUBLIC_RELEASE_CHECKLIST.md`,
and attach large reproducibility artifacts through a dedicated artifact host.
