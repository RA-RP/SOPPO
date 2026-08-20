# AAAI-27 Reproducibility Checklist

This directory contains the checklist as a standalone AAAI-27 LaTeX project.
The questions and formatting are copied from `AuthorKit27` without
modification; only the requested response lines are filled.

## Overleaf

1. Upload `ReproducibilityChecklist.tex` and `aaai2027.sty` to a new Overleaf
   project.
2. Set `ReproducibilityChecklist.tex` as the Main document if Overleaf does
   not select it automatically.
3. Compile with pdfLaTeX.
4. Download the resulting PDF and upload it to the dedicated checklist field
   in OpenReview. Do not merge it into the paper PDF.

## Current Status

The answers are a conservative snapshot of what is documented in
`aaai_opd_sft_trajectory_en/main_en.tex` and the current `human_read-ch.md`.
They are not automatically final. Recheck the following before the full-paper
deadline:

- Add direct citations for every training, probe, and evaluation dataset.
- State the random seeds and how many independent trajectories were run.
- Add complete hardware, operating-system, and library-version provenance.
- List final training, generation, probing, and analysis hyperparameters.
- Decide the code-release license and include the promised code artifact.
- Align answers about uncertainty and statistical testing with the final
  analyses actually reported.
- Strengthen the formal statement and proof of the local optimality result if
  the paper continues to claim a theoretical contribution.

After any paper or supplementary-material change, update the corresponding
answers in `ReproducibilityChecklist.tex` before compiling the submitted PDF.
