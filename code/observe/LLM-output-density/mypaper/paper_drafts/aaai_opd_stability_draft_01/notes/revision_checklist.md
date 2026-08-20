# Revision Checklist

## Evidence Gates

- [ ] Cycle 04 full or high-sample GSM8K result imported.
- [ ] At least two OPD arms or seeds available.
- [ ] Matched SFT control exists with `GSM8K_gain_gap <= 0.02`.
- [ ] OOD-lite comparison repeated under Cycle 04.
- [ ] Geometry expanded beyond layer 14.
- [ ] Principal-angle or singular-vector artifact status documented.

## Paper Quality Gates

- [ ] Abstract states the claim boundary explicitly.
- [ ] Introduction distinguishes stability from improvement.
- [ ] Related work includes OPD, SFT degradation, and spectral adaptation.
- [ ] Method section can be reproduced from local artifacts.
- [ ] Results do not rely on `eval_limit=200` for improvement claims.
- [ ] Discussion states what would falsify K1-stability.
- [ ] Limitations section remains visible and specific.
- [ ] AAAI page length checked after compiling with official style files.

## Citation Integrity Gates

- [ ] Every OPD mechanism claim is tied to a cited OPD source.
- [ ] Every fine-tuning degradation claim is tied to a cited fine-tuning source.
- [ ] Every spectral mechanism claim is framed as analogy unless directly supported.
- [ ] Local experimental numbers are traceable to Cycle 03 result files.
- [ ] No unpublished Cycle 04 result is written as evidence before import.

