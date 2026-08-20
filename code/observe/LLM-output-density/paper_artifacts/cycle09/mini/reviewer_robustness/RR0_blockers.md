# RR0 Blockers

Created UTC: 2026-07-27T11:52:06.587114+00:00

## Status Counts

| task   | status                                 |   n |
|:-------|:---------------------------------------|----:|
| RR1A   | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT |  96 |
| RR1B   | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT |  96 |
| RR2    | BLOCKED_MISSING_ARTIFACT               |  36 |
| RR2    | READY_REUSE                            |  60 |
| RR3    | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT |  96 |
| RR4    | READY_REUSE                            |  96 |
| RR5    | READY_REUSE                            |  96 |
| RR6    | BLOCKED_PROTOCOL_MISMATCH              |  48 |
| RR6    | READY_REUSE                            |  48 |

## Blocker Reasons

| task   | status                                 | blocker_reason                                                                           |   n |
|:-------|:---------------------------------------|:-----------------------------------------------------------------------------------------|----:|
| RR1A   | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT | missing formal per-sample factor bundle; new forward required for exact sample bootstrap |  96 |
| RR1B   | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT | missing formal per-sample factor bundle; new forward required for exact sample bootstrap |  96 |
| RR2    | BLOCKED_MISSING_ARTIFACT               | missing Stage4 current or base direction.pt singular spectrum                            |  36 |
| RR3    | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT | RR3 is new forward and requires explicit Theory GO                                       |  96 |
| RR6    | BLOCKED_PROTOCOL_MISMATCH              | RR6 is Llama OPD vs frozenSelf0-KD only                                                  |  48 |

## Immediate Gate

- RR1A/RR1B cells are not READY_REUSE unless formal per-sample factor bundles exist.
- RR3 remains new-forward and requires explicit Theory GO.
- READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT cells must not be launched without Theory GO.
