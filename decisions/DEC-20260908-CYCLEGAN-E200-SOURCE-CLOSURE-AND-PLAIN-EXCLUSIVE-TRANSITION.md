# CycleGAN e200 source closure and matched-plain exclusive transition

## Decision

5090B CycleGAN completed the frozen 200-data-epoch protocol naturally at
1,710,600 updates.  The e200 milestone, sidecar, scientific state, five-epoch
source-bound export set, and terminal run state are hash closed.  No paired
performance value was read and no process was signalled to produce this closure.

The byte hash of `full_state_latest.pt` differs from the permanent e200 milestone
because they are separate serializations.  Their sidecars bind the same
scientific-state hash, so the source-bound exporter correctly selects the permanent
e200 milestone; this is not a state divergence.

## Scheduling effect

CycleGAN's supervisor, trainer, continuity guard, exporter, and export-recovery
supervisor all exited naturally after terminal closure.  The already-running 5090B
matched plain retained supervisor/trainer/exporter PIDs 534983/534984/534982 and
continued from e39 without restart, migration, or protocol change.  It is now the
only training lane on that GPU.  Its first post-CycleGAN complete epoch will be used
to replace the prior co-resident throughput estimate; no intermediate paired metric
is involved.

The hardened 5090B-to-4090A relay immediately began copying the five CycleGAN
milestones.  It verified e100 and was transferring e125 at capture time.  It remains
fail-closed and will publish `IMPORT_LANE.json` only after all hashes pass.  Unified
evaluation still waits for both the fixed imports and the AM-TNC GPU-release gate.

## Boundaries

- CycleGAN remains an external official-loss baseline, not a matched UNSB delta.
- Main-table checkpoint remains e200; there is no best-checkpoint selection.
- Confirmation20 remains sealed.
- Import completion and the first exclusive matched-plain epoch are subsequent
  operational events, not reasons to start or modify another experiment.

Evidence:
`evidence/paper_aio/PAPER_AIO_CYCLEGAN_E200_SOURCE_CLOSURE_AND_PLAIN_EXCLUSIVE_TRANSITION_20260908T093100.json`.
