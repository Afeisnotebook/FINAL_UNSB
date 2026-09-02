# Local evaluation node materialization

## Decision

Materialize one read-only copy of the full paper dataset on the local GTX1660
host.  The copy is an engineering prerequisite for moving fixed-checkpoint
evaluation and target-blind terminal audits off the long-training GPUs.  It is
not a new experiment and does not authorize confirmation20.

The source is the 4090A dataset because that host is reachable through an
already installed key over the local network.  The source command uses Linux
idle I/O priority (`ionice -c3`) and process niceness 19.  Existing training is
never paused, migrated, or reconfigured for the transfer.

## Fail-closed boundary

- The source is read only.
- The destination must be a strict child of the explicitly supplied local
  workspace root and must not already contain `dataset`.
- The receiving archive is retained until a separate full manifest/content
  hash gate passes.  It is not silently deleted after extraction.
- A completed byte transfer is not sufficient evidence that the dataset is
  usable.  No evaluation is authorized before the canonical manifest gate,
  local train-view materialization, checkpoint source verification, and a
  repeat-evaluation gate all pass.
- Training targets, paired discovery scores, and confirmation20 are never read
  by the mirror scheduler.

## Why this shortens the paper path

The current unified evaluator waits for 4090A AM-TNC e200 before acquiring that
GPU.  A verified local evaluation node can consume exported fixed checkpoints
without delaying plain, Proposal-only, ST-CGR, AM-TNC, CUT, CycleGAN, or the
future 5090B matched plain.  This advances paper tables and the full-data
terminal audit while preserving every training transition.
