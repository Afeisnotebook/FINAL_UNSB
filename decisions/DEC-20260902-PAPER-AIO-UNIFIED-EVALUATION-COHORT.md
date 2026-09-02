# Require a source-bound one-container cohort for the final first-wave table

Date: 2026-09-02  
Authority: user-approved common-4090 evaluation protocol

## Decision

Training remains host-separated and checkpoints may never continue training on
another host. Final evaluation is a distinct read-only operation. Each source
host must first export a receipt for e100, e125, e150, e175 and e200 which binds
the checkpoint, full-state sidecar, scientific state, lane semantics, training
commit/protocol, manifest and source host.

Copied checkpoints are evaluated in one selected 4090 environment with the
same current evaluator and frozen `68f53a8e...` CRN bundle identity. The cohort
lock requires all twenty first-wave cells: plain, Proposal-only, CUT and
CycleGAN across five epochs. It also enforces discovery80/five bundles at e200,
fixed NFE cells for UNSB, read-only loading and identical evaluation runtime.

`PAPER_RESULTS.json` may not report `FIRST_WAVE_COMPLETE` without this cohort
lock. The common evaluator improves measurement comparability; it does not
turn CUT or CycleGAN into matched comparisons against a 4090-trained plain and
does not merge host-separated training deltas. Existing metric files with a
different source receipt or result are never silently overwritten.
