# Decision: rebind DCLGAN evaluation and addendum to the current V7 delivery chain

Date: 2026-09-13

## Finding

The durable DCLGAN evaluator was healthy as a process but referenced the
obsolete `FINAL_UNSB_PAPER_UNIFIED_EVAL_DYNAMIC_V1` cohort. That cohort had
already failed closed because its GPU-release dependency was obsolete. The
DCLGAN addendum had failed closed on the same old base portfolio. Left
unchanged, the local DCLGAN trajectory could finish and be imported correctly
while evaluation and the paper addendum remained permanently blocked.

## Decision and action

A replacement evaluator and nonblocking addendum were deployed from the clean
V7 control checkout at `da41652`. The evaluator now waits on
`FINAL_UNSB_PAPER_UNIFIED_EVAL_V7_DA41652`; the addendum waits on the matching
V7 final-delivery output and the replacement DCLGAN result. The frozen DCLGAN
adapter, upstream commit, manifest, fixed epochs, common evaluator runtime and
shared GPU lock are unchanged.

Both replacement supervisors and children reached their expected waiting
states, and their joint health watcher remained healthy across two polls. Only
then were the old evaluator, supervisor and health watcher retired. The old
blocked addendum supervisor and child were already absent; its alert-only
watcher was retired. All old directories and evidence remain present.

## Scientific boundary

This is a control-DAG correction, not a new experiment. It did not restart or
modify local DCLGAN training, AM-TNC training, a checkpoint, an evaluation
schedule, an algorithm or a hyperparameter. No performance value was read or
used. The addendum remains nonblocking with respect to the core portfolio,
cross-runtime deltas remain forbidden, and `confirmation20` remains sealed.

Evidence:
`evidence/paper_aio/PAPER_AIO_DCLGAN_V7_DELIVERY_REBIND_20260913T185629.json`.
