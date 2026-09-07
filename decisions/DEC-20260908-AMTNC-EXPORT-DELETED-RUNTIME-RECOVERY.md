# AM-TNC exporter deleted-runtime recovery closure

## Decision

The live AM-TNC training and source exporter remain untouched.  The old export
recovery supervisor is retired because its frozen restart command referenced
the removed `unsb_cov` Python path.  A new hash-pinned supervisor adopts the
existing exporter by its historical argv identity but uses the verified
isolated evaluator runtime for every future launch.

## Why this is necessary

The earlier recovery closure protected the AM-TNC trainer and the evaluation
waiters, but a full `/proc/*/exe` audit exposed one remaining gap: exporter PID
2182586 was healthy only because its deleted interpreter inode was still live,
while recovery PID 426584 could not recreate that child after an exit.  This
was not current data loss, but it was a credible e200 delivery failure mode.

## Safety boundary

- Trainer and supervisor PIDs 3446758/3446757 were not signalled or restarted.
- Exporter PID 2182586 was not signalled or restarted.
- No checkpoint was loaded, copied, or modified.
- The new runtime is a control-plane export runtime, not a new training cohort.
- The old recovery supervisor and its health watcher were retired only after
  the replacement adopted PID 2182586 and passed two health polls.
- The old files and evidence remain on disk.

The authoritative evidence is
`evidence/paper_aio/PAPER_AIO_AMTNC_EXPORT_DELETED_RUNTIME_RECOVERY_CLOSURE_20260908T054500.json`.
