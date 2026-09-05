# DEC-20260906: Recover CycleGAN from the sealed e125 full state

## Decision

Terminate only the CycleGAN trainer after its frozen progress watcher reached
`ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS`, then let the original
scientific supervisor resume from the already-published e125 full state.  CUT
and every other lane remain untouched.

## Evidence and boundary

The e125 latest checkpoint and permanent milestone were published before the
inline evaluation began.  The latest checkpoint SHA256 is
`ce5e58740ba14e6f024dc0e0afd5ef623befbbd1dadc92132d50c432843099b1`; it
deserializes as the frozen full-state schema at update 1,069,125 and carries
scientific-state SHA256
`ac896c95d4f6c2cc331c70e3e1f2679e7e441d65aca41cdae0d29efe459585f2`.
No signal was sent before the registered 10,800-second progress threshold.

The original supervisor stayed alive and launched PID 493306 with the same
`--resume` command, training commit and protocol fingerprint.  This is an
engineering recovery, not a restart from e0 and not a scientific method change.
Completion remains provisional until a checkpoint beyond e125 is published.

## Missing inline metric

The interrupted inline e125 metric was not published.  It must not be
recomputed by co-residing another evaluator on the already busy 5090B.  The
source-bound exporter depends on fixed checkpoints and sidecars rather than
inline metric JSON, and the final one-runtime unified evaluator will recompute
e125 from the frozen milestone.  Therefore no scientific result is lost and
no intermediate performance value is needed to route recovery.

## Scientific firewall

No paired metric was read, confirmation20 remains sealed, no checkpoint crossed
hosts, and neither algorithm nor optimizer, scheduler, batch, data order, or
external-baseline protocol changed.

Evidence: `evidence/paper_aio/PAPER_AIO_CYCLEGAN_E125_INLINE_EVALUATION_STALL_RECOVERY_20260906T012500.json`
