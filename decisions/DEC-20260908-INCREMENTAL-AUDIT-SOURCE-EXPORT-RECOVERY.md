# Incremental audit source-export recovery closure

## Decision

Keep every live training and incremental exporter process in place.  Make the
three commit-pinned recovery supervisors at `2c26c9a` authoritative for the
AM-TNC, ST-CGR and Proposal incremental audit exporters.

## Why

The fixed e100/e150/e200 terminal audit depends on source-bound checkpoint
exports.  Those exporters were healthy, but none had a durable restart layer;
the AM-TNC exporter additionally retained the deleted `unsb_cov` interpreter
inode.  A later exporter exit could therefore have broken the audit delivery
chain even though training itself remained recoverable.

The supervisor now accepts both terminal and incremental export contracts.
For AM-TNC it recognises the old process by the historical Python path but
uses only the verified isolated evaluator runtime for any future restart.
ST-CGR and Proposal retain their existing valid runtimes.

## Fail-closed evidence

The first deployment at `6dc1b6f` rejected all three incremental contracts
before process adoption because their target-blind scheduling field has a
different name from the terminal exporter contract.  No exporter, checkpoint,
or recovery state was changed.  Commit `2c26c9a` added schema-specific
validation; the full suite reports 750 passing tests and each remote checkout
reports five passing targeted tests.

## Non-interference

- Exporter PIDs 2274260, 351230 and 29411 are unchanged.
- All recovery supervisors report `MONITORING_EXISTING_EXPORT` with zero
  restarts, and all health watchers passed two polls without alerts.
- No checkpoint or performance value was read.
- No training process, algorithm, sampler, RNG, protocol, or runtime cohort
  was changed.
- confirmation20 remains sealed.

The authoritative receipt is
`evidence/paper_aio/PAPER_AIO_INCREMENTAL_AUDIT_SOURCE_EXPORT_RECOVERY_CLOSURE_20260908T062200.json`.
