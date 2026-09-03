# First e100 incremental relay readiness

## Decision

Keep the running training and audit schedules unchanged.  The deployed
4090A-plain e100 path is ready to release the first target-blind terminal audit
as soon as the immutable e100 checkpoint and sidecar appear.

Before that milestone, the real deployment contracts were compared end to end:

1. 4090A incremental source exporter;
2. Windows incremental relay;
3. local terminal-audit successor and audit-only preflight;
4. the shared GTX1660 non-blocking GPU lock.

Lane, host, training commit, protocol fingerprint, manifest, fixed epochs,
remote export root and local import root all match.  The preflight is released,
the GPU lock can be acquired, and the existing e080 checkpoint/sidecar pair
confirms the expected milestone naming convention.  No e100 artifact exists yet,
so the current empty partial set is the expected state rather than a stalled
relay.

## Regression closure

A new integration test constructs a scientific-state-bound e100 checkpoint,
passes it through the real incremental exporter representation, a fake pinned
SFTP transport, the import-set hash binding and the terminal successor's
readiness resolver.  It proves that the first partial `[100]` set is accepted
without waiting for e150/e200 and remains bound to its checkpoint and export
receipt hashes.

This test does not run a performance evaluation, expose paired data, authorize
training, or select a checkpoint.  It exists to prevent a six-hour-delayed
interface failure when the first real e100 milestone arrives.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_FIRST_E100_RELAY_READINESS_20260903T145419.json`.
