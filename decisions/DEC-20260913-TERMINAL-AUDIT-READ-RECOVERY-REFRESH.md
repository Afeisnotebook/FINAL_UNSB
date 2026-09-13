# Decision: preserve the healthy terminal-audit child and bind the next-failure recovery

## Decision

Keep terminal-audit child PID `16424` running. It has remained healthy since
`2026-09-13T12:32:58+08:00`, is waiting only for fixed imports or the local GPU,
and retains the three completed target-blind plain audit cells.

The control log contains nine launches and eight failed children. Every recorded
failure is the same transient Windows `PermissionError` while reading an
incremental-import membership file during atomic publication. There is no NaN,
OOM, corrupted checkpoint, performance access, or training-state mutation.

Do not hot-replace the healthy frozen child. If it exits again, the recovery path
must first deploy the already tested canonical-reader retry implementation at
commit `f2a601a0e74d8a0000fe8e641f6c64bc532c3378`, then relaunch from the same
read-only audit authority. Eleven launches remain in the current supervisor
budget, so no immediate destructive intervention is justified.

This decision does not authorize terminal JVP to co-reside with DCLGAN on the
GTX1660. It does not read paired metrics, change training, open confirmation20,
or alter any algorithm protocol.
