# AM-TNC e195 first-failure interlock

The repaired AM-TNC chain inherited a generic supervisor budget of three
consecutive failures. That policy is appropriate for ordinary infrastructure
interruptions, but not for the current decisive e195 numerical gate: another
failure after the precision repair would be new causal evidence and must not
be replayed blindly for another hour.

Commit `cb082ef` adds a metric-blind first-failure interlock. It binds the
current supervisor and recovery-guard PIDs to their exact command fragments
and polls only the supervisor control state. Healthy training is never
signalled. If the first post-repair failure is recorded, it stops the outer
guard first and then the supervisor before their 30-second restart delay can
launch another child. It does not signal the scientific trainer directly,
load a checkpoint, inspect a performance value, or change the protocol.

The live interlock is PID `28640` and is in
`MONITORING_ZERO_FAILURE_RECOVERY`. A trigger is intentionally an engineering
pause requiring forensic review, not an algorithm decision. If e200 completes
without another failure, the interlock exits without action.
