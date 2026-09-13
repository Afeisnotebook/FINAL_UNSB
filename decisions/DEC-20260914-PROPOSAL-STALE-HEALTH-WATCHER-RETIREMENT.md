# Proposal stale health watcher retirement

At Proposal e191, the original 5090C health watcher (PID 13587) still bound its
supervisor check to retired PID 9729.  It therefore emitted
`ALERT_PID_DEAD` even though the exact-resume training chain was healthy under
supervisor/trainer PIDs 90767/90768.

The replacement chain was checked before action: the hash-pinned training
guard, the current-supervisor progress watcher, the final source-bound export
recovery chain, and the incremental export recovery chain were all live and
reported zero alerts.  PID 13587 was then terminated after an exact command
line check.  No training, checkpoint, sampler, RNG, export payload, algorithm,
or protocol state was changed, and the old state files were retained as
historical evidence.

The authoritative Proposal health sources are now the current training-guard
and progress-watch health states plus the two export recovery health states.
The retired legacy `ALERT` file must not be interpreted as a current training
failure or restarted.

Paired performance was not read, confirmation20 remains sealed, and Proposal
continues unchanged to e200.
