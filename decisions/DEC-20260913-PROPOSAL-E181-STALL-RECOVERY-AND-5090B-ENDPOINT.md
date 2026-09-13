# Decision: recover Proposal from e181 and escalate the matched-control endpoint

## Decision

The 5090C Proposal process was no longer treated as a healthy long epoch. Its last normal epoch took about 4,891 seconds, while the e181 heartbeat exceeded the predeclared 14,400-second metric-blind stall bound. Sixteen GPU samples were all zero, the process consumed one CPU core, and it made no file or byte-I/O progress during a 31.3-second diagnostic window.

After the frozen recovery preflight verified the clean training/control commits, pinned Python hash, protocol authorization and the actual e181 checkpoint SHA, only the stalled trainer was terminated. The existing outer guard recovered a new supervisor and trainer from exactly e181 / 1,548,093 updates. GPU work resumed without changing the lane, protocol, runtime cohort, sampler, RNG or matched-control relation.

The former progress watcher was tied to the retired supervisor PID. It has been replaced with a read-only watcher bound to the recovered supervisor and a separate health watcher. The old watcher is retired so stale alerts cannot be mistaken for the current process state.

This is the third recorded Proposal long-process stall, after e75 and e135. It is an engineering/runtime reliability event whose root cause remains unlocalized. It is not evidence that Proposal is beneficial or harmful. The next scientific-state closure is the first complete post-recovery e182 checkpoint.

Separately, the 5090B endpoint currently refuses connections. This is not recorded as a training failure or checkpoint loss because the internal state cannot be observed. No replacement experiment is authorized. Provider reachability must be restored first; then the existing matched-plain supervisor, trainer, guard and full-state checkpoint must be audited before any recovery action.

## Consequences

- Proposal continues from e181; at most one incomplete epoch of computation was discarded.
- AM-TNC independently closed e192 with matching full/scientific hashes and continues unchanged.
- ST-CGR and local DCLGAN remain healthy and untouched.
- Proposal/ST-CGR strict matched deltas remain on the 5090B matched-control critical path.
- Intermediate paired performance, best-checkpoint selection and confirmation20 remain unavailable to execution control.

Evidence: `evidence/paper_aio/PAPER_AIO_PROPOSAL_E181_STALL_EXACT_RECOVERY_AND_CONTROL_ENDPOINT_ALERT_20260913T105819.json`.
