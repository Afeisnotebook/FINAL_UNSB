# Local incremental relay durable recovery

Date: 2026-09-08

The live audit found that the AM-TNC and ST-CGR local incremental checkpoint
relays (old PIDs 19068 and 6152) and their combined health watcher (PID 18736)
had stopped at approximately 21:41 on 2026-09-07. Proposal relay PID 14528,
the remote training processes, fixed checkpoints and terminal-audit successors
remained healthy. The old bare processes retained no exit log, so the original
exit cause is explicitly unresolved rather than guessed.

The recovery is limited to the read-only e100/e150/e200 delivery path. Commit
`e066f0885b3c4c10717a86efaea9a7091856b18c` adds a dedicated supervisor that
pins the relay contract, source blobs, Python executable and exact command;
rejects duplicates or PID-command mismatches; retains child output; and restarts
only a nonterminal metric-blind relay. Password values remain inherited process
state and are not persisted. Full regression is 748 passed.

AM-TNC now runs under supervisor PID 16540 with child PID 24160. ST-CGR runs
under supervisor PID 20856 with child PID 16752. A deliberate termination of
the first ST-CGR child proved bounded automatic recovery from the same frozen
contract; its restart count of two therefore means initial launch plus this
test, not two incidents. Replacement combined health PID 27152 covers both
supervisors and child states, Proposal relay, terminal audit/pathology chains
and disk capacity, and is healthy with zero alerts.

No training, algorithm, sampler, checkpoint, metric, schedule or confirmation
state was changed. The available epoch lists correctly remain empty until the
source lanes reach e100. Machine-readable evidence is
`evidence/paper_aio/PAPER_AIO_LOCAL_INCREMENTAL_RELAY_DURABLE_RECOVERY_20260908T051100.json`.
