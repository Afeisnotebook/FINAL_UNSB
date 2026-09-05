# DEC-20260905: Guard only the recoverable matched-plain wait

## Decision

Add a frozen recovery guard around the already-running 5090B cross-host plain
successor, without restarting or modifying that successor, CUT, CycleGAN, or
any scientific state.

The guard may relaunch the exact frozen successor command only while its state
is `WAITING_FOR_PREDECESSOR_E200` and the recorded process is no longer alive.
It adopts the existing PID while healthy.  Its recovery authority ends as soon
as the successor enters engineering gates, the co-resident capacity gate, a
co-resident release wait, or plain-supervisor execution.  A blocked, unknown,
scientific-boundary-violating, or post-wait state is never restarted.

## Why

The existing health watcher and progress watcher could report loss of successor
PID 148465, but neither could recover it.  That left the nearest critical
transition—CUT e200 to fresh-e0 strict matched plain—dependent on one long-lived
waiting process.  A death during the safe waiting phase could therefore add up
to the next manual or two-hour Goal inspection interval.

Widening recovery past the waiting phase was rejected.  A successor death after
it has launched a gate or training supervisor does not prove that the child
work also died; an automatic relaunch could collide with live locks or duplicate
work.  Existing supervisor and lane monitors remain authoritative after handoff.

## Deployment

- Guard commit: `c8040ef3c03b37bae186a3b3a983758473fe4ae4`
- Guard PID: `441694`
- Adopted successor PID: `148465`
- Initial status: `MONITORING_EXISTING_WAITING_SUCCESSOR`
- Initial restart count: `0`
- CUT and CycleGAN trainer PIDs remained `5771` and `11846`.
- The first deployment attempt failed closed before state creation because it
  incorrectly required the legacy process cwd to equal its source checkout.
  No scientific process was signalled or changed.  The final version freezes
  absolute cwd, full command, source SHA256, training checkout, and protocol
  identity independently.

## Scientific boundary

The guard reads only process identity and metric-blind control state.  It cannot
read paired performance, cannot open confirmation20, cannot change protocol,
and cannot launch any lane other than the already-authorized fresh-e0 matched
plain successor.

Evidence: `evidence/paper_aio/PAPER_AIO_5090B_MATCHED_WAIT_RECOVERY_GUARD_20260905T170034.json`
