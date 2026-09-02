# Paper critical-path and co-location scheduling

Date: 2026-09-02

The paper program is scheduled by terminal-result makespan, not by displayed
GPU occupancy.  No scientific configuration, paired controller, confirmation
split, or checkpoint-selection rule changes in this decision.

4090A remains a sequential matched cohort.  Proposal passed its engineering
gates, but the overlapping gate made the active plain epoch about 70% slower.
Because plain is the required control for Proposal and downstream paper
adjudication, full Proposal is left behind the existing metric-blind plain-e200
successor.

5090B keeps CUT and CycleGAN co-resident.  CycleGAN saved an exact e1 full
state in 2380.45 seconds while CUT's measured co-resident epoch was 1579.48
seconds versus 1053.76 seconds isolated.  Under the measured contention ratio,
the remaining pair is estimated at 131.59 hours co-resident versus 143.55 hours
sequential.  GPU memory, CPU, IO, checkpoint, OOM, and NaN gates all passed.
The obsolete CUT-to-CycleGAN successor was terminated so it cannot launch a
duplicate process.

5090A now carries an explicit `USER_CAPACITY_OVERRIDE`.  Its audited worst-case
incremental write is 24 GiB; a 2x safety factor makes the effective floor 48
GiB, while 188.22 GiB was free.  No file was deleted.  Temporary co-resident
gates roughly doubled ST-CGR epoch time and limited plain to 2.114 updates/s,
so full plain is not started concurrently.  Instead, a persistent scheduler
starts the already-authorized same-host plain immediately after the existence
of ST-CGR's complete-e200 trajectory artifact, without reading that artifact's
metrics.  This plain remains useful for whichever evidence-derived candidate
is subsequently frozen; no candidate full run starts before posthoc strict
adjudication.

DDSB remains a reproduction-incomplete baseline, not a negative result.  The
local GTX1660 has completed every presently unblocked target-blind audit and
Input evaluation and is reserved for the ST-CGR terminal adjudication and
incoming deterministic evaluations.  Filling it with a new hypothesis before
that evidence exists would violate the route-one discovery contract.

Machine-readable details, PIDs, queues, dependencies, and time estimates are in
`evidence/paper_aio/PAPER_AIO_CRITICAL_PATH_AND_COLOCATION_AUDIT_20260902.json`.
