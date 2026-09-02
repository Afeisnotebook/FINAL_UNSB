# Realtime critical-path rescheduling

Date: 2026-09-02  
Scope: paper full-data execution and evidence-derived new-algorithm path

The live audit preserved every healthy scientific process.  The 4090A plain,
5090B CUT/CycleGAN pair, 5090A matched plain, and the transient candidate gate
remain continuous; no checkpoint, data, or historical evidence was deleted.
The 5090A disk policy is the user-authorized 24 GiB worst-case estimate with a
2x safety margin, hence a 48 GiB effective floor.  About 187 GiB remained at
the audit, so the former fixed 200 GiB threshold is not a blocker.

Co-location is decided by terminal makespan, not memory occupancy.  Proposal
remains behind its 4090A matched plain because the prior gate slowed plain by
about 70 percent.  CUT and CycleGAN remain together on 5090B because their
measured schedule saves about 12 hours, and the obsolete duplicate successor
is absent.  On 5090A, ST-CGR used only about 2.1 GiB beside the parent plain,
but its 1000-update full-data continuous gate took about 1122 seconds versus
the isolated small25 compute rate of 1.724 updates/s.  This roughly twofold
compute slowdown makes full co-residence a longer path.  The full e1 probe is
allowed to finish and preserve an exact checkpoint, after which ST-CGR waits
for the same-host plain e200 and resumes from e1.

A durable metric-blind continuation controller is committed at `3cfff2f` and
runs on 5090A as PID 308472 in `after_parent` mode.  It first waits for the
activation successor to complete the runtime, evidence-lock, authorization and
e1 gates.  It then waits only for the parent supervisor's `COMPLETE_E200`
engineering state, launches the frozen candidate supervisor with exact resume,
and starts the source-bound exporter.  It never reads an evaluation artifact or
performance value and cannot open confirmation20.

The current critical path is therefore genuine work rather than an idle
successor: 4090A plain then Proposal; 5090A transient gate, matched plain, then
ST-CGR; and the co-resident external baselines on 5090B.  DDSB remains
`reproduction_incomplete`, not negative, because no authoritative public source
or sufficiently complete formula/full-state lock was found.  The local GTX1660
has completed the terminal spectral audit, target-blind diagnostics, Input
reference, candidate invariants and schema gates.  Its next licensed work is
incoming unified evaluation; starting an unmotivated experiment merely to show
GPU utilization is prohibited.

Machine-readable live states, PIDs, queues, estimates and dependencies are in
`evidence/paper_aio/PAPER_AIO_REALTIME_SCHEDULING_AUDIT_20260902.json`.
