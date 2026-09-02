# 5090C global portfolio reschedule

Date: 2026-09-02  
Scope: full-data paper critical path and retained multi-algorithm frontier

The new 5090C runs full Proposal-only.  After its exact 2000-update runtime twin
matched 5090A at e0, network/optimizer state, sampler/RNG state and step core,
Proposal passed zero-intervention identity, exact resume and repeated evaluation.
It therefore uses the still-running 5090A plain as its strict matched control.
It must not be compared with 4090A plain.

This changes the old queue.  Only after 5090C Proposal, its source exporter and
the 4090 import relay were healthy, the waiting 4090A Proposal successor and its
duplicate exporter were stopped.  They had not begun training.  The healthy
4090A plain was not interrupted.  Its new metric-blind successor is AM-TNC,
which preserves a mathematically independent Adam-metric geometry route.  The
result is three active full-data algorithm paths: Proposal running on 5090C,
AM-TNC waiting behind 4090A plain, and ST-CGR waiting from an exact e1 state
behind 5090A plain.

The legacy 5090C repeated-evaluation receipt initially failed authorization
because the old runner stored the frozen evaluation-bundle identity in a field
that authorization interpreted as the training protocol identity.  Commit
`85eb306` adds a source-bound control-plane migration.  It verifies the frozen
scientific checkout and unchanged duplicate-evaluation hashes, separates the
two identities without loading a checkpoint or performance values, and lets
the original frozen runner issue the authorization.  The scientific training
checkout remains clean at `e4a5eed`.

Measured Proposal throughput projects about 10.76 days, so its e200 endpoint is
around September 13 afternoon.  A one-day extension of the nominal ten-day
5090C rental is the minimum; two days is the safer operational margin.  The
4090A plain projects to September 5, and AM-TNC to approximately September
11--12.  CUT and CycleGAN project to September 6 and 8.  This yields the first
matched full-data in-house algorithm result around September 11--12 and a
two-algorithm set around September 13--14, several days before the prior
plain-to-Proposal-to-AM-TNC serial arrangement.

HJCGR is deferred, not falsified.  Its full-data cost is longer than the added
card rental and its host-separated small25 terminal sign was inconsistent.
DDSB remains reproduction-incomplete until an authoritative source/formula lock
exists.  Starting an extra external baseline on 5090C was rejected because CUT
and CycleGAN are already running and it would displace an evidence-supported
in-house trajectory.  No paired performance value controlled this scheduling
decision; confirmation20 remains sealed.

The machine-readable portfolio, PIDs, hashes, completion estimates and rejected
alternatives are recorded in
`configs/FULL_DATA_METHOD_PORTFOLIO.json` and
`evidence/paper_aio/PAPER_AIO_5090C_GLOBAL_RESCHEDULE_20260902.json`.
