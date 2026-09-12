# E175+ lease and export-readiness refresh

All five live trajectories were verified through real trainer processes, active
GPU work, current complete-epoch heartbeats, and the authoritative recovery
health chains. No training, checkpoint, runtime, or queue was changed. Old
health watchers that still point at retired pre-recovery processes are not the
authority for the active AM-TNC recovery lane.

Recent complete-epoch throughput projects AM-TNC e200 for September 13 around
18:40, Proposal for September 14 around 07:34--08:02, ST-CGR for September 15
around 04:43--06:03, local DCLGAN for September 15 around 03:25--07:23, and the
5090B matched plain for September 15 around 18:48--19:25. These projections use
no paired result and do not authorize early stopping or algorithm changes.

Fixed source-bound protection already exists through e150 for Proposal and
ST-CGR and through e100 for the 5090B matched plain; AM-TNC has its e178 failure
state and earlier fixed milestones protected off-host. Each source e200 exporter
or its recovery supervisor is healthy. The remaining risk is therefore the
unexported tail, not loss of all scientific evidence.

Provider expiry is not observable over SSH. The user should keep 5090C through
at least September 15 12:00, 5090A through September 16 00:00, and 5090B through
September 16 12:00 (all UTC+08). Under those recorded horizons the conservative
post-training margins are about 28, 18, and 16.6 hours respectively. The 5090B
matched control remains the lease-critical cloud path. Reforecast again by
September 14 06:00 or immediately on any e200 or health event.

Evidence: `evidence/paper_aio/PAPER_AIO_E175_PLUS_LEASE_AND_EXPORT_READINESS_REFRESH_20260913T011000.json`.
