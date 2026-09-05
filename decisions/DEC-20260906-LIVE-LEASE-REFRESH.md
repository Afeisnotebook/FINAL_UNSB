# Live throughput and lease refresh

Date: 2026-09-06  
Scope: target-blind completion timing and capacity risk

The last five completed epoch wall times keep AM-TNC, ST-CGR and Proposal
inside their currently declared availability windows. Their conservative e200
projections are approximately September 13 13:44, September 14 16:36 and
September 13 22:26, respectively. Local DCLGAN is projected around September
14 20:53 to September 15 03:22 and no longer creates a remote-GPU lease
requirement.

The remaining tight path is the 5090B matched plain. If the pre-registered
capacity gate rejects CycleGAN co-residence, the conservative sequence is
CycleGAN e200 near September 8 01:52 followed by about 133.9 hours of isolated
plain training. That places plain e200 near September 13 15:47, leaving only
about 8.2 hours before the currently recorded September 14 availability
boundary, before export time and ordinary throughput variance. To prevent a
nearly complete legal control from being lost, 5090B should be guaranteed
through at least September 15 if the existing rental does not already cover
that date.

This recommendation does not change the capacity gate or scientific schedule.
CUT, CycleGAN and their successor remain untouched, and the gate must still use
measured makespan rather than paired results. The prior request to extend 5090B
for DCLGAN is retired because DCLGAN now runs locally; the recommendation is
solely for the indispensable matched control. Exact trace fields, projections
and boundaries are recorded in
`evidence/paper_aio/PAPER_AIO_LIVE_THROUGHPUT_LEASE_REFORECAST_20260906T023903.json`.
