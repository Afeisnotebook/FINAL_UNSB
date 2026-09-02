# 5090C Proposal e1 throughput decision

Date: 2026-09-02  
Scope: first complete full-data epoch and scheduling consequence

5090C Proposal completed its first full data epoch without an engineering or
scientific-boundary failure. It performed 8,553 updates in 4,696.60 seconds,
wrote full scientific state hash
`d6767a1e8e05ac0d77888f19fd53916db824bf28ae5e6a39cbfb46f2904a9be8`,
and recorded no paired-controller or confirmation20 access. The trainer,
supervisor, source exporter and 4090 import relay remain healthy.

This rate differs from the preceding 1,000-update projection by only about
1.1%. Keeping the current lane on 5090C therefore remains preferable to moving
it back behind 4090A plain. Constant-rate projection places e200 at approximately
2026-09-13 17:07 +08:00. A nominal ten-day rental is short: one extra day is the
minimum and two days provide the appropriate operational margin. No algorithm
or training setting is changed to fit the rental.

The second isolated 5090A plain epoch also completed in 2,325.01 seconds. The
two-epoch isolated mean is 2,410.47 seconds, versus 4,223.34 seconds during the
temporary co-resident period. This confirms that waiting to resume ST-CGR until
plain e200 reduces total makespan. The current projections are approximately
September 8 for 5090A plain and September 19--21 for ST-CGR. The ST-CGR date is
a throughput bracket, not a performance claim.

No paired performance value informed this decision. The exact 5090C/5090A
runtime relation and final per-image CRN gate remain mandatory before Proposal
delta can be reported. Complete source identities, PIDs, dates and boundaries
are recorded in
`evidence/paper_aio/PAPER_AIO_5090C_PROPOSAL_E1_CALIBRATION_20260902.json`.
