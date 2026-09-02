# Paper critical path after isolated-throughput recalibration

Date: 2026-09-02  
Scope: scheduling facts and completion estimates only

The pre-5090C queue is no longer the correct schedule, but the already-executed
5090C reschedule remains the best available allocation. Proposal-only is now
running independently on 5090C and is strictly matched to 5090A plain through
the exact 2000-update runtime relation. 4090A plain therefore releases into
AM-TNC, while 5090A plain releases into the already-authorized ST-CGR e1 state.
This preserves three mathematically distinct in-house full-data paths rather
than a Proposal/ST-CGR-only frontier.

The first isolated 5090A plain epoch took 2495.93 seconds, versus 4159.10
seconds under temporary ST-CGR co-residence. This moves its projected e200 from
approximately September 12 to September 8 afternoon. Applying the measured
isolation bracket to ST-CGR moves its provisional result window from September
24 to September 20--22. The estimate remains a bracket until its first isolated
continuation epoch; it is not a scientific result.

5090C Proposal has not yet completed its first full epoch at this snapshot. Its
1000-update gate still projects an e200 endpoint around September 13 afternoon,
roughly 18 hours beyond a nominal ten-day rental. At least one extra rental day
is therefore provisionally required; two days is the safe margin. The first
full epoch is the next automatic recalibration trigger and does not require any
paired metric.

No healthy training was interrupted. No new lane was added merely to occupy a
GPU. HJCGR remains deferred rather than falsified, and DDSB remains
reproduction-incomplete rather than negative. Local GTX1660 is legitimately
waiting for fixed checkpoint imports because all currently unblocked
target-blind audits and evaluation interfaces are complete.

The complete live snapshot, PIDs, DAG, runtime relations, rejected alternatives
and completion windows are recorded in
`evidence/paper_aio/PAPER_AIO_ISOLATED_THROUGHPUT_RECALIBRATION_20260902.json`.
