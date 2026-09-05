# Non-destructive acceleration audit

The e44 ST-CGR and e50 Proposal boundary tests reproduced all transition-defining
state hashes exactly under the original and GPU-local NUMA schedules.  Neither
schedule was adopted: ST-CGR fell from 1.655 to 0.930 updates/s and Proposal
fell from 1.639 to 0.618 updates/s.  Including both branch costs, their projected
remaining times increased rather than decreased.  Both live trainers resumed
in memory with every TID restored to the original `0-127` affinity; their source
checkpoints were unchanged and their runtime cohorts remain unchanged.

No narrower-affinity follow-up is justified.  The observed increase in CPU time,
context switches and reduced mean GPU utilization shows that compressing the
existing thread pools onto one NUMA-local logical-CPU set creates more contention
than it removes.  The single-NUMA 4090A AM-TNC lane remains on its existing
`0-27` CPU set.  No second training lane was co-located on an active remote GPU.

The otherwise idle local GTX 1660 now runs the already authorized DCLGAN lane
from its exact update-1000 state.  Its first full epoch closed at 8,553 updates,
and a source-bound export successor is armed.  The local terminal JVP pipeline
retains all completed receipts but is serialized behind DCLGAN with the same GPU
lock.  The three waiting 5090B DCLGAN processes were retired before training
started there, so CUT, CycleGAN and the matched-plain successor are unchanged and
no duplicate DCLGAN will launch.  This is a resource-priority decision, not a
mechanism failure or a change to the internal-algorithm frontier.

Raw operational receipts are stored beside the compact audit evidence.  No
paired performance value was read, confirmation20 remains sealed, and no
checkpoint was migrated or rolled back.
