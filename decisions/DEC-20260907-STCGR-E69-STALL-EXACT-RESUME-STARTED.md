# ST-CGR e69 stall: exact-resume recovery started

At the 2026-09-07 07:43 +08:00 read-only refresh, the 5090A ST-CGR trainer had
not advanced its epoch-boundary heartbeat or full-state checkpoint since e69 at
02:29. The progress watcher had independently crossed its 7,200-second limit and
reported `ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS`: five GPU samples were
zero and a 30-second process sample showed CPU ticks but no file I/O. No OOM,
CUDA error, NaN, traceback, or disk-capacity fault was found.

The e69 checkpoint was recomputed as
`0a777ccfe9628a07d0d146820c5aebf07b0b7fb4d6978a03e78555ca77e0b541`, matched
its sidecar, and loaded on CPU with networks, optimizers, schedulers, method
state, all RNG state, and both samplers present. This made the live child an
engineering stall rather than a healthy training process.

Only stalled child PID 431032 received SIGTERM. The durable supervisor and all
successors were retained. The frozen `--resume` command started PID 863376 from
the e69 full state; two post-start samples showed active GPU work. No scientific
configuration, runtime cohort, data order, or metric policy changed. Closure is
deliberately pending until the replacement child writes a hash-verified e70
state. The maximum loss is the incomplete work after e69, not a completed epoch.

The simultaneous 4090A AM-TNC, 5090C Proposal, 5090B CycleGAN/matched plain, and
local DCLGAN jobs were read only and remained healthy. Confirmation20 stayed
sealed and no performance value was read.
