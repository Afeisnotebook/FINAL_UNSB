# CycleGAN post-e175-checkpoint stall: exact resume started

CycleGAN wrote its e175 rolling and permanent full-state checkpoints at
1,496,775 updates, then failed to publish the e175 metric and heartbeat. The
normal e150 evaluation took about 41 seconds; this post-checkpoint phase made no
GPU or file-I/O progress for more than the registered 10,800-second watchdog
limit. The live-progress watcher therefore formally reported
`ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS`.

Both e175 checkpoint copies were hashed, and the permanent state loaded on CPU
with all networks, optimizers, schedulers, method state, RNG sources, and both
samplers. Only stalled child PID 493306 received SIGTERM. The original
supervisor started PID 662728 with the frozen `--resume` command. The progress
watcher returned to `HEALTHY_WITHIN_EPOCH_BOUND`, with new I/O and GPU work.
The co-resident 5090B matched plain process was neither signaled nor modified
and remained healthy.

Because resume starts after the already committed e175 update, it does not
repeat the interrupted e175 evaluation. This does not invalidate the trajectory:
the source-bound exporter carries the immutable e175 checkpoint, and the common
4090A evaluator will reconstruct the required e175 paper metric read-only. The
metric payload is not used for training or scheduling. Recovery closure remains
pending until the replacement child writes e176.
