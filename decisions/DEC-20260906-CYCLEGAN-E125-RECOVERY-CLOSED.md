# Decision: close the CycleGAN e125 recovery incident at e126

Date: 2026-09-06

Status: **EXACT RECOVERY CLOSED; NORMAL e126→e200 TRAINING CONTINUES**

The original CycleGAN trainer stalled inside the inline e125 evaluation after
the complete e125 full state and fixed milestone had already been published.
After the formal 10,800-second progress threshold fired, only that trainer was
terminated. Its original supervisor resumed the same lane from the complete
e125 state without changing the code, protocol, sampler, RNG, optimizer, model,
run directory, or host.

The recovered trainer has now published a complete e126 checkpoint and
heartbeat at 1,077,678 updates. The e126 binary checkpoint hash agrees with its
sidecar; the scientific-state hash is new and the progress watcher and outer
continuity guard both follow the recovered PID. This is the preregistered proof
that the exact e125 recovery crossed the fault boundary. The recovery incident
is therefore closed, while the same supervisor continues ordinary e126→e200
training.

The missing inline e125 metric is not reconstructed inside the training
process. The immutable e125 milestone remains available and the final unified
read-only evaluator will recompute that fixed cell. CUT was neither interrupted
nor restarted, and the 5090B matched-plain successor remains in its original
wait-only state until CUT e200.

No performance value was read and no training or successor decision used a
paired metric. confirmation20 remains sealed.

Evidence:
`evidence/paper_aio/PAPER_AIO_CYCLEGAN_E126_RECOVERY_CLOSURE_20260906T020917.json`
