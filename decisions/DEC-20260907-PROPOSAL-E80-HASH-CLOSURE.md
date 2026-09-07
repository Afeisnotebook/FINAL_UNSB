# Proposal e80 fixed-milestone closure

Proposal completed e80 at 684,240 updates and continued on the same 5090C
runtime. The permanent e080 checkpoint, rolling latest state, sidecars, and
metric artifact were hashed. The metric file was not parsed. The checkpoint
loads on CPU with networks, optimizers, schedulers, method state, all RNG
sources, and both independent samplers present.

The supervisor, trainer, exporter, health watcher, and progress watcher remain
alive. The progress watcher reports `HEALTHY_WITHIN_EPOCH_BOUND`; the previous
e75 engineering recovery is therefore still closed. No completed epoch was
lost and no runtime or protocol change occurred.

A deliberately conservative projection using the slower e80 milestone epoch,
including its evaluation overhead, places e200 around 2026-09-14 08:20 +08:00.
This remains inside the declared 5090C availability window, but leaves only
about 16 hours of headroom. The existing 2026-09-11 lease reforecast remains a
real decision gate. It must use throughput and remaining work only, never the
intermediate metric payload.
