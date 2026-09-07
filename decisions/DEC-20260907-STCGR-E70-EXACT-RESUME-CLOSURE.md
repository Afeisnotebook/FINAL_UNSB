# ST-CGR e70 exact-resume closure

The replacement ST-CGR child completed e70 at 598,710 updates from the
hash-verified e69 state. Its 4,708.7-second epoch is within the established
long-run throughput envelope. The new checkpoint hash matches its sidecar,
loads on CPU, and contains networks, optimizers, schedulers, method state, all
four RNG sources, and both samplers.

The original supervisor, continuation process, export successor, health
watcher, and progress watcher stayed in place. The progress watcher now reports
`HEALTHY_WITHIN_EPOCH_BOUND`, and the trainer continued using the GPU after the
write. Therefore the e69 stall recovery is closed: no completed epoch was lost,
the scientific runtime cohort did not change, and the lane continues to e200.

No paired performance value was read, no scheduling decision used a metric,
and confirmation20 remains sealed. Based on this recovered epoch, e200 remains
inside the declared 5090A availability window with roughly 37 hours of headroom.
