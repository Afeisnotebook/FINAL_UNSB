# Five-node lease-safe refresh

At 2026-09-07 19:25 +08:00, all five active training lanes were observed alive and advancing without consulting paired performance. The AM-TNC deleted-runtime recovery guard continued to monitor the original healthy supervisor with zero restarts.

The current schedule remains unchanged. Proposal is projected to finish near 2026-09-14 07:38, leaving about 16.4 hours before the declared 5090C boundary. ST-CGR is projected near 2026-09-14 12:43, leaving about 35.3 hours. The 5090B matched plain projection is computed in two phases: co-resident with CycleGAN until its projected e200, then at the previously measured isolated plain rate. This gives a projected completion near 2026-09-12 21:49 and about 26.2 hours of headroom.

No lease extension, queue change, restart, migration, or scientific-protocol change is authorized by this refresh. Re-open the decision early only on a committed trigger: Proposal headroom below twelve hours including export allowance, ST-CGR crossing 2026-09-15 12:00, failure of CycleGAN to release the shared GPU after source-bound e200 export, or any fail-closed health state.

The refresh does not read paired metrics, open confirmation20, select checkpoints, or alter any running process.
