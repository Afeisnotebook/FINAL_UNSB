# ST-CGR e80 fixed-milestone closure

ST-CGR completed e80 at 684,240 updates and continued on the same 5090A
runtime. The permanent e080 checkpoint, rolling latest state, both sidecars,
and the metric artifact were hashed. The metric file was not parsed.

The e80 checkpoint loads on CPU with G/F/D/E, optimizers, schedulers, ST-CGR
and proposal-family method state, all four RNG sources, and both independent
samplers present. The permanent milestone and rolling latest serialization have
the same scientific-state hash even though their file hashes differ, as
expected for separately serialized full-state artifacts.

The continuation process, supervisor, trainer, exporter, health watcher,
progress watcher, and export-recovery chain remain alive. Current health is
`HEALTHY_WITHIN_EPOCH_BOUND`; the e69 engineering recovery remains closed and
no completed epoch was lost. The incremental target-blind exporter is
registered only for e100/e150/e200, so e80 not appearing in that export set is
intentional rather than a missing-delivery defect.

A conservative projection using the e80 epoch including milestone overhead
places e200 around 2026-09-14 14:10 +08:00, leaving about 34 hours before the
declared 5090A availability boundary. No lease extension is triggered now.
This projection is operational only: no performance value was read and no
paired result controls training or scheduling.

Complete receipt:
`evidence/paper_aio/PAPER_AIO_STCGR_E80_HASH_CLOSURE_20260907T224848.json`.
