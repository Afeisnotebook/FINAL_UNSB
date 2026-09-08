# 5090B matched plain e40 exclusive-throughput closure

The fresh-e0 5090B matched plain completed e40 without any restart, resume, process
migration, or protocol change while transitioning naturally from CycleGAN
co-residency to GPU exclusivity.  The permanent e40 checkpoint, sidecar, scientific
state, and metric artifact are hash closed; metric values were not read.

The first complete exclusive epoch took 2,402.323 seconds versus 4,656.142 seconds
for the immediately preceding co-resident epoch, a 48.4% wall-time reduction.  This
is an engineering throughput observation only.  It does not select a method or use
paired performance.

At this single-epoch rate, the remaining 160 epochs project to approximately
September 12 at 20:45 +08.  The durable supervisor, trainer, exporter, progress
watcher, and export-recovery supervisor remain unchanged and healthy.  Therefore the
correct action is to continue the existing process, not restart or replace it.

Evidence:
`evidence/paper_aio/PAPER_AIO_5090B_MATCHED_PLAIN_E40_EXCLUSIVE_THROUGHPUT_20260908T100200.json`.
