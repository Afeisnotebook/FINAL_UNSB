# Global reschedule and next external-source gate

At 2026-09-03 10:54 +08:00, every active full-data lane and its durable
supervision chain was alive and advancing without reading any paired metric.
The measured epochs were: 4090A plain e71, 5090A ST-CGR e8, 5090B CUT e56,
5090B CycleGAN e33 and 5090C Proposal e11.  Their current assignments are
retained.

Keeping Proposal on 5090C is not inertia.  It is the shortest path to the most
portable existing algorithm result, while AM-TNC already has a same-host
matched path behind 4090A plain.  Replacing Proposal with AM-TNC would discard
a healthy e11 trajectory and delay Proposal; colocating another long train
would slow the roughly 80-minute Proposal epoch and does not shorten the first
legal matched-result horizon.  ST-CGR remains the only GPU training lane on
5090A.  CUT and CycleGAN remain colocated on 5090B, followed by the already
armed exact-runtime and dynamic-capacity matched-plain successor.

The expected horizons, using only observed epoch time, are different and must
not be collapsed into one makespan: plain/CUT/CycleGAN should permit the first
core table around September 8; AM-TNC and Proposal plus legal controls should
permit the first reliable algorithm result set around September 12--14; the
layered Proposal/AM-TNC/ST-CGR set is expected around September 15.  These are
engineering projections, not performance decisions.

The idle engineering path was used to remove a later dependency.  Both DCLGAN
and NEGCUT have author repositories.  DCLGAN at commit `f7a7b8e` passed a
128-pixel batch-one data-dependent-initialization and one-update CPU smoke on
PyTorch 2.6.  NEGCUT at `c7dbe3d` exposed an upstream device-selection failure
in the same CPU smoke.  Both repositories also contain a malformed string
`choices` declaration for their explicit mode flag, and neither upstream
trainer provides the full optimizer/scheduler/sampler/RNG recovery required by
this project.  DCLGAN is therefore the next external engineering target;
NEGCUT is deferred as an engineering and licensing matter, not falsified.

DCLGAN is not authorized for GPU training yet.  Its source-bound adapter must
preserve official losses while explicitly labeling the shared 200-epoch,
128-pixel, no-flip exposure adaptation and must pass exact full-state resume,
repeated evaluation, confirmation-lock and throughput gates.  No upstream code
is vendored under the current license review.  The authoritative source and
protocol lock is `configs/PAPER_DCLGAN_NEGCUT_SOURCE_GATE.json`; compact live
evidence is
`evidence/paper_aio/PAPER_AIO_GLOBAL_RESCHEDULE_AND_EXTERNAL_SOURCE_GATE_20260903T105419.json`.
