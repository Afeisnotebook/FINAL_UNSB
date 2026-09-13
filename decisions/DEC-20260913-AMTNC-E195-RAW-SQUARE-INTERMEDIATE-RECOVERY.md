# AM-TNC e195 raw-square intermediate recovery

## Decision

Replace the failed `d8ee501` e195 replay with a source-bound continuation at
`5676c91`.  Resume from the immutable finite e194 checkpoint at commit
`478211c`, not from a non-finite state and not from a selected performance
checkpoint.  Continue the fixed protocol to e200 only after the raw-square
counterexample, exact-runtime tests, deterministic one-step replay and
dynamics-preserving migration gates pass.

## Why the first repair failed

The first Adam precision guard checked whether the mathematical next second
moment

`beta2 * v + (1 - beta2) * g^2`

exceeded float32 range.  PyTorch Adam computes the raw `g * g` product before
applying `1 - beta2`.  A gradient near `2e19` therefore overflows the float32
intermediate even though the weighted recurrence is still representable.  The
old trigger missed exactly this interval.  This is an implementation trigger
omission, not evidence that AM-TNC's optimization geometry or restoration
quality failed.

## Repair boundary

The new guard promotes only the affected Adam `exp_avg_sq` tensor to float64
when either the raw square or the weighted next recurrence would exceed
float32 range.  It does not clip, rescale or skip a gradient; it does not
change the loss, AM-TNC projection, betas, epsilon, learning rate, sampler,
RNG, batch size, data order or total updates.

The exact 4090 environment proved that an unguarded float32 square is infinite,
the guard promotes once, the stored second moment remains finite and the
float64 recurrence is exact.  Two independent e194-to-e194+1 branches produced
the same checkpoint bytes, and their transition-defining state equals the
previous repair's e194+1 state.  Migration changed provenance metadata only.

## Failure containment and next gate

The first-failure interlock stopped the failed `d8ee501` supervisor and guard
before any blind replay.  No e195 checkpoint was published and the finite e194
source remains intact.  The `5676c91` supervisor, outer guard, first-failure
interlock, health watcher and source-bound e200 exporter are now live.  Any new
failure will again freeze the control processes on its first occurrence.

The repair is not closed until e195 produces a finite full-state checkpoint,
records an actual precision promotion, and an isolated exact-resume probe
preserves the float64 optimizer state.  Performance remains unadjudicated;
paired metrics were not read and confirmation20 remains sealed.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_E195_RAW_SQUARE_INTERMEDIATE_RECOVERY_STARTED_20260913T161543.json`.
