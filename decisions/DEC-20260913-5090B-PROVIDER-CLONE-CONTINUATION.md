# Decision: continue 5090B matched plain on the verified provider clone

## Decision

The provider stopped the original 5090B instance for a billing renewal. The
replacement instance is a data-disk clone, not the same physical GPU: its GPU
UUID is different, so the old PIDs and old host-identity receipt are historical
only.

The cloned disk contains a complete e116 / 992,148-update matched-plain full
state. Its 201,489,371-byte checkpoint matches the atomic sidecar SHA256, all
recorded losses are finite, and the checkpoint remained byte-identical through
the recovery audit. At most the incomplete e117 computation was lost.

Continuation from e116 is authorized because the replacement passed all three
metric-blind gates before the main run was touched:

1. the physical-host gate recorded a new RTX 5090 GPU identity;
2. a fresh 2,000-update plain runtime twin reproduced the frozen e0 and step
   cores exactly, with no differences from the existing 5090A/5090B cohort;
3. two isolated e116-to-e116+8 resume branches produced byte-identical full
   states and identical scientific-state hashes while preserving the source
   checkpoint.

The training checkout, protocol, manifest, seed, batch size, data order,
optimizer and algorithm remain frozen. The resumed control is therefore a
segmented exact-runtime-cohort trajectory. It must not be relabelled as a
same-physical-host run. Final runtime-relation admission and manuscript
metadata must disclose the provider-clone boundary before matched deltas are
published.

The main e116 trajectory may now resume under a new durable supervisor, outer
recovery guard, metric-blind progress watcher and source exporters. No other
healthy training is changed.

## Boundaries

- No paired performance was read or used for this recovery.
- No best checkpoint was selected; e116 is the latest complete atomic state.
- No cross-host method checkpoint is imported and no non-equivalent runtime
  delta is admitted.
- confirmation20 remains sealed.
- A failed later process is recovered only from a complete state produced on
  the replacement segment; the old snapshot PIDs are never reused.

Evidence:
`evidence/paper_aio/PAPER_AIO_5090B_PROVIDER_CLONE_CONTINUATION_GATE_20260913T121300.json`.
