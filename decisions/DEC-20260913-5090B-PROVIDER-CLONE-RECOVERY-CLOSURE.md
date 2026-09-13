# 5090B provider clone recovery closure

## Decision

Continue the existing `5090B_MATCHED_PLAIN` trajectory from its hash-verified e116 full state on the provider clone. Classify the result as a **segmented exact-runtime-cohort continuation**, not as an uninterrupted same-physical-host run.

This preserves the already completed 116 data epochs and the legally admitted matched-control role. It does not authorize treating the new GPU UUID as the old physical machine, and the physical segment boundary must be disclosed in the final runtime relation and manuscript evidence.

## Why continuation is admissible

- The cloned disk retained the complete e116 checkpoint and sidecar. The actual checkpoint SHA256 and scientific-state SHA256 match the recorded sidecar.
- A new isolated 2,000-update plain twin reproduced both the frozen e0 core and the step core exactly.
- Two independent e116-to-e116+8 resume branches produced identical full-state and scientific-state hashes, while the parent checkpoint remained unchanged.
- The real resumed trajectory closed e117 at update 1,000,701. Both the checkpoint bytes and recomputed scientific-state hash match the new sidecar.
- Seed, sampler/RNG state, batch size, optimizer, code commit, manifest, protocol fingerprint and data-epoch definition were not changed.

At most the incomplete portion of e117 that was in memory when the provider stopped the old instance was lost. No complete epoch was lost and no training result was selected from paired performance.

## Delivery-chain change

The old endpoint-specific relays were not reused. Two new key-authenticated, pinned-host-key chains were deployed:

1. The local incremental audit relay re-verified and published the existing e100 source-bound import, then waits for e150/e200.
2. The 4090A unified-evaluation relay authenticated to the clone and now waits for the complete e200 export set.

Each chain has a frozen contract, recovery supervisor and health watcher. Only after both new chains were healthy were the obsolete old-endpoint workers retired. No checkpoint, import receipt or historical evidence was deleted. The unified evaluation successor remains unchanged and will still resolve the logical source label `5090B_MATCHED_PLAIN`.

## Scientific boundary

This event is infrastructure recovery, not algorithm evidence. It neither supports nor refutes plain UNSB, Proposal-only, ST-CGR or any other mechanism. Intermediate paired metrics were not read; confirmation20 remains sealed; e200 and the preregistered e150/e175/e200 sustained protocol remain authoritative.

Compact evidence: `evidence/paper_aio/PAPER_AIO_5090B_PROVIDER_CLONE_E117_AND_RELAY_CLOSURE_20260913T125800.json`.
