# Proposal e181 recovery closure and five-lane refresh

## Decision

Close the Proposal-only e181 process-stall recovery as an engineering recovery after the same resumed trajectory produced complete, hash-bound e182 through e187 states. Continue Proposal-only, AM-TNC, ST-CGR, DCLGAN and the 5090B provider-clone matched plain without intervention.

## Evidence

- Proposal-only reached e187 with one total guard restart and zero consecutive no-progress restarts. Its current checkpoint bytes match the recorded full-state SHA256, and the guard and trainer are live.
- AM-TNC reached e197, ST-CGR e177, local DCLGAN e173 and the provider-clone matched plain e126. Each current full-state checkpoint was independently hashed during this audit; every value matches its sidecar.
- All observed training supervisors and trainers are alive. Health watchers report zero current alerts, and the 5090B clone has sufficient storage for its remaining write bound plus headroom.
- The 5090B trajectory remains a segmented exact-runtime-cohort continuation from e116 on a different physical GPU. This refresh does not relabel it as an uninterrupted same-host run.

## Scientific boundary

This is an infrastructure and continuity decision only. It does not adjudicate the performance of Proposal-only, AM-TNC, ST-CGR, DCLGAN or plain UNSB. No paired performance value was read, no best checkpoint was selected, and confirmation20 remains sealed. Proposal/ST-CGR matched deltas remain unavailable until the 5090B clone completes e200, exports a source-bound set and passes the final runtime-relation review.

Compact evidence: `evidence/paper_aio/PAPER_AIO_PROPOSAL_E181_RECOVERY_CLOSURE_AND_FIVE_LANE_REFRESH_20260913T192900.json`.
