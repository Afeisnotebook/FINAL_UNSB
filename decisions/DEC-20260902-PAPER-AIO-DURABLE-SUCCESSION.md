# Paper AIO durable lane succession

Date: 2026-09-02  
Scope: already-approved first-wave full-data lanes

The two sequential resource assignments are now guarded by persistent,
fail-closed successor processes:

- 4090A: `plain e200 -> Proposal gates -> Proposal e200`;
- 5090B: `CUT e200 -> CycleGAN gates -> CycleGAN e200`.

The transition trigger is only the predecessor supervisor's fixed terminal
state `COMPLETE_E200`. No metric, paired target, intermediate score, or best
checkpoint participates in the decision. Before the successor starts, the
orchestrator reruns a fresh disk/data preflight, the successor-specific exact
resume gate, and repeated evaluation. Proposal additionally requires the
zero-intervention identity witness and same-runtime-output-root authorization.

Any predecessor failure, insufficient disk, stale protocol, changed scientific
commit, or failed successor gate blocks the chain and leaves a machine-readable
state. It does not silently skip a lane or start a substitute. The orchestrator
is operational code from commit `15a503e`; both scientific runs remain pinned to
commit `31f2fb8` and paper protocol fingerprint `68f53a8e...`.

Deployment receipts are recorded in
`evidence/paper_aio/PAPER_AIO_DURABLE_SUCCESSORS_20260902.json`.
