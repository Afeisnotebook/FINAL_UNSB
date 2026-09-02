# ST-CGR small25 e200 promotion decision

Date: 2026-09-02

`G4-01-STRATIFIED-TIME-CONDITIONAL-GF` completed the frozen small25,
seed-2026, batch-1 trajectory at 30,000 updates.  Only after the complete e200
trajectory and its source-bound terminal receipt existed were paired discovery
metrics read.

The preregistered long-horizon gate passes.  The e150/e175/e200 macro PSNR
deltas average `+1.0446466894 dB`, the e200 delta is `+0.2863084249 dB`, all
three late points have at least four positive domains, the late average worst
domain is `-0.1967468925 dB`, and the SSIM and LPIPS guards both pass.  The
candidate absolute trajectory rises from `17.8941 dB` at e150 to `18.0631 dB`
at e200, so the relative result is not produced by a collapsing candidate or
matched plain control.

This is evidence for the new estimator, not merely a revalidation of Proposal.
ST-CGR keeps the exact native uniform marginal for each bridge-time draw and
replaces the two iid-with-replacement post-D/E G/F times by an ordered uniform
pair without replacement.  The derivation proves the same conditional mean as
native UNSB and removes the positive-semidefinite between-time covariance term
`V_mu / (2(T-1))` relative to Proposal-only.  The frozen fixed-state audit had
already shown the predicted covariance reduction on both Proposal and HJCGR
parent states; the completed long run now supplies the required trajectory
evidence.

The decision is to promote ST-CGR through the already frozen paper runtime,
evidence-lock and authorization gates.  It is not yet a paper result: no
full-data e200 result or cross-seed claim exists, confirmation20 remains sealed,
and the same-host full-data plain must finish before matched adjudication.  The
candidate activation successor stops after one full-data epoch so that
co-resident makespan can be decided using throughput, memory and IO only.

Machine-readable compact evidence is in
`evidence/remote_route1_offload/STCGR_SMALL25_E200_TERMINAL_20260902.json`.
