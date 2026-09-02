# Separate training source identity from the common paper CRN bundle

Date: 2026-09-02  
Authority: frozen full-data evaluation protocol and dynamic-candidate contract

## Decision

The protocol fingerprint continues to identify all code that can change a
training transition or evaluation implementation. A new candidate therefore
has a different fingerprint from the already-running first-wave plain. That
difference must not also change the latent and rollout-noise bundles used for
matched evaluation.

All current and future paper checkpoints consequently use the frozen first-wave
fingerprint `68f53a8e...` solely as the CRN bundle seed identity. This is not a
claim that candidate training belongs to the old source identity. Candidate
training retains its own current protocol fingerprint and must pass the
cross-code gate. Adjudication computes a delta only when every late image has
the same domain, stem, order, replicate, NFE and bundle hash as its same-host
plain reference.

The e200 record now exposes each of five replicate-level macro metrics and
their population standard deviation. Dynamic evidence-locked candidates are
included in `PAPER_RESULTS.json`; they must additionally pass the absolute
plain-collapse guard. A unified first wave cannot be marked complete without
plain, Proposal-only, CUT and CycleGAN. None of these post-hoc checks may
control training or open confirmation20.
