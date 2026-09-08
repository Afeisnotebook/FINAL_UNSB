# DEC-20260908: Make the paper reference ledger part of the claim-freeze chain

## Decision

The post-result review draft, committed review decision, claim-freeze receipt,
and downstream distribution evaluator must all carry the same committed
reference-ledger identity.  A missing, uncommitted, hash-drifted, or
bibliography-inconsistent ledger now fails closed.

## Reason

The pre-result ledger commit made the bibliography reproducible, but a
standalone document could still be omitted by a hand-built freeze receipt.
That would allow the empirical claim set to become immutable without proving
that required UNSB, baseline, mathematical-neighbor, and dataset-lineage
citations were reviewed.  The new gate closes that delivery gap without
touching training or deciding which algorithms succeed.

The integration deliberately does not claim that metadata proves novelty.
The related-work collision audit remains the claim authority, and every entry
in `submission_day_refresh_required` must still be checked from its primary
source before submission.

## Operational effect

- Existing healthy training, checkpoint, export, and runtime-cohort processes
  are unchanged.
- The first effect occurs only after the complete e200 portfolio exists and an
  explicit human/Codex claim review is authored.
- Distribution metrics cannot start from a legacy or synthetic freeze receipt
  that lacks the committed citation binding.
- `confirmation20` remains sealed and is not authorized by this gate.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_REFERENCE_LEDGER_CLAIM_FREEZE_INTEGRATION_20260908T133500.json`.
