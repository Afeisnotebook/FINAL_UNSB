# Decision: disclose both AM-TNC numerical recovery stages before result review

Date: 2026-09-13

## Decision

The manuscript-facing Methods, Limitations, and Reproducibility drafts now
describe both admitted AM-TNC method-only numerical recovery stages before any
epoch-200 performance result is inspected:

1. the epoch-178 float32 Adam-metric reduction overflow, for which only the
   same non-finite reduction is recomputed in float64; and
2. the epoch-194 raw-gradient-square overflow before Adam's weighted
   `exp_avg_sq` recurrence, for which only affected second-moment state is
   promoted to float64 before evaluating the unchanged recurrence.

The epoch-195 gate facts are also frozen in prose: six precision promotions,
all model and optimizer tensors finite, and exact full-state reload preserving
the promoted dtype. The comparison remains an audited, same-host,
`sequential_method_only_recovery_v1` relation and is explicitly not called a
byte-identical runtime continuation.

## Why this was necessary

The machine-readable runtime registry already bound both stages, but the
paper-facing prose still described only the first. Leaving that mismatch until
after results were available would create avoidable provenance ambiguity and
could make a later correction look result-contingent. This refresh closes that
gap while all language remains pre-result and claim-free.

## What did not change

- no training, supervisor, queue, checkpoint, optimizer, algorithm, or
  hyperparameter was changed;
- no paired performance value was read or used;
- no checkpoint was selected and no comparison was promoted beyond the
  registered recovery relation;
- `confirmation20` remains sealed.

The hash-bound Methods, Reproducibility, Limitations, and manuscript-branching
contracts were refreshed in dependency order. Tests now fail if either AM-TNC
recovery stage disappears from the public prose.

The strict JSON pass also found a pre-existing duplicate
`latest_scientific_state_sha256` key inside the same AM-TNC run object in
`PROJECT_STATE.json`. Both values were identical. Only the later duplicate key
was removed, so the retained value and parsed state did not change.

Evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_TWO_STAGE_PRE_RESULT_DISCLOSURE_20260913T181827.json`.
