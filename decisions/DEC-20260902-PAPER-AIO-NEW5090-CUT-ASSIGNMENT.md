# DEC-20260902: Assign the new RTX 5090 to the standalone CUT paper baseline

## Decision

The new `5090B` host runs the official-loss CUT lane from the frozen paper
training commit `31f2fb8badaf8293a2ed2744963035575df7d7a6`.  It does not run Proposal
against the existing `4090A` plain trajectory.

## Why this preserves the paper protocol

Proposal requires either a same-runtime matched plain trajectory or an exact
cross-4090 runtime-cohort receipt.  An RTX 5090 is a separate runtime and its
state hashes cannot be merged with the RTX 4090 plain run to manufacture a
matched delta.  CUT is a standalone external baseline: its e200 absolute
metrics can be generated independently and all final checkpoints will still be
evaluated in the common 4090 evaluation container.

This assignment therefore adds a necessary paper opponent without weakening
the causal comparison for our algorithm.  It is a resource decision, not an
algorithm selection and not evidence that CUT is the only relevant baseline.

## Gates and durable execution

- Full manifest and all 18,306 image contents verified.
- Training view counts are exactly 8,553/8,553, discovery 480/480 and sealed
  confirmation 120/120.
- Environment is Python 3.11.16, PyTorch 2.8.0+cu128, torchvision
  0.23.0+cu128, TF32 disabled and deterministic CUBLAS configured.
- The 1,000-update continuous state equals the 500+resume state exactly:
  `1d730404a6707bcedea694873760578382b3e99aaec19daadaf19ecb80d036a4`.
- Repeated lane-blind evaluation is exact:
  `0fd748a19c0d11e15e5f50b07714881e4fe7c8eff5fe119ae1eec5793b9f0523`.
- The long-run authorization passed with no paired controller and unopened
  confirmation20.
- A durable same-host supervisor owns exact resume through fixed e200.

The gate measured 8.24 updates/s including initialization and state saving,
which implies about 57.7 training hours before milestone-evaluation overhead.
This replaces the earlier unmeasured several-day estimate for this host.

## Resource frontier after launch

- `4090A`: full plain UNSB, running.
- `5090B`: CUT, running.
- old `5090`: audit-only because its preserved route-1 evidence leaves less
  than the 200 GiB training gate; those artifacts are not deleted implicitly.
- Proposal: starts on `4090A` after matched plain unless another RTX 4090 passes
  the exact runtime-twin cohort gate.

With only this one additional training-capable card, the project is on the
one-card degraded schedule rather than the proposed four-training-GPU first
wave.  Algorithm discovery remains a separate route-1 obligation and is not
replaced by these baseline runs.

## Non-negotiable boundaries

- No intermediate paired metric controls training or stopping.
- No cross-4090/5090 matched delta is reported.
- No best-checkpoint result replaces fixed e200.
- confirmation20 remains sealed.
- This action item is not described as the unique final algorithm.
