# Independent candidate completion handoff

Status: accepted, scheduling-only

## Decision

Long-running candidates that share a GPU host no longer need to wait for an
unrelated portfolio peer before publishing their canonical source-bound e200
receipt.  A candidate may publish that receipt immediately after its own
complete e200 trajectory exists.  A downstream, already-frozen successor may
then use the released GPU slot.

This policy first applies on the 5090 host: Proposal-only publishes its receipt
as soon as its full e200 trajectory is complete, allowing HJCGR to run beside
the still-running AM-TNC trajectory.

## Scientific boundary

- The successor reads only the existence of the completed trajectory.  It does
  not read PSNR, SSIM, LPIPS, rankings, domains, or intermediate checkpoints to
  decide whether to publish.
- The candidate's own frozen source worktree builds and signs the receipt.
- Training formula, common e0, seed, small25 view, batch size, target e200,
  checkpoint selection, and confirmation20 state are unchanged.
- The ordinary portfolio successor later rebuilds the `_5090` receipt from the
  same frozen code and artifacts.  Canonical promotion still requires exact
  byte equality, so the independent publication gains no authority by being
  earlier.
- This changes wall-clock scheduling only.  It is not an exit threshold,
  paired controller, checkpoint selection rule, or scientific promotion.

## Multi-algorithm meaning

The final scientific deliverable may retain two or three related viable
algorithms when their mechanisms and evidence warrant it.  `ACTION_PRIORITY`
orders the next unit of compute; it is not an exclusivity rule and does not
collapse `ALGORITHM_SET` to one winner.  Compute should therefore keep the
most informative independent mechanisms moving while preserving exact matched
adjudication for every retained member.

## Implementation

`operations/local_route1_single_candidate_receipt_successor.py` provides the
fail-closed durable bridge.  Its contract freezes both orchestration and
candidate-source Git identities, hashes every receipt source, requires a
canonical candidate path, and rejects any paired-metric scheduling flag.
