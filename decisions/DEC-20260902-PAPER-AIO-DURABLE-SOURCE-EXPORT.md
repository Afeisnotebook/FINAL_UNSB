# Arm source-bound checkpoint export independently of training succession

Date: 2026-09-02  
Authority: unified evaluation cohort contract

## Decision

The existing plain-to-Proposal and CUT-to-CycleGAN successors must remain
focused on exact training recovery. A second, GPU-free successor is assigned to
each of the four source lanes. It waits only for that lane's sealed
`COMPLETE_E200` state, then emits source-bound receipts for e100, e125, e150,
e175 and e200.

The exporter is pinned to a clean control checkout and verifies the frozen
training commit/protocol in every checkpoint. It does not evaluate images,
inspect performance values, copy checkpoints, resume training or schedule an
algorithm. Therefore exporting may safely occur while the training successor
starts the next lane on the same GPU.

Any missing milestone, changed source hash, mismatched training identity or
repeated source-lane engineering failure blocks that lane's export set. It may
not be silently omitted from the later unified cohort.
