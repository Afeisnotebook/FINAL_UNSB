# Durable unified evaluation successor

Date: 2026-09-02
Scope: first-wave post-training delivery

The four source-bound export/import relays made every fixed checkpoint durable,
but the final transition from imported artifacts to one-container evaluation,
cohort locking and adjudication still required a person to return at the right
time. That manual gap is now closed by a detached successor on 4090A.

The successor waits only on the complete, predeclared import set and the fixed
AM-TNC `COMPLETE_E200` GPU-release state. It does not parse metric values. Once
both dependencies are ready, it verifies every receipt, checkpoint and sidecar
hash; evaluates Input, plain, Proposal-only, CUT and CycleGAN at the frozen
e100/e125/e150/e175/e200 epochs; locks the one-container cohort; and then runs
posthoc adjudication. Existing valid receipts are reusable after hash checking,
so an engineering interruption cannot force already completed evaluations to
be repeated.

Waiting for AM-TNC is an operational GPU-exclusivity decision, not a scientific
selection. It prevents paper evaluation from slowing the 4090A plain-to-AM-TNC
critical path. The evaluator uses its own GPU lock and is separately watched by
a detached metric-blind health process. Both processes have PPID 1 and were
observed healthy after more than one poll.

The first deployment at commit `6f4cc9e` was retired while it was still in the
pure waiting state because its successful terminal label was not yet recognized
by the existing generic health watcher. It had loaded no checkpoint, generated
no metric and changed no scientific state. Its output remains preserved. The
corrected immutable deployment is bound to commit `03a5948`.

This automation closes only the first-wave delivery gap. It does not freeze the
final algorithm set, adjudicate later full-data candidates, open confirmation20,
select a best checkpoint, or authorize metric-driven training decisions. Exact
PIDs, paths, hashes and boundary flags are recorded in
`evidence/paper_aio/PAPER_AIO_UNIFIED_EVALUATION_SUCCESSOR_DEPLOYMENT_20260902.json`.
