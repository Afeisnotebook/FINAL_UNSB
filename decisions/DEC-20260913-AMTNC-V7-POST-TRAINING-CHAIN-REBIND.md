# Decision: rebind the post-training evaluation chain to the current AM-TNC run

Date: 2026-09-13

## Finding

A read-only critical-path audit found that the durable first-wave unified
evaluation successor still used the release state of the first numerical
recovery run (`478211c`). The authoritative AM-TNC trajectory had already moved
to the second-stage `5676c91` run. The stale state remained `CHILD_RUNNING` and
could never certify the current run's `COMPLETE_E200` event. Left unchanged,
training could finish correctly while unified evaluation waited forever.

## Decision and action

A complete V7 control chain was deployed from clean commit `da41652` with the
same pinned evaluator runtime, manifest, import roots, fixed epochs, common GPU
lock, and scientific boundaries. Its only operational correction is that GPU
release and AM-TNC export readiness are bound to the authoritative
`5676c91` run.

The V7 chain contains independently supervised successors for:

- first-wave unified evaluation;
- AM-TNC matched evaluation;
- ST-CGR evaluation after its source-bound import and cohort gate; and
- final multi-algorithm delivery.

All four children reached their expected waiting states, and the shared health
watcher remained healthy across two polls. Only then were nine superseded,
wait-only processes terminated. Their directories and historical artifacts
were retained. No checkpoint or result was deleted.

## Scientific boundary

The replacement did not signal or restart AM-TNC training, did not read a
performance value, and did not change an algorithm, hyperparameter, sampler,
RNG state, checkpoint rule, fixed evaluation epoch, or runtime relation.
`confirmation20` remains sealed. The active AM-TNC training and export chain
was verified alive after the retirement.

Evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_V7_POST_TRAINING_CHAIN_REBIND_20260913T183727.json`.
