# AM-TNC e195 precision closure and staged V6 delivery

## Decision

Accept the e195 engineering gate and continue the fixed AM-TNC trajectory to
e200. Register the e178 geometry-reduction repair and the e194 Adam
second-moment repair as two explicit, sequential method-only numerical
recovery boundaries. Use the new V6 evaluator and final-delivery waiters for
the final `5676c91` source-bound export.

This decision does **not** claim byte-identical runtime, unchanged source code,
or positive performance. It preserves the legal same-host 4090A plain
comparison because both repairs retained the mathematical update on their
finite representable paths, changed no sampling or hyperparameter state, and
were admitted before reading the terminal performance.

## Evidence at e195

The hash-bound e195 checkpoint closed at update 1,667,835. All model and
optimizer tensors are finite. Six affected Adam `exp_avg_sq` tensors were
promoted to float64 at the predeclared raw-square/weighted-recurrence guard.
An isolated e195+1 resume retained those six promoted tensors, remained finite,
and preserved the source checkpoint.

The first durable successor invocation failed before checkpoint inspection
because its standalone subprocess lacked the training repository on
`PYTHONPATH`. Commit `282d840` fixes that control-plane import path. The same
frozen checkpoint gate then passed, and the successor published
`COMPLETE_E195_PRECISION_RECOVERY_GATE`. This incident had no scientific-state
effect.

## Provenance and delivery

The registry now retains both the original single-stage `31f2fb8 -> 478211c`
relation and the final staged `31f2fb8 -> 478211c -> 5676c91` relation. The
terminal portfolio revalidates the two stage records instead of trusting a
copied PASS label and carries the complete stage list into the paper artifact.

The V6 AM-TNC evaluator and final-delivery successor are bound to commit
`83d4b06`, the final raw-square recovery export root, the pinned evaluator
Python, and the original 4090A plain export. They are healthy and waiting for
e200. The obsolete V5 waiters pointed to the stopped e194 source and were
retired only after V6 became healthy; their artifacts were retained. The
unified first-wave and ST-CGR chains were not changed.

## Scientific boundary

This closes an implementation representation failure, not the AM-TNC paper
claim. The fixed e200 checkpoint and e150/e175/e200 sustained adjudication
remain authoritative. No paired performance value controlled this action and
confirmation20 remains sealed.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_E195_PRECISION_CLOSURE_AND_STAGED_V6_DELIVERY_20260913T173026.json`.
