# AM-TNC e178→e200 overflow-safe recovery

AM-TNC is no longer stopped by the e178 numerical failure. The failure was
localized to float32 overflow in three Adam-metric square/cross-product
reductions; the sampled gradients, Adam scales and scaled vectors were finite.
This is an implementation-level numerical defect, not evidence that the
AM-TNC mechanism or its long-run performance has failed.

The repair retains the frozen float32 path bitwise whenever its geometry is
finite. Only a nonfinite metric product is recomputed using float64 operands;
the gradient samples, projection formula, hyperparameters, RNG and sampler are
unchanged. A read-only replay from the protected e178 checkpoint completed
6,250 updates, crossed the exact former failure at offset 6,245 with one
precision fallback, and ended with finite model and optimizer state.

Continuation uses branch `amtnc-e178-overflow-recovery` at commit
`478211cbbd85c4052458073e30ad746fef86563f`, which is the frozen parent
training commit plus only the overflow-safe AM-TNC operator and its tests. The
original e178 run and checkpoint remain untouched. A publish-last migration to
a new output root changed provenance metadata only; its dynamics-only hash is
identical before and after migration. An attempted migration against current
`main` was rejected before full-state load because later runner metadata had
changed, and that rejected output was quarantined rather than overwritten.

The repaired lane is now running from e178 toward fixed e200 under supervisor
PID 3764454, trainer PID 3764455, recovery guard PID 3764845 and health watcher
PID 3766094. No paired result was read, confirmation20 remains sealed, and no
performance conclusion is made until the fixed long-run evaluation closes.
