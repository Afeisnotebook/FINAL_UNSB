# AM-TNC e179 overflow crossing and delivery chain

The overflow-safe AM-TNC continuation has saved e179 at update 1,530,987,
crossing the exact e178-to-e179 interval where the frozen float32 operator
failed three times.  The scientific state is finite, the training supervisor,
outer guard and health watcher are healthy, and no recovery restart occurred.
This converts the earlier 6,250-update read-only replay into a successful live
full-state crossing; it still does not establish a performance benefit.

The incident remains an implementation-level numerical defect in the
AM-TNC-specific Adam-metric reduction, not a mechanism falsification.  The
sampled gradients, Adam scales and scaled replica vectors were finite; only
three float32 square/cross products overflowed.  The repair keeps the original
float32 operator bitwise whenever finite and uses float64 only to evaluate a
nonfinite metric product.  It changes no gradient sample, projection formula,
hyperparameter, RNG or sampler state.

The recovered run used a distinct provenance host label, which made the old
static-pair evaluator reject its otherwise same-physical-host relation to the
4090A plain.  Commit `2b45f88` therefore adds a read-only source view with a
unique control root and the physical host label `4090A`; scientific lane and
live supervisor entries remain symlinks to the recovered source.  Its exporter,
recovery supervisor and health watcher are running with zero restarts.  A
replacement AM-TNC evaluator and final-delivery waiter now consume this
normalized source-bound export.  Existing valid unified and ST-CGR waiters are
reused, and the blocked first AM evaluator remains historical evidence.

AM-TNC continues unchanged to fixed e200.  Its algorithmic disposition will be
decided only after the same-host fixed e150/e175/e200/e200 evaluation closes;
no intermediate paired result was read and confirmation20 remains sealed.
