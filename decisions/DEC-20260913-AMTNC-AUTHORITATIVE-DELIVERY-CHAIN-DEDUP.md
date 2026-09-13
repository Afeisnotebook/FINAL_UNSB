# AM-TNC authoritative delivery-chain deduplication

AM-TNC remained healthy at e194 and no training process was restarted, migrated,
or modified.  A read-only process audit found three pre-existing non-authoritative
paths that were no longer part of the V5 paper delivery graph: the legacy AM-TNC
evaluator bound to the pre-recovery export namespace, the original dynamic
ST-CGR evaluator that had been replaced by the BC99604 recovery chain, and the
BC99604 final-delivery waiter whose AM-TNC input was superseded by V5.

Those exact waiters, their control supervisors, and the health watcher that only
covered the retired c464d6a chain were terminated with `SIGTERM`.  Their files,
states, logs, and checkpoints were retained.  The authoritative V5 AM-TNC and
final-delivery supervisors remain healthy, and the BC99604 unified and ST-CGR
supervisors remain alive because V5 explicitly consumes their result states.

This is an operational deduplication, not an empirical adjudication.  No paired
metric was read, confirmation20 remains sealed, and the fixed e200 training and
evaluation protocol is unchanged.  Its purpose is to prevent obsolete waiters
from contending for the single 4090 evaluation lock or publishing stale duplicate
artifacts when the final checkpoints arrive.
