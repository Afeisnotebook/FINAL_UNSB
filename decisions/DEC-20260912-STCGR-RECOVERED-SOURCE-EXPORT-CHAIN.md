# ST-CGR recovered five-milestone source export

ST-CGR training is healthy and unchanged. The alert detected on 5090A came
from the original five-milestone exporter, which had terminated against the
historical engineering pause before the exact-resume training supervisor took
over. The surviving incremental audit exporter covers e100/e150/e200, but it
cannot replace the paper evaluator's required e100/e125/e150/e175/e200 source
set. Leaving this state unchanged would therefore have caused a late delivery
failure despite successful e200 training.

A separately named, allow-listed, read-only source view now exposes the same
ST-CGR lane and source host to a replacement five-milestone exporter. It does
not load or copy checkpoints and does not modify training, optimizer, sampler,
RNG or protocol state. The exporter writes to the original export destination,
which is already monitored by the existing healthy 5090A-to-4090A relay. Its
exporter, recovery supervisor and health watcher are live and waiting for the
fixed e200 completion.

The two old health watchers were stopped only after dedicated healthy watchers
covered the recovered training supervisor, incremental audit exporter and new
five-milestone exporter. They watched dead pre-recovery PIDs and could only
produce stale alerts. No training or live exporter was stopped.

This is a delivery-control repair, not a scientific intervention or a result
claim. Performance has not been read, confirmation20 remains sealed, and the
paper decision remains fixed-e200 evaluation under the registered runtime
relations.
