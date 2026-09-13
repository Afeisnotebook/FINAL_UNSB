# AM-TNC export health-watcher deduplication

Two historical health watchers remained alive after their monitored recovery
processes had been superseded.  Both reported deterministic false alerts against
dead PIDs, while the current normalized and incremental export chains were alive
and independently healthy.

Only the two stale watchers were terminated.  The direct exporter, normalized
exporter, incremental exporter, all active recovery supervisors, and AM-TNC
training were preserved.  No files or checkpoints were removed.  The current
health authorities are PID `3785120` for normalized e200 export and PID `3802961`
for incremental audit export.

This prevents future monitoring turns from treating historical dead-PID alerts
as a current delivery failure.  It changes no training state, algorithm, metric,
or evaluation protocol; paired performance was not read and confirmation20
remains sealed.
