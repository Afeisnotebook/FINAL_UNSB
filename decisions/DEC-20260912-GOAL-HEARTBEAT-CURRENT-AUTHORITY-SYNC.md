# Return the Goal heartbeat to low-frequency current-authority monitoring

The ten-minute emergency heartbeat had served the AM-TNC e178 recovery, but its
prompt still focused on the first e179 crossing and fixed exporter PIDs after
the recovery reached e180. Continuing that prompt would spend monitoring quota
and risk treating normal dynamic control-process replacement as a failure.

The active heartbeat now runs every two hours, resolves all live PIDs from the
three current authority files, treats archived RF/G3 successor fields as
non-authoritative, and retains the source-bound AM recovery and all scientific
hard boundaries. Healthy waits are quiet; completion, failure, recovery,
delivery transitions and user decisions remain notification events.

No credential is persisted in the prompt. No training, checkpoint, algorithm,
protocol, metric or confirmation state changed.
