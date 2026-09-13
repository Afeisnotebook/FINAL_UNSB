# 5090B clone e134 health and authority repair

The expired 5090B instance has been replaced by the user-provided clone at
`connect.weste.seetacloud.com:43172`.  The replacement did not silently assume
same-host identity: it passed the 2000-update exact runtime-cohort gate and two
independent 8-update resume branches before the sealed e116 full state was
continued.  The live trajectory has now reached e134 / 1,146,102 updates with
zero guard restarts, zero health alerts, and no completed-epoch loss.

The trainer, supervisor, outer recovery guard, progress/health watches, full
e200 exporter and incremental e100/e150/e200 exporter are all live.  Local
incremental delivery has already reverified e100, while both the local and
4090A terminal relays remain durably bound to port 43172 and wait for the
source-bound e200 export.  The clone has about 222 GiB free against a 24 GiB
conservative remaining-write allowance.

During the authority refresh, `configs/FULL_DATA_METHOD_PORTFOLIO.json` was
found to contain two `performance_adjudicated` keys in the historical AM-TNC
V5 event.  Later terminal-result fields had also been copied into that
event-time record.  The V5 object is restored to its original waiting state;
the final AM-TNC adjudication remains only in the separate V8 terminal object.
A regression test now rejects duplicate keys in all three scheduling-authority
JSON files.

No healthy training process, checkpoint, sampler, RNG, algorithm, protocol, or
queue was changed.  No paired performance was read, and confirmation20 remains
sealed.  The 5090B matched comparison remains a segmented exact-runtime-cohort
continuation and must be disclosed as such after e200; it is not relabelled as
a same-physical-host run.
