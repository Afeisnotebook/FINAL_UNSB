# Active AM-TNC V5 pointer global synchronization

The delivery matrix correction exposed the same stale V2 replacement pointers
inside the current AM-TNC run objects in `PROJECT_STATE.json` and the full-data
method portfolio. Those active objects are now synchronized to the V5
evaluation/final-delivery supervisors and the verified e182 live state.

Historical V2--V4 evidence is preserved and remains valid as history, but it is
not recovery authority. Historical `latest_reaudited` fields are also preserved
when e182 received only a live heartbeat/sidecar verification rather than a new
full CPU checkpoint audit.

No process was restarted or signalled, no checkpoint or runtime changed, and no
performance value was read. This change removes contradictory current pointers
that could otherwise direct a future operator toward a retired waiter.

Evidence: `evidence/paper_aio/PAPER_AIO_ACTIVE_AMTNC_V5_POINTER_GLOBAL_SYNC_20260913T012600.json`.
