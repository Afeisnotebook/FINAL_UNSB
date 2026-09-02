# Durable health watch for paper long runs

Date: 2026-09-02  
Scope: operational liveness only

The existing paper supervisors exactly resume a child that exits, but they do
not distinguish a healthy long computation from a child process that remains
alive while its epoch heartbeat stops advancing. Four independent host-local
health watchers now close that visibility gap for 4090A, 5090A, 5090B and
5090C.

Each watcher runs from the source-bound control commit `a51d4cd`, is detached
with PPID 1, and refreshes a machine-readable state every 60 seconds. It checks
only process liveness, allow-listed status/epoch fields, heartbeat age,
scientific-boundary flags and real remaining disk capacity. It never loads a
checkpoint, copies a loss, reads performance, changes a scheduler, kills a
process, or restarts training. A problem therefore becomes a durable alert for
Codex to inspect and recover under the existing authorization; the monitor
cannot silently change the experiment while trying to repair it.

The 4090A watcher covers plain, its exporter, the AM-TNC successor/future lane,
and all four remote import relays. The 5090A watcher covers plain, its exporter
and the ST-CGR continuation controller. The 5090B and 5090C watchers cover each
running lane, supervisor and exporter. All four refreshed beyond their initial
write, all observed states are healthy, and every watcher has PPID 1.

5090B exposed a control-plane deployment incident: cloning from a linked
worktree yielded an object-incomplete checkout, and a subsequent direct GitHub
clone did not finish within the bounded wait. Neither attempt started the
watcher or touched training. Deployment recovered through a verified complete
Git bundle and `git fsck`; the failed directories are preserved for audit.

The 5090A disk decision continues to use `USER_CAPACITY_OVERRIDE`. Its watcher
compares approximately 184.5 GiB free against the actual conservative 48 GiB
requirement; it does not reintroduce the removed fixed 200 GiB gate.

Exact PIDs, paths, contract/state hashes, thresholds, disk calculations and the
recovery record are in
`evidence/paper_aio/PAPER_AIO_DURABLE_HEALTH_WATCH_DEPLOYMENT_20260902.json`.
