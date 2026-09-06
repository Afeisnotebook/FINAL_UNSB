# Decision: accept AM-TNC e40 and preserve the live deleted-inode process

Date: 2026-09-07

Status: **E40 FIXED MILESTONE HASH-CLOSED; DELETED-RUNTIME RECOVERY REMAINS HEALTHY**

The 4090A AM-TNC lane reached the preregistered e40 milestone without restarting
the trainer or supervisor that survived the accidental environment cleanup.  The
e40 checkpoint was recomputed on the source host and matches its sidecar
(`e94a2cc2...8f695`); its scientific-state hash is `8fdcaa03...344fb`.  The
metric artifact is retained and hash-bound, but its contents were not parsed.

The isolated recovery runtime successfully loaded the e40 full-state checkpoint
on CPU and verified its schema, step and physical epoch.  The active trainer PID
3446758 still maps the deleted original executable, while the three original
Python entrypoints remain read-only relays to the isolated byte-matched runtime.
The outer guard continues to monitor the existing supervisor with zero restarts
and zero health alerts.

This is the first fixed training milestone created after the cleanup incident.
It materially reduces the loss risk: all work through e40 is now protected by a
source-hashed, independently loadable full state.  The live process must not be
restarted merely to remove the `(deleted)` marker.  If it fails, the frozen guard
may recover from the latest verified complete epoch; at most the incomplete epoch
is at risk.  GPU next-step bitwise equivalence to the still-running deleted-inode
process is not claimed.

No performance value was read, no scheduling or protocol decision was made, and
confirmation20 remains sealed.

Evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_E40_DELETED_RUNTIME_CONTINUITY_20260907T001540.json`
