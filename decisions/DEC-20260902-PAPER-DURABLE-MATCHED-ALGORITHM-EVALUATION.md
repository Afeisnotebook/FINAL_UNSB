# Durable matched evaluation for AM-TNC and ST-CGR

Date: 2026-09-02
Scope: post-e200 algorithm delivery

The first-wave unified evaluator uses 5090A plain because Proposal-only on
5090C has an exact 2000-update runtime relation to that control. This does not
make 4090A and 5090A interchangeable. Reusing the first-wave plain for AM-TNC
would therefore create an invalid cross-host delta even though inference ran
inside one container.

The post-training DAG is consequently partitioned into two legal matched
cohorts. AM-TNC is evaluated with the 4090A plain from which its static lane was
initialized, in a separate output root. ST-CGR is evaluated inside the existing
first-wave output only after its 5090A checkpoints have been verified and the
5090A-plain first-wave cohort has been locked. Both successors evaluate all
fixed e100/e125/e150/e175/e200 checkpoints and produce a terminal method
disposition; neither can select a checkpoint or feed a metric back into
training or scheduling.

ST-CGR's candidate lock, authorization and cross-code runtime gate were copied
from 5090A through a pinned-host, hash-verified, one-shot metadata relay. The
portable evaluation authority binds all three expected hashes. The candidate
lock contains frozen prior small25 evidence, so the receipt truthfully records
that this evidence was transferred; it was not used to schedule, change or
stop the already authorized full-data run.

The first metadata relay attempt stopped before opening a remote connection
because the exact evaluator Python lacks optional `paramiko`. The identical
frozen contract was then executed with the host's existing control-plane
Python, which provides `paramiko`. No checkpoint was loaded and no training or
scientific state was touched by the failed attempt.

Both algorithm successors and their independent health watcher are detached
with PPID 1. A shared GPU lock serializes all evaluator work. Exact process IDs,
paths, hashes, source relations, current states and boundary flags are in
`evidence/paper_aio/PAPER_AIO_DURABLE_ALGORITHM_EVALUATION_SUCCESSORS_20260902.json`.

A post-deployment race audit found that the first ST-CGR successor considered
the imported lane receipt sufficient before checking its completed import-set
membership. The window is extremely short and would only have caused a
fail-closed controller exit, but leaving it would weaken unattended delivery.
Both initial successors were still waiting with zero evaluations, so they and
their watcher were replaced without touching training. The current successor
requires the complete import set and is bound to commit `0492097`; the retired
control files remain archived for audit.
