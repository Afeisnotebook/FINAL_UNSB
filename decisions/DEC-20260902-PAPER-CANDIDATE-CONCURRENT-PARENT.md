# Concurrent parent readiness for a full-data candidate

Date: 2026-09-02

The old paper candidate gate coupled two different requirements: permission to
start an evidence-qualified candidate and permission to compute its final
matched delta.  Requiring parent plain e200 for both forced the new candidate
to wait an additional full training duration even though both independent
lanes start from frozen e0 states and never exchange writable state.

The gate now exposes two explicit modes.  Its conservative default remains
`complete_e200`.  The scheduling path may request `authorized_running` only
after the candidate has a positive source-bound small25 e200 receipt and the
same-host parent plain has a pinned preflight, authorization, e0, runtime twin,
and healthy exact-resume supervisor.  Candidate-code equivalence, disabled
identity, full-state resume, evaluation repeat, capacity and source locks are
unchanged.

This does not authorize an early scientific comparison.  Parent plain and the
candidate must both finish exactly 1,710,600 updates and pass unified terminal
evaluation before any matched delta or paper claim is produced.  It changes
only execution order and can remove roughly 60–70 hours from the full-data
new-algorithm critical path.  An e1 co-residence capacity gate still decides
whether simultaneous execution lowers total makespan; otherwise one lane is
exactly resumed sequentially.
