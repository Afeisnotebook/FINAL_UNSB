# New 5090 endpoint pending reachability and Goal arming

The user supplied `connect.weste.seetacloud.com:44804` again as a new RTX 5090
resource.  DNS resolves, but repeated TCP probes through 00:58:34 +08:00 could
not reach the SSH port.  Authentication therefore never began.  This is an
endpoint-startup or mapping wait, not an algorithm failure and not a reason to
stop the existing paper Goal.

The same endpoint previously resolved to registered host 5090B with GPU UUID
`GPU-578d4047-4c22-8c6a-d216-0f7938e99194`.  Provider reprovisioning can change
that identity, so neither a new-GPU claim nor a duplicate-host claim is made
until the live UUID is recollected.  The active 43172 provider clone is a
different physical GPU and continues the matched-plain trajectory unchanged.

The existing `final-unsb-goal` heartbeat was updated rather than creating a
competing scheduler.  It temporarily checks hourly, retries 44804 without
destructive action, performs the GPU-UUID gate when reachable, refreshes the
then-current experiment DAG and only then selects a non-duplicate,
evidence-authorized task that can close within the real lease.  HJCGR is not
automatically launched merely because the GPU is empty; its full-data cost and
matched-control gates still apply.  AM-TNC follow-up requires a new derivation
from terminal causal evidence, DDSB remains reproduction-incomplete, and no
archived route-1 successor may be revived.

Any subsequently authorized long run must receive a frozen checkout and
fingerprint, full-state checkpoints, training supervisor, outer guard,
progress/health watches, source-bound full and incremental exporters, and an
off-host recovery copy before the monitor returns to its normal two-hour
cadence.  No credentials were written to Git or the automation prompt.  No
paired metric was read, confirmation20 remains sealed, and no healthy process
or queue was changed.
