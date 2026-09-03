# Progress-aware supervisor failure budget

The paper-lane supervisor formerly counted every child-process failure over its
entire lifetime toward `maximum_consecutive_failures`.  A successful exact
resume that advanced many complete data epochs did not reset that streak.  On a
multi-day run, three unrelated infrastructure failures could therefore produce
`BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE` even though the lane had made
durable progress between them.

Commit `a89874ff378dbb5dd7721efb87a0345536d621ff` changes only operational retry
accounting.  The supervisor records the complete atomic checkpoint step before
and after each child invocation.  If the child advances to a later checkpoint,
the previous streak is cleared and the terminating invocation becomes failure
one.  If no later checkpoint exists, failures continue to accumulate and the
third consecutive same-state failure still fails closed.  A corrupt JSON
sidecar or a sidecar for another lane cannot claim progress.  The child command,
scientific checkout, seed, sampler, protocol, target updates and resume loader
are unchanged.

No healthy live supervisor or trainer was restarted to adopt this change.  The
patch is the required operational source for future lane launches and for any
future reviewed supervisor replacement after an infrastructure exit.  Existing
progress sentinels and the active Goal continue to watch the currently loaded
supervisors.  This avoids trading a speculative control-plane improvement for
an interruption of healthy training.

The targeted supervisor suite passed 5 tests and the full repository passed 645
tests.  No checkpoint or performance file was opened, no paired value informed
the change, and confirmation20 remains sealed.
