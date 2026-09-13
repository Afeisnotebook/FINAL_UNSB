# DCLGAN exporter dynamic-child continuity

## Decision

Keep the current DCLGAN source-export child PID `3944` under supervisor PID `19524`. Do not restart or hot-replace it merely because the formerly recorded PID `20760` is absent.

The supervisor log proves that the preceding child exited while atomically replacing its JSON state on Windows (`WinError 5`). This is the already classified publish-lock failure mode, not a DCLGAN training or algorithm failure. The supervisor automatically started the current child, which has remained healthy and is waiting for e200 without copying a checkpoint.

The downstream local-to-4090A push chain and current V7 DCLGAN evaluation/addendum chain are also alive and waiting on their legitimate dependencies. At e200 they remain responsible for source-bound export, publish-last import and common-runtime evaluation.

If PID `3944` later exits, the frozen supervisor and its dynamic state—not this PID value—remain authoritative. No healthy training, scientific protocol, paired metric or confirmation20 state was changed.

Compact evidence: `evidence/paper_aio/PAPER_AIO_DCLGAN_EXPORTER_DYNAMIC_CHILD_CONTINUITY_20260913T194600.json`.
