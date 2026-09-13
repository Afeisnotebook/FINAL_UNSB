# Stale read-only SSH poll retirement

Four old local PowerShell/OpenSSH poll pairs remained blocked after read-only
status queries issued on September 12 and 13.  Their commands only read AM,
ST-CGR, or Proposal state with `cat`, `find`, `ps`, `nvidia-smi`, and `df`;
they had no monitoring, recovery, training, or delivery responsibility.

After verifying that none contained a launch, training, relay, or termination
operation, all eight wrapper processes were stopped.  A follow-up scan found no
remaining stale FINAL_UNSB SSH poll.  The only remaining long-lived matching
processes are the formal Python relays for the 5090B clone.  The DCLGAN,
ST-CGR, Proposal, and matched-plain trainers remained live.

No remote process, checkpoint, result, sampler, RNG, algorithm, or protocol was
changed.  Paired performance was not read and confirmation20 remains sealed.
