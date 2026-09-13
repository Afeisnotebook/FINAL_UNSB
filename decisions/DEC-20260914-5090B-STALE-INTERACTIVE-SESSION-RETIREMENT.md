# 5090B stale interactive-session retirement

Two unsupervised local interactive connection pairs remained open against the
5090B clone endpoint on port 43172: one OpenSSH shell and one Paramiko/getpass
session.  Neither process pair participated in training, incremental relay,
final relay, recovery, or health monitoring.

After exact endpoint and command-type checks, the four interactive processes
were stopped.  The six formal local relay/recovery/health processes remained
live, and the remote supervisor, trainer, guard, progress watcher, and health
watcher remained live with the trainer still present on the GPU.  No live local
process points to retired port 44804.

No credential value is recorded in this evidence.  No checkpoint, RNG,
sampler, training process, result, algorithm, or protocol was changed.  Paired
performance was not read and confirmation20 remains sealed.
