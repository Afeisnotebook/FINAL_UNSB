# DCLGAN source-bound adapter and formal local gate

Date: 2026-09-03

Superseded for cross-platform authorization by
`DEC-20260903-DCLGAN-PORTABLE-GATE-AND-DURABLE-QUEUE.md`: this commit's raw-byte
source hashes were CRLF-sensitive.  Its completed local gate remains provenance,
not a portable authorization and not a scientific failure.

DCLGAN is the next external-baseline engineering target, not a substitute for
Proposal, ST-CGR or AM-TNC.  The adapter is frozen at commit
`f03b3ffbfc00cd2b3f740057fb9a391b1ab05add` and binds the author repository at
`f7a7b8e2712a5e2d13a535e5e64bb310546e949d` by tracked source hashes.  The
upstream repository is not vendored because its NVIDIA NC/StyleGAN2-derived
license boundary must remain explicit.

The adapter preserves the official DCLGAN objective while applying the frozen
paper exposure protocol.  It saves both generators, both feature networks,
both discriminators, all optimizers and schedulers, the deterministic sampler,
and Python/NumPy/CPU/CUDA RNG state.  The data-dependent-initialization batch is
reused by the first optimizer update.  Read-only evaluation uses `G_A` for the
degraded-A to clean-B direction and rejects confirmation20 before manifest or
image access.

PyTorch 2.6 reports CUDA `ReflectionPad2d` backward as nondeterministic.  The
adapter therefore replaces exactly the expected 50 reflection-pad modules with
a slice/flip/concatenate implementation.  Its forward is bit-exact to the
upstream pad and its CUDA backward is repeatable; any future coverage change
fails closed.  This is an engineering determinism adaptation, not an algorithm
change, and must be disclosed in reproduction materials.

Short CUDA exact-resume and discovery70 repeat gates passed, but they are only
diagnostic.  A clean frozen checkout is now running the formal local gate:
preflight, confirmation lock, continuous-1000 versus 500+resume, an independent
1000-update capacity run, repeated update-1000 discovery70 evaluation, and
authorization.  The supervisor and health watcher are durable and metric
blind.  No remote GPU is assigned, and no full-data DCLGAN training is
authorized at this point.

Passing the local gate will make the implementation portable enough to queue;
it will not prove a target runtime.  The eventual training GPU must repeat the
same complete gate from this frozen commit and obtain a host-bound authorization
before e200 may start.  Existing 4090A, 5090A, 5090B and 5090C training was not
interrupted or modified for this work.
