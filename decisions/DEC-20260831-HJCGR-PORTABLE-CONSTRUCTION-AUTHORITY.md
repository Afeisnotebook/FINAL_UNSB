# HJCGR portable construction authority

## Decision

HJCGR is constructed once from the complete, source-bound 4090 HJ and
Proposal-only evidence. A replay host must consume an immutable portable copy
of that construction evidence. Its own Proposal-only e200 result is independent
cross-runtime evidence and cannot requalify, alter, or block the already-frozen
HJCGR algorithm.

## Reason

The previous 5090 successor waited for its local Proposal-only receipt and then
called the authoritative-host parent gate against that destination trajectory.
This mixed two distinct questions:

1. whether the evidence was sufficient to define HJCGR; and
2. whether the frozen algorithm reproduces on another runtime.

The first question was already answered on the authoritative 4090 run. Reusing
the destination result as a second construction gate could suppress the HJCGR
replay solely because of a host-specific trajectory, even though no formula or
training-control input had changed.

## Boundary

- The portable authority embeds the exact parent-evidence object and binds it to
  the source anchor, causal matrix, reversal atlas, Proposal-only receipt, card,
  and trajectory hashes.
- A destination replay must have byte-identical anchor, causal-matrix, and atlas
  sources. Changed or malformed authority fails closed.
- Destination Proposal-only still runs through e200 and remains visible in the
  host-separated related-algorithm adjudication.
- No destination paired metric enters the HJCGR formula, optimizer, controller,
  activation, or schedule.
- This repair changes orchestration semantics only. It does not change the
  HJCGR formula, implementation sources, hyperparameters, or algorithm
  fingerprint.

## Tests

The regression suite proves that a negative destination Proposal-only trajectory
cannot block replay under a valid source authority, while modified authority or
modified source bindings are rejected.
