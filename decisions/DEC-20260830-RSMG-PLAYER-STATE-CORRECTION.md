# RSMG player-state semantic incident and correction boundary

## Decision

The trajectory identified as `G1-02-SAMPLING-VARIANCE` is frozen as an
`implementation_failure`. It is not a scientific negative result and cannot be
ranked, resumed, revised from its checkpoints, or used to close the sampling
variance mechanism.

The frozen derivation card requires the stochastic replicas for each native
player to be conditionally iid at that player's current state. Native UNSB
commits D, then E, then joint G/F. The implementation constructed both views
once before D and reused them for E and G/F after the opponents had changed.
The G/F estimator therefore did not satisfy the premise of the registered
conditional unbiasedness proof.

## Correction boundary

The engineering replacement will receive a new candidate ID and fingerprints,
will restart from the common e0, and will preserve the invalid artifacts. D/E
may share one iid two-replica bundle at their common pre-opponent state, matching
their independent native losses. After both opponent steps are committed, G/F
must receive a fresh iid two-replica bundle generated at the updated opponent
state. This yields an unbiased estimator for each native player at the state at
which that player is actually optimized, while retaining batch-1 PatchNCE and
the native D/E/G/F update order.

Before any long run, a new coupled-game gate must prove generation order and
RNG provenance, one-replica exact identity, exact full-state resume, and the
fixed-state mean/variance invariant for the fresh G/F estimator. Passing the
old abstract estimator gate is insufficient.

4090 and 5090 will each run the correction as an isolated batch-1 stream from
e0 alongside the unaffected BVCP stream. Scientific batch size remains one;
the additional hardware is used for independent concurrent trajectories, not
to change the UNSB game.

No paired metric caused this correction and no paired metric may control the
replacement. `confirmation20` remains sealed.
