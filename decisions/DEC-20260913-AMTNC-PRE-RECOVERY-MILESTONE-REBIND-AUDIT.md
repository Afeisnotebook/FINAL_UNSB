# AM-TNC pre-recovery milestone rebind audit

## Decision

Keep the V7 AM-TNC exporter bound to the current `5676c91` recovery view, but
record both hash namespaces for epochs 150 and 175.  The recovery-view
checkpoints are exact computational copies of the original pre-recovery
milestones with provenance metadata rebound to the admitted recovery commit,
protocol fingerprint and e0 receipt.  They are not a second early trajectory.

## Evidence

A recursive CPU comparison loaded the original and recovery-view checkpoints
for both fixed epochs and compared every transition-defining field.  Network,
optimizer, scheduler, method, Python/NumPy/CPU/CUDA RNG, primary/secondary
sampler, lane and counter state are exactly equal.  The only three differing
leaves are `metadata.git_commit`, `metadata.protocol_fingerprint` and
`metadata.e0_scientific_state_sha256`.

Consequently the checkpoint-byte and sidecar scientific hashes legitimately
differ, but that difference is provenance rebinding rather than model or
training-state divergence.  Final delivery must report the recovery-view
hashes for the files it actually evaluates and retain the original hashes as
the pre-recovery provenance anchors; it must not substitute one namespace for
the other.

## Boundary

No training, checkpoint, supervisor, exporter or queue was modified.  No
paired performance value was read, no best checkpoint was selected and
`confirmation20` remains sealed.  This audit does not validate AM-TNC
performance or upgrade the two-stage recovery to byte-identical continuation.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_PRE_RECOVERY_MILESTONE_REBIND_AUDIT_20260913T200211.json`.
