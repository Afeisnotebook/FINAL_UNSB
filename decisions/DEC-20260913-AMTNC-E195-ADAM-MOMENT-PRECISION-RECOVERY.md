# AM-TNC e195 Adam second-moment precision recovery

AM-TNC reached a new fail-closed numerical boundary after completing e194.
Two independent attempts from the same finite e194 full state failed at the
e195 checkpoint with a non-finite Adam `exp_avg_sq` tensor. The e194 networks,
optimizer states, samplers, and RNG states were finite. A read-only audit found
that a finite but extreme late gradient could overflow when squared in
float32, even though the normalized Adam parameter update remains finite.

This is an implementation-capacity failure, not a paired-performance result
and not evidence that the AM-TNC mechanism is scientifically ineffective. It
does show that the current AM-TNC trajectory can expose substantially more
severe optimizer conditioning than plain UNSB, so numerical robustness and
its cost must be reported as part of the method if the final e200 result is
scientifically useful.

The repair preserves the mathematical Adam update instead of clipping the
gradient, skipping the step, changing a hyperparameter, or suppressing an
AM-TNC correction. Before `Adam.step`, a float64 conservative bound determines
whether the next second moment is representable in float32. Only an affected
`exp_avg_sq` state tensor is promoted to float64. Every representable step
keeps the prior bitwise path, and full-state resume explicitly restores the
higher-precision optimizer state rather than allowing PyTorch to silently
cast it back to parameter dtype.

The implementation is isolated on branch
`amtnc-e195-adam-moment-recovery` at commit `d8ee501`. Local and remote tests,
including CUDA promotion and resume-dtype checks, passed. The finite e194
state was copied off host, migrated with identical dynamics-only hashes, and
completed one real update under the repaired source. A guarded source-bound
production chain now replays e195 with zero supervisor failures, zero guard
restarts, and no access to paired performance or confirmation20.

The earlier provider cleanup is not the root cause of this failure. It removed
the original runtime path and made recovery fragile, but the Adam overflow was
reproduced deterministically in the isolated restored runtime. No completed
epoch is lost and no healthy unrelated process was touched.

This decision does not yet close the incident. The decisive gate is a complete
finite e195 checkpoint with a recorded precision-promotion event and a
successful full-state reload. Only after that gate may the same fixed protocol
continue to e200. A further failure must fail closed and be localized; it must
not trigger another blind restart. Fixed e200 matched evaluation remains the
only performance adjudication.
