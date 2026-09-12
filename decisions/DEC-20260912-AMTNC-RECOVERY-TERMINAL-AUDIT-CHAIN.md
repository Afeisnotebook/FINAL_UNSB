# AM-TNC recovery-lineage terminal-audit delivery

AM-TNC is operationally healthy at completed epoch 180 and has crossed the
former e178-to-e179 failure. The incident was localized to overflow in
float32 intermediate Adam-metric products while the source gradients, Adam
scales and replica vectors remained finite. The recovery retains the original
float32 path whenever finite and recomputes only a nonfinite reduction in
float64. It does not change sampled gradients, the projection, method
hyperparameters, RNG or sampler state.

The old local terminal-audit relay was not a valid continuation authority for
the repaired training lineage: its supervisor died during Windows atomic
state publication and its source still referred to the pre-recovery AM-TNC
run. No AM-TNC audit cell had executed. The old imported checkpoints were
moved to a recoverable archive, not deleted, and a separately named,
hash-bound recovery-lineage relay now exposes e100 and e150 while waiting for
e200. The existing terminal-audit supervisor remains read-only and will not
run the AM-TNC cells until the required recovery-lineage import is complete.

Two deployment mistakes failed closed before changing training or
checkpoints: a remote exporter command-identity mismatch, and a local
old-lineage hash mismatch. The corrected exporter, local relay, their recovery
supervisors and health watchers are healthy. The unified evaluator, AM-TNC and
ST-CGR disposition waiters, and corrected final-delivery successor remain
waiting for their fixed inputs.

This decision repairs provenance and delivery only. It does not adjudicate
AM-TNC performance, does not claim the algorithm succeeds or fails, and does
not authorize confirmation20. The scientific decision remains fixed-e200
evaluation under the registered matched-control relation.
