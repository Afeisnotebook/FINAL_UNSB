# AM-TNC recovery relation and V5 delivery gate

AM-TNC is operationally healthy and had completed e182 at 1,556,646 updates
when this decision was captured. The active trainer and its recovery supervisor
were not restarted, modified, or migrated. The epoch-178 incident remains a
method-specific numerical implementation defect: finite Adam-metric vectors
overflowed only when their float32 square/cross products were reduced. It is
not evidence that AM-TNC is scientifically ineffective, and its restoration
benefit remains unadjudicated until the fixed e200 evaluation.

The post-e200 path contained a real fail-closed defect. The recovered AM-TNC
checkpoints carry protocol fingerprint `d2ac81...`, whereas the same-host plain
trajectory carries parent fingerprint `68f53a...`. The previous evaluator
would therefore reject the comparison as an unexplained runtime mismatch even
though the pre-result migration, localization, replay, and unchanged-state
receipts already established a narrower method-only relation. Commit
`a738846` adds that explicit relation and removes the false `same_source_runtime`
label from final delivery.

The recovery namespace rematerializes fixed e100/e125/e150/e175 checkpoints
with the recovery provenance fingerprint. This does not mean they were trained
under the recovery code: the migration gate compared every source checkpoint
and destination after stripping provenance metadata and required identical
dynamics-only hashes. Commit `9214838` binds those four hashes, makes the
artifact normalization explicit, and accepts only the resulting audited
relation. The true intervention boundary remains e178; no byte-identical
runtime claim is made.

The final read-only probe used the real AM-TNC e150/e175/recovery identity and
the real 4090A plain exports. All three fixed late comparisons passed the
audited relation, and final-portfolio validation passed. The replacement V5
AM evaluator and final-delivery waiter are pinned to `9214838`, use the existing
dedicated evaluator runtime, and are healthy in waiting states. Obsolete V2,
V3, and V4 AM/final waiters were terminated only after their replacement was
healthy; their files were retained. Existing unified/ST-CGR evaluators and all
training/export processes were preserved.

Scientific boundaries are unchanged: no paired performance value was read,
e200 remains the primary checkpoint, confirmation20 remains sealed, and this
decision authorizes no algorithm selection. The next adjudication is the
fixed e200 matched evaluation, not another training change.
