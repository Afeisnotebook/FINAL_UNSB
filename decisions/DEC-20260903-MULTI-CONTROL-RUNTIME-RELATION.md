# Multiple exact runtime controls for one method

Date: 2026-09-03

The original paper runtime registry stored one relation object per method lane.
That was sufficient while Proposal on 5090C had only 5090A plain, but it would
force the new 5090B fresh-e0 plain either to overwrite the original relation or
to remain unusable. Overwriting would erase provenance and make reruns depend
on whichever control happened to finish last.

The evaluator now accepts either the original object or a list of exact
relations per lane. It selects exactly one relation by the method and plain
source-host labels present in the imported metric receipts. No match fails
closed; duplicate matches fail as ambiguous. Existing single-relation files
remain valid and unchanged.

The new `runtime-relation` stage materializes a relation candidate only from
the method runtime-twin receipt, the prospective plain runtime-twin receipt and
the method authorization receipt. Both twins must be exact 2000-update cohort
passes with identical e0/step cores, protocol, manifest and normalized runtime
environment. The authorization must hash-bind the method twin. The stage
rejects confirmation access, differences, approximate status and overwriting.

This implementation does not pre-authorize the future 5090B relation. The
5090B receipt does not exist until CUT e200 releases the slot and the gate
actually runs. Only after that primary evidence is reviewed may its generated
candidate be appended to the committed registry. No training, metric reading,
delta calculation or confirmation access is performed by this change.

