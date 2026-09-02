# FINAL_UNSB full-data paper runner

This runner is independent of `research.local_route1`. It freezes the
8,553-image-per-side official image-proportional unpaired measure, batch one,
seed 2026 and 200 data epochs (1,710,600 optimizer updates). Paired discovery
targets are evaluation-only; confirmation20 is unaddressable here.

## Required order on every training host

```text
python -m research.paper_aio.run --stage materialize ...
python -m research.paper_aio.run --stage preflight --node-role training ...
python -m research.paper_aio.run --stage resume-gate --lane LANE ...
python -m research.paper_aio.run --stage zero-intervention-gate ...  # Proposal host
python -m research.paper_aio.run --stage evaluation-repeat-gate --lane LANE --checkpoint PATH ...
python -m research.paper_aio.run --stage authorize --lane LANE ...
python -m research.paper_aio.run --stage train --lane LANE --resume ...
```

`train` refuses to start without a current lane authorization. Proposal must
declare either `--matched-plain-mode same_runtime_output_root` or provide an
exact cross-4090 twin receipt with
`--matched-plain-mode exact_cross_4090_cohort --runtime-receipt PATH`.
The 5090 is always a separate runtime and its values are never subtracted from
a 4090 plain trajectory.

A fresh cross-host plain may be scheduled into a slot released by one member
of a co-resident pair.  In that case
`operations.paper_aio_cross_host_plain_successor` can run an exact two-epoch
engineering pause before committing to the remaining 198 epochs.  The second
epoch excludes the e1 milestone evaluation and supplies a clean observed
co-resident training time.  The successor compares projected completion when
continuing immediately versus waiting for the remaining companion, using only
training heartbeats and preregistered isolated-time references.  If the
minimum wall-clock saving is not met, the new plain remains safely paused at
its full-state e2 checkpoint and resumes only after the companion releases the
GPU.  The immutable `CORESIDENT_MAKESPAN_CAPACITY_GATE.json` records the
decision; no paired metric, checkpoint quality, or algorithm setting enters
the gate.

The frozen scientific protocol retains its original conservative 200 GiB
disk default. A task-specific user capacity waiver is supplied as a separate,
hashed operational receipt with `--capacity-override-receipt` and a bound
`--host-label`. The runner recomputes the effective floor from the receipt's
worst-case incremental-write estimate and safety multiplier. An override is
never permission to delete data, checkpoints, or historical evidence.

## Audits and evaluation

```text
python -m research.paper_aio.run --stage evaluation-repeat-gate --lane LANE --checkpoint PATH ...
python -m research.paper_aio.run --stage terminal-audit --lane LANE --checkpoint PATH ...
python -m research.paper_aio.run --stage adjudicate ...
```

The terminal audit is target-blind. It records bridge increment/endpoint
spectra, local and propagated perturbation gains, NFE4-to-NFE5 change, and
forced-time native G/F gradient mean, variance, component conflict and Adam
preconditioned norm. Paired performance is attached only afterwards by the
adjudicator; it never controls training.

Paper complexity uses an immutable checkpoint through `--stage complexity`.
It records per-network parameter counts, fixed NFE 1--5 inference latency, one
complete optimizer-step latency, and CUDA peak memory. The profiler uses one
training input only, never reads the paired target, verifies the source
checkpoint hash before and after, and discards the in-memory model used for
training-step timing. FLOPs are deliberately not claimed because the custom
stochastic bridge and lazy PatchNCE operators are not fully covered by one
audited counter.

Source-host exports can be transferred to one evaluation host with
`operations/paper_aio_export_relay.py`.  The relay pins the SSH host key,
accepts authentication only through a named environment variable, never
persists the credential, and verifies the export receipt, checkpoint and
sidecar hashes before publishing an import set.  It transfers no metric file
and cannot mutate a source checkpoint.  `--relay-id` distinguishes multiple
run roots on one physical source host without changing the source-host label
recorded in the scientific receipts.

Candidate checkpoints are evaluated with `--candidate-authority`.  The
portable authority contains only the frozen lane identity, training
commit/protocol and source hashes; it deliberately excludes performance
values and cannot authorize training.  This removes source-host absolute paths
from unified evaluation without weakening candidate-code verification.

Git contains protocol, code, compact receipts and decisions only. Views,
checkpoints and full logs stay outside the repository.

## Evidence-locked new candidates

A route-1 candidate is not another static lane name.  After a complete
positive small25 e200 trajectory, use the following fail-closed sequence on
the same host/runtime that produced the paper plain reference:

```text
python -m research.paper_aio.run --stage candidate-runtime-gate \
  --candidate-id ID --candidate-terminal-receipt RECEIPT \
  --candidate-trajectory TRAJECTORY --candidate-derivation-card CARD \
  --candidate-implementation IMPLEMENTATION --parent-output PARENT \
  --parent-runtime-receipt PARENT_TWIN --parent-e0 PARENT_E0 \
  --parent-scientific-git-commit COMMIT --parent-protocol-fingerprint FP \
  --parent-readiness-mode authorized_running ...

python -m research.paper_aio.run --stage candidate-lock \
  --candidate-id ID --candidate-terminal-receipt RECEIPT \
  --candidate-trajectory TRAJECTORY --candidate-derivation-card CARD \
  --candidate-implementation IMPLEMENTATION \
  --candidate-runtime-gate CANDIDATE_RUNTIME_GATE \
  --parent-output PARENT --parent-scientific-git-commit COMMIT \
  --parent-protocol-fingerprint FP \
  --parent-readiness-mode authorized_running ...

python -m research.paper_aio.run --stage authorize --lane candidate \
  --candidate-id ID ...
python -m research.paper_aio.run --stage train --lane candidate \
  --candidate-id ID --resume ...
```

The runtime gate independently proves exact parent/candidate e0 scientific
cores, the native 2000-update transition, disabled-candidate identity,
candidate full-state resume, and repeated evaluation.  The evidence lock
deliberately records `full_data_authorized=false`; only a second fresh
authorization bound to the current Git commit and protocol fingerprint may
start the 1,710,600-update run.  A negative/incomplete small25 trajectory,
stale source hash, different host runtime, or missing healthy authorized
parent plain fails closed; matched adjudication separately requires parent
plain e200.

`authorized_running` removes a scheduling-only dependency: after the
candidate's small25 e200 receipt exists, its full run may overlap a healthy,
authorized, exact-resume same-host plain run. It does not relax the comparison
boundary. The parent plain must still reach e200 and enter the unified locked
evaluation cohort before any matched delta or paper claim can be produced.
The default remains `complete_e200`; concurrent readiness must be requested
explicitly and is recorded in the runtime gate, candidate lock, and fresh
authorization.

Training source identity and evaluation randomness are deliberately separate.
All paper checkpoints use the frozen first-wave `68f53a8e...` bundle seed so a
new candidate source fingerprint cannot silently change its rollout noise.
Adjudication checks every late image's domain, stem, order, replicate, NFE and
bundle hash before computing method-minus-plain deltas.  Terminal metrics also
report the macro PSNR/SSIM/LPIPS standard deviation across the five fixed
rollout bundles (population standard deviation, `ddof=0`).

## Posthoc terminal-pathology adjudication

The full-data terminal audit is target-blind. It first freezes and validates
all e100/e150/e200 audits for 4090A plain, 5090C Proposal, 4090A AM-TNC and
5090A ST-CGR, including unchanged parent state/RNG and the complete
X_t-to-NFE5 rollout Jacobian. Only after the exact 12-cell audit state is
complete may `terminal_adjudicate` open source-bound discovery metrics.
The durable successor can then acquire the shared local GPU lock, evaluate the
same 12 imported checkpoints in one frozen evaluator, create the absolute-path
metric binding, and run the adjudicator without an unmonitored manual gap.

The preregistered lead-lag test uses e100-to-e150 diagnostics and labels the
future e150-to-e200 change on the common discovery70, replicate 0 and NFE 5.
It confirms a mechanism only when the same spectral-collapse or perturbation-
amplification rule supports at least two probes and three domains. Metric
bindings use absolute receipt/metric paths, and every metric must come from
one evaluator runtime with the frozen manifest and CRN bundle. A positive
decision authorizes writing a derivation card only; it never starts a repair
module or controls an existing training run.

```text
python -m research.paper_aio.terminal_adjudicate \
  --audit-root AUDITS --metric-bindings METRIC_BINDINGS.json \
  --output TERMINAL_PATHOLOGY_DECISION.json
```

## One-container final evaluation

Training checkpoints never resume across hosts. After each source lane ends,
bind every e100/e125/e150/e175/e200 checkpoint before copying it:

```text
python -m research.paper_aio.run --stage checkpoint-export --lane LANE \
  --epoch EPOCH --checkpoint SOURCE.pt --source-sidecar SOURCE.pt.json \
  --source-host-label HOST --receipt-output EXPORT.json
```

Copy the checkpoint and export JSON to the chosen 4090 evaluation container,
then evaluate them read-only:

```text
python -m research.paper_aio.run --stage input-evaluate ...
python -m research.paper_aio.run --stage unified-evaluate --lane LANE \
  --source-receipt EXPORT.json --copied-checkpoint COPIED.pt ...
python -m research.paper_aio.run --stage unified-lock ...
python -m research.paper_aio.run --stage adjudicate ...
```

`input-evaluate` adds the deterministic degraded-input row using discovery80,
the same resize/metric implementation and the same evaluator environment; it
does not create a trainable lane. `unified-lock` requires that Input receipt
plus all five frozen epochs for plain, Proposal-only, CUT and CycleGAN in one
environment and evaluator fingerprint. It checks every
copied checkpoint hash, metric hash, CRN policy and read-only flag. Only this
cohort can change `PAPER_RESULTS.json` from `FIRST_WAVE_INCOMPLETE` to
`FIRST_WAVE_COMPLETE`; host-separated training remains explicit and no
CUT/CycleGAN-minus-plain matched claim is created.

Unified evaluation is not, by itself, a training-runtime match.  Every imported
metric records its source host, training protocol fingerprint and manifest.
When a static UNSB method and plain have different source-host labels,
`unified-lock` and adjudication additionally require the metric-blind exact
relation in `configs/PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json`.  Same-host
static methods require equal training protocol and manifest; a dynamic
cross-code candidate remains bound by its candidate runtime gate.  Thus a
common evaluator cannot accidentally turn two non-equivalent training hosts
into a matched delta.

One method may have more than one independently proven plain control. The
registry therefore accepts either the legacy single relation object or a list
of relations for that lane, and selects exactly one entry by the method/plain
source-host pair. A future entry is generated from primary receipts before it
is committed:

```text
python -m research.paper_aio.run --stage runtime-relation --lane proposal \
  --method-runtime-receipt METHOD_TWIN.json \
  --plain-runtime-receipt PLAIN_TWIN.json \
  --method-authorization-receipt METHOD_AUTHORIZATION.json \
  --method-source-host-label 5090C \
  --plain-source-host-label 5090B_MATCHED_PLAIN \
  --receipt-output RELATION_CANDIDATE.json
```

The command requires identical 2000-update e0/step cores, protocol, manifest
and normalized runtime environment. It creates a candidate only; a later Git
decision must still append it to the registry. Missing or duplicate host-pair
relations fail closed.

When the standard Proposal relation and the cross-code ST-CGR relation are both
available, prepare the exact Git review payload without editing the tracked
registry:

```text
python -m operations.paper_aio_relation_registry_review \
  --registry configs/PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json \
  --candidate PROPOSAL_RELATION.json --expected-candidate-sha256 SHA256 \
  --candidate STCGR_RELATION.json --expected-candidate-sha256 SHA256 \
  --candidate-state PROPOSAL_SUCCESSOR_STATE.json \
  --expected-candidate-state-sha256 SHA256 \
  --candidate-state STCGR_SUCCESSOR_STATE.json \
  --expected-candidate-state-sha256 SHA256 \
  --required-lane proposal \
  --required-lane G4-01-STRATIFIED-TIME-CONDITIONAL-GF \
  --method-host proposal=5090C \
  --method-host G4-01-STRATIFIED-TIME-CONDITIONAL-GF=5090A \
  --plain-source-host 5090B_MATCHED_PLAIN --output REVIEW_OUTPUT
```

The review command revalidates type-specific proof fields, all primary hashes,
the 2000-update identity, the exact method/plain host pairs, and the immutable
completion state that binds each successor to the candidate path and digest.
It also verifies absence of metric fields and uniqueness against the current
registry. It emits a deterministic proposed registry and a compact receipt. It
never edits Git or authorizes a comparison; Codex must inspect and apply the
proposed object as an explicit registry commit.

`paper_aio_relation_registry_review_successor` can be armed before either
candidate exists. It waits for both immutable completion states, derives their
advertised paths and hashes only after exact release, and runs the same review
command. Its terminal artifact is still only a Git review proposal: it cannot
edit the tracked registry, authorize a comparison, or start evaluation.

First-wave completion is deliberately not the algorithm/claim freeze.
`ALGORITHM_SET.json` keeps DDSB as `REPRODUCTION_INCOMPLETE`, distinguishes
deferred and engineering-blocked lanes, and remains
`FIRST_WAVE_EVIDENCE_READY_CANDIDATES_PENDING` until every evidence-authorized
new algorithm has received its own disposition. Confirmation is still locked
at that point.

The complete fixed sequence can be armed before any checkpoint is available:

```text
python -m operations.paper_aio_unified_evaluation_successor \
  --repo-root CONTROL_CHECKOUT --output-root OUTPUT \
  --import-root VERIFIED_IMPORT_ROOT --data-root DATA \
  --train-view MATERIALIZED_VIEW --manifest FULL_DATA_MANIFEST.csv \
  --lane-source plain=5090A --lane-source proposal=5090C \
  --lane-source cut=5090B --lane-source cyclegan=5090B \
  --gpu-release-state AMTNC_SUPERVISOR.json \
  --gpu-release-status COMPLETE_E200 --gpu-lock EVALUATION.lock
```

This successor remains metric-blind while waiting and while deciding what to
run. It resumes only from hash-validated fixed receipts, uses a GPU lock, and
emits a monitor-recognizable terminal status after unified lock and posthoc
adjudication. It automates first-wave delivery only; later candidate cohorts
still require their separately authorized disposition path.

## Matched posthoc algorithm delivery

A common evaluation container does not make different training hosts matched.
The durable posthoc successor therefore has two explicit modes. `static_pair`
builds an isolated same-host comparison root for 4090A plain and AM-TNC.
`dynamic_candidate` adds ST-CGR only to the already locked cohort containing
its 5090A plain. Both modes fix e100/e125/e150/e175/e200 and emit a separate
algorithm disposition after the terminal adjudicator runs:

```text
python -m operations.paper_aio_algorithm_evaluation_successor \
  --mode static_pair --method-lane amtnc \
  --plain-source-host 4090A --method-source-host 4090A ...

python -m operations.paper_aio_algorithm_evaluation_successor \
  --mode dynamic_candidate \
  --method-lane G4-01-STRATIFIED-TIME-CONDITIONAL-GF \
  --method-source-host 5090A --candidate-authority AUTHORITY.json \
  --candidate-metadata-receipt METADATA_IMPORT.json \
  --first-wave-cohort UNIFIED_EVALUATION_COHORT.json ...
```

Dynamic candidate metadata is imported once with
`operations.paper_aio_candidate_metadata_relay.py`. It verifies the pinned SSH
host key and binds the candidate lock, authorization and runtime gate to hashes
already frozen by the portable evaluation authority. Prior small25 evidence is
transported as immutable provenance, never as a full-data controller.

## Final discovery delivery

`operations.paper_aio_final_delivery_successor.py` waits for the fixed
first-wave, AM-TNC and ST-CGR terminal dispositions. It then profiles every
predeclared e200 model in the 4090 evaluator and emits
`PAPER_ALGORITHM_PORTFOLIO.json`. The compact portfolio preserves the separate
4090A and 5090A matched plains, includes all passing or failing current
implementations, and carries parameter, latency and memory evidence.

This is a discovery-stage delivery, not an automatic claim freeze. It never
chooses a checkpoint, never changes a run, does not claim unaudited FLOPs, and
leaves `paper_claims_frozen=false` and `confirmation_authorized=false` for the
subsequent human/Codex paper review.

## Post-freeze distribution metrics

`research.paper_aio.distribution` implements the preregistered KID/FID stage,
but it cannot run during discovery.  `--stage distribution` requires a freeze
receipt already committed to this repository.  That receipt must freeze the
algorithm, baseline, e200 result and paper-claim set while leaving confirmation
unauthorized and closed.

After that gate, the evaluator uses discovery80 at 128 pixels.  UNSB-family
lanes use the five fixed CRN replicates at NFE=5; deterministic lanes use one
replicate.  Six-domain macro KID is primary.  Pooled KID and pooled FID are
supplementary, and FID is labeled as a 480-image small-sample estimate.  The
receipt records the Clean-FID and Inception identities, RNG-isolated KID seeds,
checkpoint hash, CRN identity and image quantization.  It never opens
confirmation20 or selects a lane/checkpoint.
