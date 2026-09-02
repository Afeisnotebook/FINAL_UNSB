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

First-wave completion is deliberately not the algorithm/claim freeze.
`ALGORITHM_SET.json` keeps DDSB as `REPRODUCTION_INCOMPLETE`, distinguishes
deferred and engineering-blocked lanes, and remains
`FIRST_WAVE_EVIDENCE_READY_CANDIDATES_PENDING` until every evidence-authorized
new algorithm has received its own disposition. Confirmation is still locked
at that point.
