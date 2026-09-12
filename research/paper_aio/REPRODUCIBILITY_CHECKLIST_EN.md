# Reproducibility and Artifact Checklist

Status: `PRE-RESULT / NO EMPIRICAL CLAIM`

This checklist defines what must be disclosed and released for the controlled
FINAL_UNSB All-in-One study. It is not an empirical result and does not select
a method. The executable protocol, frozen theory bundle, baseline portfolio,
and admitted runtime-relation registry override this prose if they differ.

## A. Immutable study identity

The final artifact package must record all of the following for every reported
lane:

- Git commit and a clean-worktree receipt for the training source;
- protocol fingerprint and lane-configuration SHA256;
- full data-manifest SHA256
  `02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744`;
- common-e0 scientific-state SHA256 for each comparison cohort;
- host label, GPU UUID, GPU model, runtime receipt, and container or environment
  identity;
- seed, output directory, exact command, start time, terminal time, and whether
  another process shared the GPU;
- checkpoint, sidecar, metric artifact, export receipt, and final result hashes.

Changing any of these values creates a new run identity. A copied checkpoint is
not admitted until its source receipt, copied-file hash, sidecar, scientific
state, training commit, protocol fingerprint, and source host have all been
verified.

## B. Data and split disclosure

The controlled study uses one six-domain All-in-One model. The final paper and
artifact README must name and cite every source dataset, state its license or
redistribution restriction, and publish the manifest-building procedure even
when images cannot be redistributed.

The frozen manifest contains:

- 8,553 training identities on each side;
- 480 discovery identities, or 80 per domain;
- 120 sealed confirmation identities, or 20 per domain.

The released manifest must contain only identifiers and paths permitted by the
dataset licenses. Its row count, partition counts, domain counts, stem
uniqueness, per-file SHA256 values, and aggregate SHA256 must be checked before
training and again before unified evaluation.

Training follows `official_image_proportional_unpaired`: each data epoch uses a
seeded permutation of all A identities, while B is independently sampled from
the complete B marginal. Paired target paths, domain labels, and discovery
metrics are unavailable to the sampler and training controller. Domain
imbalance is handled by six-domain macro reporting, not by changing the main
training measure.

## C. Frozen training protocol

For the UNSB family, report and preserve:

- seed 2026, batch size 1, 128-by-128 resize-and-crop inputs, no flip, and zero
  data-loader workers;
- 8,553 optimizer updates per data epoch and 1,710,600 updates over 200 constant
  data epochs;
- Adam learning rate (10^{-4}), betas ((0.5,0.999)), and no decay stage;
- GAN/SB/NCE weights of one, \(\tau=0.01\), five bridge steps, PatchNCE layers
  0/4/8/12/16, temperature 0.07, and 256 patches;
- deterministic flags, TF32 state, CUDA/cuDNN versions, PyTorch/torchvision
  versions, and a complete package lock.

The paper must state the number of stochastic views and affected players for
every proposed operator. Proposal-only and ST-CGR modify G/F after the realized
D/E commits. AM-TNC modifies D, E, and G/F at their respective sequential
states. These extra views must be reflected in wall time and compute reporting;
they must not be described as a larger image batch.

External baselines retain their source-bound objectives. CUT, CycleGAN, and
DCLGAN use the controlled 100-epoch constant plus 100-epoch linear-decay
schedule, learning rate (2\times10^{-4}), and the same data exposure, image
size, and seed. A baseline that has not passed its authoritative formula,
optimizer-order, checkpoint/resume, and evaluation gates receives no reproduced
number. In particular, DDSB remains `REPRODUCTION_INCOMPLETE` until those gates
pass.

## D. Minimum executable command record

Each host-specific run receipt must preserve the fully resolved values of
`<output>`, `<manifest>`, `<data-root>`, `<train-view>`, `<host-label>`, and
`<gpu>`. The public command skeletons are:

```text
python -m research.paper_aio.run --stage preflight \
  --output <output> --manifest <manifest> --data-root <data-root> \
  --train-view <train-view> --host-label <host-label>

python -m research.paper_aio.run --stage materialize \
  --output <output> --manifest <manifest> --data-root <data-root> \
  --train-view <train-view> --host-label <host-label>

python -m research.paper_aio.run --stage resume-gate --lane <lane> \
  --output <output> --manifest <manifest> --data-root <data-root> \
  --train-view <train-view> --gpu <gpu>

python -m research.paper_aio.run --stage authorize --lane <lane> \
  --output <output> [--matched-plain-mode <mode>] \
  [--runtime-receipt <runtime-receipt>]

python -m research.paper_aio.run --stage train --lane <lane> --resume \
  --output <output> --manifest <manifest> --data-root <data-root> \
  --train-view <train-view> --gpu <gpu>

python -m research.paper_aio.run --stage evaluate --lane <lane> \
  --epoch <fixed-epoch> --checkpoint <checkpoint> --output <output> \
  --manifest <manifest> --data-root <data-root> --train-view <train-view> \
  --gpu <gpu>
```

Candidate and imported-checkpoint commands must additionally record the frozen
candidate authority or source-export receipt. Supervisor, exporter, relay, and
evaluator commands must be published beside their contract hashes. Secrets, SSH passwords, and private host keys must never appear in Git artifacts.

## E. Full-state and recovery requirements

Permanent full-state checkpoints are required at e1/e5/e10/e20/e40/e60/e80/
e100/e125/e150/e175/e200, with `latest` atomically replaced after every complete
epoch. A valid state includes:

- all method networks and method-specific state;
- every optimizer and scheduler;
- image pools where applicable;
- Python, NumPy, CPU, and CUDA RNG states;
- independent primary and secondary sampler states;
- global update, physical data epoch, source commit, and protocol fingerprint.

Continuous execution and split execution must pass the registered exact-resume
gate. Recovery must preserve the source host/runtime cohort and use a frozen
command unless a separately registered method-only numerical recovery relation
binds the unchanged transition state and discloses the non-identical runtime.
A healthy run is never restarted merely to remove a diagnostic label.
If an environment or process is damaged, the artifact record must distinguish
the incident, preserved state, recovery runtime, executable probe, remaining
equivalence limit, and any recomputed work.

The 4090A AM-TNC cleanup and epoch-178 overflow incident is therefore disclosed
as an engineering provenance item. The original process first retained a
deleted interpreter inode, then stopped when finite Adam-metric vectors
overflowed only in float32 square/cross-product reductions. The admitted
continuation uses a hash-audited isolated runtime, unchanged transition state,
bitwise-identical finite path, and float64 only for a non-finite metric
reduction. This fact neither establishes nor invalidates empirical benefit and
must not be described as byte-identical runtime continuation. The single
recovery export namespace rematerializes e100/e125/e150/e175 only after the
migration gate proves a dynamics-only hash identity for each source checkpoint;
the new provenance fingerprint must not be mistaken for recovery-code training
at those earlier epochs.

## F. Runtime cohorts and legal comparisons

A matched delta is legal only when the runtime-relation registry admits the
method/plain pair. The receipt must bind both source commits, protocol
fingerprints, manifests, e0 scientific states, deterministic flags, hardware
identity, and the relevant twin or reviewed relation evidence.

- AM-TNC is compared only with its same-host 4090A plain and, after epoch 178,
  only through the registered metric-blind method-only recovery relation.
- Proposal and ST-CGR may use the admitted fresh-e0 5090 relation only after the
  committed registry and imported checkpoint receipts pass.
- Results from a non-admitted host are absolute trajectories, never silently
  merged matched deltas.
- Cross-host wall time is operational telemetry, not an algorithmic speedup
  ratio.

Co-resident training must be declared. Capacity admission is based on a
metric-blind makespan gate; it cannot use intermediate image-quality values.

## G. Frozen evaluation and statistical reporting

Evaluation uses lane-blind common-random-number bundles. The artifact package
must publish the bundle fingerprint and per-image records, not only aggregate
means.

- Fixed trajectory epochs use 70 discovery images per domain for PSNR/SSIM.
- LPIPS is added at e100/e125/e150/e175/e200.
- UNSB-family e100/e150/e200 reports NFE 1 through 5; NFE 5 is primary and is
  not selected from observed results.
- The main table uses e200. Sustained behavior uses the fixed
  e150/e175/e200 triplet. No best-checkpoint result is allowed.
- Terminal evaluation uses all 80 discovery images per domain and five fixed
  rollout bundles, reporting stochastic mean and dispersion.
- Report six-domain macro values, every domain value, positive-domain coverage,
  worst-domain behavior, absolute trajectories, parameter count, peak memory,
  optimizer-step time, and stochastic-view count.
- Report KID as the primary distribution metric and FID as supplementary with
  its small-sample limitation.

The first full-data wave uses a single seed (2026) and must be described as such. It cannot
support a multi-seed stability claim. The 20 confirmation images per domain
remain sealed until the method set, runtime relations, claims, tables, and
figures are committed. Confirmation is opened at most once and cannot change an
algorithm, NFE, checkpoint, or claim.

## H. Required public artifact inventory

The final release must contain, or explain the lawful omission of:

1. source commit, environment lock, protocol, manifest builder, and manifest;
2. lane authorization, runtime relation, common-e0, and exact-resume receipts;
3. fixed checkpoint sidecars and source-bound export/import receipts;
4. per-image trajectory and terminal metrics with evaluation-bundle identity;
5. terminal causal-audit records and proof that parent state/RNG were restored;
   the final terminal-pathology decision must bind all 12 fixed target-blind
   audit receipts and all 12 posthoc metric receipts before claim freeze;
6. parameter, memory, optimizer-step, NFE, KID, and FID receipts;
7. algorithm derivation cards, pseudocode, identity/self-null or unbiasedness
   boundaries, and implementation hashes;
8. external-baseline source lineage and any reproduction failure record;
9. committed claim-review decision, freeze receipt, deterministic table and
   figure receipts, and the single confirmation-session receipt if authorized;
10. a limitations statement covering single-seed evidence, stochastic compute,
    small-sample distribution metrics, deferred methods, and runtime incidents.

## I. Pre-submission audit

Before the abstract or conclusion is finalized, a reviewer independent of the
training scheduler must verify:

- every numerical claim resolves to a source-bound fixed-e200 or sustained
  artifact;
- every delta has an admitted matched relation;
- every proposed method is represented in the result branch, including negative
  outcomes;
- deferred methods are not called falsified, and reproduction-incomplete
  baselines are not assigned invented values;
- covariance or unbiasedness statements stay within the frozen pre-Adam scope;
- no statement implies that variance reduction guarantees image quality,
  convergence, or terminal singularity repair;
- the committed claim freeze hash-binds a complete positive or negative
  terminal-pathology adjudication, and that adjudication did not control or
  modify training;
- code, protocol, theory, tables, figures, and manuscript claim hashes match the
  committed freeze;
- confirmation20 was either never opened or was opened exactly once under the
  committed authorization.

Only after this audit may result-dependent words such as *improves*,
*outperforms*, *stable*, or *best* enter the manuscript.
