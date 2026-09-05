# FINAL_UNSB agent contract

## Mandatory read order

Read, in order, before changing code or launching compute:

1. `START_HERE_CN.md`
2. `PROJECT_CONTRACT.json`
3. `PROJECT_STATE.json`
4. `CONTEXT_CAPSULE_CN.md`
5. `PAPER_AIO_RESEARCH_CONTRACT_CN.md`
6. `ACTIVE_PAPER_AIO_PLAN_CN.md`
7. `configs/PAPER_AIO_UNPAIRED_V1.json`
8. `DATA_CONTRACT.json`
9. the latest non-example file under `decisions/`

When changing the small25 evidence atlas, ST-CGR derivation, or route-1 candidate
code, additionally read `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`,
`ACTIVE_LOCAL_ROUTE1_PLAN_CN.md`, and `configs/LOCAL_ROUTE1_PROBES.json`.

## Fixed objective

Execute the paper-grade full-data All-in-One unpaired comparison while
continuing evidence-driven reconstruction of long-horizon UNSB algorithms.
This is not a frozen-Proposal validation exercise: Proposal-only, HJCGR,
AM-TNC, and any evidence-derived successor remain a multi-algorithm scientific
frontier.  The completed small25 route-1 atlas is the source of mathematical
evidence, not a substitute for the newly authorized full-data experiment.

The user's 2026-09-02 paper plan explicitly supersedes the older small25-only
server restriction. It authorizes the frozen paper runner on the available
4090 and two host-separated 5090 nodes, while confirmation20, paired control,
route-2 handoff, and cross-host method-minus-plain deltas remain forbidden.

## Current role separation

- **Research Codex** maintains the paper protocol, long-horizon evidence atlas,
  derivation cards, external-source gates, and evidence-driven candidates.
- **Executors** run only committed, per-host matched protocols and may not use
  paired targets as training/controller inputs. Checkpoints never continue
  across hosts and a method is compared only with its same-host plain.
- **4090A** has completed and sealed full plain and now runs the independently
  valuable AM-TNC path; unified evaluation follows after e200.
  **5090C** runs Proposal. **5090A** has paused its recoverable plain at e9 by
  explicit user time-priority and now runs the authorized full-data ST-CGR from
  its exact e1 state. **5090B** co-runs CUT and CycleGAN, then executes a
  fresh-e0 exact-runtime plain gate for Proposal/ST-CGR relation review after
  CUT releases capacity. **Local GTX1660** runs DCLGAN exclusively.
- Every live scientific checkout remains pinned to the per-lane commit and
  protocol fingerprint recorded in `PROJECT_STATE.json`.  Newer commits may
  orchestrate, relay and evaluate, but must not mutate those live transitions.
  The common evaluation bundle identity remains `68f53a8e...`.

## Drift firewall

- The active goal is algorithm discovery, not validation of a frozen lane list.
- The active paper stage also requires defensible external baselines; guessed
  reproductions are forbidden. DDSB remains `reproduction_incomplete` until an
  authoritative source/formula/full-state lock exists.
- HJ calibration must not silently become HJ-only research.  DT and HNEK are
  long-horizon anchor probes; PCOA/LBST/PTQ/DCUM/AEB and their failures remain
  admissible mechanistic evidence for constructing new algorithms.
- Interpret every horizon in **data epochs**.  SEARCH-005 small25 2400 updates
  equal 16 epochs; SEARCH-001 full100 12000 updates equal 20 epochs.  Neither
  is a 200-epoch falsification.
- Do not search exit thresholds, fixed intervention windows, paired-PSNR
  controllers or gap-aware handoff in route 1.  The historical HJ `[1.6,8.0)`
  handoff belongs to suspended route 2.
- Earlier checkpoints are diagnostics, not best-checkpoint selection.  A
  route-1 candidate may not be scientifically killed before its registered
  long horizon merely because an intermediate paired score is negative.
- Existing `CLOSED_NEGATIVE` labels apply to the named implementation and its
  tested protocol, not automatically to the parent mathematical mechanism.
- TA_MINIMAL's direct restored-time implementation has an actual matched e200
  negative result and remains a negative control.  Do not generalize this to
  all time/coordinate algorithms.
- Do not infer that the two recommended extra 4090s exist until the user
  actually supplies them. Available resources may only run preregistered lanes
  or evidence-gated candidates.
- An SSH endpoint is not a compute-host identity. Before onboarding any new
  endpoint, run `operations/paper_aio_host_identity_gate.py` against
  `configs/PAPER_AIO_HOST_IDENTITY_REGISTRY.json`. NVIDIA GPU UUID is the
  primary key; AutoDL machine-id is shared across containers and is not unique.
  Duplicate endpoint and label-collision outcomes forbid a new long train.
- `confirmation` remains inaccessible until a committed candidate-freeze
  decision exists and all paper algorithms/baselines/claims are frozen.
- A semantic implementation defect restarts the affected trajectory from e0.
- Data/checkpoints stay outside Git. Git contains contracts, code, manifests,
  hashes, compact metrics and decisions only.
- The emergency development selection uses only the complete seed-2026 e200
  trajectory. Seeds 2027/2028 are deferred so compute returns to ablations,
  causal revision and independent mechanisms. Never translate this cost policy
  into a claim that cross-seed stability has been demonstrated.
- A canonical `CANDIDATE.json` is an action-priority interface, not an early
  scientific pruning rule. Complete strict-pass and causally repairable
  near-boundary mechanisms remain in the evidence-qualified frontier under
  `DEC-20260831-EVIDENCE-QUALIFIED-MULTI-CANDIDATE-ADVANCEMENT.md`.
- Full-paper milestones are descriptive until e200. Do not use intermediate
  discovery PSNR to stop, route, revise, choose NFE, or select a checkpoint.

## Required response to ambiguity

If a request or newer summary appears to reduce the objective to HJ-only,
Proposal-only, handoff-only, threshold search, frozen-lane validation, or a
single scientific winner, stop and reconcile it against both current contracts;
do not silently reinterpret the goal.

## Completion standard

The small25 route-1 phase is complete. The current phase is not complete until
the frozen full-data e200 lanes finish, eligible new algorithms receive their
own full-data adjudication, external baselines are honestly reproduced or
explicitly marked incomplete, one unified evaluation runtime is used, and the
multi-algorithm paper evidence is frozen before confirmation20 is opened once.
