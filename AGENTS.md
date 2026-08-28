# FINAL_UNSB agent contract

## Mandatory read order

Read, in order, before changing code or launching compute:

1. `START_HERE_CN.md`
2. `PROJECT_CONTRACT.json`
3. `PROJECT_STATE.json`
4. `CONTEXT_CAPSULE_CN.md`
5. `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`
6. `ACTIVE_LOCAL_ROUTE1_PLAN_CN.md`
7. `configs/LOCAL_ROUTE1_PROBES.json`
8. `DATA_CONTRACT.json`
9. the latest non-example file under `decisions/`

## Fixed objective

Continue local route-1 research on the GTX 1660 and discover/reconstruct a new
UNSB algorithm whose benefit can remain valid over a true 200-data-epoch
horizon.  DT, HJ, HNEK, PCOA and the later sampling/teacher/latent mechanisms
are evidence probes, not a frozen candidate list.  HJ is the first temporal
positive control because it owns the clearest historical delayed-benefit
trajectory; it is not the only research direction or the presumed winner.

The prior four-server/four-lane plan is suspended.  Do not prepare or authorize
4090 work unless the user explicitly reopens it after the local route-1 gate.

## Current role separation

- **Research Codex** maintains the long-horizon evidence atlas, performs
  lineage audits, constructs derivation cards and implements evidence-driven
  route-1 candidates.
- **Local executor** runs only committed matched local protocols and may not
  use paired targets as training/controller inputs.
- Historical server tooling remains in the repository for provenance only and
  is inactive during this phase.

## Drift firewall

- The active goal is algorithm discovery, not validation of a frozen lane list.
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
- Do not launch 4090/server work in the current phase.
- `confirmation` remains inaccessible until a committed candidate-freeze
  decision exists. Discovery results may not change code or lane settings.
- A semantic implementation defect restarts the affected trajectory from e0.
- Data/checkpoints stay outside Git. Git contains contracts, code, manifests,
  hashes, compact metrics and decisions only.

## Required response to ambiguity

If a request or newer summary appears to reduce the objective to HJ-only,
handoff-only, threshold search, frozen-lane validation or immediate 4090 work,
stop and reconcile it against `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`; do not
silently reinterpret the goal.

## Completion standard

The local phase is not complete merely when HJ is reproduced.  Completion
requires: a corrected long-horizon evidence atlas spanning the principal probe
families; at least one new evidence-derived algorithm with a derivation card;
a matched true-200-epoch local trajectory for the promoted candidate; and an
honest candidate/fallback decision.  Positive benefit is a scientific target,
not a guaranteed outcome.
