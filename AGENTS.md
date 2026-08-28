# FINAL_UNSB agent contract

## Mandatory read order

Read, in order, before changing code or launching compute:

1. `START_HERE_CN.md`
2. `PROJECT_CONTRACT.json`
3. `PROJECT_STATE.json`
4. `CONTEXT_CAPSULE_CN.md`
5. `configs/FOUR_LANES.json`
6. `DATA_CONTRACT.json`
7. the latest file under `decisions/`

## Fixed objective

Run a leak-free, seed-2026, 200-data-epoch comparison of exactly four frozen
lanes on the six-domain corpus and select one current best non-plain candidate.
This is a single-seed development decision, not paper confirmation.

## Role separation

- **Master Codex** may inspect, merge reports, run contract tests, request the
  missing server paths, and issue the explicit run authorization.
- **Server Codex/executor** may install the frozen environment, materialize the
  data view, run preflight, run its assigned lane, resume exact checkpoints,
  evaluate registered milestones, and return compact reports.
- Server executors may not redesign methods, change thresholds, shorten data,
  change seed, open confirmation, or interpret scientific novelty.

## Drift firewall

- The only executable training/evaluation entrypoints are under `production`;
  do not recreate or invoke an upstream `src/train.py` or `src/test.py`.
- No fifth lane and no hyperparameter grid during the four-lane run.
- No TA_MINIMAL, CLOCK_ACTIVE, TACT, KCK, DCUM, LBST, PTQ, AEB or PCOA rescue.
- HJ activation is defined in physical data epochs `[1.6, 8.0)`, never by a
  copied absolute small-view step number and never by paired metrics.
- Primary selection checkpoint is e200. Earlier checkpoints are trajectory
  diagnostics, not a best-checkpoint search.
- `confirmation` remains inaccessible until a committed candidate-freeze
  decision exists. Discovery results may not change code or lane settings.
- A semantic implementation defect restarts the affected lane from e0. It is
  not repaired in place after effect metrics have been read.
- Data/checkpoints stay outside Git. Git contains configs, manifests, hashes,
  compact metrics and decisions only.

## Required response to ambiguity

If a request conflicts with the frozen objective, stop and write a proposed
decision under `decisions/proposals/`; do not silently reinterpret the goal.
Missing machine paths are operational inputs, not permission to change science.

## Completion standard

The project is not complete until all four lane identities are comparable,
e200 is reached or a documented engineering hard stop occurs, discovery is
ranked against matched plain, one candidate is frozen, and claim boundaries
are reported honestly.
