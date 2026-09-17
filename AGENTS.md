# FINAL_UNSB agent contract

## Mandatory read order

Before answering scientific questions or changing files, read in order:

1. `FINAL_HANDOFF_CN.md`
2. `FINAL_HANDOFF.json`
3. `docs/CURRENT_PROJECT_STATE_CN.md`
4. `CLAIM_BOUNDARIES.md`
5. `configs/DOCUMENT_AUTHORITY_REGISTRY.json`
6. `archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json`
7. `archive/paper_aio/final_v10/PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json`
8. `research/paper_aio/RESEARCH_HANDOFF_CN.md`
9. `DISASTER_RECOVERY_CN.md`
10. Read historical contracts, evidence and decisions only when the task needs
    provenance or the user explicitly authorizes a new scientific phase.

Run `python tools/verify_git_only_recovery.py` and
`python tools/validate_contracts.py` before trusting a new clone.

## Current phase

The full-data discovery execution phase is complete and archived. There is no
live training queue authorized by this repository. The current gate is an
explicit paper-claim review and confirmation-policy freeze. Confirmation20 is
sealed.

Historical PIDs, heartbeats, SSH endpoints, supervisors, successors and
`running` fields are provenance. They must not be used to restart work.

## Canonical scientific outcome

- Proposal-only is the sole in-house method that passes the preregistered
  full-data long-horizon gate: late-three macro PSNR delta +1.723565 dB and
  e200 delta +0.839347 dB.
- Current ST-CGR and AM-TNC implementations fail the gate. This does not
  falsify their parent mechanism families.
- HJCGR is deferred. DDSB is reproduction incomplete. Neither is a negative
  result.
- CUT, DCLGAN and CycleGAN have higher fixed-e200 absolute PSNR than Proposal;
  an overall SOTA claim is forbidden.
- Evidence is seed 2026 only. Cross-seed stability is not established.
- Terminal low-variance/singular-drift pathology was not confirmed and no
  repair module may be claimed from it.

Exact results come only from the final archived portfolio, not from old
operational ledgers or intermediate checkpoints.

## Authority and historical documents

`configs/DOCUMENT_AUTHORITY_REGISTRY.json` defines the conflict order.

- Files under `archive/paper_aio/final_v10/` are canonical compact results.
- `PROJECT_STATE.json`, `configs/FULL_DATA_METHOD_PORTFOLIO.json` and
  `configs/PAPER_DELIVERY_COMPLETION_MATRIX.json` are append-only operational
  ledgers. Only their final overlay/top-level current entrypoint is current;
  nested live fields are historical.
- `ACTIVE_LOCAL_ROUTE1_PLAN_CN.md`, `LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md`,
  `configs/LOCAL_ROUTE1_PROBES.json` and `HYPOTHESIS_LEDGER.json` are historical
  small25 inputs and evidence. Their ACTIVE tokens do not authorize execution.
- `configs/PAPER_AIO_UNPAIRED_V1.json` and pre-result theory/method/checklist
  artifacts are frozen scientific inputs. Their pre-result or active schema
  tokens remain for hash provenance, not present-tense scheduling.
- Evidence and decisions are not rewritten merely to sound current.

## Scientific hard boundaries

- Never use paired metrics to control training, scheduling, algorithm choice,
  NFE, exit time or checkpoint selection.
- Never select the best checkpoint; e200 is primary and e150/e175/e200 is the
  sustained window.
- Never merge deltas across non-equivalent runtime cohorts.
- Never open confirmation20 without a new committed claim/confirmation freeze.
- Never treat an implementation failure, deferral or incomplete reproduction
  as mechanism falsification.
- Never auto-import independent proof work into the canonical result.
- Do not claim multi-seed stability, overall SOTA or a confirmed terminal
  singularity repair.

## Changes and new compute

Documentation, manuscript work and read-only analysis may proceed from the
archived record. New training, confirmation access or algorithm search requires
an explicit user request and a new committed decision/contract. Old run
authorizations and successors cannot be revived.

Data, checkpoints and full logs stay outside Git. Git contains code, manifests,
hashes, compact evidence and decisions. `DISASTER_RECOVERY_CN.md` defines what
Git-only recovery does and does not guarantee.

## Completion standard

Discovery is complete. The broader paper lifecycle is not complete until
claims, disclosures, algorithm set and confirmation policy are reviewed and
frozen; any authorized confirmation is performed once; and manuscript tables,
figures and text are bound to the final archive.
