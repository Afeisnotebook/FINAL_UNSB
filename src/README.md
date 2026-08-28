# Embedded UNSB model library

This directory is an audited model/data library, not an executable project.
The old upstream `train.py`, `test.py` and shell entrypoints were deliberately
removed because they do not save the full training state or enforce this
repository's data, authorization and confirmation contracts.

Use only these root-level entrypoints:

- `python -m production.train_lane`
- `python -m production.evaluate_lane`
- `python -m production.rank_lanes`

The upstream project is <https://github.com/cyclomon/UNSB>. Its MIT license is
retained in `src/LICENSE`. The deterministic baseline provenance is recorded in
the root `PROJECT_STATE.json` and `evidence/PROVENANCE.md`.
