# DEC-20260908: Freeze a primary-source reference core before results

## Decision

Freeze a machine-checked, pre-result bibliography containing the stable references already required by the paper protocol, algorithm theory, external-baseline portfolio, and six-domain data lineage. Keep volatile 2026 preprints and OpenReview neighbors outside that core until a mandatory submission-day refresh.

## Why

The repository already constrained empirical claims and related-work novelty, but it had no hash-bound bibliography or citation inventory. Leaving citation assembly until after results would make it easy to omit a required baseline, blur a dataset's original source with the later aggregation source, or accidentally treat a changing preprint record as stable metadata.

The new ledger separates three obligations:

1. foundations and controlled or gated baselines;
2. mathematical and optimization neighbors relevant to the three frozen operators;
3. original dataset lineage plus the later All-in-One aggregation source.

This is a metadata closure, not a novelty proof. The existing related-work boundary remains authoritative for overlap and claim scope. Eleven volatile neighbors are explicitly excluded from the core BibTeX file and must be re-read from their primary pages before the final claim freeze or submission.

## Consequences

- The stable core now contains 20 unique citation keys with primary-source URLs.
- A regression test binds the exact BibTeX bytes, requires all ledger entries to be represented exactly once, checks complete group coverage, and preserves the no-result/no-training scientific boundaries.
- No performance value was read, no algorithm was selected, no process or queue was changed, and `confirmation20` remains sealed.
- Paper drafting can cite the stable core now, but final submission is fail-closed until the volatile-neighbor refresh is completed.

## Validation

- `python tools/validate_contracts.py`: PASS
- `python -m pytest tests/test_paper_reference_ledger.py -q`: 3 passed
- full `python -m pytest -q`: 766 passed
- `bibtex` smoke check: PASS without warnings
- `python -m compileall -q research operations tests tools`: PASS
- `git diff --check`: PASS

Compact evidence: `evidence/paper_aio/PAPER_AIO_PRE_RESULT_REFERENCE_LEDGER_20260908T124000.json`.
