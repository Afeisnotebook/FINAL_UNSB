# Decision: correct and test the limitations contract commit identity

Date: 2026-09-08

Status: accepted engineering correction; no scientific or runtime change.

The first status commit for the pre-result limitations freeze expanded the visible short hash `4298cb2` to a non-existent 40-character value instead of resolving the Git object. That value would have caused a later committed-file identity gate to fail even though the bound document and contract bytes were correct.

The authoritative commit is `4298cb2383f5c694b4a63204150002594f1cd71b`. Direct `git show` verification proves that its manuscript branching contract blob is byte-identical to the current file and has SHA256 `6fe0fd12258b7e789bfbe37d495af3f4bbaa516ac87ce739f7e1f5fc8c4c1fec`.

All live status references and the original compact evidence were corrected. A new recursive regression test now rejects any recorded `branch_contract_commit` in `PROJECT_STATE.json`, `FULL_DATA_METHOD_PORTFOLIO.json`, or `PAPER_DELIVERY_COMPLETION_MATRIX.json` unless the Git object resolves and reproduces the recorded contract hash. The full suite passes 763 tests.

No process, checkpoint, queue, metric, or confirmation data was touched.

Evidence: `evidence/paper_aio/PAPER_AIO_LIMITATIONS_COMMIT_IDENTITY_CORRECTION_20260908T112200.json`.
