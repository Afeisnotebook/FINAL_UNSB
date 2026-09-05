# Frozen manuscript-table delivery

Date: 2026-09-06  
Scope: deterministic post-freeze paper reporting

The final result path previously produced complete JSON evidence but still left
manual copying between the portfolio and the manuscript. That is an avoidable
late-stage integrity risk. Commit
`a4419ee9c51fbe90c0949d5dbc59752bde3017a0` adds a deterministic exporter for
the primary e200 macro table, the fixed e150/e175/e200 algorithm trajectories,
all six-domain algorithm deltas, complexity, approved claims and a readable
Markdown summary.

The exporter cannot run from a draft or merely complete portfolio. It requires
the existing two-stage algorithm/baseline/claim freeze receipt to be committed,
rehashes the external portfolio, and proves that its own working source equals
its committed Git blob. Every generated table is immutable and the final
receipt publishes all output hashes. DCLGAN remains a standalone external
comparison and is never presented as a matched UNSB delta.

This is a reporting-only interface outside the scientific training fingerprint.
It has not been deployed because the e200 portfolio is not complete. It does
not rank methods, select checkpoints, authorize confirmation20 or affect any
running/scheduled experiment. Full verification is recorded in
`evidence/paper_aio/PAPER_AIO_FROZEN_MANUSCRIPT_TABLE_INTERFACE_20260906T023358.json`.
