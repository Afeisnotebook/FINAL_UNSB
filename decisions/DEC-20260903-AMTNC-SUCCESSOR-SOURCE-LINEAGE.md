# AM-TNC successor source-lineage hardening

## Finding

The waiting 4090A `plain -> AM-TNC` successor correctly pinned the scientific
training checkout, but its v1 state did not bind the separate control checkout
or the AM-TNC identity-gate file that it will execute after plain e200.  Both
checkouts were currently clean, so this was a latent provenance risk rather than
an observed training error.

## Decision and deployment

Upgrade only the waiting controller to successor contract v2.  The v2 contract
binds and continuously revalidates:

- the control checkout commit and clean status;
- SHA256 for the successor and AM-TNC identity-gate sources;
- the scientific training checkout commit and clean status;
- output, manifest, data, view, predecessor, successor, GPU and protocol
  fingerprint.

Drift before release or between gates now fails closed.  The old waiting process
and its health watcher were retired only after commit `5b4739a` was available on
4090A.  The new PID 2518921 reached `WAITING_FOR_PREDECESSOR_E200`, wrote the
immutable v2 contract, detached to PPID 1 and was then covered by the replacement
zero-alert health watcher PID 2519397.

The running plain supervisor/trainer, AM-TNC exporter and every remote relay were
left untouched.  No model, optimizer, checkpoint, sampler, RNG, algorithm
configuration or training schedule changed.

## Verification

- Repository tests: `590 passed`.
- Remote control commit: `5b4739aa082cbe67d05a5c1ff06b27c039ea1ff8`.
- Scientific commit remains
  `31f2fb8badaf8293a2ed2744963035575df7d7a6`.
- Contract SHA256:
  `698546636284f3c0ecbed5f67c76fb80ccbe7692973ca995569ca467f409193b`.
- Replacement health: `HEALTHY`, zero alerts.
- Plain training remained `CHILD_RUNNING` throughout the handoff.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_AMTNC_SUCCESSOR_V2_DEPLOYED_20260903T150451.json`.
