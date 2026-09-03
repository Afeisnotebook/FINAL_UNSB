# Future dynamic-control integration rehearsal

## Decision

Keep the current training schedule unchanged and accept the paper-delivery control
path only after an integration test proves that the real tracked runtime-relation
registry can evolve from its current shape to the expected post-5090B shape.

The rehearsal covers both method paths that lost their originally planned 5090A
e200 control when the user prioritized ST-CGR:

- Proposal-only on 5090C can retain its historical 5090A relation while adding a
  second reviewed relation to `5090B_MATCHED_PLAIN`.
- ST-CGR on 5090A can add its cross-code candidate-to-parent and parent-to-plain
  proof chain to the same registry.
- Both relations reach the late-three final-delivery validator without selecting a
  checkpoint or reading any performance value.

This is an interface and fail-closed delivery test. It is not evidence that either
algorithm improves PSNR, and it does not authorize a matched delta before the real
5090B e200 runtime receipt and both reviewed relation candidates exist.

## Why now

5090A plain is intentionally paused at e9 and has no automatic resume path. The
future 5090B control therefore has high fan-out: it must support Proposal-only and
ST-CGR after their complete trajectories arrive. Earlier unit tests covered
registry review and final delivery separately, but did not exercise the tracked
single-dict Proposal relation through the future multi-relation registry and both
final validators in one test.

Finding this class of mismatch after e200 would waste the wall-clock time saved by
the ST-CGR-only scheduling decision. The new integration test closes that risk
without touching any trainer, successor, checkpoint, data, or scientific protocol.

## Result

- Targeted integration test: `1 passed`.
- Full repository test suite: `588 passed`.
- No active training or successor was interrupted or modified.
- No paired metric was read; confirmation20 remains sealed.
- No runtime relation was added to the production registry. Real evidence remains
  mandatory and the production path continues to fail closed until it arrives.

Compact evidence:
`evidence/paper_aio/PAPER_AIO_FUTURE_DYNAMIC_CONTROL_INTEGRATION_20260903T144212.json`.
