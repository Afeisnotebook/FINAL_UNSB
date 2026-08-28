# Decision: reopen local route-1 long-horizon algorithm discovery

Date: 2026-08-29

## Decision

Suspend the four-4090/four-frozen-lane execution objective. Continue locally on
the GTX 1660. The active objective is to discover/reconstruct a new long-horizon
UNSB algorithm from multi-method evidence, not to validate HJ or any fixed list.

## Evidence that changed the state

The prior local “long” gates used incompatible clocks:

- small25 2400 updates / 150 samples = 16 data epochs;
- full100 12000 updates / 600 samples = 20 data epochs.

Historical continuous Layer-0 HJ was negative at e100 and became positive only
from e125 through e200. Therefore the short local gates cannot support a general
200-epoch mechanism-falsification claim. HNEK, DT, PCOA and most later mechanisms
were also judged within similarly short horizons; the issue is not HJ-specific.

## Consequences

- HJ becomes the first temporal positive control, not the unique direction.
- Plain/HJ/HNEK/DT form the initial long-horizon anchor atlas.
- Later mechanisms remain an evidence pool for constructing new algorithms;
  current-implementation failures remain recorded without becoming universal
  parent-mechanism death sentences.
- Route-2 handoff and all server work are inactive until explicitly reopened.
- No scientific early stop before the registered long horizon based only on an
  intermediate paired metric.
