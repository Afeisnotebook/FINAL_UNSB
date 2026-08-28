# DEC-0001 — Freeze the final four-lane portfolio

Date: 2026-08-28

## Decision

Freeze `P0_PLAIN`, `P1_HJ_HANDOFF`, `P2_HNEK` and `P3_MACRO_MARGINAL` as the
only four e200 lanes. Run local preflight before renting 4090s. Do not include
TA/KCK/DCUM/PCOA or a newly invented fifth method.

## Reason

Plain is required. HJ is the strongest local native-inheritance signal. HNEK
is the only historical e200-positive bridge-native anchor. Macro-marginal is
the only major full-scale failure class not represented by balanced small/full
views. Four lanes maximize coverage of independent hypotheses under the fixed
one-week budget.

## Boundary

This decision optimizes the chance of finding a development candidate. A
cleaner 2x2 HNEK-by-measure factorial was considered and rejected because it
would remove HJ, currently the strongest local candidate. No run is authorized
until local and four-server preflight identities pass.
