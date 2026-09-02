# DDSB source gate: reproduction incomplete, not a negative result

Date: 2026-09-02  
Scope: full-data paper baseline only

## Decision

DDSB is a required contemporary comparison, but its lane remains fail-closed.
The NeurIPS 2025 paper and supplement define DOT and DTC at a conceptual and
equation level, yet an author-maintained public source repository was not found
during the 2026-09-02 audit. The paper's own reproducibility checklist says the
source was *promised* after publication; it does not provide a repository URL.

The paper fixes several high-level values (five steps, tau 0.01, DOT lambda
0.01, batch one, 400 epochs, 2e-4 initial learning rate, 512 resolution) and
describes a U-Net/AdaIN generator and a three-layer degradation model. It does
not uniquely determine the complete network graph, MINE/discriminator details,
multi-network update ordering, stop-gradient boundaries, DTC reduction, all
loss coefficients, or exact resume state. Those choices can materially change
the method. Implementing them by guesswork would create a local derivative,
not a defensible DDSB reproduction.

Therefore:

- no DDSB long run starts until a source/formula/full-state lock passes;
- the result is labelled `reproduction_incomplete`, never a DDSB failure;
- the reserved 5090 lane falls back to official-loss CycleGAN after its own
  data, disk and exact-resume gates pass;
- the paper remains cited and DDSB can be added later if authoritative source
  becomes available, without changing our algorithm or opening confirmation20.

Machine-readable evidence:
`evidence/paper_aio/DDSB_SOURCE_GATE_20260902.json`.
