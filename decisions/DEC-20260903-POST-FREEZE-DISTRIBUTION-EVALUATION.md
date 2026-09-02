# Post-freeze KID/FID evaluation boundary

Date: 2026-09-03
Scope: paper distribution metrics only

The frozen paper protocol requires KID as the primary distribution metric and
FID as a small-sample supplement, but it also requires both metrics to remain
unavailable until the algorithm, baseline and claim set is frozen.  The current
full-data runs and discovery adjudication are not such a freeze.

The repository now contains a read-only distribution evaluator, but no
distribution run or successor has been launched.  The evaluator refuses to
start unless its lane appears in an exact JSON freeze receipt that is already
committed in this repository.  The receipt must bind the full-data manifest,
the frozen evaluation bundle, the source e200 portfolio and the facts that
algorithm configuration, baseline configuration, e200 results and paper claims
are frozen.  It must also state that confirmation remains unauthorized and
closed.

After that future decision, each selected e200 lane is rendered at 128 pixels
on discovery80 in the common evaluator.  UNSB-family methods use all five fixed
CRN replicates at NFE=5; deterministic external methods and Input use one
replicate.  The primary statistic is six-domain macro KID.  Pooled KID and
pooled FID are supplementary, with FID explicitly labeled as a 480-image
small-sample estimate.  Clean-FID version, source module, feature-model state,
quantization, checkpoint, CRN set and RNG-isolated KID seed are recorded.

This interface does not change training, select a checkpoint, choose an
algorithm, or authorize confirmation20.  Deployment remains a future action
after the committed paper freeze.
