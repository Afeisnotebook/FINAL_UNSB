# ST-CGR fixed-state decision

The preregistered target-blind audit authorizes
`G4-01-STRATIFIED-TIME-CONDITIONAL-GF` to proceed through executable
engineering gates and, if those gates pass, one small25/seed2026/e200 run.

This is not a PSNR result and does not authorize full-data training. The
decision is based on a bias-corrected gradient covariance audit at the exact
post-D/E boundary used by Proposal-only. Both Proposal and HJCGR parents passed
at e60/e100/e150/e200. Their pooled pre-Adam covariance-trace ratios were
0.7894 and 0.7878 respectively, with all eight checkpoint-specific ratios
strictly below the preregistered 0.95 material threshold.

The candidate preserves the native uniform bridge-time marginal in each G/F
replica, changes no objective or endpoint law, reads no paired target, and has
the same network-evaluation count as Proposal-only. Its only change is the
ordered coupling of the two time indices: the second index is uniform over the
four indices not used by the first view.

The next admissible action is therefore:

1. freeze the evidence-bound candidate identity and source hashes;
2. prove zero-intervention identity, exact resume, source-state isolation and
   recoverable ordered-pair counters;
3. run the existing 400-update engineering micro gate without paired metrics;
4. only after all gates pass, restart from the common small25 e0 and train to
   e200 without early stopping or paired control.

Full-data promotion remains blocked until the fixed e150/e175/e200 and e200
small25 scientific criteria are evaluated. Confirmation20 remains sealed.
