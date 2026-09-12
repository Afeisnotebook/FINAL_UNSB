# Methods and Experimental Protocol (Pre-result Draft)

Status: `PRE-RESULT / NO EMPIRICAL CLAIM`

This document is an English manuscript draft, not a new scientific authority.
The executable protocol is `configs/PAPER_AIO_UNPAIRED_V1.json`; the canonical
mathematical boundary is `configs/PAPER_ALGORITHM_THEORY_BUNDLE.json` together
with `research/paper_aio/ALGORITHM_THEORY_MAP_CN.md`.  If this prose and those
artifacts differ, the machine-readable protocol and frozen theory bundle win.
No full-data performance value was read or used while writing this draft.

## 1. Problem setting

Let \(\pi_A\) be a mixture of six degraded-image domains and \(\pi_B\) the
corresponding clean-image marginal.  Training observes two unpaired image
collections and never queries the clean counterpart of a sampled degraded
image.  We learn one All-in-One mapping shared by all six domains; domain
labels are not provided to the model or to an online controller.

UNSB transports \(X_0\sim\pi_A\) toward \(\pi_B\) through a finite Markov
bridge.  For bridge coordinates
\(0=t_0<t_1<\cdots<t_T=1\), one transition has the form

\[
X_{t_{j+1}}=(1-\mu_{j+1})X_{t_j}
             +\mu_{j+1}\widehat X_B(X_{t_j};\theta_G)
             +\sigma_{j+1}\varepsilon_j,
\]

where

\[
\mu_{j+1}=\frac{t_{j+1}-t_j}{1-t_j},\qquad
\sigma_{j+1}^2=\mu_{j+1}(1-\mu_{j+1})\tau(1-t_j),
\]

and \(\varepsilon_j\) is native bridge noise.  We retain the canonical UNSB
bridge construction, endpoint law, GAN term, Schrödinger-bridge term, and
PatchNCE term.  Our question is narrower: can the stochastic estimator used
inside the sequential UNSB game be changed without using paired supervision
so that a beneficial training effect, if present, survives 200 data epochs?

## 2. The sequential stochastic-game boundary

One native optimizer update commits the players in the order

\[
D\;\longrightarrow\;E\;\longrightarrow\;(G,F).
\]

Let \(b_k\) denote the official batch-1 unpaired draw at update \(k\), and let
\(S_k^{DE}\) be the realized network and optimizer state after the discriminator
\(D\) and encoder \(E\) have committed their updates.  A stochastic G/F view is

\[
g_{I,\omega}=g_{GF}(S_k^{DE},b_k;I,\omega),
\]

where \(I\in\{0,\ldots,T-1\}\) is the native uniform bridge-time index and
\(\omega\) collects endpoint latent noise, bridge noise, PatchNCE latent noise,
and patch sampling.  Define

\[
\mu_i=\mathbb E_\omega[g_{i,\omega}],\qquad
\bar\mu=\frac1T\sum_{i=0}^{T-1}\mu_i,
\]

\[
\bar\Sigma=\frac1T\sum_i\operatorname{Cov}_\omega(g_{i,\omega}),\qquad
V_\mu=\frac1T\sum_i(\mu_i-\bar\mu)(\mu_i-\bar\mu)^\top.
\]

All unbiasedness statements below are conditional pre-Adam statements: given
the realized player state and official unpaired batch, the submitted gradient
estimator has the parent estimator's conditional mean.  They do not assert
equality of the expected Adam displacement, equality of the finite-step
training kernel, or convergence of the full adversarial game.

The probability statements require finite second moments.  Views called iid
are conditionally independent and identically distributed given state and
batch; replicas called exchangeable have a swap-invariant conditional joint
law.  These conditions describe the frozen implementation, not any arbitrary
program that happens to execute two forward passes.

## 3. Player-selective conditional averaging (Proposal-only)

Proposal-only leaves the native single-view D and E updates unchanged.  Only
after those players create the actual \(S_k^{DE}\) does it draw two complete
conditionally iid G/F views:

\[
\widehat g_P=\frac12
\left(g_{I_1,\omega_1}+g_{I_2,\omega_2}\right),
\qquad
(I_r,\omega_r)\stackrel{\mathrm{iid}}{\sim}\mathcal P_{GF}.
\]

The two gradients are averaged before a single native G/F Adam commit.  Thus

\[
\mathbb E[\widehat g_P\mid S_k^{DE},b_k]=\bar\mu,
\qquad
\operatorname{Cov}(\widehat g_P\mid S_k^{DE},b_k)
=\frac12(\bar\Sigma+V_\mu).
\]

The construction reduces conditional view variance for G/F but not variance
from the sampled image identity or domain.  It is not equivalent to increasing
the dataset batch size.  It is also not equivalent to averaging every player:
changing the stochastic D/E update changes the endogenous state seen by G/F.

### Algorithmic order

1. Draw one official unpaired batch \(b_k\).
2. Execute and commit the native one-view D update.
3. Execute and commit the native one-view E update.
4. At the resulting \(S_k^{DE}\), draw two fresh complete G/F views.
5. Average their pre-Adam gradients and commit G and F once.

Disabling the proposal dispatches the native UNSB update.  The method uses no
teacher, output anchor, domain label, paired target, validation score, or exit
window.

## 4. Stratified-time conditional G/F resampling (ST-CGR)

ST-CGR keeps the Proposal-only player boundary and number of views, but changes
the joint law of their time indices.  It samples an ordered pair uniformly
without replacement:

\[
\Pr(I_1=i,I_2=j)=\frac{1}{T(T-1)},\qquad i\ne j.
\]

Each coordinate still has the native uniform marginal \(1/T\).  Conditional on
the selected time indices, all non-time randomness is drawn independently in
the two views.  The committed estimator is

\[
\widehat g_S=\frac12
\left(g_{I_1,\omega_1}+g_{I_2,\omega_2}\right).
\]

Consequently,

\[
\mathbb E[\widehat g_S\mid S_k^{DE},b_k]=\bar\mu,
\]

\[
\operatorname{Cov}(\widehat g_S)
=\frac12\bar\Sigma+\frac{T-2}{2(T-1)}V_\mu.
\]

Relative to the iid Proposal pair,

\[
\operatorname{Cov}(\widehat g_P)-
\operatorname{Cov}(\widehat g_S)
=\frac{1}{2(T-1)}V_\mu\succeq0.
\]

For the frozen \(T=5\) protocol, the removed term is \(V_\mu/8\).  If all
time-conditional means coincide, ST-CGR and Proposal have the same covariance.
The method does not change the time marginal, learn a sampler, apply importance
weights, or alter the bridge coordinate or endpoint law.

## 5. Adam-metric tangential consensus (AM-TNC)

AM-TNC is an independent stochastic-geometry route rather than another member
of the Proposal/ST-CGR time-coupling lineage.  At each native player boundary,
it forms two conditionally exchangeable stochastic gradients \(g_1,g_2\) and
writes

\[
m=\frac{g_1+g_2}{2},\qquad d=\frac{g_1-g_2}{2}.
\]

Before drawing the current gradients, it freezes the positive diagonal metric
from the previous Adam second moment,

\[
A=\operatorname{diag}\left((\sqrt{v_{\mathrm{prev}}}+\epsilon)^{-1}\right).
\]

When \(\lVert Am\rVert>0\), define

\[
c=\frac{\langle Am,Ad\rangle}{\lVert Am\rVert^2},\qquad
\widehat g_A=m+d-cm.
\]

The residual disagreement is tangential to the consensus direction in the
frozen Adam metric:

\[
A(\widehat g_A-m)=Ad-\operatorname{Proj}_{Am}(Ad).
\]

Swapping the replicas leaves \(m\) unchanged and negates \(d\), \(c\), and the
correction.  Therefore

\[
\widehat g_A(g_1,g_2)+\widehat g_A(g_2,g_1)=2m.
\]

Under exchangeability, the pre-Adam conditional mean equals the two-view
consensus mean.  If \(\lVert Am\rVert=0\), positivity of \(A\) implies \(m=0\);
the implementation returns the ordered first gradient, whose swap-pair mean is
again zero.  If the two gradients are identical, the explicit identity branch
returns that gradient.  A single-replica configuration dispatches plain UNSB.

AM-TNC is applied to D, E, and joint G/F at their actual sequential states.  It
does not solve a multi-task conflict problem and does not prove that the
nonlinear Adam parameter displacement is unbiased.  Its additional stochastic
views also make its compute cost distinct from Proposal and ST-CGR.

## 6. Relationship among the three operators

| Operator | Players modified | Replica coupling | Direct object | Strict pre-Adam statement |
|---|---|---|---|---|
| Proposal-only | G/F only | two complete iid views | post-D/E conditional G/F view variance | parent conditional mean; half conditional covariance |
| ST-CGR | G/F only | time without replacement; other randomness independent | between-time conditional-mean variance inside Proposal | parent conditional mean; removes \(V_\mu/[2(T-1)]\) relative to Proposal |
| AM-TNC | D, E, and G/F | exchangeable same-player replicas | radial/tangential disagreement in a frozen Adam metric | swap-pair mean returns to consensus |

Proposal and ST-CGR form a two-level conditional-estimator family.  AM-TNC is
an independent all-player geometry.  HJCGR remains a documented small-view
parent mechanism but is not a first-wave full-data method; its deferred status
is not a mechanism falsification.  Full-data results determine which, if any,
of these operators may enter the final empirical claim.

## 7. Data and unpaired training protocol

We use six restoration domains in one shared All-in-One model.  The frozen
manifest contains 8,553 training identities per side, 480 discovery identities,
and 120 sealed confirmation identities.  One data epoch is exactly 8,553
batch-1 optimizer updates; 200 data epochs therefore contain 1,710,600 updates.

At each data epoch, the A identities are traversed by a seeded permutation.
The B identity is sampled independently from the complete B marginal, without
querying the paired target.  This is the frozen
`official_image_proportional_unpaired` measure.  We do not balance domains,
provide a domain label, enlarge the image batch, or use discovery metrics to
change the sampler or method.

All UNSB-family lanes use seed 2026, 128-by-128 resize-and-crop inputs, no
horizontal flip, and deterministic execution with TF32 disabled.  The UNSB
optimizer is Adam with learning rate \(10^{-4}\), \((\beta_1,\beta_2)=(0.5,
0.999)\), 200 constant epochs, and no decay phase.  GAN, SB, and NCE weights
are all one; \(\tau=0.01\); \(T=5\); PatchNCE uses layers
0, 4, 8, 12, and 16, temperature 0.07, and 256 sampled patches.

Proposal, ST-CGR, and their admitted matched plain share a reviewed runtime
cohort and fresh initialization relation. AM-TNC is compared only with its
same-host 4090A plain trajectory. After the epoch-178 float32 metric-reduction
overflow, that comparison is no longer described as byte-identical runtime
identity: it requires the pre-result, metric-blind method-only recovery relation
that binds the unchanged transition state, finite-path bitwise identity,
float64-only overflow fallback, and the localization/replay/migration receipts.
The recovery export rematerializes e100/e125/e150/e175 with new provenance
metadata only; the migration receipt binds dynamics-only identity for every
such fixed checkpoint, so this artifact normalization is not represented as
retraining under the recovery code.
A result without an admitted exact or explicitly disclosed recovery relation is
reported as an absolute trajectory, not as a matched delta.

CycleGAN, CUT, and DCLGAN retain their method-specific official objectives.
The controlled external schedule uses Adam with learning rate \(2\times10^{-4}\)
and \((\beta_1,\beta_2)=(0.5,0.999)\), with 100 constant epochs followed by
100 linear-decay epochs.  Their data exposure, image size, seed, and e200
terminal reporting point match the controlled main protocol.  These are
source-bound controlled reproductions, not claims of byte-for-byte identity to
every environment in the original papers.

## 8. Checkpoints, recovery, and non-interference

Permanent full-state checkpoints are written at data epochs
1, 5, 10, 20, 40, 60, 80, 100, 125, 150, 175, and 200; `latest` is refreshed at
every complete epoch.  A recoverable state contains G/F/D/E or the corresponding
external networks, optimizers, schedulers, image pools where applicable,
method state, Python/NumPy/CPU/CUDA RNG, independent A/B sampler state, global
update, and physical data epoch.  Recovery is permitted only through the
source-bound exact-resume path on the same host/runtime cohort, except for the
separately disclosed AM-TNC method-only numerical recovery described above.

Every lane uses an isolated writable run directory.  Exporters publish hashes
and receipts after a complete checkpoint is stable.  Read-only virtual audits
must restore the parent model, optimizer, sampler, RNG, mode, and
`requires_grad` state exactly.  A missing process, stale heartbeat, NaN/OOM,
save failure, or violated disk bound is an engineering event; an intermediate
quality score is not a scheduling signal.

## 9. Frozen evaluation protocol

Trajectory evaluation uses a lane-blind common-random-number bundle.  Fixed
milestones use 70 discovery images per domain for PSNR and SSIM.  LPIPS is
added at e100, e125, e150, e175, and e200.  At e100, e150, and e200, UNSB-family
methods report all fixed NFE values from one through five; the primary NFE is
five and is not selected from the results.

The main table uses e200, never the best checkpoint.  Sustained behavior is
summarized by the fixed e150/e175/e200 triplet.  The terminal evaluation uses
all 80 discovery images per domain and five predetermined rollout bundles to
report the stochastic mean and dispersion.  Six-domain macro metrics,
per-domain coverage, worst-domain behavior, and absolute trajectories are all
retained.  KID is the primary distribution metric; FID is supplementary and
is reported with its small-sample limitation.

The 20 confirmation identities per domain remain sealed until algorithms,
baseline configurations, runtime relations, result claims, and manuscript
tables are frozen.  Confirmation can be opened once and cannot be used to
change an algorithm, NFE, checkpoint, or claim.

## 10. Reporting and interpretation boundaries

The final report includes fixed-e200 PSNR, SSIM, and LPIPS; sustained
e150/e175/e200 behavior; per-domain results; stochastic rollout dispersion;
controlled optimizer-step time; peak memory; and parameter count.  Cross-host
epoch time is operational telemetry, not an algorithm-cost ratio.  Unequal
numbers of stochastic views are disclosed.

The first full-data wave uses one seed.  It cannot establish multi-seed
stability.  A covariance theorem does not imply a quality improvement, and a
positive terminal result does not prove convergence or repair of terminal
singularity.  DDSB remains a required related-work comparator but receives no
reproduced number until an authoritative implementation passes the formula,
optimizer-order, resume, and evaluation gates.  Deferred HJCGR and NEGCUT are
not reported as mechanism failures.

Empirical language such as *improves*, *outperforms*, *stable*, or *best* must
be inserted only from the committed post-result claim freeze.  Until then, the
method section describes operators and proved estimator properties, not their
unknown full-data utility.
