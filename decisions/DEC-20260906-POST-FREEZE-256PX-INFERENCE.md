# Decision: implement one fixed post-freeze 256px inference sensitivity

Date: 2026-09-06

Status: **CODE READY; NOT AUTHORIZED OR SCHEDULED BEFORE PAPER FREEZE**

The completion matrix still lacked the preregistered higher-resolution
supplement. This was an engineering dependency that could be closed without
reading intermediate performance or touching a healthy training process.

We implement exactly one spatial policy: resize source and paired metric target
to 256×256 with bicubic interpolation, then run the fixed e200 model as a whole
fully convolutional image. The controlled 128px table remains primary. The
supplement is not described as native-resolution restoration, and it exposes
no resolution, tiling, NFE, checkpoint, or retraining choice.

The evaluator is intentionally an `operations` module rather than a change to
`research/paper_aio/*.py`. Therefore it does not change the scientific training
fingerprint of lanes already in flight. Its own script hash and Git commit are
recorded in every receipt and must be identical across the locked cohort.

Execution remains forbidden until the algorithm, baseline, e200 result, and
claim set have passed the existing two-stage committed freeze. Only discovery80
is then read. UNSB-family lanes use NFE=5 and five fixed lane-blind 256px CRN
bundles; deterministic lanes use one pass. Input, all frozen methods and
baselines, portable candidates, and source-bound DCLGAN must be evaluated in
one evaluator runtime before the supplementary cohort can lock.

This decision does not alter any training queue, authorize confirmation20, or
permit its metrics to select or schedule an algorithm.

Evidence:
`evidence/paper_aio/PAPER_AIO_POST_FREEZE_HIGH_RESOLUTION_INTERFACE_20260906T020449.json`
