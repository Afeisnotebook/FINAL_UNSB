# Durable full-data discovery delivery

Date: 2026-09-02
Scope: fixed terminal results and complexity

Training, checkpoint export, import and matched evaluation were already
durable, but a complete paper discovery package still required a person to run
the e200 complexity profiles and merge two distinct legal comparison cohorts.
That final handoff is now represented by one detached successor on 4090A.

The successor waits for first-wave, AM-TNC and ST-CGR terminal evaluation
states, the complete verified import set and the AM-TNC source-bound export.
These are metric-blind completion dependencies. Once all are ready it profiles
the fixed e200 checkpoint for plain, Proposal-only, CUT, CycleGAN, AM-TNC and
ST-CGR in one evaluator runtime. It records parameters, NFE-specific inference
latency, training-step latency and peak memory. It deliberately does not report
FLOPs because the stochastic bridge and lazy PatchNCE path are not covered by a
single audited counter.

The final portfolio preserves each legal comparison relation: Proposal and
ST-CGR use 5090A plain; AM-TNC uses 4090A plain. It can retain several passing
algorithms and labels a failing lane only as a failure of the current
implementation and protocol. HJCGR remains deferred rather than falsified and
DDSB remains reproduction-incomplete rather than a negative result.

The resulting discovery portfolio is review-ready, not a silent paper-claim
freeze. It cannot authorize confirmation20 and cannot use result values to
modify training, scheduling or checkpoint selection. Exact process, contract,
complexity and output details are in
`evidence/paper_aio/PAPER_AIO_FINAL_DISCOVERY_DELIVERY_SUCCESSOR_20260902.json`.
