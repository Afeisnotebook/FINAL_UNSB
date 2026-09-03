# DCLGAN portable source lock and durable 5090B queue

Date: 2026-09-03

The first source-bound DCLGAN gate at `f03b3ff` completed locally, but its raw
byte hashes were not portable across Git checkouts with different line-ending
policies.  Linux consequently rejected the README hash before any target
training.  This supersedes that source-lock representation only; it is neither
a DCLGAN mechanism failure nor a failed training result.

Commit `e45973a42e8c309db2cff76eb4f36c8b1f7b123b` changes tracked text hashing to
`portable_text_lf_v1`.  Windows and Linux now verify the same author commit,
the same normalized source hashes and adapter fingerprint
`bf4eecb236e97fbba990e59fc9024b790f9de0b64f1cac4632d5a7d20d66ba44`.
The fresh GTX1660 formal gate then passed all frozen stages: source/data
preflight, confirmation rejection, continuous 1000 versus 500+resume exact
state, an independent 1000-update capacity run, repeated discovery evaluation
without parent-state mutation, and host-bound authorization.  The old raw-hash
receipt remains retained as provenance but must not authorize another host.

5090B has a clean detached checkout and a clean author-source clone made from a
recorded Git bundle.  A metric-blind target successor is armed behind
`COMPLETE_PLAIN_E200`, with a separate source-bound exporter and health watch.
When released it must repeat the complete gate on 5090B; only a target-host
authorization with the frozen fingerprint may start the 1,710,600-update run.
The current CUT and CycleGAN trainers are untouched, and no DCLGAN GPU process
is running while the successor waits.

This ordering protects the high-fanout matched-plain control before adding the
second-priority external baseline.  DCLGAN is an external comparator and does
not replace Proposal, ST-CGR or AM-TNC.  NEGCUT remains deferred for its known
device/resume/licensing engineering gates and is not mechanism-falsified.
