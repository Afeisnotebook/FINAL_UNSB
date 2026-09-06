# Decision: accept the CycleGAN e150 fixed milestone

Date: 2026-09-06

Status: **E150 CHECKPOINT AND METRIC ARTIFACT HASH-CLOSED; TRAINING CONTINUES**

CycleGAN on 5090B has crossed the preregistered e150 fixed milestone and has
continued to e152.  The e150 checkpoint was recomputed from the source host and
matches its sidecar (`9aaa474e...8a27`); the scientific-state hash is
`8ca369aa...edaa`.  The e150 metric artifact is retained and hash-bound, but its
contents were not parsed and did not influence training or scheduling.

The earlier e125 inline-evaluation incident remains closed: the same recovered
trainer PID 493306 continues under supervisor PID 11845, and the outer guard has
performed zero restarts.  The exact-runtime 5090B matched plain is independently
healthy at e11 after its preregistered metric-blind capacity gate.  Both lanes
remain co-resident because that already-frozen gate projected a positive
makespan saving; this milestone review makes no new scheduling decision.

No algorithm, optimizer, data order, checkpoint, or process was changed.
No paired performance value was read, and confirmation20 remains sealed.

Evidence:
`evidence/paper_aio/PAPER_AIO_CYCLEGAN_E150_HASH_CLOSURE_AND_5090B_HEALTH_20260906T204629.json`
