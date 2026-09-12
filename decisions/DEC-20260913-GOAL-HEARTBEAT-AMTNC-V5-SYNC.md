# Goal heartbeat synchronized to AM-TNC V5

The persistent two-hour FINAL_UNSB Goal monitor is now synchronized to the
audited AM-TNC V5 evaluation and final-delivery chain. It explicitly treats
V2, V3, and V4 waiters as normally retired and forbids restoring them merely
because their historical PIDs are absent. Active process identities remain
dynamic and must be read from the latest state files rather than copied from
the automation prompt.

At synchronization, all five incomplete fixed-protocol lanes were verified by
live process handles and current full-state files: AM-TNC e182, ST-CGR e164,
Proposal e177, 5090B matched plain e105, and local DCLGAN e158. Their GPUs and
storage were healthy. The new V5 AM evaluator/final waiter and its health watch
were also live with zero alerts. No restart, queue change, paired metric read,
or scientific intervention was warranted.

The monitor remains active every two hours with failed-run-only notification.
It preserves the paper north star and confirmation20 seal while allowing the
healthy long jobs to run without unnecessary agent or GPU disturbance.
