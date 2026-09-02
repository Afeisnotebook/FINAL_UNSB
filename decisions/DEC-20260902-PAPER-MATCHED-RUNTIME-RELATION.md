# Paper matched-runtime relation gate

Date: 2026-09-02  
Scope: unified evaluation and method-minus-plain adjudication

Moving Proposal from 4090A to 5090C shortened the training critical path, but
one-container evaluation alone cannot prove that a 5090C checkpoint and a
5090A plain checkpoint are a matched training experiment.  The evaluator must
therefore preserve two independent facts: every metric was recomputed with the
same CRN/evaluator, and the two training runtimes were proven equivalent before
long training began.

`configs/PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json` records the metric-blind
5090C/5090A relation: identical 2000-update e0 and step cores, training protocol,
manifest, normalized environment and receipt hashes.  It contains no PSNR,
SSIM, LPIPS, FID, KID, ranking or delta.  This registry is part of the unified
evaluator fingerprint, not a change to any running training checkout.

The unified evaluator now records source host, training protocol and manifest
on every metric and receipt.  If Proposal and plain have different source-host
labels, `unified-lock` fails unless that exact relation passes.  Adjudication
also includes the relation in every late trajectory point; CRN equality without
runtime equality is no longer sufficient for a scientific pass.  Same-host
AM-TNC still requires identical training protocol and manifest.  Dynamic
ST-CGR remains governed by its same-host cross-code candidate runtime gate.

This closes an evaluation-layer loophole without reading performance, changing
training, merging a non-equivalent host delta, opening confirmation20 or
selecting a checkpoint.
