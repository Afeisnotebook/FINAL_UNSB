# DEC-0002 — Accept local engineering preflight

Date: 2026-08-28

## Decision

Accept the repository for four-server preflight. Do not authorize long training
until the four RTX 4090 environments independently reproduce the canonical
manifest, protocol fingerprint and a common e0 hash.

## Material correction found locally

The imported HJ inactive path drew its PatchNCE latent directly on CUDA, while
plain drew on CPU and then transferred it. The two distributions were the same,
but their RNG streams were not: HJ therefore diverged at e0 before its declared
window. HJ now uses the canonical CPU draw. The corrected implementation has:

- the same e0 hash as plain/HNEK/macro;
- exact plain network, optimizer, scheduler and RNG state before HJ activation;
- a read-only cumulative counter proving that active HJ optimizer steps occur.

This is an implementation correction, not a favorable-effect selection.

## Boundary

The local smoke used 12 training identities and has no scientific performance
meaning. No local smoke PSNR is retained as evidence. Confirmation remains sealed.
