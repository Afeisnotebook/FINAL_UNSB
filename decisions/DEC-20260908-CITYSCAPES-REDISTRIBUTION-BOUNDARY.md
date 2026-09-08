# Decision: bind the official Cityscapes redistribution boundary

Date: 2026-09-08

Status: accepted pre-result release constraint; no training or scientific-result change.

The current official Cityscapes Terms and Conditions were checked directly. They authorize limited use under the registered scientific-use relationship, but prohibit making the dataset available to third parties and extend that prohibition to modified or derived material from which the data can be reconstructed. The accompanying license agreement also prohibits distributing the dataset or modified versions, permits abstract nonrecoverable derivatives such as trained models subject to its conditions, requires attribution, and excludes commercial use.

Consequently, FINAL_UNSB now treats FoggyCityscapes, RainCityscapes, and RSCityscapes as confirmed no-redistribution chains unless separate written permission is obtained. Image bytes and recoverable transformed copies remain outside the public repository. This replaces the weaker wording that Cityscapes terms merely had to be checked.

The result is deliberately narrow. It does not establish the governing license for the MPMF-Net aggregate bundle, LowLightTrafficData, RainDS-syn, or SnowTrafficData, and it does not prove byte identity between local files and upstream archives. Those remain `license_not_verified_do_not_redistribute`. The terms must be rechecked before public artifact release because the authoritative page may change.

This is a reproducibility and release-policy record, not legal advice. It does not change the frozen dataset, manifest, split, training, evaluation, or confirmation state.

Evidence: `evidence/paper_aio/PAPER_AIO_CITYSCAPES_TERMS_AUDIT_20260908T115000.json`.
