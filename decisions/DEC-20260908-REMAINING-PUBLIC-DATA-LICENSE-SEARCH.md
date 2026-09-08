# Decision: close the public-page search for remaining dataset redistribution terms

Date: 2026-09-08

Status: accepted scoped negative search; author or rights-holder confirmation remains open.

The MPMF-Net, RainDS_CCN, and VideoDesnowing primary project pages were rechecked after the Cityscapes terms audit. They expose dataset download and citation information, but the inspected repository trees and README files do not state an explicit license authorizing redistribution of the relevant image bytes.

This does not prove that no governing license exists. It proves only that the inspected public project pages do not supply the authority needed to relax FINAL_UNSB's current release policy. A download link is not a redistribution grant, and a code license cannot be silently extended to dataset content.

The project therefore keeps the remaining unresolved domains and the MPMF aggregate bundle at `license_not_verified_do_not_redistribute`. The next valid resolution is written confirmation from an author or rights holder, or a separately verified governing data license. Until then, reproducibility must use acquisition/build instructions and hashes rather than redistributed images.

No data, manifest, training, checkpoint, metric, or confirmation state changed.

Evidence: `evidence/paper_aio/PAPER_AIO_REMAINING_PUBLIC_DATA_LICENSE_SEARCH_20260908T120000.json`.
