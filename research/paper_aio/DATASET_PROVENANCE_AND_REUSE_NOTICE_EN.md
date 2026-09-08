# Dataset Provenance and Reuse Notice

Status: `PRE-RESULT / NO EMPIRICAL CLAIM / CITYSCAPES TERMS VERIFIED / BUNDLE REDISTRIBUTION NOT CLEARED`

This notice records what is known, what is inferred, and what remains unverified
about the six-domain image corpus used by the controlled FINAL_UNSB study.  It
does not authorize redistribution of any image.  The frozen executable manifest
and data contract remain the authority for the bytes used by the experiments.

## 1. Local corpus identity

The canonical manifest is
`manifests/FULL_DATA_MANIFEST.csv`, SHA256
`02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744`.
It identifies 9,153 physical input/target identities:

| Domain | Physical | Train | Discovery | Sealed confirmation |
| --- | ---: | ---: | ---: | ---: |
| FoggyCityscapes | 4,575 | 4,475 | 80 | 20 |
| LowLightTrafficData | 736 | 636 | 80 | 20 |
| RainCityscapes | 1,188 | 1,088 | 80 | 20 |
| RainDS-syn | 200 | 100 | 80 | 20 |
| RSCityscapes | 1,188 | 1,088 | 80 | 20 |
| SnowTrafficData | 1,266 | 1,166 | 80 | 20 |
| **Total** | **9,153** | **8,553** | **480** | **120** |

The local dataset directories contain image pairs only.  No README, license,
download receipt, or original archive metadata was retained beside them.  The
manifest proves the bytes and split used by this study, but it does not by itself
prove where those bytes were downloaded.

## 2. Upstream bundle attribution

The six directory names exactly match the six testing datasets distributed by
the official MPMF-Net repository for *Multi-axis Prompt and Multi-dimension
Fusion Network for All-in-one Weather-degraded Image Restoration* (AAAI 2025).
The repository was audited at commit
`13feb0da6c33cbef3e686b21ef4e34b259c2bcd0`; its README provides separate
download links for all six names and cites DOI `10.1609/aaai.v39i8.32898`.

This establishes a high-confidence naming and protocol lineage, not a
cryptographic download lineage.  Until an original archive receipt or an
upstream checksum is recovered, the paper must say that the local corpus is
*consistent with the MPMF-Net six-test-set bundle* rather than claiming a
verified byte-for-byte copy of that download.

Primary upstream references:

- [MPMF-Net official repository, audited commit](https://github.com/chdwyb/MPMF-Net/tree/13feb0da6c33cbef3e686b21ef4e34b259c2bcd0)
- [MPMF-Net AAAI 2025 paper](https://ojs.aaai.org/index.php/AAAI/article/view/32898)

The audited MPMF-Net commit contains no top-level `LICENSE` file, and GitHub's
license endpoint returned 404 during this audit.  A download link is not a
redistribution license.  Consequently, FINAL_UNSB must not publish the image
files or a repackaged archive without written permission or a separately
verified governing license.

## 3. Per-domain citation chain

The MPMF-Net paper assigns the following degradation roles and citations.  The
status column deliberately distinguishes a citation from a verified data
license.

| Local domain | Role in MPMF-Net | Upstream citation chain | Reuse status for this project |
| --- | --- | --- | --- |
| FoggyCityscapes | haze | Sakaridis, Dai, and Van Gool, *Semantic Foggy Scene Understanding with Synthetic Data*, IJCV 2018 | Derived from Cityscapes. The current official Cityscapes terms prohibit third-party distribution of the dataset and recoverable modified versions. Do not redistribute. |
| RainCityscapes | rain-by-haze | Hu et al., *Depth-Attentional Features for Single-Image Rain Removal*, CVPR 2019 | The [official DAF-Net repository](https://github.com/xw-hu/DAF-Net/tree/ebf08cf2f357aa361861e7863942417d9d562b15) directs dataset access through Cityscapes. Its Apache-2.0 code license is not a data license, while Cityscapes' data terms prohibit redistribution. Do not redistribute. |
| RSCityscapes | rain-by-snow | Wen et al., *Restoring Vision in Rain-by-Snow Weather with Simple Attention-Based Sampling Cross-Hierarchy Transformer*, Pattern Recognition 2024 | MPMF-Net states that it is generated from RainCityscapes. In the absence of separate permission, the Cityscapes-derived chain remains subject to the no-redistribution boundary. Do not redistribute. |
| SnowTrafficData | snow | Chen et al., *Snow Removal in Video: A New Dataset and A Novel Method*, ICCV 2023 | Citation verified through the MPMF-Net paper; no license covering the local packaged images was verified. Do not redistribute. |
| LowLightTrafficData | low light | Li et al., *Benchmarking Single-Image Dehazing and Beyond*, TIP 2019 (cited as 2018 by MPMF-Net) | The MPMF-Net attribution is retained verbatim in the paper trail, but the construction and governing data terms of this derived low-light subset remain unverified. Do not redistribute. |
| RainDS-syn | rain-by-raindrop | Quan et al., *Removing Raindrops and Rain Streaks in One Go*, CVPR 2021 | Original benchmark citation verified; no license covering the local 200-pair subset was verified. Do not redistribute. |

Useful primary paper links:

- [Foggy Cityscapes / IJCV project paper](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)
- [RainCityscapes / CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.html)
- [Snow Removal in Video / ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Snow_Removal_in_Video_A_New_Dataset_and_A_Novel_Method_ICCV_2023_paper.html)
- [RainDS / CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Quan_Removing_Raindrops_and_Rain_Streaks_in_One_Go_CVPR_2021_paper.html)

### Cityscapes terms audit

The [official Cityscapes Terms and Conditions](https://www.cityscapes-dataset.com/license/)
were re-read on 2026-09-08.  Section 4.2 states that protected dataset content
must not be made accessible to third parties and extends that restriction to
modified or derived works from which the data can be reconstructed.  The
license agreement separately prohibits distribution of the dataset or modified
versions, while allowing abstract derivative representations such as trained
models only when the source data cannot be recovered.  It also limits the
licensed use to non-commercial purposes and requires attribution.

This is stronger evidence than merely failing to locate a license.  It confirms
that FINAL_UNSB must not ship FoggyCityscapes, RainCityscapes, or RSCityscapes
image bytes, transformed copies, or sample bundles that permit reconstruction,
unless the rights holder supplies separate written permission.  It does not by
itself prove the governing terms of the three non-Cityscapes domains or of the
MPMF-Net aggregate download, so the bundle-wide status remains unresolved.
This project records the policy boundary for reproducibility and release; it
does not offer legal advice.

## 4. Nonstandard reuse in FINAL_UNSB

MPMF-Net describes these six collections as testing datasets.  FINAL_UNSB does
not reproduce that paired supervised protocol.  It repurposes their combined
physical image pools into a new controlled unpaired research corpus:

1. 100 identities per domain are frozen before training: 80 discovery and 20
   sealed confirmation identities;
2. all remaining 8,553 identities form the training side;
3. A is traversed by seeded permutation and B is sampled independently from the
   full B marginal;
4. same-stem targets and domain labels are unavailable to training;
5. every compared method is retrained on this same custom split.

Therefore our values are **not directly comparable** to the paired, original
test-set numbers reported by MPMF-Net or its cited source papers.  The paper must
describe this as a controlled custom split of existing evaluation collections,
not as the official training split of any upstream benchmark.  External methods
in our main table are valid only when retrained under the registered FINAL_UNSB
exposure protocol; literature-reported numbers belong in a separate context
table, if used at all.

## 5. Release policy and actions before submission

- Keep all image bytes, archives, checkpoints containing sample reconstructions,
  and non-cleared path inventories out of the public Git repository.
- Publish code, split-generation logic, aggregate counts, and manifest hashes.
  Publish stems or per-file hashes only after their governing terms are checked.
- Cite MPMF-Net plus every underlying paper listed above.
- Ask the MPMF-Net authors to confirm the bundle's redistribution terms and, if
  possible, provide an archive checksum.  Record the response as evidence.
- Treat the official Cityscapes no-redistribution rule as binding for the three
  Cityscapes-derived chains; retain any user-specific historical account terms
  because they may be stricter.
- If permission remains unavailable, release a manifest builder that operates on
  user-obtained upstream files and a hash verifier, not the files themselves.
- Preserve the phrase `license_not_verified_do_not_redistribute` for every
  unresolved domain.  Absence of a license must never be rewritten as public
  domain or research-use permission.

This provenance gap does not invalidate already running controlled comparisons:
all lanes consume the same hash-locked local corpus.  It is, however, a paper and
artifact-release limitation that must be resolved or disclosed before submission.
