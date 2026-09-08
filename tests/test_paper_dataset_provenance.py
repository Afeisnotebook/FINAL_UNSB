from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "PAPER_DATASET_PROVENANCE_CONTRACT.json"
NOTICE = ROOT / "research" / "paper_aio" / "DATASET_PROVENANCE_AND_REUSE_NOTICE_EN.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_provenance_document_and_local_authorities_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for binding in [contract["document"], *contract["local_authorities"].values()]:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]


def test_provenance_counts_match_canonical_manifest() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    counts: Counter[tuple[str, str]] = Counter()
    with (ROOT / "manifests" / "FULL_DATA_MANIFEST.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            counts[(row["domain"], row["split"])] += 1

    assert set(contract["domains"]) == {
        "FoggyCityscapes",
        "LowLightTrafficData",
        "RainCityscapes",
        "RainDS-syn",
        "RSCityscapes",
        "SnowTrafficData",
    }
    for domain, record in contract["domains"].items():
        assert counts[(domain, "train")] == record["train"]
        assert counts[(domain, "discovery")] == 80
        assert counts[(domain, "confirmation")] == 20
        assert sum(counts[(domain, split)] for split in ("train", "discovery", "confirmation")) == record["physical"]

    assert sum(counts.values()) == contract["corpus_counts"]["physical_identities"]
    assert sum(value for (domain, split), value in counts.items() if split == "train") == contract["corpus_counts"]["train_identities"]


def test_provenance_notice_does_not_overstate_lineage_or_permission() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    text = NOTICE.read_text(encoding="utf-8")
    required = [
        "PRE-RESULT / NO EMPIRICAL CLAIM / CITYSCAPES TERMS VERIFIED / BUNDLE REDISTRIBUTION NOT CLEARED",
        "consistent with the MPMF-Net six-test-set bundle",
        "not directly comparable",
        "license_not_verified_do_not_redistribute",
        "controlled custom split",
        "same-stem targets and domain labels are unavailable to training",
        "official Cityscapes Terms and Conditions",
        "must not ship FoggyCityscapes, RainCityscapes, or RSCityscapes",
        "does not offer legal advice",
    ]
    for phrase in required:
        assert phrase in text

    assert contract["upstream_bundle_lineage"]["original_archive_receipt_retained_locally"] is False
    cityscapes = contract["cityscapes_terms"]
    assert cityscapes["url"] == "https://www.cityscapes-dataset.com/license/"
    assert cityscapes["third_party_dataset_access_allowed"] is False
    assert cityscapes["modified_or_derived_distribution_allowed_when_source_can_be_recovered"] is False
    assert set(cityscapes["affected_local_domains"]) == {
        "FoggyCityscapes", "RainCityscapes", "RSCityscapes",
    }
    assert contract["release_policy"]["image_bytes_allowed_in_public_repository"] is False
    assert all(value is False for value in contract["hard_boundaries"].values())
    for domain, item in contract["domains"].items():
        status = item["license_status"]
        if domain in cityscapes["affected_local_domains"]:
            assert "redistribut" in status
        else:
            assert "do_not_redistribute" in status


def test_provenance_contract_is_registered_without_changing_training() -> None:
    matrix = json.loads(
        (ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json").read_text(encoding="utf-8")
    )
    portfolio = json.loads(
        (ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json").read_text(encoding="utf-8")
    )
    project = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    matrix_item = next(
        item for item in matrix["nonblocking_extensions"]
        if item["id"] == "dataset_provenance_and_custom_reuse"
    )
    records = [
        matrix_item,
        portfolio["dataset_provenance_and_custom_reuse"],
        project["paper_aio_20260902"]["dataset_provenance_and_custom_reuse"],
    ]
    for record in records:
        assert record["document"] == str(NOTICE.relative_to(ROOT)).replace("\\", "/")
        assert record["contract"] == str(CONTRACT.relative_to(ROOT)).replace("\\", "/")
        blob = subprocess.check_output(
            ["git", "show", f"{record['contract_commit']}:configs/PAPER_DATASET_PROVENANCE_CONTRACT.json"],
            cwd=ROOT,
        )
        assert hashlib.sha256(blob).hexdigest() == record["contract_sha256"]
        assert record["training_queue_changed"] is False
        assert record["performance_values_read"] is False
        assert record["confirmation20_opened"] is False
