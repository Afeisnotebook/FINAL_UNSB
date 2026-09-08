import hashlib
import json
import re
from pathlib import Path

from research.paper_aio import reference_ledger


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "configs" / "PAPER_REFERENCE_LEDGER.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_reference_ledger_binds_complete_core_bibliography() -> None:
    ledger = _load_ledger()
    assert ledger["schema"] == "final-unsb-paper-reference-ledger-v1"
    assert ledger["status"] == "PRE_RESULT_PRIMARY_METADATA_LOCK_CORE_STABLE_VOLATILE_REFRESH_REQUIRED"

    bib_path = ROOT / ledger["bib"]["path"]
    assert bib_path.is_file()
    assert _sha256(bib_path) == ledger["bib"]["sha256"]

    bib_keys = re.findall(r"^@\w+\{([^,]+),", bib_path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    entry_keys = [entry["citation_key"] for entry in ledger["entries"]]
    assert len(bib_keys) == ledger["bib"]["entry_count"] == 20
    assert len(bib_keys) == len(set(bib_keys))
    assert len(entry_keys) == len(set(entry_keys))
    assert set(bib_keys) == set(entry_keys)


def test_reference_ledger_primary_sources_and_required_groups_are_closed() -> None:
    ledger = _load_ledger()
    entry_keys = {entry["citation_key"] for entry in ledger["entries"]}

    for entry in ledger["entries"]:
        assert entry["metadata_status"] == "verified_primary"
        assert entry["primary_url"].startswith("https://")
        assert isinstance(entry["year"], int)
        assert entry["roles"]

    grouped_keys = {
        key
        for keys in ledger["required_groups"].values()
        for key in keys
    }
    assert grouped_keys == entry_keys


def test_volatile_neighbors_are_excluded_and_scientific_boundaries_hold() -> None:
    ledger = _load_ledger()
    core_keys = {entry["citation_key"] for entry in ledger["entries"]}
    volatile = ledger["submission_day_refresh_required"]

    assert volatile
    assert len({entry["working_key"] for entry in volatile}) == len(volatile)
    assert core_keys.isdisjoint({entry["working_key"] for entry in volatile})
    for entry in volatile:
        assert entry["status"] == "volatile_not_in_core_bib"
        assert entry["primary_url"].startswith("https://")

    authority = ledger["authority"]
    assert authority["metadata_lock_is_novelty_proof"] is False
    assert authority["volatile_entries_require_fresh_primary_source_review_before_submission"] is True
    assert all(value is False for value in ledger["scientific_boundaries"].values())


def test_runtime_reference_validator_binds_ledger_and_bibliography() -> None:
    reference = reference_ledger.reference_ledger_reference(root=ROOT)
    assert reference["path"] == "configs/PAPER_REFERENCE_LEDGER.json"
    assert reference["bibliography_path"] == (
        "research/paper_aio/PAPER_REFERENCES_CORE_PRE_RESULT.bib"
    )
    assert reference["entry_count"] == 20
    assert reference["submission_day_refresh_count"] == 11
    assert reference["sha256"] == _sha256(LEDGER_PATH)
