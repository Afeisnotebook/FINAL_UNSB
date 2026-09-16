import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "paper_aio" / "final_v10"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_v10_archive_manifest_is_complete_and_hash_exact():
    manifest = json.loads((ARCHIVE / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE_HASH_VERIFIED_COMPACT_PAPER_ARCHIVE"
    assert manifest["entry_count"] == 21
    assert manifest["total_bytes"] == 320096
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]


def test_final_v10_archive_binds_terminal_portfolios_and_keeps_confirmation_sealed():
    portfolio = ARCHIVE / "PAPER_ALGORITHM_PORTFOLIO.json"
    augmented = ARCHIVE / "PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json"
    assert _sha256(portfolio) == "f70ca229f2abae7026a5d60f6f359164dfd14e7da6e58e2ceeae70718ed1e36c"
    assert _sha256(augmented) == "c1a8229c446c65a94fcf266320467208e11d7842a1c440faf4070db1884e919e"
    payload = json.loads(augmented.read_text(encoding="utf-8"))
    assert payload["accepted_algorithms"] == ["proposal"]
    assert payload["confirmation20_opened"] is False
    assert payload["confirmation_authorized"] is False


def test_server_inventory_keeps_independent_proofs_outside_canonical_result():
    inventory = json.loads((ARCHIVE / "SERVER_ASSET_INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["canonical_result_only_on_rental_server"] is False
    proofs = inventory["server_only_noncanonical_work"]
    assert proofs["5090A_independent_audit_20260915"]["sync_status"] == "ACTIVE_NOT_SYNCED_NOT_TOUCHED"
    assert inventory["destructive_action_taken"] is False
    assert inventory["confirmation20_opened"] is False
