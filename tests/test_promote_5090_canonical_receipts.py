from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations import local_route1_promote_5090_canonical_receipts as promotion
from research.local_route1.cross_runtime_portfolio import REPLAY_IDS, RESULT_SCHEMA
from research.local_route1.protocol import file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> None:
    rows = []
    for candidate_id in REPLAY_IDS:
        receipt = _write(
            tmp_path / "operations" / "terminal_receipts"
            / f"{candidate_id}_5090.json",
            {
                "candidate_id": candidate_id,
                "algorithm_fingerprint": f"algorithm-{candidate_id}",
                "confirmation20_opened": False,
            },
        )
        rows.append({
            "candidate_id": candidate_id,
            "algorithm_fingerprint": f"algorithm-{candidate_id}",
            "receipt_path": str(receipt.resolve()),
            "receipt_sha256": file_sha256(receipt),
        })
    _write(tmp_path / "operations" / promotion.RESULT_FILE, {
        "schema": RESULT_SCHEMA,
        "status": "CROSS_RUNTIME_5090_PORTFOLIO_COMPLETE_E200",
        "candidate_results": rows,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })


def test_promotes_validated_receipts_by_exact_bytes(monkeypatch, tmp_path: Path):
    _fixture(tmp_path)
    monkeypatch.setattr(
        promotion, "_validate_receipt",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    result = promotion.promote_canonical_receipts(tmp_path)
    assert result["status"] == "CANONICAL_5090_RECEIPTS_REGISTERED"
    assert [row["candidate_id"] for row in result["promoted_receipts"]] == list(
        REPLAY_IDS
    )
    for candidate_id in REPLAY_IDS:
        source = (
            tmp_path / "operations" / "terminal_receipts"
            / f"{candidate_id}_5090.json"
        )
        destination = (
            tmp_path / "operations" / "terminal_receipts" / f"{candidate_id}.json"
        )
        assert destination.read_bytes() == source.read_bytes()


def test_existing_changed_canonical_receipt_fails_closed(monkeypatch, tmp_path: Path):
    _fixture(tmp_path)
    monkeypatch.setattr(
        promotion, "_validate_receipt",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    candidate_id = REPLAY_IDS[0]
    _write(
        tmp_path / "operations" / "terminal_receipts" / f"{candidate_id}.json",
        {"candidate_id": "changed"},
    )
    with pytest.raises(RuntimeError, match="canonical 5090 receipt differs"):
        promotion.promote_canonical_receipts(tmp_path)
