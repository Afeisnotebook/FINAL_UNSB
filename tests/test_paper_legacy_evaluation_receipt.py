from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from operations.paper_aio_migrate_legacy_evaluation_receipt import migrate


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _receipts(root: Path, *, repeat_fingerprint: str = "eval") -> None:
    _write(root / "gates" / "PREFLIGHT.json", {
        "status": "PASS", "node_role": "training", "protocol_fingerprint": "train",
        "manifest": {"content_hashes_verified": True}, "confirmation20_opened": False,
    })
    _write(root / "gates" / "RESUME_GATE_proposal.json", {
        "status": "PASS", "lane_id": "proposal", "protocol_fingerprint": "train",
        "continuous_core_sha256": "same", "resumed_core_sha256": "same",
        "total_updates": 1000, "split_updates": 500, "confirmation20_opened": False,
    })
    _write(root / "gates" / "EVALUATION_REPEAT_proposal.json", {
        "schema": "final-unsb-paper-evaluation-repeat-gate-v1", "status": "PASS",
        "lane_id": "proposal", "first_result_sha256": "same",
        "second_result_sha256": "same", "protocol_fingerprint": repeat_fingerprint,
        "split": "discovery", "confirmation20_opened": False,
    })


def test_migrates_only_the_overloaded_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _receipts(tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "commit\n" if "rev-parse" in args[0] else "")
    result = migrate(
        output=tmp_path, lane="proposal", scientific_repo=tmp_path,
        required_scientific_commit="commit", training_fingerprint="train",
        evaluation_fingerprint="eval",
    )
    assert result["protocol_fingerprint"] == "train"
    assert result["evaluation_bundle_fingerprint"] == "eval"
    assert result["first_result_sha256"] == result["second_result_sha256"]
    assert result["migration"]["checkpoint_loaded"] is False
    assert result["migration"]["performance_values_read"] is False


def test_rejects_nonidentical_repeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _receipts(tmp_path)
    path = tmp_path / "gates" / "EVALUATION_REPEAT_proposal.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["second_result_sha256"] = "different"
    _write(path, value)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "commit\n" if "rev-parse" in args[0] else "")
    with pytest.raises(RuntimeError, match="not deterministic"):
        migrate(
            output=tmp_path, lane="proposal", scientific_repo=tmp_path,
            required_scientific_commit="commit", training_fingerprint="train",
            evaluation_fingerprint="eval",
        )
