from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations import paper_aio_paused_plain_resume_successor as successor
from research.paper_aio.protocol import file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _predecessor(status: str = "CANDIDATE_SUPERVISOR_RUNNING") -> dict:
    return {
        "schema": "final-unsb-paper-candidate-continuation-v1",
        "status": status,
        "candidate_id": "G4-STCGR",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_predecessor_decision_is_metric_blind() -> None:
    assert successor.predecessor_decision(
        "CANDIDATE_SUPERVISOR_RUNNING", timed_out=False,
    ) == "WAIT"
    assert successor.predecessor_decision(
        "COMPLETE_CANDIDATE_E200", timed_out=False,
    ) == "START"
    assert successor.predecessor_decision(
        "BLOCKED_SUPERVISOR_EXIT", timed_out=False,
    ) == "BLOCK"
    assert successor.predecessor_decision(None, timed_out=True) == "TIMEOUT"


def test_predecessor_rejects_boundary_contamination() -> None:
    successor.validate_predecessor(
        _predecessor("COMPLETE_CANDIDATE_E200"),
        candidate_id="G4-STCGR", require_complete=True,
    )
    broken = _predecessor("COMPLETE_CANDIDATE_E200")
    broken["performance_values_read"] = True
    with pytest.raises(RuntimeError, match="invalid or not complete"):
        successor.validate_predecessor(
            broken, candidate_id="G4-STCGR", require_complete=True,
        )


def test_paused_plain_requires_exact_epoch_and_file_hash(tmp_path: Path) -> None:
    lane = tmp_path / "lanes" / "plain"
    checkpoint = lane / "full_state_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"full scientific state")
    digest = file_sha256(checkpoint)
    sidecar = {
        "schema": "final-unsb-paper-aio-full-state-v1",
        "lane_id": "plain",
        "step": 9 * 8_553,
        "physical_epoch_completed": 9,
        "target_steps": 1_710_600,
        "full_state_sha256": digest,
        "scientific_state_sha256": "s" * 64,
        "metadata": {
            "lane_id": "plain", "git_commit": "g" * 40,
            "protocol_fingerprint": "p" * 64, "seed": 2026,
            "batch_size": 1, "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    }
    _write(Path(str(checkpoint) + ".json"), sidecar)
    result = successor.validate_paused_plain(
        training_output=tmp_path, required_epoch=9,
        required_full_state_sha256=digest,
        required_scientific_state_sha256="s" * 64,
        required_git_commit="g" * 40,
        required_protocol_fingerprint="p" * 64,
    )
    assert result["step"] == 76_977
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="frozen identity"):
        successor.validate_paused_plain(
            training_output=tmp_path, required_epoch=9,
            required_full_state_sha256=digest,
            required_scientific_state_sha256="s" * 64,
            required_git_commit="g" * 40,
            required_protocol_fingerprint="p" * 64,
        )


def test_plain_authorization_must_remain_sealed(tmp_path: Path) -> None:
    auth = {
        "schema": "final-unsb-paper-lane-authorization-v1",
        "status": "PASS", "lane_id": "plain",
        "protocol_fingerprint": "p" * 64, "failures": [],
        "paired_metric_control": False, "confirmation20_opened": False,
    }
    path = tmp_path / "gates" / "LANE_AUTHORIZATION_plain.json"
    _write(path, auth)
    successor.validate_plain_authorization(
        training_output=tmp_path,
        required_protocol_fingerprint="p" * 64,
    )
    auth["confirmation20_opened"] = True
    _write(path, auth)
    with pytest.raises(RuntimeError, match="absent or changed"):
        successor.validate_plain_authorization(
            training_output=tmp_path,
            required_protocol_fingerprint="p" * 64,
        )
