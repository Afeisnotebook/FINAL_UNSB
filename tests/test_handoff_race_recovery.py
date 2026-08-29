from pathlib import Path

import pytest

from operations.local_route1_handoff_race_recovery import (
    paused_state,
    recovery_paths,
    validate_interrupted_state,
)


def _state(**changes):
    result = {
        "status": "CHUNK_RUNNING",
        "lane": "hnek",
        "start_data_epoch": 0,
        "target_data_epoch": 5,
        "executor_pid": 101,
        "child_pid": 202,
        "git_commit": "frozen",
        "protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "confirmation20_opened": False,
    }
    result.update(changes)
    return result


def test_recovery_only_accepts_bounded_first_hnek_chunk():
    assert validate_interrupted_state(_state()) == (101, 202)
    for invalid in (
        _state(lane="hj"),
        _state(start_data_epoch=5, target_data_epoch=10),
        _state(target_data_epoch=6),
        _state(executor_pid=-1),
    ):
        with pytest.raises(RuntimeError):
            validate_interrupted_state(invalid)


def test_paused_state_removes_active_lane_but_preserves_identity(tmp_path):
    original = _state()
    quarantine = tmp_path / "operations" / "quarantine" / "hnek"
    result = paused_state(original, partial_epoch=1, quarantine=quarantine)
    assert result["status"] == "PAUSED_AFTER_HJ_E200_FOR_VERIFIED_HNEK_IMPORT"
    assert "lane" not in result and "child_pid" not in result
    assert result["previous_lane"] == "hnek"
    assert result["previous_child_pid"] == 202
    assert result["git_commit"] == "frozen"
    assert result["protocol_fingerprint"] == "protocol"
    assert result["manifest_sha256"] == "manifest"
    assert result["confirmation20_opened"] is False


def test_recovery_artifacts_are_scoped_to_canonical_operations(tmp_path):
    canonical = tmp_path / "run"
    paths = recovery_paths(canonical)
    assert paths["record"].parent == canonical.resolve() / "operations"
    assert paths["events"].parent == canonical.resolve() / "operations"
    assert paths["quarantine"] == canonical.resolve() / "operations" / "quarantine"
