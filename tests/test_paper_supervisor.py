from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from operations.paper_aio_supervisor import (
    authorization_path,
    checkpoint_step,
    child_command,
    failure_count_after_run,
    lane_identity,
)


def test_static_paper_supervisor_contract_is_unchanged(tmp_path: Path) -> None:
    assert lane_identity("plain", None) == ("plain", "static")
    assert authorization_path(tmp_path, "plain", None).name == (
        "LANE_AUTHORIZATION_plain.json"
    )


def test_candidate_paper_supervisor_uses_candidate_authorization_and_id(
    tmp_path: Path,
) -> None:
    candidate_id = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"
    args = argparse.Namespace(
        lane="candidate", candidate_id=candidate_id,
        manifest=tmp_path / "manifest.csv", data_root=tmp_path / "data",
        train_view=tmp_path / "view", gpu=0,
    )
    assert authorization_path(tmp_path, "candidate", candidate_id).name == (
        f"CANDIDATE_AUTHORIZATION_{candidate_id}.json"
    )
    command = child_command(args, tmp_path)
    assert command[command.index("--lane") + 1] == "candidate"
    assert command[command.index("--candidate-id") + 1] == candidate_id


def test_candidate_paper_supervisor_fails_closed_without_safe_id() -> None:
    with pytest.raises(ValueError, match="requires a safe"):
        lane_identity("candidate", None)
    with pytest.raises(ValueError, match="only valid"):
        lane_identity("plain", "unexpected")


def test_supervisor_failure_streak_resets_only_after_checkpoint_progress(
    tmp_path: Path,
) -> None:
    lane = tmp_path / "lanes" / "plain"
    lane.mkdir(parents=True)
    assert checkpoint_step(tmp_path, "plain") is None

    sidecar = lane / "full_state_latest.pt.json"
    sidecar.write_text('{"lane_id":"plain","step":8553}\n', encoding="utf-8")
    assert checkpoint_step(tmp_path, "plain") == 8553
    assert failure_count_after_run(
        2, checkpoint_before=8553, checkpoint_after=17106,
    ) == (1, True)

    assert failure_count_after_run(
        1, checkpoint_before=17106, checkpoint_after=17106,
    ) == (2, False)
    assert failure_count_after_run(
        2, checkpoint_before=17106, checkpoint_after=17106,
    ) == (3, False)


def test_supervisor_does_not_credit_invalid_or_cross_lane_sidecar(tmp_path: Path) -> None:
    lane = tmp_path / "lanes" / "plain"
    lane.mkdir(parents=True)
    sidecar = lane / "full_state_latest.pt.json"
    sidecar.write_text('{"lane_id":"proposal","step":8553}\n', encoding="utf-8")
    assert checkpoint_step(tmp_path, "plain") is None
    sidecar.write_text('{not-json}\n', encoding="utf-8")
    assert checkpoint_step(tmp_path, "plain") is None
