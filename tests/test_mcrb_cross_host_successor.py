from __future__ import annotations

import copy

import pytest

from operations.local_route1_mcrb_cross_host_successor import (
    CANDIDATE_ID,
    REMOTE_FAIL,
    REMOTE_PASS,
    validate_remote_receipt,
    validate_remote_trajectory,
)


def _trajectory(status: str = REMOTE_PASS) -> dict:
    return {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "candidate_id": CANDIDATE_ID,
        "status": status,
        "trajectory": [
            {"epoch": 150, "updates": 22500},
            {"epoch": 175, "updates": 26250},
            {"epoch": 200, "updates": 30000},
        ],
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }


def _receipt(status: str = REMOTE_PASS) -> dict:
    return {
        "schema": "final-unsb-route1-candidate-terminal-receipt-v1",
        "candidate_id": CANDIDATE_ID,
        "trajectory_status": status,
        "terminal_integrity": {
            "status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET",
            "milestone_checkpoint_sha256": {"200": "checkpoint"},
            "metric_sha256": {"200": "metric"},
            "evaluation_crn_matched_to_plain": True,
            "paired_metric_used_for_training_or_control": False,
            "confirmation20_opened": False,
        },
        "paired_metrics_used_for_training_or_control": False,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "confirmation20_opened": False,
    }


@pytest.mark.parametrize("status", [REMOTE_PASS, REMOTE_FAIL])
def test_complete_remote_mcrb_e200_is_valid_routing_evidence(status: str) -> None:
    trajectory = validate_remote_trajectory(_trajectory(status))
    receipt = validate_remote_receipt(_receipt(status), trajectory)
    assert receipt["trajectory_status"] == status


def test_intermediate_remote_result_cannot_route_4090_replay() -> None:
    value = _trajectory()
    value["trajectory"][-1] = {"epoch": 175, "updates": 26250}
    with pytest.raises(RuntimeError, match="e200/30000"):
        validate_remote_trajectory(value)


def test_remote_receipt_must_be_post_trajectory_and_target_blind() -> None:
    trajectory = _trajectory()
    value = _receipt()
    value["paired_metrics_used_only_after_complete_trajectory"] = False
    with pytest.raises(RuntimeError, match="before trajectory completion"):
        validate_remote_receipt(value, trajectory)

    value = _receipt()
    value["paired_metrics_used_for_training_or_control"] = True
    with pytest.raises(RuntimeError, match="training or control"):
        validate_remote_receipt(value, trajectory)


def test_remote_receipt_status_and_same_host_plain_crn_are_bound() -> None:
    trajectory = _trajectory(REMOTE_FAIL)
    value = _receipt(REMOTE_PASS)
    with pytest.raises(RuntimeError, match="status mismatch"):
        validate_remote_receipt(value, trajectory)

    value = _receipt(REMOTE_FAIL)
    value["terminal_integrity"]["evaluation_crn_matched_to_plain"] = False
    with pytest.raises(RuntimeError, match="same-host plain CRN"):
        validate_remote_receipt(value, trajectory)


def test_remote_confirmation_access_is_fail_closed() -> None:
    trajectory = _trajectory()
    trajectory["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="confirmation20"):
        validate_remote_trajectory(trajectory)

    trajectory = _trajectory()
    receipt = copy.deepcopy(_receipt())
    receipt["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="confirmation20"):
        validate_remote_receipt(receipt, trajectory)

    receipt = copy.deepcopy(_receipt())
    receipt["terminal_integrity"]["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="terminal integrity opened confirmation20"):
        validate_remote_receipt(receipt, trajectory)


def test_remote_terminal_integrity_cannot_hide_paired_control() -> None:
    receipt = copy.deepcopy(_receipt())
    receipt["terminal_integrity"]["paired_metric_used_for_training_or_control"] = True
    with pytest.raises(RuntimeError, match="terminal integrity used paired"):
        validate_remote_receipt(receipt, _trajectory())
