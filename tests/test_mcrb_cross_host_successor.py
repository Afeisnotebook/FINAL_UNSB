from __future__ import annotations

import copy
from pathlib import Path

import pytest

from operations.local_route1_mcrb_cross_host_successor import (
    CANDIDATE_ID,
    REMOTE_FAIL,
    REMOTE_PASS,
    MCRBCrossHostSuccessor,
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


def _orchestration_harness(status: str):
    successor = object.__new__(MCRBCrossHostSuccessor)
    calls: list[object] = []
    successor.contract_path = Path("contract.json")
    successor.event = lambda name, **fields: calls.append(("event", name, fields))
    successor.state = lambda name, **fields: calls.append(("state", name, fields))
    successor.wait_for_remote_e200 = lambda: (_trajectory(status), _receipt(status))
    revision = {"ranking": [{"candidate_id": "G2-FULL"}]}
    successor.wait_for_amtnc_revision = lambda: revision
    successor.prepare_mcrb_4090 = lambda: calls.append("prepare_4090")
    receipt_path = Path("mcrb-receipt.json")
    successor.run_mcrb_4090 = lambda: calls.append("run_4090") or receipt_path

    def finalize(value, receipt):
        calls.append(("finalize", value, receipt))
        return {"selected_candidate_id": "WINNER"}

    successor.materialize_final_selection = finalize
    successor.start_winner_ablation_successor = (
        lambda selection: calls.append(("start_ablations", selection)) or 1234
    )
    return successor, calls, revision, receipt_path


def test_complete_remote_negative_skips_4090_then_finalizes() -> None:
    successor, calls, revision, _ = _orchestration_harness(REMOTE_FAIL)
    assert successor.run() == 0
    assert "prepare_4090" not in calls
    assert "run_4090" not in calls
    assert ("finalize", revision, None) in calls
    finalization = calls.index(("finalize", revision, None))
    ablation = next(index for index, row in enumerate(calls) if row[0] == "start_ablations")
    assert finalization < ablation


def test_complete_remote_positive_replays_before_final_selection() -> None:
    successor, calls, revision, receipt_path = _orchestration_harness(REMOTE_PASS)
    assert successor.run() == 0
    prepare = calls.index("prepare_4090")
    replay = calls.index("run_4090")
    finalize = calls.index(("finalize", revision, receipt_path))
    ablation = next(index for index, row in enumerate(calls) if row[0] == "start_ablations")
    assert prepare < replay < finalize < ablation
