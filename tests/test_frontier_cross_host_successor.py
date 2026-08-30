from __future__ import annotations

from pathlib import Path

import pytest

from operations.local_route1_frontier_cross_host_successor import (
    NO_REPLAY,
    REPLAY_READY,
    FrontierCrossHostSuccessor,
    validate_remote_decision,
)
from research.local_route1.frontier_adjudication import FRONTIER_IDS


def _remote(candidate_id: str | None) -> tuple[dict, dict]:
    replay = candidate_id is not None
    algorithm = None if candidate_id is None else f"algorithm-{candidate_id}"
    decision = {
        "schema": "final-unsb-route1-frontier-4090-replay-decision-v1",
        "status": REPLAY_READY if replay else NO_REPLAY,
        "recommended_candidate_id": candidate_id,
        "recommended_algorithm_fingerprint": algorithm,
        "complete_e200_only": True,
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    adjudication = {
        "schema": "final-unsb-route1-frontier-e200-adjudication-v1",
        "recommended_4090_replay_candidate_id": candidate_id,
        "recommended_4090_replay_algorithm_fingerprint": algorithm,
        "selected_frontier_candidate_id": FRONTIER_IDS[0],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    return decision, adjudication


def test_remote_strict_terminal_decision_can_request_one_replay() -> None:
    decision, adjudication = _remote(FRONTIER_IDS[1])
    assert validate_remote_decision(decision, adjudication) == FRONTIER_IDS[1]


def test_remote_negative_terminal_decision_skips_replay() -> None:
    decision, adjudication = _remote(None)
    assert validate_remote_decision(decision, adjudication) is None


def test_remote_intermediate_or_cross_host_control_is_rejected() -> None:
    decision, adjudication = _remote(FRONTIER_IDS[0])
    decision["intermediate_metric_routing"] = True
    with pytest.raises(RuntimeError, match="intermediate_metric_routing"):
        validate_remote_decision(decision, adjudication)
    decision, adjudication = _remote(FRONTIER_IDS[0])
    adjudication["cross_host_deltas_merged"] = True
    with pytest.raises(RuntimeError, match="cross_host_deltas_merged"):
        validate_remote_decision(decision, adjudication)


def test_remote_decision_cannot_substitute_an_unfrozen_candidate() -> None:
    decision, adjudication = _remote("UNREGISTERED")
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_remote_decision(decision, adjudication)


def test_negative_orchestration_never_prepares_4090_replay(tmp_path: Path) -> None:
    successor = object.__new__(FrontierCrossHostSuccessor)
    successor.operations = tmp_path
    successor.wait_local_ablations = lambda: None
    decision, adjudication = _remote(None)
    successor.wait_remote_decision = lambda: (None, decision, adjudication)
    calls: list[str] = []
    successor.prepare_replay = lambda _candidate: calls.append("prepare")
    successor.run_replay = lambda *_args: calls.append("replay")
    states = []
    successor.state = lambda status, **fields: states.append((status, fields))
    assert successor.run() == 0
    assert calls == []
    assert states[-1][0] == "COMPLETE_REMOTE_FRONTIER_NEGATIVE_4090_REPLAY_SKIPPED"


def test_positive_orchestration_waits_then_runs_exactly_one_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor = object.__new__(FrontierCrossHostSuccessor)
    successor.operations = tmp_path
    order: list[str] = []
    successor.wait_local_ablations = lambda: order.append("local_complete")
    decision, adjudication = _remote(FRONTIER_IDS[0])
    successor.wait_remote_decision = lambda: (
        order.append("remote_complete") or FRONTIER_IDS[0], decision, adjudication,
    )
    contract = tmp_path / "contract.json"
    successor.prepare_replay = lambda candidate: order.append(f"prepare:{candidate}") or contract
    receipt = tmp_path / "receipt.json"
    successor.run_replay = lambda candidate, value: order.append(
        f"replay:{candidate}:{value.name}"
    ) or receipt
    successor.state = lambda status, **fields: order.append(f"state:{status}")
    monkeypatch.setattr(
        "operations.local_route1_frontier_cross_host_successor._validate_receipt",
        lambda path: {
            "algorithm_fingerprint": "algorithm",
            "trajectory_status": "status",
        },
    )
    monkeypatch.setattr(
        "operations.local_route1_frontier_cross_host_successor.support.file_sha256",
        lambda path: "receipt-sha",
    )
    assert successor.run() == 0
    assert order[:4] == [
        "local_complete",
        "remote_complete",
        f"prepare:{FRONTIER_IDS[0]}",
        f"replay:{FRONTIER_IDS[0]}:contract.json",
    ]
    assert sum(value.startswith("replay:") for value in order) == 1

