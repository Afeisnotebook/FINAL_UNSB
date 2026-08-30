from __future__ import annotations

from pathlib import Path

import pytest

import operations.local_route1_winner_ablation_successor as successor_module
from operations.local_route1_winner_ablation_successor import (
    WinnerAblationSuccessor,
)


def test_winner_ablation_e200_executors_are_strictly_sequential(
    tmp_path: Path, monkeypatch,
) -> None:
    successor = object.__new__(WinnerAblationSuccessor)
    successor.contract = {
        "python": "/python",
        "e200_execution_policy": "SEQUENTIAL_SINGLE_STREAM_BY_MEASURED_WALL_CLOCK",
    }
    successor.repo = tmp_path
    successor.run_root = tmp_path / "run"
    successor.operations = successor.run_root / "operations"
    successor.operations.mkdir(parents=True)
    successor._init_executor_contract = (
        lambda candidate_id: tmp_path / f"{candidate_id}.json"
    )

    lifecycle: list[tuple[str, str]] = []
    states: list[tuple[str, dict]] = []
    successor.state = lambda name, **fields: states.append((name, fields))
    successor.event = lambda name, **fields: lifecycle.append(
        (name, fields["candidate_id"]),
    )

    class FakeProcess:
        next_pid = 100

        def __init__(self, candidate_id: str):
            self.candidate_id = candidate_id
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def wait(self):
            lifecycle.append(("wait", self.candidate_id))
            return 0

    def popen(command, **_kwargs):
        candidate_id = Path(command[-1]).stem
        lifecycle.append(("start", candidate_id))
        return FakeProcess(candidate_id)

    monkeypatch.setattr(successor_module.subprocess, "Popen", popen)
    monkeypatch.setattr(successor_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        successor_module.support, "current_epoch", lambda _root, _candidate: 12,
    )

    successor.run_e200(["proposal", "observable"])

    assert lifecycle == [
        ("start", "proposal"),
        ("wait", "proposal"),
        ("WINNER_ABLATION_E200_CANDIDATE_COMPLETE", "proposal"),
        ("start", "observable"),
        ("wait", "observable"),
        ("WINNER_ABLATION_E200_CANDIDATE_COMPLETE", "observable"),
    ]
    assert [state for state, _ in states] == [
        "WINNER_ABLATION_E200_RUNNING_SINGLE_STREAM",
        "WINNER_ABLATION_E200_RUNNING_SINGLE_STREAM",
    ]
    assert states[1][1]["completed_candidate_ids"] == ["proposal"]


def test_first_winner_ablation_failure_prevents_second_e200_start(
    tmp_path: Path, monkeypatch,
) -> None:
    successor = object.__new__(WinnerAblationSuccessor)
    successor.contract = {
        "python": "/python",
        "e200_execution_policy": "SEQUENTIAL_SINGLE_STREAM_BY_MEASURED_WALL_CLOCK",
    }
    successor.repo = tmp_path
    successor.run_root = tmp_path / "run"
    successor.operations = successor.run_root / "operations"
    successor.operations.mkdir(parents=True)
    successor._init_executor_contract = (
        lambda candidate_id: tmp_path / f"{candidate_id}.json"
    )
    successor.state = lambda *_args, **_fields: None
    successor.event = lambda *_args, **_fields: None
    started: list[str] = []

    class FailedProcess:
        pid = 101

        def poll(self):
            return 1

        def wait(self):
            return 1

    def popen(command, **_kwargs):
        candidate_id = Path(command[-1]).stem
        started.append(candidate_id)
        return FailedProcess()

    monkeypatch.setattr(successor_module.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="proposal"):
        successor.run_e200(["proposal", "observable"])

    assert started == ["proposal"]
