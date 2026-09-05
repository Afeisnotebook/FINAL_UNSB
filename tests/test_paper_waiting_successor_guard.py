import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import paper_aio_waiting_successor_guard as guard


def _state(status: str, *, pid: int = 4242) -> dict:
    return {
        "schema": guard.CHILD_STATE_SCHEMA,
        "status": status,
        "pid": pid,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_state_decision_recovers_only_predecessor_wait() -> None:
    assert guard.child_state_decision(_state(guard.WAIT_STATUS)) == "WAIT"
    assert guard.child_state_decision(_state(guard.COMPLETE_STATUS)) == "COMPLETE"
    for status in guard.HANDOFF_STATUSES:
        assert guard.child_state_decision(_state(status)) == "HANDOFF"
    assert guard.child_state_decision(_state("BLOCKED_TEST")) == "BLOCK"
    assert guard.child_state_decision(_state("UNKNOWN")) == "BLOCK"
    assert guard.child_state_decision({**_state(guard.WAIT_STATUS), "paired_metric_control": True}) == "BLOCK"
    assert guard.child_state_decision({**_state(guard.WAIT_STATUS), "confirmation20_opened": True}) == "BLOCK"


def test_legacy_successor_cwd_is_frozen_but_need_not_equal_source_checkout(
    tmp_path, monkeypatch
) -> None:
    runtime = tmp_path / "python"
    runtime.write_bytes(b"runtime")
    control = tmp_path / "control"
    source = control / "operations" / "paper_aio_cross_host_plain_successor.py"
    source.parent.mkdir(parents=True)
    source.write_text("successor\n", encoding="utf-8")
    cwd = tmp_path / "operator-home"
    cwd.mkdir()
    training = tmp_path / "training"
    training.mkdir()
    output = tmp_path / "output"
    command = [
        str(runtime),
        str(source),
        "--training-repo",
        str(training),
        "--training-output",
        str(output),
        "--required-training-git-commit",
        "train123",
        "--required-protocol-fingerprint",
        "protocol123",
        "--host-label",
        "5090B_MATCHED_PLAIN",
        "--predecessor-state",
        str(tmp_path / "predecessor.json"),
        "--peer-runtime-receipt",
        str(tmp_path / "peer.json"),
        "--manifest",
        str(tmp_path / "manifest.csv"),
        "--data-root",
        str(tmp_path / "data"),
    ]
    payload = {
        "schema": guard.COMMAND_SCHEMA,
        "role": guard.ROLE,
        "cwd": str(cwd),
        "state_path": str(
            output / "operations" / "CROSS_HOST_PLAIN_SUCCESSOR_STATE.json"
        ),
        "child_source_sha256": guard._sha256(source),
        "command": command,
    }
    path = tmp_path / "command.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        guard,
        "_git",
        lambda repo, *args: "train123" if args[-1] == "HEAD" else "",
    )
    result = guard.validate_child_command(path)
    assert result["cwd"] == str(cwd.resolve())
    assert result["child_source"] == str(source.resolve())


def _contract(tmp_path: Path, child_state: Path) -> dict:
    return {
        "schema": guard.CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING_ONLY",
        "repo": str(tmp_path),
        "control_git_commit": "abc123",
        "control_source": str(tmp_path / "guard.py"),
        "control_source_sha256": "source",
        "child_command_path": str(tmp_path / "command.json"),
        "child_command_sha256": "command",
        "child": {
            "command": [sys.executable, "successor.py"],
            "cwd": str(tmp_path),
            "state_path": str(child_state),
        },
        "poll_seconds": 5,
        "restart_delay_seconds": 1,
        "max_restarts": 2,
        "timeout_hours": 24.0,
    }


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        repo=tmp_path,
        required_control_git_commit="abc123",
        child_command=tmp_path / "command.json",
        output=tmp_path / "output",
        poll_seconds=5,
        restart_delay_seconds=1,
        max_restarts=2,
        timeout_hours=24.0,
    )


def test_guard_adopts_exact_live_waiter_without_duplicate(tmp_path, monkeypatch):
    child_state = tmp_path / "state.json"
    child_state.write_text(json.dumps(_state(guard.WAIT_STATUS)), encoding="utf-8")
    contract = _contract(tmp_path, child_state)
    monkeypatch.setattr(guard, "_contract", lambda args: contract)
    monkeypatch.setattr(guard, "_verify", lambda value: None)
    monkeypatch.setattr(guard, "_lock", lambda handle: None)
    monkeypatch.setattr(
        guard,
        "_process_command",
        lambda pid: (contract["child"]["command"], Path(contract["child"]["cwd"])),
    )

    def finish_after_adoption(_seconds):
        child_state.write_text(
            json.dumps(_state(guard.COMPLETE_STATUS)), encoding="utf-8"
        )

    monkeypatch.setattr(guard.time, "sleep", finish_after_adoption)
    monkeypatch.setattr(
        guard.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("live waiter was duplicated"),
    )
    result = guard.run(_args(tmp_path))
    assert result["status"] == "COMPLETE_CHILD_TERMINAL"
    assert result["restart_count"] == 0


def test_guard_restarts_dead_waiter_with_exact_frozen_command(tmp_path, monkeypatch):
    child_state = tmp_path / "state.json"
    child_state.write_text(json.dumps(_state(guard.WAIT_STATUS)), encoding="utf-8")
    contract = _contract(tmp_path, child_state)
    monkeypatch.setattr(guard, "_contract", lambda args: contract)
    monkeypatch.setattr(guard, "_verify", lambda value: None)
    monkeypatch.setattr(guard, "_lock", lambda handle: None)
    monkeypatch.setattr(guard, "_process_command", lambda pid: None)
    monkeypatch.setattr(guard.time, "sleep", lambda seconds: None)
    launches = []

    class Child:
        pid = 5151

    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        child_state.write_text(
            json.dumps(_state("PREDECESSOR_COMPLETE_STARTING_EXACT_GATES", pid=5151)),
            encoding="utf-8",
        )
        return Child()

    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    result = guard.run(_args(tmp_path))
    assert len(launches) == 1
    assert launches[0][0][0] == contract["child"]["command"]
    assert result["status"] == "HANDOFF_STARTED_RECOVERY_RELINQUISHED"
    assert result["restart_count"] == 1


@pytest.mark.parametrize(
    "status,expected",
    [
        ("RUNNING_EXACT_ENGINEERING_GATES", "HANDOFF_STARTED_RECOVERY_RELINQUISHED"),
        ("BLOCKED_TEST", "BLOCKED_UNSAFE_CHILD_STATE"),
    ],
)
def test_guard_never_restarts_after_waiting_phase(
    tmp_path, monkeypatch, status, expected
):
    child_state = tmp_path / "state.json"
    child_state.write_text(json.dumps(_state(status, pid=0)), encoding="utf-8")
    contract = _contract(tmp_path, child_state)
    monkeypatch.setattr(guard, "_contract", lambda args: contract)
    monkeypatch.setattr(guard, "_verify", lambda value: None)
    monkeypatch.setattr(guard, "_lock", lambda handle: None)
    monkeypatch.setattr(
        guard.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("unsafe phase was restarted"),
    )
    result = guard.run(_args(tmp_path))
    assert result["status"] == expected
    assert result["restart_count"] == 0
