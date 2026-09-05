import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import operations.paper_aio_control_supervisor as supervisor
from operations.paper_aio_control_supervisor import (
    COMMAND_SCHEMA,
    _pid_alive,
    child_state_decision,
    validate_child_command,
)


def _command(tmp_path: Path, role: str, module: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    python = tmp_path / "python.exe"
    python.write_bytes(b"runtime")
    state = tmp_path / "state.json"
    path = tmp_path / f"{role}.json"
    path.write_text(
        json.dumps(
            {
                "schema": COMMAND_SCHEMA,
                "role": role,
                "cwd": str(repo),
                "state_path": str(state),
                "command": [
                    str(python),
                    "-u",
                    "-m",
                    module,
                    "--repo",
                    str(repo),
                    "--required-control-git-commit",
                    "abc123",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_fixed_child_command_accepts_only_bound_audit_module(tmp_path):
    path = _command(
        tmp_path,
        "terminal_audit",
        "operations.paper_aio_local_terminal_audit_successor",
    )
    result = validate_child_command(
        path,
        role="terminal_audit",
        repo=(tmp_path / "repo").resolve(),
        required_commit="abc123",
    )
    assert result["state_path"] == str((tmp_path / "state.json").resolve())


def test_pid_liveness_distinguishes_current_and_exited_process():
    assert _pid_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    assert not _pid_alive(child.pid)


def test_atomic_state_write_retries_transient_replace_denial(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    real_replace = Path.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient reader lock")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)
    supervisor._atomic_json(path, {"status": "HEALTHY"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "HEALTHY"}
    assert attempts == 3


def test_fixed_child_command_rejects_training_or_confirmation(tmp_path):
    path = _command(
        tmp_path,
        "terminal_audit",
        "operations.paper_aio_local_terminal_audit_successor",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["command"].extend(["--stage", "train"])
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden work"):
        validate_child_command(
            path,
            role="terminal_audit",
            repo=(tmp_path / "repo").resolve(),
            required_commit="abc123",
        )


def test_fixed_child_command_rejects_relative_state_path(tmp_path):
    path = _command(
        tmp_path,
        "terminal_audit",
        "operations.paper_aio_local_terminal_audit_successor",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["state_path"] = "relative/state.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="state path must be absolute"):
        validate_child_command(
            path,
            role="terminal_audit",
            repo=(tmp_path / "repo").resolve(),
            required_commit="abc123",
        )


def test_terminal_audit_state_decision_is_fail_closed():
    base = {
        "schema": "final-unsb-paper-local-terminal-audit-successor-state-v1",
        "status": "WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert child_state_decision("terminal_audit", base) == "WAIT"
    assert child_state_decision(
        "terminal_audit",
        {**base, "status": "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS"},
    ) == "COMPLETE"
    assert child_state_decision(
        "terminal_audit", {**base, "paired_metric_control": True}
    ) == "BLOCK"
    assert child_state_decision(
        "terminal_audit", {**base, "performance_values_read": True}
    ) == "BLOCK"
    assert child_state_decision(
        "terminal_audit", {**base, "status": "BLOCKED_TEST"}
    ) == "BLOCK"


def test_terminal_pathology_allows_only_posthoc_performance_state():
    base = {
        "schema": "final-unsb-paper-terminal-pathology-successor-state-v1",
        "status": "WAITING_FOR_COMPLETE_TARGET_BLIND_AUDITS",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert child_state_decision("terminal_pathology", base) == "WAIT"
    complete = {
        **base,
        "status": "COMPLETE_POSTHOC_TERMINAL_PATHOLOGY_ADJUDICATION",
        "performance_values_read_posthoc": True,
    }
    assert child_state_decision("terminal_pathology", complete) == "COMPLETE"
    assert child_state_decision(
        "terminal_pathology", {**base, "confirmation20_opened": True}
    ) == "BLOCK"


def test_local_export_push_state_is_recoverable_but_fail_closed() -> None:
    base = {
        "schema": "final-unsb-paper-local-export-push-state-v1",
        "status": "WAITING_FOR_COMPLETE_LOCAL_EXPORT_OR_TRANSIENT_NETWORK",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert child_state_decision("local_export_push", base) == "WAIT"
    assert child_state_decision(
        "local_export_push",
        {**base, "status": "COMPLETE_VERIFIED_REMOTE_IMPORT"},
    ) == "COMPLETE"
    assert child_state_decision(
        "local_export_push", {**base, "status": "FAIL_CLOSED_TEST"},
    ) == "BLOCK"
    assert child_state_decision(
        "local_export_push", {**base, "performance_values_read": True},
    ) == "BLOCK"


def _run_contract(tmp_path: Path, child_state: Path) -> dict:
    return {
        "schema": supervisor.CONTRACT_SCHEMA,
        "status": "FROZEN",
        "role": "terminal_audit",
        "repo": str(tmp_path),
        "control_git_commit": "abc123",
        "control_source": str(tmp_path / "source.py"),
        "control_source_sha256": "source",
        "child_command_path": str(tmp_path / "command.json"),
        "child_command_sha256": "command",
        "child": {
            "command": ["python", "-u", "-m", "fixed.module"],
            "cwd": str(tmp_path),
            "state_path": str(child_state),
        },
        "poll_seconds": 5,
        "restart_delay_seconds": 1,
        "max_restarts": 2,
        "timeout_hours": 24.0,
        "performance_values_available_to_supervisor": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _run_args(tmp_path: Path) -> SimpleNamespace:
    command = tmp_path / "command.json"
    command.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        repo=tmp_path,
        required_control_git_commit="abc123",
        role="terminal_audit",
        child_command=command,
        output=tmp_path / "supervisor",
        poll_seconds=5,
        restart_delay_seconds=1,
        max_restarts=2,
        timeout_hours=24.0,
    )


def _audit_state(status: str, *, pid: int = 0) -> dict:
    return {
        "schema": "final-unsb-paper-local-terminal-audit-successor-state-v1",
        "status": status,
        "pid": pid,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_supervisor_adopts_live_child_without_duplicate_launch(tmp_path, monkeypatch):
    child_state = tmp_path / "child.json"
    child_state.write_text(
        json.dumps(_audit_state("WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU", pid=4242)),
        encoding="utf-8",
    )
    contract = _run_contract(tmp_path, child_state)
    monkeypatch.setattr(supervisor, "_contract", lambda args: contract)
    monkeypatch.setattr(supervisor, "_verify", lambda value: None)
    monkeypatch.setattr(supervisor, "_pid_alive", lambda pid: pid == 4242)

    def complete_after_adoption(_seconds):
        child_state.write_text(
            json.dumps(
                _audit_state(
                    "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS",
                    pid=4242,
                )
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(supervisor.time, "sleep", complete_after_adoption)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("adoption launched a duplicate child"),
    )
    result = supervisor.run(_run_args(tmp_path))
    assert result["status"] == "COMPLETE_CHILD_TERMINAL"
    assert result["restart_count"] == 0


def test_supervisor_restarts_dead_waiting_child_and_accepts_terminal_state(
    tmp_path, monkeypatch
):
    child_state = tmp_path / "child.json"
    child_state.write_text(
        json.dumps(_audit_state("WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU", pid=4242)),
        encoding="utf-8",
    )
    contract = _run_contract(tmp_path, child_state)
    monkeypatch.setattr(supervisor, "_contract", lambda args: contract)
    monkeypatch.setattr(supervisor, "_verify", lambda value: None)
    monkeypatch.setattr(supervisor, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)

    class CompletedChild:
        pid = 5151
        returncode = 0

        def poll(self):
            child_state.write_text(
                json.dumps(
                    _audit_state(
                        "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS",
                        pid=self.pid,
                    )
                ),
                encoding="utf-8",
            )
            return self.returncode

    launches = []

    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return CompletedChild()

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    result = supervisor.run(_run_args(tmp_path))
    assert len(launches) == 1
    assert result["status"] == "COMPLETE_CHILD_TERMINAL"
    assert result["restart_count"] == 1


def test_supervisor_never_restarts_blocked_child(tmp_path, monkeypatch):
    child_state = tmp_path / "child.json"
    child_state.write_text(
        json.dumps(_audit_state("BLOCKED_TEST", pid=0)), encoding="utf-8"
    )
    contract = _run_contract(tmp_path, child_state)
    monkeypatch.setattr(supervisor, "_contract", lambda args: contract)
    monkeypatch.setattr(supervisor, "_verify", lambda value: None)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("blocked child was restarted"),
    )
    result = supervisor.run(_run_args(tmp_path))
    assert result["status"] == "BLOCKED_CHILD_STATE"
    assert result["restart_count"] == 0
