"""Recover a frozen paper successor only while it is safely waiting.

This guard closes a narrow operational gap: a successor can spend days waiting
for a metric-blind predecessor, while an ordinary health watcher can only
report that the successor died.  The guard adopts the existing process and may
restart the exact frozen command only in ``WAITING_FOR_PREDECESSOR_E200``.

Once the successor begins engineering gates, a capacity probe, or the plain
training handoff, this guard deliberately relinquishes recovery.  Restarting at
those phases could collide with an orphaned gate or training supervisor.  The
existing lane/supervisor health monitors remain authoritative after handoff.
No metric or confirmation data is read by this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - deployment is POSIX; tests run on Windows too.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None


COMMAND_SCHEMA = "final-unsb-paper-waiting-successor-command-v1"
CONTRACT_SCHEMA = "final-unsb-paper-waiting-successor-guard-contract-v1"
STATE_SCHEMA = "final-unsb-paper-waiting-successor-guard-state-v1"
CHILD_STATE_SCHEMA = "final-unsb-paper-cross-host-plain-successor-v2"
ROLE = "cross_host_plain_successor_wait"
WAIT_STATUS = "WAITING_FOR_PREDECESSOR_E200"
COMPLETE_STATUS = "COMPLETE_PLAIN_E200"
HANDOFF_STATUSES = {
    "PREDECESSOR_COMPLETE_STARTING_EXACT_GATES",
    "RUNNING_EXACT_ENGINEERING_GATES",
    "RUNNING_METRIC_BLIND_CO_RESIDENT_CAPACITY_GATE",
    "WAITING_FOR_CO_RESIDENT_RELEASE_AFTER_CAPACITY_GATE",
    "PLAIN_SUPERVISOR_RUNNING",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _argument(command: list[str], name: str) -> str:
    indices = [index for index, value in enumerate(command) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise RuntimeError(f"frozen successor command requires exactly one {name}")
    return command[indices[0] + 1]


def validate_child_command(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    command = payload.get("command")
    if (
        payload.get("schema") != COMMAND_SCHEMA
        or payload.get("role") != ROLE
        or not isinstance(command, list)
        or len(command) < 20
        or any(not isinstance(value, str) or "\n" in value for value in command)
    ):
        raise RuntimeError("frozen successor command payload is invalid")

    python = Path(command[0]).resolve()
    source = Path(command[1]).resolve()
    cwd = Path(str(payload.get("cwd", ""))).resolve()
    if not python.is_file() or not source.is_file():
        raise RuntimeError("frozen successor runtime/source is missing")
    if source.name != "paper_aio_cross_host_plain_successor.py":
        raise RuntimeError("guard can only recover the cross-host plain successor")
    # Legacy successors were intentionally launched with absolute script paths
    # from the operator's home directory.  The cwd need not equal the source
    # checkout; both are frozen independently in the immutable command payload.
    if not cwd.is_dir() or not cwd.is_absolute():
        raise RuntimeError("successor cwd is missing or not absolute")
    if payload.get("child_source_sha256") != _sha256(source):
        raise RuntimeError("frozen successor source hash changed")

    training_repo = Path(_argument(command, "--training-repo")).resolve()
    training_output = Path(_argument(command, "--training-output")).resolve()
    required_training_commit = _argument(
        command, "--required-training-git-commit"
    )
    if (
        _git(training_repo, "rev-parse", "HEAD") != required_training_commit
        or _git(training_repo, "status", "--porcelain")
    ):
        raise RuntimeError("frozen successor training checkout changed")
    state_path = Path(str(payload.get("state_path", ""))).resolve()
    expected_state = (
        training_output / "operations" / "CROSS_HOST_PLAIN_SUCCESSOR_STATE.json"
    ).resolve()
    if state_path != expected_state or not state_path.is_absolute():
        raise RuntimeError("successor state path is not bound to its output")
    if any("confirmation" in value.lower() for value in command):
        raise RuntimeError("successor command attempts confirmation access")
    if _argument(command, "--host-label") != "5090B_MATCHED_PLAIN":
        raise RuntimeError("guard is bound to the registered matched-plain lane")
    return {
        **payload,
        "command": command,
        "cwd": str(cwd),
        "state_path": str(state_path),
        "child_source": str(source),
        "training_repo": str(training_repo),
        "training_output": str(training_output),
        "required_training_commit": required_training_commit,
        "required_protocol_fingerprint": _argument(
            command, "--required-protocol-fingerprint"
        ),
    }


def child_state_decision(state: dict[str, Any]) -> str:
    """Return WAIT, HANDOFF, COMPLETE, or BLOCK without reading metrics."""
    if not state:
        return "BLOCK"
    if (
        state.get("schema") != CHILD_STATE_SCHEMA
        or state.get("performance_values_read") is not False
        or state.get("paired_metric_control") is not False
        or state.get("confirmation20_opened") is not False
    ):
        return "BLOCK"
    status = str(state.get("status", ""))
    if status == WAIT_STATUS:
        return "WAIT"
    if status == COMPLETE_STATUS:
        return "COMPLETE"
    if status in HANDOFF_STATUSES:
        return "HANDOFF"
    return "BLOCK"


def _process_command(pid: int) -> tuple[list[str], Path] | None:
    if pid <= 0 or os.name == "nt":
        return None
    proc = Path("/proc") / str(pid)
    try:
        command = [
            value.decode("utf-8")
            for value in (proc / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        cwd = Path(os.readlink(proc / "cwd")).resolve()
        return command, cwd
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _process_matches(pid: int, command: list[str], cwd: Path) -> bool:
    observed = _process_command(pid)
    return observed is not None and observed == (command, cwd)


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("waiting-successor guard checkout is not frozen")
    child_path = args.child_command.resolve()
    child = validate_child_command(child_path)
    source = repo / "operations" / "paper_aio_waiting_successor_guard.py"
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING_ONLY",
        "repo": str(repo),
        "control_git_commit": commit,
        "control_source": str(source),
        "control_source_sha256": _sha256(source),
        "child_command_path": str(child_path),
        "child_command_sha256": _sha256(child_path),
        "child": child,
        "poll_seconds": int(args.poll_seconds),
        "restart_delay_seconds": int(args.restart_delay_seconds),
        "max_restarts": int(args.max_restarts),
        "timeout_hours": float(args.timeout_hours),
        "recoverable_child_status": WAIT_STATUS,
        "handoff_statuses": sorted(HANDOFF_STATUSES),
        "performance_values_available_to_guard": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
        or _sha256(Path(contract["control_source"]))
        != contract["control_source_sha256"]
        or _sha256(Path(contract["child_command_path"]))
        != contract["child_command_sha256"]
    ):
        raise RuntimeError("waiting-successor guard frozen identity changed")
    if validate_child_command(Path(contract["child_command_path"])) != contract["child"]:
        raise RuntimeError("frozen successor command changed")


def _state_payload(contract: dict[str, Any], *, status: str, **extra) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "control_git_commit": contract["control_git_commit"],
        "child_state_path": contract["child"]["state_path"],
        "recoverable_child_status": WAIT_STATUS,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def _lock(handle) -> None:
    if _fcntl is None:
        raise RuntimeError("waiting-successor guard requires POSIX file locking")
    _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 5 <= args.poll_seconds <= 600 or not 1 <= args.restart_delay_seconds <= 600:
        raise ValueError("unsafe waiting-successor polling configuration")
    if not 1 <= args.max_restarts <= 10 or args.timeout_hours < 24:
        raise ValueError("unsafe waiting-successor recovery budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "WAITING_SUCCESSOR_GUARD_CONTRACT.json"
    state_path = output / "WAITING_SUCCESSOR_GUARD_STATE.json"
    lock_path = output / "WAITING_SUCCESSOR_GUARD.lock"
    log_path = output / "WAITING_SUCCESSOR_CHILD.log"
    contract = _contract(args)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("waiting-successor guard contract changed")
    else:
        _atomic_json(contract_path, contract)

    started = time.time()
    restart_count = 0
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        _lock(lock_handle)
        while True:
            _verify(contract)
            child_state = _read_json(Path(contract["child"]["state_path"]))
            decision = child_state_decision(child_state)
            child_pid = int(child_state.get("pid", 0) or 0)
            if decision == "COMPLETE":
                result = _state_payload(
                    contract,
                    status="COMPLETE_CHILD_TERMINAL",
                    restart_count=restart_count,
                    child_status=child_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if decision == "HANDOFF":
                result = _state_payload(
                    contract,
                    status="HANDOFF_STARTED_RECOVERY_RELINQUISHED",
                    restart_count=restart_count,
                    child_pid=child_pid,
                    child_status=child_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK":
                result = _state_payload(
                    contract,
                    status="BLOCKED_UNSAFE_CHILD_STATE",
                    restart_count=restart_count,
                    child_pid=child_pid,
                    child_status=child_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                result = _state_payload(
                    contract,
                    status="BLOCKED_WAITING_SUCCESSOR_TIMEOUT",
                    restart_count=restart_count,
                )
                _atomic_json(state_path, result)
                return result

            command = contract["child"]["command"]
            cwd = Path(contract["child"]["cwd"])
            observed = _process_command(child_pid)
            if observed is not None:
                if observed != (command, cwd):
                    result = _state_payload(
                        contract,
                        status="BLOCKED_CHILD_PID_OR_COMMAND_DRIFT",
                        restart_count=restart_count,
                        child_pid=child_pid,
                    )
                    _atomic_json(state_path, result)
                    return result
                _atomic_json(
                    state_path,
                    _state_payload(
                        contract,
                        status="MONITORING_EXISTING_WAITING_SUCCESSOR",
                        restart_count=restart_count,
                        child_pid=child_pid,
                        child_status=child_state.get("status"),
                    ),
                )
                time.sleep(contract["poll_seconds"])
                continue
            if restart_count >= contract["max_restarts"]:
                result = _state_payload(
                    contract,
                    status="BLOCKED_RESTART_BUDGET_EXHAUSTED",
                    restart_count=restart_count,
                    child_status=child_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result

            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{time.time():.3f}] launch frozen waiting successor\n")
                log.flush()
                child = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                restart_count += 1
                _atomic_json(
                    state_path,
                    _state_payload(
                        contract,
                        status="RESTARTED_FROZEN_WAITING_SUCCESSOR",
                        restart_count=restart_count,
                        child_pid=child.pid,
                    ),
                )
            time.sleep(contract["restart_delay_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--child-command", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--restart-delay-seconds", type=int, default=15)
    value.add_argument("--max-restarts", type=int, default=5)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not str(result["status"]).startswith("BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
