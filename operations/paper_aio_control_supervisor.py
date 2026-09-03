"""Restart fixed local paper control children without widening scientific scope.

The training supervisors own scientific full-state recovery.  This narrower
supervisor keeps long-lived audit-only successors alive while checkpoints are
still arriving.  It adopts an already-running child when deployed, freezes the
exact command and source identity, never reads metric values, and refuses to
restart a completed, blocked, or boundary-violating state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised on Linux deployment only.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows is covered instead.
    _fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-control-supervisor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-control-supervisor-state-v1"
COMMAND_SCHEMA = "final-unsb-paper-control-child-command-v1"
ROLE_SPECS = {
    "terminal_audit": {
        "module": "operations.paper_aio_local_terminal_audit_successor",
        "child_schema": "final-unsb-paper-local-terminal-audit-successor-state-v1",
        "final_status": "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS",
        "performance_must_remain_false": True,
    },
    "terminal_pathology": {
        "module": "operations.paper_aio_terminal_pathology_successor",
        "child_schema": "final-unsb-paper-terminal-pathology-successor-state-v1",
        "final_status": "COMPLETE_POSTHOC_TERMINAL_PATHOLOGY_ADJUDICATION",
        "performance_must_remain_false": False,
    },
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
        raise RuntimeError(f"fixed child command requires exactly one {name}")
    return command[indices[0] + 1]


def validate_child_command(
    path: Path, *, role: str, repo: Path, required_commit: str
) -> dict[str, Any]:
    payload = _read_json(path)
    spec = ROLE_SPECS[role]
    command = payload.get("command")
    if (
        payload.get("schema") != COMMAND_SCHEMA
        or payload.get("role") != role
        or not isinstance(command, list)
        or len(command) < 8
        or any(not isinstance(value, str) or "\n" in value for value in command)
    ):
        raise RuntimeError("fixed child command payload is invalid")
    python = Path(command[0]).resolve()
    if not python.is_file() or command[1:4] != ["-u", "-m", spec["module"]]:
        raise RuntimeError("fixed child command module/runtime is invalid")
    if Path(str(payload.get("cwd", ""))).resolve() != repo:
        raise RuntimeError("fixed child command cwd is not the frozen repo")
    if Path(_argument(command, "--repo")).resolve() != repo:
        raise RuntimeError("fixed child command repo differs from its cwd")
    if _argument(command, "--required-control-git-commit") != required_commit:
        raise RuntimeError("fixed child command commit differs from supervisor")
    raw_state_path = payload.get("state_path")
    if not isinstance(raw_state_path, str) or not Path(raw_state_path).is_absolute():
        raise RuntimeError("fixed child state path must be absolute")
    state_path = Path(raw_state_path).resolve()
    lowered = [value.lower() for value in command]
    if "train" in lowered or any("confirmation" in value for value in lowered):
        raise RuntimeError("fixed audit control child attempts forbidden work")
    return {
        **payload,
        "command": command,
        "cwd": str(repo),
        "state_path": str(state_path),
    }


def child_state_decision(role: str, state: dict[str, Any]) -> str:
    """Return WAIT, COMPLETE, or BLOCK without inspecting any metric value."""
    if not state:
        return "WAIT"
    spec = ROLE_SPECS[role]
    if state.get("schema") != spec["child_schema"]:
        return "BLOCK"
    if (
        state.get("paired_metric_control") is not False
        or state.get("confirmation20_opened") is not False
    ):
        return "BLOCK"
    if spec["performance_must_remain_false"] and state.get(
        "performance_values_read"
    ) is not False:
        return "BLOCK"
    status = str(state.get("status", ""))
    if status == spec["final_status"]:
        return "COMPLETE"
    if status.startswith(("BLOCKED", "FAIL")):
        return "BLOCK"
    return "WAIT"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill(pid, 0) is unreliable on Windows.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _acquire_lock(handle) -> bool:
    if _fcntl is None:
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("control supervisor repo is not at its frozen commit")
    command_path = args.child_command.resolve()
    child = validate_child_command(
        command_path,
        role=args.role,
        repo=repo,
        required_commit=commit,
    )
    source = repo / "operations" / "paper_aio_control_supervisor.py"
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "role": args.role,
        "repo": str(repo),
        "control_git_commit": commit,
        "control_source": str(source),
        "control_source_sha256": _sha256(source),
        "child_command_path": str(command_path),
        "child_command_sha256": _sha256(command_path),
        "child": child,
        "poll_seconds": int(args.poll_seconds),
        "restart_delay_seconds": int(args.restart_delay_seconds),
        "max_restarts": int(args.max_restarts),
        "timeout_hours": float(args.timeout_hours),
        "performance_values_available_to_supervisor": False,
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
        raise RuntimeError("control supervisor frozen identity changed")
    observed = validate_child_command(
        Path(contract["child_command_path"]),
        role=contract["role"],
        repo=repo,
        required_commit=contract["control_git_commit"],
    )
    if observed != contract["child"]:
        raise RuntimeError("fixed child command changed")


def _state_payload(contract: dict[str, Any], *, status: str, **extra) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "role": contract["role"],
        "control_git_commit": contract["control_git_commit"],
        "child_state_path": contract["child"]["state_path"],
        "restart_count": int(extra.pop("restart_count", 0)),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.role not in ROLE_SPECS:
        raise ValueError("unsupported control child role")
    if not 5 <= args.poll_seconds <= 600 or not 1 <= args.restart_delay_seconds <= 600:
        raise ValueError("unsafe control supervisor polling configuration")
    if not 1 <= args.max_restarts <= 20 or args.timeout_hours < 24:
        raise ValueError("unsafe control supervisor recovery budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "CONTROL_SUPERVISOR_CONTRACT.json"
    state_path = output / "CONTROL_SUPERVISOR_STATE.json"
    lock_path = output / "CONTROL_SUPERVISOR.lock"
    log_path = output / "CONTROL_CHILD.log"
    contract = _contract(args)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("control supervisor contract changed")
    else:
        _atomic_json(contract_path, contract)

    started = time.time()
    restart_count = 0
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if not _acquire_lock(lock_handle):
            raise RuntimeError("control supervisor is already running")
        while True:
            _verify(contract)
            child_state_path = Path(contract["child"]["state_path"])
            child_state = _read_json(child_state_path) if child_state_path.is_file() else {}
            decision = child_state_decision(contract["role"], child_state)
            if decision == "COMPLETE":
                result = _state_payload(
                    contract,
                    status="COMPLETE_CHILD_TERMINAL",
                    restart_count=restart_count,
                    child_status=child_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK":
                result = _state_payload(
                    contract,
                    status="BLOCKED_CHILD_STATE",
                    restart_count=restart_count,
                    child_status=child_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                result = _state_payload(
                    contract,
                    status="BLOCKED_CONTROL_SUPERVISOR_TIMEOUT",
                    restart_count=restart_count,
                )
                _atomic_json(state_path, result)
                return result

            existing_pid = int(child_state.get("pid", 0) or 0)
            if existing_pid != os.getpid() and _pid_alive(existing_pid):
                _atomic_json(
                    state_path,
                    _state_payload(
                        contract,
                        status="MONITORING_EXISTING_CHILD",
                        restart_count=restart_count,
                        child_pid=existing_pid,
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
                log.write(
                    f"\n[{time.time():.3f}] launch {json.dumps(contract['child']['command'])}\n"
                )
                log.flush()
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                child = subprocess.Popen(
                    contract["child"]["command"],
                    cwd=contract["child"]["cwd"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
                restart_count += 1
                while child.poll() is None:
                    _verify(contract)
                    observed = (
                        _read_json(child_state_path) if child_state_path.is_file() else {}
                    )
                    observed_decision = child_state_decision(contract["role"], observed)
                    if observed_decision == "BLOCK":
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        result = _state_payload(
                            contract,
                            status="BLOCKED_CHILD_STATE",
                            restart_count=restart_count,
                            child_pid=child.pid,
                            child_status=observed.get("status"),
                        )
                        _atomic_json(state_path, result)
                        return result
                    _atomic_json(
                        state_path,
                        _state_payload(
                            contract,
                            status="CHILD_RUNNING",
                            restart_count=restart_count,
                            child_pid=child.pid,
                            child_status=observed.get("status"),
                        ),
                    )
                    time.sleep(contract["poll_seconds"])
                _atomic_json(
                    state_path,
                    _state_payload(
                        contract,
                        status="CHILD_EXITED_CHECKING_RECOVERY",
                        restart_count=restart_count,
                        child_pid=child.pid,
                        child_returncode=child.returncode,
                    ),
                )
            time.sleep(contract["restart_delay_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--role", choices=sorted(ROLE_SPECS), required=True)
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
