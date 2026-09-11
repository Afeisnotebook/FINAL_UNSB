"""Keep the fail-closed DDSB public-source watcher alive without widening scope.

The child only polls frozen publication pages and repository queries.  This
supervisor adopts an existing child, freezes both repositories, the interpreter,
the child command and its contract, and restarts only a dead child whose latest
state still requests an authoritative source.  A source candidate remains a
manual-review event and can never authorize training here.
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

try:  # pragma: no cover - Linux deployments use flock.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows is exercised in tests/deployment.
    _fcntl = None


CONTRACT_SCHEMA = "final-unsb-ddsb-source-watch-recovery-contract-v1"
STATE_SCHEMA = "final-unsb-ddsb-source-watch-recovery-state-v1"
SOURCE_CONTRACT_SCHEMA = "final-unsb-ddsb-source-watch-contract-v1"
SOURCE_STATE_SCHEMA = "final-unsb-ddsb-source-watch-state-v1"
WAIT_STATUSES = {"WAITING_FOR_AUTHORITATIVE_SOURCE", "TRANSIENT_NETWORK_ERROR"}
REVIEW_STATUSES = {
    "AUTHORITATIVE_SOURCE_CANDIDATE_REVIEW_REQUIRED",
    "UNVERIFIED_SOURCE_CANDIDATE_REVIEW_REQUIRED",
}


def _read_json(path: Path) -> dict[str, Any]:
    for attempt in range(12):
        try:
            raw = path.read_text(encoding="utf-8-sig")
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.05 * (2**attempt), 1.0))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(0.05 * (2**attempt), 1.0))
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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(query_limited_information, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                and int(code.value) == still_active
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def source_state_decision(state: dict[str, Any]) -> str:
    if not state:
        return "BLOCK"
    if (
        state.get("schema") != SOURCE_STATE_SCHEMA
        or state.get("training_authorized") is not False
        or state.get("training_started") is not False
        or state.get("paired_metric_control") is not False
        or state.get("confirmation20_opened") is not False
    ):
        return "BLOCK"
    status = str(state.get("status", ""))
    if status in WAIT_STATUSES:
        return "WAIT"
    if status in REVIEW_STATUSES:
        return "REVIEW"
    if status == "TIMED_OUT_WITHOUT_AUTHORITY_DECISION":
        return "TIMEOUT"
    return "BLOCK"


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


def _repo_identity(repo: Path, expected: str, label: str) -> None:
    if _git(repo, "rev-parse", "HEAD") != expected or _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"{label} repo is not at its frozen clean commit")


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    control_repo = args.control_repo.resolve(strict=True)
    source_repo = args.source_repo.resolve(strict=True)
    _repo_identity(control_repo, args.required_control_git_commit, "control")
    _repo_identity(source_repo, args.required_source_git_commit, "source")
    python = args.python.resolve(strict=True)
    source_script = (source_repo / "operations" / "paper_aio_ddsb_source_watch.py").resolve(
        strict=True
    )
    control_script = (
        control_repo / "operations" / "paper_aio_ddsb_source_watch_recovery.py"
    ).resolve(strict=True)
    watch_output = args.watch_output.resolve(strict=True)
    source_contract_path = watch_output / "DDSB_SOURCE_WATCH_CONTRACT.json"
    source_contract = _read_json(source_contract_path)
    source_script_sha256 = _sha256(source_script)
    if (
        source_contract.get("schema") != SOURCE_CONTRACT_SCHEMA
        or source_contract.get("status") != "FROZEN_FAIL_CLOSED"
        or Path(str(source_contract.get("control_script", ""))).resolve() != source_script
        or source_contract.get("control_script_sha256") != source_script_sha256
        or Path(str(source_contract.get("output", ""))).resolve() != watch_output
        or source_contract.get("automatic_source_acceptance") is not False
        or source_contract.get("automatic_training_authorization") is not False
        or source_contract.get("paired_metric_control") is not False
        or source_contract.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("DDSB source-watch contract is not the frozen fail-closed child")
    command = [
        str(python),
        "-u",
        str(source_script),
        "--output",
        str(watch_output),
        "--poll-seconds",
        str(int(source_contract["poll_seconds"])),
        "--timeout-hours",
        str(float(source_contract["timeout_hours"])),
        "--request-timeout-seconds",
        str(float(source_contract["request_timeout_seconds"])),
    ]
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_FAIL_CLOSED",
        "control_repo": str(control_repo),
        "control_git_commit": args.required_control_git_commit,
        "control_script": str(control_script),
        "control_script_sha256": _sha256(control_script),
        "source_repo": str(source_repo),
        "source_git_commit": args.required_source_git_commit,
        "source_script": str(source_script),
        "source_script_sha256": source_script_sha256,
        "python": str(python),
        "python_sha256": _sha256(python),
        "watch_output": str(watch_output),
        "source_contract": str(source_contract_path),
        "source_contract_sha256": _sha256(source_contract_path),
        "child_command": command,
        "initial_child_pid": int(args.initial_child_pid),
        "poll_seconds": int(args.poll_seconds),
        "restart_delay_seconds": int(args.restart_delay_seconds),
        "max_restarts": int(args.max_restarts),
        "timeout_hours": float(args.timeout_hours),
        "automatic_source_acceptance": False,
        "automatic_training_authorization": False,
        "performance_values_available": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify(contract: dict[str, Any]) -> None:
    control_repo = Path(contract["control_repo"])
    source_repo = Path(contract["source_repo"])
    _repo_identity(control_repo, contract["control_git_commit"], "control")
    _repo_identity(source_repo, contract["source_git_commit"], "source")
    checks = (
        (contract["control_script"], contract["control_script_sha256"]),
        (contract["source_script"], contract["source_script_sha256"]),
        (contract["python"], contract["python_sha256"]),
        (contract["source_contract"], contract["source_contract_sha256"]),
    )
    if any(_sha256(Path(path)) != expected for path, expected in checks):
        raise RuntimeError("DDSB source-watch recovery identity changed")


def _state(contract: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "control_git_commit": contract["control_git_commit"],
        "source_git_commit": contract["source_git_commit"],
        "watch_output": contract["watch_output"],
        "restart_count": int(extra.pop("restart_count", 0)),
        "automatic_source_acceptance": False,
        "automatic_training_authorization": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 10 <= args.poll_seconds <= 600 or not 1 <= args.restart_delay_seconds <= 600:
        raise ValueError("unsafe recovery timing")
    if not 1 <= args.max_restarts <= 20 or args.timeout_hours < 24:
        raise ValueError("unsafe recovery budget")
    output = args.supervisor_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "DDSB_SOURCE_WATCH_RECOVERY_CONTRACT.json"
    state_path = output / "DDSB_SOURCE_WATCH_RECOVERY_STATE.json"
    lock_path = output / "DDSB_SOURCE_WATCH_RECOVERY.lock"
    log_path = output / "DDSB_SOURCE_WATCH_RECOVERY.log"
    contract = _contract(args)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("DDSB source-watch recovery contract changed")
    else:
        _atomic_json(contract_path, contract)
    _verify(contract)

    source_state_path = Path(contract["watch_output"]) / "DDSB_SOURCE_WATCH_STATE.json"
    initial_state = _read_json(source_state_path)
    if (
        int(initial_state.get("watcher_pid", 0) or 0) != contract["initial_child_pid"]
        or not _pid_alive(contract["initial_child_pid"])
        or source_state_decision(initial_state) != "WAIT"
    ):
        raise RuntimeError("initial DDSB source watcher is not uniquely adoptable")

    started = time.time()
    restart_count = 0
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if not _acquire_lock(lock_handle):
            raise RuntimeError("DDSB source-watch recovery supervisor already running")
        while True:
            _verify(contract)
            source_state = _read_json(source_state_path)
            decision = source_state_decision(source_state)
            child_pid = int(source_state.get("watcher_pid", 0) or 0)
            child_status = str(source_state.get("status", ""))
            if decision == "REVIEW":
                result = _state(
                    contract,
                    "COMPLETE_MANUAL_SOURCE_REVIEW_REQUIRED",
                    restart_count=restart_count,
                    child_pid=child_pid,
                    child_status=child_status,
                )
                _atomic_json(state_path, result)
                return result
            if decision == "TIMEOUT":
                result = _state(
                    contract,
                    "COMPLETE_SOURCE_WATCH_TIMEOUT_NO_AUTHORITY_DECISION",
                    restart_count=restart_count,
                    child_pid=child_pid,
                    child_status=child_status,
                )
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK":
                result = _state(
                    contract,
                    "BLOCKED_SOURCE_STATE_BOUNDARY",
                    restart_count=restart_count,
                    child_pid=child_pid,
                    child_status=child_status,
                )
                _atomic_json(state_path, result)
                return result
            if _pid_alive(child_pid):
                _atomic_json(
                    state_path,
                    _state(
                        contract,
                        "MONITORING_EXISTING_SOURCE_WATCH",
                        restart_count=restart_count,
                        child_pid=child_pid,
                        child_status=child_status,
                    ),
                )
                time.sleep(contract["poll_seconds"])
                continue
            if restart_count >= contract["max_restarts"]:
                result = _state(
                    contract,
                    "BLOCKED_RESTART_BUDGET_EXHAUSTED",
                    restart_count=restart_count,
                    child_pid=child_pid,
                    child_status=child_status,
                )
                _atomic_json(state_path, result)
                return result

            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{time.time():.3f}] restart frozen DDSB source watcher\n")
                log.flush()
                child = subprocess.Popen(
                    contract["child_command"],
                    cwd=contract["source_repo"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            restart_count += 1
            _atomic_json(
                state_path,
                _state(
                    contract,
                    "SOURCE_WATCH_RESTARTED",
                    restart_count=restart_count,
                    child_pid=child.pid,
                    prior_child_pid=child_pid,
                    child_status=child_status,
                ),
            )
            time.sleep(contract["restart_delay_seconds"])
            if time.time() - started >= contract["timeout_hours"] * 3600:
                result = _state(
                    contract,
                    "BLOCKED_RECOVERY_SUPERVISOR_TIMEOUT",
                    restart_count=restart_count,
                )
                _atomic_json(state_path, result)
                return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-repo", type=Path, required=True)
    parser.add_argument("--required-control-git-commit", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--required-source-git-commit", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--watch-output", type=Path, required=True)
    parser.add_argument("--initial-child-pid", type=int, required=True)
    parser.add_argument("--supervisor-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--restart-delay-seconds", type=int, default=10)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser.parse_args()


def main() -> int:
    run(arguments())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
