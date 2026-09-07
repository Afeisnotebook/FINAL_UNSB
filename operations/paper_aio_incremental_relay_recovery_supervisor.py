"""Durably recover frozen local incremental checkpoint relays.

The incremental relays are read-only delivery workers: they copy only the
pre-registered e100/e150/e200 checkpoint bundles and never inspect metrics.
This supervisor pins the relay contract, source checkout, runtime and command,
then adopts or restarts exactly one matching relay.  Authentication remains in
the inherited environment and is never written to disk or the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from operations.paper_aio_export_relay_recovery_supervisor import (
    _git,
    _read_json,
    _sha256,
    relay_source_identity,
)

try:  # pragma: no cover - Linux deployment path.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows deployment path.
    _fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-incremental-relay-recovery-contract-v1"
STATE_SCHEMA = "final-unsb-paper-incremental-relay-recovery-state-v1"
RELAY_CONTRACT_SCHEMA = "final-unsb-paper-incremental-audit-relay-contract-v1"
RELAY_STATE_SCHEMA = "final-unsb-paper-incremental-audit-relay-state-v1"
COMPLETE_RELAY_STATUS = "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT"
REQUIRED_EPOCHS = [100, 150, 200]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish state atomically, tolerating short Windows reader handles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil
        except ImportError:
            return False
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def relay_state_decision(value: dict[str, Any]) -> str:
    """Return WAIT, COMPLETE or BLOCK without reading performance values."""
    if not value:
        return "WAIT"
    if (
        value.get("schema") != RELAY_STATE_SCHEMA
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        return "BLOCK"
    status = str(value.get("status", ""))
    if status == COMPLETE_RELAY_STATUS:
        return "COMPLETE"
    if status.startswith(("FAIL", "BLOCK", "ERROR")):
        return "BLOCK"
    return "WAIT"


def _validate_relay_contract(value: dict[str, Any]) -> None:
    if (
        value.get("schema") != RELAY_CONTRACT_SCHEMA
        or value.get("status") != "FROZEN_WAITING"
        or value.get("required_epochs") != REQUIRED_EPOCHS
        or value.get("password_persisted") is not False
        or value.get("performance_values_available_to_scheduler") is not False
        or value.get("paired_metric_control") is not False
        or value.get("source_checkpoint_mutation") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("incremental relay contract violates the frozen boundary")
    password_env = str(value.get("password_env", ""))
    if not password_env.startswith("FINAL_UNSB_"):
        raise RuntimeError("incremental relay password variable is not namespaced")
    for key in (
        "relay_id", "source_host_label", "source_host", "source_user",
        "expected_host_key_sha256", "remote_export_root", "lane_id",
        "required_training_git_commit", "required_training_protocol_fingerprint",
        "required_manifest_sha256", "control_script", "control_script_sha256",
        "base_relay_script_sha256",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise RuntimeError(f"incremental relay contract lacks {key}")


def render_relay_command(python: Path, relay: dict[str, Any]) -> list[str]:
    return [
        str(Path(python).resolve()),
        str(Path(relay["control_script"]).resolve()),
        "--destination-root", str(Path(relay["destination_root"]).resolve()),
        "--relay-id", str(relay["relay_id"]),
        "--source-host-label", str(relay["source_host_label"]),
        "--source-host", str(relay["source_host"]),
        "--source-port", str(int(relay["source_port"])),
        "--source-user", str(relay["source_user"]),
        "--expected-host-key-sha256", str(relay["expected_host_key_sha256"]),
        "--remote-export-root", str(relay["remote_export_root"]),
        "--lane", str(relay["lane_id"]),
        "--required-training-git-commit",
        str(relay["required_training_git_commit"]),
        "--required-training-protocol-fingerprint",
        str(relay["required_training_protocol_fingerprint"]),
        "--required-manifest-sha256", str(relay["required_manifest_sha256"]),
        "--password-env", str(relay["password_env"]),
        "--poll-seconds", str(int(relay["poll_seconds"])),
        "--timeout-hours", str(float(relay["timeout_hours"])),
    ]


def _parse_command(command: list[str]) -> dict[str, Any] | None:
    if len(command) < 4:
        return None
    try:
        result: dict[str, Any] = {
            "python": str(Path(command[0]).resolve()),
            "script": str(Path(command[1]).resolve()),
        }
    except (OSError, ValueError):
        return None
    index = 2
    while index < len(command):
        if index + 1 >= len(command) or not command[index].startswith("--"):
            return None
        key, value = command[index], command[index + 1]
        if key in result:
            return None
        result[key] = value
        index += 2
    return result


def command_matches_contract(
    command: list[str], python: Path, relay: dict[str, Any],
) -> bool:
    observed = _parse_command(command)
    expected = _parse_command(render_relay_command(python, relay))
    if observed is None or expected is None:
        return False
    for key in ("--source-port", "--poll-seconds"):
        try:
            if int(observed[key]) != int(expected[key]):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        observed.pop(key)
        expected.pop(key)
    try:
        if float(observed["--timeout-hours"]) != float(expected["--timeout-hours"]):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    observed.pop("--timeout-hours")
    expected.pop("--timeout-hours")
    if os.name == "nt":
        for key in ("python", "script", "--destination-root"):
            observed[key] = str(observed[key]).casefold()
            expected[key] = str(expected[key]).casefold()
    return observed == expected


def _process_commands() -> Iterable[tuple[int, list[str]]]:
    if os.name == "nt":  # psutil is part of the pinned local evaluation runtime.
        try:
            import psutil
        except ImportError as error:  # pragma: no cover - fail closed on deployment.
            raise RuntimeError("Windows relay recovery requires psutil") from error
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = process.info.get("cmdline") or []
                if command:
                    yield int(process.info["pid"]), [str(part) for part in command]
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            command = [
                part.decode("utf-8")
                for part in (item / "cmdline").read_bytes().split(b"\0") if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError):
            continue
        if command:
            yield int(item.name), command


def _matching_processes(python: Path, relay: dict[str, Any]) -> list[int]:
    return sorted(
        pid for pid, command in _process_commands()
        if pid != os.getpid() and command_matches_contract(command, python, relay)
    )


def _acquire_lock(handle) -> bool:
    if _fcntl is not None:
        try:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    import msvcrt

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write("0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("incremental recovery checkout is not frozen")
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError("incremental relay runtime is absent")
    relay_path = args.relay_contract.resolve()
    relay = _read_json(relay_path)
    _validate_relay_contract(relay)
    script = Path(relay["control_script"]).resolve()
    base = script.with_name("paper_aio_export_relay.py")
    if (
        not script.is_file()
        or _sha256(script) != relay["control_script_sha256"]
        or not base.is_file()
        or _sha256(base) != relay["base_relay_script_sha256"]
    ):
        raise RuntimeError("incremental relay source differs from its frozen contract")
    child_repo = script.parents[1]
    child_identity = relay_source_identity(child_repo)
    password_env = str(relay["password_env"])
    if not os.environ.get(password_env):
        raise RuntimeError(f"missing incremental relay password environment: {password_env}")
    source = repo / "operations" / Path(__file__).name
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "control_repo": str(repo),
        "control_git_commit": commit,
        "control_source": str(source),
        "control_source_sha256": _sha256(source),
        "python": str(python),
        "python_sha256": _sha256(python),
        "relay_contract": str(relay_path),
        "relay_contract_sha256": _sha256(relay_path),
        "relay_state": str(args.relay_state.resolve()),
        "relay_id": relay["relay_id"],
        "relay_source_repo": str(child_repo),
        "relay_source_identity_mode": child_identity["mode"],
        "relay_source_git_commit": child_identity["git_commit"],
        "relay_source_sha256": relay["control_script_sha256"],
        "relay_base_source_sha256": relay["base_relay_script_sha256"],
        "command": render_relay_command(python, relay),
        "password_env": password_env,
        "password_value_persisted": False,
        "poll_seconds": int(args.poll_seconds),
        "restart_delay_seconds": int(args.restart_delay_seconds),
        "max_restarts": int(args.max_restarts),
        "timeout_hours": float(args.timeout_hours),
        "performance_values_available_to_supervisor": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify(contract: dict[str, Any]) -> dict[str, Any]:
    repo = Path(contract["control_repo"])
    relay_path = Path(contract["relay_contract"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
        or _sha256(Path(contract["control_source"])) != contract["control_source_sha256"]
        or _sha256(Path(contract["python"])) != contract["python_sha256"]
        or _sha256(relay_path) != contract["relay_contract_sha256"]
    ):
        raise RuntimeError("incremental recovery frozen identity changed")
    relay = _read_json(relay_path)
    _validate_relay_contract(relay)
    script = Path(relay["control_script"]).resolve()
    child_identity = relay_source_identity(Path(contract["relay_source_repo"]))
    if (
        child_identity["mode"] != contract["relay_source_identity_mode"]
        or child_identity["git_commit"] != contract["relay_source_git_commit"]
        or _sha256(script) != contract["relay_source_sha256"]
        or _sha256(script.with_name("paper_aio_export_relay.py"))
        != contract["relay_base_source_sha256"]
        or render_relay_command(Path(contract["python"]), relay) != contract["command"]
        or not os.environ.get(contract["password_env"])
    ):
        raise RuntimeError("incremental relay runtime, source or secret binding changed")
    return relay


def _state(contract: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "relay_id": contract["relay_id"],
        "control_git_commit": contract["control_git_commit"],
        "relay_state": contract["relay_state"],
        "restart_count": int(extra.pop("restart_count", 0)),
        "password_value_persisted": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 5 <= args.poll_seconds <= 600 or not 1 <= args.restart_delay_seconds <= 600:
        raise ValueError("unsafe incremental recovery timing")
    if not 1 <= args.max_restarts <= 20 or args.timeout_hours < 24:
        raise ValueError("unsafe incremental recovery budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "INCREMENTAL_RELAY_RECOVERY_CONTRACT.json"
    state_path = output / "INCREMENTAL_RELAY_RECOVERY_STATE.json"
    lock_path = output / "INCREMENTAL_RELAY_RECOVERY.lock"
    log_path = output / "INCREMENTAL_RELAY_CHILD.log"
    contract = _freeze(args)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("incremental recovery contract changed")
    else:
        _atomic_json(contract_path, contract)
    started = time.time()
    restart_count = 0
    with lock_path.open("a+", encoding="utf-8") as lock:
        if not _acquire_lock(lock):
            raise RuntimeError("incremental recovery supervisor already running")
        while True:
            relay = _verify(contract)
            relay_state_path = Path(contract["relay_state"])
            relay_state = _read_json(relay_state_path) if relay_state_path.is_file() else {}
            decision = relay_state_decision(relay_state)
            if decision == "COMPLETE":
                result = _state(
                    contract, "COMPLETE_RELAY_TERMINAL",
                    restart_count=restart_count, child_status=relay_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK":
                result = _state(
                    contract, "BLOCKED_RELAY_STATE",
                    restart_count=restart_count, child_status=relay_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                result = _state(
                    contract, "BLOCKED_RECOVERY_TIMEOUT", restart_count=restart_count,
                )
                _atomic_json(state_path, result)
                return result

            matches = _matching_processes(Path(contract["python"]), relay)
            advertised = int(relay_state.get("pid", 0) or 0)
            if advertised > 0 and _pid_alive(advertised) and advertised not in matches:
                result = _state(
                    contract, "BLOCKED_LIVE_PID_COMMAND_MISMATCH",
                    restart_count=restart_count, child_pid=advertised,
                )
                _atomic_json(state_path, result)
                return result
            if len(matches) > 1:
                result = _state(
                    contract, "BLOCKED_DUPLICATE_RELAY_PROCESSES",
                    restart_count=restart_count, child_pids=matches,
                )
                _atomic_json(state_path, result)
                return result
            if matches:
                _atomic_json(
                    state_path,
                    _state(
                        contract, "MONITORING_EXISTING_RELAY",
                        restart_count=restart_count, child_pid=matches[0],
                        child_status=relay_state.get("status"),
                    ),
                )
                time.sleep(contract["poll_seconds"])
                continue
            if restart_count >= contract["max_restarts"]:
                result = _state(
                    contract, "BLOCKED_RESTART_BUDGET_EXHAUSTED",
                    restart_count=restart_count, child_status=relay_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{time.time():.3f}] launch incremental relay {contract['relay_id']}\n")
                log.flush()
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                subprocess.Popen(
                    contract["command"], cwd=contract["relay_source_repo"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
            restart_count += 1
            _atomic_json(
                state_path,
                _state(contract, "RELAY_RESTARTED", restart_count=restart_count),
            )
            time.sleep(contract["restart_delay_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--relay-contract", type=Path, required=True)
    value.add_argument("--relay-state", type=Path, required=True)
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
