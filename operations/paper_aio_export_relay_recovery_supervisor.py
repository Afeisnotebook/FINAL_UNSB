"""Durably adopt and recover frozen cross-host checkpoint import relays.

The relay itself is metric-blind and source-bound.  This supervisor adds only
process continuity: it verifies the relay's frozen contract, source checkout,
command semantics and password-variable presence, adopts an already-running
matching process, and restarts it only while its state is non-terminal.  It
never persists authentication material or reads a scientific metric value.
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

try:  # pragma: no cover - Linux deployment path.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-export-relay-recovery-contract-v1"
STATE_SCHEMA = "final-unsb-paper-export-relay-recovery-state-v1"
RELAY_CONTRACT_SCHEMA = "final-unsb-paper-export-relay-contract-v1"
RELAY_STATE_SCHEMA = "final-unsb-paper-export-relay-state-v1"
COMPLETE_RELAY_STATUS = "COMPLETE_VERIFIED_IMPORT_SET"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
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
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def relay_source_identity(repo: Path) -> dict[str, Any]:
    """Bind a Git checkout when present, otherwise retain blob-hash identity.

    Some already-deployed relay controls are deliberately minimal immutable
    copies without ``.git`` metadata.  Their frozen relay contract still binds
    the exact executable script SHA256; absence of repository metadata must not
    be misreported as absence of source identity.
    """
    repo = Path(repo).resolve()
    if (repo / ".git").exists():
        if _git(repo, "status", "--porcelain"):
            raise RuntimeError("relay source checkout is dirty")
        return {
            "mode": "git_commit_and_script_sha256",
            "repo": str(repo),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
        }
    return {
        "mode": "contract_script_sha256",
        "repo": str(repo),
        "git_commit": None,
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def relay_state_decision(value: dict[str, Any]) -> str:
    """Return WAIT, COMPLETE or BLOCK without consulting metric payloads."""
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
        or value.get("password_persisted") is not False
        or value.get("performance_values_available_to_scheduler") is not False
        or value.get("paired_metric_control") is not False
        or value.get("source_checkpoint_mutation") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("relay contract violates the frozen paper boundary")
    lanes = value.get("lanes")
    if not isinstance(lanes, list) or not lanes or len(lanes) != len(set(lanes)):
        raise RuntimeError("relay contract lane set is invalid")
    password_env = str(value.get("password_env", ""))
    if not password_env.startswith("FINAL_UNSB_"):
        raise RuntimeError("relay password variable is not namespaced")


def render_relay_command(python: Path, relay: dict[str, Any]) -> list[str]:
    command = [
        str(python.resolve()),
        str(Path(relay["control_script"]).resolve()),
        "--destination-root", str(Path(relay["destination_root"]).resolve()),
        "--relay-id", str(relay["relay_id"]),
        "--source-host-label", str(relay["source_host_label"]),
        "--source-host", str(relay["source_host"]),
        "--source-port", str(int(relay["source_port"])),
        "--source-user", str(relay["source_user"]),
        "--expected-host-key-sha256", str(relay["expected_host_key_sha256"]),
        "--remote-export-root", str(relay["remote_export_root"]),
    ]
    for lane in relay["lanes"]:
        command.extend(["--lane", str(lane)])
    command.extend([
        "--password-env", str(relay["password_env"]),
        "--poll-seconds", str(int(relay["poll_seconds"])),
        "--timeout-hours", str(float(relay["timeout_hours"])),
    ])
    return command


def _parse_relay_command(command: list[str]) -> dict[str, Any] | None:
    if len(command) < 4:
        return None
    result: dict[str, Any] = {
        "python": str(Path(command[0]).resolve()),
        "script": str(Path(command[1]).resolve()),
        "lanes": [],
    }
    index = 2
    while index < len(command):
        key = command[index]
        if not key.startswith("--") or index + 1 >= len(command):
            return None
        value = command[index + 1]
        if key == "--lane":
            result["lanes"].append(value)
        elif key in result:
            return None
        else:
            result[key] = value
        index += 2
    return result


def command_matches_contract(command: list[str], python: Path, relay: dict[str, Any]) -> bool:
    observed = _parse_relay_command(command)
    expected = _parse_relay_command(render_relay_command(python, relay))
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
    return observed == expected


def _proc_command(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8") for part in raw.split(b"\0") if part]


def _matching_processes(python: Path, relay: dict[str, Any]) -> list[int]:
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            command = _proc_command(int(item.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError):
            continue
        if command_matches_contract(command, python, relay):
            found.append(int(item.name))
    return sorted(found)


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("recovery supervisor checkout is not frozen")
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError("relay runtime is absent")
    relay_contract_path = args.relay_contract.resolve()
    relay_state_path = args.relay_state.resolve()
    relay = _read_json(relay_contract_path)
    _validate_relay_contract(relay)
    script = Path(relay["control_script"]).resolve()
    if not script.is_file() or _sha256(script) != relay["control_script_sha256"]:
        raise RuntimeError("relay source differs from its frozen contract")
    child_repo = script.parents[1]
    child_identity = relay_source_identity(child_repo)
    password_env = str(relay["password_env"])
    if not os.environ.get(password_env):
        raise RuntimeError(f"missing relay password environment: {password_env}")
    source = repo / "operations" / "paper_aio_export_relay_recovery_supervisor.py"
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "control_repo": str(repo),
        "control_git_commit": commit,
        "control_source": str(source),
        "control_source_sha256": _sha256(source),
        "relay_contract": str(relay_contract_path),
        "relay_contract_sha256": _sha256(relay_contract_path),
        "relay_state": str(relay_state_path),
        "relay_id": relay["relay_id"],
        "relay_source_repo": str(child_repo),
        "relay_source_identity_mode": child_identity["mode"],
        "relay_source_git_commit": child_identity["git_commit"],
        "relay_source_sha256": relay["control_script_sha256"],
        "python": str(python),
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
    child_repo = Path(contract["relay_source_repo"])
    relay_path = Path(contract["relay_contract"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
        or _sha256(Path(contract["control_source"])) != contract["control_source_sha256"]
        or _sha256(relay_path) != contract["relay_contract_sha256"]
    ):
        raise RuntimeError("relay recovery frozen identity changed")
    child_identity = relay_source_identity(child_repo)
    if (
        child_identity["mode"] != contract["relay_source_identity_mode"]
        or child_identity["git_commit"] != contract["relay_source_git_commit"]
    ):
        raise RuntimeError("relay source identity mode changed")
    relay = _read_json(relay_path)
    _validate_relay_contract(relay)
    script = Path(relay["control_script"]).resolve()
    if _sha256(script) != contract["relay_source_sha256"]:
        raise RuntimeError("relay source hash changed")
    if render_relay_command(Path(contract["python"]), relay) != contract["command"]:
        raise RuntimeError("relay command changed")
    if not os.environ.get(contract["password_env"]):
        raise RuntimeError("relay password environment disappeared")
    return relay


def _state(contract: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "relay_id": contract["relay_id"],
        "control_git_commit": contract["control_git_commit"],
        "relay_source_git_commit": contract["relay_source_git_commit"],
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
        raise ValueError("unsafe relay recovery timing")
    if not 1 <= args.max_restarts <= 20 or args.timeout_hours < 24:
        raise ValueError("unsafe relay recovery budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "RELAY_RECOVERY_CONTRACT.json"
    state_path = output / "RELAY_RECOVERY_STATE.json"
    lock_path = output / "RELAY_RECOVERY.lock"
    log_path = output / "RELAY_CHILD.log"
    contract = _freeze(args)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("relay recovery contract changed")
    else:
        _atomic_json(contract_path, contract)
    started = time.time()
    restart_count = 0
    with lock_path.open("a+", encoding="utf-8") as lock:
        if _fcntl is not None:
            try:
                _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("relay recovery supervisor already running") from error
        while True:
            relay = _verify(contract)
            relay_state_path = Path(contract["relay_state"])
            relay_state = _read_json(relay_state_path) if relay_state_path.is_file() else {}
            decision = relay_state_decision(relay_state)
            if decision == "COMPLETE":
                result = _state(
                    contract, "COMPLETE_RELAY_TERMINAL",
                    restart_count=restart_count,
                    child_status=relay_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK":
                result = _state(
                    contract, "BLOCKED_RELAY_STATE",
                    restart_count=restart_count,
                    child_status=relay_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                result = _state(contract, "BLOCKED_RECOVERY_TIMEOUT", restart_count=restart_count)
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
                        restart_count=restart_count,
                        child_pid=matches[0], child_status=relay_state.get("status"),
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
                log.write(f"\n[{time.time():.3f}] launch relay {contract['relay_id']}\n")
                log.flush()
                child = subprocess.Popen(
                    contract["command"], cwd=contract["relay_source_repo"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
            restart_count += 1
            _atomic_json(
                state_path,
                _state(
                    contract, "RELAY_RESTARTED",
                    restart_count=restart_count, child_pid=child.pid,
                ),
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
