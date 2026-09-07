"""Durably adopt and recover one frozen source-bound export successor.

Training supervisors protect scientific progress, while an export successor
turns the completed fixed checkpoints into hash-bound receipts.  This control
process protects only that receipt handoff.  It adopts an existing exporter,
rehashes both its own control source and the exporter's frozen source/contract,
and restarts the exporter only while its metric-blind state is non-terminal.
It never loads a checkpoint, reads performance values, or changes training.
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

try:  # pragma: no cover - deployed on Linux.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-export-successor-recovery-contract-v1"
STATE_SCHEMA = "final-unsb-paper-export-successor-recovery-state-v1"
EXPORT_CONTRACT_SCHEMA = "final-unsb-paper-export-successor-contract-v1"
EXPORT_STATE_SCHEMA = "final-unsb-paper-export-successor-state-v1"
COMPLETE_STATUS = "COMPLETE_SOURCE_BOUND_EXPORT_SET"
SOURCE_RELATIVES = (
    "operations/paper_aio_export_successor.py",
    "research/paper_aio/unified.py",
    "research/paper_aio/protocol.py",
)


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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def export_state_decision(value: dict[str, Any]) -> str:
    """Return WAIT, COMPLETE, or BLOCK without consulting metric payloads."""
    if not value:
        return "WAIT"
    if (
        value.get("schema") != EXPORT_STATE_SCHEMA
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        return "BLOCK"
    status = str(value.get("status", ""))
    if status == COMPLETE_STATUS:
        return "COMPLETE"
    if status.startswith(("BLOCK", "FAIL", "FATAL", "ERROR")):
        return "BLOCK"
    return "WAIT"


def _validate_export_contract(value: dict[str, Any]) -> None:
    if (
        value.get("schema") != EXPORT_CONTRACT_SCHEMA
        or value.get("status") != "FROZEN_WAITING"
        or value.get("performance_values_available_to_scheduling") is not False
        or value.get("paired_metric_control") is not False
        or value.get("checkpoint_copy_performed") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("export successor contract violates the frozen boundary")
    epochs = value.get("epochs")
    if epochs != [100, 125, 150, 175, 200]:
        raise RuntimeError("export successor fixed epoch set changed")
    lane = str(value.get("lane_id", ""))
    if not lane or any(character in lane for character in ("/", "\\", "\n")):
        raise RuntimeError("export successor lane is invalid")
    sources = value.get("control_source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_RELATIVES):
        raise RuntimeError("export successor source hash set is invalid")


def render_export_command(python: Path, contract: dict[str, Any]) -> list[str]:
    repo = Path(contract["control_repo"]).resolve()
    return [
        str(python.resolve()),
        str((repo / "operations" / "paper_aio_export_successor.py").resolve()),
        "--repo", str(repo),
        "--source-output", str(Path(contract["source_output"]).resolve()),
        "--destination", str(Path(contract["destination"]).resolve()),
        "--lane", str(contract["lane_id"]),
        "--source-host-label", str(contract["source_host_label"]),
        "--required-training-git-commit", str(contract["required_training_git_commit"]),
        "--required-training-protocol-fingerprint",
        str(contract["required_training_protocol_fingerprint"]),
        "--poll-seconds", str(int(contract["poll_seconds"])),
        "--timeout-hours", str(float(contract["timeout_hours"])),
    ]


def _parse_command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any] | None:
    if len(command) < 4:
        return None
    python = Path(command[0])
    script = Path(command[1])
    if not script.is_absolute():
        if cwd is None:
            return None
        script = cwd / script
    result: dict[str, Any] = {
        "python": str(python.resolve()),
        "script": str(script.resolve()),
    }
    index = 2
    while index < len(command):
        key = command[index]
        if not key.startswith("--") or index + 1 >= len(command) or key in result:
            return None
        result[key] = command[index + 1]
        index += 2
    for key in ("--repo", "--source-output", "--destination"):
        if key not in result:
            return None
        result[key] = str(Path(result[key]).resolve())
    return result


def command_matches_contract(
    command: list[str], *, cwd: Path, python: Path, contract: dict[str, Any]
) -> bool:
    observed = _parse_command(command, cwd=cwd)
    expected = _parse_command(render_export_command(python, contract), cwd=None)
    if observed is None or expected is None:
        return False
    for key in ("--poll-seconds",):
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


def _proc_command_and_cwd(pid: int) -> tuple[list[str], Path]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    command = [part.decode("utf-8") for part in raw.split(b"\0") if part]
    cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    return command, cwd


def _matching_processes(python: Path, contract: dict[str, Any]) -> list[int]:
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            command, cwd = _proc_command_and_cwd(int(item.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError):
            continue
        if command_matches_contract(command, cwd=cwd, python=python, contract=contract):
            found.append(int(item.name))
    return sorted(found)


def _verify_export_source(contract: dict[str, Any]) -> None:
    repo = Path(contract["export_control_repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["export_control_git_commit"]
        or _git(repo, "status", "--porcelain")
    ):
        raise RuntimeError("export successor control checkout changed")
    for relative, expected in contract["export_source_sha256"].items():
        if _sha256(repo / relative) != expected:
            raise RuntimeError(f"export successor source changed: {relative}")


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("recovery supervisor checkout is not frozen")
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError("export successor runtime is absent")
    export_contract_path = args.export_contract.resolve()
    export_state_path = args.export_state.resolve()
    export = _read_json(export_contract_path)
    _validate_export_contract(export)
    export_repo = Path(export["control_repo"]).resolve()
    source = repo / "operations" / "paper_aio_export_successor_recovery_supervisor.py"
    value = {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "control_repo": str(repo),
        "control_git_commit": commit,
        "control_source": str(source),
        "control_source_sha256": _sha256(source),
        "export_contract": str(export_contract_path),
        "export_contract_sha256": _sha256(export_contract_path),
        "export_state": str(export_state_path),
        "lane_id": export["lane_id"],
        "export_control_repo": str(export_repo),
        "export_control_git_commit": export["control_git_commit"],
        "export_source_sha256": export["control_source_sha256"],
        "python": str(python),
        "command": render_export_command(python, export),
        "poll_seconds": int(args.poll_seconds),
        "restart_delay_seconds": int(args.restart_delay_seconds),
        "max_restarts": int(args.max_restarts),
        "timeout_hours": float(args.timeout_hours),
        "checkpoint_loaded": False,
        "performance_values_available_to_supervisor": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _verify_export_source(value)
    return value


def _verify(contract: dict[str, Any]) -> dict[str, Any]:
    repo = Path(contract["control_repo"])
    path = Path(contract["export_contract"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
        or _sha256(Path(contract["control_source"])) != contract["control_source_sha256"]
        or _sha256(path) != contract["export_contract_sha256"]
    ):
        raise RuntimeError("export recovery frozen identity changed")
    _verify_export_source(contract)
    export = _read_json(path)
    _validate_export_contract(export)
    if render_export_command(Path(contract["python"]), export) != contract["command"]:
        raise RuntimeError("export successor command changed")
    return export


def _state(contract: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "lane_id": contract["lane_id"],
        "control_git_commit": contract["control_git_commit"],
        "export_control_git_commit": contract["export_control_git_commit"],
        "export_state": contract["export_state"],
        "restart_count": int(extra.pop("restart_count", 0)),
        "checkpoint_loaded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 5 <= args.poll_seconds <= 600 or not 1 <= args.restart_delay_seconds <= 600:
        raise ValueError("unsafe export recovery timing")
    if not 1 <= args.max_restarts <= 20 or args.timeout_hours < 24:
        raise ValueError("unsafe export recovery budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "EXPORT_RECOVERY_CONTRACT.json"
    state_path = output / "EXPORT_RECOVERY_STATE.json"
    lock_path = output / "EXPORT_RECOVERY.lock"
    log_path = output / "EXPORT_CHILD.log"
    contract = _freeze(args)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("export recovery contract changed")
    else:
        _atomic_json(contract_path, contract)
    started = time.time()
    restart_count = 0
    with lock_path.open("a+", encoding="utf-8") as lock:
        if _fcntl is not None:
            try:
                _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("export recovery supervisor already running") from error
        while True:
            export = _verify(contract)
            export_state_path = Path(contract["export_state"])
            export_state = _read_json(export_state_path) if export_state_path.is_file() else {}
            decision = export_state_decision(export_state)
            if decision == "COMPLETE":
                result = _state(
                    contract, "COMPLETE_EXPORT_TERMINAL", restart_count=restart_count,
                    child_status=export_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK":
                result = _state(
                    contract, "BLOCKED_EXPORT_STATE", restart_count=restart_count,
                    child_status=export_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                result = _state(contract, "BLOCKED_RECOVERY_TIMEOUT", restart_count=restart_count)
                _atomic_json(state_path, result)
                return result

            matches = _matching_processes(Path(contract["python"]), export)
            advertised = int(export_state.get("pid", 0) or 0)
            if advertised > 0 and _pid_alive(advertised) and advertised not in matches:
                result = _state(
                    contract, "BLOCKED_LIVE_PID_COMMAND_MISMATCH",
                    restart_count=restart_count, child_pid=advertised,
                )
                _atomic_json(state_path, result)
                return result
            if len(matches) > 1:
                result = _state(
                    contract, "BLOCKED_DUPLICATE_EXPORT_PROCESSES",
                    restart_count=restart_count, child_pids=matches,
                )
                _atomic_json(state_path, result)
                return result
            if matches:
                _atomic_json(
                    state_path,
                    _state(
                        contract, "MONITORING_EXISTING_EXPORT",
                        restart_count=restart_count, child_pid=matches[0],
                        child_status=export_state.get("status"),
                    ),
                )
                time.sleep(contract["poll_seconds"])
                continue
            if restart_count >= contract["max_restarts"]:
                result = _state(
                    contract, "BLOCKED_RESTART_BUDGET_EXHAUSTED",
                    restart_count=restart_count, child_status=export_state.get("status"),
                )
                _atomic_json(state_path, result)
                return result
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{time.time():.3f}] launch export successor {contract['lane_id']}\n")
                log.flush()
                child = subprocess.Popen(
                    contract["command"], cwd=contract["export_control_repo"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            restart_count += 1
            _atomic_json(
                state_path,
                _state(
                    contract, "EXPORT_RESTARTED", restart_count=restart_count,
                    child_pid=child.pid,
                ),
            )
            time.sleep(contract["restart_delay_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--export-contract", type=Path, required=True)
    value.add_argument("--export-state", type=Path, required=True)
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
