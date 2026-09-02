"""Metric-blind durable health watcher for long paper experiments.

The watcher is deliberately outside the scientific runner.  It never loads a
checkpoint and reads only an allow-list of liveness fields from heartbeat or
control-state JSON.  It does not kill, restart, or reschedule work.  Its job is
to turn a dead PID, stale heartbeat, blocked controller, scientific-boundary
violation, or genuine disk shortage into one durable machine-readable alert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable


CONTRACT_SCHEMA = "final-unsb-paper-health-watch-contract-v1"
STATE_SCHEMA = "final-unsb-paper-health-watch-state-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_BLOCKED_PREFIXES = ("BLOCKED", "FAIL", "FATAL")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"health state must be a JSON object: {path}")
    return payload


def parse_watch(value: str) -> dict[str, Any]:
    """Parse NAME|PID|ABSOLUTE_STATE|MAX_STALE_S|ALLOW_MISSING_S."""
    fields = str(value).split("|")
    if len(fields) != 5:
        raise ValueError("--watch requires NAME|PID|STATE|MAX_STALE|ALLOW_MISSING")
    name, pid_text, path_text, stale_text, missing_text = fields
    if not _SAFE_ID.fullmatch(name):
        raise ValueError(f"unsafe health-watch name: {name!r}")
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"health-watch state path must be absolute: {path}")
    pid = int(pid_text)
    max_stale = float(stale_text)
    allow_missing = float(missing_text)
    if pid < 0 or max_stale < 0 or allow_missing < 0:
        raise ValueError("health-watch numeric fields must be non-negative")
    if pid == 0 and max_stale == 0:
        raise ValueError("health watch must check a PID, state freshness, or both")
    return {
        "name": name,
        "pid": pid,
        "state_path": str(path),
        "max_stale_seconds": max_stale,
        "allow_missing_seconds": allow_missing,
    }


def process_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid),
        )
        if not handle:
            # Access denied means the process exists but is not queryable.  A
            # missing PID reports ERROR_INVALID_PARAMETER instead.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminal(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", ""))
    if status == "COMPLETE_E200" or status.startswith("COMPLETE_SUCCESSOR_E200"):
        return True
    epoch = payload.get("data_epoch")
    try:
        return float(epoch) >= 200.0
    except (TypeError, ValueError):
        return False


def evaluate_watch(
    spec: dict[str, Any], *, now: float, watch_started: float,
    alive: Callable[[int], bool] = process_alive,
) -> dict[str, Any]:
    path = Path(spec["state_path"])
    pid = int(spec["pid"])
    pid_ok = alive(pid)
    result: dict[str, Any] = {
        "name": spec["name"],
        "pid": pid,
        "pid_alive": pid_ok,
        "state_path": str(path),
        "state_exists": path.is_file(),
    }
    if not path.is_file():
        initial_age = max(0.0, now - watch_started)
        result["watch_age_seconds"] = initial_age
        result["health"] = (
            "WAITING_FOR_INITIAL_STATE"
            if initial_age <= float(spec["allow_missing_seconds"]) and pid_ok
            else "ALERT_STATE_MISSING" if pid_ok else "ALERT_PID_DEAD_AND_STATE_MISSING"
        )
        return result

    payload = read_json(path)
    status = payload.get("status")
    age = max(0.0, now - path.stat().st_mtime)
    result.update({
        "state_age_seconds": age,
        "upstream_status": status,
        "data_epoch": payload.get("data_epoch"),
        "updates": payload.get("updates"),
        "paired_metric_control": payload.get("paired_metric_control"),
        "paired_controller_access": payload.get("paired_controller_access"),
        "confirmation20_opened": payload.get("confirmation20_opened"),
    })
    boundary_violation = any(
        payload.get(key) is True for key in (
            "paired_metric_control", "paired_controller_access", "confirmation20_opened",
        )
    )
    if boundary_violation:
        result["health"] = "ALERT_SCIENTIFIC_BOUNDARY"
    elif status and str(status).startswith(_BLOCKED_PREFIXES):
        result["health"] = "ALERT_UPSTREAM_BLOCKED"
    elif _terminal(payload):
        result["health"] = "TERMINAL"
    elif not pid_ok:
        result["health"] = "ALERT_PID_DEAD"
    elif float(spec["max_stale_seconds"]) and age > float(spec["max_stale_seconds"]):
        result["health"] = "ALERT_STATE_STALE"
    else:
        result["health"] = "HEALTHY"
    return result


def evaluate_contract(
    contract: dict[str, Any], *, now: float, watch_started: float,
    alive: Callable[[int], bool] = process_alive,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    rows = [
        evaluate_watch(spec, now=now, watch_started=watch_started, alive=alive)
        for spec in contract["watches"]
    ]
    disk = disk_usage(contract["disk_path"])
    free_gib = float(disk.free) / (1024 ** 3)
    required_gib = (
        float(contract["estimated_remaining_write_gib"])
        + float(contract["minimum_headroom_gib"])
    )
    disk_health = "HEALTHY" if free_gib >= required_gib else "ALERT_REAL_CAPACITY_RISK"
    alerts = [row for row in rows if str(row["health"]).startswith("ALERT")]
    if disk_health.startswith("ALERT"):
        alerts.append({"name": "disk", "health": disk_health})
    all_terminal = bool(rows) and all(row["health"] == "TERMINAL" for row in rows)
    return {
        "schema": STATE_SCHEMA,
        "status": "ALERT" if alerts else "ALL_TERMINAL" if all_terminal else "HEALTHY",
        "host_label": contract["host_label"],
        "captured_unix_time": now,
        "watch_started_unix_time": watch_started,
        "watches": rows,
        "disk": {
            "path": contract["disk_path"],
            "free_gib": free_gib,
            "estimated_remaining_write_gib": contract["estimated_remaining_write_gib"],
            "minimum_headroom_gib": contract["minimum_headroom_gib"],
            "required_gib": required_gib,
            "health": disk_health,
            "user_capacity_override": contract["user_capacity_override"],
        },
        "alert_count": len(alerts),
        "checkpoint_loaded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--watch", action="append", required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--estimated-remaining-write-gib", type=float, required=True)
    parser.add_argument("--minimum-headroom-gib", type=float, default=4.0)
    parser.add_argument("--user-capacity-override", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=480.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def proposed_contract(args: argparse.Namespace) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(str(args.host_label)):
        raise ValueError("health-watch host label must be a safe identifier")
    if not args.disk_path.is_absolute():
        raise ValueError("health-watch disk path must be absolute")
    if int(args.poll_seconds) < 30 or int(args.poll_seconds) > 600:
        raise ValueError("health-watch poll interval must be in [30,600]")
    if float(args.timeout_hours) < 24:
        raise ValueError("health-watch timeout must be at least 24 hours")
    if float(args.estimated_remaining_write_gib) < 0 or float(args.minimum_headroom_gib) < 0:
        raise ValueError("health-watch capacity estimates must be non-negative")
    watches = [parse_watch(value) for value in args.watch]
    names = [row["name"] for row in watches]
    if len(names) != len(set(names)):
        raise ValueError("health-watch names must be unique")
    script = Path(__file__).resolve()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "host_label": str(args.host_label),
        "control_script": str(script),
        "control_script_sha256": file_sha256(script),
        "output": str(args.output.resolve()),
        "watches": watches,
        "disk_path": str(args.disk_path.resolve()),
        "estimated_remaining_write_gib": float(args.estimated_remaining_write_gib),
        "minimum_headroom_gib": float(args.minimum_headroom_gib),
        "user_capacity_override": bool(args.user_capacity_override),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "checkpoint_loaded": False,
        "performance_values_available": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def freeze_contract(output: Path, proposed: dict[str, Any]) -> Path:
    path = Path(output).resolve() / "HEALTH_WATCH_CONTRACT.json"
    if path.is_file():
        if read_json(path) != proposed:
            raise RuntimeError("paper health-watch contract changed")
        return path
    atomic_json(path, proposed)
    return path


def acquire_lock(output: Path) -> Path:
    path = Path(output).resolve() / "HEALTH_WATCH.lock"
    if path.is_file():
        try:
            current = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            current = 0
        if current > 0 and process_alive(current):
            raise RuntimeError(f"paper health watcher already running: pid={current}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    return path


def main() -> int:
    args = arguments()
    output = args.output.resolve()
    contract = proposed_contract(args)
    freeze_contract(output, contract)
    if args.once:
        state = evaluate_contract(contract, now=time.time(), watch_started=time.time())
        atomic_json(output / "HEALTH_WATCH_STATE.json", state)
        return 2 if state["status"] == "ALERT" else 0

    lock = acquire_lock(output)
    started = time.time()
    deadline = started + float(contract["timeout_hours"]) * 3600.0
    try:
        while True:
            if file_sha256(Path(contract["control_script"])) != contract["control_script_sha256"]:
                raise RuntimeError("paper health-watch control script changed")
            state = evaluate_contract(contract, now=time.time(), watch_started=started)
            state["watcher_pid"] = os.getpid()
            atomic_json(output / "HEALTH_WATCH_STATE.json", state)
            if state["status"] == "ALL_TERMINAL":
                return 0
            if time.time() >= deadline:
                state["status"] = "TIMED_OUT"
                atomic_json(output / "HEALTH_WATCH_STATE.json", state)
                return 3
            time.sleep(int(contract["poll_seconds"]))
    except Exception as error:
        atomic_json(output / "HEALTH_WATCH_FATAL.json", {
            "schema": STATE_SCHEMA,
            "status": "FATAL",
            "host_label": contract["host_label"],
            "error_type": type(error).__name__,
            "error": str(error),
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        return 4
    finally:
        if lock.is_file() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
