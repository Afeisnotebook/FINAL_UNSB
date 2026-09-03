"""Metric-blind live-process progress sentinel for a supervised paper lane.

The ordinary health watcher proves that a PID and heartbeat file exist.  This
sentinel covers the narrower failure mode where the trainer remains alive but
stops consuming input or completing epochs.  It is diagnostic only: it never
signals a process, changes a checkpoint, or launches a replacement child.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-paper-live-progress-watch-v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def supervisor_children(supervisor_pid: int, proc_root: Path = Path("/proc")) -> list[int]:
    path = proc_root / str(int(supervisor_pid)) / "task" / str(int(supervisor_pid)) / "children"
    if not path.is_file():
        return []
    return [int(value) for value in path.read_text(encoding="utf-8").split()]


def parse_process_stat(raw: str) -> dict[str, int | str]:
    """Parse the fields needed from Linux ``/proc/PID/stat`` safely."""
    right = raw.rfind(")")
    if right < 0:
        raise ValueError("malformed process stat")
    pid = int(raw[: raw.find(" ")])
    fields = raw[right + 2 :].split()
    if len(fields) < 20:
        raise ValueError("incomplete process stat")
    return {
        "pid": pid,
        "state": fields[0],
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
    }


def process_snapshot(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    root = proc_root / str(int(pid))
    stat = parse_process_stat((root / "stat").read_text(encoding="utf-8"))
    io_values: dict[str, int] = {}
    for line in (root / "io").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        io_values[key] = int(raw.strip())
    return {**stat, "io": io_values}


def process_start_unix(
    start_ticks: int, *, now: float, uptime_seconds: float, clock_ticks: int,
) -> float:
    boot_unix = float(now) - float(uptime_seconds)
    return boot_unix + float(start_ticks) / float(clock_ticks)


def effective_progress_age(
    *, now: float, heartbeat_mtime: float, child_started_unix: float,
) -> float:
    # A supervisor exact-resume creates a new child before the old epoch-level
    # heartbeat changes.  Child age therefore prevents a false stale alarm.
    return max(
        0.0,
        min(float(now) - float(heartbeat_mtime), float(now) - float(child_started_unix)),
    )


def classify_probe(
    *, effective_age_seconds: float, stall_seconds: float,
    before: dict[str, Any], after: dict[str, Any],
) -> tuple[str, bool]:
    if float(effective_age_seconds) < float(stall_seconds):
        return "HEALTHY_WITHIN_EPOCH_BOUND", False
    before_io = before.get("io") or {}
    after_io = after.get("io") or {}
    io_keys = ("rchar", "wchar", "read_bytes", "write_bytes", "syscr", "syscw")
    if any(int(after_io.get(key, 0)) > int(before_io.get(key, 0)) for key in io_keys):
        return "STALE_EPOCH_HEARTBEAT_BUT_PROCESS_IO_PROGRESSING", False
    if int(after.get("cpu_ticks", 0)) > int(before.get("cpu_ticks", 0)):
        return "ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS", True
    return "ALERT_LIVE_PROCESS_NO_PROGRESS", True


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--checkpoint-sidecar", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=int, default=7200)
    parser.add_argument("--sample-seconds", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.stall_seconds < 600:
        raise SystemExit("stall threshold must be at least 600 seconds")
    if args.sample_seconds < 5 or args.sample_seconds > 120:
        raise SystemExit("sample interval must be in [5,120] seconds")
    if args.poll_seconds < 30 or args.poll_seconds > 600:
        raise SystemExit("poll interval must be in [30,600] seconds")
    output = args.output.resolve()
    state_path = output / "PROGRESS_WATCH_STATE.json"
    contract_path = output / "PROGRESS_WATCH_CONTRACT.json"
    contract = {
        "schema": SCHEMA,
        "status": "FROZEN_DIAGNOSTIC_ONLY",
        "pid": os.getpid(),
        "host_label": args.host_label,
        "supervisor_pid": args.supervisor_pid,
        "heartbeat": str(args.heartbeat.resolve()),
        "checkpoint_sidecar": str(args.checkpoint_sidecar.resolve()),
        "stall_seconds": args.stall_seconds,
        "sample_seconds": args.sample_seconds,
        "poll_seconds": args.poll_seconds,
        "signals_processes": False,
        "launches_training": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    if contract_path.is_file() and read_json(contract_path) != contract:
        raise SystemExit("progress-watch contract changed")
    if not contract_path.is_file():
        atomic_json(contract_path, contract)
    started = time.time()
    alert_count = 0
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    while time.time() - started <= args.timeout_hours * 3600:
        now = time.time()
        base = {
            **contract,
            "observed_at": now,
            "alert_count": alert_count,
        }
        children = supervisor_children(args.supervisor_pid)
        if len(children) != 1 or not args.heartbeat.is_file():
            atomic_json(state_path, {
                **base,
                "status": "WAITING_FOR_EXACTLY_ONE_SUPERVISOR_CHILD_AND_HEARTBEAT",
                "children": children,
                "alert": False,
            })
            time.sleep(args.poll_seconds)
            continue
        child = children[0]
        try:
            before = process_snapshot(child)
            uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
            child_started = process_start_unix(
                int(before["start_ticks"]), now=now, uptime_seconds=uptime,
                clock_ticks=clock_ticks,
            )
            age = effective_progress_age(
                now=now, heartbeat_mtime=args.heartbeat.stat().st_mtime,
                child_started_unix=child_started,
            )
            time.sleep(args.sample_seconds)
            after = process_snapshot(child)
            status, alert = classify_probe(
                effective_age_seconds=age, stall_seconds=args.stall_seconds,
                before=before, after=after,
            )
            if alert:
                alert_count += 1
            sidecar = (
                read_json(args.checkpoint_sidecar)
                if args.checkpoint_sidecar.is_file() else {}
            )
            atomic_json(state_path, {
                **base,
                "status": status,
                "alert": alert,
                "alert_count": alert_count,
                "trainer_pid": child,
                "trainer_state": after["state"],
                "child_started_unix": child_started,
                "heartbeat_age_seconds": now - args.heartbeat.stat().st_mtime,
                "effective_progress_age_seconds": age,
                "checkpoint_step": sidecar.get("step"),
                "io_delta": {
                    key: int((after.get("io") or {}).get(key, 0))
                    - int((before.get("io") or {}).get(key, 0))
                    for key in ("rchar", "wchar", "read_bytes", "write_bytes", "syscr", "syscw")
                },
                "cpu_ticks_delta": int(after["cpu_ticks"]) - int(before["cpu_ticks"]),
            })
        except (FileNotFoundError, ProcessLookupError):
            atomic_json(state_path, {
                **base,
                "status": "CHILD_CHANGED_DURING_DIAGNOSTIC_SAMPLE",
                "trainer_pid": child,
                "alert": False,
            })
        time.sleep(args.poll_seconds)
    atomic_json(state_path, {
        **contract,
        "status": "TIMEOUT",
        "alert": True,
        "alert_count": alert_count + 1,
        "observed_at": time.time(),
    })
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
