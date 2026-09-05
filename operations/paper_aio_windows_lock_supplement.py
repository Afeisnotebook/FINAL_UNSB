"""Hold the legacy append-mode lock byte while an existing task is alive.

This narrow compatibility guard is needed only for an exclusive runner that
was started before the two-byte Windows lock was introduced.  It reads no
checkpoint or metric and exits automatically with the protected process.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from operations.paper_aio_health_watch import process_alive


SCHEMA = "final-unsb-paper-windows-lock-supplement-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> int:
    if os.name != "nt":
        raise RuntimeError("supplementary byte lock is Windows-only")
    import msvcrt

    lock_path = args.gpu_lock.resolve()
    protected_state = args.protected_state.resolve()
    state_path = args.state.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "protected_pid": args.protected_pid,
        "gpu_lock": str(lock_path),
        "protected_state": str(protected_state),
        "locked_byte_offset": 1,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    with lock_path.open("a+", encoding="utf-8") as handle:
        handle.seek(1)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        atomic_json(state_path, {**base, "status": "LOCK_HELD"})
        while process_alive(args.protected_pid):
            if protected_state.is_file():
                payload = json.loads(protected_state.read_text(encoding="utf-8"))
                if any(payload.get(key) is True for key in (
                    "performance_values_read", "paired_metric_control",
                    "confirmation20_opened",
                )):
                    raise RuntimeError("protected task crossed a scientific boundary")
                if payload.get("status") != "CHILD_RUNNING":
                    break
            time.sleep(args.poll_seconds)
        atomic_json(state_path, {**base, "status": "PROTECTED_TASK_TERMINAL_LOCK_RELEASED"})
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--protected-pid", type=int, required=True)
    value.add_argument("--protected-state", type=Path, required=True)
    value.add_argument("--state", type=Path, required=True)
    value.add_argument("--poll-seconds", type=float, default=30.0)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
