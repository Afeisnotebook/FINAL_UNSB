"""Run one durable child while holding the shared paper-evaluation GPU lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-paper-exclusive-gpu-runner-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def acquire_lock(handle) -> None:
    handle.seek(0)
    if handle.read(1) == "":
        handle.write("0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def run(args: argparse.Namespace) -> int:
    if not args.command:
        raise RuntimeError("child command is empty")
    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command:
        raise RuntimeError("child command is empty")
    lock_path = args.gpu_lock.resolve()
    state_path = args.state.resolve()
    log_path = args.log.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_hash = file_sha256(Path(__file__))
    with lock_path.open("a+", encoding="utf-8") as lock:
        acquire_lock(lock)
        with log_path.open("ab") as log:
            child = subprocess.Popen(
                command, cwd=args.cwd.resolve(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
            base = {
                "schema": SCHEMA,
                "pid": os.getpid(),
                "child_pid": child.pid,
                "gpu_lock": str(lock_path),
                "cwd": str(args.cwd.resolve()),
                "command": command,
                "control_source_sha256": source_hash,
                "performance_values_read": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            }
            atomic_json(state_path, {**base, "status": "CHILD_RUNNING"})
            returncode = child.wait()
            atomic_json(state_path, {
                **base,
                "status": "COMPLETE" if returncode == 0 else "CHILD_FAILED",
                "child_returncode": returncode,
                "wall_seconds": time.time() - started,
            })
            return returncode


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--state", type=Path, required=True)
    value.add_argument("--log", type=Path, required=True)
    value.add_argument("--cwd", type=Path, required=True)
    value.add_argument("command", nargs=argparse.REMAINDER)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
