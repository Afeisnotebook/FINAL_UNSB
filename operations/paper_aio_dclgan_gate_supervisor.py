"""Durably execute the metric-blind DCLGAN 1000-update GPU gate.

The supervisor uses a frozen clean checkout and retries only engineering
interruptions.  Its stages and stopping points are fixed before execution;
no performance value is parsed or used for control.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _repo_identity(repo: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo, text=True,
    ).strip()
    return commit, dirty


def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if handle.read(1) == b"":
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--required-git-commit", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=int, default=30)
    return parser.parse_args()


def _stage_commands(args: argparse.Namespace, python: Path) -> list[tuple[str, list[str]]]:
    adapter = args.repo / "operations" / "paper_aio_dclgan_adapter.py"
    common = [
        str(python), str(adapter),
        "--upstream-root", str(args.upstream_root),
        "--manifest", str(args.manifest),
        "--train-view", str(args.train_view),
        "--output", str(args.output),
    ]
    checkpoint = args.output / "lanes" / "dclgan" / "full_state_latest.pt"
    train = common + [
        "--stage", "train", "--gpu", str(args.gpu),
        "--stop-after-updates", "1000", "--resume",
    ]
    return [
        ("preflight", common + ["--stage", "preflight"]),
        (
            "confirmation_lock",
            common + ["--stage", "confirmation-lock-gate"],
        ),
        (
            "exact_resume_1000_500",
            common + [
                "--stage", "exact-resume-gate", "--gpu", str(args.gpu),
                "--gate-total-updates", "1000",
                "--gate-split-updates", "500",
            ],
        ),
        ("capacity_train_1000", train),
        (
            "evaluation_repeat",
            common + [
                "--stage", "evaluation-repeat-gate",
                "--data-root", str(args.data_root),
                "--checkpoint", str(checkpoint), "--gpu", str(args.gpu),
            ],
        ),
        ("authorize", common + ["--stage", "authorize"]),
    ]


def main() -> int:
    args = _arguments()
    args.repo = args.repo.resolve()
    args.upstream_root = args.upstream_root.resolve()
    args.manifest = args.manifest.resolve()
    args.train_view = args.train_view.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.max_attempts < 1 or args.retry_seconds < 10:
        raise RuntimeError("DCLGAN supervisor retry policy is unsafe")
    commit, dirty = _repo_identity(args.repo)
    if commit != args.required_git_commit or dirty:
        raise RuntimeError("DCLGAN supervisor checkout is not the frozen clean commit")
    state_path = args.output / "operations" / "DCLGAN_GATE_SUPERVISOR.json"
    lock = _lock(args.output / "operations" / "DCLGAN_GATE_SUPERVISOR.lock")
    started = time.time()
    try:
        for stage, command in _stage_commands(args, Path(sys.executable)):
            for attempt in range(1, args.max_attempts + 1):
                log_path = args.output / "logs" / f"{stage}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(
                        command, cwd=args.repo, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                    _write(state_path, {
                        "schema": "final-unsb-paper-dclgan-gate-supervisor-v1",
                        "status": "CHILD_RUNNING",
                        "pid": os.getpid(), "child_pid": child.pid,
                        "stage": stage, "attempt": attempt,
                        "required_git_commit": args.required_git_commit,
                        "output": str(args.output),
                        "started_unix_time": started,
                        "updated_unix_time": time.time(),
                        "performance_values_read": False,
                        "paired_metric_control": False,
                        "confirmation20_opened": False,
                    })
                    returncode = child.wait()
                if returncode == 0:
                    _write(state_path, {
                        "schema": "final-unsb-paper-dclgan-gate-supervisor-v1",
                        "status": "STAGE_COMPLETE",
                        "pid": os.getpid(), "stage": stage,
                        "attempt": attempt,
                        "required_git_commit": args.required_git_commit,
                        "updated_unix_time": time.time(),
                        "performance_values_read": False,
                        "paired_metric_control": False,
                        "confirmation20_opened": False,
                    })
                    break
                _write(state_path, {
                    "schema": "final-unsb-paper-dclgan-gate-supervisor-v1",
                    "status": "RETRY_WAIT" if attempt < args.max_attempts else "FAILED",
                    "pid": os.getpid(), "stage": stage, "attempt": attempt,
                    "child_returncode": returncode,
                    "required_git_commit": args.required_git_commit,
                    "updated_unix_time": time.time(),
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                })
                if attempt == args.max_attempts:
                    return returncode or 2
                time.sleep(args.retry_seconds)
        _write(state_path, {
            "schema": "final-unsb-paper-dclgan-gate-supervisor-v1",
            "status": "COMPLETE_HOST_BOUND_GPU_GATE_AUTHORIZED",
            "pid": os.getpid(),
            "required_git_commit": args.required_git_commit,
            "output": str(args.output),
            "wall_seconds": time.time() - started,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
