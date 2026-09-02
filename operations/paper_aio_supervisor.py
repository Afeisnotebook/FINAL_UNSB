"""Durable same-host supervisor for an authorized paper lane.

The child runner owns exact scientific state. This supervisor only restarts the
same committed command after an infrastructure interruption; it cannot change
the lane, seed, protocol, checkpoint, metrics, or stopping epoch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--restart-delay-seconds", type=int, default=30)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    return parser.parse_args()


def lane_identity(lane: str, candidate_id: str | None) -> tuple[str, str]:
    if lane != "candidate":
        if candidate_id is not None:
            raise ValueError("--candidate-id is only valid with --lane candidate")
        if not _SAFE_ID.fullmatch(str(lane)):
            raise ValueError(f"unsafe paper lane: {lane!r}")
        return str(lane), "static"
    value = str(candidate_id or "")
    if not _SAFE_ID.fullmatch(value):
        raise ValueError("--lane candidate requires a safe --candidate-id")
    return value, "candidate"


def authorization_path(output: Path, lane: str, candidate_id: str | None) -> Path:
    identity, kind = lane_identity(lane, candidate_id)
    name = (
        f"CANDIDATE_AUTHORIZATION_{identity}.json" if kind == "candidate" else
        f"LANE_AUTHORIZATION_{identity}.json"
    )
    return Path(output) / "gates" / name


def child_command(args: argparse.Namespace, output: Path) -> list[str]:
    identity, kind = lane_identity(args.lane, args.candidate_id)
    command = [
        sys.executable, "-m", "research.paper_aio.run",
        "--stage", "train", "--lane", args.lane, "--resume",
        "--output", str(output), "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--gpu", str(args.gpu),
    ]
    if kind == "candidate":
        command += ["--candidate-id", identity]
    return command


def main() -> int:
    args = arguments()
    repo = args.repo.resolve()
    output = args.output.resolve()
    identity, lane_kind = lane_identity(args.lane, args.candidate_id)
    authorization = authorization_path(output, args.lane, args.candidate_id)
    if not authorization.is_file():
        raise SystemExit(f"authorized paper lane required: {authorization}")
    log = output / "logs" / f"SUPERVISOR_{identity}.log"
    heartbeat = output / "gates" / f"SUPERVISOR_{identity}.json"
    command = child_command(args, output)
    failures = 0
    while True:
        started = time.time()
        atomic_json(heartbeat, {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "CHILD_RUNNING",
            "lane_id": identity,
            "lane_kind": lane_kind,
            "command": command,
            "started_unix_time": started,
            "consecutive_failures": failures,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{time.time():.3f}] starting child\n")
            handle.flush()
            completed = subprocess.run(
                command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT,
                check=False,
            )
        state_path = output / "lanes" / identity / "RUN_STATE.json"
        state = {}
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        if completed.returncode == 0 and state.get("status") == "COMPLETE_E200":
            atomic_json(heartbeat, {
                "schema": "final-unsb-paper-supervisor-v1",
                "status": "COMPLETE_E200",
                "lane_id": identity,
                "lane_kind": lane_kind,
                "child_returncode": completed.returncode,
                "run_state": state,
                "confirmation20_opened": False,
            })
            return 0
        failures += 1
        if failures >= int(args.maximum_consecutive_failures):
            atomic_json(heartbeat, {
                "schema": "final-unsb-paper-supervisor-v1",
                "status": "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE",
                "lane_id": identity,
                "lane_kind": lane_kind,
                "child_returncode": completed.returncode,
                "consecutive_failures": failures,
                "last_run_state": state,
                "confirmation20_opened": False,
            })
            return completed.returncode or 2
        atomic_json(heartbeat, {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "WAITING_TO_EXACT_RESUME",
            "lane_id": identity,
            "lane_kind": lane_kind,
            "child_returncode": completed.returncode,
            "consecutive_failures": failures,
            "restart_delay_seconds": int(args.restart_delay_seconds),
            "confirmation20_opened": False,
        })
        time.sleep(max(1, int(args.restart_delay_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
