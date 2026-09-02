"""Durable same-host supervisor for an authorized paper lane.

The child runner owns exact scientific state. This supervisor only restarts the
same committed command after an infrastructure interruption; it cannot change
the lane, seed, protocol, checkpoint, metrics, or stopping epoch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--restart-delay-seconds", type=int, default=30)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    repo = args.repo.resolve()
    output = args.output.resolve()
    authorization = output / "gates" / f"LANE_AUTHORIZATION_{args.lane}.json"
    if not authorization.is_file():
        raise SystemExit(f"authorized paper lane required: {authorization}")
    log = output / "logs" / f"SUPERVISOR_{args.lane}.log"
    heartbeat = output / "gates" / f"SUPERVISOR_{args.lane}.json"
    command = [
        sys.executable, "-m", "research.paper_aio.run",
        "--stage", "train", "--lane", args.lane, "--resume",
        "--output", str(output), "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--gpu", str(args.gpu),
    ]
    failures = 0
    while True:
        started = time.time()
        atomic_json(heartbeat, {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "CHILD_RUNNING",
            "lane_id": args.lane,
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
        state_path = output / "lanes" / args.lane / "RUN_STATE.json"
        state = {}
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        if completed.returncode == 0 and state.get("status") == "COMPLETE_E200":
            atomic_json(heartbeat, {
                "schema": "final-unsb-paper-supervisor-v1",
                "status": "COMPLETE_E200",
                "lane_id": args.lane,
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
                "lane_id": args.lane,
                "child_returncode": completed.returncode,
                "consecutive_failures": failures,
                "last_run_state": state,
                "confirmation20_opened": False,
            })
            return completed.returncode or 2
        atomic_json(heartbeat, {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "WAITING_TO_EXACT_RESUME",
            "lane_id": args.lane,
            "child_returncode": completed.returncode,
            "consecutive_failures": failures,
            "restart_delay_seconds": int(args.restart_delay_seconds),
            "confirmation20_opened": False,
        })
        time.sleep(max(1, int(args.restart_delay_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
