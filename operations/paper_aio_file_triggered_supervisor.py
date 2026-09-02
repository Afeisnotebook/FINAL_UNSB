"""Start one authorized paper lane after a metric-blind file trigger.

This is an operational scheduling bridge.  It observes only the existence of
an explicitly named completion artifact, never reads its scientific payload,
and then hands an already-authorized lane to ``paper_aio_supervisor.py``.
The training worktree, protocol, and lane are pinned in the command line.
"""

from __future__ import annotations

import argparse
import json
import os
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


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def trigger_decision(*, trigger_exists: bool, timed_out: bool) -> str:
    if trigger_exists:
        return "START"
    if timed_out:
        return "TIMEOUT"
    return "WAIT"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--trigger-description", required=True)
    parser.add_argument("--required-training-git-commit", required=True)
    parser.add_argument("--required-protocol-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=480.0)
    return parser.parse_args()


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    repo = args.repo.resolve()
    output = args.output.resolve()
    trigger = args.trigger.resolve()
    if int(args.poll_seconds) < 10:
        raise RuntimeError("trigger polling interval must be at least 10 seconds")
    if float(args.timeout_hours) < 12:
        raise RuntimeError("trigger timeout must be at least 12 hours")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if head != args.required_training_git_commit:
        raise RuntimeError(
            f"training worktree moved: expected {args.required_training_git_commit}, "
            f"observed {head}"
        )
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip():
        raise RuntimeError(f"training worktree is dirty: {repo}")
    protocol = read_json(output / "PAPER_PROTOCOL.json")
    observed = protocol.get("protocol_fingerprint")
    if observed != args.required_protocol_fingerprint:
        raise RuntimeError(
            f"paper protocol moved: expected {args.required_protocol_fingerprint}, "
            f"observed {observed}"
        )
    authorization = read_json(output / "gates" / f"LANE_AUTHORIZATION_{args.lane}.json")
    if (
        authorization.get("status") != "PASS"
        or authorization.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
    ):
        raise RuntimeError(f"lane authorization is not pinned PASS: {args.lane}")
    return repo, output, trigger


def main() -> int:
    args = arguments()
    repo, output, trigger = validate_frozen_inputs(args)
    operations = output / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    label = f"FILE_TRIGGERED_SUPERVISOR_{args.lane}"
    state_path = operations / f"{label}_STATE.json"
    lock_path = operations / f"{label}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"file-triggered supervisor lock already exists: {lock_path}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.close(descriptor)
    started = time.time()
    timeout_seconds = float(args.timeout_hours) * 3600.0
    while True:
        decision = trigger_decision(
            trigger_exists=trigger.is_file(),
            timed_out=(time.time() - started) >= timeout_seconds,
        )
        atomic_json(state_path, {
            "schema": "final-unsb-paper-file-triggered-supervisor-v1",
            "status": {
                "WAIT": "WAITING_FOR_METRIC_BLIND_TRIGGER",
                "START": "TRIGGER_OBSERVED_STARTING_SUPERVISOR",
                "TIMEOUT": "BLOCKED_TRIGGER_TIMEOUT",
            }[decision],
            "pid": os.getpid(),
            "lane_id": args.lane,
            "trigger": str(trigger),
            "trigger_description": args.trigger_description,
            "training_git_commit": args.required_training_git_commit,
            "protocol_fingerprint": args.required_protocol_fingerprint,
            "trigger_payload_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        if decision == "START":
            break
        if decision == "TIMEOUT":
            return 3
        time.sleep(int(args.poll_seconds))
    # Revalidate after the wait so a moved worktree or receipt fails closed.
    validate_frozen_inputs(args)
    command = [
        sys.executable, str(repo / "operations" / "paper_aio_supervisor.py"),
        "--repo", str(repo), "--output", str(output),
        "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--lane", args.lane, "--gpu", str(args.gpu),
    ]
    completed = subprocess.run(command, cwd=repo, check=False)
    supervisor_path = output / "gates" / f"SUPERVISOR_{args.lane}.json"
    supervisor = read_json(supervisor_path) if supervisor_path.is_file() else {}
    ok = completed.returncode == 0 and supervisor.get("status") == "COMPLETE_E200"
    atomic_json(state_path, {
        "schema": "final-unsb-paper-file-triggered-supervisor-v1",
        "status": "COMPLETE_E200" if ok else "BLOCKED_SUPERVISOR_EXIT",
        "pid": os.getpid(),
        "lane_id": args.lane,
        "trigger": str(trigger),
        "trigger_description": args.trigger_description,
        "training_git_commit": args.required_training_git_commit,
        "protocol_fingerprint": args.required_protocol_fingerprint,
        "child_returncode": completed.returncode,
        "supervisor_status": supervisor.get("status"),
        "trigger_payload_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    return 0 if ok else (completed.returncode or 4)


if __name__ == "__main__":
    raise SystemExit(main())
