"""Activate one full-data candidate after its terminal small25 receipt exists.

The successor is deliberately narrow. It waits for completed artifacts, runs
the frozen cross-code runtime gate, materializes the evidence lock, issues the
separate authorization, and executes exactly one full-data epoch as a
throughput/capacity probe. It never reads intermediate metrics and never
continues beyond e1 without a separate makespan decision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_trigger_decision(*, all_exist: bool, timed_out: bool) -> str:
    if all_exist:
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
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--derivation-card", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--parent-runtime-receipt", type=Path, required=True)
    parser.add_argument("--parent-e0", type=Path, required=True)
    parser.add_argument("--parent-scientific-git-commit", required=True)
    parser.add_argument("--parent-protocol-fingerprint", required=True)
    parser.add_argument("--capacity-override-receipt", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--required-candidate-git-commit", required=True)
    parser.add_argument("--required-candidate-protocol-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=480.0)
    return parser.parse_args()


def git_identity(repo: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"candidate worktree is dirty: {repo}")
    return head


def run_logged(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.time():.3f}] {json.dumps(command)}\n")
        handle.flush()
        return subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def main() -> int:
    args = arguments()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if int(args.poll_seconds) < 10 or float(args.timeout_hours) < 12:
        raise RuntimeError("activation successor polling/timeout is unsafe")
    head = git_identity(repo)
    if head != args.required_candidate_git_commit:
        raise RuntimeError("candidate activation worktree commit mismatch")
    env_check = subprocess.run(
        [sys.executable, "-c", (
            "from research.paper_aio.protocol import protocol_fingerprint; "
            "print(protocol_fingerprint())"
        )], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if env_check != args.required_candidate_protocol_fingerprint:
        raise RuntimeError("candidate activation protocol fingerprint mismatch")

    operations = output / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    state_path = operations / "CANDIDATE_ACTIVATION_SUCCESSOR_STATE.json"
    lock_path = operations / "CANDIDATE_ACTIVATION_SUCCESSOR.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"candidate activation lock already exists: {lock_path}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.close(descriptor)

    required_files = [
        args.terminal_receipt.resolve(), args.trajectory.resolve(),
        args.derivation_card.resolve(), args.implementation.resolve(),
        args.parent_runtime_receipt.resolve(), args.parent_e0.resolve(),
        args.capacity_override_receipt.resolve(),
        args.parent_output.resolve() / "gates" / "SUPERVISOR_plain.json",
    ]
    started = time.time()
    while True:
        missing = [str(path) for path in required_files if not path.is_file()]
        decision = file_trigger_decision(
            all_exist=not missing,
            timed_out=time.time() - started >= float(args.timeout_hours) * 3600,
        )
        atomic_json(state_path, {
            "schema": "final-unsb-paper-candidate-activation-successor-v1",
            "status": {
                "WAIT": "WAITING_FOR_COMPLETE_ARTIFACTS",
                "START": "COMPLETE_ARTIFACTS_OBSERVED_STARTING_GATES",
                "TIMEOUT": "BLOCKED_ARTIFACT_TIMEOUT",
            }[decision],
            "pid": os.getpid(),
            "candidate_id": args.candidate_id,
            "candidate_git_commit": head,
            "candidate_protocol_fingerprint": env_check,
            "parent_readiness_mode": "authorized_running",
            "missing": missing,
            "artifact_payloads_read_by_scheduler": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        if decision == "START":
            break
        if decision == "TIMEOUT":
            return 3
        time.sleep(int(args.poll_seconds))

    common = [
        "--output", str(output), "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()), "--gpu", str(args.gpu),
    ]
    evidence = [
        "--candidate-id", args.candidate_id,
        "--candidate-terminal-receipt", str(args.terminal_receipt.resolve()),
        "--candidate-trajectory", str(args.trajectory.resolve()),
        "--candidate-derivation-card", str(args.derivation_card.resolve()),
        "--candidate-implementation", str(args.implementation.resolve()),
        "--parent-output", str(args.parent_output.resolve()),
        "--parent-scientific-git-commit", args.parent_scientific_git_commit,
        "--parent-protocol-fingerprint", args.parent_protocol_fingerprint,
        "--parent-readiness-mode", "authorized_running",
    ]
    runtime_gate = (
        output / "candidate_runtime_gate" / args.candidate_id
        / "CANDIDATE_RUNTIME_GATE.json"
    )
    commands = [
        [
            sys.executable, "-m", "research.paper_aio.run",
            "--stage", "candidate-runtime-gate", *evidence,
            "--parent-runtime-receipt", str(args.parent_runtime_receipt.resolve()),
            "--parent-e0", str(args.parent_e0.resolve()),
            "--capacity-override-receipt", str(args.capacity_override_receipt.resolve()),
            "--host-label", args.host_label, *common,
        ],
        [
            sys.executable, "-m", "research.paper_aio.run",
            "--stage", "candidate-lock", *evidence,
            "--candidate-runtime-gate", str(runtime_gate), *common,
        ],
        [
            sys.executable, "-m", "research.paper_aio.run",
            "--stage", "authorize", "--lane", "candidate",
            "--candidate-id", args.candidate_id, *common,
        ],
        [
            sys.executable, "-m", "research.paper_aio.run",
            "--stage", "train", "--lane", "candidate",
            "--candidate-id", args.candidate_id, "--resume",
            "--engineering-stop-after-updates", "8553", *common,
        ],
    ]
    log = output / "logs" / "CANDIDATE_ACTIVATION_SUCCESSOR.log"
    labels = ("runtime_gate", "evidence_lock", "authorization", "e1_capacity_probe")
    for index, (label, command) in enumerate(zip(labels, commands), start=1):
        atomic_json(state_path, {
            "schema": "final-unsb-paper-candidate-activation-successor-v1",
            "status": "RUNNING_" + label.upper(),
            "pid": os.getpid(), "candidate_id": args.candidate_id,
            "stage_index": index, "stage_count": len(commands),
            "candidate_git_commit": head,
            "candidate_protocol_fingerprint": env_check,
            "parent_readiness_mode": "authorized_running",
            "artifact_payloads_read_by_scheduler": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        code = run_logged(command, cwd=repo, log=log)
        if code:
            atomic_json(state_path, {
                "schema": "final-unsb-paper-candidate-activation-successor-v1",
                "status": "BLOCKED_ACTIVATION_GATE_FAILURE",
                "pid": os.getpid(), "candidate_id": args.candidate_id,
                "failed_stage": label, "child_returncode": code,
                "candidate_git_commit": head,
                "candidate_protocol_fingerprint": env_check,
                "parent_readiness_mode": "authorized_running",
                "artifact_payloads_read_by_scheduler": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            })
            return code

    atomic_json(state_path, {
        "schema": "final-unsb-paper-candidate-activation-successor-v1",
        "status": "E1_CAPACITY_PROBE_COMPLETE_AWAITING_MAKESPAN_DECISION",
        "pid": os.getpid(), "candidate_id": args.candidate_id,
        "candidate_git_commit": head,
        "candidate_protocol_fingerprint": env_check,
        "parent_readiness_mode": "authorized_running",
        "candidate_continued_beyond_e1": False,
        "artifact_payloads_read_by_scheduler": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        traceback.print_exc()
        raise SystemExit(2) from error
