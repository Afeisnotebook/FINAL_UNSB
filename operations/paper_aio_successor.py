"""Fail-closed successor orchestration for frozen paper lanes.

This process waits for an already-authorized predecessor lane to reach its
fixed e200 terminal state.  It then runs the successor's frozen engineering
gates and, only if every gate passes, hands the lane to paper_aio_supervisor.
It never reads metrics and cannot select a checkpoint or alter the protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


RUNNING_STATES = {"CHILD_RUNNING", "WAITING_TO_EXACT_RESUME"}
BLOCKED_PREFIXES = ("BLOCKED", "FAIL")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predecessor_decision(status: str | None) -> str:
    """Return WAIT, START, or BLOCK without consulting scientific metrics."""
    if status == "COMPLETE_E200":
        return "START"
    if status and status.startswith(BLOCKED_PREFIXES):
        return "BLOCK"
    return "WAIT"


def gate_commands(
    *, python: str, output: Path, manifest: Path, data_root: Path,
    train_view: Path, successor: str, gpu: int,
) -> list[list[str]]:
    common = [
        "--output", str(output),
        "--manifest", str(manifest),
        "--data-root", str(data_root),
        "--train-view", str(train_view),
        "--gpu", str(gpu),
    ]
    commands = [
        [python, "-m", "research.paper_aio.run", "--stage", "preflight", *common],
        [
            python, "-m", "research.paper_aio.run", "--stage", "resume-gate",
            "--lane", successor, *common,
        ],
    ]
    if successor == "proposal":
        commands.append([
            python, "-m", "research.paper_aio.run",
            "--stage", "zero-intervention-gate", *common,
        ])
    elif successor == "amtnc":
        commands.append([
            python,
            str(Path(__file__).with_name("paper_aio_amtnc_identity_gate.py")),
            "--output", str(output),
            "--manifest", str(manifest),
            "--train-view", str(train_view),
            "--gpu", str(gpu),
        ])
    checkpoint = (
        output / "resume_gate" / successor / "continuous" / "lanes"
        / successor / "full_state_latest.pt"
    )
    commands.append([
        python, "-m", "research.paper_aio.run",
        "--stage", "evaluation-repeat-gate", "--lane", successor,
        "--checkpoint", str(checkpoint), *common,
    ])
    authorize = [
        python, "-m", "research.paper_aio.run", "--stage", "authorize",
        "--lane", successor, *common,
    ]
    if successor == "proposal":
        authorize.extend(["--matched-plain-mode", "same_runtime_output_root"])
    commands.append(authorize)
    return commands


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--predecessor", required=True)
    parser.add_argument(
        "--successor",
        choices=("proposal", "amtnc", "hjcgr", "cyclegan"),
        required=True,
        help=(
            "Frozen static paper lane to gate and start after the predecessor. "
            "Adding a lane here changes only metric-blind scheduling; the "
            "scientific lane definition remains pinned by the training repo."
        ),
    )
    parser.add_argument("--required-git-commit", required=True)
    parser.add_argument("--required-protocol-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


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
    manifest = args.manifest.resolve()
    data_root = args.data_root.resolve()
    train_view = args.train_view.resolve()
    label = f"{args.predecessor}_TO_{args.successor}"
    state_path = output / "gates" / f"SUCCESSOR_{label}.json"
    log = output / "logs" / f"SUCCESSOR_{label}.log"
    predecessor_state = output / "gates" / f"SUPERVISOR_{args.predecessor}.json"

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if head != args.required_git_commit:
        atomic_json(state_path, {
            "schema": "final-unsb-paper-successor-v1",
            "status": "BLOCKED_SCIENTIFIC_COMMIT_MISMATCH",
            "expected": args.required_git_commit,
            "observed": head,
            "confirmation20_opened": False,
        })
        return 2

    while True:
        predecessor = read_json(predecessor_state) if predecessor_state.is_file() else {}
        decision = predecessor_decision(predecessor.get("status"))
        atomic_json(state_path, {
            "schema": "final-unsb-paper-successor-v1",
            "status": (
                "WAITING_FOR_PREDECESSOR_E200" if decision == "WAIT"
                else "PREDECESSOR_COMPLETE_STARTING_GATES" if decision == "START"
                else "BLOCKED_PREDECESSOR_NOT_RECOVERABLE"
            ),
            "predecessor": args.predecessor,
            "successor": args.successor,
            "predecessor_status": predecessor.get("status"),
            "scientific_git_commit": head,
            "required_protocol_fingerprint": args.required_protocol_fingerprint,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        if decision == "START":
            break
        if decision == "BLOCK":
            return 3
        time.sleep(max(10, int(args.poll_seconds)))

    commands = gate_commands(
        python=sys.executable, output=output, manifest=manifest,
        data_root=data_root, train_view=train_view,
        successor=args.successor, gpu=args.gpu,
    )
    for index, command in enumerate(commands, start=1):
        atomic_json(state_path, {
            "schema": "final-unsb-paper-successor-v1",
            "status": "RUNNING_SUCCESSOR_GATES",
            "predecessor": args.predecessor,
            "successor": args.successor,
            "gate_index": index,
            "gate_count": len(commands),
            "scientific_git_commit": head,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        code = run_logged(command, cwd=repo, log=log)
        if code:
            atomic_json(state_path, {
                "schema": "final-unsb-paper-successor-v1",
                "status": "BLOCKED_SUCCESSOR_GATE_FAILURE",
                "predecessor": args.predecessor,
                "successor": args.successor,
                "gate_index": index,
                "child_returncode": code,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            })
            return code

    protocol = read_json(output / "PAPER_PROTOCOL.json")
    observed = protocol.get("protocol_fingerprint")
    if observed != args.required_protocol_fingerprint:
        atomic_json(state_path, {
            "schema": "final-unsb-paper-successor-v1",
            "status": "BLOCKED_PROTOCOL_FINGERPRINT_MISMATCH",
            "expected": args.required_protocol_fingerprint,
            "observed": observed,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        return 4

    atomic_json(state_path, {
        "schema": "final-unsb-paper-successor-v1",
        "status": "SUCCESSOR_SUPERVISOR_RUNNING",
        "predecessor": args.predecessor,
        "successor": args.successor,
        "scientific_git_commit": head,
        "protocol_fingerprint": observed,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    command = [
        sys.executable, str(repo / "operations" / "paper_aio_supervisor.py"),
        "--repo", str(repo), "--output", str(output),
        "--manifest", str(manifest), "--data-root", str(data_root),
        "--train-view", str(train_view), "--lane", args.successor,
        "--gpu", str(args.gpu),
    ]
    code = run_logged(command, cwd=repo, log=log)
    supervisor = output / "gates" / f"SUPERVISOR_{args.successor}.json"
    child = read_json(supervisor) if supervisor.is_file() else {}
    atomic_json(state_path, {
        "schema": "final-unsb-paper-successor-v1",
        "status": (
            "COMPLETE_SUCCESSOR_E200"
            if code == 0 and child.get("status") == "COMPLETE_E200"
            else "BLOCKED_SUCCESSOR_SUPERVISOR_EXIT"
        ),
        "predecessor": args.predecessor,
        "successor": args.successor,
        "child_returncode": code,
        "successor_supervisor_status": child.get("status"),
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    return code if code else (0 if child.get("status") == "COMPLETE_E200" else 5)


if __name__ == "__main__":
    raise SystemExit(main())
