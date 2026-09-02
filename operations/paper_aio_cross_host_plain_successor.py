"""Start a fresh runtime-matched plain lane after a metric-blind predecessor.

The successor is intentionally narrow: it can only train the frozen paper
``plain`` lane from e0.  It waits for an external predecessor supervisor to
reach COMPLETE_E200, runs the complete engineering gate chain, and requires a
2000-update exact runtime match to a named peer before long training begins.
It never reads a metric file or continues a checkpoint across hosts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is POSIX; tests import on Windows
    fcntl = None


BLOCKED_PREFIXES = ("BLOCKED", "FAIL")
TERMINAL_SUPERVISOR_STATES = {"COMPLETE_E200"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def predecessor_decision(status: str | None, *, timed_out: bool) -> str:
    if status in TERMINAL_SUPERVISOR_STATES:
        return "START"
    if status and status.startswith(BLOCKED_PREFIXES):
        return "BLOCK"
    if timed_out:
        return "TIMEOUT"
    return "WAIT"


def validate_runtime_receipt(
    receipt: dict, *, host_label: str, required_protocol_fingerprint: str,
) -> None:
    if (
        receipt.get("schema") != "final-unsb-paper-runtime-twin-receipt-v1"
        or receipt.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or receipt.get("host_label") != host_label
        or receipt.get("updates") != 2000
        or receipt.get("protocol_fingerprint") != required_protocol_fingerprint
        or receipt.get("exact_runtime_equivalence") is not True
        or receipt.get("differences") != {}
        or receipt.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("cross-host plain runtime twin did not pass exactly")


def gate_commands(args: argparse.Namespace) -> list[list[str]]:
    common = [
        "--output", str(args.training_output.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--gpu", str(args.gpu),
    ]
    python = str(args.python.resolve())
    twin_checkpoint = (
        args.training_output.resolve() / "runtime_twin" / args.host_label
        / "lanes" / "plain" / "full_state_latest.pt"
    )
    return [
        [
            python, "-m", "research.paper_aio.run", "--stage", "preflight",
            "--host-label", args.host_label, *common,
        ],
        [
            python, "-m", "research.paper_aio.run", "--stage", "resume-gate",
            "--lane", "plain", *common,
        ],
        [
            python, "-m", "research.paper_aio.run", "--stage", "runtime-twin",
            "--host-label", args.host_label,
            "--peer-receipt", str(args.peer_runtime_receipt.resolve()), *common,
        ],
        [
            python, "-m", "research.paper_aio.run",
            "--stage", "evaluation-repeat-gate", "--lane", "plain",
            "--checkpoint", str(twin_checkpoint), *common,
        ],
        [
            python, "-m", "research.paper_aio.run", "--stage", "authorize",
            "--lane", "plain", *common,
        ],
    ]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--peer-runtime-receipt", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--required-training-git-commit", required=True)
    parser.add_argument("--required-protocol-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser.parse_args(argv)


def git_identity(repo: Path) -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip())
    return head, dirty


def run_logged(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.time():.3f}] {json.dumps(command)}\n")
        handle.flush()
        return subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def state_payload(args: argparse.Namespace, *, status: str, **extra) -> dict:
    return {
        "schema": "final-unsb-paper-cross-host-plain-successor-v1",
        "status": status,
        "pid": os.getpid(),
        "predecessor_state": str(args.predecessor_state.resolve()),
        "lane_id": "plain",
        "host_label": args.host_label,
        "source_host_label": args.source_host_label,
        "training_git_commit": args.required_training_git_commit,
        "required_protocol_fingerprint": args.required_protocol_fingerprint,
        "fresh_e0_required": True,
        "cross_host_checkpoint_resume": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.poll_seconds < 30 or args.timeout_hours < 24:
        raise RuntimeError("successor polling/timeout is unsafe")
    repo = args.training_repo.resolve()
    output = args.training_output.resolve()
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError(f"frozen Python runtime is missing: {python}")
    head, dirty = git_identity(repo)
    if head != args.required_training_git_commit or dirty:
        raise RuntimeError("frozen plain training checkout identity changed")
    peer = read_json(args.peer_runtime_receipt.resolve())
    if (
        peer.get("schema") != "final-unsb-paper-runtime-twin-receipt-v1"
        or peer.get("updates") != 2000
        or peer.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
        or peer.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("peer runtime receipt is invalid")

    operations = output / "operations"
    state_path = operations / "CROSS_HOST_PLAIN_SUCCESSOR_STATE.json"
    lock_path = operations / "CROSS_HOST_PLAIN_SUCCESSOR.lock"
    log = output / "logs" / "CROSS_HOST_PLAIN_SUCCESSOR.log"
    operations.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("cross-host plain successor requires POSIX file locking")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        while True:
            predecessor = (
                read_json(args.predecessor_state.resolve())
                if args.predecessor_state.is_file() else {}
            )
            decision = predecessor_decision(
                predecessor.get("status"),
                timed_out=(time.time() - started) >= args.timeout_hours * 3600,
            )
            atomic_json(state_path, state_payload(
                args,
                status={
                    "WAIT": "WAITING_FOR_PREDECESSOR_E200",
                    "START": "PREDECESSOR_COMPLETE_STARTING_EXACT_GATES",
                    "BLOCK": "BLOCKED_PREDECESSOR_NOT_RECOVERABLE",
                    "TIMEOUT": "BLOCKED_PREDECESSOR_TIMEOUT",
                }[decision],
                predecessor_status=predecessor.get("status"),
            ))
            if decision == "START":
                break
            if decision in {"BLOCK", "TIMEOUT"}:
                return 3
            time.sleep(args.poll_seconds)

        head, dirty = git_identity(repo)
        if head != args.required_training_git_commit or dirty:
            raise RuntimeError("frozen checkout moved while successor was waiting")
        commands = gate_commands(args)
        for index, command in enumerate(commands, start=1):
            atomic_json(state_path, state_payload(
                args, status="RUNNING_EXACT_ENGINEERING_GATES",
                gate_index=index, gate_count=len(commands),
            ))
            code = run_logged(command, cwd=repo, log=log)
            if code:
                atomic_json(state_path, state_payload(
                    args, status="BLOCKED_ENGINEERING_GATE_FAILURE",
                    gate_index=index, child_returncode=code,
                ))
                return code

        runtime_path = output / "gates" / f"RUNTIME_TWIN_{args.host_label}.json"
        validate_runtime_receipt(
            read_json(runtime_path), host_label=args.host_label,
            required_protocol_fingerprint=args.required_protocol_fingerprint,
        )
        protocol = read_json(output / "PAPER_PROTOCOL.json")
        if protocol.get("protocol_fingerprint") != args.required_protocol_fingerprint:
            raise RuntimeError("plain protocol fingerprint changed after gates")

        export_log = output / "logs" / "EXPORT_SUCCESSOR_plain.log"
        export_handle = export_log.open("a", encoding="utf-8")
        export_command = [
            str(python), str(repo / "operations" / "paper_aio_export_successor.py"),
            "--repo", str(repo), "--source-output", str(output),
            "--destination", str(output / "exports"), "--lane", "plain",
            "--source-host-label", args.source_host_label,
            "--required-training-git-commit", args.required_training_git_commit,
            "--required-training-protocol-fingerprint",
            args.required_protocol_fingerprint,
            "--poll-seconds", "60", "--timeout-hours", str(args.timeout_hours),
        ]
        exporter = subprocess.Popen(
            export_command, cwd=repo, stdout=export_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        supervisor_command = [
            str(python), str(repo / "operations" / "paper_aio_supervisor.py"),
            "--repo", str(repo), "--output", str(output),
            "--manifest", str(args.manifest.resolve()),
            "--data-root", str(args.data_root.resolve()),
            "--train-view", str(args.train_view.resolve()),
            "--lane", "plain", "--gpu", str(args.gpu),
        ]
        supervisor_log = output / "logs" / "SUPERVISOR_plain.log"
        supervisor_handle = supervisor_log.open("a", encoding="utf-8")
        supervisor = subprocess.Popen(
            supervisor_command, cwd=repo, stdout=supervisor_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        while supervisor.poll() is None:
            atomic_json(state_path, state_payload(
                args, status="PLAIN_SUPERVISOR_RUNNING",
                supervisor_pid=supervisor.pid,
                export_successor_pid=exporter.pid,
                exact_runtime_equivalence=True,
            ))
            time.sleep(args.poll_seconds)
        supervisor_handle.close()
        export_handle.close()
        supervisor_state_path = output / "gates" / "SUPERVISOR_plain.json"
        supervisor_state = (
            read_json(supervisor_state_path) if supervisor_state_path.is_file() else {}
        )
        complete = (
            supervisor.returncode == 0
            and supervisor_state.get("status") == "COMPLETE_E200"
        )
        atomic_json(state_path, state_payload(
            args,
            status="COMPLETE_PLAIN_E200" if complete
            else "BLOCKED_PLAIN_SUPERVISOR_EXIT",
            supervisor_pid=supervisor.pid,
            export_successor_pid=exporter.pid,
            supervisor_returncode=supervisor.returncode,
            supervisor_status=supervisor_state.get("status"),
            exact_runtime_equivalence=True,
        ))
        return 0 if complete else (supervisor.returncode or 5)


if __name__ == "__main__":
    raise SystemExit(main())
