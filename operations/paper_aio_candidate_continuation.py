"""Durably continue an e1-probed paper candidate under a frozen schedule.

The controller consumes engineering state only.  It never reads an evaluation
file or a performance value.  ``co_resident`` starts after the candidate e1
probe while the matched plain is healthy; ``after_parent`` waits for the
matched plain supervisor to reach its fixed e200 terminal state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ACTIVATION_READY = "E1_CAPACITY_PROBE_COMPLETE_AWAITING_MAKESPAN_DECISION"
PARENT_RUNNING = {"CHILD_RUNNING", "WAITING_TO_EXACT_RESUME"}
BLOCKED_PREFIXES = ("BLOCKED", "FAIL")

try:
    import fcntl
except ImportError:  # pragma: no cover - the controller runs on Linux hosts.
    fcntl = None


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def continuation_decision(
    *, activation_status: str | None, parent_status: str | None, mode: str,
) -> str:
    """Return a metric-blind wait/start/block decision."""
    if activation_status and activation_status.startswith(BLOCKED_PREFIXES):
        return "BLOCK_ACTIVATION"
    if activation_status != ACTIVATION_READY:
        return "WAIT_ACTIVATION"
    if parent_status and parent_status.startswith(BLOCKED_PREFIXES):
        return "BLOCK_PARENT"
    if mode == "co_resident":
        return "START" if parent_status in PARENT_RUNNING | {"COMPLETE_E200"} else "WAIT_PARENT"
    if mode == "after_parent":
        return "START" if parent_status == "COMPLETE_E200" else "WAIT_PARENT"
    raise ValueError(f"unknown continuation mode: {mode}")


def _git_identity(repo: Path) -> tuple[str, str]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip()
    return head, dirty


def _protocol_fingerprint(python: Path, repo: Path) -> str:
    return subprocess.check_output(
        [str(python), "-c", (
            "from research.paper_aio.protocol import protocol_fingerprint; "
            "print(protocol_fingerprint())"
        )], cwd=repo, text=True,
    ).strip()


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-repo", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--required-candidate-git-commit", required=True)
    parser.add_argument("--required-candidate-protocol-fingerprint", required=True)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--mode", choices=("co_resident", "after_parent"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=480.0)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if int(args.poll_seconds) < 30 or float(args.timeout_hours) < 24:
        raise RuntimeError("candidate continuation polling/timeout is unsafe")
    repo = args.candidate_repo.resolve()
    output = args.candidate_output.resolve()
    python = args.candidate_python.resolve()
    head, dirty = _git_identity(repo)
    if head != args.required_candidate_git_commit or dirty:
        raise RuntimeError("frozen candidate checkout identity changed")
    fingerprint = _protocol_fingerprint(python, repo)
    if fingerprint != args.required_candidate_protocol_fingerprint:
        raise RuntimeError("frozen candidate protocol fingerprint changed")

    operations = output / "operations"
    state_path = operations / "CANDIDATE_CONTINUATION_STATE.json"
    lock_path = operations / "CANDIDATE_CONTINUATION.lock"
    operations.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("candidate continuation requires POSIX file locking")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("candidate continuation is already running") from error

        activation_path = operations / "CANDIDATE_ACTIVATION_SUCCESSOR_STATE.json"
        parent_path = args.parent_output.resolve() / "gates" / "SUPERVISOR_plain.json"
        started = time.time()
        while True:
            activation = _read(activation_path)
            parent = _read(parent_path)
            decision = continuation_decision(
                activation_status=activation.get("status"),
                parent_status=parent.get("status"), mode=args.mode,
            )
            _write(state_path, {
                "schema": "final-unsb-paper-candidate-continuation-v1",
                "status": decision,
                "pid": os.getpid(),
                "mode": args.mode,
                "candidate_id": args.candidate_id,
                "activation_status": activation.get("status"),
                "parent_status": parent.get("status"),
                "candidate_git_commit": head,
                "candidate_protocol_fingerprint": fingerprint,
                "performance_values_read": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            })
            if decision == "START":
                break
            if decision.startswith("BLOCK"):
                return 3
            if time.time() - started > float(args.timeout_hours) * 3600:
                raise TimeoutError("candidate continuation exceeded its frozen timeout")
            time.sleep(int(args.poll_seconds))

        lane_root = output / "lanes" / args.candidate_id
        run_state = _read(lane_root / "RUN_STATE.json")
        if not (
            run_state.get("status") == "ENGINEERING_PAUSE"
            and int(run_state.get("final_updates", -1)) == 8_553
            and float(run_state.get("final_data_epoch", -1)) == 1.0
            and run_state.get("confirmation20_opened") is False
        ):
            raise RuntimeError("candidate e1 exact-resume state is missing or invalid")
        authorization = (
            output / "gates" / f"CANDIDATE_AUTHORIZATION_{args.candidate_id}.json"
        )
        if not authorization.is_file():
            raise RuntimeError("candidate full-data authorization is missing")

        export_state_path = operations / f"EXPORT_SUCCESSOR_{args.candidate_id}_STATE.json"
        export_state = _read(export_state_path)
        export_pid = export_state.get("pid") if _process_alive(export_state.get("pid")) else None
        if export_pid is None:
            export_log = output / "logs" / f"EXPORT_SUCCESSOR_{args.candidate_id}.log"
            export_handle = export_log.open("a", encoding="utf-8")
            export_command = [
                str(python), str(repo / "operations" / "paper_aio_export_successor.py"),
                "--repo", str(repo), "--source-output", str(output),
                "--destination", str(output / "exports"), "--lane", args.candidate_id,
                "--source-host-label", args.source_host_label,
                "--required-training-git-commit", head,
                "--required-training-protocol-fingerprint", fingerprint,
                "--poll-seconds", str(args.poll_seconds),
                "--timeout-hours", str(args.timeout_hours),
            ]
            export_process = subprocess.Popen(
                export_command, cwd=repo, stdout=export_handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            export_handle.close()
            export_pid = export_process.pid

        supervisor_command = [
            str(python), str(repo / "operations" / "paper_aio_supervisor.py"),
            "--repo", str(repo), "--output", str(output),
            "--manifest", str(args.manifest.resolve()),
            "--data-root", str(args.data_root.resolve()),
            "--train-view", str(args.train_view.resolve()),
            "--lane", "candidate", "--candidate-id", args.candidate_id,
            "--gpu", str(args.gpu),
        ]
        _write(state_path, {
            "schema": "final-unsb-paper-candidate-continuation-v1",
            "status": "CANDIDATE_SUPERVISOR_RUNNING",
            "pid": os.getpid(), "mode": args.mode,
            "candidate_id": args.candidate_id,
            "candidate_git_commit": head,
            "candidate_protocol_fingerprint": fingerprint,
            "export_successor_pid": export_pid,
            "supervisor_command": supervisor_command,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        supervisor_log = output / "logs" / f"CONTINUATION_{args.candidate_id}.log"
        with supervisor_log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                supervisor_command, cwd=repo, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        supervisor = _read(output / "gates" / f"SUPERVISOR_{args.candidate_id}.json")
        complete = completed.returncode == 0 and supervisor.get("status") == "COMPLETE_E200"
        _write(state_path, {
            "schema": "final-unsb-paper-candidate-continuation-v1",
            "status": "COMPLETE_CANDIDATE_E200" if complete else "BLOCKED_SUPERVISOR_EXIT",
            "pid": os.getpid(), "mode": args.mode,
            "candidate_id": args.candidate_id,
            "child_returncode": completed.returncode,
            "supervisor_status": supervisor.get("status"),
            "export_successor_pid": export_pid,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        return 0 if complete else (completed.returncode or 4)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"candidate continuation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
