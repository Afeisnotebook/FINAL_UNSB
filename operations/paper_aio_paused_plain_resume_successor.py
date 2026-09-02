"""Resume one paused paper plain lane after a fixed candidate reaches e200.

This control-plane successor is intentionally metric-blind.  It waits for the
candidate continuation controller to report its complete fixed e200 terminal
state, revalidates the exact paused plain checkpoint and frozen authorization,
and starts the original scientific checkout's plain supervisor.  It never
loads paired metrics, changes a training option, or continues a checkpoint on
another host.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from research.paper_aio.protocol import file_sha256

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is POSIX
    fcntl = None


STATE_SCHEMA = "final-unsb-paper-paused-plain-resume-successor-v1"
BLOCKED_PREFIXES = ("BLOCKED", "FAIL", "FATAL")


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def git_identity(repo: Path) -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip())
    return head, dirty


def predecessor_decision(status: str | None, *, timed_out: bool) -> str:
    if status == "COMPLETE_CANDIDATE_E200":
        return "START"
    if status and status.startswith(BLOCKED_PREFIXES):
        return "BLOCK"
    if timed_out:
        return "TIMEOUT"
    return "WAIT"


def validate_predecessor(
    payload: dict[str, Any], *, candidate_id: str, require_complete: bool,
) -> None:
    status = payload.get("status")
    if (
        payload.get("schema") != "final-unsb-paper-candidate-continuation-v1"
        or payload.get("candidate_id") != candidate_id
        or payload.get("performance_values_read") is not False
        or payload.get("paired_metric_control") is not False
        or payload.get("confirmation20_opened") is not False
        or (require_complete and status != "COMPLETE_CANDIDATE_E200")
    ):
        raise RuntimeError("candidate predecessor state is invalid or not complete")


def validate_paused_plain(
    *, training_output: Path, required_epoch: int,
    required_full_state_sha256: str, required_scientific_state_sha256: str,
    required_git_commit: str, required_protocol_fingerprint: str,
) -> dict[str, Any]:
    lane_root = Path(training_output) / "lanes" / "plain"
    checkpoint = lane_root / "full_state_latest.pt"
    sidecar_path = Path(str(checkpoint) + ".json")
    sidecar = read_json(sidecar_path)
    metadata = sidecar.get("metadata") or {}
    expected_updates = int(required_epoch) * 8_553
    if (
        sidecar.get("schema") != "final-unsb-paper-aio-full-state-v1"
        or sidecar.get("lane_id") != "plain"
        or int(sidecar.get("step", -1)) != expected_updates
        or int(sidecar.get("physical_epoch_completed", -1)) != int(required_epoch)
        or int(sidecar.get("target_steps", -1)) != 1_710_600
        or sidecar.get("full_state_sha256") != required_full_state_sha256
        or sidecar.get("scientific_state_sha256") != required_scientific_state_sha256
        or metadata.get("lane_id") != "plain"
        or metadata.get("git_commit") != required_git_commit
        or metadata.get("protocol_fingerprint") != required_protocol_fingerprint
        or int(metadata.get("seed", -1)) != 2026
        or int(metadata.get("batch_size", -1)) != 1
        or metadata.get("paired_controller_access") is not False
        or metadata.get("confirmation20_opened") is not False
        or not checkpoint.is_file()
        or file_sha256(checkpoint) != required_full_state_sha256
    ):
        raise RuntimeError("paused plain full state no longer matches its frozen identity")
    return sidecar


def validate_plain_authorization(
    *, training_output: Path, required_protocol_fingerprint: str,
) -> None:
    authorization = read_json(
        Path(training_output) / "gates" / "LANE_AUTHORIZATION_plain.json",
    )
    if (
        authorization.get("schema") != "final-unsb-paper-lane-authorization-v1"
        or authorization.get("status") != "PASS"
        or authorization.get("lane_id") != "plain"
        or authorization.get("protocol_fingerprint") != required_protocol_fingerprint
        or authorization.get("failures") != []
        or authorization.get("paired_metric_control") is not False
        or authorization.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("frozen plain authorization is absent or changed")


def matching_plain_train_pids(training_output: Path) -> list[int]:
    marker = str(Path(training_output).resolve())
    result: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return result
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if (
            "research.paper_aio.run" in command
            and "--stage train" in command
            and "--lane plain" in command
            and marker in command
        ):
            result.append(int(entry.name))
    return sorted(result)


def state_payload(args: argparse.Namespace, *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "candidate_id": args.candidate_id,
        "plain_lane_id": "plain",
        "plain_source_host_label": args.plain_source_host_label,
        "required_resume_epoch": int(args.required_resume_epoch),
        "cross_host_checkpoint_resume": False,
        "training_configuration_changed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def ensure_export_successor(args: argparse.Namespace) -> int:
    output = args.training_output.resolve()
    state = read_json(output / "operations" / "EXPORT_SUCCESSOR_plain_STATE.json")
    if process_alive(state.get("pid")):
        return int(state["pid"])
    command = [
        str(args.python.resolve()),
        str(args.training_repo.resolve() / "operations" / "paper_aio_export_successor.py"),
        "--repo", str(args.training_repo.resolve()),
        "--source-output", str(output),
        "--destination", str(output / "exports"),
        "--lane", "plain",
        "--source-host-label", args.plain_source_host_label,
        "--required-training-git-commit", args.required_training_git_commit,
        "--required-training-protocol-fingerprint",
        args.required_protocol_fingerprint,
        "--poll-seconds", str(args.poll_seconds),
        "--timeout-hours", "480",
    ]
    log = output / "logs" / "EXPORT_SUCCESSOR_plain_restart.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=args.training_repo.resolve(), stdout=handle,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    handle.close()
    return process.pid


def run(args: argparse.Namespace) -> int:
    if args.poll_seconds < 30 or args.poll_seconds > 600:
        raise RuntimeError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise RuntimeError("timeout must be at least 24 hours")
    control_head, control_dirty = git_identity(args.control_repo.resolve())
    if control_head != args.required_control_git_commit or control_dirty:
        raise RuntimeError("plain-resume control checkout identity changed")
    training_head, training_dirty = git_identity(args.training_repo.resolve())
    if training_head != args.required_training_git_commit or training_dirty:
        raise RuntimeError("frozen plain scientific checkout identity changed")
    if not args.python.is_file():
        raise RuntimeError("frozen plain Python runtime is missing")
    fingerprint = subprocess.check_output(
        [str(args.python.resolve()), "-c", (
            "from research.paper_aio.protocol import protocol_fingerprint; "
            "print(protocol_fingerprint())"
        )], cwd=args.training_repo.resolve(), text=True,
    ).strip()
    if fingerprint != args.required_protocol_fingerprint:
        raise RuntimeError("frozen plain protocol fingerprint changed")
    validate_plain_authorization(
        training_output=args.training_output,
        required_protocol_fingerprint=args.required_protocol_fingerprint,
    )
    validate_paused_plain(
        training_output=args.training_output,
        required_epoch=args.required_resume_epoch,
        required_full_state_sha256=args.required_full_state_sha256,
        required_scientific_state_sha256=args.required_scientific_state_sha256,
        required_git_commit=args.required_training_git_commit,
        required_protocol_fingerprint=args.required_protocol_fingerprint,
    )

    operations = args.successor_output.resolve() / "operations"
    state_path = operations / "PAUSED_PLAIN_RESUME_SUCCESSOR_STATE.json"
    lock_path = operations / "PAUSED_PLAIN_RESUME_SUCCESSOR.lock"
    operations.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("paused-plain successor requires POSIX file locking")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            predecessor = read_json(args.predecessor_state.resolve())
            status = predecessor.get("status")
            decision = predecessor_decision(
                status, timed_out=time.time() - started >= args.timeout_hours * 3600,
            )
            atomic_json(state_path, state_payload(
                args,
                status={
                    "WAIT": "WAITING_FOR_STCGR_COMPLETE_E200",
                    "START": "STCGR_COMPLETE_VALIDATING_PAUSED_PLAIN",
                    "BLOCK": "BLOCKED_STCGR_ENGINEERING_FAILURE",
                    "TIMEOUT": "BLOCKED_STCGR_TIMEOUT",
                }[decision],
                predecessor_status=status,
            ))
            if decision == "START":
                validate_predecessor(
                    predecessor, candidate_id=args.candidate_id,
                    require_complete=True,
                )
                break
            if decision in {"BLOCK", "TIMEOUT"}:
                return 3
            if predecessor:
                validate_predecessor(
                    predecessor, candidate_id=args.candidate_id,
                    require_complete=False,
                )
            time.sleep(args.poll_seconds)

        validate_paused_plain(
            training_output=args.training_output,
            required_epoch=args.required_resume_epoch,
            required_full_state_sha256=args.required_full_state_sha256,
            required_scientific_state_sha256=args.required_scientific_state_sha256,
            required_git_commit=args.required_training_git_commit,
            required_protocol_fingerprint=args.required_protocol_fingerprint,
        )
        active = matching_plain_train_pids(args.training_output)
        if active:
            raise RuntimeError(f"plain trainer already exists: {active}")
        export_pid = ensure_export_successor(args)
        command = [
            str(args.python.resolve()),
            str(args.training_repo.resolve() / "operations" / "paper_aio_supervisor.py"),
            "--repo", str(args.training_repo.resolve()),
            "--output", str(args.training_output.resolve()),
            "--manifest", str(args.manifest.resolve()),
            "--data-root", str(args.data_root.resolve()),
            "--train-view", str(args.train_view.resolve()),
            "--lane", "plain", "--gpu", str(args.gpu),
        ]
        log = args.training_output.resolve() / "logs" / "SUPERVISOR_plain_resumed.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        supervisor = subprocess.Popen(
            command, cwd=args.training_repo.resolve(), stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        while supervisor.poll() is None:
            atomic_json(state_path, state_payload(
                args, status="PLAIN_SUPERVISOR_RUNNING_EXACT_RESUME",
                supervisor_pid=supervisor.pid, export_successor_pid=export_pid,
            ))
            time.sleep(args.poll_seconds)
        handle.close()
        supervisor_state = read_json(
            args.training_output.resolve() / "gates" / "SUPERVISOR_plain.json",
        )
        complete = (
            supervisor.returncode == 0
            and supervisor_state.get("status") == "COMPLETE_E200"
        )
        atomic_json(state_path, state_payload(
            args,
            status="COMPLETE_PLAIN_E200" if complete else "BLOCKED_PLAIN_SUPERVISOR_EXIT",
            supervisor_pid=supervisor.pid, export_successor_pid=export_pid,
            supervisor_returncode=supervisor.returncode,
            supervisor_status=supervisor_state.get("status"),
        ))
        return 0 if complete else (supervisor.returncode or 4)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-repo", type=Path, required=True)
    parser.add_argument("--required-control-git-commit", required=True)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--required-training-git-commit", required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--successor-output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--plain-source-host-label", required=True)
    parser.add_argument("--required-protocol-fingerprint", required=True)
    parser.add_argument("--required-resume-epoch", type=int, required=True)
    parser.add_argument("--required-full-state-sha256", required=True)
    parser.add_argument("--required-scientific-state-sha256", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=960.0)
    return parser.parse_args(argv)


def main() -> int:
    args = arguments()
    try:
        return run(args)
    except Exception as error:
        output = args.successor_output.resolve()
        atomic_json(
            output / "operations" / "PAUSED_PLAIN_RESUME_SUCCESSOR_STATE.json",
            state_payload(
                args, status="BLOCKED_FAIL_CLOSED", error_type=type(error).__name__,
            ),
        )
        print(f"paused-plain resume successor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
