"""Queue DCLGAN behind a fixed metric-blind predecessor on one target GPU.

The successor runs the complete host-bound 1000-update gate and only then
hands the exact update-1000 state to the durable e200 supervisor.  It never
reads a metric value and cannot start from a checkpoint on another host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-paper-dclgan-target-successor-v1"
CONTRACT_SCHEMA = "final-unsb-paper-dclgan-target-successor-contract-v1"
BLOCKED_PREFIXES = ("BLOCKED", "FAIL", "FATAL")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def repo_identity(repo: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).strip()
    return commit, dirty


def acquire_lock(path: Path):
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


def predecessor_decision(
    payload: dict[str, Any], required_status: str,
) -> str:
    if any(
        payload.get(key) is True
        for key in ("paired_metric_control", "confirmation20_opened")
    ):
        return "BLOCK"
    status = str(payload.get("status", ""))
    if status == required_status:
        return "START"
    if status.startswith(BLOCKED_PREFIXES):
        return "BLOCK"
    return "WAIT"


def state_payload(args: argparse.Namespace, *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "predecessor_state": str(args.predecessor_state),
        "required_predecessor_status": args.required_predecessor_status,
        "required_git_commit": args.required_git_commit,
        "required_adapter_fingerprint": args.required_adapter_fingerprint,
        "target_host_label": args.target_host_label,
        "output": str(args.output),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    commit, dirty = repo_identity(args.repo)
    if commit != args.required_git_commit or dirty:
        raise RuntimeError("DCLGAN target successor checkout is not frozen and clean")
    sources = {}
    for relative in (
        "operations/paper_aio_dclgan_adapter.py",
        "operations/paper_aio_dclgan_gate_supervisor.py",
        "operations/paper_aio_dclgan_long_supervisor.py",
        "operations/paper_aio_dclgan_target_successor.py",
        "configs/PAPER_DCLGAN_NEGCUT_SOURCE_GATE.json",
    ):
        sources[relative] = file_sha256(args.repo / relative)
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "repo": str(args.repo),
        "upstream_root": str(args.upstream_root),
        "manifest": str(args.manifest),
        "train_view": str(args.train_view),
        "data_root": str(args.data_root),
        "output": str(args.output),
        "predecessor_state": str(args.predecessor_state),
        "required_predecessor_status": args.required_predecessor_status,
        "required_git_commit": args.required_git_commit,
        "required_adapter_fingerprint": args.required_adapter_fingerprint,
        "target_host_label": args.target_host_label,
        "gpu": int(args.gpu),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "source_sha256": sources,
        "performance_values_available_to_scheduling": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def freeze_contract(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    proposed = build_contract(args)
    path = args.output / "operations" / "DCLGAN_TARGET_SUCCESSOR_CONTRACT.json"
    if path.is_file():
        existing = read_json(path)
        if existing != proposed:
            raise RuntimeError("DCLGAN target successor contract changed")
        return path, existing
    atomic_json(path, proposed)
    return path, proposed


def verify_contract(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    commit, dirty = repo_identity(repo)
    if commit != contract["required_git_commit"] or dirty:
        raise RuntimeError("DCLGAN target successor checkout changed")
    for relative, expected in contract["source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"DCLGAN successor source changed: {relative}")


def gate_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.repo / "operations" / "paper_aio_dclgan_gate_supervisor.py"),
        "--repo",
        str(args.repo),
        "--upstream-root",
        str(args.upstream_root),
        "--manifest",
        str(args.manifest),
        "--train-view",
        str(args.train_view),
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.output),
        "--gpu",
        str(args.gpu),
        "--required-git-commit",
        args.required_git_commit,
    ]


def long_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.repo / "operations" / "paper_aio_dclgan_long_supervisor.py"),
        "--repo",
        str(args.repo),
        "--upstream-root",
        str(args.upstream_root),
        "--manifest",
        str(args.manifest),
        "--train-view",
        str(args.train_view),
        "--output",
        str(args.output),
        "--gpu",
        str(args.gpu),
        "--required-git-commit",
        args.required_git_commit,
        "--required-adapter-fingerprint",
        args.required_adapter_fingerprint,
    ]


def run_child(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.time():.3f}] {json.dumps(command)}\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--required-predecessor-status", required=True)
    parser.add_argument("--target-host-label", required=True)
    parser.add_argument("--required-git-commit", required=True)
    parser.add_argument("--required-adapter-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    for name in (
        "repo", "upstream_root", "manifest", "train_view", "data_root",
        "output", "predecessor_state",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.poll_seconds < 30 or args.timeout_hours < 24:
        raise RuntimeError("DCLGAN target successor waiting policy is unsafe")
    state_path = args.output / "operations" / "DCLGAN_TARGET_SUCCESSOR_STATE.json"
    lock = acquire_lock(args.output / "operations" / "DCLGAN_TARGET_SUCCESSOR.lock")
    started = time.time()
    try:
        contract_path, contract = freeze_contract(args)
        while True:
            verify_contract(contract)
            predecessor = (
                read_json(args.predecessor_state)
                if args.predecessor_state.is_file()
                else {}
            )
            decision = predecessor_decision(
                predecessor,
                args.required_predecessor_status,
            )
            waiting_status = {
                "WAIT": "WAITING_FOR_PREDECESSOR",
                "START": "PREDECESSOR_COMPLETE_STARTING_TARGET_GATE",
                "BLOCK": "BLOCKED_PREDECESSOR_OR_SCIENTIFIC_BOUNDARY",
            }[decision]
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status=waiting_status,
                    predecessor_status=predecessor.get("status"),
                    contract=str(contract_path),
                    contract_sha256=file_sha256(contract_path),
                    elapsed_seconds=time.time() - started,
                ),
            )
            if decision == "START":
                break
            if decision == "BLOCK":
                return 2
            if time.time() - started > args.timeout_hours * 3600:
                atomic_json(
                    state_path,
                    state_payload(args, status="BLOCKED_TIMEOUT_WAITING_PREDECESSOR"),
                )
                return 3
            time.sleep(args.poll_seconds)

        log_path = args.output / "logs" / "DCLGAN_TARGET_SUCCESSOR.log"
        for stage, command in (
            ("TARGET_GATE_RUNNING", gate_command(args)),
            ("LONG_SUPERVISOR_RUNNING", long_command(args)),
        ):
            verify_contract(contract)
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status=stage,
                    command=command,
                    updated_unix_time=time.time(),
                ),
            )
            returncode = run_child(command, cwd=args.repo, log_path=log_path)
            if returncode:
                atomic_json(
                    state_path,
                    state_payload(
                        args,
                        status=f"BLOCKED_{stage}_EXIT",
                        child_returncode=returncode,
                        updated_unix_time=time.time(),
                    ),
                )
                return returncode

        supervisor_path = args.output / "gates" / "SUPERVISOR_dclgan.json"
        supervisor = read_json(supervisor_path)
        if supervisor.get("status") != "COMPLETE_E200":
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status="BLOCKED_LONG_SUPERVISOR_NOT_TERMINAL",
                    supervisor_status=supervisor.get("status"),
                ),
            )
            return 4
        atomic_json(
            state_path,
            state_payload(
                args,
                status="COMPLETE_DCLGAN_E200",
                supervisor_status="COMPLETE_E200",
                wall_seconds=time.time() - started,
            ),
        )
        return 0
    finally:
        lock.close()


def main() -> int:
    return run(arguments())


if __name__ == "__main__":
    raise SystemExit(main())
