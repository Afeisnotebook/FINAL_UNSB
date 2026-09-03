"""Durably continue an authorized DCLGAN lane from update 1000 to e200.

The target-host authorization is deliberately host-bound.  This supervisor
never reads evaluation values; it only verifies frozen identities, launches
the source-bound adapter with exact resume, and retries engineering exits.
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

# This file is launched by absolute path from a detached supervisor.  Make the
# frozen checkout importable without depending on the caller's PYTHONPATH.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from operations import paper_aio_dclgan_adapter as adapter  # noqa: E402


SCHEMA = "final-unsb-paper-dclgan-long-supervisor-v1"
TERMINAL_UPDATES = 1_710_600
BLOCKED_STATUS = "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE"


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


def training_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.repo / "operations" / "paper_aio_dclgan_adapter.py"),
        "--upstream-root",
        str(args.upstream_root),
        "--manifest",
        str(args.manifest),
        "--train-view",
        str(args.train_view),
        "--output",
        str(args.output),
        "--stage",
        "train",
        "--gpu",
        str(args.gpu),
        "--stop-after-updates",
        str(TERMINAL_UPDATES),
        "--resume",
    ]


def validate_authorization(args: argparse.Namespace) -> dict[str, Any]:
    commit, dirty = repo_identity(args.repo)
    if commit != args.required_git_commit or dirty:
        raise RuntimeError("DCLGAN long supervisor checkout is not frozen and clean")
    source = adapter.verify_upstream(args.upstream_root)
    fingerprint = adapter.adapter_fingerprint(
        upstream_receipt=source,
        manifest_path=args.manifest,
    )
    if fingerprint != args.required_adapter_fingerprint:
        raise RuntimeError("DCLGAN adapter fingerprint differs from the frozen contract")
    path = args.output / "gates" / "DCLGAN_LONG_TRAINING_AUTHORIZATION.json"
    if not path.is_file():
        raise RuntimeError("DCLGAN target-host authorization is missing")
    authorization = read_json(path)
    expected_host = adapter.runtime_host_identity(args.gpu)
    failures = []
    if authorization.get("status") != "PASS_LONG_TRAINING_AUTHORIZED":
        failures.append("authorization status")
    if authorization.get("adapter_git_commit") != commit:
        failures.append("adapter commit")
    if authorization.get("adapter_fingerprint") != fingerprint:
        failures.append("adapter fingerprint")
    if authorization.get("upstream_commit") != source.get("commit"):
        failures.append("upstream commit")
    if authorization.get("runtime_host") != expected_host:
        failures.append("runtime host")
    if authorization.get("confirmation20_opened") is not False:
        failures.append("confirmation boundary")
    if failures:
        raise RuntimeError(
            "DCLGAN target-host authorization is stale: " + ", ".join(failures)
        )
    return authorization


def terminal_decision(output: Path) -> dict[str, Any]:
    lane = output / "lanes" / adapter.LANE_ID
    state_path = lane / "RUN_STATE.json"
    latest = lane / "full_state_latest.pt"
    latest_sidecar = Path(str(latest) + ".json")
    if not state_path.is_file():
        return {"complete": False, "reason": "run_state_missing"}
    state = read_json(state_path)
    if (
        state.get("status") != "COMPLETE_E200"
        or int(state.get("final_updates", -1)) != TERMINAL_UPDATES
        or float(state.get("final_data_epoch", -1.0)) != 200.0
    ):
        return {
            "complete": False,
            "reason": "run_state_not_terminal",
            "run_status": state.get("status"),
            "final_updates": state.get("final_updates"),
        }
    if not latest.is_file() or not latest_sidecar.is_file():
        raise RuntimeError("DCLGAN terminal latest checkpoint is incomplete")
    sidecar = read_json(latest_sidecar)
    if (
        sidecar.get("schema") != adapter.FULL_STATE_SCHEMA
        or sidecar.get("lane_id") != adapter.LANE_ID
        or int(sidecar.get("step", -1)) != TERMINAL_UPDATES
        or sidecar.get("full_state_sha256") != adapter.file_sha256(latest)
        or sidecar.get("metadata", {}).get("confirmation20_opened") is not False
    ):
        raise RuntimeError("DCLGAN terminal latest checkpoint failed identity checks")
    e200 = lane / "milestones" / "e200.pt"
    e200_sidecar = Path(str(e200) + ".json")
    if not e200.is_file() or not e200_sidecar.is_file():
        raise RuntimeError("DCLGAN e200 milestone is missing")
    e200_metadata = read_json(e200_sidecar)
    if (
        e200_metadata.get("full_state_sha256") != adapter.file_sha256(e200)
        or int(e200_metadata.get("step", -1)) != TERMINAL_UPDATES
    ):
        raise RuntimeError("DCLGAN e200 milestone hash differs")
    return {
        "complete": True,
        "final_updates": TERMINAL_UPDATES,
        "latest_checkpoint_sha256": sidecar["full_state_sha256"],
        "e200_checkpoint_sha256": e200_metadata["full_state_sha256"],
        "scientific_state_sha256": e200_metadata["scientific_state_sha256"],
    }


def state_payload(args: argparse.Namespace, *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "lane_id": adapter.LANE_ID,
        "required_git_commit": args.required_git_commit,
        "required_adapter_fingerprint": args.required_adapter_fingerprint,
        "output": str(args.output),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--required-git-commit", required=True)
    parser.add_argument("--required-adapter-fingerprint", required=True)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-seconds", type=int, default=60)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    for name in ("repo", "upstream_root", "manifest", "train_view", "output"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.max_attempts < 1 or args.retry_seconds < 10:
        raise RuntimeError("DCLGAN long supervisor retry policy is unsafe")
    state_path = args.output / "gates" / "SUPERVISOR_dclgan.json"
    lock = acquire_lock(args.output / "operations" / "DCLGAN_LONG_SUPERVISOR.lock")
    started = time.time()
    try:
        validate_authorization(args)
        terminal = terminal_decision(args.output)
        if terminal["complete"]:
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status="COMPLETE_E200",
                    wall_seconds=0.0,
                    terminal=terminal,
                ),
            )
            return 0
        command = training_command(args)
        log_path = args.output / "logs" / "DCLGAN_LONG_SUPERVISOR.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.max_attempts + 1):
            validate_authorization(args)
            with log_path.open("a", encoding="utf-8") as log:
                child = subprocess.Popen(
                    command,
                    cwd=args.repo,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                atomic_json(
                    state_path,
                    state_payload(
                        args,
                        status="CHILD_RUNNING",
                        child_pid=child.pid,
                        attempt=attempt,
                        started_unix_time=started,
                        updated_unix_time=time.time(),
                    ),
                )
                returncode = child.wait()
            if returncode == 0:
                terminal = terminal_decision(args.output)
                if terminal["complete"]:
                    atomic_json(
                        state_path,
                        state_payload(
                            args,
                            status="COMPLETE_E200",
                            attempt=attempt,
                            wall_seconds=time.time() - started,
                            terminal=terminal,
                        ),
                    )
                    return 0
            status = "RETRY_WAIT" if attempt < args.max_attempts else BLOCKED_STATUS
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status=status,
                    attempt=attempt,
                    child_returncode=returncode,
                    terminal=terminal_decision(args.output),
                    updated_unix_time=time.time(),
                ),
            )
            if attempt == args.max_attempts:
                return returncode or 2
            time.sleep(args.retry_seconds)
        return 2
    finally:
        lock.close()


def main() -> int:
    return run(arguments())


if __name__ == "__main__":
    raise SystemExit(main())
