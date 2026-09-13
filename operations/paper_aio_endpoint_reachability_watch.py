"""Durably wait for a paper-compute SSH endpoint to become reachable.

This is an engineering-only interlock.  It performs a TCP connection probe,
records no credential, does not authenticate, does not collect GPU identity,
and cannot launch training.  A reachable result is deliberately
``REVIEW_REQUIRED``: the separate physical-GPU identity gate must run before
the endpoint can enter the compute DAG.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


CONTRACT_SCHEMA = "final-unsb-paper-endpoint-reachability-contract-v1"
STATE_SCHEMA = "final-unsb-paper-endpoint-reachability-state-v1"
FINAL_STATUS = "REACHABLE_REVIEW_REQUIRED"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def validate_control_repo(repo: Path, required_commit: str) -> None:
    if _git(repo, "rev-parse", "HEAD") != required_commit:
        raise RuntimeError("endpoint watch control checkout is at the wrong commit")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("endpoint watch control checkout is dirty")


def probe_tcp(
    host: str,
    port: int,
    timeout_seconds: float,
    *,
    connector: Callable[..., Any] = socket.create_connection,
) -> dict[str, Any]:
    try:
        addresses = sorted({row[4][0] for row in socket.getaddrinfo(host, port)})
    except OSError:
        addresses = []
    connection = None
    try:
        connection = connector((host, port), timeout=timeout_seconds)
        peer = connection.getpeername()
        return {
            "reachable": True,
            "resolved_addresses": addresses,
            "peer_address": str(peer[0]),
            "peer_port": int(peer[1]),
            "error_type": None,
        }
    except OSError as error:
        return {
            "reachable": False,
            "resolved_addresses": addresses,
            "peer_address": None,
            "peer_port": None,
            "error_type": type(error).__name__,
        }
    finally:
        if connection is not None:
            connection.close()


def make_state(
    *,
    host: str,
    port: int,
    attempt: int,
    result: dict[str, Any],
    started_unix_time: float,
    observed_unix_time: float,
) -> dict[str, Any]:
    reachable = result.get("reachable") is True
    return {
        "schema": STATE_SCHEMA,
        "status": FINAL_STATUS if reachable else "WAITING_FOR_TCP_REACHABILITY",
        "pid": os.getpid(),
        "host": host,
        "port": int(port),
        "attempt": int(attempt),
        "started_unix_time": float(started_unix_time),
        "observed_unix_time": float(observed_unix_time),
        "reachable": reachable,
        "resolved_addresses": list(result.get("resolved_addresses", [])),
        "peer_address": result.get("peer_address"),
        "peer_port": result.get("peer_port"),
        "error_type": result.get("error_type"),
        "authentication_attempted": False,
        "gpu_identity_collected": False,
        "identity_gate_required": True,
        "long_training_launched": False,
        "credentials_recorded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--required-control-git-commit", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=7.0)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    repo = args.repo.resolve()
    output = args.output.resolve()
    state_path = output / "ENDPOINT_REACHABILITY_STATE.json"
    contract_path = output / "ENDPOINT_REACHABILITY_CONTRACT.json"
    validate_control_repo(repo, str(args.required_control_git_commit))
    if not 1 <= args.port <= 65535:
        raise RuntimeError("endpoint port is outside the TCP range")
    if args.poll_seconds <= 0 or args.connect_timeout_seconds <= 0:
        raise RuntimeError("endpoint watch timing values must be positive")

    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "host": str(args.host),
        "port": int(args.port),
        "repo": str(repo),
        "required_control_git_commit": str(args.required_control_git_commit),
        "poll_seconds": float(args.poll_seconds),
        "connect_timeout_seconds": float(args.connect_timeout_seconds),
        "timeout_hours": float(args.timeout_hours),
        "final_status": FINAL_STATUS,
        "action_on_reachable": "STOP_AND_REQUIRE_GPU_IDENTITY_REVIEW",
        "credentials_recorded": False,
        "training_launch_capability": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _atomic_json(contract_path, contract)

    started = time.time()
    deadline = started + float(args.timeout_hours) * 3600.0
    attempt = 0
    while True:
        attempt += 1
        observed = time.time()
        result = probe_tcp(
            str(args.host), int(args.port), float(args.connect_timeout_seconds)
        )
        state = make_state(
            host=str(args.host),
            port=int(args.port),
            attempt=attempt,
            result=result,
            started_unix_time=started,
            observed_unix_time=observed,
        )
        _atomic_json(state_path, state)
        if state["status"] == FINAL_STATUS:
            return 0
        if time.time() >= deadline:
            state["status"] = "TIMEOUT_WAITING_FOR_TCP_REACHABILITY"
            state["observed_unix_time"] = time.time()
            _atomic_json(state_path, state)
            return 2
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
