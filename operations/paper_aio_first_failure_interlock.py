"""Fail-stop a recovery supervisor before it can blindly replay a new failure.

This interlock is intentionally outside the scientific training process.  It
observes only the metric-blind supervisor state and acts only after the first
post-repair child failure.  The recovery guard is stopped first, followed by
the lane supervisor, so neither layer can launch another child while the new
failure is being audited.  Healthy training is never signalled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time
from typing import Any


SCHEMA = "final-unsb-paper-first-failure-interlock-v1"
FAILURE_STATES = {
    "WAITING_TO_EXACT_RESUME",
    "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def should_interlock(state: dict[str, Any]) -> bool:
    return (
        str(state.get("status")) in FAILURE_STATES
        or int(state.get("consecutive_failures", 0)) > 0
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cmdline(pid: int) -> str:
    return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    )


def _assert_process(pid: int, fragments: list[str], role: str) -> None:
    if not _pid_alive(pid):
        raise RuntimeError(f"{role} PID is absent: {pid}")
    command = _cmdline(pid)
    missing = [fragment for fragment in fragments if fragment not in command]
    if missing:
        raise RuntimeError(f"{role} PID identity mismatch: missing {missing!r}")


def _stopped(pid: int) -> bool:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    return any(line.startswith("State:\tT") for line in status.splitlines())


def monitor(args: argparse.Namespace) -> int:
    if os.name != "posix":
        raise RuntimeError("first-failure interlock is supported only on Linux")
    state_path = args.supervisor_state.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "FIRST_FAILURE_INTERLOCK_CONTRACT.json"
    state_output = output / "FIRST_FAILURE_INTERLOCK_STATE.json"

    _assert_process(
        args.supervisor_pid,
        list(args.required_supervisor_command_fragment),
        "supervisor",
    )
    _assert_process(
        args.guard_pid, list(args.required_guard_command_fragment), "guard"
    )
    initial = _read(state_path)
    if (
        initial.get("schema") != "final-unsb-paper-supervisor-v1"
        or initial.get("lane_id") != args.lane_id
        or should_interlock(initial)
    ):
        raise RuntimeError("interlock requires a healthy zero-failure initial state")

    contract = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "lane_id": args.lane_id,
        "supervisor_pid": args.supervisor_pid,
        "guard_pid": args.guard_pid,
        "supervisor_state": str(state_path),
        "required_supervisor_command_fragments": list(
            args.required_supervisor_command_fragment
        ),
        "required_guard_command_fragments": list(
            args.required_guard_command_fragment
        ),
        "action": "SIGSTOP_GUARD_THEN_SUPERVISOR_AFTER_FIRST_FAILURE_ONLY",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    if contract_path.exists() and _read(contract_path) != contract:
        raise RuntimeError("refusing to replace a different interlock contract")
    _atomic(contract_path, contract)

    started = time.monotonic()
    while time.monotonic() - started < float(args.timeout_hours) * 3600:
        state = _read(state_path)
        if state.get("lane_id") != args.lane_id:
            raise RuntimeError("supervisor lane identity changed")
        if state.get("status") == "COMPLETE_E200":
            _atomic(state_output, {
                **contract,
                "status": "COMPLETE_E200_NO_INTERLOCK_REQUIRED",
                "captured_unix_time": time.time(),
            })
            return 0
        if should_interlock(state):
            # Stop the outer guard first.  Otherwise it could replace a lane
            # supervisor that we deliberately freeze for failure forensics.
            _assert_process(
                args.guard_pid,
                list(args.required_guard_command_fragment),
                "guard",
            )
            os.kill(args.guard_pid, signal.SIGSTOP)
            _assert_process(
                args.supervisor_pid,
                list(args.required_supervisor_command_fragment),
                "supervisor",
            )
            os.kill(args.supervisor_pid, signal.SIGSTOP)
            time.sleep(0.25)
            result = {
                **contract,
                "status": "FIRST_FAILURE_INTERCEPTED_SUPERVISORS_STOPPED",
                "captured_unix_time": time.time(),
                "observed_supervisor_status": state.get("status"),
                "observed_consecutive_failures": int(
                    state.get("consecutive_failures", 0)
                ),
                "guard_stopped": _stopped(args.guard_pid),
                "supervisor_stopped": _stopped(args.supervisor_pid),
            }
            _atomic(state_output, result)
            return 2
        if not _pid_alive(args.supervisor_pid) or not _pid_alive(args.guard_pid):
            _atomic(state_output, {
                **contract,
                "status": "CONTROL_PID_MISSING_WITHOUT_RECORDED_FAILURE",
                "captured_unix_time": time.time(),
                "supervisor_alive": _pid_alive(args.supervisor_pid),
                "guard_alive": _pid_alive(args.guard_pid),
            })
            return 3
        _atomic(state_output, {
            **contract,
            "status": "MONITORING_ZERO_FAILURE_RECOVERY",
            "captured_unix_time": time.time(),
            "observed_supervisor_status": state.get("status"),
            "observed_consecutive_failures": int(
                state.get("consecutive_failures", 0)
            ),
        })
        time.sleep(max(0.1, float(args.poll_seconds)))
    _atomic(state_output, {
        **contract,
        "status": "TIMEOUT_WITHOUT_FAILURE_OR_E200",
        "captured_unix_time": time.time(),
    })
    return 4


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--supervisor-state", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--lane-id", required=True)
    value.add_argument("--supervisor-pid", type=int, required=True)
    value.add_argument("--guard-pid", type=int, required=True)
    value.add_argument(
        "--required-supervisor-command-fragment", action="append", default=[]
    )
    value.add_argument("--required-guard-command-fragment", action="append", default=[])
    value.add_argument("--poll-seconds", type=float, default=1.0)
    value.add_argument("--timeout-hours", type=float, default=24.0)
    return value


if __name__ == "__main__":
    raise SystemExit(monitor(parser().parse_args()))
