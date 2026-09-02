"""Durably adjudicate terminal pathology after all target-blind audits.

The successor does not parse the paired metric-binding file until the complete
fixed audit set has passed.  Its output can authorize derivation, never training.
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


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from research.paper_aio.protocol import file_sha256  # noqa: E402
from research.paper_aio.terminal_adjudicate import (  # noqa: E402
    AUDIT_EPOCHS,
    PROBES,
    adjudicate_terminal_pathology,
)


CONTRACT_SCHEMA = "final-unsb-paper-terminal-pathology-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-terminal-pathology-successor-state-v1"
SOURCE_RELATIVES = (
    "operations/paper_aio_terminal_pathology_successor.py",
    "operations/paper_aio_local_terminal_audit_successor.py",
    "research/paper_aio/terminal_adjudicate.py",
    "research/paper_aio/terminal_audit.py",
    "research/paper_aio/protocol.py",
    "research/paper_aio/unified.py",
    "research/local_route1/runtime.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def audit_release(path: Path) -> str:
    if not Path(path).is_file():
        return "WAIT"
    value = _read_json(path)
    if (
        value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        return "BLOCKED"
    status = str(value.get("status", ""))
    if "FAIL" in status or "BLOCK" in status:
        return "BLOCKED"
    if status != "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS":
        return "WAIT"
    expected = {f"{probe_id}:e{epoch}" for probe_id in PROBES for epoch in AUDIT_EPOCHS}
    return "READY" if set(value.get("completed", [])) == expected else "BLOCKED"


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("terminal pathology successor checkout is not frozen")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "repo": str(repo),
        "control_git_commit": commit,
        "control_source_sha256": {
            relative: file_sha256(repo / relative) for relative in SOURCE_RELATIVES
        },
        "audit_successor_state": str(args.audit_successor_state.resolve()),
        "audit_root": str(args.audit_root.resolve()),
        "metric_bindings": str(args.metric_bindings.resolve()),
        "destination": str(args.destination.resolve()),
        "fixed_epochs": list(AUDIT_EPOCHS),
        "fixed_probes": list(PROBES),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "paired_files_parsed_before_all_audits": False,
        "paired_metric_control": False,
        "algorithm_or_module_auto_start": False,
        "confirmation20_opened": False,
    }


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    if _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"] or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("terminal pathology successor checkout moved")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"terminal pathology source changed: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.poll_seconds <= 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise ValueError("timeout must be at least 24 hours")
    contract = _contract(args)
    output = args.output.resolve()
    contract_path = output / "TERMINAL_PATHOLOGY_SUCCESSOR_CONTRACT.json"
    state_path = output / "TERMINAL_PATHOLOGY_SUCCESSOR_STATE.json"
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("terminal pathology successor contract changed")
    else:
        _write_json(contract_path, contract)

    started = time.time()
    while True:
        _verify_control(contract)
        release = audit_release(Path(contract["audit_successor_state"]))
        if release == "BLOCKED":
            raise RuntimeError("target-blind terminal audit release is invalid")
        if release == "READY":
            binding_path = Path(contract["metric_bindings"])
            if binding_path.is_file():
                result = adjudicate_terminal_pathology(
                    audit_root=Path(contract["audit_root"]),
                    metric_bindings=binding_path,
                    destination=Path(contract["destination"]),
                )
                state = {
                    "schema": STATE_SCHEMA,
                    "status": "COMPLETE_POSTHOC_TERMINAL_PATHOLOGY_ADJUDICATION",
                    "pid": os.getpid(),
                    "decision_status": result["status"],
                    "terminal_pathology_confirmed": result[
                        "terminal_pathology_confirmed"
                    ],
                    "decision_sha256": file_sha256(Path(contract["destination"])),
                    "all_target_blind_audits_validated_before_paired_metric_read": True,
                    "performance_values_read_posthoc": True,
                    "paired_metric_control": False,
                    "algorithm_or_module_auto_started": False,
                    "confirmation20_opened": False,
                }
                _write_json(state_path, state)
                return state
            status = "WAITING_FOR_POSTHOC_UNIFIED_METRIC_BINDINGS"
        else:
            status = "WAITING_FOR_COMPLETE_TARGET_BLIND_AUDITS"
        if time.time() - started > contract["timeout_hours"] * 3600:
            raise TimeoutError("terminal pathology successor timed out")
        _write_json(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "status": status,
                "pid": os.getpid(),
                "audit_release": release,
                "metric_binding_exists": Path(contract["metric_bindings"]).is_file(),
                "elapsed_seconds": time.time() - started,
                "paired_files_parsed": False,
                "performance_values_read": False,
                "paired_metric_control": False,
                "algorithm_or_module_auto_started": False,
                "confirmation20_opened": False,
            },
        )
        time.sleep(contract["poll_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--audit-successor-state", type=Path, required=True)
    value.add_argument("--audit-root", type=Path, required=True)
    value.add_argument("--metric-bindings", type=Path, required=True)
    value.add_argument("--destination", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
