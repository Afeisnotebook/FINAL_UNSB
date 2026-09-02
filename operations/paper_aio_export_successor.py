"""Durably export source-bound receipts after one frozen paper lane reaches e200.

This process is control-plane only. It never evaluates a checkpoint, reads a
performance value, resumes training, copies model weights across hosts, or
opens confirmation20.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# This executable is launched by absolute path from immutable control
# worktrees.  Without this bootstrap Python exposes only ``operations/`` on
# sys.path, making the public script depend on a caller-supplied PYTHONPATH.
# Keep the control-plane executable self-contained; this does not participate
# in a model transition or alter the scientific paper protocol fingerprint.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from research.paper_aio.protocol import ROOT, file_sha256, protocol_fingerprint
from research.paper_aio.unified import UNIFIED_EPOCHS, export_checkpoint_receipt


SCHEMA = "final-unsb-paper-export-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-export-successor-state-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SOURCE_RELATIVES = (
    "operations/paper_aio_export_successor.py",
    "research/paper_aio/unified.py",
    "research/paper_aio/protocol.py",
)

try:  # Linux training hosts.
    import fcntl as _fcntl
except ImportError:  # Windows CPU tests and local orchestration.
    _fcntl = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _dirty(repo: Path) -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()


def source_state_decision(source_output: Path, lane_id: str) -> dict[str, Any]:
    lane_root = Path(source_output) / "lanes" / lane_id
    state_path = lane_root / "RUN_STATE.json"
    supervisor_path = Path(source_output) / "gates" / f"SUPERVISOR_{lane_id}.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    supervisor = _read_json(supervisor_path) if supervisor_path.is_file() else {}
    if supervisor.get("status") == "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE":
        return {
            "status": "BLOCKED_SOURCE_LANE_ENGINEERING_FAILURE",
            "run_state": state.get("status"),
            "supervisor_status": supervisor.get("status"),
        }
    if (
        state.get("status") == "COMPLETE_E200"
        and int(state.get("final_updates", -1)) == 1_710_600
        and float(state.get("final_data_epoch", -1)) == 200.0
        and state.get("confirmation20_opened") is False
    ):
        return {
            "status": "READY_COMPLETE_E200",
            "run_state": state.get("status"),
            "supervisor_status": supervisor.get("status"),
        }
    return {
        "status": "WAITING_FOR_COMPLETE_E200",
        "run_state": state.get("status"),
        "supervisor_status": supervisor.get("status"),
    }


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if repo != ROOT.resolve():
        raise RuntimeError("paper export successor must execute from its declared control repo")
    lane_id = str(args.lane)
    if not _SAFE_ID.fullmatch(lane_id):
        raise ValueError(f"unsafe paper export lane: {lane_id!r}")
    if _dirty(repo):
        raise RuntimeError("paper export successor control checkout must be clean")
    commit = _head(repo)
    sources = {
        relative: file_sha256(repo / relative) for relative in SOURCE_RELATIVES
    }
    return {
        "schema": SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(repo),
        "control_git_commit": commit,
        "control_protocol_fingerprint": protocol_fingerprint(),
        "control_source_sha256": sources,
        "source_output": str(args.source_output.resolve()),
        "destination": str(args.destination.resolve()),
        "lane_id": lane_id,
        "source_host_label": str(args.source_host_label),
        "required_training_git_commit": str(args.required_training_git_commit),
        "required_training_protocol_fingerprint": str(
            args.required_training_protocol_fingerprint
        ),
        "epochs": list(UNIFIED_EPOCHS),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "performance_values_available_to_scheduling": False,
        "paired_metric_control": False,
        "checkpoint_copy_performed": False,
        "confirmation20_opened": False,
    }


def _freeze_contract(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    proposed = _contract(args)
    path = (
        args.source_output.resolve() / "operations"
        / f"EXPORT_SUCCESSOR_{args.lane}_CONTRACT.json"
    )
    if path.is_file():
        existing = _read_json(path)
        for key, value in proposed.items():
            if existing.get(key) != value:
                raise RuntimeError(f"paper export successor contract changed for {key}")
        return path, existing
    _write_json(path, proposed)
    return path, proposed


def _verify_control_checkout(contract: dict[str, Any]) -> None:
    repo = Path(contract["control_repo"])
    if _head(repo) != contract["control_git_commit"] or _dirty(repo):
        raise RuntimeError("paper export successor control checkout changed")
    if protocol_fingerprint() != contract["control_protocol_fingerprint"]:
        raise RuntimeError("paper export successor protocol fingerprint changed")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"paper export successor source changed: {relative}")


def _export_all(contract: dict[str, Any]) -> dict[str, Any]:
    source_output = Path(contract["source_output"])
    destination = Path(contract["destination"])
    lane_id = contract["lane_id"]
    exports = []
    for epoch in contract["epochs"]:
        checkpoint = (
            source_output / "lanes" / lane_id / "milestones" / f"e{int(epoch):03d}.pt"
        )
        sidecar = Path(str(checkpoint) + ".json")
        receipt_path = destination / lane_id / f"e{int(epoch):03d}.export.json"
        receipt = export_checkpoint_receipt(
            checkpoint=checkpoint, sidecar=sidecar, lane_id=lane_id,
            epoch=int(epoch), host_label=contract["source_host_label"],
            destination=receipt_path,
        )
        if receipt.get("training_git_commit") != contract["required_training_git_commit"]:
            raise RuntimeError(f"{lane_id} e{epoch} training commit differs")
        if (
            receipt.get("training_protocol_fingerprint")
            != contract["required_training_protocol_fingerprint"]
        ):
            raise RuntimeError(f"{lane_id} e{epoch} training protocol differs")
        exports.append({
            "epoch": int(epoch),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "scientific_state_sha256": receipt["scientific_state_sha256"],
        })
    result = {
        "schema": "final-unsb-paper-source-export-set-v1",
        "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
        "lane_id": lane_id,
        "source_host_label": contract["source_host_label"],
        "epochs": list(contract["epochs"]),
        "exports": exports,
        "performance_values_read": False,
        "checkpoint_copy_performed": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json(destination / lane_id / "EXPORT_SET.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.poll_seconds) < 30 or int(args.poll_seconds) > 600:
        raise ValueError("paper export poll interval must be in [30,600] seconds")
    if float(args.timeout_hours) < 24:
        raise ValueError("paper export timeout must allow at least 24 hours")
    contract_path, contract = _freeze_contract(args)
    source_output = Path(contract["source_output"])
    state_path = (
        source_output / "operations" / f"EXPORT_SUCCESSOR_{args.lane}_STATE.json"
    )
    lock_path = source_output / "operations" / f"EXPORT_SUCCESSOR_{args.lane}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            if _fcntl is None:
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise RuntimeError("paper export successor is already running") from error
        while True:
            _verify_control_checkout(contract)
            decision = source_state_decision(source_output, contract["lane_id"])
            _write_json(state_path, {
                "schema": STATE_SCHEMA,
                "status": decision["status"],
                "pid": os.getpid(),
                "lane_id": contract["lane_id"],
                "contract": str(contract_path),
                "contract_sha256": file_sha256(contract_path),
                "elapsed_seconds": time.time() - started,
                "performance_values_read": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            })
            if decision["status"] == "READY_COMPLETE_E200":
                result = _export_all(contract)
                _write_json(state_path, {
                    "schema": STATE_SCHEMA,
                    "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
                    "pid": os.getpid(),
                    "lane_id": contract["lane_id"],
                    "result": result,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                })
                return result
            if decision["status"].startswith("BLOCKED"):
                raise RuntimeError(f"source lane blocked: {decision}")
            if time.time() - started > float(contract["timeout_hours"]) * 3600:
                raise TimeoutError("paper export successor exceeded its frozen timeout")
            time.sleep(int(contract["poll_seconds"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--required-training-git-commit", required=True)
    parser.add_argument("--required-training-protocol-fingerprint", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=480)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
