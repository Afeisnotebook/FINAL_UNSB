"""Publish fixed audit checkpoints as soon as immutable milestones exist.

This exporter is deliberately separate from the terminal e200 paper exporter.
It exposes only the preregistered e100/e150/e200 checkpoints used by the
target-blind terminal audit.  It never evaluates a model, reads a performance
file, changes training state, or opens confirmation20.
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

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from research.paper_aio.protocol import file_sha256  # noqa: E402
from research.paper_aio.unified import (  # noqa: E402
    EXPORT_SCHEMA,
    export_checkpoint_receipt,
)


CONTRACT_SCHEMA = "final-unsb-paper-incremental-audit-export-contract-v1"
STATE_SCHEMA = "final-unsb-paper-incremental-audit-export-state-v1"
SET_SCHEMA = "final-unsb-paper-incremental-audit-export-set-v1"
AUDIT_EPOCHS = (100, 150, 200)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SOURCE_RELATIVES = (
    "operations/paper_aio_incremental_audit_export.py",
    "research/paper_aio/unified.py",
    "research/paper_aio/protocol.py",
    "research/local_route1/runtime.py",
)

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - deployment is POSIX.
    _fcntl = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    lane = str(args.lane)
    if not _SAFE_ID.fullmatch(lane) or not _SAFE_ID.fullmatch(args.source_host_label):
        raise ValueError("lane and source host label must be safe identifiers")
    if not 30 <= int(args.poll_seconds) <= 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if float(args.timeout_hours) < 24:
        raise ValueError("timeout must be at least 24 hours")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("incremental exporter control checkout is not frozen")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(repo),
        "control_git_commit": commit,
        "control_source_sha256": {
            relative: file_sha256(repo / relative) for relative in SOURCE_RELATIVES
        },
        "source_output": str(args.source_output.resolve()),
        "destination": str(args.destination.resolve()),
        "lane_id": lane,
        "source_host_label": str(args.source_host_label),
        "required_training_git_commit": str(args.required_training_git_commit),
        "required_training_protocol_fingerprint": str(
            args.required_training_protocol_fingerprint
        ),
        "required_manifest_sha256": str(args.required_manifest_sha256),
        "audit_epochs": list(AUDIT_EPOCHS),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "checkpoint_copy_performed": False,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["control_repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
    ):
        raise RuntimeError("incremental exporter control checkout moved")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"incremental exporter source changed: {relative}")


def validate_existing_receipt(
    value: dict[str, Any], *, contract: dict[str, Any], epoch: int,
) -> None:
    if (
        value.get("schema") != EXPORT_SCHEMA
        or value.get("status") != "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT"
        or value.get("lane_id") != contract["lane_id"]
        or int(value.get("epoch", -1)) != int(epoch)
        or int(value.get("updates", -1)) != int(epoch) * 8553
        or value.get("source_host_label") != contract["source_host_label"]
        or value.get("training_git_commit")
        != contract["required_training_git_commit"]
        or value.get("training_protocol_fingerprint")
        != contract["required_training_protocol_fingerprint"]
        or value.get("manifest_sha256") != contract["required_manifest_sha256"]
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"existing incremental export receipt differs at e{epoch}")


def available_exports(contract: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(contract["source_output"]) / "lanes" / contract["lane_id"]
    destination = Path(contract["destination"]) / contract["lane_id"]
    rows: list[dict[str, Any]] = []
    for epoch in contract["audit_epochs"]:
        checkpoint = source / "milestones" / f"e{int(epoch):03d}.pt"
        sidecar = Path(str(checkpoint) + ".json")
        receipt_path = destination / f"e{int(epoch):03d}.export.json"
        if checkpoint.is_file() != sidecar.is_file():
            continue
        if not checkpoint.is_file():
            continue
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
            validate_existing_receipt(receipt, contract=contract, epoch=int(epoch))
        else:
            receipt = export_checkpoint_receipt(
                checkpoint=checkpoint,
                sidecar=sidecar,
                lane_id=contract["lane_id"],
                epoch=int(epoch),
                host_label=contract["source_host_label"],
                destination=receipt_path,
            )
            validate_existing_receipt(receipt, contract=contract, epoch=int(epoch))
        rows.append({
            "epoch": int(epoch),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "scientific_state_sha256": receipt["scientific_state_sha256"],
        })
    return rows


def export_set(contract: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [int(row["epoch"]) for row in rows]
    complete = available == list(contract["audit_epochs"])
    return {
        "schema": SET_SCHEMA,
        "status": (
            "COMPLETE_INCREMENTAL_AUDIT_EXPORT_SET"
            if complete else "PARTIAL_INCREMENTAL_AUDIT_EXPORT_SET"
        ),
        "lane_id": contract["lane_id"],
        "source_host_label": contract["source_host_label"],
        "required_epochs": list(contract["audit_epochs"]),
        "available_epochs": available,
        "exports": rows,
        "checkpoint_copy_performed": False,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = _contract(args)
    source_output = Path(contract["source_output"])
    operations = source_output / "operations"
    contract_path = operations / f"INCREMENTAL_AUDIT_EXPORT_{contract['lane_id']}_CONTRACT.json"
    state_path = operations / f"INCREMENTAL_AUDIT_EXPORT_{contract['lane_id']}_STATE.json"
    lock_path = operations / f"INCREMENTAL_AUDIT_EXPORT_{contract['lane_id']}.lock"
    set_path = Path(contract["destination"]) / contract["lane_id"] / "INCREMENTAL_AUDIT_EXPORT_SET.json"
    operations.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("incremental export contract changed")
    else:
        _write_json(contract_path, contract)
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        if _fcntl is None:
            raise RuntimeError("incremental source exporter requires POSIX file locking")
        _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        while True:
            _verify_control(contract)
            rows = available_exports(contract)
            current = export_set(contract, rows)
            _write_json(set_path, current)
            _write_json(state_path, {
                "schema": STATE_SCHEMA,
                "status": current["status"],
                "pid": os.getpid(),
                "lane_id": contract["lane_id"],
                "available_epochs": current["available_epochs"],
                "required_epochs": current["required_epochs"],
                "set": str(set_path.resolve()),
                "set_sha256": file_sha256(set_path),
                "elapsed_seconds": time.time() - started,
                "performance_values_read": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            })
            if current["status"] == "COMPLETE_INCREMENTAL_AUDIT_EXPORT_SET":
                return current
            supervisor_path = (
                source_output / "gates" / f"SUPERVISOR_{contract['lane_id']}.json"
            )
            if supervisor_path.is_file():
                status = str(_read_json(supervisor_path).get("status", ""))
                if status.startswith(("BLOCKED", "FAIL")):
                    raise RuntimeError(f"source supervisor is blocked: {status}")
                if status == "COMPLETE_E200":
                    raise RuntimeError("source completed e200 without all audit milestones")
            if time.time() - started > contract["timeout_hours"] * 3600:
                raise TimeoutError("incremental audit exporter timed out")
            time.sleep(contract["poll_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--source-output", type=Path, required=True)
    value.add_argument("--destination", type=Path, required=True)
    value.add_argument("--lane", required=True)
    value.add_argument("--source-host-label", required=True)
    value.add_argument("--required-training-git-commit", required=True)
    value.add_argument("--required-training-protocol-fingerprint", required=True)
    value.add_argument("--required-manifest-sha256", required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"incremental audit exporter failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
