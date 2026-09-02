"""Incrementally relay fixed, source-bound terminal-audit checkpoints.

The relay accepts only the preregistered e100/e150/e200 source export set.  It
publishes a partial verified import receipt after each newly available epoch so
the local target-blind audit can overlap the remaining long training.  It does
not expose metrics, modify a source checkpoint, or authorize training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from operations.paper_aio_export_relay import (
    SourceExportNotReady,
    TransientRelayNetwork,
    _bytes_sha256,
    _connect,
    _download_verified,
    _read_json_bytes,
    _remote_path,
    _sftp_read,
    _write_bytes,
    _write_json,
    file_sha256,
    validate_export_receipt,
)


CONTRACT_SCHEMA = "final-unsb-paper-incremental-audit-relay-contract-v1"
STATE_SCHEMA = "final-unsb-paper-incremental-audit-relay-state-v1"
SOURCE_SET_SCHEMA = "final-unsb-paper-incremental-audit-export-set-v1"
IMPORT_LANE_SCHEMA = "final-unsb-paper-incremental-audit-imported-lane-v1"
IMPORT_SET_SCHEMA = "final-unsb-paper-incremental-audit-import-set-v1"
AUDIT_EPOCHS = (100, 150, 200)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

try:
    import fcntl as _fcntl
except ImportError:  # Windows local coordinator.
    _fcntl = None


class IncrementalImportNotReady(Exception):
    """A partial lane receipt is not yet atomically bound by its relay set."""


def _inside(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes incremental import root: {path}") from error
    return path


def incremental_import_lane_path(
    import_root: Path, lane_id: str, host_label: str,
) -> Path:
    return (
        Path(import_root).resolve() / "sources" / host_label / lane_id
        / "INCREMENTAL_IMPORT_LANE.json"
    )


def validate_source_set(
    value: dict[str, Any], *, lane_id: str, source_host_label: str,
) -> list[dict[str, Any]]:
    required = value.get("required_epochs")
    available = value.get("available_epochs")
    rows = value.get("exports")
    if (
        value.get("schema") != SOURCE_SET_SCHEMA
        or value.get("status") not in {
            "PARTIAL_INCREMENTAL_AUDIT_EXPORT_SET",
            "COMPLETE_INCREMENTAL_AUDIT_EXPORT_SET",
        }
        or value.get("lane_id") != lane_id
        or value.get("source_host_label") != source_host_label
        or required != list(AUDIT_EPOCHS)
        or not isinstance(available, list)
        or not isinstance(rows, list)
        or value.get("checkpoint_copy_performed") is not False
        or value.get("source_checkpoint_mutation") is not False
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("invalid incremental source export set")
    epochs = [int(epoch) for epoch in available]
    if epochs != sorted(set(epochs)) or any(epoch not in AUDIT_EPOCHS for epoch in epochs):
        raise RuntimeError("incremental source epochs are invalid")
    if len(rows) != len(epochs):
        raise RuntimeError("incremental source row count differs")
    by_epoch: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid incremental source row")
        epoch = int(row.get("epoch", -1))
        if epoch in by_epoch or epoch not in epochs:
            raise RuntimeError("duplicate or unavailable incremental source epoch")
        if not isinstance(row.get("receipt_sha256"), str):
            raise RuntimeError("incremental source row lacks receipt hash")
        _remote_path(str(row.get("receipt", "")), "incremental receipt")
        by_epoch[epoch] = row
    if sorted(by_epoch) != epochs:
        raise RuntimeError("incremental source rows differ from available epochs")
    complete = epochs == list(AUDIT_EPOCHS)
    if complete != (value["status"] == "COMPLETE_INCREMENTAL_AUDIT_EXPORT_SET"):
        raise RuntimeError("incremental source completion status differs")
    return [by_epoch[epoch] for epoch in epochs]


def _matching_incremental_sets(
    import_root: Path, lane_id: str, host_label: str, lane_path: Path,
) -> list[dict[str, Any]]:
    root = Path(import_root).resolve()
    lane_path = _inside(lane_path, root, "incremental lane receipt")
    lane_sha = file_sha256(lane_path)
    matches = []
    for path in sorted((root / "operations").glob("INCREMENTAL_IMPORT_SET_*.json")):
        value = _read_json_bytes(path.read_bytes(), str(path))
        advertised = Path(str(value.get("lane_import_receipt", "")))
        try:
            advertised = _inside(advertised, root, "advertised incremental lane")
        except RuntimeError:
            continue
        if (
            value.get("schema") == IMPORT_SET_SCHEMA
            and value.get("status") in {
                "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
                "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
            }
            and value.get("source_host_label") == host_label
            and value.get("lane_id") == lane_id
            and value.get("required_epochs") == list(AUDIT_EPOCHS)
            and advertised == lane_path
            and value.get("lane_import_receipt_sha256") == lane_sha
            and value.get("checkpoint_copy_performed") is True
            and value.get("source_checkpoint_mutation") is False
            and value.get("performance_values_read") is False
            and value.get("paired_metric_control") is False
            and value.get("confirmation20_opened") is False
        ):
            matches.append({"path": path, "sha256": file_sha256(path)})
    return matches


def validate_incremental_import_lane(
    path: Path, *, import_root: Path, lane_id: str, host_label: str,
) -> list[dict[str, Any]]:
    root = Path(import_root).resolve()
    path = _inside(path, root, "incremental lane receipt")
    value = _read_json_bytes(path.read_bytes(), str(path))
    memberships = _matching_incremental_sets(root, lane_id, host_label, path)
    if not memberships:
        raise IncrementalImportNotReady(
            "incremental lane lacks a verified relay-set binding"
        )
    available = value.get("available_epochs")
    rows = value.get("imports")
    if (
        value.get("schema") != IMPORT_LANE_SCHEMA
        or value.get("status") not in {
            "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
            "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
        }
        or value.get("source_host_label") != host_label
        or value.get("lane_id") != lane_id
        or value.get("required_epochs") != list(AUDIT_EPOCHS)
        or not isinstance(available, list)
        or not isinstance(rows, list)
        or value.get("checkpoint_copy_performed") is not True
        or value.get("source_checkpoint_mutation") is not False
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("invalid incremental imported lane")
    epochs = [int(epoch) for epoch in available]
    if epochs != sorted(set(epochs)) or any(epoch not in AUDIT_EPOCHS for epoch in epochs):
        raise RuntimeError("incremental imported epochs are invalid")
    if len(rows) != len(epochs):
        raise RuntimeError("incremental imported row count differs")
    by_epoch: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid incremental imported row")
        epoch = int(row.get("epoch", -1))
        if epoch in by_epoch or epoch not in epochs:
            raise RuntimeError("duplicate incremental imported epoch")
        receipt = _inside(Path(str(row.get("export_receipt", ""))), root, "export receipt")
        checkpoint = _inside(Path(str(row.get("checkpoint", ""))), root, "checkpoint")
        sidecar = _inside(Path(str(row.get("sidecar", ""))), root, "sidecar")
        for item, key in (
            (receipt, "export_receipt_sha256"),
            (checkpoint, "checkpoint_sha256"),
            (sidecar, "sidecar_sha256"),
        ):
            if not item.is_file() or file_sha256(item) != row.get(key):
                raise RuntimeError(f"incremental imported file hash differs: {item}")
        export = _read_json_bytes(receipt.read_bytes(), str(receipt))
        validate_export_receipt(
            export, lane_id=lane_id, epoch=epoch, source_host_label=host_label,
        )
        if (
            export.get("training_git_commit") != value.get("training_git_commit")
            or export.get("training_protocol_fingerprint")
            != value.get("training_protocol_fingerprint")
            or export.get("manifest_sha256") != value.get("manifest_sha256")
        ):
            raise RuntimeError("incremental import training identity differs")
        by_epoch[epoch] = {
            "epoch": epoch,
            "export_receipt": receipt,
            "checkpoint": checkpoint,
            "source_host_label": host_label,
            "export_receipt_sha256": row["export_receipt_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "import_set_receipts": memberships,
        }
    complete = epochs == list(AUDIT_EPOCHS)
    if complete != (value["status"] == "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT"):
        raise RuntimeError("incremental import completion status differs")
    return [by_epoch[epoch] for epoch in epochs]


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    lane = str(args.lane)
    relay_id = str(args.relay_id)
    if not _SAFE_ID.fullmatch(lane) or not _SAFE_ID.fullmatch(relay_id):
        raise ValueError("relay and lane identifiers must be safe")
    if not _SAFE_ID.fullmatch(args.source_host_label):
        raise ValueError("source host label must be safe")
    if not str(args.password_env).startswith("FINAL_UNSB_"):
        raise ValueError("password environment must use FINAL_UNSB_ prefix")
    if not str(args.expected_host_key_sha256).startswith("SHA256:"):
        raise ValueError("relay requires a pinned SSH host key")
    if not 30 <= int(args.poll_seconds) <= 600 or float(args.timeout_hours) < 24:
        raise ValueError("unsafe relay polling or timeout")
    script = Path(__file__).resolve()
    base = Path(__file__).with_name("paper_aio_export_relay.py")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_script": str(script),
        "control_script_sha256": file_sha256(script),
        "base_relay_script_sha256": file_sha256(base),
        "relay_id": relay_id,
        "source_host_label": str(args.source_host_label),
        "source_host": str(args.source_host),
        "source_port": int(args.source_port),
        "source_user": str(args.source_user),
        "expected_host_key_sha256": str(args.expected_host_key_sha256),
        "remote_export_root": _remote_path(args.remote_export_root, "export root"),
        "destination_root": str(args.destination_root.resolve()),
        "lane_id": lane,
        "required_training_git_commit": str(args.required_training_git_commit),
        "required_training_protocol_fingerprint": str(
            args.required_training_protocol_fingerprint
        ),
        "required_manifest_sha256": str(args.required_manifest_sha256),
        "required_epochs": list(AUDIT_EPOCHS),
        "password_env": str(args.password_env),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "password_persisted": False,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "source_checkpoint_mutation": False,
        "confirmation20_opened": False,
    }


def _verify_control(contract: dict[str, Any]) -> None:
    script = Path(contract["control_script"])
    base = script.with_name("paper_aio_export_relay.py")
    if (
        file_sha256(script) != contract["control_script_sha256"]
        or file_sha256(base) != contract["base_relay_script_sha256"]
    ):
        raise RuntimeError("incremental relay control source changed")


def _import_available(sftp, contract: dict[str, Any]) -> dict[str, Any]:
    lane = contract["lane_id"]
    remote_set = str(
        PurePosixPath(contract["remote_export_root"])
        / lane / "INCREMENTAL_AUDIT_EXPORT_SET.json"
    )
    set_bytes = _sftp_read(sftp, remote_set)
    source_set = _read_json_bytes(set_bytes, remote_set)
    rows = validate_source_set(
        source_set, lane_id=lane,
        source_host_label=contract["source_host_label"],
    )
    lane_root = (
        Path(contract["destination_root"]) / "sources"
        / contract["source_host_label"] / lane
    )
    imported = []
    for row in rows:
        epoch = int(row["epoch"])
        receipt_remote = _remote_path(row["receipt"], "incremental receipt")
        receipt_bytes = _sftp_read(sftp, receipt_remote)
        if _bytes_sha256(receipt_bytes) != row["receipt_sha256"]:
            raise RuntimeError("incremental export receipt hash differs")
        receipt = _read_json_bytes(receipt_bytes, receipt_remote)
        validate_export_receipt(
            receipt, lane_id=lane, epoch=epoch,
            source_host_label=contract["source_host_label"],
        )
        if (
            receipt.get("training_git_commit")
            != contract["required_training_git_commit"]
            or receipt.get("training_protocol_fingerprint")
            != contract["required_training_protocol_fingerprint"]
            or receipt.get("manifest_sha256") != contract["required_manifest_sha256"]
        ):
            raise RuntimeError("incremental source training identity differs")
        checkpoint = lane_root / f"e{epoch:03d}.pt"
        sidecar = lane_root / f"e{epoch:03d}.pt.json"
        local_receipt = lane_root / f"e{epoch:03d}.export.json"
        _download_verified(
            sftp, receipt["source_checkpoint"], checkpoint,
            receipt["checkpoint_sha256"],
        )
        _download_verified(
            sftp, receipt["source_sidecar"], sidecar, receipt["sidecar_sha256"],
        )
        if local_receipt.is_file() and file_sha256(local_receipt) != row["receipt_sha256"]:
            raise RuntimeError("existing incremental receipt differs")
        if not local_receipt.is_file():
            _write_bytes(local_receipt, receipt_bytes)
        imported.append({
            "epoch": epoch,
            "export_receipt": str(local_receipt.resolve()),
            "export_receipt_sha256": row["receipt_sha256"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": receipt["sidecar_sha256"],
            "scientific_state_sha256": receipt["scientific_state_sha256"],
        })
    available = [int(row["epoch"]) for row in imported]
    complete = available == list(AUDIT_EPOCHS)
    result = {
        "schema": IMPORT_LANE_SCHEMA,
        "status": (
            "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT"
            if complete else "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT"
        ),
        "source_host_label": contract["source_host_label"],
        "lane_id": lane,
        "required_epochs": list(AUDIT_EPOCHS),
        "available_epochs": available,
        "source_export_set_sha256": _bytes_sha256(set_bytes),
        "training_git_commit": contract["required_training_git_commit"],
        "training_protocol_fingerprint": contract[
            "required_training_protocol_fingerprint"
        ],
        "manifest_sha256": contract["required_manifest_sha256"],
        "imports": imported,
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    lane_path = incremental_import_lane_path(
        Path(contract["destination_root"]), lane, contract["source_host_label"],
    )
    _write_json(lane_path, result)
    operations = Path(contract["destination_root"]) / "operations"
    set_result = {
        "schema": IMPORT_SET_SCHEMA,
        "status": result["status"],
        "relay_id": contract["relay_id"],
        "source_host_label": contract["source_host_label"],
        "lane_id": lane,
        "required_epochs": list(AUDIT_EPOCHS),
        "available_epochs": available,
        "lane_import_receipt": str(lane_path.resolve()),
        "lane_import_receipt_sha256": file_sha256(lane_path),
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json(
        operations / f"INCREMENTAL_IMPORT_SET_{contract['relay_id']}.json",
        set_result,
    )
    return set_result


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = _contract(args)
    root = Path(contract["destination_root"])
    operations = root / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    contract_path = operations / f"INCREMENTAL_RELAY_{contract['relay_id']}_CONTRACT.json"
    state_path = operations / f"INCREMENTAL_RELAY_{contract['relay_id']}_STATE.json"
    lock_path = operations / f"INCREMENTAL_RELAY_{contract['relay_id']}.lock"
    if contract_path.is_file():
        current = _read_json_bytes(contract_path.read_bytes(), str(contract_path))
        if current != contract:
            raise RuntimeError("incremental relay contract changed")
    else:
        _write_json(contract_path, contract)
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        if _fcntl is None:
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        while True:
            _verify_control(contract)
            try:
                client = _connect(contract)
                try:
                    with client.open_sftp() as sftp:
                        result = _import_available(sftp, contract)
                finally:
                    client.close()
            except (SourceExportNotReady, TransientRelayNetwork) as error:
                _write_json(state_path, {
                    "schema": STATE_SCHEMA,
                    "status": "WAITING_FOR_INCREMENTAL_SOURCE_EXPORT",
                    "pid": os.getpid(),
                    "relay_id": contract["relay_id"],
                    "last_condition": type(error).__name__,
                    "elapsed_seconds": time.time() - started,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                })
                result = None
            if result is not None:
                _write_json(state_path, {
                    "schema": STATE_SCHEMA,
                    "status": result["status"],
                    "pid": os.getpid(),
                    "relay_id": contract["relay_id"],
                    "available_epochs": result["available_epochs"],
                    "elapsed_seconds": time.time() - started,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                })
                if result["status"] == "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT":
                    return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                raise TimeoutError("incremental audit relay timed out")
            time.sleep(contract["poll_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--destination-root", type=Path, required=True)
    value.add_argument("--relay-id", required=True)
    value.add_argument("--source-host-label", required=True)
    value.add_argument("--source-host", required=True)
    value.add_argument("--source-port", type=int, required=True)
    value.add_argument("--source-user", required=True)
    value.add_argument("--expected-host-key-sha256", required=True)
    value.add_argument("--remote-export-root", required=True)
    value.add_argument("--lane", required=True)
    value.add_argument("--required-training-git-commit", required=True)
    value.add_argument("--required-training-protocol-fingerprint", required=True)
    value.add_argument("--required-manifest-sha256", required=True)
    value.add_argument("--password-env", required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"incremental audit relay failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
