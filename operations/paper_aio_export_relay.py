"""Relay source-bound paper checkpoints into one evaluation staging root.

The relay is deliberately metric-blind.  It waits for source-side export sets,
copies only their immutable receipts, checkpoints and sidecars, and verifies
every advertised SHA256 before publishing a local import set.  Authentication
is supplied through an environment variable and is never persisted.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


CONTRACT_SCHEMA = "final-unsb-paper-export-relay-contract-v1"
STATE_SCHEMA = "final-unsb-paper-export-relay-state-v1"
IMPORT_LANE_SCHEMA = "final-unsb-paper-imported-lane-v1"
IMPORT_SET_SCHEMA = "final-unsb-paper-import-set-v1"
EXPORT_SET_SCHEMA = "final-unsb-paper-source-export-set-v1"
EXPORT_RECEIPT_SCHEMA = "final-unsb-paper-checkpoint-export-v1"
UNIFIED_EPOCHS = (100, 125, 150, 175, 200)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

try:
    import fcntl as _fcntl
except ImportError:  # Windows coordinator.
    _fcntl = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON from {label}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {label}")
    return payload


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _write_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remote_path(value: str, label: str) -> str:
    path = PurePosixPath(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe remote {label}: {value!r}")
    return str(path)


def validate_export_set(
    payload: dict[str, Any], *, lane_id: str, source_host_label: str,
) -> list[dict[str, Any]]:
    if (
        payload.get("schema") != EXPORT_SET_SCHEMA
        or payload.get("status") != "COMPLETE_SOURCE_BOUND_EXPORT_SET"
        or payload.get("lane_id") != lane_id
        or payload.get("source_host_label") != source_host_label
        or payload.get("epochs") != list(UNIFIED_EPOCHS)
        or payload.get("performance_values_read") is not False
        or payload.get("checkpoint_copy_performed") is not False
        or payload.get("paired_metric_control") is not False
        or payload.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"invalid source export set for {lane_id}")
    rows = payload.get("exports")
    if not isinstance(rows, list) or len(rows) != len(UNIFIED_EPOCHS):
        raise RuntimeError(f"incomplete source export rows for {lane_id}")
    by_epoch: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid source export row for {lane_id}")
        epoch = int(row.get("epoch", -1))
        if epoch in by_epoch or epoch not in UNIFIED_EPOCHS:
            raise RuntimeError(f"duplicate or unexpected source epoch for {lane_id}")
        if not isinstance(row.get("receipt_sha256"), str):
            raise RuntimeError(f"source export row lacks receipt hash for {lane_id}")
        _remote_path(str(row.get("receipt", "")), "receipt path")
        by_epoch[epoch] = row
    if tuple(sorted(by_epoch)) != UNIFIED_EPOCHS:
        raise RuntimeError(f"source export epochs differ for {lane_id}")
    return [by_epoch[epoch] for epoch in UNIFIED_EPOCHS]


def validate_export_receipt(
    payload: dict[str, Any], *, lane_id: str, epoch: int,
    source_host_label: str,
) -> None:
    if (
        payload.get("schema") != EXPORT_RECEIPT_SCHEMA
        or payload.get("status") != "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT"
        or payload.get("lane_id") != lane_id
        or int(payload.get("epoch", -1)) != int(epoch)
        or int(payload.get("updates", -1)) != int(epoch) * 8553
        or payload.get("source_host_label") != source_host_label
        or payload.get("paired_metric_control") is not False
        or payload.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"invalid checkpoint export receipt for {lane_id} e{epoch}")
    for key in (
        "source_checkpoint", "source_sidecar", "checkpoint_sha256",
        "sidecar_sha256", "scientific_state_sha256", "training_git_commit",
        "training_protocol_fingerprint", "manifest_sha256",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise RuntimeError(f"checkpoint export receipt lacks {key}")
    _remote_path(payload["source_checkpoint"], "checkpoint path")
    _remote_path(payload["source_sidecar"], "sidecar path")


def host_key_sha256(key) -> str:
    return "SHA256:" + base64.b64encode(
        hashlib.sha256(key.asbytes()).digest()
    ).decode("ascii").rstrip("=")


class PinnedHostKeyPolicy:
    def __init__(self, expected: str):
        self.expected = str(expected)

    def missing_host_key(self, client, hostname: str, key) -> None:
        actual = host_key_sha256(key)
        if actual != self.expected:
            raise RuntimeError(
                f"SSH host key mismatch for {hostname}: expected {self.expected}, got {actual}"
            )
        client.get_host_keys().add(hostname, key.get_name(), key)


def _connect(contract: dict[str, Any]):
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("paper export relay requires paramiko") from error
    password = os.environ.get(contract["password_env"])
    if not password:
        raise RuntimeError(
            f"missing relay password environment: {contract['password_env']}"
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(
        PinnedHostKeyPolicy(contract["expected_host_key_sha256"])
    )
    client.connect(
        hostname=contract["source_host"], port=int(contract["source_port"]),
        username=contract["source_user"], password=password,
        look_for_keys=False, allow_agent=False, timeout=30,
        banner_timeout=30, auth_timeout=30,
    )
    return client


def _sftp_read(sftp, path: str) -> bytes:
    with sftp.open(path, "rb") as handle:
        return handle.read()


def _download_verified(sftp, remote: str, destination: Path, expected: str) -> str:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = file_sha256(destination)
        if actual != expected:
            raise RuntimeError(f"existing imported file hash differs: {destination}")
        return actual
    temporary = destination.with_name(destination.name + f".{os.getpid()}.part")
    try:
        sftp.get(remote, str(temporary))
        actual = file_sha256(temporary)
        if actual != expected:
            raise RuntimeError(f"downloaded file hash differs: {remote}")
        temporary.replace(destination)
        return actual
    finally:
        if temporary.exists():
            temporary.unlink()


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    lanes = [str(value) for value in args.lane]
    if len(set(lanes)) != len(lanes) or any(not _SAFE_ID.fullmatch(value) for value in lanes):
        raise ValueError("relay lanes must be unique safe identifiers")
    if not _SAFE_ID.fullmatch(str(args.source_host_label)):
        raise ValueError("source host label must be a safe identifier")
    if not str(args.password_env).startswith("FINAL_UNSB_"):
        raise ValueError("relay password environment must use a FINAL_UNSB_ prefix")
    if not str(args.expected_host_key_sha256).startswith("SHA256:"):
        raise ValueError("relay requires a pinned SHA256 SSH host key")
    if int(args.poll_seconds) < 30 or int(args.poll_seconds) > 600:
        raise ValueError("relay poll interval must be in [30,600] seconds")
    if float(args.timeout_hours) < 24:
        raise ValueError("relay timeout must be at least 24 hours")
    script = Path(__file__).resolve()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_script": str(script),
        "control_script_sha256": file_sha256(script),
        "source_host_label": str(args.source_host_label),
        "source_host": str(args.source_host),
        "source_port": int(args.source_port),
        "source_user": str(args.source_user),
        "expected_host_key_sha256": str(args.expected_host_key_sha256),
        "remote_export_root": _remote_path(
            str(args.remote_export_root), "export root"
        ),
        "destination_root": str(args.destination_root.resolve()),
        "lanes": lanes,
        "epochs": list(UNIFIED_EPOCHS),
        "password_env": str(args.password_env),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "password_persisted": False,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "source_checkpoint_mutation": False,
        "confirmation20_opened": False,
    }


def _freeze_contract(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    proposed = _contract(args)
    root = args.destination_root.resolve()
    path = root / "operations" / f"IMPORT_RELAY_{args.source_host_label}_CONTRACT.json"
    if path.is_file():
        current = _read_json_bytes(path.read_bytes(), str(path))
        if current != proposed:
            raise RuntimeError("paper import relay contract changed")
        return path, current
    _write_json(path, proposed)
    return path, proposed


def _verify_script(contract: dict[str, Any]) -> None:
    script = Path(contract["control_script"])
    if not script.is_file() or file_sha256(script) != contract["control_script_sha256"]:
        raise RuntimeError("paper import relay control script changed")


def _import_lane(sftp, contract: dict[str, Any], lane_id: str) -> dict[str, Any]:
    remote_root = PurePosixPath(contract["remote_export_root"])
    set_path = str(remote_root / lane_id / "EXPORT_SET.json")
    set_bytes = _sftp_read(sftp, set_path)
    export_set = _read_json_bytes(set_bytes, set_path)
    rows = validate_export_set(
        export_set, lane_id=lane_id,
        source_host_label=contract["source_host_label"],
    )
    lane_root = (
        Path(contract["destination_root"]) / "sources"
        / contract["source_host_label"] / lane_id
    )
    imported = []
    for row in rows:
        epoch = int(row["epoch"])
        receipt_remote = _remote_path(row["receipt"], "receipt path")
        receipt_bytes = _sftp_read(sftp, receipt_remote)
        if _bytes_sha256(receipt_bytes) != row["receipt_sha256"]:
            raise RuntimeError(f"source receipt hash differs for {lane_id} e{epoch}")
        receipt = _read_json_bytes(receipt_bytes, receipt_remote)
        validate_export_receipt(
            receipt, lane_id=lane_id, epoch=epoch,
            source_host_label=contract["source_host_label"],
        )
        checkpoint = lane_root / f"e{epoch:03d}.pt"
        sidecar = lane_root / f"e{epoch:03d}.pt.json"
        local_receipt = lane_root / f"e{epoch:03d}.export.json"
        _download_verified(
            sftp, receipt["source_checkpoint"], checkpoint,
            receipt["checkpoint_sha256"],
        )
        _download_verified(
            sftp, receipt["source_sidecar"], sidecar,
            receipt["sidecar_sha256"],
        )
        if local_receipt.is_file() and file_sha256(local_receipt) != row["receipt_sha256"]:
            raise RuntimeError(f"existing local receipt differs for {lane_id} e{epoch}")
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
    result = {
        "schema": IMPORT_LANE_SCHEMA,
        "status": "COMPLETE_VERIFIED_IMPORTED_LANE",
        "source_host_label": contract["source_host_label"],
        "lane_id": lane_id,
        "epochs": list(UNIFIED_EPOCHS),
        "source_export_set_sha256": _bytes_sha256(set_bytes),
        "imports": imported,
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json(lane_root / "IMPORT_LANE.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_path, contract = _freeze_contract(args)
    root = Path(contract["destination_root"])
    operations = root / "operations"
    state_path = operations / f"IMPORT_RELAY_{contract['source_host_label']}_STATE.json"
    final_path = operations / f"IMPORT_SET_{contract['source_host_label']}.json"
    lock_path = operations / f"IMPORT_RELAY_{contract['source_host_label']}.lock"
    operations.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            if _fcntl is None:
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                _fcntl.flock(lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise RuntimeError("paper import relay is already running") from error
        while True:
            _verify_script(contract)
            complete: dict[str, Any] = {}
            try:
                client = _connect(contract)
                try:
                    with client.open_sftp() as sftp:
                        for lane_id in contract["lanes"]:
                            complete[lane_id] = _import_lane(sftp, contract, lane_id)
                finally:
                    client.close()
            except (FileNotFoundError, EOFError, OSError, socket.error) as error:
                _write_json(state_path, {
                    "schema": STATE_SCHEMA,
                    "status": "WAITING_FOR_COMPLETE_SOURCE_EXPORTS_OR_TRANSIENT_NETWORK",
                    "pid": os.getpid(),
                    "source_host_label": contract["source_host_label"],
                    "lanes": contract["lanes"],
                    "contract": str(contract_path),
                    "contract_sha256": file_sha256(contract_path),
                    "last_error_type": type(error).__name__,
                    "elapsed_seconds": time.time() - started,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                })
            else:
                result = {
                    "schema": IMPORT_SET_SCHEMA,
                    "status": "COMPLETE_VERIFIED_IMPORT_SET",
                    "source_host_label": contract["source_host_label"],
                    "lanes": contract["lanes"],
                    "epochs": list(UNIFIED_EPOCHS),
                    "lane_imports": {
                        lane_id: {
                            "receipt": str(
                                root / "sources" / contract["source_host_label"]
                                / lane_id / "IMPORT_LANE.json"
                            ),
                            "receipt_sha256": file_sha256(
                                root / "sources" / contract["source_host_label"]
                                / lane_id / "IMPORT_LANE.json"
                            ),
                        }
                        for lane_id in contract["lanes"]
                    },
                    "checkpoint_copy_performed": True,
                    "source_checkpoint_mutation": False,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                }
                _write_json(final_path, result)
                _write_json(state_path, {
                    "schema": STATE_SCHEMA,
                    "status": "COMPLETE_VERIFIED_IMPORT_SET",
                    "pid": os.getpid(),
                    "source_host_label": contract["source_host_label"],
                    "result": str(final_path),
                    "result_sha256": file_sha256(final_path),
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                })
                return result
            if time.time() - started > float(contract["timeout_hours"]) * 3600:
                raise TimeoutError("paper import relay exceeded its frozen timeout")
            time.sleep(int(contract["poll_seconds"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-port", type=int, required=True)
    parser.add_argument("--source-user", required=True)
    parser.add_argument("--expected-host-key-sha256", required=True)
    parser.add_argument("--remote-export-root", required=True)
    parser.add_argument("--lane", action="append", required=True)
    parser.add_argument("--password-env", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=480)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"paper export relay failed: {error}", file=sys.stderr)
        raise
