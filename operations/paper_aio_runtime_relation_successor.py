"""Relay a future exact plain runtime receipt into a review-only relation.

The successor runs on the unified-evaluation host.  It waits for one immutable
remote runtime-twin receipt, pins the SSH host key, validates it against the
already imported method receipt and authorization, and materializes a
metric-blind runtime-relation *candidate*.  It never edits the committed
relation registry, starts training, reads performance values, or authorizes a
comparison.
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

from operations.paper_aio_export_relay import (
    SourceExportNotReady,
    TransientRelayNetwork,
    _connect,
    _sftp_read,
    _write_bytes,
    _write_json,
    _remote_path,
    file_sha256,
)
from research.paper_aio.runtime_relation import exact_runtime_relation_payload

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is POSIX
    fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-runtime-relation-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-runtime-relation-successor-state-v1"
RUNTIME_SCHEMA = "final-unsb-paper-runtime-twin-receipt-v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON from {label}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object from {label}")
    return result


def contains_performance_field(value: Any) -> bool:
    forbidden = ("psnr", "ssim", "lpips", "fid", "kid", "ranking", "delta")
    if isinstance(value, dict):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or contains_performance_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_performance_field(item) for item in value)
    return False


def validate_plain_runtime_receipt(
    receipt: dict[str, Any], *, host_label: str,
    protocol_fingerprint: str, manifest_sha256: str,
) -> None:
    if (
        receipt.get("schema") != RUNTIME_SCHEMA
        or receipt.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or receipt.get("host_label") != host_label
        or int(receipt.get("updates", -1)) != 2000
        or receipt.get("protocol_fingerprint") != protocol_fingerprint
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("exact_runtime_equivalence") is not True
        or receipt.get("differences") != {}
        or receipt.get("confirmation20_opened") is not False
        or contains_performance_field(receipt)
    ):
        raise RuntimeError("remote plain runtime receipt is not an exact sealed gate")


def git_identity(repo: Path) -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip())
    return head, dirty


def proposed_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    head, dirty = git_identity(repo)
    if head != args.required_control_git_commit or dirty:
        raise RuntimeError("runtime-relation successor checkout identity changed")
    if not args.password_env.startswith("FINAL_UNSB_"):
        raise RuntimeError("password environment name must use FINAL_UNSB_ prefix")
    if not args.expected_host_key_sha256.startswith("SHA256:"):
        raise RuntimeError("a pinned SHA256 SSH host key is required")
    if args.poll_seconds < 30 or args.poll_seconds > 600:
        raise RuntimeError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise RuntimeError("timeout must be at least 24 hours")
    _remote_path(args.remote_plain_runtime_receipt, "plain runtime receipt")
    for path, expected, label in (
        (args.method_runtime_receipt, args.required_method_runtime_sha256, "method runtime"),
        (args.method_authorization_receipt, args.required_method_authorization_sha256, "method authorization"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"{label} receipt is absent or changed")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_REVIEW_ONLY_SUCCESSOR",
        "control_repo": str(repo),
        "control_git_commit": head,
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__).resolve()),
        "lane_id": args.lane_id,
        "method_source_host_label": args.method_source_host_label,
        "plain_source_host_label": args.plain_source_host_label,
        "method_runtime_receipt": str(args.method_runtime_receipt.resolve()),
        "method_runtime_receipt_sha256": args.required_method_runtime_sha256,
        "method_authorization_receipt": str(args.method_authorization_receipt.resolve()),
        "method_authorization_receipt_sha256": args.required_method_authorization_sha256,
        "protocol_fingerprint": args.protocol_fingerprint,
        "manifest_sha256": args.manifest_sha256,
        "source_host": args.source_host,
        "source_port": int(args.source_port),
        "source_user": args.source_user,
        "expected_host_key_sha256": args.expected_host_key_sha256,
        "password_env": args.password_env,
        "remote_plain_runtime_receipt": args.remote_plain_runtime_receipt,
        "destination_output": str(args.destination_output.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "password_persisted": False,
        "registry_edited": False,
        "comparison_authorized": False,
        "training_authorized_or_scheduled": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def state_payload(contract: dict[str, Any], *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "lane_id": contract["lane_id"],
        "method_source_host_label": contract["method_source_host_label"],
        "plain_source_host_label": contract["plain_source_host_label"],
        "registry_edited": False,
        "comparison_authorized": False,
        "training_authorized_or_scheduled": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def publish_exact_bytes(path: Path, value: bytes) -> str:
    digest = hashlib.sha256(value).hexdigest()
    if path.is_file():
        if file_sha256(path) != digest:
            raise RuntimeError(f"immutable receipt destination changed: {path}")
    else:
        _write_bytes(path, value)
    if file_sha256(path) != digest:
        raise RuntimeError(f"runtime receipt publication failed: {path}")
    return digest


def publish_relation_candidate(
    *, contract: dict[str, Any], plain_runtime_receipt: Path,
    destination: Path,
) -> dict[str, Any]:
    expected = exact_runtime_relation_payload(
        lane_id=contract["lane_id"],
        method_source_host_label=contract["method_source_host_label"],
        plain_source_host_label=contract["plain_source_host_label"],
        method_runtime_receipt=Path(contract["method_runtime_receipt"]),
        plain_runtime_receipt=plain_runtime_receipt,
        method_authorization_receipt=Path(contract["method_authorization_receipt"]),
    )
    if destination.is_file():
        if read_json(destination) != expected:
            raise RuntimeError("existing runtime-relation candidate differs")
    else:
        _write_json(destination, expected)
    return expected


def run(args: argparse.Namespace) -> int:
    contract = proposed_contract(args)
    output = args.destination_output.resolve()
    operations = output / "operations"
    contract_path = operations / "RUNTIME_RELATION_SUCCESSOR_CONTRACT.json"
    state_path = operations / "RUNTIME_RELATION_SUCCESSOR_STATE.json"
    operations.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("runtime-relation successor contract changed")
    else:
        _write_json(contract_path, contract)
    plain_path = (
        output / "primary_receipts" / contract["plain_source_host_label"]
        / "RUNTIME_TWIN.json"
    )
    relation_path = (
        output / "relation_candidates"
        / f"{contract['lane_id']}__{contract['method_source_host_label']}"
        f"__{contract['plain_source_host_label']}.json"
    )
    lock_path = operations / "RUNTIME_RELATION_SUCCESSOR.lock"
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("runtime-relation successor requires POSIX file locking")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if time.time() - started >= contract["timeout_hours"] * 3600:
                _write_json(state_path, state_payload(
                    contract, status="BLOCKED_TIMEOUT_WAITING_FOR_EXACT_PLAIN_RECEIPT",
                ))
                return 3
            try:
                client = _connect(contract)
                try:
                    with client.open_sftp() as sftp:
                        blob = _sftp_read(sftp, contract["remote_plain_runtime_receipt"])
                finally:
                    client.close()
            except (SourceExportNotReady, TransientRelayNetwork) as error:
                _write_json(state_path, state_payload(
                    contract,
                    status="WAITING_FOR_EXACT_PLAIN_RUNTIME_RECEIPT",
                    transient_condition=type(error).__name__,
                ))
                time.sleep(contract["poll_seconds"])
                continue
            receipt = json_bytes(blob, "remote plain runtime receipt")
            validate_plain_runtime_receipt(
                receipt, host_label=contract["plain_source_host_label"],
                protocol_fingerprint=contract["protocol_fingerprint"],
                manifest_sha256=contract["manifest_sha256"],
            )
            plain_sha = publish_exact_bytes(plain_path, blob)
            relation = publish_relation_candidate(
                contract=contract, plain_runtime_receipt=plain_path,
                destination=relation_path,
            )
            _write_json(state_path, state_payload(
                contract,
                status="COMPLETE_REVIEW_ONLY_RUNTIME_RELATION_CANDIDATE",
                plain_runtime_receipt=str(plain_path),
                plain_runtime_receipt_sha256=plain_sha,
                relation_candidate=str(relation_path),
                relation_candidate_sha256=file_sha256(relation_path),
                exact_runtime_equivalence=relation.get("differences") == {},
            ))
            return 0


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--required-control-git-commit", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--method-source-host-label", required=True)
    parser.add_argument("--plain-source-host-label", required=True)
    parser.add_argument("--method-runtime-receipt", type=Path, required=True)
    parser.add_argument("--required-method-runtime-sha256", required=True)
    parser.add_argument("--method-authorization-receipt", type=Path, required=True)
    parser.add_argument("--required-method-authorization-sha256", required=True)
    parser.add_argument("--protocol-fingerprint", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--source-port", type=int, required=True)
    parser.add_argument("--source-user", required=True)
    parser.add_argument("--expected-host-key-sha256", required=True)
    parser.add_argument("--password-env", required=True)
    parser.add_argument("--remote-plain-runtime-receipt", required=True)
    parser.add_argument("--destination-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser.parse_args(argv)


def main() -> int:
    args = arguments()
    try:
        return run(args)
    except Exception as error:
        try:
            output = args.destination_output.resolve()
            contract_path = output / "operations" / "RUNTIME_RELATION_SUCCESSOR_CONTRACT.json"
            if contract_path.is_file():
                contract = read_json(contract_path)
                _write_json(
                    output / "operations" / "RUNTIME_RELATION_SUCCESSOR_STATE.json",
                    state_payload(
                        contract, status="BLOCKED_FAIL_CLOSED",
                        error_type=type(error).__name__,
                    ),
                )
        finally:
            print(f"runtime-relation successor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
