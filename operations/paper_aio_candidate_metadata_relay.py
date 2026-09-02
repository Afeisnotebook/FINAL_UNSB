"""Import frozen dynamic-candidate evidence into the paper evaluator.

The large checkpoint relay intentionally copies only model checkpoints and
their export receipts.  Dynamic candidates additionally need the exact
candidate lock, authorization, and cross-code runtime gate at adjudication
time.  This one-shot relay copies those three small immutable JSON artifacts,
pins the SSH host key, validates hashes against the portable evaluation
authority, and never authorizes or schedules training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from operations.paper_aio_export_relay import (
    _connect,
    _sftp_read,
    _write_bytes,
    _write_json,
    file_sha256,
)


CONTRACT_SCHEMA = "final-unsb-paper-candidate-metadata-relay-contract-v1"
RECEIPT_SCHEMA = "final-unsb-paper-candidate-metadata-import-v1"
AUTHORITY_SCHEMA = "final-unsb-paper-portable-candidate-evaluation-authority-v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid candidate metadata JSON: {label}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"expected candidate metadata object: {label}")
    return result


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def validate_authority(path: Path, candidate_id: str) -> dict[str, Any]:
    authority = _read(path)
    evidence = authority.get("source_evidence") or {}
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != "FROZEN_EVALUATION_ONLY_AUTHORITY"
        or authority.get("candidate_id") != candidate_id
        or authority.get("evaluation_only") is not True
        or authority.get("authorizes_training") is not False
        or authority.get("performance_metric_values_included") is not False
        or authority.get("paired_metric_control") is not False
        or authority.get("confirmation20_opened") is not False
        or any(
            not isinstance(evidence.get(key), str) or len(evidence[key]) != 64
            for key in (
                "candidate_lock_sha256",
                "runtime_gate_sha256",
                "authorization_sha256",
            )
        )
    ):
        raise RuntimeError("portable candidate authority is invalid")
    return authority


def validate_candidate_metadata(
    *, candidate_id: str, authority: dict[str, Any], lock: dict[str, Any],
    authorization: dict[str, Any], runtime_gate: dict[str, Any],
) -> None:
    evidence = authority["source_evidence"]
    if (
        lock.get("schema") != "final-unsb-paper-candidate-lock-v1"
        or lock.get("status") != "PASS_FULL_DATA_CANDIDATE_LOCK"
        or lock.get("candidate_id") != candidate_id
        or lock.get("full_data_authorized") is not False
        or lock.get("paired_metric_control") is not False
        or lock.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate lock is invalid")
    if (
        authorization.get("schema")
        != "final-unsb-paper-candidate-authorization-v1"
        or authorization.get("status")
        != "PASS_FULL_DATA_CANDIDATE_AUTHORIZATION"
        or authorization.get("candidate_id") != candidate_id
        or authorization.get("candidate_lock_sha256")
        != evidence["candidate_lock_sha256"]
        or authorization.get("candidate_runtime_gate_sha256")
        != evidence["runtime_gate_sha256"]
        or authorization.get("paired_metric_control") is not False
        or authorization.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate authorization is invalid")
    if (
        runtime_gate.get("schema")
        != "final-unsb-paper-candidate-runtime-gate-v1"
        or runtime_gate.get("status") != "PASS_CROSS_CODE_CANDIDATE_RUNTIME"
        or runtime_gate.get("candidate_id") != candidate_id
        or runtime_gate.get("manifest_sha256")
        != "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
        or runtime_gate.get("e0_scientific_core_exact") is not True
        or runtime_gate.get("plain_2000_transition_exact") is not True
        or runtime_gate.get("zero_intervention_identity_exact") is not True
        or runtime_gate.get("candidate_resume_exact") is not True
        or runtime_gate.get("candidate_evaluation_repeat_exact") is not True
        or runtime_gate.get("paired_metric_control") is not False
        or runtime_gate.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate runtime gate is invalid")


def _contract(args: argparse.Namespace, authority: dict[str, Any]) -> dict[str, Any]:
    repo = args.repo.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("candidate metadata relay checkout must be clean")
    head = _git(repo, "rev-parse", "HEAD")
    if head != args.required_control_git_commit:
        raise RuntimeError("candidate metadata relay checkout moved")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_ONE_SHOT",
        "control_repo": str(repo),
        "control_git_commit": head,
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__).resolve()),
        "candidate_id": args.candidate_id,
        "authority": str(args.authority.resolve()),
        "authority_sha256": file_sha256(args.authority),
        "expected_hashes": dict(authority["source_evidence"]),
        "source_host_label": args.source_host_label,
        "source_host": args.source_host,
        "source_port": int(args.source_port),
        "source_user": args.source_user,
        "expected_host_key_sha256": args.expected_host_key_sha256,
        "password_env": args.password_env,
        "remote_candidate_lock": args.remote_candidate_lock,
        "remote_authorization": args.remote_authorization,
        "remote_runtime_gate": args.remote_runtime_gate,
        "destination_output": str(args.destination_output.resolve()),
        "password_persisted": False,
        "training_authorized_or_scheduled": False,
        "paired_performance_used_for_training_or_scheduling": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_authority(args.authority, args.candidate_id)
    contract = _contract(args, authority)
    output = args.destination_output.resolve()
    operations = output / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    contract_path = operations / f"CANDIDATE_METADATA_RELAY_{args.candidate_id}_CONTRACT.json"
    if contract_path.is_file():
        if _read(contract_path) != contract:
            raise RuntimeError("candidate metadata relay contract changed")
    else:
        _write_json(contract_path, contract)

    client = _connect(contract)
    try:
        with client.open_sftp() as sftp:
            blobs = {
                "candidate_lock": _sftp_read(sftp, args.remote_candidate_lock),
                "authorization": _sftp_read(sftp, args.remote_authorization),
                "runtime_gate": _sftp_read(sftp, args.remote_runtime_gate),
            }
    finally:
        client.close()

    expected = authority["source_evidence"]
    hashes = {
        key: hashlib.sha256(value).hexdigest()
        for key, value in blobs.items()
    }
    if (
        hashes["candidate_lock"] != expected["candidate_lock_sha256"]
        or hashes["authorization"] != expected["authorization_sha256"]
        or hashes["runtime_gate"] != expected["runtime_gate_sha256"]
    ):
        raise RuntimeError("candidate metadata differs from portable authority")
    lock = _json_bytes(blobs["candidate_lock"], "candidate lock")
    authorization = _json_bytes(blobs["authorization"], "authorization")
    runtime_gate = _json_bytes(blobs["runtime_gate"], "runtime gate")
    validate_candidate_metadata(
        candidate_id=args.candidate_id, authority=authority, lock=lock,
        authorization=authorization, runtime_gate=runtime_gate,
    )

    lock_path = output / "candidate_locks" / args.candidate_id / "CANDIDATE_LOCK.json"
    authorization_path = output / "gates" / f"CANDIDATE_AUTHORIZATION_{args.candidate_id}.json"
    runtime_path = output / "candidate_runtime_gate" / args.candidate_id / "CANDIDATE_RUNTIME_GATE.json"
    for path, key in (
        (lock_path, "candidate_lock"),
        (authorization_path, "authorization"),
        (runtime_path, "runtime_gate"),
    ):
        _write_bytes(path, blobs[key])
        if file_sha256(path) != hashes[key]:
            raise RuntimeError(f"candidate metadata publish failed: {path}")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "COMPLETE_VERIFIED_CANDIDATE_METADATA_IMPORT",
        "candidate_id": args.candidate_id,
        "source_host_label": args.source_host_label,
        "authority": str(args.authority.resolve()),
        "authority_sha256": file_sha256(args.authority),
        "candidate_lock": str(lock_path.resolve()),
        "candidate_lock_sha256": hashes["candidate_lock"],
        "authorization": str(authorization_path.resolve()),
        "authorization_sha256": hashes["authorization"],
        "runtime_gate": str(runtime_path.resolve()),
        "runtime_gate_sha256": hashes["runtime_gate"],
        "frozen_prior_evidence_transferred": True,
        "training_authorized_or_scheduled": False,
        "paired_performance_used_for_training_or_scheduling": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    receipt_path = operations / f"CANDIDATE_METADATA_IMPORT_{args.candidate_id}.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--candidate-id", required=True)
    value.add_argument("--authority", type=Path, required=True)
    value.add_argument("--source-host-label", required=True)
    value.add_argument("--source-host", required=True)
    value.add_argument("--source-port", type=int, required=True)
    value.add_argument("--source-user", required=True)
    value.add_argument("--expected-host-key-sha256", required=True)
    value.add_argument("--password-env", required=True)
    value.add_argument("--remote-candidate-lock", required=True)
    value.add_argument("--remote-authorization", required=True)
    value.add_argument("--remote-runtime-gate", required=True)
    value.add_argument("--destination-output", type=Path, required=True)
    return value


def main() -> int:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
