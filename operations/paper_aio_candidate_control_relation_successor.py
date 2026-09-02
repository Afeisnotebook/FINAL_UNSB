"""Build a review-only cross-host control relation for a frozen candidate.

The successor waits for an independently verified plain runtime receipt.  It
then verifies the candidate-to-parent cross-code gate and the parent-to-plain
runtime identity as two explicit proof links.  It never edits the committed
relation registry, evaluates a checkpoint, or starts training.
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

from operations.paper_aio_export_relay import _write_json
from research.paper_aio.protocol import file_sha256
from research.paper_aio.runtime_relation import (
    exact_candidate_control_relation_payload,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is POSIX
    fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-candidate-control-relation-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-candidate-control-relation-successor-state-v1"
SOURCE_CONTRACT_SCHEMA = "final-unsb-paper-runtime-relation-successor-contract-v1"
SOURCE_STATE_SCHEMA = "final-unsb-paper-runtime-relation-successor-state-v1"


class SourceRelationNotReady(RuntimeError):
    """The upstream exact plain receipt has not been published yet."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def git_identity(repo: Path) -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True,
    ).strip())
    return head, dirty


def _pinned(path: Path, digest: str, label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file() or file_sha256(result) != digest:
        raise RuntimeError(f"{label} is absent or changed")
    return result


def proposed_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    head, dirty = git_identity(repo)
    if head != args.required_control_git_commit or dirty:
        raise RuntimeError("candidate-control successor checkout identity changed")
    if args.poll_seconds < 30 or args.poll_seconds > 600:
        raise RuntimeError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise RuntimeError("timeout must be at least 24 hours")
    source_contract_path = _pinned(
        args.source_successor_contract, args.required_source_successor_contract_sha256,
        "source runtime-relation successor contract",
    )
    source_contract = read_json(source_contract_path)
    if (
        source_contract.get("schema") != SOURCE_CONTRACT_SCHEMA
        or source_contract.get("plain_source_host_label")
        != args.plain_source_host_label
        or source_contract.get("protocol_fingerprint")
        != args.plain_protocol_fingerprint
        or source_contract.get("manifest_sha256") != args.manifest_sha256
        or source_contract.get("registry_edited") is not False
        or source_contract.get("comparison_authorized") is not False
        or source_contract.get("performance_values_read") is not False
        or source_contract.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("source successor contract is not a sealed plain receipt source")
    evidence = {}
    for name, path, digest in (
        ("candidate_runtime_gate", args.candidate_runtime_gate,
         args.required_candidate_runtime_gate_sha256),
        ("candidate_authorization", args.candidate_authorization,
         args.required_candidate_authorization_sha256),
        ("candidate_metadata_import", args.candidate_metadata_import,
         args.required_candidate_metadata_import_sha256),
        ("candidate_authority", args.candidate_authority,
         args.required_candidate_authority_sha256),
    ):
        fixed = _pinned(path, digest, name)
        evidence[name] = {"path": str(fixed), "sha256": digest}
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_REVIEW_ONLY_SUCCESSOR",
        "control_repo": str(repo),
        "control_git_commit": head,
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__).resolve()),
        "candidate_id": args.candidate_id,
        "method_source_host_label": args.method_source_host_label,
        "plain_source_host_label": args.plain_source_host_label,
        "plain_protocol_fingerprint": args.plain_protocol_fingerprint,
        "manifest_sha256": args.manifest_sha256,
        "candidate_evidence": evidence,
        "source_successor_contract": str(source_contract_path),
        "source_successor_contract_sha256": args.required_source_successor_contract_sha256,
        "source_successor_state": str(args.source_successor_state.resolve()),
        "plain_runtime_receipt": str(args.plain_runtime_receipt.resolve()),
        "destination_output": str(args.destination_output.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
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
        "candidate_id": contract["candidate_id"],
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


def require_verified_source(contract: dict[str, Any]) -> Path:
    state_path = Path(contract["source_successor_state"])
    plain_path = Path(contract["plain_runtime_receipt"]).resolve()
    if not state_path.is_file():
        raise SourceRelationNotReady("source successor state is absent")
    state = read_json(state_path)
    status = state.get("status")
    if status != "COMPLETE_REVIEW_ONLY_RUNTIME_RELATION_CANDIDATE":
        if isinstance(status, str) and status.startswith("BLOCKED"):
            raise RuntimeError("source runtime-relation successor is blocked")
        raise SourceRelationNotReady("source runtime-relation successor is waiting")
    if (
        state.get("schema") != SOURCE_STATE_SCHEMA
        or Path(str(state.get("plain_runtime_receipt", ""))).resolve() != plain_path
        or state.get("plain_source_host_label") != contract["plain_source_host_label"]
        or state.get("plain_runtime_receipt_sha256") != file_sha256(plain_path)
        or state.get("exact_runtime_equivalence") is not True
        or state.get("registry_edited") is not False
        or state.get("comparison_authorized") is not False
        or state.get("performance_values_read") is not False
        or state.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("source successor completion does not bind the plain receipt")
    return plain_path


def publish_relation(contract: dict[str, Any], plain_path: Path, destination: Path) -> dict[str, Any]:
    evidence = contract["candidate_evidence"]
    result = exact_candidate_control_relation_payload(
        candidate_id=contract["candidate_id"],
        method_source_host_label=contract["method_source_host_label"],
        plain_source_host_label=contract["plain_source_host_label"],
        candidate_runtime_gate=Path(evidence["candidate_runtime_gate"]["path"]),
        candidate_authorization=Path(evidence["candidate_authorization"]["path"]),
        candidate_metadata_import=Path(evidence["candidate_metadata_import"]["path"]),
        candidate_authority=Path(evidence["candidate_authority"]["path"]),
        plain_runtime_receipt=plain_path,
    )
    if destination.is_file():
        if read_json(destination) != result:
            raise RuntimeError("existing candidate-control relation differs")
    else:
        _write_json(destination, result)
    return result


def run(args: argparse.Namespace) -> int:
    contract = proposed_contract(args)
    output = Path(contract["destination_output"])
    operations = output / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    contract_path = operations / "CANDIDATE_CONTROL_RELATION_SUCCESSOR_CONTRACT.json"
    state_path = operations / "CANDIDATE_CONTROL_RELATION_SUCCESSOR_STATE.json"
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("candidate-control successor contract changed")
    else:
        _write_json(contract_path, contract)
    relation_path = (
        output / "relation_candidates"
        / f"{contract['candidate_id']}__{contract['method_source_host_label']}"
        f"__{contract['plain_source_host_label']}.json"
    )
    started = time.time()
    lock_path = operations / "CANDIDATE_CONTROL_RELATION_SUCCESSOR.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("candidate-control successor requires POSIX file locking")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if time.time() - started >= contract["timeout_hours"] * 3600:
                _write_json(state_path, state_payload(
                    contract, status="BLOCKED_TIMEOUT_WAITING_FOR_VERIFIED_PLAIN_RECEIPT",
                ))
                return 3
            try:
                plain_path = require_verified_source(contract)
            except SourceRelationNotReady as error:
                _write_json(state_path, state_payload(
                    contract, status="WAITING_FOR_VERIFIED_PLAIN_RUNTIME_RECEIPT",
                    transient_condition=type(error).__name__,
                ))
                time.sleep(contract["poll_seconds"])
                continue
            result = publish_relation(contract, plain_path, relation_path)
            _write_json(state_path, state_payload(
                contract,
                status="COMPLETE_REVIEW_ONLY_CANDIDATE_CONTROL_RELATION",
                plain_runtime_receipt=str(plain_path),
                plain_runtime_receipt_sha256=file_sha256(plain_path),
                relation_candidate=str(relation_path),
                relation_candidate_sha256=file_sha256(relation_path),
                proof_chain=result["proof_chain"],
                exact_runtime_equivalence=result.get("differences") == {},
            ))
            return 0


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--required-control-git-commit", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--method-source-host-label", required=True)
    parser.add_argument("--plain-source-host-label", required=True)
    parser.add_argument("--plain-protocol-fingerprint", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--candidate-runtime-gate", type=Path, required=True)
    parser.add_argument("--required-candidate-runtime-gate-sha256", required=True)
    parser.add_argument("--candidate-authorization", type=Path, required=True)
    parser.add_argument("--required-candidate-authorization-sha256", required=True)
    parser.add_argument("--candidate-metadata-import", type=Path, required=True)
    parser.add_argument("--required-candidate-metadata-import-sha256", required=True)
    parser.add_argument("--candidate-authority", type=Path, required=True)
    parser.add_argument("--required-candidate-authority-sha256", required=True)
    parser.add_argument("--source-successor-contract", type=Path, required=True)
    parser.add_argument("--required-source-successor-contract-sha256", required=True)
    parser.add_argument("--source-successor-state", type=Path, required=True)
    parser.add_argument("--plain-runtime-receipt", type=Path, required=True)
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
            contract_path = (
                output / "operations" / "CANDIDATE_CONTROL_RELATION_SUCCESSOR_CONTRACT.json"
            )
            if contract_path.is_file():
                contract = read_json(contract_path)
                _write_json(
                    output / "operations" / "CANDIDATE_CONTROL_RELATION_SUCCESSOR_STATE.json",
                    state_payload(
                        contract, status="BLOCKED_FAIL_CLOSED",
                        error_type=type(error).__name__,
                    ),
                )
        finally:
            print(f"candidate-control successor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
