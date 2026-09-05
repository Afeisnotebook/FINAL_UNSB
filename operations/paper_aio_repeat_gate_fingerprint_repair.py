"""Repair the one known legacy repeated-evaluation fingerprint mix-up.

The e4a5eed gate wrote the frozen evaluation-bundle fingerprint into the
``protocol_fingerprint`` field.  That made a deterministic PASS receipt fail
the subsequent training-protocol authorization.  This tool performs one
fail-closed metadata migration.  It preserves the original bytes, requires
the exact recorded gate-5 failure, and never loads metrics or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def immutable_copy(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"immutable archive changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def repair(
    *,
    output: Path,
    repair_id: str,
    required_training_protocol_fingerprint: str,
    required_evaluation_bundle_fingerprint: str,
    required_failed_successor_state_sha256: str,
    required_legacy_evaluation_receipt_sha256: str,
    required_failed_authorization_sha256: str,
    required_runtime_receipt_sha256: str,
    host_label: str,
) -> dict:
    if not _SAFE_ID.fullmatch(repair_id):
        raise RuntimeError("unsafe repair id")
    output = Path(output).resolve()
    operations = output / "operations"
    gates = output / "gates"
    state_path = operations / "CROSS_HOST_PLAIN_SUCCESSOR_STATE.json"
    evaluation_path = gates / "EVALUATION_REPEAT_plain.json"
    authorization_path = gates / "LANE_AUTHORIZATION_plain.json"
    runtime_path = gates / f"RUNTIME_TWIN_{host_label}.json"
    protocol_path = output / "PAPER_PROTOCOL.json"
    required_files = (
        state_path,
        evaluation_path,
        authorization_path,
        runtime_path,
        protocol_path,
    )
    if any(not path.is_file() for path in required_files):
        raise RuntimeError("legacy gate repair inputs are incomplete")
    expected_hashes = {
        state_path: required_failed_successor_state_sha256,
        evaluation_path: required_legacy_evaluation_receipt_sha256,
        authorization_path: required_failed_authorization_sha256,
        runtime_path: required_runtime_receipt_sha256,
    }
    for path, expected in expected_hashes.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"legacy gate repair input changed: {path.name}")

    state = read_json(state_path)
    evaluation = read_json(evaluation_path)
    authorization = read_json(authorization_path)
    runtime = read_json(runtime_path)
    protocol = read_json(protocol_path)
    if (
        state.get("status") != "BLOCKED_ENGINEERING_GATE_FAILURE"
        or state.get("gate_index") != 5
        or state.get("child_returncode") != 1
        or state.get("host_label") != host_label
        or state.get("required_protocol_fingerprint")
        != required_training_protocol_fingerprint
    ):
        raise RuntimeError("successor is not the exact recorded gate-5 failure")
    if (
        evaluation.get("schema")
        != "final-unsb-paper-evaluation-repeat-gate-v1"
        or evaluation.get("status") != "PASS"
        or evaluation.get("lane_id") != "plain"
        or evaluation.get("first_result_sha256")
        != evaluation.get("second_result_sha256")
        or evaluation.get("protocol_fingerprint")
        != required_evaluation_bundle_fingerprint
        or "evaluation_bundle_fingerprint" in evaluation
        or evaluation.get("split") != "discovery"
        or evaluation.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("receipt is not the known legacy fingerprint mix-up")
    if (
        authorization.get("status") != "FAIL"
        or authorization.get("lane_id") != "plain"
        or authorization.get("failures")
        != ["lane-specific repeated evaluation receipt is stale"]
        or authorization.get("evaluation_repeat_gate_sha256")
        != required_legacy_evaluation_receipt_sha256
        or authorization.get("paired_metric_control") is not False
        or authorization.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("authorization failure is not the known legacy failure")
    if (
        protocol.get("protocol_fingerprint")
        != required_training_protocol_fingerprint
        or (protocol.get("evaluation") or {}).get("bundle_seed_fingerprint")
        != required_evaluation_bundle_fingerprint
        or protocol.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("paper protocol identities changed")
    if (
        runtime.get("schema") != "final-unsb-paper-runtime-twin-receipt-v1"
        or runtime.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or runtime.get("host_label") != host_label
        or runtime.get("updates") != 2000
        or runtime.get("protocol_fingerprint")
        != required_training_protocol_fingerprint
        or runtime.get("exact_runtime_equivalence") is not True
        or runtime.get("differences") != {}
        or runtime.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("runtime twin is not exact")
    if (gates / "SUPERVISOR_plain.json").exists() or (
        output / "lanes" / "plain" / "HEARTBEAT.json"
    ).exists():
        raise RuntimeError("plain training already exists; metadata repair refused")

    archive = operations / "gate_fingerprint_repair" / repair_id / "original"
    immutable_copy(state_path, archive / state_path.name)
    immutable_copy(evaluation_path, archive / evaluation_path.name)
    immutable_copy(authorization_path, archive / authorization_path.name)
    corrected = {
        **evaluation,
        "protocol_fingerprint": required_training_protocol_fingerprint,
        "evaluation_bundle_fingerprint": required_evaluation_bundle_fingerprint,
    }
    atomic_json(evaluation_path, corrected)
    result = {
        "schema": "final-unsb-paper-repeat-gate-fingerprint-repair-v1",
        "status": "PASS_KNOWN_LEGACY_METADATA_REPAIR",
        "repair_id": repair_id,
        "host_label": host_label,
        "output": str(output),
        "original": {
            "successor_state_sha256": required_failed_successor_state_sha256,
            "evaluation_receipt_sha256": (
                required_legacy_evaluation_receipt_sha256
            ),
            "authorization_sha256": required_failed_authorization_sha256,
        },
        "corrected_evaluation_receipt_sha256": file_sha256(evaluation_path),
        "runtime_receipt_sha256_before_and_after": required_runtime_receipt_sha256,
        "training_protocol_fingerprint": required_training_protocol_fingerprint,
        "evaluation_bundle_fingerprint": required_evaluation_bundle_fingerprint,
        "checkpoint_loaded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    receipt_path = operations / f"REPEAT_GATE_FINGERPRINT_REPAIR_{repair_id}.json"
    atomic_json(receipt_path, result)
    return {**result, "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path)}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-id", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--required-training-protocol-fingerprint", required=True)
    parser.add_argument("--required-evaluation-bundle-fingerprint", required=True)
    parser.add_argument("--required-failed-successor-state-sha256", required=True)
    parser.add_argument("--required-legacy-evaluation-receipt-sha256", required=True)
    parser.add_argument("--required-failed-authorization-sha256", required=True)
    parser.add_argument("--required-runtime-receipt-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    print(json.dumps(repair(
        output=args.output,
        repair_id=args.repair_id,
        host_label=args.host_label,
        required_training_protocol_fingerprint=(
            args.required_training_protocol_fingerprint
        ),
        required_evaluation_bundle_fingerprint=(
            args.required_evaluation_bundle_fingerprint
        ),
        required_failed_successor_state_sha256=(
            args.required_failed_successor_state_sha256
        ),
        required_legacy_evaluation_receipt_sha256=(
            args.required_legacy_evaluation_receipt_sha256
        ),
        required_failed_authorization_sha256=(
            args.required_failed_authorization_sha256
        ),
        required_runtime_receipt_sha256=args.required_runtime_receipt_sha256,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
