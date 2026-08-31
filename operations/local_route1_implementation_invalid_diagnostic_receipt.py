"""Materialize a complete e200 receipt that is permanently ineligible to rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
    _median_epoch_seconds,
    _validate_terminal_artifacts,
    _validate_trajectory,
)
from research.local_route1.protocol import ROOT, file_sha256, load_protocol
from research.local_route1.runtime import write_json


DIAGNOSTIC_RECEIPT_SCHEMA = (
    "final-unsb-route1-implementation-invalid-diagnostic-receipt-v1"
)
DIAGNOSTIC_SIDECAR_SCHEMA = (
    "final-unsb-route1-implementation-invalid-diagnostic-receipt-sidecar-v1"
)
INCIDENTS = {
    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER": (
        "evidence/remote_route1_offload/"
        "AMMCRB_ABSOLUTE_MARGIN_SEMANTIC_INCIDENT_20260831.json"
    ),
    "G1-03-STATE-FEEDBACK-MISSING": (
        "evidence/remote_route1_offload/"
        "MCRB_ABSOLUTE_MARGIN_SEMANTIC_INCIDENT_20260831.json"
    ),
}
SOURCE_RELATIVES = (
    "operations/local_route1_implementation_invalid_diagnostic_receipt.py",
    "research/local_route1/generation1_adjudication.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def build_diagnostic_receipt(output_root: Path, candidate_id: str) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if candidate_id not in INCIDENTS:
        raise RuntimeError("candidate has no frozen implementation semantic incident")
    incident_path = (ROOT / INCIDENTS[candidate_id]).resolve()
    incident = _read_json(incident_path)
    if (
        incident.get("schema") != "final-unsb-route1-semantic-incident-v1"
        or incident.get("candidate_id") != candidate_id
        or incident.get("classification") != "implementation_failure"
        or incident.get("scientific_conclusion_allowed") is not False
        or incident.get("parent_mechanism_falsified") is not False
        or incident.get("paired_metric_used_for_discovery_or_repair") is not False
        or incident.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("semantic incident does not authorize a diagnostic receipt")

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    matches = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError("implementation-invalid candidate ledger identity is not unique")
    record = matches[0]
    binding = record.get("engineering_incident") or {}
    if (
        record.get("status") != "IMPLEMENTATION_INVALID"
        or record.get("scientific_result_admissible") is not False
        or binding.get("path") != INCIDENTS[candidate_id]
        or binding.get("sha256") != file_sha256(incident_path)
    ):
        raise RuntimeError("ledger does not bind the implementation-invalid incident")

    candidate_root = output_root / "candidates" / candidate_id
    trajectory_path = candidate_root / "CANDIDATE_TRAJECTORY.json"
    trajectory = _read_json(trajectory_path)
    algorithm_fingerprint = str(
        (incident.get("invalid_identity") or {}).get("algorithm_fingerprint", "")
    )
    _validate_trajectory(
        trajectory=trajectory,
        candidate_id=candidate_id,
        algorithm_fingerprint=algorithm_fingerprint,
    )
    if trajectory.get("status") not in (POSITIVE_STATUS, NEGATIVE_STATUS):
        raise RuntimeError("diagnostic trajectory is not a complete e200 outcome")
    candidate_fingerprint = str(trajectory.get("candidate_fingerprint", ""))
    latest_sidecar = _read_json(candidate_root / "full_state_latest.pt.json")
    metadata = latest_sidecar.get("metadata") or {}
    authority = SimpleNamespace(
        algorithm_fingerprint=algorithm_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        base_protocol_fingerprint=str(metadata.get("base_protocol_fingerprint", "")),
    )
    terminal = _validate_terminal_artifacts(
        output_root=output_root,
        candidate_id=candidate_id,
        registration=authority,
    )
    execution_state_path = (
        output_root / "operations" / f"CANDIDATE_EXECUTION_STATE_{candidate_id}.json"
    )
    execution = _read_json(execution_state_path)
    if (
        execution.get("status") != "CANDIDATE_E200_COMPLETE_ADJUDICATION_REQUIRED"
        or execution.get("candidate_id") != candidate_id
        or int(execution.get("data_epoch", -1)) != 200
        or int(execution.get("updates", -1)) != 30000
        or execution.get("algorithm_fingerprint") != algorithm_fingerprint
        or execution.get("candidate_fingerprint") != candidate_fingerprint
        or execution.get("paired_controller_access") is not False
        or execution.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("implementation-invalid executor state is not complete e200")

    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    invalid_identity = incident["invalid_identity"]
    if file_sha256(card_path) != invalid_identity.get("derivation_card_sha256"):
        raise RuntimeError("diagnostic derivation card differs from the semantic incident")
    implementation = _read_json(implementation_path)
    projection = str(invalid_identity.get("projection_source", ""))
    source_rows = {
        str(row.get("path")): str(row.get("sha256"))
        for row in implementation.get("source_files", []) if isinstance(row, dict)
    }
    if source_rows.get(projection) != invalid_identity.get("projection_source_sha256"):
        raise RuntimeError("diagnostic implementation source differs from the incident")

    ranking_fields = {
        key: trajectory.get(key)
        for key in (
            "late_three_mean_macro_psnr_delta",
            "e200_macro_psnr_delta",
            "late_points_with_four_of_six_positive_domains",
            "late_average_worst_domain_delta",
            "candidate_best_to_terminal_three_point_rolling_drawdown",
            "late_mean_macro_ssim_delta",
            "late_mean_macro_lpips_delta",
        )
    }
    protocol = load_protocol()
    plain_verification_path = (
        output_root / "operations" / "milestone_verifications"
        / "PLAIN_E200_VERIFICATION.json"
    )
    return {
        "schema": DIAGNOSTIC_RECEIPT_SCHEMA,
        "status": "ACCEPTED_IMPLEMENTATION_INVALID_COMPLETE_E200_DIAGNOSTIC",
        "candidate_id": candidate_id,
        "trajectory_status": trajectory["status"],
        "algorithm_fingerprint": algorithm_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "candidate_training_core_fingerprint": metadata[
            "candidate_training_core_fingerprint"
        ],
        "base_e0_scientific_state_sha256": metadata[
            "base_e0_scientific_state_sha256"
        ],
        "base_protocol_fingerprint": metadata["base_protocol_fingerprint"],
        "manifest_sha256": protocol["manifest"]["sha256"],
        "training_git_commit": metadata["training_git_commit"],
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "ranking_fields": ranking_fields,
        "median_epoch_wall_seconds": _median_epoch_seconds(candidate_root),
        "terminal_integrity": terminal,
        "plain_e200_verification_path": str(plain_verification_path),
        "plain_e200_verification_sha256": file_sha256(plain_verification_path),
        "incident_path": incident_path.relative_to(ROOT).as_posix(),
        "incident_sha256": file_sha256(incident_path),
        "derivation_card_sha256": file_sha256(card_path),
        "implementation_sha256": file_sha256(implementation_path),
        "execution_state_path": str(execution_state_path),
        "execution_state_sha256": file_sha256(execution_state_path),
        "receipt_source_sha256": file_sha256(ROOT / SOURCE_RELATIVES[0]),
        "scientific_ranking_eligible": False,
        "parent_mechanism_falsified": False,
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def materialize_diagnostic_receipt(
    output_root: Path, candidate_id: str, receipt_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    path = (
        output_root / "operations" / "diagnostic_receipts" / f"{candidate_id}.json"
        if receipt_path is None else Path(receipt_path).resolve()
    )
    receipt = build_diagnostic_receipt(output_root, candidate_id)
    write_json(path, receipt)
    write_json(Path(str(path) + ".sha256.json"), {
        "schema": DIAGNOSTIC_SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
        "scientific_ranking_eligible": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", choices=sorted(INCIDENTS), required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        materialize_diagnostic_receipt(args.output, args.candidate_id, args.receipt),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
