"""Create a source-bound terminal receipt inside one candidate's frozen code identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.local_route1.candidates import load_candidate_registration
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
    _median_epoch_seconds,
    _validate_terminal_artifacts,
    _validate_trajectory,
)
from research.local_route1.protocol import ROOT, file_sha256, git_commit, load_protocol
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-candidate-terminal-receipt-v1"
SIDECAR_SCHEMA = "final-unsb-route1-candidate-terminal-receipt-sidecar-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/generation1_adjudication.py",
    "research/local_route1/candidates.py",
    "research/local_route1/candidate_runner.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def build_receipt(output_root: Path, candidate_id: str) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    registration = load_candidate_registration(
        output_root, candidate_id, require_gate=True,
    )
    candidate_root = output_root / "candidates" / candidate_id
    trajectory_path = candidate_root / "CANDIDATE_TRAJECTORY.json"
    if not trajectory_path.is_file():
        raise RuntimeError("terminal receipt requires a completed candidate trajectory")
    trajectory = _read_json(trajectory_path)
    _validate_trajectory(
        trajectory=trajectory,
        candidate_id=candidate_id,
        algorithm_fingerprint=registration.algorithm_fingerprint,
    )
    if trajectory.get("status") not in (POSITIVE_STATUS, NEGATIVE_STATUS):
        raise RuntimeError("terminal receipt requires a scientifically adjudicable e200 status")

    terminal = _validate_terminal_artifacts(
        output_root=output_root,
        candidate_id=candidate_id,
        registration=registration,
    )
    latest_sidecar = _read_json(candidate_root / "full_state_latest.pt.json")
    metadata = latest_sidecar.get("metadata", {})
    if (
        metadata.get("candidate_training_core_fingerprint")
        != registration.candidate_training_core_fingerprint
    ):
        raise RuntimeError("terminal training-core fingerprint changed")
    training_git_commit = str(metadata.get("training_git_commit", ""))
    if len(training_git_commit) != 40:
        raise RuntimeError("terminal checkpoint lacks its frozen training commit")

    protocol = load_protocol()
    plain_verification_path = (
        output_root / "operations" / "milestone_verifications"
        / "PLAIN_E200_VERIFICATION.json"
    )
    if not plain_verification_path.is_file():
        raise RuntimeError("same-host plain e200 verification is missing")
    plain_verification = _read_json(plain_verification_path)
    if (
        plain_verification.get("status") != "ACCEPTED_MILESTONE"
        or plain_verification.get("identity", {}).get("probe_id") != "plain"
        or int(plain_verification.get("identity", {}).get("data_epoch", -1)) != 200
    ):
        raise RuntimeError("same-host plain e200 is not accepted")

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
    sources = {relative: file_sha256(ROOT / relative) for relative in SOURCE_RELATIVES}
    return {
        "schema": SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": trajectory["status"],
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "candidate_training_core_fingerprint": registration.candidate_training_core_fingerprint,
        "base_e0_scientific_state_sha256": registration.base_e0_scientific_state_sha256,
        "base_protocol_fingerprint": registration.base_protocol_fingerprint,
        "manifest_sha256": protocol["manifest"]["sha256"],
        "training_git_commit": training_git_commit,
        "verification_git_commit": git_commit(),
        "verification_source_sha256": sources,
        "receipt_source_sha256": sources[SOURCE_RELATIVES[0]],
        "hypothesis_freeze_sha256": registration.hypothesis_freeze_sha256,
        "derivation_card_sha256": file_sha256(registration.card_path),
        "implementation_sha256": file_sha256(registration.implementation_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "ranking_fields": ranking_fields,
        "median_epoch_wall_seconds": _median_epoch_seconds(candidate_root),
        "terminal_integrity": terminal,
        "plain_e200_verification_path": str(plain_verification_path),
        "plain_e200_verification_sha256": file_sha256(plain_verification_path),
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }


def materialize_receipt(
    output_root: Path, candidate_id: str, receipt_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    path = (
        output_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
        if receipt_path is None else Path(receipt_path).resolve()
    )
    receipt = build_receipt(output_root, candidate_id)
    write_json(path, receipt)
    write_json(Path(str(path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        materialize_receipt(args.output, args.candidate_id, args.receipt),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
