from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from operations.local_route1_implementation_invalid_diagnostic_receipt import (
    DIAGNOSTIC_RECEIPT_SCHEMA,
    DIAGNOSTIC_SIDECAR_SCHEMA,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.repaired_frontier_adjudication import (
    ACTIONABLE_STATUS,
    FALLBACK_STATUS,
    INVALID_DIAGNOSTIC_INCIDENTS,
    RANKABLE_IDS,
    adjudicate_repaired_frontier,
)


INVALID_FINGERPRINTS = {
    candidate_id: json.loads(
        (ROOT / relative).read_text(encoding="utf-8")
    )["invalid_identity"]["algorithm_fingerprint"]
    for candidate_id, relative in INVALID_DIAGNOSTIC_INCIDENTS.items()
}


def _receipt(
    tmp_path: Path,
    candidate_id: str,
    late: float,
    *,
    status: str,
    algorithm_fingerprint: str | None = None,
    plain: str = "plain",
) -> Path:
    trajectory_path = tmp_path / f"{candidate_id}.trajectory.json"
    trajectory_path.write_text(json.dumps({
        "candidate_id": candidate_id,
        "status": status,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
    }), encoding="utf-8")
    path = tmp_path / f"{candidate_id}.json"
    path.write_text(json.dumps({
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": status,
        "algorithm_fingerprint": algorithm_fingerprint or f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "plain_e200_verification_sha256": plain,
        "training_git_commit": "a" * 40,
        "receipt_source_sha256": file_sha256(
            ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
        ),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": late,
            "e200_macro_psnr_delta": late,
            "late_points_with_four_of_six_positive_domains": 2,
            "late_average_worst_domain_delta": -0.2,
            "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
            "late_mean_macro_ssim_delta": 0.01,
            "late_mean_macro_lpips_delta": -0.01,
        },
        "median_epoch_wall_seconds": 1.0,
        "terminal_integrity": {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    Path(str(path) + ".sha256.json").write_text(json.dumps({
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    }), encoding="utf-8")
    return path


def _diagnostic_receipt(
    tmp_path: Path, candidate_id: str, late: float,
) -> Path:
    path = tmp_path / f"{candidate_id}.diagnostic.json"
    path.write_text(json.dumps({
        "schema": DIAGNOSTIC_RECEIPT_SCHEMA,
        "status": "ACCEPTED_IMPLEMENTATION_INVALID_COMPLETE_E200_DIAGNOSTIC",
        "candidate_id": candidate_id,
        "trajectory_status": POSITIVE_STATUS,
        "algorithm_fingerprint": INVALID_FINGERPRINTS[candidate_id],
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "plain_e200_verification_sha256": "plain",
        "training_git_commit": "a" * 40,
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": late,
            "e200_macro_psnr_delta": late,
        },
        "terminal_integrity": {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
        "receipt_source_sha256": file_sha256(
            ROOT / "operations"
            / "local_route1_implementation_invalid_diagnostic_receipt.py"
        ),
        "scientific_ranking_eligible": False,
        "parent_mechanism_falsified": False,
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    Path(str(path) + ".sha256.json").write_text(json.dumps({
        "schema": DIAGNOSTIC_SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    }), encoding="utf-8")
    return path


def _frontier(
    tmp_path: Path, *, all_negative: bool = False,
) -> tuple[list[Path], list[Path]]:
    lates = (-0.1, -0.2, -0.3) if all_negative else (0.1, 0.4, 0.2)
    statuses = (
        (NEGATIVE_STATUS,) * 3 if all_negative
        else (POSITIVE_STATUS,) * 3
    )
    rankable = [
        _receipt(tmp_path, candidate_id, late, status=status)
        for candidate_id, late, status in zip(RANKABLE_IDS, lates, statuses)
    ]
    invalid = [
        _diagnostic_receipt(tmp_path, candidate_id, 9.0)
        for candidate_id in INVALID_DIAGNOSTIC_INCIDENTS
    ]
    return rankable, invalid


def test_repaired_frontier_preserves_multiple_candidates_and_excludes_invalid(
    tmp_path: Path,
) -> None:
    rankable, invalid = _frontier(tmp_path)
    result = adjudicate_repaired_frontier(
        rankable, invalid, tmp_path / "adjudication.json",
    )
    assert result["status"] == ACTIONABLE_STATUS
    assert result["action_priority_candidate_id"] == RANKABLE_IDS[1]
    assert result["priority_alternate_candidate_ids"] == [RANKABLE_IDS[2], RANKABLE_IDS[0]]
    assert result["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert len(result["ranking"]) == 3
    assert all(
        row["candidate_id"] not in INVALID_DIAGNOSTIC_INCIDENTS
        for row in result["ranking"]
    )
    assert len(result["implementation_invalid_diagnostics"]) == 2
    assert all(
        row["scientific_ranking_eligible"] is False
        for row in result["implementation_invalid_diagnostics"]
    )


def test_repaired_frontier_all_negative_retains_actionable_fallbacks(
    tmp_path: Path,
) -> None:
    rankable, invalid = _frontier(tmp_path, all_negative=True)
    result = adjudicate_repaired_frontier(
        rankable, invalid, tmp_path / "adjudication.json",
    )
    assert result["status"] == FALLBACK_STATUS
    assert result["action_priority_candidate_id"] == RANKABLE_IDS[0]
    assert result["priority_alternate_candidate_ids"] == [RANKABLE_IDS[1], RANKABLE_IDS[2]]
    assert result["recommended_4090_replay_queue"] == []


def test_repaired_frontier_replays_complete_evidence_backed_repair_alternate(
    tmp_path: Path,
) -> None:
    rankable, invalid = _frontier(tmp_path, all_negative=True)
    repaired = rankable[1]
    payload = json.loads(repaired.read_text(encoding="utf-8"))
    payload["ranking_fields"]["late_three_mean_macro_psnr_delta"] = 0.05
    payload["ranking_fields"]["e200_macro_psnr_delta"] = -0.2
    repaired.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = Path(str(repaired) + ".sha256.json")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["receipt_sha256"] = file_sha256(repaired)
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")

    result = adjudicate_repaired_frontier(
        rankable, invalid, tmp_path / "adjudication.json",
    )
    assert result["recommended_4090_replay_queue"] == [RANKABLE_IDS[1]]
    assert RANKABLE_IDS[1] in result["evidence_preserved_candidate_ids"]


def test_repaired_frontier_rejects_invalid_diagnostic_not_bound_to_incident(
    tmp_path: Path,
) -> None:
    rankable, invalid = _frontier(tmp_path)
    payload = json.loads(invalid[0].read_text(encoding="utf-8"))
    payload["algorithm_fingerprint"] = "wrong"
    invalid[0].write_text(json.dumps(payload), encoding="utf-8")
    sidecar = Path(str(invalid[0]) + ".sha256.json")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["receipt_sha256"] = file_sha256(invalid[0])
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not bound to its incident"):
        adjudicate_repaired_frontier(rankable, invalid, tmp_path / "result.json")


def test_repaired_frontier_requires_one_same_host_plain_authority(
    tmp_path: Path,
) -> None:
    rankable, invalid = _frontier(tmp_path)
    payload = json.loads(rankable[-1].read_text(encoding="utf-8"))
    payload["plain_e200_verification_sha256"] = "other"
    rankable[-1].write_text(json.dumps(payload), encoding="utf-8")
    sidecar = Path(str(rankable[-1]) + ".sha256.json")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["receipt_sha256"] = file_sha256(rankable[-1])
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="same-host authority"):
        adjudicate_repaired_frontier(rankable, invalid, tmp_path / "result.json")
