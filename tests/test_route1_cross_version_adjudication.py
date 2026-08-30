from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from operations.local_route1_cross_version_adjudicate import adjudicate
from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_NEGATIVE_STATUS,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.protocol import ROOT, file_sha256


def _receipt(
    tmp_path: Path, candidate_id: str, late: float,
    *, trajectory_status: str = POSITIVE_STATUS,
) -> Path:
    path = tmp_path / f"{candidate_id}.json"
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": trajectory_status,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "training_git_commit": "a" * 40,
        "receipt_source_sha256": file_sha256(
            ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
        ),
        "trajectory_sha256": f"trajectory-{candidate_id}",
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
        "terminal_integrity": {
            "status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET",
        },
        "plain_e200_verification_sha256": "plain",
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    Path(str(path) + ".sha256.json").write_text(json.dumps({
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    }), encoding="utf-8")
    return path


def test_cross_version_receipts_rank_without_loading_candidate_code(tmp_path):
    low = _receipt(tmp_path, "G1-LOW", 0.1)
    high = _receipt(tmp_path, "G1-HIGH", 0.3)
    result = adjudicate([low, high], tmp_path / "result.json")
    assert result["selected_candidate_id"] == "G1-HIGH"
    assert result["winner_not_loaded_under_a_different_training_core"] is True
    assert result["seed_freeze_performed"] is False
    assert [row["candidate_id"] for row in result["ranking"]] == ["G1-HIGH", "G1-LOW"]


def test_cross_version_receipts_reject_different_plain_authority(tmp_path):
    first = _receipt(tmp_path, "G1-A", 0.1)
    second = _receipt(tmp_path, "G1-B", 0.2)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["plain_e200_verification_sha256"] = "different"
    second.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = Path(str(second) + ".sha256.json")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["receipt_sha256"] = file_sha256(second)
    sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="plain_e200_verification_sha256"):
        adjudicate([first, second], tmp_path / "result.json")


def test_cross_version_all_negative_emits_the_negative_successor_status(tmp_path):
    first = _receipt(
        tmp_path, "G1-A", -0.1, trajectory_status=NEGATIVE_STATUS,
    )
    second = _receipt(
        tmp_path, "G1-B", -0.2, trajectory_status=NEGATIVE_STATUS,
    )
    result = adjudicate([first, second], tmp_path / "result.json")
    assert result["status"] == CROSS_VERSION_NEGATIVE_STATUS
    assert result["selection_role"] == "current_best_fallback"
