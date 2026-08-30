from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from operations.local_route1_frontier_terminal_successor import validate_completion
from research.local_route1.frontier_adjudication import (
    FRONTIER_IDS,
    NO_REPLAY_STATUS,
    REPLAY_STATUS,
    adjudicate_frontier,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.protocol import ROOT, file_sha256


def _receipt(
    tmp_path: Path, candidate_id: str, late: float, *, status: str,
) -> Path:
    path = tmp_path / f"{candidate_id}.json"
    path.write_text(json.dumps({
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": status,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "plain_e200_verification_sha256": "plain",
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


def test_frontier_strict_pass_recommends_only_best_for_4090(tmp_path: Path) -> None:
    first = _receipt(tmp_path, FRONTIER_IDS[0], 0.2, status=POSITIVE_STATUS)
    second = _receipt(tmp_path, FRONTIER_IDS[1], 0.4, status=POSITIVE_STATUS)
    result = adjudicate_frontier([first, second], tmp_path / "result.json")
    assert result["status"] == REPLAY_STATUS
    assert result["recommended_4090_replay_candidate_id"] == FRONTIER_IDS[1]
    assert result["strict_gate_pass_candidate_ids"] == [FRONTIER_IDS[1], FRONTIER_IDS[0]]
    assert result["cross_host_deltas_merged"] is False


def test_frontier_all_negative_preserves_rank_without_replay(tmp_path: Path) -> None:
    first = _receipt(tmp_path, FRONTIER_IDS[0], -0.2, status=NEGATIVE_STATUS)
    second = _receipt(tmp_path, FRONTIER_IDS[1], -0.1, status=NEGATIVE_STATUS)
    result = adjudicate_frontier([first, second], tmp_path / "result.json")
    assert result["status"] == NO_REPLAY_STATUS
    assert result["selected_frontier_candidate_id"] == FRONTIER_IDS[1]
    assert result["recommended_4090_replay_candidate_id"] is None


def test_frontier_rejects_receipts_from_different_same_host_plain(tmp_path: Path) -> None:
    first = _receipt(tmp_path, FRONTIER_IDS[0], 0.2, status=POSITIVE_STATUS)
    second = _receipt(tmp_path, FRONTIER_IDS[1], 0.4, status=POSITIVE_STATUS)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["plain_e200_verification_sha256"] = "other-plain"
    second.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = Path(str(second) + ".sha256.json")
    sidecar.write_text(json.dumps({
        "schema": SIDECAR_SCHEMA,
        "candidate_id": FRONTIER_IDS[1],
        "receipt_sha256": file_sha256(second),
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="same-host authority"):
        adjudicate_frontier([first, second], tmp_path / "result.json")


def test_terminal_completion_requires_exact_e200_receipt_set(tmp_path: Path) -> None:
    records = {}
    for candidate_id in FRONTIER_IDS:
        path = tmp_path / f"{candidate_id}.json"
        path.write_text("{}", encoding="utf-8")
        records[candidate_id] = {
            "path": path.name,
            "sha256": file_sha256(path),
        }
    complete = {
        "schema": "final-unsb-route1-frontier-e200-complete-v1",
        "status": "FRONTIER_E200_COMPLETE_ADJUDICATION_REQUIRED",
        "candidate_receipts": records,
        "selection_seeds": [2026],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    assert validate_completion(tmp_path, complete) == [
        tmp_path / f"{candidate_id}.json" for candidate_id in FRONTIER_IDS
    ]
    complete["candidate_receipts"].pop(FRONTIER_IDS[1])
    with pytest.raises(RuntimeError, match="receipt set"):
        validate_completion(tmp_path, complete)

