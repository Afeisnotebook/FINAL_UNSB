from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from research.local_route1.complete_5090_frontier import (
    STATUS,
    materialize_complete_5090_frontier,
)
from research.local_route1.cross_runtime_portfolio import RESULT_SCHEMA
from research.local_route1.extended_repaired_frontier import (
    SCHEMA as EXTENDED_SCHEMA,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.protocol import ROOT, file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipt(root: Path, candidate_id: str, late: float) -> Path:
    trajectory = _write(
        root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json",
        {
            "candidate_id": candidate_id,
            "status": POSITIVE_STATUS,
            "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
        },
    )
    path = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    _write(path, {
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": POSITIVE_STATUS,
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
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": file_sha256(trajectory),
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
    })
    _write(Path(str(path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    })
    return path


def test_complete_5090_frontier_adds_two_cross_runtime_candidates(
    tmp_path: Path,
) -> None:
    old = [_receipt(tmp_path, f"OLD-{index}", 0.1 - index) for index in range(3)]
    cross = [_receipt(tmp_path, "STRICT-REPLAY", 0.8), _receipt(tmp_path, "NEAR-REPLAY", 0.3)]
    _write(tmp_path / "operations" / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json", {
        "schema": EXTENDED_SCHEMA,
        "ranking": [
            {
                "candidate_id": json.loads(path.read_text())["candidate_id"],
                "receipt_path": str(path.resolve()),
                "receipt_sha256": file_sha256(path),
            }
            for path in old
        ],
        "parent_ablation_results": [],
        "observable_only_candidate_ids_excluded_from_ranking": [],
        "paired_controller_access": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "operations" / "CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json", {
        "schema": RESULT_SCHEMA,
        "status": "CROSS_RUNTIME_5090_PORTFOLIO_COMPLETE_E200",
        "candidate_results": [
            {
                "candidate_id": json.loads(path.read_text())["candidate_id"],
                "receipt_path": str(path.resolve()),
                "receipt_sha256": file_sha256(path),
            }
            for path in cross
        ],
        "paired_controller_access": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    })

    result = materialize_complete_5090_frontier(tmp_path)
    assert result["status"] == STATUS
    assert result["action_priority_candidate_id"] == "STRICT-REPLAY"
    assert result["rankable_complete_e200_candidate_count"] == 5
    assert {row["source_role"] for row in result["ranking"]} == {
        "5090_repaired_frontier",
        "4090_evidence_qualified_5090_replay",
    }


def test_complete_5090_frontier_rejects_missing_two_replays(tmp_path: Path) -> None:
    old = [_receipt(tmp_path, f"OLD-{index}", 0.1) for index in range(3)]
    replay = _receipt(tmp_path, "ONLY-ONE", 0.2)
    _write(tmp_path / "operations" / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json", {
        "schema": EXTENDED_SCHEMA,
        "ranking": [
            {"receipt_path": str(path), "receipt_sha256": file_sha256(path)}
            for path in old
        ],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "operations" / "CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json", {
        "schema": RESULT_SCHEMA,
        "status": "CROSS_RUNTIME_5090_PORTFOLIO_COMPLETE_E200",
        "candidate_results": [{
            "receipt_path": str(replay), "receipt_sha256": file_sha256(replay),
        }],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    with pytest.raises(RuntimeError, match="five expected"):
        materialize_complete_5090_frontier(tmp_path)
