from __future__ import annotations

import json
from pathlib import Path

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from research.local_route1.frontier_final_delivery import (
    CANDIDATE_SCHEMA,
    FINAL_SELECTION,
    POINTER_SCHEMA,
    _same_host_selection,
    materialize_frontier_final_delivery,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.protocol import ROOT, file_sha256


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _receipt(
    root: Path, candidate_id: str, late: float, *, status: str,
    commit: str = "a" * 40,
) -> Path:
    path = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    trajectory = root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    _write(trajectory, {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "candidate_id": candidate_id,
        "status": status,
        "trajectory": [{"epoch": 200, "macro_psnr_delta": late}],
        "confirmation20_opened": False,
    })
    payload = {
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
        "training_git_commit": commit,
        "receipt_source_sha256": file_sha256(
            ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
        ),
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
    }
    _write(path, payload)
    _write(Path(str(path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    })
    return path


def _source_and_contract(root: Path, candidate_id: str, commit: str = "a" * 40) -> None:
    _write(root / "derive" / "cards" / f"{candidate_id}.json", {
        "candidate_id": candidate_id,
        "name": f"Name {candidate_id}",
        "unsb_object": "object",
        "formula": "formula",
        "identity_or_unbiased_condition": "identity",
        "objective_change": False,
        "estimator_change": True,
        "coordinate_change": False,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": "proof",
        "compute_cost": "cost",
        "memory_cost": "memory",
        "recovery_state_cost": "state",
        "expected_applicable_state": "state",
        "falsifying_experiment": "falsifier",
    })
    _write(root / "derive" / "implementations" / f"{candidate_id}.json", {
        "candidate_id": candidate_id,
    })
    _write(root / "operations" / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json", {
        "candidate_id": candidate_id,
        "candidate_git_commit": commit,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
    })


def test_same_host_replay_can_replace_prefrontier_fallback(tmp_path: Path) -> None:
    base = _receipt(tmp_path, "BASE", -0.1, status=NEGATIVE_STATUS)
    replay = _receipt(tmp_path, "REPLAY", 0.2, status=POSITIVE_STATUS)
    selection, path, receipt = _same_host_selection(tmp_path, "BASE", {
        "status": "COMPLETE_ONE_FRONTIER_4090_REPLAY_ADJUDICATION_REQUIRED",
        "receipt_path": str(replay),
        "receipt_sha256": file_sha256(replay),
    })
    assert selection["selected_candidate_id"] == "REPLAY"
    assert path == replay
    assert receipt["candidate_id"] == "REPLAY"
    assert selection["cross_host_deltas_merged"] is False
    assert base.is_file()


def test_no_replay_materializes_idempotent_frontier_complete_delivery(
    tmp_path: Path,
) -> None:
    candidate_id = "BASE"
    _receipt(tmp_path, candidate_id, -0.1, status=NEGATIVE_STATUS)
    _source_and_contract(tmp_path, candidate_id)
    final = tmp_path / "final"
    _write(final / "CANDIDATE.json", {
        "candidate_id": candidate_id,
        "confirmation20_opened": False,
        "paired_metrics_used_for_training_or_control": False,
    })
    _write(final / "RESULTS.json", {
        "selected_candidate_id": candidate_id,
        "winner_ablation_results": {"full": "evidence"},
        "confirmation20_opened": False,
        "paired_metrics_used_for_training_or_control": False,
    })
    _write(final / "ALTERNATES.json", {
        "selected_candidate_id": candidate_id,
        "alternates": [{
            "candidate_id": "BASE-ALT",
            "role": "tested_alternate",
            "trajectory_status": NEGATIVE_STATUS,
            "reason_not_selected": "lower rank",
        }],
        "confirmation20_opened": False,
    })
    (final / "FINAL_ROUTE1_REPORT.md").write_text("base report", encoding="utf-8")
    decision = {
        "schema": "final-unsb-route1-frontier-4090-replay-decision-v1",
        "status": "NO_4090_REPLAY_FRONTIER_CURRENT_IMPLEMENTATIONS_NEGATIVE",
        "recommended_candidate_id": None,
        "recommended_algorithm_fingerprint": None,
        "complete_e200_only": True,
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    adjudication = {
        "schema": "final-unsb-route1-frontier-e200-adjudication-v1",
        "selected_frontier_candidate_id": "F1-REMOTE",
        "recommended_4090_replay_candidate_id": None,
        "recommended_4090_replay_algorithm_fingerprint": None,
        "ranking": [{
            "candidate_id": "F1-REMOTE",
            "trajectory_status": NEGATIVE_STATUS,
        }],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(tmp_path / "operations" / "FRONTIER_5090_TERMINAL_ENVELOPE.json", {
        "decision": decision,
        "adjudication": adjudication,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    cross_result = {
        "schema": "final-unsb-route1-frontier-cross-host-result-v1",
        "status": "COMPLETE_REMOTE_FRONTIER_NEGATIVE_4090_REPLAY_SKIPPED",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(tmp_path / "operations" / "FRONTIER_CROSS_HOST_RESULT.json", cross_result)
    selection, receipt_path, _ = _same_host_selection(
        tmp_path, candidate_id, cross_result,
    )
    selection_path = tmp_path / "operations" / FINAL_SELECTION
    ablation_path = tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
    _write(ablation_path, {
        "schema": "final-unsb-route1-winner-ablation-adjudication-v1",
        "status": "COMPLETE_NO_SELECTION_CHANGE",
        "roles": {"projected_or_full": {"candidate_id": candidate_id}},
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "operations" / "FRONTIER_WINNER_ABLATION_RESULT.json", {
        "schema": "final-unsb-route1-frontier-winner-ablation-result-v1",
        "status": "REUSED_PRE_FRONTIER_SELECTED_WINNER_ABLATIONS",
        "selected_candidate_id": candidate_id,
        "selected_receipt_path": str(receipt_path),
        "selected_receipt_sha256": file_sha256(receipt_path),
        "post_ablation_selection_path": str(selection_path),
        "post_ablation_selection_sha256": file_sha256(selection_path),
        "post_ablation_selection": selection,
        "winner_ablation_adjudication_path": str(ablation_path),
        "winner_ablation_adjudication_sha256": file_sha256(ablation_path),
        "winner_ablation_evidence": {"projected_or_full": "evidence"},
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    pointer = materialize_frontier_final_delivery(tmp_path)
    assert pointer["schema"] == POINTER_SCHEMA
    assert pointer["selected_candidate_id"] == candidate_id
    candidate = json.loads((final / "CANDIDATE.json").read_text(encoding="utf-8"))
    assert candidate["schema"] == CANDIDATE_SCHEMA
    assert candidate["candidate_id"] == candidate_id
    assert (final / "pre_frontier_delivery" / "CANDIDATE.json").is_file()
    assert len(json.loads((final / "ALTERNATES.json").read_text())["alternates"]) == 2
    assert materialize_frontier_final_delivery(tmp_path) == pointer
