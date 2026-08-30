from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from research.local_route1.frontier_final_delivery import (
    CANDIDATE_SCHEMA,
    FINAL_SELECTION,
    POINTER_SCHEMA,
    _executor_contract,
    _frontier_challenger_ablation_results,
    _same_host_selection,
    _source_files,
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
    domains = [
        "FoggyCityscapes", "LowLightTrafficData", "RSCityscapes",
        "RainCityscapes", "RainDS-syn", "SnowTrafficData",
    ]
    plain_metric = {
        "schema": "evaluation-v1", "count_per_domain": 70,
        "protocol_fingerprint": "protocol", "evaluation_input_sha256": "crn",
        "macro_psnr": 10.0, "macro_ssim": 0.5, "macro_lpips": 0.3,
        "domains": {
            domain: {"psnr": 10.0, "ssim": 0.5, "lpips": 0.3}
            for domain in domains
        },
        "confirmation20_opened": False,
    }
    candidate_metric = {
        **plain_metric,
        "macro_psnr": 10.0 + late,
        "macro_ssim": 0.51,
        "macro_lpips": 0.29,
        "domains": {
            domain: {"psnr": 10.0 + late, "ssim": 0.51, "lpips": 0.29}
            for domain in domains
        },
    }
    _write(root / "anchors" / "plain" / "metrics" / "e200.json", plain_metric)
    _write(root / "candidates" / candidate_id / "metrics" / "e200.json", candidate_metric)
    _write(trajectory, {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "candidate_id": candidate_id,
        "status": status,
        "trajectory": [{
            "epoch": 200, "updates": 30000,
            "macro_psnr": 10.0 + late, "plain_macro_psnr": 10.0,
            "macro_psnr_delta": late,
        }],
        "paired_metrics_used_for_training_or_gate": False,
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
        "algorithm_hyperparameters": {"strength": 1.0},
    })
    implementation_path = root / "derive" / "implementations" / f"{candidate_id}.json"
    _write(implementation_path, {
        "candidate_id": candidate_id,
        "model": "route1_test",
        "method": {"enabled": True},
        "source_files": [{"path": "src/models/test.py", "sha256": "source"}],
    })
    _write(root / "operations" / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json", {
        "schema": "final-unsb-route1-candidate-executor-contract-v1",
        "candidate_id": candidate_id,
        "candidate_git_commit": commit,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "manifest_sha256": "manifest",
        "target_data_epochs": 200,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    receipt_path = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["derivation_card_sha256"] = file_sha256(
        root / "derive" / "cards" / f"{candidate_id}.json"
    )
    receipt["implementation_sha256"] = file_sha256(implementation_path)
    _write(receipt_path, receipt)
    _write(Path(str(receipt_path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(receipt_path),
    })


def _role_row(root: Path, candidate_id: str) -> dict:
    receipt_path = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "candidate_id": candidate_id,
        "trajectory_status": receipt["trajectory_status"],
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


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


def test_final_source_and_executor_contract_are_receipt_bound(tmp_path: Path) -> None:
    candidate_id = "BOUND"
    receipt_path = _receipt(tmp_path, candidate_id, 0.2, status=POSITIVE_STATUS)
    _source_and_contract(tmp_path, candidate_id)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _source_files(tmp_path, candidate_id, receipt)
    _executor_contract(tmp_path, receipt)

    card_path = tmp_path / "derive" / "cards" / f"{candidate_id}.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["formula"] = "tampered"
    _write(card_path, card)
    with pytest.raises(RuntimeError, match="derivation card changed"):
        _source_files(tmp_path, candidate_id, receipt)

    contract_path = (
        tmp_path / "operations" / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["target_data_epochs"] = 199
    _write(contract_path, contract)
    with pytest.raises(RuntimeError, match="executor contract mismatch"):
        _executor_contract(tmp_path, receipt)


def test_no_replay_materializes_idempotent_frontier_complete_delivery(
    tmp_path: Path,
) -> None:
    candidate_id = "BASE"
    _receipt(tmp_path, candidate_id, -0.1, status=NEGATIVE_STATUS)
    _source_and_contract(tmp_path, candidate_id)
    proposal_id = "BASE-PROPOSAL"
    observable_id = "BASE-OBSERVABLE"
    _receipt(tmp_path, proposal_id, -0.2, status=NEGATIVE_STATUS)
    _receipt(tmp_path, observable_id, 0.0, status=NEGATIVE_STATUS)
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
    roles = {
        "proposal_only": _role_row(tmp_path, proposal_id),
        "observable_only": _role_row(tmp_path, observable_id),
        "projected_or_full": _role_row(tmp_path, candidate_id),
    }
    _write(ablation_path, {
        "schema": "final-unsb-route1-winner-ablation-adjudication-v1",
        "status": "COMPLETE_NO_SELECTION_CHANGE",
        "roles": roles,
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
        "winner_ablation_evidence": roles,
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
    assert candidate["selected_fixed_checkpoint"] == {
        "data_epoch": 200, "updates": 30000, "best_checkpoint_selection": False,
    }
    assert candidate["absolute_relative_domain_trajectory"][-1]["data_epoch"] == 200
    assert set(candidate["ablation_evidence"]["experimental_results"]) == {
        "proposal_only", "observable_only", "projected_or_full",
    }
    assert candidate["conclusion_boundaries"]["proxy_distortion"][
        "cross_host_delta_merge"
    ] is False
    assert "python -m operations.local_route1_candidate_executor" in candidate[
        "reproduction"
    ]["seed2026_e200"]
    assert (final / "pre_frontier_delivery" / "CANDIDATE.json").is_file()
    assert len(json.loads((final / "ALTERNATES.json").read_text())["alternates"]) == 2
    results = json.loads((final / "RESULTS.json").read_text(encoding="utf-8"))
    assert results["host_separated_complete_frontier"]["cross_host_deltas_merged"] is False
    report = (final / "FINAL_ROUTE1_REPORT.md").read_text(encoding="utf-8")
    for heading in ("## 科学结论", "## 工程失败与科学结果的边界", "## 代理失真边界", "## 尚未验证"):
        assert heading in report
    assert "晚期差距收窄是耐久性改善证据" in report
    assert materialize_frontier_final_delivery(tmp_path) == pointer


def test_reselected_base_retains_complete_frontier_challenger_ablations(
    tmp_path: Path,
) -> None:
    full_id = "F1-FULL"
    proposal_id = "F1-PROPOSAL"
    observable_id = "F1-OBSERVABLE"
    for candidate_id in (full_id, proposal_id, observable_id):
        _receipt(
            tmp_path, candidate_id, 0.1,
            status=POSITIVE_STATUS if candidate_id != observable_id else NEGATIVE_STATUS,
        )
    roles = {
        "proposal_only": _role_row(tmp_path, proposal_id),
        "observable_only": _role_row(tmp_path, observable_id),
        "projected_or_full": _role_row(tmp_path, full_id),
    }
    path = tmp_path / "operations" / "FRONTIER_WINNER_ABLATION_ADJUDICATION.json"
    adjudication = {
        "roles": roles,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    _write(path, adjudication)
    result = {
        "status": "PRE_FRONTIER_SELECTED_WINNER_RETAINED_AFTER_FRONTIER_ABLATIONS",
        "frontier_full_candidate_id": full_id,
        "frontier_challenger_ablation_adjudication_path": str(path),
        "frontier_challenger_ablation_adjudication_sha256": file_sha256(path),
        "frontier_challenger_ablation_evidence": roles,
    }

    retained = _frontier_challenger_ablation_results(tmp_path, result)

    assert retained is not None
    assert retained["frontier_full_candidate_id"] == full_id
    assert set(retained["experimental_results"]) == {
        "proposal_only", "observable_only", "projected_or_full",
    }


def test_replayed_frontier_proposal_winner_keeps_own_complete_evidence(
    tmp_path: Path,
) -> None:
    base_id = "BASE"
    full_id = "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING"
    proposal_id = "ABL-F1-01-PCNR-PROPOSAL-ONLY"
    observable_id = "ABL-F1-01-PCNR-OBSERVABLE-ONLY"
    for candidate_id, late, status in (
        (base_id, -0.1, NEGATIVE_STATUS),
        (full_id, 0.2, POSITIVE_STATUS),
        (proposal_id, 0.3, POSITIVE_STATUS),
        (observable_id, 0.0, NEGATIVE_STATUS),
    ):
        _receipt(tmp_path, candidate_id, late, status=status)
    for candidate_id in (base_id, full_id, proposal_id):
        _source_and_contract(tmp_path, candidate_id)

    final = tmp_path / "final"
    _write(final / "CANDIDATE.json", {
        "candidate_id": base_id,
        "confirmation20_opened": False,
        "paired_metrics_used_for_training_or_control": False,
    })
    _write(final / "RESULTS.json", {
        "selected_candidate_id": base_id,
        "ranking": [{
            "candidate_id": base_id, "trajectory_status": NEGATIVE_STATUS,
            "ranking_fields": {"late_three_mean_macro_psnr_delta": -0.1,
                               "e200_macro_psnr_delta": -0.1},
        }],
        "winner_ablation_results": {"projected_or_full": "base"},
        "confirmation20_opened": False,
        "paired_metrics_used_for_training_or_control": False,
    })
    _write(final / "ALTERNATES.json", {
        "selected_candidate_id": base_id,
        "alternates": [],
        "confirmation20_opened": False,
    })
    (final / "FINAL_ROUTE1_REPORT.md").write_text("base report", encoding="utf-8")

    full_receipt = tmp_path / "operations" / "terminal_receipts" / f"{full_id}.json"
    decision = {
        "schema": "final-unsb-route1-frontier-4090-replay-decision-v1",
        "status": "REPLAY_REQUEST_READY_REQUIRES_4090_SOURCE_BOUND_EXECUTOR",
        "recommended_candidate_id": full_id,
        "recommended_algorithm_fingerprint": f"algorithm-{full_id}",
        "complete_e200_only": True,
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    adjudication = {
        "schema": "final-unsb-route1-frontier-e200-adjudication-v1",
        "selected_frontier_candidate_id": full_id,
        "recommended_4090_replay_candidate_id": full_id,
        "recommended_4090_replay_algorithm_fingerprint": f"algorithm-{full_id}",
        "ranking": [{
            "candidate_id": full_id, "trajectory_status": POSITIVE_STATUS,
            "ranking_fields": {"late_three_mean_macro_psnr_delta": 0.2,
                               "e200_macro_psnr_delta": 0.2},
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
    _write(tmp_path / "operations" / "FRONTIER_CROSS_HOST_RESULT.json", {
        "schema": "final-unsb-route1-frontier-cross-host-result-v1",
        "status": "COMPLETE_ONE_FRONTIER_4090_REPLAY_ADJUDICATION_REQUIRED",
        "candidate_id": full_id,
        "receipt_path": str(full_receipt),
        "receipt_sha256": file_sha256(full_receipt),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })

    roles = {
        "proposal_only": _role_row(tmp_path, proposal_id),
        "observable_only": _role_row(tmp_path, observable_id),
        "projected_or_full": _role_row(tmp_path, full_id),
    }
    ablation_path = tmp_path / "operations" / "FRONTIER_WINNER_ABLATION_ADJUDICATION.json"
    _write(ablation_path, {
        "schema": "final-unsb-route1-winner-ablation-adjudication-v1",
        "status": "SEED2026_PROPOSAL_ONLY_CHALLENGER_READY",
        "roles": roles,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    post_path = tmp_path / "operations" / "ROUTE1_FRONTIER_POST_ABLATION_SELECTION.json"
    post = {
        "selected_candidate_id": proposal_id,
        "ranking": [
            {"candidate_id": proposal_id, "trajectory_status": POSITIVE_STATUS,
             "ranking_fields": {"late_three_mean_macro_psnr_delta": 0.3,
                                "e200_macro_psnr_delta": 0.3}},
            {"candidate_id": full_id, "trajectory_status": POSITIVE_STATUS,
             "ranking_fields": {"late_three_mean_macro_psnr_delta": 0.2,
                                "e200_macro_psnr_delta": 0.2}},
            {"candidate_id": base_id, "trajectory_status": NEGATIVE_STATUS,
             "ranking_fields": {"late_three_mean_macro_psnr_delta": -0.1,
                                "e200_macro_psnr_delta": -0.1}},
        ],
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(post_path, post)
    proposal_receipt = (
        tmp_path / "operations" / "terminal_receipts" / f"{proposal_id}.json"
    )
    _write(tmp_path / "operations" / "FRONTIER_WINNER_ABLATION_RESULT.json", {
        "schema": "final-unsb-route1-frontier-winner-ablation-result-v1",
        "status": "FRONTIER_SELECTED_ALGORITHM_ABLATIONS_COMPLETE",
        "frontier_full_candidate_id": full_id,
        "selected_candidate_id": proposal_id,
        "selected_algorithm_fingerprint": f"algorithm-{proposal_id}",
        "selected_receipt_path": str(proposal_receipt),
        "selected_receipt_sha256": file_sha256(proposal_receipt),
        "post_ablation_selection_path": str(post_path),
        "post_ablation_selection_sha256": file_sha256(post_path),
        "winner_ablation_adjudication_path": str(ablation_path),
        "winner_ablation_adjudication_sha256": file_sha256(ablation_path),
        "winner_ablation_evidence": roles,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })

    pointer = materialize_frontier_final_delivery(tmp_path)
    assert pointer["selected_candidate_id"] == proposal_id
    candidate = json.loads((final / "CANDIDATE.json").read_text(encoding="utf-8"))
    assert candidate["candidate_id"] == proposal_id
    assert candidate["source_bound_terminal_receipt"]["sha256"] == file_sha256(
        proposal_receipt
    )
    assert candidate["ablation_evidence"]["experimental_results"][
        "projected_or_full"
    ]["candidate_id"] == full_id
    results = json.loads((final / "RESULTS.json").read_text(encoding="utf-8"))
    assert results["host_separated_complete_frontier"]["remote5090_frontier"][
        "selected_frontier_candidate_id"
    ] == full_id
