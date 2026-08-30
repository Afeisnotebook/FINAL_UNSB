from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from operations.local_route1_cross_version_adjudicate import SCHEMA as CROSS_SCHEMA
from operations.local_route1_winner_ablation_adjudicate import (
    SCHEMA as ABLATION_SCHEMA,
    SINGLE_SEED_CHALLENGE_STATUS,
)
from research.local_route1.cross_version_final_delivery import (
    SCHEMA,
    materialize_cross_version_final_delivery,
)
from research.local_route1.ablation_challenger_selection import (
    WORKSPACE_SCHEMA,
    adjudicate_ablation_challenger_selection,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.candidate_defect_audit import CROSS_VERSION_NEGATIVE_STATUS
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.seed_validation import MULTI_SEED_ADJUDICATION_SCHEMA
from research.local_route1.single_seed_development import (
    materialize_single_seed_development_freeze,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metric(psnr: float, *, probe_id: str = "plain", protocol: str = "crn") -> dict:
    domains = {
        f"d{index}": {"psnr": psnr + index / 100.0, "ssim": 0.7, "lpips": 0.2}
        for index in range(6)
    }
    return {
        "schema": "evaluation",
        "count_per_domain": 70,
        "protocol_fingerprint": protocol,
        "evaluation_input_sha256": "inputs",
        "macro_psnr": sum(row["psnr"] for row in domains.values()) / 6,
        "macro_ssim": 0.7,
        "macro_lpips": 0.2,
        "domains": domains,
        "probe_id": probe_id,
        "confirmation20_opened": False,
    }


def _trajectory(
    candidate_id: str, algorithm: str, delta: float, *, status: str = POSITIVE_STATUS,
) -> dict:
    return {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "status": status,
        "candidate_id": candidate_id,
        "algorithm_fingerprint": algorithm,
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "trajectory": [{"epoch": 200}],
        "late_three_mean_macro_psnr_delta": delta,
        "e200_macro_psnr_delta": delta,
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }


def _method(
    root: Path, candidate_id: str, delta: float, *, status: str = POSITIVE_STATUS,
) -> tuple[Path, dict]:
    algorithm = f"algorithm-{candidate_id}"
    card_path = root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = root / "derive" / "implementations" / f"{candidate_id}.json"
    _write(card_path, {
        "name": f"name-{candidate_id}",
        "formula": "operator formula",
        "unsb_object": "native object",
        "identity_or_unbiased_condition": "identity",
        "target_inaccessibility_proof": "unpaired only",
        "algorithm_hyperparameters": {"fixed": 1},
        "compute_cost": "fixed compute",
        "memory_cost": "fixed memory",
        "recovery_state_cost": "RNG",
    })
    _write(implementation_path, {
        "model": "test-model", "method": {"fixed": 1}, "source_files": [],
    })
    trajectory_path = root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    _write(trajectory_path, _trajectory(candidate_id, algorithm, delta, status=status))
    _write(root / "candidates" / candidate_id / "metrics" / "e200.json", _metric(
        20.0 + delta, probe_id=candidate_id,
    ))
    trace = root / "candidates" / candidate_id / "TRAIN_TRACE.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(json.dumps({"epoch_wall_seconds": 10.0}) + "\n", encoding="utf-8")
    receipt_path = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": status,
        "algorithm_fingerprint": algorithm,
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "training_git_commit": "a" * 40,
        "receipt_source_sha256": file_sha256(
            ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
        ),
        "derivation_card_sha256": file_sha256(card_path),
        "implementation_sha256": file_sha256(implementation_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": delta,
            "e200_macro_psnr_delta": delta,
            "late_points_with_four_of_six_positive_domains": 2,
            "late_average_worst_domain_delta": -0.2,
            "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
            "late_mean_macro_ssim_delta": 0.01,
            "late_mean_macro_lpips_delta": -0.01,
        },
        "median_epoch_wall_seconds": 10.0,
        "terminal_integrity": {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
        "plain_e200_verification_sha256": "plain",
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    _write(receipt_path, receipt)
    _write(Path(str(receipt_path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(receipt_path),
    })
    return receipt_path, receipt


def test_cross_version_final_delivery_requires_and_includes_long_ablations(tmp_path):
    _write(tmp_path / "anchors" / "plain" / "metrics" / "e200.json", _metric(20.0))
    full_id, runner_id = "G1-FULL", "G1-RUNNER"
    full_path, full = _method(tmp_path, full_id, 0.3)
    runner_path, runner = _method(tmp_path, runner_id, 0.1)
    proposal_path, proposal = _method(tmp_path, "ABL-PROPOSAL", 0.05)
    observable_path, observable = _method(tmp_path, "ABL-OBSERVE", 0.0)

    ranking = []
    for rank, receipt in enumerate((full, runner), start=1):
        ranking.append({
            "rank": rank,
            "candidate_id": receipt["candidate_id"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "candidate_training_core_fingerprint": receipt[
                "candidate_training_core_fingerprint"
            ],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "ranking_fields": receipt["ranking_fields"],
            "median_epoch_wall_seconds": receipt["median_epoch_wall_seconds"],
            "terminal_integrity": receipt["terminal_integrity"],
        })
    cross_path = tmp_path / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    _write(cross_path, {
        "schema": CROSS_SCHEMA,
        "status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE",
        "ranking": ranking,
            "selected_candidate_id": full_id,
            "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
            "selected_candidate_fingerprint": full["candidate_fingerprint"],
            "selected_training_git_commit": full["training_git_commit"],
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "candidates" / full_id / "MULTI_SEED_ADJUDICATION.json", {
        "schema": MULTI_SEED_ADJUDICATION_SCHEMA,
        "status": "ROUTE1_SUSTAINED_LOCAL",
        "candidate_id": full_id,
        "algorithm_fingerprint": full["algorithm_fingerprint"],
        "included_seeds": [2026, 2027],
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    })
    seed_root = tmp_path / "seed_validation" / "seed2027"
    _write(seed_root / "SEED_VALIDATION_SUMMARY.json", {
        "status": "COMPLETE", "trajectory": [{"epoch": 200}],
        "paired_metric_changed_algorithm": False, "confirmation20_opened": False,
    })
    _write(seed_root / "candidate" / "metrics" / "e200.json", _metric(
        20.2, probe_id=full_id, protocol="seed-crn",
    ))
    _write(seed_root / "plain" / "metrics" / "e200.json", _metric(
        20.0, protocol="seed-crn",
    ))

    with pytest.raises(RuntimeError, match="requires winner proposal"):
        materialize_cross_version_final_delivery(tmp_path)

    roles = {}
    for role, path, receipt in (
        ("proposal_only", proposal_path, proposal),
        ("observable_only", observable_path, observable),
        ("projected_or_full", full_path, full),
    ):
        roles[role] = {
            "candidate_id": receipt["candidate_id"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "trajectory_status": receipt["trajectory_status"],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "receipt_path": str(path.resolve()),
            "receipt_sha256": file_sha256(path),
            "ranking_fields": receipt["ranking_fields"],
        }
    _write(tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json", {
        "schema": ABLATION_SCHEMA,
        "status": "COMPLETE_NO_SELECTION_CHANGE",
        "selected_candidate_id": full_id,
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "source_cross_version_adjudication_sha256": file_sha256(cross_path),
        "roles": roles,
        "observable_only_identity": {
            "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
        },
        "proposal_only_out_ranks_full": False,
        "selection_change_blocked_pending_seed_validation": False,
        "selection_changed": False,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    result = materialize_cross_version_final_delivery(tmp_path)
    assert result["schema"] == SCHEMA
    assert result["candidate_id"] == full_id
    assert result["classification"] == "route1_sustained_local"
    assert (tmp_path / "final" / "CANDIDATE.json").is_file()
    results = json.loads((tmp_path / "final" / "RESULTS.json").read_text())
    assert set(results["winner_ablation_results"]) == {
        "proposal_only", "observable_only", "projected_or_full",
    }
    alternates = json.loads((tmp_path / "final" / "ALTERNATES.json").read_text())
    assert [row["candidate_id"] for row in alternates["alternates"]] == [
        runner_id, proposal["candidate_id"],
    ]


def _multi(candidate_id: str, algorithm: str, gain: float) -> dict:
    return {
        "schema": MULTI_SEED_ADJUDICATION_SCHEMA,
        "status": "ROUTE1_SUSTAINED_LOCAL",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": algorithm,
        "included_seeds": [2026, 2027],
        "combined_late_three_mean_macro_psnr_delta": gain,
        "combined_late_average_positive_domains": 4.5,
        "combined_late_average_worst_domain_delta": -0.2,
        "algorithm_changes_after_seed2026_freeze": False,
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }


@pytest.mark.parametrize(
    "full_gain,proposal_gain,expected_status,expected_candidate",
    [
        (0.2, 0.4, "CHALLENGER_SELECTED_AFTER_FROZEN_SEEDS", "ABL-PROPOSAL"),
        (0.5, 0.4, "FULL_WINNER_RETAINED_AFTER_CHALLENGER_SEEDS", "G1-FULL"),
    ],
)
def test_cross_version_final_delivery_resolves_frozen_seed_ablation_challenger(
    tmp_path, full_gain, proposal_gain, expected_status, expected_candidate,
):
    _write(tmp_path / "anchors" / "plain" / "metrics" / "e200.json", _metric(20.0))
    full_path, full = _method(tmp_path, "G1-FULL", 0.3)
    runner_path, runner = _method(tmp_path, "G1-RUNNER", 0.1)
    proposal_path, proposal = _method(tmp_path, "ABL-PROPOSAL", 0.4)
    observable_path, observable = _method(tmp_path, "ABL-OBSERVE", 0.0)
    ranking = []
    for rank, receipt in enumerate((full, runner), start=1):
        ranking.append({
            "rank": rank,
            "candidate_id": receipt["candidate_id"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "candidate_training_core_fingerprint": receipt[
                "candidate_training_core_fingerprint"
            ],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "ranking_fields": receipt["ranking_fields"],
            "median_epoch_wall_seconds": receipt["median_epoch_wall_seconds"],
            "terminal_integrity": receipt["terminal_integrity"],
        })
    cross_path = tmp_path / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    _write(cross_path, {
        "schema": CROSS_SCHEMA,
        "status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE",
        "ranking": ranking,
            "selected_candidate_id": full["candidate_id"],
            "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
            "selected_candidate_fingerprint": full["candidate_fingerprint"],
            "selected_training_git_commit": full["training_git_commit"],
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    full_multi_path = (
        tmp_path / "candidates" / full["candidate_id"]
        / "MULTI_SEED_ADJUDICATION.json"
    )
    _write(full_multi_path, _multi(
        full["candidate_id"], full["algorithm_fingerprint"], full_gain,
    ))
    full_seed_root = tmp_path / "seed_validation" / "seed2027"
    _write(full_seed_root / "SEED_VALIDATION_SUMMARY.json", {
        "status": "COMPLETE", "trajectory": [{"epoch": 200}],
        "paired_metric_changed_algorithm": False, "confirmation20_opened": False,
    })
    _write(full_seed_root / "candidate" / "metrics" / "e200.json", _metric(
        20.3, probe_id=full["candidate_id"], protocol="seed-crn",
    ))
    _write(full_seed_root / "plain" / "metrics" / "e200.json", _metric(
        20.0, protocol="seed-crn",
    ))
    roles = {}
    for role, path, receipt in (
        ("proposal_only", proposal_path, proposal),
        ("observable_only", observable_path, observable),
        ("projected_or_full", full_path, full),
    ):
        roles[role] = {
            "candidate_id": receipt["candidate_id"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "trajectory_status": receipt["trajectory_status"],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "receipt_path": str(path.resolve()),
            "receipt_sha256": file_sha256(path),
            "ranking_fields": receipt["ranking_fields"],
        }
    ablation_path = tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
    _write(ablation_path, {
        "schema": ABLATION_SCHEMA,
        "status": "ABLATION_CHALLENGER_REQUIRES_FROZEN_SEED_VALIDATION",
        "selected_candidate_id": full["candidate_id"],
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "source_cross_version_adjudication_sha256": file_sha256(cross_path),
        "roles": roles,
        "observable_only_identity": {
            "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
        },
        "proposal_only_out_ranks_full": True,
        "selection_change_blocked_pending_seed_validation": True,
        "selection_changed": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    with pytest.raises(RuntimeError, match="requires completed frozen-seed selection"):
        materialize_cross_version_final_delivery(tmp_path)

    workspace = (
        tmp_path / "ablation_challenger_seed_validation" / proposal["candidate_id"]
    )
    workspace_record = workspace / "CHALLENGER_SEED_WORKSPACE.json"
    _write(workspace_record, {
        "schema": WORKSPACE_SCHEMA,
        "source_root": str(tmp_path.resolve()),
        "workspace_root": str(workspace.resolve()),
        "candidate_id": proposal["candidate_id"],
        "full_winner_seed_namespace_reused": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    challenger_multi_path = (
        workspace / "candidates" / proposal["candidate_id"]
        / "MULTI_SEED_ADJUDICATION.json"
    )
    _write(challenger_multi_path, _multi(
        proposal["candidate_id"], proposal["algorithm_fingerprint"], proposal_gain,
    ))
    seed_root = workspace / "seed_validation" / "seed2027"
    _write(seed_root / "SEED_VALIDATION_SUMMARY.json", {
        "status": "COMPLETE", "trajectory": [{"epoch": 200}],
        "paired_metric_changed_algorithm": False, "confirmation20_opened": False,
    })
    _write(seed_root / "candidate" / "metrics" / "e200.json", _metric(
        20.4, probe_id=proposal["candidate_id"], protocol="seed-crn",
    ))
    _write(seed_root / "plain" / "metrics" / "e200.json", _metric(
        20.0, protocol="seed-crn",
    ))
    selection = adjudicate_ablation_challenger_selection(tmp_path, workspace)
    assert selection["status"] == expected_status

    result = materialize_cross_version_final_delivery(tmp_path)
    assert result["candidate_id"] == expected_candidate
    assert result["ablation_challenger_selection"]["status"] == expected_status
    alternates = json.loads((tmp_path / "final" / "ALTERNATES.json").read_text())
    expected_first_alternate = (
        full["candidate_id"]
        if expected_candidate == proposal["candidate_id"]
        else proposal["candidate_id"]
    )
    assert [row["candidate_id"] for row in alternates["alternates"]] == [
        expected_first_alternate, runner["candidate_id"],
    ]


def test_single_seed_emergency_policy_selects_complete_e200_ablation_challenger(
    tmp_path,
):
    _write(tmp_path / "anchors" / "plain" / "metrics" / "e200.json", _metric(20.0))
    full_path, full = _method(tmp_path, "G1-FULL", 0.3)
    _, runner = _method(tmp_path, "G1-RUNNER", 0.1)
    proposal_path, proposal = _method(tmp_path, "ABL-PROPOSAL", 0.4)
    observable_path, observable = _method(tmp_path, "ABL-OBSERVE", 0.0)
    ranking = []
    for rank, receipt in enumerate((full, runner), start=1):
        ranking.append({
            "rank": rank,
            "candidate_id": receipt["candidate_id"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "candidate_training_core_fingerprint": receipt[
                "candidate_training_core_fingerprint"
            ],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "ranking_fields": receipt["ranking_fields"],
            "median_epoch_wall_seconds": receipt["median_epoch_wall_seconds"],
            "terminal_integrity": receipt["terminal_integrity"],
        })
    cross_path = tmp_path / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    _write(cross_path, {
        "schema": CROSS_SCHEMA,
        "status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE",
        "ranking": ranking,
        "selected_candidate_id": full["candidate_id"],
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "selected_candidate_fingerprint": full["candidate_fingerprint"],
        "selected_training_git_commit": full["training_git_commit"],
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    freeze = materialize_single_seed_development_freeze(tmp_path)
    assert freeze["included_seeds"] == [2026]
    assert freeze["cross_seed_stability_claimed"] is False

    roles = {}
    for role, path, receipt in (
        ("proposal_only", proposal_path, proposal),
        ("observable_only", observable_path, observable),
        ("projected_or_full", full_path, full),
    ):
        roles[role] = {
            "candidate_id": receipt["candidate_id"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "trajectory_status": receipt["trajectory_status"],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "receipt_path": str(path.resolve()),
            "receipt_sha256": file_sha256(path),
            "ranking_fields": receipt["ranking_fields"],
        }
    _write(tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json", {
        "schema": ABLATION_SCHEMA,
        "status": SINGLE_SEED_CHALLENGE_STATUS,
        "selected_candidate_id": full["candidate_id"],
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "source_cross_version_adjudication_sha256": file_sha256(cross_path),
        "roles": roles,
        "observable_only_identity": {
            "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
        },
        "proposal_only_out_ranks_full": True,
        "selection_change_blocked_pending_seed_validation": False,
        "selection_ready_under_single_seed_development_policy": True,
        "selection_changed": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    result = materialize_cross_version_final_delivery(tmp_path)
    assert result["candidate_id"] == proposal["candidate_id"]
    assert result["classification"] == "single_seed_development_signal"
    assert result["single_seed_development_freeze"]["included_seeds"] == [2026]
    assert result["ablation_challenger_selection"]["status"] == (
        "SINGLE_SEED_ABLATION_CHALLENGER_SELECTED"
    )
    commands = result["reproduction_commands"]
    assert "seed_validation" not in commands
    assert commands["deferred_seed_validation"] == {
        "status": "DEFERRED_BY_SINGLE_SEED_EMERGENCY_POLICY",
        "requires_new_user_authorization": True,
        "included_in_current_execution": False,
        "deferred_seeds": [2027, 2028],
        "command_template": (
            "python operations/local_route1_seed_executor.py --contract "
            "<SELECTED_SEED_ROOT>/operations/SEED_EXECUTOR_CONTRACT_ABL-PROPOSAL_s<SEED>.json"
        ),
    }
    results = json.loads((tmp_path / "final" / "RESULTS.json").read_text())
    assert results["seed_results"] == {}
    assert results["multi_seed_adjudication"] is None
    assert results["cross_seed_stability_claimed"] is False


def test_single_seed_terminal_negative_selection_delivers_honest_fallback(tmp_path):
    _write(tmp_path / "anchors" / "plain" / "metrics" / "e200.json", _metric(20.0))
    full_path, full = _method(
        tmp_path, "G2-FALLBACK", -0.1, status=NEGATIVE_STATUS,
    )
    _, runner = _method(
        tmp_path, "G1-RUNNER", -0.3, status=NEGATIVE_STATUS,
    )
    proposal_path, proposal = _method(
        tmp_path, "ABL-PROPOSAL", -0.2, status=NEGATIVE_STATUS,
    )
    observable_path, observable = _method(
        tmp_path, "ABL-OBSERVE", 0.0, status=NEGATIVE_STATUS,
    )
    ranking = []
    for rank, receipt in enumerate((full, runner), start=1):
        ranking.append({
            "rank": rank,
            "candidate_id": receipt["candidate_id"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "candidate_training_core_fingerprint": receipt[
                "candidate_training_core_fingerprint"
            ],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "ranking_fields": receipt["ranking_fields"],
            "median_epoch_wall_seconds": receipt["median_epoch_wall_seconds"],
            "terminal_integrity": receipt["terminal_integrity"],
        })
    selection_path = tmp_path / "operations" / "ROUTE1_FINAL_E200_SELECTION.json"
    _write(selection_path, {
        "schema": CROSS_SCHEMA,
        "status": CROSS_VERSION_NEGATIVE_STATUS,
        "ranking": ranking,
        "selected_candidate_id": full["candidate_id"],
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "selected_candidate_fingerprint": full["candidate_fingerprint"],
        "selected_training_git_commit": full["training_git_commit"],
        "selection_role": "current_best_fallback",
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    freeze = materialize_single_seed_development_freeze(tmp_path)
    assert freeze["development_signal_classification"] == (
        "current_best_seed2026_e200_fallback"
    )
    roles = {}
    for role, path, receipt in (
        ("proposal_only", proposal_path, proposal),
        ("observable_only", observable_path, observable),
        ("projected_or_full", full_path, full),
    ):
        roles[role] = {
            "candidate_id": receipt["candidate_id"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "trajectory_status": receipt["trajectory_status"],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "receipt_path": str(path.resolve()),
            "receipt_sha256": file_sha256(path),
            "ranking_fields": receipt["ranking_fields"],
        }
    _write(tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json", {
        "schema": ABLATION_SCHEMA,
        "status": "COMPLETE_NO_SELECTION_CHANGE",
        "selected_candidate_id": full["candidate_id"],
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "source_cross_version_adjudication_sha256": file_sha256(selection_path),
        "roles": roles,
        "observable_only_identity": {
            "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
        },
        "proposal_only_out_ranks_full": False,
        "selection_change_blocked_pending_seed_validation": False,
        "selection_changed": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    result = materialize_cross_version_final_delivery(tmp_path)
    assert result["candidate_id"] == full["candidate_id"]
    assert result["classification"] == "weak_fallback_single_seed_development"
    assert result["source_e200_selection"]["status"] == CROSS_VERSION_NEGATIVE_STATUS
