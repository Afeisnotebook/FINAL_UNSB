from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.local_route1 import related_multi_algorithm_final_delivery as delivery
from research.local_route1.protocol import file_sha256


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(candidate_id: str, late: float, e200: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_role": "test",
        "classification": "strict_sustained",
        "trajectory_status": "LONG_HORIZON_POSITIVE_CANDIDATE",
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "training_git_commit": "a" * 40,
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": late,
            "e200_macro_psnr_delta": e200,
            "late_points_with_four_of_six_positive_domains": 3,
            "late_average_worst_domain_delta": -0.2,
            "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
            "late_mean_macro_ssim_delta": 0.01,
            "late_mean_macro_lpips_delta": -0.01,
        },
        "receipt_path": "unused",
        "receipt_sha256": "unused",
        "trajectory_path": "unused",
        "trajectory_sha256": "unused",
        "median_epoch_wall_seconds": 10.0,
    }


def _related_row(candidate_id: str, late: float, e200: float) -> dict:
    trajectory = {
        "candidate_id": candidate_id,
        "status": "LONG_HORIZON_POSITIVE_CANDIDATE",
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }
    return {
        "candidate_id": candidate_id,
        "classification": "strict_sustained_local_signal",
        "strict_checks": {"late_three_positive": True},
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "training_git_commit": "b" * 40,
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "late_three_mean_macro_psnr_delta": late,
        "e200_macro_psnr_delta": e200,
        "late_points_with_four_of_six_positive_domains": 3,
        "late_average_worst_domain_delta": -0.2,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
        "rolling_drawdown_db": 0.1,
        "median_epoch_wall_seconds": 10.0,
        "terminal_receipt_path": "unused",
        "terminal_receipt_sha256": "unused",
        "trajectory_path": "unused",
        "trajectory_sha256": "unused",
        "trajectory_snapshot": trajectory,
    }


def test_final_delivery_keeps_multiple_viable_algorithms(monkeypatch, tmp_path: Path):
    complete_pointer_path = _write(tmp_path / "complete-pointer.json", {"ok": True})
    base_path = _write(tmp_path / "base.json", {"ok": True})
    related_paths = {
        key: _write(tmp_path / f"{key}.json", {"ok": True})
        for key in ("remote4090", "remote5090", "combined")
    }
    base = {
        "same_host_authority": {
            "base_e0_scientific_state_sha256": "e0",
            "base_protocol_fingerprint": "protocol",
            "manifest_sha256": "manifest",
        },
        "ranking": [
            _row("BASE", 0.4, 0.3),
            _row(delivery.PROPOSAL, 0.45, 0.35),
            {
                **_row(delivery.PCNR, -0.5, -0.03),
                "classification": "closed_current_operator",
            },
        ],
    }
    _write(
        tmp_path / "operations" / "COMPLETE_FRONTIER_4090_ADJUDICATION.json",
        base,
    )
    _write(tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json", {
        "status": "COMPLETE_CAUSAL_AUDIT",
        "sampling_variance_summaries": [
            {
                "probe": "hj",
                "axes": {
                    "latent_time_bridge_rng": {
                        "rows": 22,
                        "variance_dominated_rows": 22,
                        "mean_variance_fraction": 0.87,
                    },
                },
            },
            {
                "probe": "hnek",
                "axes": {
                    "latent_time_bridge_rng": {
                        "rows": 18,
                        "variance_dominated_rows": 0,
                        "mean_variance_fraction": 0.52,
                    },
                    "independent_unpaired_batch": {
                        "rows": 18,
                        "variance_dominated_rows": 9,
                        "mean_variance_fraction": 0.70,
                    },
                },
            },
        ],
    })
    host4090 = {"ranking": [
        _related_row(delivery.HPCGR, 0.8, 0.5),
        _related_row(delivery.HJCGR, 0.6, 0.4),
    ]}
    host5090 = {"ranking": []}
    combined = {
        "status": "MULTIPLE_VIABLE_ALGORITHMS",
        "algorithms": [],
        "cross_host_deltas_merged": False,
        "cross_runtime_is_not_cross_seed": True,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }

    monkeypatch.setattr(
        delivery, "_complete_delivery",
        lambda _root: ({"status": "complete"}, complete_pointer_path),
    )
    monkeypatch.setattr(delivery, "_base_frontier", lambda _root: (base, base_path))
    monkeypatch.setattr(
        delivery, "_related_inputs",
        lambda _root: (host4090, host5090, combined, related_paths),
    )
    monkeypatch.setattr(
        delivery, "_candidate_domain_trajectory",
        lambda _root, candidate_id: {"candidate_id": candidate_id},
    )
    monkeypatch.setattr(
        delivery, "_terminal_row",
        lambda _root, candidate_id, host_label: {
            **_related_row(candidate_id, 0.25, 0.15),
            "host_label": host_label,
        },
    )

    sources = {}
    executors = {}
    _write(tmp_path / "evidence" / "ANCHOR_TRAJECTORIES.json", {
        "schema": "local-route1-anchor-summary-v1",
        "summaries": [
            {
                "probe_id": "hnek",
                "complete_e200": True,
                "late_three_mean_macro_psnr_delta": 0.5,
                "trajectory": [{"epoch": 200, "macro_psnr_delta": 0.25}],
            },
            {
                "probe_id": "hj",
                "complete_e200": True,
                "late_three_mean_macro_psnr_delta": 0.2,
                "trajectory": [{"epoch": 200, "macro_psnr_delta": 0.1}],
            },
        ],
    })
    _write(tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json", {
        "schema": "final-unsb-route1-winner-ablation-adjudication-v1",
        "status": "ABLATION_CHALLENGER_READY_FOR_SINGLE_SEED_DEVELOPMENT_SELECTION",
        "roles": {
            "proposal_only": {
                "candidate_id": delivery.PROPOSAL,
                "ranking_fields": {
                    "late_three_mean_macro_psnr_delta": 0.54,
                    "e200_macro_psnr_delta": 0.45,
                },
            },
            "observable_only": {
                "candidate_id": "OBSERVABLE",
                "ranking_fields": {
                    "late_three_mean_macro_psnr_delta": 0.0,
                    "e200_macro_psnr_delta": 0.0,
                },
            },
            "projected_or_full": {
                "candidate_id": delivery.PCRSMG_FULL,
                "ranking_fields": {
                    "late_three_mean_macro_psnr_delta": 0.62,
                    "e200_macro_psnr_delta": -0.001,
                },
            },
        },
        "observable_only_identity": {
            "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
            "candidate_dynamics_state_sha256": "same-dynamics",
            "plain_dynamics_state_sha256": "same-dynamics",
            "matched_plain_delta_exact_zero": True,
        },
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        "proposal_only_strict_gate_pass": True,
        "projected_or_full_strict_gate_pass": False,
        "proposal_only_out_ranks_full": True,
    })
    hjpcnr_trajectory_path = _write(
        tmp_path / "candidates" / delivery.HJPCNR / "CANDIDATE_TRAJECTORY.json",
        {
            "candidate_id": delivery.HJPCNR,
            "paired_metrics_used_for_training_or_gate": False,
            "confirmation20_opened": False,
        },
    )
    _write(tmp_path / "operations" / delivery.HJPCNR_RECEIPT, {
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": delivery.HJPCNR,
        "training_git_commit": "d" * 40,
        "verification_git_commit": "d" * 40,
        "trajectory_sha256": file_sha256(hjpcnr_trajectory_path),
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": 0.25,
            "e200_macro_psnr_delta": 0.15,
        },
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })

    for candidate_id in (
        "BASE", delivery.PROPOSAL, delivery.PCNR, delivery.HPCGR, delivery.HJCGR,
        delivery.HJPCNR,
    ):
        receipt_path = _write(
            tmp_path / "operations" / "terminal_receipts" / f"{candidate_id}.json",
            {"candidate_id": candidate_id},
        )
        trajectory_path = _write(
            tmp_path / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json",
            {
                "candidate_id": candidate_id,
                **(
                    {
                        "paired_metrics_used_for_training_or_gate": False,
                        "confirmation20_opened": False,
                    }
                    if candidate_id == delivery.HJPCNR else {}
                ),
            },
        )
        card_path = _write(
            tmp_path / "derive" / "cards" / f"{candidate_id}.json",
            {
                "candidate_id": candidate_id,
                "name": candidate_id,
                "formula": f"formula-{candidate_id}",
            },
        )
        implementation_path = _write(
            tmp_path / "derive" / "implementations" / f"{candidate_id}.json",
            {"candidate_id": candidate_id, "model": candidate_id, "method": {}},
        )
        sources[candidate_id] = (
            {
                "candidate_id": candidate_id,
                "algorithm_fingerprint": f"algorithm-{candidate_id}",
                "candidate_fingerprint": f"candidate-{candidate_id}",
                "training_git_commit": "c" * 40,
            },
            receipt_path,
            {"candidate_id": candidate_id},
            trajectory_path,
            {"candidate_id": candidate_id, "name": candidate_id},
            card_path,
            {"candidate_id": candidate_id, "model": candidate_id, "method": {}},
        )
        executor_path = _write(
            tmp_path / "operations" / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json",
            {"candidate_id": candidate_id},
        )
        executors[candidate_id] = (
            executor_path,
            {
                "candidate_git_commit": "c" * 40,
                "algorithm_fingerprint": f"algorithm-{candidate_id}",
                "candidate_fingerprint": f"candidate-{candidate_id}",
            },
        )
        assert file_sha256(implementation_path)

    monkeypatch.setattr(
        delivery, "_selected_source",
        lambda _root, row: sources[row["candidate_id"]],
    )
    monkeypatch.setattr(
        delivery, "_executor_contract",
        lambda _root, receipt: executors[receipt["candidate_id"]],
    )

    pointer = delivery.materialize_related_multi_algorithm_final_delivery(tmp_path)
    assert pointer["status"] == "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_COMPLETE"
    assert pointer["action_priority_candidate_id"] == delivery.HPCGR
    assert pointer["strict_viable_candidate_count"] == 5
    algorithm_set = json.loads(
        (tmp_path / delivery.FINAL_SUBDIR / "ALGORITHM_SET.json").read_text()
    )
    assert algorithm_set["status"] == "MULTIPLE_VIABLE_ALGORITHMS"
    assert algorithm_set["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert set(algorithm_set["strict_viable_candidate_ids"]) == {
        "BASE", delivery.PROPOSAL, delivery.HPCGR, delivery.HJCGR,
        delivery.HJPCNR,
    }
    assert algorithm_set["action_priority_is_not_scientific_exclusivity"] is True
    hjpcnr_member = next(
        row for row in algorithm_set["members"]
        if row["candidate_id"] == delivery.HJPCNR
    )
    assert hjpcnr_member["risk"]["posthoc_gain_source_development_control"] is True
    assert algorithm_set["related_conditional_estimator_family"][
        "gain_source_controls"
    ][0]["candidate_id"] == delivery.HJPCNR
    decomposition = algorithm_set["mechanism_gain_source_decomposition"]
    by_id = {row["candidate_id"]: row for row in decomposition["members"]}
    assert by_id[delivery.PROPOSAL][
        "matched_compositional_increment_over_parent"
    ]["late_three_macro_psnr_delta"] == 0.45
    assert by_id[delivery.HPCGR][
        "matched_compositional_increment_over_parent"
    ]["late_three_macro_psnr_delta"] == pytest.approx(0.3)
    assert by_id[delivery.HJCGR][
        "matched_compositional_increment_over_parent"
    ]["e200_macro_psnr_delta"] == pytest.approx(0.3)
    assert decomposition["shared_estimator_positive_increment_count"] == 3
    assert decomposition[
        "matched_increment_is_not_additive_causal_attribution"
    ] is True
    assert decomposition["compute_only_control"]["dynamics_state_exact_plain"] is True
    assert decomposition["compute_only_control"][
        "does_not_claim_native_compute_budget_equivalence"
    ] is True
    assert decomposition["player_scope_control"]["gf_only"][
        "e200_macro_psnr_delta"
    ] == 0.45
    assert decomposition["player_scope_control"]["all_players"][
        "e200_macro_psnr_delta"
    ] == -0.001
    resampling = decomposition["conditional_resampling_control"]
    assert resampling["resampling_only"]["candidate_id"] == delivery.PCNR
    assert resampling["resampling_plus_two_view_mean"][
        "candidate_id"
    ] == delivery.PROPOSAL
    assert resampling["two_view_mean_increment_over_resampling_only"][
        "e200_macro_psnr_delta"
    ] == pytest.approx(0.48)
    hj_factorial = decomposition["hj_specific_factorial_control"]
    assert hj_factorial["one_fresh_view"]["candidate_id"] == delivery.HJPCNR
    assert hj_factorial["two_fresh_view_mean"]["candidate_id"] == delivery.HJCGR
    assert hj_factorial["one_view_increment_over_hj"][
        "e200_macro_psnr_delta"
    ] == pytest.approx(0.05)
    assert hj_factorial["two_view_mean_increment_over_one_view"][
        "e200_macro_psnr_delta"
    ] == pytest.approx(0.25)
    assert decomposition["stochastic_variance_scope"][
        "conditioning_includes_official_unpaired_batch"
    ] is True
    assert algorithm_set["related_conditional_estimator_family"][
        "stochastic_variance_scope"
    ]["not_reduced_components"]
    alignment = decomposition["variance_axis_alignment"]
    assert alignment["members"][delivery.HJCGR]["latent_time_bridge_rng"][
        "variance_dominated_rows"
    ] == 22
    assert alignment["members"][delivery.HPCGR]["latent_time_bridge_rng"][
        "variance_dominated_rows"
    ] == 0
    assert alignment["members"][delivery.HPCGR][
        "unaddressed_parent_axis"
    ] == "independent_unpaired_batch"
    assert decomposition["optimizer_nonlinearity_boundary"]
    candidate = json.loads(
        (tmp_path / delivery.FINAL_SUBDIR / "CANDIDATE.json").read_text()
    )
    alternates = json.loads(
        (tmp_path / delivery.FINAL_SUBDIR / "ALTERNATES.json").read_text()
    )
    assert candidate["candidate_id"] == delivery.HPCGR
    assert candidate["algorithm"]["reproduction"]["seed2026_e200"]
    assert candidate["action_priority_is_not_scientific_exclusivity"] is True
    assert len(alternates["alternates"]) == 2
    assert delivery.HPCGR not in {
        row["candidate_id"] for row in alternates["alternates"]
    }
    assert delivery.materialize_related_multi_algorithm_final_delivery(tmp_path) == pointer
