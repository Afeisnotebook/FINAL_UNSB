from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.local_route1 import related_goal_completion_audit as audit
from research.local_route1.related_multi_algorithm_final_delivery import (
    POINTER,
    PUBLISHED_FILES,
)


def _write(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload) if isinstance(payload, dict) else payload,
        encoding="utf-8",
    )


def test_related_goal_audit_requires_multi_algorithm_semantics(monkeypatch, tmp_path: Path):
    related = tmp_path / "related"
    family_ids = (audit.PROPOSAL, audit.HPCGR, audit.HJCGR)
    member_ids = (*family_ids, audit.AMTNC)
    gain_source = {
        "schema": "final-unsb-route1-related-gain-source-decomposition-v1",
        "status": "SHARED_ESTIMATOR_IMPROVES_MULTIPLE_PARENT_FIELDS",
        "members": [
            {
                "candidate_id": candidate_id,
                "parent": {
                    "parent_id": parent_id,
                    "late_three_mean_macro_psnr_delta": parent_late,
                    "e200_macro_psnr_delta": parent_e200,
                },
                "composed": {
                    "late_three_mean_macro_psnr_delta": child_late,
                    "e200_macro_psnr_delta": child_e200,
                },
                "matched_compositional_increment_over_parent": {
                    "late_three_macro_psnr_delta": child_late - parent_late,
                    "e200_macro_psnr_delta": child_e200 - parent_e200,
                },
            }
            for candidate_id, parent_id, parent_late, parent_e200,
            child_late, child_e200 in (
                (audit.PROPOSAL, "plain", 0.0, 0.0, 0.4, 0.3),
                (audit.HPCGR, "hnek", 0.2, 0.1, 0.5, 0.3),
                (audit.HJCGR, "hj", 0.1, 0.05, 0.3, 0.2),
            )
        ],
        "shared_estimator_positive_increment_candidate_ids": list(family_ids),
        "shared_estimator_positive_increment_count": 3,
        "matched_increment_is_not_additive_causal_attribution": True,
        "compute_only_control": {
            "schema": "final-unsb-route1-related-compute-only-control-v1",
            "status": "EXTRA_VIEW_OBSERVATION_IS_EXACT_PLAIN_E200_DYNAMICS",
            "source_path": "operations/WINNER_ABLATION_ADJUDICATION.json",
            "source_sha256": "a" * 64,
            "candidate_dynamics_state_sha256": "same-dynamics",
            "plain_dynamics_state_sha256": "same-dynamics",
            "dynamics_state_exact_plain": True,
            "late_three_mean_macro_psnr_delta": 0.0,
            "e200_macro_psnr_delta": 0.0,
            "rules_out_wall_clock_or_observer_side_effect_only": True,
            "does_not_claim_native_compute_budget_equivalence": True,
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
        "player_scope_control": {
            "schema": "final-unsb-route1-related-player-scope-control-v1",
            "status": "GF_ONLY_REPLICATION_SUSTAINS_E200_WHILE_FULL_PLAYER_REPLICATION_DOES_NOT",
            "gf_only": {
                "candidate_id": audit.PROPOSAL,
                "late_three_mean_macro_psnr_delta": 0.54,
                "e200_macro_psnr_delta": 0.45,
                "strict_gate_pass": True,
            },
            "all_players": {
                "candidate_id": "G1-02B-PLAYER-CONDITIONAL-RSMG",
                "late_three_mean_macro_psnr_delta": 0.62,
                "e200_macro_psnr_delta": -0.001,
                "strict_gate_pass": False,
            },
            "gf_only_minus_all_players": {
                "late_three_macro_psnr_delta": -0.08,
                "e200_macro_psnr_delta": 0.451,
            },
            "does_not_claim_additive_single_path_causality": True,
            "paired_metrics_used_only_after_complete_e200_trajectories": True,
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
        "conditional_resampling_control": {
            "schema": "final-unsb-route1-related-conditional-resampling-control-v1",
            "status": "FRESH_POST_DE_RESAMPLING_ALONE_FAILS_WHILE_TWO_VIEW_GF_MEAN_PASSES",
            "source_path": "operations/COMPLETE_FRONTIER_4090_ADJUDICATION.json",
            "source_sha256": "b" * 64,
            "resampling_only": {
                "candidate_id": audit.PCNR,
                "late_three_mean_macro_psnr_delta": -0.53,
                "e200_macro_psnr_delta": -0.03,
            },
            "resampling_plus_two_view_mean": {
                "candidate_id": audit.PROPOSAL,
                "late_three_mean_macro_psnr_delta": 0.54,
                "e200_macro_psnr_delta": 0.45,
            },
            "two_view_mean_increment_over_resampling_only": {
                "late_three_macro_psnr_delta": 1.07,
                "e200_macro_psnr_delta": 0.48,
            },
            "only_tested_operator_scope": True,
            "does_not_claim_global_necessity": True,
            "does_not_claim_additive_single_path_causality": True,
            "paired_metrics_used_only_after_complete_e200_trajectories": True,
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
        "stochastic_variance_scope": {
            "conditioning_includes_official_unpaired_batch": True,
            "reduced_components": "within-batch native G/F view randomness",
            "not_reduced_components": "across-batch data sampling",
            "iid_requirement": "two conditionally iid views before one commit",
        },
        "variance_axis_alignment": {
            "schema": "final-unsb-route1-related-variance-axis-alignment-v1",
            "operator_axis": "within_batch_latent_time_bridge_and_feature_sampling",
            "causal_matrix_path": "audit/LONG_CAUSAL_MATRIX.json",
            "causal_matrix_sha256": "c" * 64,
            "members": {
                audit.PROPOSAL: {
                    "alignment": "empirically_aligned_by_completed_factorial_controls",
                },
                audit.HJCGR: {
                    "alignment": "directly_aligned_with_parent_audited_variance_axis",
                    "latent_time_bridge_rng": {
                        "rows": 22,
                        "variance_dominated_rows": 22,
                        "mean_variance_fraction": 0.87,
                    },
                },
                audit.HPCGR: {
                    "alignment": "compositional_transfer_hypothesis_not_direct_axis_repair",
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
                    "unaddressed_parent_axis": "independent_unpaired_batch",
                },
            },
            "shared_theorem_does_not_imply_shared_failure_mode": True,
            "hpcgr_viability_must_be_decided_by_complete_e200_trajectory": True,
            "paired_metrics_used_for_formula_or_training_control": False,
            "confirmation20_opened": False,
        },
        "optimizer_nonlinearity_boundary": "pre-Adam only",
        "cross_host_metrics_merged": False,
    }
    algorithm_set = {
        "schema": audit.ALGORITHM_SET_SCHEMA,
        "status": "MULTIPLE_VIABLE_ALGORITHMS",
        "action_priority_candidate_id": audit.HPCGR,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "strict_viable_candidate_ids": [audit.HPCGR, audit.HJCGR],
        "positive_but_fragile_candidate_ids": [audit.PROPOSAL, audit.AMTNC],
        "members": [
            {
                "candidate_id": candidate_id,
                "disposition": (
                    "strict_viable_algorithm"
                    if candidate_id in (audit.HPCGR, audit.HJCGR)
                    else "positive_but_fragile_algorithm"
                ),
                "mathematics": {"formula": f"formula-{candidate_id}"},
                "risk": {"single_seed_only": True},
                "reproduction": {
                    "seed2026_e200": f"run-{candidate_id}",
                    "deferred_seed_validation": [2027, 2028],
                },
                "source_bound": {"receipt": candidate_id},
                "absolute_relative_domain_trajectory": [candidate_id],
            }
            for candidate_id in member_ids
        ],
        "related_conditional_estimator_family": {
            "members": [
                {"candidate_id": candidate_id} for candidate_id in family_ids
            ],
            "conditional_expectation_property": "expectation preserved",
            "conditional_covariance_property": "covariance halved",
            "unbiased_mathematical_object": "pre-Adam joint G/F stochastic gradient estimator",
            "conditioning_scope": "fixed post-D/E parent state",
            "stochastic_variance_scope": {
                "conditioning_includes_official_unpaired_batch": True,
                "reduced_components": "within-batch native G/F view randomness",
                "not_reduced_components": "across-batch data sampling",
            },
            "adam_boundary": "no expected displacement equality claim",
            "finite_step_coupling_change_is_intended": True,
            "native_de_stochasticity_retained": True,
            "membership_is_not_assumed_viability": True,
        },
        "independent_mechanism_members": [
            {"candidate_id": audit.AMTNC, "mechanism": "independent"},
        ],
        "mechanism_gain_source_decomposition": gain_source,
        "cross_runtime_related_evidence": {
            "algorithms": [{
                "host_results": [{
                    "trajectory_snapshot": {
                        "candidate_id": audit.HPCGR,
                        "confirmation20_opened": False,
                    },
                }],
            }],
            "cross_runtime_is_not_cross_seed": True,
            "cross_host_deltas_merged": False,
            "cross_seed_stability_claimed": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
        "cross_host_deltas_merged": False,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(related / "ALGORITHM_SET.json", algorithm_set)
    _write(related / "ACTION_PRIORITY.json", {
        "schema": audit.ACTION_SCHEMA,
        "status": "CURRENT_NEXT_ACTION_PRIORITY",
        "candidate_id": audit.HPCGR,
        "action_priority_is_not_scientific_exclusivity": True,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    selected_member = next(
        row for row in algorithm_set["members"]
        if row["candidate_id"] == audit.HPCGR
    )
    _write(related / "CANDIDATE.json", {
        "schema": audit.CANDIDATE_SCHEMA,
        "status": "CURRENT_ACTION_PRIORITY_FROM_MULTI_ALGORITHM_SET",
        "candidate_id": audit.HPCGR,
        "canonical_candidate_is_action_priority_only": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "algorithm": selected_member,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(related / "ALTERNATES.json", {
        "schema": audit.ALTERNATES_SCHEMA,
        "status": "TWO_EVIDENCE_RANKED_ALTERNATES",
        "action_priority_candidate_id": audit.HPCGR,
        "alternates": [
            {
                "candidate_id": audit.HJCGR,
                "algorithm": next(
                    row for row in algorithm_set["members"]
                    if row["candidate_id"] == audit.HJCGR
                ),
            },
            {
                "candidate_id": audit.PROPOSAL,
                "algorithm": next(
                    row for row in algorithm_set["members"]
                    if row["candidate_id"] == audit.PROPOSAL
                ),
            },
        ],
        "action_priority_is_not_scientific_exclusivity": True,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(related / "RELATED_RESULTS.json", {
        "schema": audit.RESULTS_SCHEMA,
        "status": "RELATED_MULTI_ALGORITHM_E200_COMPLETE",
        "mechanism_gain_source_decomposition": gain_source,
        "cross_host_deltas_merged": False,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(related / "RELATED_FINAL_REPORT.md", "report")
    _write(related / POINTER, {"pointer": True})

    monkeypatch.setattr(
        audit, "audit_complete_delivery",
        lambda _path: {"status": "compatibility-proven"},
    )
    monkeypatch.setattr(
        audit, "validate_related_delivery",
        lambda _path: {
            "algorithm_set_status": "MULTIPLE_VIABLE_ALGORITHMS",
            "action_priority_candidate_id": audit.HPCGR,
            "strict_viable_candidate_count": 2,
        },
    )
    monkeypatch.setattr(
        audit, "_domain_trajectory",
        lambda rows, label: {"label": label, "rows": rows},
    )

    result = audit.audit_related_goal_completion(tmp_path / "compatibility", related)
    assert result["action_priority_candidate_id"] == audit.HPCGR
    assert result["strict_viable_candidate_ids"] == [audit.HPCGR, audit.HJCGR]
    assert result["action_priority_is_not_scientific_exclusivity"] is True
    assert result["alternate_candidate_ids"] == [audit.HJCGR, audit.PROPOSAL]
    assert result["gain_source_proof"][
        "positive_increment_candidate_ids"
    ] == list(family_ids)
    assert result["gain_source_proof"]["additive_causality_not_claimed"] is True
    assert result["gain_source_proof"]["compute_only_control_proven"] is True
    assert result["gain_source_proof"][
        "native_compute_budget_equivalence_not_claimed"
    ] is True
    assert result["gain_source_proof"]["player_scope_control_proven"] is True
    assert result["gain_source_proof"][
        "conditional_resampling_control_proven"
    ] is True
    assert result["gain_source_proof"]["within_batch_variance_scope_proven"] is True
    assert result["gain_source_proof"][
        "parent_variance_axis_alignment_proven"
    ] is True
    assert result["gain_source_proof"][
        "pre_adam_unbiasedness_boundary_proven"
    ] is True
    assert result["completion_claim_allowed"] is False
    assert set(result["source_delivery_sha256"]) == {POINTER, *PUBLISHED_FILES}

    algorithm_set["mechanism_gain_source_decomposition"]["members"][0][
        "matched_compositional_increment_over_parent"
    ]["e200_macro_psnr_delta"] += 1.0
    _write(related / "ALGORITHM_SET.json", algorithm_set)
    with pytest.raises(RuntimeError, match="gain-source arithmetic changed"):
        audit.audit_related_goal_completion(tmp_path / "compatibility", related)
