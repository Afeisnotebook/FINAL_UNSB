"""Final Goal audit that requires both compatibility and algorithm-set deliveries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from operations.local_route1_related_multi_algorithm_final_relay import (
    validate_local_delivery as validate_related_delivery,
)
from research.local_route1.goal_completion_audit import (
    _domain_trajectory,
    _posthoc_boundary,
    _read_json,
    _require,
    audit_complete_delivery,
)
from research.local_route1.related_multi_algorithm_final_delivery import (
    ACTION_SCHEMA,
    ALGORITHM_SET_SCHEMA,
    ALTERNATES_SCHEMA,
    AMTNC,
    CANDIDATE_SCHEMA,
    HJCGR,
    HJPCNR,
    HPCGR,
    PCNR,
    POINTER,
    PROPOSAL,
    PUBLISHED_FILES,
    RESULTS_SCHEMA,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-related-goal-completion-audit-v1"


def _audit_gain_source(
    algorithm_set: dict[str, Any], results: dict[str, Any], member_ids: set[str],
) -> dict[str, Any]:
    family = algorithm_set.get("related_conditional_estimator_family")
    _require(isinstance(family, dict), "related estimator family is absent")
    family_ids = {
        str(row.get("candidate_id", ""))
        for row in family.get("members", []) if isinstance(row, dict)
    }
    expected_family = {PROPOSAL, HPCGR, HJCGR}
    _require(
        family_ids == expected_family,
        "related conditional-estimator family identity changed",
    )
    _require(
        family.get("membership_is_not_assumed_viability") is True,
        "related family membership was treated as assumed viability",
    )
    gain_controls = family.get("gain_source_controls")
    _require(
        isinstance(gain_controls, list)
        and len(gain_controls) == 1
        and gain_controls[0].get("candidate_id") == HJPCNR
        and gain_controls[0].get("ranked_if_same_e200_guardrails_pass") is True
        and gain_controls[0].get("posthoc_development_control") is True
        and gain_controls[0].get("confirmation_result") is False,
        "HJ-PCNR gain-source control was omitted or overclaimed",
    )
    _require(
        bool(family.get("conditional_expectation_property"))
        and bool(family.get("conditional_covariance_property")),
        "related family lost its conditional theorem",
    )
    _require(
        family.get("unbiased_mathematical_object")
        == "pre-Adam joint G/F stochastic gradient estimator"
        and bool(family.get("conditioning_scope"))
        and family.get("stochastic_variance_scope", {}).get(
            "conditioning_includes_official_unpaired_batch"
        ) is True
        and bool(family.get("stochastic_variance_scope", {}).get("reduced_components"))
        and bool(family.get("stochastic_variance_scope", {}).get("not_reduced_components"))
        and bool(family.get("adam_boundary"))
        and family.get("finite_step_coupling_change_is_intended") is True
        and family.get("native_de_stochasticity_retained") is True,
        "related family overclaimed optimizer/path equivalence or lost player scope",
    )
    independent = algorithm_set.get("independent_mechanism_members")
    independent_ids = {
        str(row.get("candidate_id", ""))
        for row in independent or [] if isinstance(row, dict)
    }
    _require(
        AMTNC in independent_ids,
        "AM-TNC independent mechanism identity is absent",
    )
    _require(
        expected_family.union({AMTNC, HJPCNR}).issubset(member_ids),
        "terminal algorithm set omitted a required related/independent member",
    )
    hjpcnr_member = next(
        (
            row for row in algorithm_set.get("members", [])
            if isinstance(row, dict) and row.get("candidate_id") == HJPCNR
        ),
        None,
    )
    _require(
        isinstance(hjpcnr_member, dict)
        and hjpcnr_member.get("risk", {}).get(
            "posthoc_gain_source_development_control"
        ) is True,
        "HJ-PCNR member lost its posthoc development label",
    )

    decomposition = algorithm_set.get("mechanism_gain_source_decomposition")
    _require(isinstance(decomposition, dict), "gain-source decomposition is absent")
    _require(
        decomposition.get("schema")
        == "final-unsb-route1-related-gain-source-decomposition-v1",
        "gain-source decomposition schema changed",
    )
    _require(
        decomposition.get("matched_increment_is_not_additive_causal_attribution")
        is True,
        "gain-source decomposition overclaims additive causality",
    )
    _require(
        decomposition.get("cross_host_metrics_merged") is False,
        "gain-source decomposition merged cross-host deltas",
    )
    compute_control = decomposition.get("compute_only_control")
    _require(
        isinstance(compute_control, dict)
        and compute_control.get("schema")
        == "final-unsb-route1-related-compute-only-control-v1"
        and compute_control.get("status")
        == "EXTRA_VIEW_OBSERVATION_IS_EXACT_PLAIN_E200_DYNAMICS"
        and compute_control.get("source_path")
        == "operations/WINNER_ABLATION_ADJUDICATION.json"
        and len(str(compute_control.get("source_sha256", ""))) == 64
        and compute_control.get("dynamics_state_exact_plain") is True
        and compute_control.get("candidate_dynamics_state_sha256")
        == compute_control.get("plain_dynamics_state_sha256")
        and float(compute_control.get("late_three_mean_macro_psnr_delta", 1.0))
        == 0.0
        and float(compute_control.get("e200_macro_psnr_delta", 1.0)) == 0.0
        and compute_control.get(
            "rules_out_wall_clock_or_observer_side_effect_only"
        ) is True
        and compute_control.get("does_not_claim_native_compute_budget_equivalence")
        is True
        and compute_control.get("paired_metrics_used_for_training_or_control")
        is False
        and compute_control.get("paired_controller_access") is False
        and compute_control.get("confirmation20_opened") is False,
        "gain-source compute-only control is absent or overclaimed",
    )
    player_scope = decomposition.get("player_scope_control")
    _require(
        isinstance(player_scope, dict)
        and player_scope.get("schema")
        == "final-unsb-route1-related-player-scope-control-v1"
        and player_scope.get("status")
        == "GF_ONLY_REPLICATION_SUSTAINS_E200_WHILE_FULL_PLAYER_REPLICATION_DOES_NOT"
        and player_scope.get("gf_only", {}).get("candidate_id") == PROPOSAL
        and player_scope.get("all_players", {}).get("candidate_id")
        == "G1-02B-PLAYER-CONDITIONAL-RSMG"
        and float(player_scope.get("gf_only", {}).get("e200_macro_psnr_delta", 0.0))
        > 0.0
        and float(player_scope.get("all_players", {}).get("e200_macro_psnr_delta", 1.0))
        <= 0.0
        and player_scope.get("gf_only", {}).get("strict_gate_pass") is True
        and player_scope.get("all_players", {}).get("strict_gate_pass") is False
        and player_scope.get("does_not_claim_additive_single_path_causality") is True
        and player_scope.get("paired_metrics_used_only_after_complete_e200_trajectories")
        is True
        and player_scope.get("paired_metrics_used_for_training_or_control") is False
        and player_scope.get("paired_controller_access") is False
        and player_scope.get("confirmation20_opened") is False,
        "gain-source player-scope control is absent or overclaimed",
    )
    gf_only = player_scope["gf_only"]
    all_players = player_scope["all_players"]
    scope_delta = player_scope.get("gf_only_minus_all_players")
    _require(
        isinstance(scope_delta, dict)
        and math.isclose(
            float(gf_only["late_three_mean_macro_psnr_delta"])
            - float(all_players["late_three_mean_macro_psnr_delta"]),
            float(scope_delta["late_three_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(gf_only["e200_macro_psnr_delta"])
            - float(all_players["e200_macro_psnr_delta"]),
            float(scope_delta["e200_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "gain-source player-scope arithmetic changed",
    )
    _require(
        bool(decomposition.get("optimizer_nonlinearity_boundary")),
        "gain-source decomposition lost the pre-Adam boundary",
    )
    variance_scope = decomposition.get("stochastic_variance_scope")
    _require(
        isinstance(variance_scope, dict)
        and variance_scope.get("conditioning_includes_official_unpaired_batch") is True
        and bool(variance_scope.get("reduced_components"))
        and bool(variance_scope.get("not_reduced_components"))
        and bool(variance_scope.get("iid_requirement")),
        "gain-source decomposition overclaimed cross-batch variance reduction",
    )
    resampling = decomposition.get("conditional_resampling_control")
    _require(
        isinstance(resampling, dict)
        and resampling.get("schema")
        == "final-unsb-route1-related-conditional-resampling-control-v1"
        and resampling.get("status")
        == "FRESH_POST_DE_RESAMPLING_ALONE_FAILS_WHILE_TWO_VIEW_GF_MEAN_PASSES"
        and resampling.get("source_path")
        == "operations/COMPLETE_FRONTIER_4090_ADJUDICATION.json"
        and len(str(resampling.get("source_sha256", ""))) == 64
        and resampling.get("resampling_only", {}).get("candidate_id") == PCNR
        and resampling.get("resampling_plus_two_view_mean", {}).get("candidate_id")
        == PROPOSAL
        and float(resampling.get("resampling_only", {}).get(
            "late_three_mean_macro_psnr_delta", 1.0
        )) <= 0.0
        and float(resampling.get("resampling_only", {}).get(
            "e200_macro_psnr_delta", 1.0
        )) <= 0.0
        and float(resampling.get("resampling_plus_two_view_mean", {}).get(
            "late_three_mean_macro_psnr_delta", 0.0
        )) > 0.0
        and float(resampling.get("resampling_plus_two_view_mean", {}).get(
            "e200_macro_psnr_delta", 0.0
        )) > 0.0
        and resampling.get("only_tested_operator_scope") is True
        and resampling.get("does_not_claim_global_necessity") is True
        and resampling.get("does_not_claim_additive_single_path_causality") is True
        and resampling.get("paired_metrics_used_only_after_complete_e200_trajectories")
        is True
        and resampling.get("paired_metrics_used_for_training_or_control") is False
        and resampling.get("paired_controller_access") is False
        and resampling.get("confirmation20_opened") is False,
        "gain-source conditional-resampling control is absent or overclaimed",
    )
    resampling_only = resampling["resampling_only"]
    two_view = resampling["resampling_plus_two_view_mean"]
    resampling_increment = resampling.get("two_view_mean_increment_over_resampling_only")
    _require(
        isinstance(resampling_increment, dict)
        and math.isclose(
            float(two_view["late_three_mean_macro_psnr_delta"])
            - float(resampling_only["late_three_mean_macro_psnr_delta"]),
            float(resampling_increment["late_three_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(two_view["e200_macro_psnr_delta"])
            - float(resampling_only["e200_macro_psnr_delta"]),
            float(resampling_increment["e200_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "gain-source conditional-resampling arithmetic changed",
    )
    hj_factorial = decomposition.get("hj_specific_factorial_control")
    hj_parent = hj_factorial.get("continuous_hj_parent") if isinstance(
        hj_factorial, dict
    ) else None
    one_view = hj_factorial.get("one_fresh_view") if isinstance(
        hj_factorial, dict
    ) else None
    two_view_hj = hj_factorial.get("two_fresh_view_mean") if isinstance(
        hj_factorial, dict
    ) else None
    one_increment = hj_factorial.get("one_view_increment_over_hj") if isinstance(
        hj_factorial, dict
    ) else None
    two_increment = hj_factorial.get(
        "two_view_mean_increment_over_one_view"
    ) if isinstance(hj_factorial, dict) else None
    _require(
        isinstance(hj_factorial, dict)
        and hj_factorial.get("schema")
        == "final-unsb-route1-hj-specific-resampling-variance-control-v1"
        and hj_factorial.get("status")
        == "COMPLETE_E200_HJ_ONE_VS_TWO_VIEW_FACTORIAL_CONTROL"
        and hj_factorial.get("source_path")
        == "operations/HJPCNR_GAIN_SOURCE_E200_RECEIPT.json"
        and len(str(hj_factorial.get("source_sha256", ""))) == 64
        and len(str(hj_factorial.get("trajectory_sha256", ""))) == 64
        and isinstance(hj_parent, dict)
        and hj_parent.get("parent_id") == "hj"
        and isinstance(one_view, dict)
        and one_view.get("candidate_id") == HJPCNR
        and isinstance(two_view_hj, dict)
        and two_view_hj.get("candidate_id") == HJCGR
        and isinstance(one_increment, dict)
        and isinstance(two_increment, dict)
        and hj_factorial.get(
            "paired_parent_result_used_only_to_authorize_completed_parent_ablation"
        ) is True
        and hj_factorial.get("paired_metrics_used_for_training_or_control") is False
        and hj_factorial.get("paired_controller_access") is False
        and hj_factorial.get("confirmation20_opened") is False,
        "HJ-specific one-view/two-view gain-source control is absent or unbound",
    )
    _require(
        math.isclose(
            float(one_view["late_three_mean_macro_psnr_delta"])
            - float(hj_parent["late_three_mean_macro_psnr_delta"]),
            float(one_increment["late_three_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(one_view["e200_macro_psnr_delta"])
            - float(hj_parent["e200_macro_psnr_delta"]),
            float(one_increment["e200_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(two_view_hj["late_three_mean_macro_psnr_delta"])
            - float(one_view["late_three_mean_macro_psnr_delta"]),
            float(two_increment["late_three_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(two_view_hj["e200_macro_psnr_delta"])
            - float(one_view["e200_macro_psnr_delta"]),
            float(two_increment["e200_macro_psnr_delta"]),
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "HJ-specific one-view/two-view gain-source arithmetic changed",
    )
    alignment = decomposition.get("variance_axis_alignment")
    alignment_members = alignment.get("members") if isinstance(alignment, dict) else None
    hj_alignment = alignment_members.get(HJCGR) if isinstance(alignment_members, dict) else None
    hnek_alignment = alignment_members.get(HPCGR) if isinstance(alignment_members, dict) else None
    _require(
        isinstance(alignment, dict)
        and alignment.get("schema")
        == "final-unsb-route1-related-variance-axis-alignment-v1"
        and alignment.get("operator_axis")
        == "within_batch_latent_time_bridge_and_feature_sampling"
        and alignment.get("causal_matrix_path") == "audit/LONG_CAUSAL_MATRIX.json"
        and len(str(alignment.get("causal_matrix_sha256", ""))) == 64
        and isinstance(hj_alignment, dict)
        and hj_alignment.get("alignment")
        == "directly_aligned_with_parent_audited_variance_axis"
        and int(hj_alignment.get("latent_time_bridge_rng", {}).get("rows", 0)) > 0
        and int(hj_alignment.get("latent_time_bridge_rng", {}).get(
            "variance_dominated_rows", -1
        )) == int(hj_alignment.get("latent_time_bridge_rng", {}).get("rows", 0))
        and isinstance(hnek_alignment, dict)
        and hnek_alignment.get("alignment")
        == "compositional_transfer_hypothesis_not_direct_axis_repair"
        and int(hnek_alignment.get("latent_time_bridge_rng", {}).get("rows", 0)) > 0
        and int(hnek_alignment.get("latent_time_bridge_rng", {}).get(
            "variance_dominated_rows", -1
        )) == 0
        and hnek_alignment.get("unaddressed_parent_axis")
        == "independent_unpaired_batch"
        and alignment.get("shared_theorem_does_not_imply_shared_failure_mode") is True
        and alignment.get("hpcgr_viability_must_be_decided_by_complete_e200_trajectory")
        is True
        and alignment.get("paired_metrics_used_for_formula_or_training_control") is False
        and alignment.get("confirmation20_opened") is False,
        "gain-source variance-axis alignment is absent or overclaimed",
    )
    rows = decomposition.get("members")
    _require(isinstance(rows, list), "gain-source member rows are malformed")
    by_id = {
        str(row.get("candidate_id", "")): row
        for row in rows if isinstance(row, dict)
    }
    _require(set(by_id) == expected_family, "gain-source member identity changed")
    expected_parents = {PROPOSAL: "plain", HPCGR: "hnek", HJCGR: "hj"}
    positive = []
    recomputed = {}
    for candidate_id, parent_id in expected_parents.items():
        row = by_id[candidate_id]
        parent = row.get("parent")
        child = row.get("composed")
        increment = row.get("matched_compositional_increment_over_parent")
        _require(
            all(isinstance(value, dict) for value in (parent, child, increment)),
            f"gain-source row is malformed: {candidate_id}",
        )
        _require(
            parent.get("parent_id") == parent_id,
            f"gain-source parent changed: {candidate_id}",
        )
        late = float(child["late_three_mean_macro_psnr_delta"]) - float(
            parent["late_three_mean_macro_psnr_delta"]
        )
        e200 = float(child["e200_macro_psnr_delta"]) - float(
            parent["e200_macro_psnr_delta"]
        )
        _require(
            math.isclose(
                late, float(increment["late_three_macro_psnr_delta"]),
                rel_tol=0.0, abs_tol=1e-12,
            )
            and math.isclose(
                e200, float(increment["e200_macro_psnr_delta"]),
                rel_tol=0.0, abs_tol=1e-12,
            ),
            f"gain-source arithmetic changed: {candidate_id}",
        )
        if late > 0.0 and e200 > 0.0:
            positive.append(candidate_id)
        recomputed[candidate_id] = {"late_three": late, "e200": e200}
    _require(
        decomposition.get("shared_estimator_positive_increment_candidate_ids")
        == positive,
        "gain-source positive member list differs from recomputation",
    )
    _require(
        int(decomposition.get("shared_estimator_positive_increment_count", -1))
        == len(positive),
        "gain-source positive count differs from recomputation",
    )
    _require(
        results.get("mechanism_gain_source_decomposition") == decomposition,
        "related results and algorithm set contain different gain-source evidence",
    )
    return {
        "family_candidate_ids": sorted(expected_family),
        "independent_candidate_ids": sorted(independent_ids),
        "positive_increment_candidate_ids": positive,
        "recomputed_matched_increments": recomputed,
        "conditional_theorem_present": True,
        "compute_only_control_proven": True,
        "player_scope_control_proven": True,
        "conditional_resampling_control_proven": True,
        "hj_specific_factorial_control_proven": True,
        "within_batch_variance_scope_proven": True,
        "parent_variance_axis_alignment_proven": True,
        "pre_adam_unbiasedness_boundary_proven": True,
        "native_compute_budget_equivalence_not_claimed": True,
        "additive_causality_not_claimed": True,
    }


def audit_related_goal_completion(
    compatibility_delivery: Path, related_delivery: Path,
) -> dict[str, Any]:
    compatibility = audit_complete_delivery(Path(compatibility_delivery))
    related_delivery = Path(related_delivery).resolve()
    pointer = validate_related_delivery(related_delivery)
    algorithm_set = _read_json(related_delivery / "ALGORITHM_SET.json")
    action = _read_json(related_delivery / "ACTION_PRIORITY.json")
    candidate = _read_json(related_delivery / "CANDIDATE.json")
    alternates = _read_json(related_delivery / "ALTERNATES.json")
    results = _read_json(related_delivery / "RELATED_RESULTS.json")

    _require(
        algorithm_set.get("schema") == ALGORITHM_SET_SCHEMA,
        "related algorithm-set schema changed",
    )
    _require(
        algorithm_set.get("status") == pointer.get("algorithm_set_status"),
        "related algorithm-set status differs from pointer",
    )
    _require(
        algorithm_set.get("action_priority_is_not_scientific_exclusivity") is True,
        "related algorithm-set made action priority exclusive",
    )
    _require(
        algorithm_set.get("algorithm_discovery_collapsed_to_single_candidate") is False,
        "related algorithm-set collapsed discovery to one candidate",
    )
    _posthoc_boundary(algorithm_set, label="related algorithm set")
    _require(action.get("schema") == ACTION_SCHEMA, "related action schema changed")
    _require(
        action.get("status") == "CURRENT_NEXT_ACTION_PRIORITY",
        "related action is not terminal",
    )
    _require(
        action.get("candidate_id") == pointer.get("action_priority_candidate_id"),
        "related action identity differs",
    )
    _require(
        action.get("action_priority_is_not_scientific_exclusivity") is True,
        "related action priority is exclusive",
    )
    _posthoc_boundary(action, label="related action")
    _require(
        candidate.get("schema") == CANDIDATE_SCHEMA
        and candidate.get("status")
        == "CURRENT_ACTION_PRIORITY_FROM_MULTI_ALGORITHM_SET",
        "related action candidate is not terminal",
    )
    _require(
        candidate.get("candidate_id") == action.get("candidate_id")
        and candidate.get("canonical_candidate_is_action_priority_only") is True
        and candidate.get("action_priority_is_not_scientific_exclusivity") is True,
        "related action candidate changed multi-algorithm semantics",
    )
    selected_algorithm = candidate.get("algorithm")
    _require(
        isinstance(selected_algorithm, dict)
        and selected_algorithm.get("candidate_id") == action.get("candidate_id")
        and isinstance(selected_algorithm.get("risk"), dict)
        and isinstance(selected_algorithm.get("reproduction"), dict)
        and bool(selected_algorithm["reproduction"].get("seed2026_e200"))
        and selected_algorithm["reproduction"].get("deferred_seed_validation")
        == [2027, 2028],
        "related action candidate is not actionable/reproducible",
    )
    _posthoc_boundary(candidate, label="related action candidate")
    _require(
        alternates.get("schema") == ALTERNATES_SCHEMA
        and alternates.get("status") == "TWO_EVIDENCE_RANKED_ALTERNATES",
        "related alternates are not terminal",
    )
    alternate_rows = alternates.get("alternates")
    alternate_ids = [
        str(row.get("candidate_id", ""))
        for row in alternate_rows or [] if isinstance(row, dict)
    ]
    _require(
        len(alternate_ids) == 2
        and len(set(alternate_ids)) == 2
        and action.get("candidate_id") not in alternate_ids,
        "related delivery does not contain two distinct alternates",
    )
    _posthoc_boundary(alternates, label="related alternates")
    _require(results.get("schema") == RESULTS_SCHEMA, "related results schema changed")
    _require(
        results.get("status") == "RELATED_MULTI_ALGORITHM_E200_COMPLETE",
        "related results are not terminal",
    )
    _posthoc_boundary(results, label="related results")

    members = algorithm_set.get("members")
    _require(isinstance(members, list) and bool(members), "algorithm set is empty")
    ids = [str(row.get("candidate_id", "")) for row in members]
    _require(len(ids) == len(set(ids)), "algorithm set contains duplicate candidates")
    members_by_id = {str(row["candidate_id"]): row for row in members}
    _require(
        selected_algorithm == members_by_id.get(str(action.get("candidate_id", ""))),
        "related action candidate differs from algorithm-set member",
    )
    for alternate in alternate_rows:
        alternate_id = str(alternate["candidate_id"])
        _require(
            alternate.get("algorithm") == members_by_id.get(alternate_id),
            f"related alternate differs from algorithm-set member: {alternate_id}",
        )
    strict = algorithm_set.get("strict_viable_candidate_ids")
    fragile = algorithm_set.get("positive_but_fragile_candidate_ids")
    _require(isinstance(strict, list), "strict viable algorithm list is malformed")
    _require(isinstance(fragile, list), "fragile algorithm list is malformed")
    _require(
        len(strict) == int(pointer.get("strict_viable_candidate_count", -1)),
        "strict viable algorithm count differs",
    )
    _require(set(strict).issubset(ids), "strict viable algorithm is absent from members")
    _require(set(fragile).issubset(ids), "fragile algorithm is absent from members")
    _require(set(alternate_ids).issubset(ids), "alternate is absent from algorithm set")
    strict_from_members = [
        str(member["candidate_id"]) for member in members
        if member.get("disposition") == "strict_viable_algorithm"
    ]
    fragile_from_members = [
        str(member["candidate_id"]) for member in members
        if member.get("disposition") == "positive_but_fragile_algorithm"
    ]
    _require(
        strict == strict_from_members,
        "strict viable list differs from member dispositions",
    )
    _require(
        fragile == fragile_from_members,
        "fragile list differs from member dispositions",
    )
    ranking = algorithm_set.get("same_host_4090_ranking")
    _require(isinstance(ranking, list) and bool(ranking), "same-host ranking is absent")
    ranking_ids = [str(row.get("candidate_id", "")) for row in ranking]
    _require(
        ranking_ids == ids
        and [int(row.get("rank", -1)) for row in ranking]
        == list(range(1, len(ranking) + 1)),
        "same-host ranking and algorithm members differ",
    )
    _require(
        ranking_ids[0] == str(action.get("candidate_id", "")),
        "action priority is not same-host rank one",
    )
    _require(
        results.get("composite_same_host_4090_ranking") == ranking,
        "related results and algorithm-set rankings differ",
    )
    for member in members:
        if member.get("disposition") in (
            "strict_viable_algorithm", "positive_but_fragile_algorithm",
        ):
            _require(
                isinstance(member.get("risk"), dict)
                and bool(member.get("reproduction", {}).get("seed2026_e200")),
                f"retained algorithm lacks risk/reproduction: {member.get('candidate_id')}",
            )
    gain_source_proof = _audit_gain_source(
        algorithm_set, results, set(ids),
    )

    trajectory_proofs = {}
    for member in members:
        candidate_id = str(member["candidate_id"])
        _require(
            isinstance(member.get("mathematics"), dict)
            and member["mathematics"].get("formula") is not None,
            f"algorithm-set member lacks mathematics: {candidate_id}",
        )
        _require(
            isinstance(member.get("source_bound"), dict),
            f"algorithm-set member lacks source binding: {candidate_id}",
        )
        trajectory_proofs[candidate_id] = _domain_trajectory(
            member.get("absolute_relative_domain_trajectory"),
            label=f"related algorithm {candidate_id}",
        )

    combined = algorithm_set.get("cross_runtime_related_evidence")
    _require(isinstance(combined, dict), "related cross-runtime evidence is absent")
    _require(
        combined.get("cross_runtime_is_not_cross_seed") is True,
        "related evidence conflates runtime with seed",
    )
    _posthoc_boundary(combined, label="related cross-runtime evidence")
    for algorithm in combined.get("algorithms", []):
        for host_result in algorithm.get("host_results", []):
            snapshot = host_result.get("trajectory_snapshot")
            _require(
                isinstance(snapshot, dict)
                and snapshot.get("confirmation20_opened") is False,
                "related host result lost its complete trajectory snapshot",
            )

    return {
        "schema": SCHEMA,
        "status": "RELATED_ROUTE1_TERMINAL_ARTIFACTS_PROVEN_FINAL_GIT_COMMIT_REQUIRED",
        "action_priority_candidate_id": pointer["action_priority_candidate_id"],
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_set_status": pointer["algorithm_set_status"],
        "strict_viable_candidate_ids": strict,
        "positive_but_fragile_candidate_ids": fragile,
        "alternate_candidate_ids": alternate_ids,
        "algorithm_member_count": len(members),
        "trajectory_proofs": trajectory_proofs,
        "gain_source_proof": gain_source_proof,
        "compatibility_goal_audit": compatibility,
        "terminal_artifact_requirements_proven": True,
        "final_repository_commit_and_push_required": True,
        "completion_claim_allowed": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        "source_delivery_sha256": {
            name: file_sha256(related_delivery / name)
            for name in (POINTER, *PUBLISHED_FILES)
        },
    }


def materialize_related_goal_completion_audit(
    compatibility_delivery: Path, related_delivery: Path, output: Path,
) -> dict[str, Any]:
    result = audit_related_goal_completion(
        compatibility_delivery, related_delivery,
    )
    write_json(Path(output).resolve(), result)
    return result
