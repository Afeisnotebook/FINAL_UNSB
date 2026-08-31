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
    AMTNC,
    HJCGR,
    HPCGR,
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
    _require(
        bool(family.get("conditional_expectation_property"))
        and bool(family.get("conditional_covariance_property")),
        "related family lost its conditional theorem",
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
        expected_family.union({AMTNC}).issubset(member_ids),
        "terminal algorithm set omitted a required related/independent member",
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
