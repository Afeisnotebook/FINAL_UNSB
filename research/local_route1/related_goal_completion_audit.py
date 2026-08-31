"""Final Goal audit that requires both compatibility and algorithm-set deliveries."""

from __future__ import annotations

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
    POINTER,
    PUBLISHED_FILES,
    RESULTS_SCHEMA,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-related-goal-completion-audit-v1"


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

