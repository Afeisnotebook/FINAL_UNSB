"""Adjudicate the repaired 5090 frontier while retaining invalid diagnostics.

Only source-bound, common-e0, complete e200 receipts are rankable.  The two
fixed-absolute-margin implementations are retained as implementation
diagnostics, but their semantic incidents make them ineligible to represent
the registered closest-point mechanisms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from operations.local_route1_cross_version_adjudicate import (
    _rank_key,
    _validate_receipt,
)
from operations.local_route1_implementation_invalid_diagnostic_receipt import (
    DIAGNOSTIC_RECEIPT_SCHEMA,
    DIAGNOSTIC_SIDECAR_SCHEMA,
    INCIDENTS as INVALID_DIAGNOSTIC_INCIDENTS,
)
from research.local_route1.frontier_advancement import (
    NEAR,
    STRICT,
    classify_complete_trajectory,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-repaired-frontier-adjudication-v1"
ACTIONABLE_STATUS = "REPAIRED_FRONTIER_COMPLETE_ACTION_PRIORITY_AVAILABLE"
FALLBACK_STATUS = "REPAIRED_FRONTIER_COMPLETE_FALLBACK_ONLY"
RANKABLE_IDS = (
    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING",
    "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER",
    "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER",
)
COMMON_AUTHORITY_FIELDS = (
    "base_e0_scientific_state_sha256",
    "base_protocol_fingerprint",
    "manifest_sha256",
    "plain_e200_verification_sha256",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validated_receipt_map(
    receipt_paths: Iterable[Path], expected_ids: Iterable[str], *, label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    paths = [Path(path).resolve() for path in receipt_paths]
    expected = tuple(expected_ids)
    if len(paths) != len(expected):
        raise RuntimeError(f"{label} requires exactly {len(expected)} receipts")
    receipts = [_validate_receipt(path) for path in paths]
    ids = [str(receipt["candidate_id"]) for receipt in receipts]
    if set(ids) != set(expected) or len(set(ids)) != len(expected):
        raise RuntimeError(f"{label} receipt identities differ from the frozen set")
    return (
        {str(receipt["candidate_id"]): receipt for receipt in receipts},
        {str(receipt["candidate_id"]): path for receipt, path in zip(receipts, paths)},
    )


def _validate_diagnostic_receipt(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    sidecar_path = Path(str(path) + ".sha256.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"implementation diagnostic receipt is missing: {path}")
    receipt = _read_json(path)
    sidecar = _read_json(sidecar_path)
    if (
        receipt.get("schema") != DIAGNOSTIC_RECEIPT_SCHEMA
        or receipt.get("status")
        != "ACCEPTED_IMPLEMENTATION_INVALID_COMPLETE_E200_DIAGNOSTIC"
        or sidecar.get("schema") != DIAGNOSTIC_SIDECAR_SCHEMA
        or sidecar.get("candidate_id") != receipt.get("candidate_id")
        or sidecar.get("receipt_sha256") != file_sha256(path)
    ):
        raise RuntimeError(f"implementation diagnostic receipt integrity failed: {path}")
    if receipt.get("scientific_ranking_eligible") is not False:
        raise RuntimeError("implementation-invalid diagnostic became scientifically rankable")
    if receipt.get("parent_mechanism_falsified") is not False:
        raise RuntimeError("implementation diagnostic incorrectly falsified its parent")
    if receipt.get("terminal_integrity", {}).get("status") != (
        "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"
    ):
        raise RuntimeError("implementation diagnostic lacks complete e200 artifacts")
    expected_source = file_sha256(
        ROOT / "operations" / "local_route1_implementation_invalid_diagnostic_receipt.py"
    )
    if receipt.get("receipt_source_sha256") != expected_source:
        raise RuntimeError("implementation diagnostic is not source-bound")
    for key in (
        "evaluation_crn_matched_to_same_host_plain",
        "paired_metrics_used_only_after_complete_trajectory",
    ):
        if receipt.get(key) is not True:
            raise RuntimeError(f"implementation diagnostic requires {key}=true")
    for key in (
        "paired_metrics_used_for_training_or_control", "paired_controller_access",
        "confirmation20_opened",
    ):
        if receipt.get(key) is not False:
            raise RuntimeError(f"implementation diagnostic requires {key}=false")
    return receipt


def _validated_diagnostic_map(
    receipt_paths: Iterable[Path], expected_ids: Iterable[str], *, label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    paths = [Path(path).resolve() for path in receipt_paths]
    expected = tuple(expected_ids)
    if len(paths) != len(expected):
        raise RuntimeError(f"{label} requires exactly {len(expected)} receipts")
    receipts = [_validate_diagnostic_receipt(path) for path in paths]
    ids = [str(receipt["candidate_id"]) for receipt in receipts]
    if set(ids) != set(expected) or len(set(ids)) != len(expected):
        raise RuntimeError(f"{label} receipt identities differ from the frozen set")
    return (
        {str(receipt["candidate_id"]): receipt for receipt in receipts},
        {str(receipt["candidate_id"]): path for receipt, path in zip(receipts, paths)},
    )


def _validate_same_host_authority(
    receipts: Iterable[dict[str, Any]],
) -> dict[str, str]:
    values = list(receipts)
    authority: dict[str, str] = {}
    for field in COMMON_AUTHORITY_FIELDS:
        unique = {str(receipt.get(field, "")) for receipt in values}
        if len(unique) != 1 or not next(iter(unique)):
            raise RuntimeError(
                f"repaired frontier receipts differ on same-host authority: {field}"
            )
        authority[field] = next(iter(unique))
    return authority


def _validate_invalid_diagnostic(
    candidate_id: str, receipt: dict[str, Any], receipt_path: Path,
) -> dict[str, Any]:
    incident_path = (ROOT / INVALID_DIAGNOSTIC_INCIDENTS[candidate_id]).resolve()
    incident = _read_json(incident_path)
    invalid_identity = incident.get("invalid_identity") or {}
    required = {
        "schema": "final-unsb-route1-semantic-incident-v1",
        "candidate_id": candidate_id,
        "classification": "implementation_failure",
        "scientific_conclusion_allowed": False,
        "parent_mechanism_falsified": False,
        "paired_metric_used_for_discovery_or_repair": False,
        "confirmation20_opened": False,
    }
    for key, expected in required.items():
        if incident.get(key) != expected:
            raise RuntimeError(f"invalid diagnostic incident changed: {candidate_id}:{key}")
    if invalid_identity.get("algorithm_fingerprint") != receipt.get(
        "algorithm_fingerprint"
    ):
        raise RuntimeError(
            f"invalid diagnostic receipt is not bound to its incident: {candidate_id}"
        )
    treatment = incident.get("running_trajectory_treatment") or incident.get(
        "completed_trajectory_treatment"
    ) or {}
    if treatment.get("rank_as_faithful_test_of_registered_formula") is not False:
        raise RuntimeError("semantic incident does not forbid scientific ranking")
    return {
        "candidate_id": candidate_id,
        "trajectory_status": receipt["trajectory_status"],
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "ranking_fields_retained_for_diagnosis_only": receipt["ranking_fields"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "incident_path": incident_path.relative_to(ROOT).as_posix(),
        "incident_sha256": file_sha256(incident_path),
        "scientific_ranking_eligible": False,
        "parent_mechanism_falsified": False,
        "exclusion_reason": "implementation_semantics_do_not_match_registered_operator",
    }


def adjudicate_repaired_frontier(
    rankable_receipt_paths: Iterable[Path],
    invalid_diagnostic_receipt_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    rankable, rankable_paths = _validated_receipt_map(
        rankable_receipt_paths, RANKABLE_IDS, label="rankable repaired frontier",
    )
    invalid, invalid_paths = _validated_diagnostic_map(
        invalid_diagnostic_receipt_paths,
        INVALID_DIAGNOSTIC_INCIDENTS,
        label="implementation-invalid diagnostic set",
    )
    authority = _validate_same_host_authority([*rankable.values(), *invalid.values()])

    rows = []
    for candidate_id, receipt in rankable.items():
        trajectory_path = Path(receipt["trajectory_path"]).resolve()
        if not trajectory_path.is_file() or file_sha256(trajectory_path) != receipt.get(
            "trajectory_sha256"
        ):
            raise RuntimeError(f"rankable trajectory integrity failed: {candidate_id}")
        trajectory = _read_json(trajectory_path)
        rows.append((receipt, trajectory, classify_complete_trajectory(receipt, trajectory)))

    ranked = sorted(rows, key=lambda row: _rank_key(row[0]))
    strict = [row for row in ranked if row[2]["classification"] == STRICT]
    preserved = [
        row for row in ranked
        if row[2]["classification"] in (STRICT, NEAR)
        or row[0].get("trajectory_status") == POSITIVE_STATUS
    ]
    action = strict[0] if strict else ranked[0]
    replay_queue = [row for row in ranked if row[2]["classification"] in (STRICT, NEAR)]
    ranking = []
    for index, (receipt, _trajectory, classification) in enumerate(ranked, start=1):
        path = rankable_paths[str(receipt["candidate_id"])]
        ranking.append({
            "rank": index,
            "candidate_id": receipt["candidate_id"],
            "classification": classification["classification"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "candidate_training_core_fingerprint": receipt[
                "candidate_training_core_fingerprint"
            ],
            "ranking_fields": receipt["ranking_fields"],
            "classification_checks": classification["checks"],
            "receipt_path": str(path),
            "receipt_sha256": file_sha256(path),
            "trajectory_path": receipt["trajectory_path"],
            "trajectory_sha256": receipt["trajectory_sha256"],
            "median_epoch_wall_seconds": receipt["median_epoch_wall_seconds"],
        })

    diagnostic_rows = [
        _validate_invalid_diagnostic(candidate_id, invalid[candidate_id], invalid_paths[candidate_id])
        for candidate_id in sorted(INVALID_DIAGNOSTIC_INCIDENTS)
    ]
    result = {
        "schema": SCHEMA,
        "status": ACTIONABLE_STATUS if strict else FALLBACK_STATUS,
        "same_host_authority": authority,
        "ranking": ranking,
        "action_priority_candidate_id": action[0]["candidate_id"],
        "action_priority_role": (
            "same_host_strict_seed2026_priority_pending_source_bound_4090_replay"
            if strict else "same_host_current_best_fallback"
        ),
        "priority_alternate_candidate_ids": [
            row[0]["candidate_id"] for row in ranked
            if row[0]["candidate_id"] != action[0]["candidate_id"]
        ][:2],
        "strict_candidate_ids": [row[0]["candidate_id"] for row in strict],
        "evidence_preserved_candidate_ids": [
            row[0]["candidate_id"] for row in preserved
        ],
        "recommended_4090_replay_queue": [
            row[0]["candidate_id"] for row in replay_queue
        ],
        "implementation_invalid_diagnostics": diagnostic_rows,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "small_score_differences_do_not_trigger_early_pruning": True,
        "rankable_complete_e200_candidate_count": len(ranked),
        "diagnostic_complete_e200_candidate_count": len(diagnostic_rows),
        "old_invalid_operators_excluded_from_scientific_ranking": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_path).resolve(), result)
    return result
