"""Adjudicate full repaired operators and every completed proposal ablation.

The original full operators remain in the table.  A proposal-only branch may
become the action priority if its own complete e200 receipt ranks higher.  The
observable-only branches are exact-dynamics negative controls and never enter
candidate ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import (
    _rank_key,
    _validate_receipt,
)
from operations.local_route1_winner_ablation_adjudicate import (
    _observable_identity,
)
from research.local_route1.frontier_advancement import (
    ALTERNATE,
    NEAR,
    STRICT,
    classify_complete_trajectory,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.protocol import file_sha256
from research.local_route1.repaired_frontier_adjudication import (
    RANKABLE_IDS,
    SCHEMA as REPAIRED_SCHEMA,
)
from research.local_route1.repaired_frontier_followups import (
    SCHEMA as FOLLOWUP_SCHEMA,
)
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-extended-repaired-frontier-adjudication-v1"
EXECUTION_SCHEMA = "final-unsb-route1-repaired-followup-execution-v1"
BASELINE_FIELDS = (
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


def _bound_json(output_root: Path, path: Path, *, schema: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(output_root):
        raise RuntimeError(f"extended repaired evidence escaped run root: {path}")
    value = _read_json(path)
    if value.get("schema") != schema:
        raise RuntimeError(f"extended repaired evidence schema changed: {path}")
    if value.get("paired_controller_access") is not False or value.get(
        "confirmation20_opened"
    ) is not False:
        raise RuntimeError("extended repaired evidence violates target-blind scope")
    return value


def _receipt_row(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = _validate_receipt(path)
    trajectory_path = Path(str(receipt["trajectory_path"])).resolve()
    if not trajectory_path.is_file() or file_sha256(trajectory_path) != receipt.get(
        "trajectory_sha256"
    ):
        raise RuntimeError("extended repaired trajectory changed")
    trajectory = _read_json(trajectory_path)
    classification = classify_complete_trajectory(receipt, trajectory)
    return receipt, trajectory, classification


def _scientific_key(
    receipt: dict[str, Any], classification: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        0 if classification["classification"] == STRICT else 1,
        *_rank_key(receipt),
    )


def materialize_extended_repaired_frontier(
    output_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    repaired_path = operations / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
    followup_path = operations / "REPAIRED_FRONTIER_FOLLOWUPS.json"
    execution_path = operations / "REPAIRED_FOLLOWUP_EXECUTION_RESULT.json"
    repaired = _bound_json(output_root, repaired_path, schema=REPAIRED_SCHEMA)
    followup = _bound_json(output_root, followup_path, schema=FOLLOWUP_SCHEMA)
    execution = _bound_json(output_root, execution_path, schema=EXECUTION_SCHEMA)
    if followup.get("source_adjudication_sha256") != file_sha256(repaired_path):
        raise RuntimeError("repaired follow-up plan no longer binds its adjudication")
    if execution.get("source_plan_sha256") != file_sha256(followup_path):
        raise RuntimeError("repaired follow-up execution no longer binds its plan")

    full_rows = repaired.get("ranking")
    if not isinstance(full_rows, list) or {
        str(row.get("candidate_id", "")) for row in full_rows
    } != set(RANKABLE_IDS):
        raise RuntimeError("extended repaired full frontier is incomplete")
    receipt_paths: dict[str, Path] = {}
    for row in full_rows:
        path = Path(str(row.get("receipt_path", ""))).resolve()
        if file_sha256(path) != row.get("receipt_sha256"):
            raise RuntimeError("extended repaired full receipt changed")
        receipt_paths[str(row["candidate_id"])] = path

    execution_parents = {
        str(row.get("parent_candidate_id", "")): row
        for row in execution.get("parent_results", [])
    }
    parent_ablations = []
    for stream in followup.get("eligible_parent_streams", []):
        parent_id = str(stream["parent_candidate_id"])
        worker = execution_parents.get(parent_id)
        if not isinstance(worker, dict) or worker.get("status") != (
            "PARENT_ABLATION_STREAM_COMPLETE_E200"
        ):
            raise RuntimeError(f"extended repaired parent stream is incomplete: {parent_id}")
        receipts = {
            str(row["candidate_id"]): Path(str(row["receipt_path"])).resolve()
            for row in worker.get("receipts", [])
        }
        ids = stream["ablation_candidate_ids"]
        proposal_id = str(ids["proposal_only"])
        observable_id = str(ids["observable_only"])
        if set(receipts) != {proposal_id, observable_id}:
            raise RuntimeError("extended repaired ablation receipt set changed")
        for row in worker["receipts"]:
            path = Path(str(row["receipt_path"])).resolve()
            if file_sha256(path) != row.get("receipt_sha256"):
                raise RuntimeError("extended repaired ablation receipt changed")
        full_receipt = _validate_receipt(receipt_paths[parent_id])
        proposal_receipt = _validate_receipt(receipts[proposal_id])
        observable_receipt = _validate_receipt(receipts[observable_id])
        for field in BASELINE_FIELDS:
            if len({
                str(value.get(field, ""))
                for value in (full_receipt, proposal_receipt, observable_receipt)
            }) != 1:
                raise RuntimeError(
                    f"extended repaired ablations differ on baseline: {field}"
                )
        identity = _observable_identity(output_root, observable_id)
        full_classification = _receipt_row(receipt_paths[parent_id])[2]
        proposal_classification = _receipt_row(receipts[proposal_id])[2]
        receipt_paths[proposal_id] = receipts[proposal_id]
        parent_ablations.append({
            "parent_candidate_id": parent_id,
            "parent_classification": stream["parent_classification"],
            "proposal_only_candidate_id": proposal_id,
            "observable_only_candidate_id": observable_id,
            "observable_only_identity": identity,
            "proposal_only_out_ranks_full": (
                _scientific_key(proposal_receipt, proposal_classification)
                < _scientific_key(full_receipt, full_classification)
            ),
            "roles": {
                "proposal_only": {
                    "candidate_id": proposal_id,
                    "receipt_path": str(receipts[proposal_id]),
                    "receipt_sha256": file_sha256(receipts[proposal_id]),
                    "ranking_fields": proposal_receipt["ranking_fields"],
                },
                "observable_only": {
                    "candidate_id": observable_id,
                    "receipt_path": str(receipts[observable_id]),
                    "receipt_sha256": file_sha256(receipts[observable_id]),
                    "ranking_fields": observable_receipt["ranking_fields"],
                },
                "projected_or_full": {
                    "candidate_id": parent_id,
                    "receipt_path": str(receipt_paths[parent_id]),
                    "receipt_sha256": file_sha256(receipt_paths[parent_id]),
                    "ranking_fields": full_receipt["ranking_fields"],
                },
            },
        })

    if set(execution_parents) != {
        str(row["parent_candidate_id"])
        for row in followup.get("eligible_parent_streams", [])
    }:
        raise RuntimeError("extended repaired execution has an unplanned parent")

    classified = [
        (*_receipt_row(path), path)
        for path in receipt_paths.values()
    ]
    authorities = {
        field: {str(row[0].get(field, "")) for row in classified}
        for field in BASELINE_FIELDS
    }
    if any(len(values) != 1 or not next(iter(values)) for values in authorities.values()):
        raise RuntimeError("extended repaired frontier is not same-host matched")
    ranked = sorted(
        classified,
        key=lambda row: _scientific_key(row[0], row[2]),
    )
    strict = [row for row in ranked if row[2]["classification"] == STRICT]
    action = ranked[0]
    preserved = [
        row for row in ranked
        if row[2]["classification"] in (STRICT, NEAR)
        or row[0].get("trajectory_status") == POSITIVE_STATUS
        or row[2]["classification"] == ALTERNATE
    ]
    ranking = []
    for rank, (receipt, _trajectory, classification, path) in enumerate(
        ranked, start=1,
    ):
        ranking.append({
            "rank": rank,
            "candidate_id": receipt["candidate_id"],
            "classification": classification["classification"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "ranking_fields": receipt["ranking_fields"],
            "classification_checks": classification["checks"],
            "receipt_path": str(path),
            "receipt_sha256": file_sha256(path),
            "trajectory_path": receipt["trajectory_path"],
            "trajectory_sha256": receipt["trajectory_sha256"],
        })
    action_id = str(action[0]["candidate_id"])
    result = {
        "schema": SCHEMA,
        "status": (
            "EXTENDED_REPAIRED_FRONTIER_STRICT_ACTION_PRIORITY_AVAILABLE"
            if strict else "EXTENDED_REPAIRED_FRONTIER_FALLBACK_ONLY"
        ),
        "same_host_authority": {
            field: next(iter(values)) for field, values in authorities.items()
        },
        "source_repaired_frontier_path": repaired_path.relative_to(
            output_root
        ).as_posix(),
        "source_repaired_frontier_sha256": file_sha256(repaired_path),
        "source_followup_plan_path": followup_path.relative_to(
            output_root
        ).as_posix(),
        "source_followup_plan_sha256": file_sha256(followup_path),
        "source_followup_execution_path": execution_path.relative_to(
            output_root
        ).as_posix(),
        "source_followup_execution_sha256": file_sha256(execution_path),
        "ranking_policy": (
            "strict sustained-e200 qualification first, then registered "
            "late-three/e200/domain/guardrail/cost key"
        ),
        "ranking": ranking,
        "action_priority_candidate_id": action_id,
        "priority_alternate_candidate_ids": [
            row[0]["candidate_id"] for row in ranked
            if row[0]["candidate_id"] != action_id
        ][:2],
        "strict_candidate_ids": [row[0]["candidate_id"] for row in strict],
        "evidence_preserved_candidate_ids": [
            row[0]["candidate_id"] for row in preserved
        ],
        "recommended_4090_replay_queue": [
            row[0]["candidate_id"] for row in strict
        ],
        "parent_ablation_results": parent_ablations,
        "observable_only_candidate_ids_excluded_from_ranking": [
            str(stream["ablation_candidate_ids"]["observable_only"])
            for stream in followup.get("eligible_parent_streams", [])
        ],
        "rankable_complete_e200_candidate_count": len(ranking),
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    output_path = (
        operations / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        if output_path is None else Path(output_path).resolve()
    )
    if not output_path.is_relative_to(output_root):
        raise RuntimeError("extended repaired frontier output escaped run root")
    write_json(output_path, result)
    return result
