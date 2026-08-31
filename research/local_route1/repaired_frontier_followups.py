"""Freeze multi-parent follow-ups for the complete repaired 5090 frontier.

The unique canonical candidate remains only an action priority.  Every
residual-feasible repair that remains strict, near-boundary, or an evidence-
backed alternate keeps an independent mechanism-ablation stream.  This module
never reads an intermediate metric and never changes a formula from paired
results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.local_route1.frontier_advancement import ALTERNATE, NEAR, STRICT
from research.local_route1.protocol import file_sha256
from research.local_route1.repaired_frontier_adjudication import (
    ACTIONABLE_STATUS,
    FALLBACK_STATUS,
    RANKABLE_IDS,
    SCHEMA as ADJUDICATION_SCHEMA,
)
from research.local_route1.runtime import write_json
from research.local_route1.winner_ablations import (
    WINNER_FAMILIES,
    materialize_parent_ablation_definitions,
)


SCHEMA = "final-unsb-route1-repaired-frontier-followups-v1"
REPAIRED_IDS = (
    "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER",
    "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_adjudication(output_root: Path, path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(output_root):
        raise RuntimeError("repaired frontier adjudication escaped the run root")
    value = _read_json(path)
    if (
        value.get("schema") != ADJUDICATION_SCHEMA
        or value.get("status") not in (ACTIONABLE_STATUS, FALLBACK_STATUS)
        or value.get("algorithm_discovery_collapsed_to_single_candidate") is not False
        or value.get("canonical_candidate_is_action_priority_only") is not True
        or value.get("paired_metrics_used_for_formula_or_training_control") is not False
        or value.get("paired_controller_access") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("repaired frontier adjudication boundary changed")
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or {
        str(row.get("candidate_id", "")) for row in ranking
    } != set(RANKABLE_IDS):
        raise RuntimeError("repaired frontier ranking is incomplete")
    for row in ranking:
        receipt = Path(str(row.get("receipt_path", ""))).resolve()
        if (
            not receipt.is_file()
            or not receipt.is_relative_to(output_root)
            or file_sha256(receipt) != row.get("receipt_sha256")
        ):
            raise RuntimeError("repaired frontier parent receipt changed")
    return value


def materialize_repaired_frontier_followups(
    output_root: Path,
    *,
    adjudication_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    adjudication_path = (
        output_root / "operations" / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        if adjudication_path is None else Path(adjudication_path).resolve()
    )
    output_path = (
        output_root / "operations" / "REPAIRED_FRONTIER_FOLLOWUPS.json"
        if output_path is None else Path(output_path).resolve()
    )
    adjudication = _validate_adjudication(output_root, adjudication_path)
    ranking = list(adjudication["ranking"])
    eligible_rows = [
        row for row in ranking
        if row["candidate_id"] in REPAIRED_IDS
        and row.get("classification") in (STRICT, NEAR, ALTERNATE)
    ]
    if len(eligible_rows) > 2:
        raise RuntimeError("repaired follow-up parent count exceeds the frozen cap")

    parent_streams = []
    for row in eligible_rows:
        parent_id = str(row["candidate_id"])
        if parent_id not in WINNER_FAMILIES:
            raise RuntimeError("eligible repaired parent lacks ablation implementation")
        freeze_filename = f"REPAIRED_FOLLOWUP_FREEZE_{parent_id}.json"
        freeze = materialize_parent_ablation_definitions(
            output_root,
            parent_id=parent_id,
            authority_path=adjudication_path,
            authority_algorithm_fingerprint=str(row["algorithm_fingerprint"]),
            freeze_filename=freeze_filename,
            authority_kind="complete_repaired_frontier_e200_adjudication",
        )
        parent_streams.append({
            "parent_rank": int(row["rank"]),
            "parent_candidate_id": parent_id,
            "parent_classification": row["classification"],
            "parent_trajectory_status": row["trajectory_status"],
            "parent_algorithm_fingerprint": row["algorithm_fingerprint"],
            "ablation_candidate_ids": freeze["ablation_candidate_ids"],
            "execution_order_within_stream": [
                freeze["ablation_candidate_ids"]["proposal_only"],
                freeze["ablation_candidate_ids"]["observable_only"],
            ],
            "freeze_path": f"operations/{freeze_filename}",
            "freeze_sha256": file_sha256(output_root / "operations" / freeze_filename),
            "non_strict_requires_target_blind_defect_audit_before_revision": (
                row["classification"] != STRICT
            ),
        })

    result = {
        "schema": SCHEMA,
        "status": (
            "MULTI_PARENT_REPAIRED_ABLATIONS_FROZEN_FOR_GATES"
            if parent_streams else
            "NO_REPAIRED_PARENT_ELIGIBLE_FOR_ABLATION_LONG_RUN"
        ),
        "source_adjudication_path": adjudication_path.relative_to(
            output_root
        ).as_posix(),
        "source_adjudication_sha256": file_sha256(adjudication_path),
        "action_priority_candidate_id": adjudication[
            "action_priority_candidate_id"
        ],
        "action_priority_is_not_an_exclusivity_rule": True,
        "eligible_parent_streams": parent_streams,
        "eligible_parent_count": len(parent_streams),
        "maximum_parallel_parent_streams": 2,
        "within_parent_execution_is_sequential": True,
        "pcnr_duplicate_full_as_proposal_not_retrained_automatically": True,
        "pcnr_reason": (
            "PCNR proposal-only is its complete source-frozen transition; a second "
            "e200 copy is identity replication rather than a distinct mechanism."
        ),
        "if_no_eligible_repair": (
            "retain complete frontier and run target-blind defect audit before the "
            "single allowed mathematical revision; do not invent a paired window"
        ),
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "paired_metrics_used_only_after_complete_e200_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(output_path, result)
    return result
