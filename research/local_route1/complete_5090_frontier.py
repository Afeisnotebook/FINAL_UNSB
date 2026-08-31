"""Adjudicate and export the complete same-host 5090 route-1 frontier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import (
    _rank_key,
    _validate_receipt,
)
from research.local_route1.cross_runtime_portfolio import RESULT_SCHEMA
from research.local_route1.extended_repaired_frontier import (
    SCHEMA as EXTENDED_SCHEMA,
)
from research.local_route1.frontier_advancement import (
    ALTERNATE,
    NEAR,
    STRICT,
    classify_complete_trajectory,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.portable_extended_frontier import (
    _canonical_json_sha256,
    _source_artifacts,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-complete-5090-frontier-adjudication-v1"
STATUS = "COMPLETE_5090_ROUTE1_FRONTIER_ACTION_PRIORITY_AVAILABLE"
PORTABLE_SCHEMA = "final-unsb-route1-portable-complete-5090-frontier-v1"
PORTABLE_STATUS = "PORTABLE_COMPLETE_5090_ROUTE1_FRONTIER_EVIDENCE"
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


def _posthoc(value: dict[str, Any], *, label: str) -> None:
    if value.get("paired_controller_access") is not False:
        raise RuntimeError(f"{label} used a paired controller")
    if value.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} opened confirmation20")
    if value.get("cross_host_deltas_merged") not in (None, False):
        raise RuntimeError(f"{label} merged cross-host deltas")


def _scientific_key(
    receipt: dict[str, Any], classification: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        0 if classification["classification"] == STRICT else 1,
        *_rank_key(receipt),
    )


def materialize_complete_5090_frontier(
    output_root: Path, *, output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    extended_path = operations / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json"
    cross_path = operations / "CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json"
    extended = _read_json(extended_path)
    cross = _read_json(cross_path)
    if extended.get("schema") != EXTENDED_SCHEMA:
        raise RuntimeError("complete 5090 source extended frontier schema changed")
    if cross.get("schema") != RESULT_SCHEMA or cross.get("status") != (
        "CROSS_RUNTIME_5090_PORTFOLIO_COMPLETE_E200"
    ):
        raise RuntimeError("complete 5090 cross-runtime portfolio is incomplete")
    _posthoc(extended, label="5090 extended frontier")
    _posthoc(cross, label="5090 cross-runtime portfolio")

    source_rows: list[tuple[Path, str]] = []
    for row in extended.get("ranking", []):
        path = Path(str(row.get("receipt_path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != row.get("receipt_sha256"):
            raise RuntimeError("complete 5090 extended receipt changed")
        source_rows.append((path, "5090_repaired_frontier"))
    for row in cross.get("candidate_results", []):
        path = Path(str(row.get("receipt_path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != row.get("receipt_sha256"):
            raise RuntimeError("complete 5090 cross-runtime receipt changed")
        source_rows.append((path, "4090_evidence_qualified_5090_replay"))
    by_id: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for path, role in source_rows:
        if not path.is_relative_to(output_root):
            raise RuntimeError("complete 5090 receipt escaped its run root")
        receipt = _validate_receipt(path)
        candidate_id = str(receipt["candidate_id"])
        previous = by_id.get(candidate_id)
        if previous is not None:
            if file_sha256(previous[1]) != file_sha256(path):
                raise RuntimeError("complete 5090 candidate has two receipts")
            continue
        by_id[candidate_id] = (receipt, path, role)
    if len(by_id) < 5:
        raise RuntimeError("complete 5090 frontier lacks the five expected candidates")
    authorities = {
        field: {str(receipt.get(field, "")) for receipt, _path, _role in by_id.values()}
        for field in BASELINE_FIELDS
    }
    if any(len(values) != 1 or not next(iter(values)) for values in authorities.values()):
        raise RuntimeError("complete 5090 frontier is not same-host/common-e0 matched")

    classified = []
    for receipt, path, role in by_id.values():
        trajectory_path = Path(str(receipt["trajectory_path"])).resolve()
        if (
            not trajectory_path.is_file()
            or not trajectory_path.is_relative_to(output_root)
            or file_sha256(trajectory_path) != receipt.get("trajectory_sha256")
        ):
            raise RuntimeError("complete 5090 candidate trajectory changed")
        trajectory = _read_json(trajectory_path)
        classification = classify_complete_trajectory(receipt, trajectory)
        classified.append((receipt, classification, path, role))
    ranked = sorted(classified, key=lambda item: _scientific_key(item[0], item[1]))
    ranking = []
    for rank, (receipt, classification, path, role) in enumerate(ranked, start=1):
        ranking.append({
            "rank": rank,
            "candidate_id": receipt["candidate_id"],
            "source_role": role,
            "classification": classification["classification"],
            "classification_checks": classification["checks"],
            "trajectory_status": receipt["trajectory_status"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "training_git_commit": receipt["training_git_commit"],
            "candidate_training_core_fingerprint": receipt[
                "candidate_training_core_fingerprint"
            ],
            "ranking_fields": receipt["ranking_fields"],
            "receipt_path": str(path),
            "receipt_sha256": file_sha256(path),
            "trajectory_path": receipt["trajectory_path"],
            "trajectory_sha256": receipt["trajectory_sha256"],
        })
    action_id = str(ranking[0]["candidate_id"])
    strict_ids = [row["candidate_id"] for row in ranking if row["classification"] == STRICT]
    preserved_ids = [
        row["candidate_id"] for row in ranking
        if row["classification"] in (STRICT, NEAR, ALTERNATE)
        or row["trajectory_status"] == POSITIVE_STATUS
    ]
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "same_host_authority": {
            field: next(iter(values)) for field, values in authorities.items()
        },
        "ranking_policy": (
            "strict sustained-e200 qualification first, then the frozen "
            "late-three/e200/domain/guardrail/cost key"
        ),
        "ranking": ranking,
        "action_priority_candidate_id": action_id,
        "priority_alternate_candidate_ids": [
            row["candidate_id"] for row in ranking if row["candidate_id"] != action_id
        ][:2],
        "strict_candidate_ids": strict_ids,
        "evidence_preserved_candidate_ids": preserved_ids,
        "parent_ablation_results": extended.get("parent_ablation_results", []),
        "observable_only_candidate_ids_excluded_from_ranking": extended.get(
            "observable_only_candidate_ids_excluded_from_ranking", []
        ),
        "source_extended_frontier_sha256": file_sha256(extended_path),
        "source_cross_runtime_result_sha256": file_sha256(cross_path),
        "cross_runtime_replay_results": cross,
        "rankable_complete_e200_candidate_count": len(ranking),
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    }
    output_path = (
        operations / "COMPLETE_5090_FRONTIER_ADJUDICATION.json"
        if output_path is None else Path(output_path).resolve()
    )
    if not output_path.is_relative_to(output_root):
        raise RuntimeError("complete 5090 frontier output escaped run root")
    write_json(output_path, result)
    return result


def validate_portable_complete_5090(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != PORTABLE_SCHEMA or value.get("status") != PORTABLE_STATUS:
        raise RuntimeError("portable complete 5090 schema/status mismatch")
    fixed = {
        "complete_source_e200_only": True,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise RuntimeError(f"portable complete 5090 changed: {key}")
    frontier = value.get("extended_adjudication")
    if (
        not isinstance(frontier, dict)
        or frontier.get("schema") != SCHEMA
        or frontier.get("status") != STATUS
        or _canonical_json_sha256(frontier) != value.get("source_frontier_sha256")
    ):
        raise RuntimeError("portable complete 5090 frontier changed")
    evidence = value.get("candidate_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(frontier.get("ranking", [])):
        raise RuntimeError("portable complete 5090 evidence count changed")
    by_id = {}
    for row in evidence:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in by_id:
            raise RuntimeError("portable complete 5090 candidate id changed")
        for sha_key, payload_key in (
            ("receipt_sha256", "receipt"),
            ("trajectory_sha256", "trajectory"),
            ("derivation_card_sha256", "derivation_card"),
            ("implementation_sha256", "implementation"),
        ):
            payload = row.get(payload_key)
            if (
                not isinstance(payload, dict)
                or _canonical_json_sha256(payload) != row.get(sha_key)
            ):
                raise RuntimeError("portable complete 5090 artifact changed")
        by_id[candidate_id] = row
    if {str(row["candidate_id"]) for row in frontier["ranking"]} != set(by_id):
        raise RuntimeError("portable complete 5090 ranking evidence changed")
    return value


def export_portable_complete_5090(
    output_root: Path, *, frontier_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    frontier_path = (
        operations / "COMPLETE_5090_FRONTIER_ADJUDICATION.json"
        if frontier_path is None else Path(frontier_path).resolve()
    )
    frontier = _read_json(frontier_path)
    if (
        frontier.get("schema") != SCHEMA
        or frontier.get("status") != STATUS
        or frontier.get("canonical_candidate_is_action_priority_only") is not True
        or frontier.get("algorithm_discovery_collapsed_to_single_candidate") is not False
    ):
        raise RuntimeError("complete 5090 frontier is not portable")
    evidence = []
    for row in frontier["ranking"]:
        path = Path(str(row["receipt_path"])).resolve()
        if file_sha256(path) != row.get("receipt_sha256"):
            raise RuntimeError("portable complete 5090 receipt changed")
        evidence.append(_source_artifacts(output_root, path))
    result = {
        "schema": PORTABLE_SCHEMA,
        "status": PORTABLE_STATUS,
        "source_frontier_sha256": file_sha256(frontier_path),
        # Compatibility name for the terminal delivery's host-separated view.
        "extended_adjudication": frontier,
        "candidate_evidence": evidence,
        "complete_source_e200_only": True,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    validate_portable_complete_5090(result)
    output_path = (
        operations / "PORTABLE_COMPLETE_5090_FRONTIER.json"
        if output_path is None else Path(output_path).resolve()
    )
    if not output_path.is_relative_to(output_root):
        raise RuntimeError("portable complete 5090 output escaped run root")
    write_json(output_path, result)
    return result
