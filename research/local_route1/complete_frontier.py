"""Adjudicate the complete same-host 4090 route-1 candidate frontier.

The pre-frontier delivery remains an immutable evidence source, but it is not
allowed to hide later repaired or Generation-3 candidates.  Only complete,
source-bound e200 receipts produced against the same 4090 plain authority are
ranked.  A canonical id is an action priority; every complete row is retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import (
    _rank_key,
    _validate_receipt,
)
from research.local_route1.frontier_advancement import (
    ALTERNATE,
    NEAR,
    STRICT,
    classify_complete_trajectory,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-complete-4090-frontier-adjudication-v1"
STATUS = "COMPLETE_4090_ROUTE1_FRONTIER_ACTION_PRIORITY_AVAILABLE"
REPAIRED_RESULT_SCHEMA = "final-unsb-route1-repaired-portfolio-4090-result-v1"
ADAM_SYNTHESIS_SCHEMA = "final-unsb-route1-residual-synthesis-4090-result-v1"
EUCLIDEAN_SYNTHESIS_SCHEMA = (
    "final-unsb-route1-residual-euclidean-synthesis-4090-result-v1"
)
BASELINE_FIELDS = (
    "base_e0_scientific_state_sha256",
    "base_protocol_fingerprint",
    "manifest_sha256",
    "plain_e200_verification_sha256",
)
RESULT_FILES = (
    "REPAIRED_PORTFOLIO_4090_RESULT.json",
    "RESIDUAL_SYNTHESIS_4090_RESULT.json",
    "RESIDUAL_EUCLIDEAN_SYNTHESIS_4090_RESULT.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _posthoc_boundary(value: dict[str, Any], *, label: str) -> None:
    if value.get("paired_controller_access") not in (None, False):
        raise RuntimeError(f"{label} used a paired controller")
    if value.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not prove confirmation20 remained closed")
    if value.get("cross_host_deltas_merged") not in (None, False):
        raise RuntimeError(f"{label} merged cross-host deltas")


def _bound_receipt(output_root: Path, path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(output_root):
        raise RuntimeError(f"complete-frontier receipt escaped run root: {path}")
    return _validate_receipt(path)


def _receipt_for_id(output_root: Path, candidate_id: str) -> Path:
    terminal = output_root / "operations" / "terminal_receipts"
    matches = [
        path for path in (
            terminal / f"{candidate_id}.json",
            terminal / f"{candidate_id}_4090.json",
        )
        if path.is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one terminal receipt for {candidate_id}, found {len(matches)}"
        )
    return matches[0].resolve()


def _pre_frontier_receipts(output_root: Path) -> list[tuple[Path, str]]:
    final = output_root / "final"
    candidate_path = final / "CANDIDATE.json"
    results_path = final / "RESULTS.json"
    if not candidate_path.is_file() or not results_path.is_file():
        raise RuntimeError("pre-frontier final delivery is incomplete")
    candidate = _read_json(candidate_path)
    results = _read_json(results_path)
    for label, value in (("pre-frontier candidate", candidate), ("pre-frontier results", results)):
        _posthoc_boundary(value, label=label)
    selected_id = str(candidate.get("candidate_id", ""))
    if not selected_id or results.get("selected_candidate_id") != selected_id:
        raise RuntimeError("pre-frontier selected candidate identity changed")
    ids = [selected_id]
    for row in results.get("ranking", []):
        if isinstance(row, dict):
            candidate_id = str(row.get("candidate_id", ""))
            if candidate_id and candidate_id not in ids:
                ids.append(candidate_id)
    return [(_receipt_for_id(output_root, candidate_id), "pre_frontier_4090") for candidate_id in ids]


def _result_receipts(
    output_root: Path,
) -> tuple[list[tuple[Path, str]], dict[str, dict[str, Any]]]:
    operations = output_root / "operations"
    values = {name: _read_json(operations / name) for name in RESULT_FILES}
    for name, value in values.items():
        _posthoc_boundary(value, label=name)
    if values[RESULT_FILES[0]].get("schema") != REPAIRED_RESULT_SCHEMA:
        raise RuntimeError("repaired 4090 portfolio result schema changed")
    if values[RESULT_FILES[1]].get("schema") != ADAM_SYNTHESIS_SCHEMA:
        raise RuntimeError("Adam synthesis result schema changed")
    if values[RESULT_FILES[2]].get("schema") != EUCLIDEAN_SYNTHESIS_SCHEMA:
        raise RuntimeError("Euclidean synthesis result schema changed")

    rows: list[tuple[Path, str]] = []
    repaired = values[RESULT_FILES[0]]
    candidate_results = repaired.get("candidate_results")
    if not isinstance(candidate_results, list):
        raise RuntimeError("repaired 4090 portfolio lacks candidate_results")
    for row in candidate_results:
        if not isinstance(row, dict):
            raise RuntimeError("repaired 4090 portfolio candidate row is malformed")
        path = Path(str(row.get("receipt_path", ""))).resolve()
        if file_sha256(path) != row.get("receipt_sha256"):
            raise RuntimeError("repaired 4090 portfolio receipt changed")
        rows.append((path, "repaired_4090_replay"))

    for name, role in (
        (RESULT_FILES[1], "generation3_adam_geometry"),
        (RESULT_FILES[2], "generation3_euclidean_geometry"),
    ):
        value = values[name]
        candidate_id = value.get("candidate_id")
        if candidate_id is None:
            continue
        path = Path(str(value.get("receipt_path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != value.get("receipt_sha256"):
            raise RuntimeError(f"{name} receipt changed")
        rows.append((path, role))
    return rows, values


def _scientific_key(
    receipt: dict[str, Any], classification: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        0 if classification["classification"] == STRICT else 1,
        *_rank_key(receipt),
    )


def materialize_complete_4090_frontier(
    output_root: Path, *, output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    source_rows = _pre_frontier_receipts(output_root)
    later_rows, terminal_results = _result_receipts(output_root)
    source_rows.extend(later_rows)

    by_id: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for path, role in source_rows:
        receipt = _bound_receipt(output_root, path)
        candidate_id = str(receipt["candidate_id"])
        previous = by_id.get(candidate_id)
        if previous is not None:
            if file_sha256(previous[1]) != file_sha256(path):
                raise RuntimeError(f"candidate has two different receipts: {candidate_id}")
            continue
        by_id[candidate_id] = (receipt, path, role)
    if not by_id:
        raise RuntimeError("complete 4090 frontier has no candidate receipts")

    authorities = {
        field: {str(receipt.get(field, "")) for receipt, _path, _role in by_id.values()}
        for field in BASELINE_FIELDS
    }
    if any(len(values) != 1 or not next(iter(values)) for values in authorities.values()):
        raise RuntimeError("complete 4090 frontier is not same-host/common-e0 matched")

    classified = []
    for receipt, path, role in by_id.values():
        trajectory_path = Path(str(receipt["trajectory_path"])).resolve()
        if (
            not trajectory_path.is_file()
            or not trajectory_path.is_relative_to(output_root)
            or file_sha256(trajectory_path) != receipt.get("trajectory_sha256")
        ):
            raise RuntimeError(f"candidate trajectory changed: {receipt['candidate_id']}")
        trajectory = _read_json(trajectory_path)
        classification = classify_complete_trajectory(receipt, trajectory)
        classified.append((receipt, trajectory, classification, path, role))
    ranked = sorted(classified, key=lambda row: _scientific_key(row[0], row[2]))
    ranking = []
    for rank, (receipt, _trajectory, classification, path, role) in enumerate(ranked, start=1):
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
        "pre_frontier_candidate_retained_in_ranking": True,
        "repaired_parent_results": terminal_results[RESULT_FILES[0]],
        "generation3_results": {
            "adam_geometry": terminal_results[RESULT_FILES[1]],
            "euclidean_geometry": terminal_results[RESULT_FILES[2]],
        },
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
        operations / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
        if output_path is None else Path(output_path).resolve()
    )
    if not output_path.is_relative_to(output_root):
        raise RuntimeError("complete 4090 frontier output escaped run root")
    write_json(output_path, result)
    return result
