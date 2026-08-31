"""Portable evidence bundle for the complete 5090 repaired frontier.

The bundle carries compact JSON evidence only.  It never transfers a model,
optimizer, checkpoint, or cross-host delta.  Every embedded receipt,
trajectory, derivation card, and implementation remains bound to its original
5090 file hash and complete-e200 adjudication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import _validate_receipt
from research.local_route1.extended_repaired_frontier import SCHEMA as EXTENDED_SCHEMA
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-portable-extended-repaired-frontier-v1"
STATUS = "PORTABLE_COMPLETE_5090_REPAIRED_FRONTIER_EVIDENCE"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bound_json(output_root: Path, path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(output_root):
        raise RuntimeError(f"portable extended evidence escaped run root: {path}")
    actual = file_sha256(path)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"portable extended evidence changed: {path}")
    value = _read_json(path)
    if _canonical_json_sha256(value) != actual:
        raise RuntimeError(f"portable extended evidence is not canonical JSON: {path}")
    return value


def _source_artifacts(output_root: Path, receipt_path: Path) -> dict[str, Any]:
    receipt_path = Path(receipt_path).resolve()
    receipt = _validate_receipt(receipt_path)
    if not receipt_path.is_relative_to(output_root):
        raise RuntimeError("portable receipt escaped source run root")
    candidate_id = str(receipt["candidate_id"])
    trajectory_path = Path(str(receipt["trajectory_path"])).resolve()
    trajectory = _bound_json(
        output_root, trajectory_path, str(receipt["trajectory_sha256"]),
    )
    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    card = _bound_json(
        output_root, card_path, str(receipt["derivation_card_sha256"]),
    )
    implementation = _bound_json(
        output_root, implementation_path, str(receipt["implementation_sha256"]),
    )
    if (
        trajectory.get("candidate_id") != candidate_id
        or card.get("candidate_id") != candidate_id
        or implementation.get("candidate_id") != candidate_id
    ):
        raise RuntimeError("portable extended candidate artifact identities differ")
    return {
        "candidate_id": candidate_id,
        "receipt": receipt,
        "receipt_sha256": file_sha256(receipt_path),
        "trajectory": trajectory,
        "trajectory_sha256": file_sha256(trajectory_path),
        "derivation_card": card,
        "derivation_card_sha256": file_sha256(card_path),
        "implementation": implementation,
        "implementation_sha256": file_sha256(implementation_path),
    }


def validate_portable_extended_frontier(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA or value.get("status") != STATUS:
        raise RuntimeError("portable extended frontier schema/status mismatch")
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
            raise RuntimeError(f"portable extended frontier changed: {key}")
    adjudication = value.get("extended_adjudication")
    if not isinstance(adjudication, dict) or adjudication.get("schema") != EXTENDED_SCHEMA:
        raise RuntimeError("portable extended frontier lacks source adjudication")
    if _canonical_json_sha256(adjudication) != value.get("source_adjudication_sha256"):
        raise RuntimeError("portable extended adjudication hash changed")
    evidence = value.get("candidate_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("portable extended frontier lacks candidate evidence")
    by_id = {}
    for row in evidence:
        if not isinstance(row, dict):
            raise RuntimeError("portable extended candidate evidence is malformed")
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in by_id:
            raise RuntimeError("portable extended candidate id is empty or duplicated")
        for key, payload_key in (
            ("receipt_sha256", "receipt"),
            ("trajectory_sha256", "trajectory"),
            ("derivation_card_sha256", "derivation_card"),
            ("implementation_sha256", "implementation"),
        ):
            payload = row.get(payload_key)
            if not isinstance(payload, dict) or _canonical_json_sha256(payload) != row.get(key):
                raise RuntimeError(f"portable extended embedded artifact changed: {candidate_id}:{payload_key}")
            if payload.get("candidate_id") != candidate_id:
                raise RuntimeError("portable extended embedded candidate identity changed")
        by_id[candidate_id] = row
    ranking_ids = {
        str(row.get("candidate_id", ""))
        for row in adjudication.get("ranking", []) if isinstance(row, dict)
    }
    if not ranking_ids or not ranking_ids.issubset(by_id):
        raise RuntimeError("portable extended ranking evidence is incomplete")
    role_ids = set()
    for parent in adjudication.get("parent_ablation_results", []):
        for role in (parent.get("roles") or {}).values():
            if isinstance(role, dict):
                role_ids.add(str(role.get("candidate_id", "")))
    if not role_ids.issubset(by_id):
        raise RuntimeError("portable extended ablation role evidence is incomplete")
    return value


def export_portable_extended_frontier(
    output_root: Path, *, adjudication_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    adjudication_path = (
        operations / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        if adjudication_path is None else Path(adjudication_path).resolve()
    )
    adjudication = _bound_json(output_root, adjudication_path)
    if (
        adjudication.get("schema") != EXTENDED_SCHEMA
        or adjudication.get("canonical_candidate_is_action_priority_only") is not True
        or adjudication.get("algorithm_discovery_collapsed_to_single_candidate") is not False
        or adjudication.get("cross_host_deltas_merged") is not False
        or adjudication.get("paired_controller_access") is not False
        or adjudication.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("extended repaired frontier is not portable")

    paths: dict[str, Path] = {}
    for row in adjudication.get("ranking", []):
        candidate_id = str(row.get("candidate_id", ""))
        path = Path(str(row.get("receipt_path", ""))).resolve()
        if not candidate_id or file_sha256(path) != row.get("receipt_sha256"):
            raise RuntimeError("extended repaired ranking receipt changed")
        paths[candidate_id] = path
    for parent in adjudication.get("parent_ablation_results", []):
        for role in (parent.get("roles") or {}).values():
            if not isinstance(role, dict):
                continue
            candidate_id = str(role.get("candidate_id", ""))
            path = Path(str(role.get("receipt_path", ""))).resolve()
            if not candidate_id or file_sha256(path) != role.get("receipt_sha256"):
                raise RuntimeError("extended repaired ablation receipt changed")
            paths[candidate_id] = path
    evidence = [_source_artifacts(output_root, path) for _id, path in sorted(paths.items())]
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_adjudication_sha256": file_sha256(adjudication_path),
        "extended_adjudication": adjudication,
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
    validate_portable_extended_frontier(result)
    output_path = (
        operations / "PORTABLE_EXTENDED_REPAIRED_FRONTIER.json"
        if output_path is None else Path(output_path).resolve()
    )
    if not output_path.is_relative_to(output_root):
        raise RuntimeError("portable extended frontier output escaped run root")
    write_json(output_path, result)
    return result
