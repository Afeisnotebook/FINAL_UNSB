"""Fail-closed single-seed development freeze for the emergency route-1 search.

This policy does not assert cross-seed stability.  It records the user's
explicit decision to use the complete seed-2026 small25/e200 result for
development selection and to defer additional seeds so the saved compute can
be spent on mechanism ablations and evidence-driven mathematical revisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import (
    SCHEMA as CROSS_SCHEMA,
    _validate_receipt,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-single-seed-development-freeze-v1"
STATUS = "FROZEN_SINGLE_SEED_DEVELOPMENT_CANDIDATE"
POSITIVE_CROSS_STATUS = "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"
POLICY = "DEFER_ADDITIONAL_SEEDS_FOR_ALGORITHM_SEARCH"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def freeze_path(output_root: Path) -> Path:
    return Path(output_root).resolve() / "operations" / "SINGLE_SEED_DEVELOPMENT_FREEZE.json"


def validate_single_seed_development_freeze(
    output_root: Path, value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    path = freeze_path(output_root)
    if value is None:
        if not path.is_file():
            raise RuntimeError("single-seed development freeze is missing")
        value = _read_json(path)
    cross_path = output_root / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    if not cross_path.is_file():
        raise RuntimeError("single-seed freeze has no cross-version e200 authority")
    cross = _read_json(cross_path)
    candidate_id = str(value.get("candidate_id", ""))
    receipt_path = (
        output_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    )
    receipt = _validate_receipt(receipt_path)
    required = {
        "schema": SCHEMA,
        "status": STATUS,
        "policy": POLICY,
        "candidate_id": cross.get("selected_candidate_id"),
        "algorithm_fingerprint": cross.get("selected_algorithm_fingerprint"),
        "candidate_fingerprint": cross.get("selected_candidate_fingerprint"),
        "training_git_commit": cross.get("selected_training_git_commit"),
        "source_cross_version_adjudication_sha256": file_sha256(cross_path),
        "source_terminal_receipt_sha256": file_sha256(receipt_path),
        "included_seeds": [2026],
        "deferred_seeds": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metric_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if cross.get("schema") != CROSS_SCHEMA or cross.get("status") != POSITIVE_CROSS_STATUS:
        raise RuntimeError("single-seed development freeze requires a positive e200 winner")
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(f"single-seed development freeze field mismatch: {key}")
    if (
        receipt.get("candidate_id") != candidate_id
        or receipt.get("algorithm_fingerprint") != value["algorithm_fingerprint"]
        or receipt.get("candidate_fingerprint") != value["candidate_fingerprint"]
        or receipt.get("training_git_commit") != value["training_git_commit"]
    ):
        raise RuntimeError("single-seed development freeze source identity changed")
    return value


def materialize_single_seed_development_freeze(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    cross_path = output_root / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    if not cross_path.is_file():
        raise RuntimeError("single-seed development freeze requires e200 adjudication")
    cross = _read_json(cross_path)
    if cross.get("schema") != CROSS_SCHEMA or cross.get("status") != POSITIVE_CROSS_STATUS:
        raise RuntimeError("single-seed development freeze requires a positive e200 winner")
    candidate_id = str(cross["selected_candidate_id"])
    receipt_path = (
        output_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    )
    receipt = _validate_receipt(receipt_path)
    value = {
        "schema": SCHEMA,
        "status": STATUS,
        "policy": POLICY,
        "decision": "decisions/DEC-20260830-ROUTE1-SINGLE-SEED-EMERGENCY.md",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": str(cross["selected_algorithm_fingerprint"]),
        "candidate_fingerprint": str(cross["selected_candidate_fingerprint"]),
        "training_git_commit": str(cross["selected_training_git_commit"]),
        "source_cross_version_adjudication_sha256": file_sha256(cross_path),
        "source_terminal_receipt_sha256": file_sha256(receipt_path),
        "included_seeds": [2026],
        "deferred_seeds": [2027, 2028],
        "evidence_scope": "small25_batch1_seed2026_true_e200",
        "allowed_use": [
            "development candidate selection",
            "proposal-only/observable-only/full mechanism ablation",
            "evidence-driven mathematical revision routing",
        ],
        "forbidden_claims": [
            "cross-seed stability has been demonstrated",
            "full-data or confirmation20 performance is established",
        ],
        "compute_reallocation_priority": [
            "winner mechanism ablations",
            "one evidence-authorized mathematical revision if needed",
            "an additional independent mechanism before repeated initialization",
        ],
        "cross_seed_stability_claimed": False,
        "paired_metric_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if (
        receipt["candidate_id"] != candidate_id
        or receipt["algorithm_fingerprint"] != value["algorithm_fingerprint"]
        or receipt["candidate_fingerprint"] != value["candidate_fingerprint"]
        or receipt["training_git_commit"] != value["training_git_commit"]
    ):
        raise RuntimeError("single-seed freeze receipt/cross identity mismatch")
    path = freeze_path(output_root)
    if path.is_file() and _read_json(path) != value:
        raise RuntimeError("single-seed development freeze already differs")
    write_json(path, value)
    return validate_single_seed_development_freeze(output_root, value)
