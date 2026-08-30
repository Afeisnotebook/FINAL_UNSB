"""Resolve the authoritative seed-2026/e200 route-1 selection.

The first Generation-1 adjudication is intentionally immutable evidence.  A
Generation-2 causal revision, or an independently authorized candidate later
run on the same host/plain authority, must therefore be ranked in a new
source-bound artifact instead of overwriting that first adjudication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from operations.local_route1_cross_version_adjudicate import (
    SCHEMA as SELECTION_SCHEMA,
    adjudicate,
)
from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
    CROSS_VERSION_NEGATIVE_STATUS,
)
from research.local_route1.runtime import write_json


FINAL_SELECTION_NAME = "ROUTE1_FINAL_E200_SELECTION.json"
REVISION_SELECTION_NAME = "CROSS_VERSION_REVISION_E200_ADJUDICATION.json"
ORIGINAL_SELECTION_NAME = "CROSS_VERSION_E200_ADJUDICATION.json"
REVISION_NEED_NAME = "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json"
POSITIVE_STATUS = "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"
REVISION_REQUIRED = "REVISION_DERIVATION_REQUIRED"
ALLOWED_STATUSES = {POSITIVE_STATUS, CROSS_VERSION_NEGATIVE_STATUS}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_e200_selection(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = _read_json(path)
    if value.get("schema") != SELECTION_SCHEMA:
        raise RuntimeError(f"route-1 e200 selection schema mismatch: {path}")
    if value.get("status") not in ALLOWED_STATUSES:
        raise RuntimeError(f"route-1 e200 selection is not terminal: {path}")
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise RuntimeError("route-1 e200 selection has no ranking")
    ids = [str(row.get("candidate_id", "")) for row in ranking]
    if not all(ids) or len(set(ids)) != len(ids):
        raise RuntimeError("route-1 e200 selection candidate identities are invalid")
    selected = str(value.get("selected_candidate_id", ""))
    selected_rows = [row for row in ranking if row.get("candidate_id") == selected]
    if len(selected_rows) != 1:
        raise RuntimeError("route-1 e200 selection winner is absent or duplicated")
    selected_row = selected_rows[0]
    for selection_key, row_key in (
        ("selected_algorithm_fingerprint", "algorithm_fingerprint"),
        ("selected_candidate_fingerprint", "candidate_fingerprint"),
        ("selected_training_git_commit", "training_git_commit"),
    ):
        if value.get(selection_key) != selected_row.get(row_key):
            raise RuntimeError(f"route-1 e200 selected identity mismatch: {selection_key}")
    for key in (
        "paired_metrics_used_for_training_or_control",
        "confirmation20_opened",
    ):
        if value.get(key) is not False:
            raise RuntimeError(f"route-1 e200 selection requires {key}=false")
    return value


def resolve_e200_selection_path(output_root: Path) -> Path:
    """Return the newest scientifically terminal source without racing revision.

    If the first ranking explicitly authorized the one allowed mathematical
    revision, the negative Generation-1 ranking is not a final winner source.
    Consumers must wait for the revision adjudication instead of silently
    freezing the pre-revision fallback.
    """
    operations = Path(output_root).resolve() / "operations"
    final_path = operations / FINAL_SELECTION_NAME
    if final_path.is_file():
        validate_e200_selection(final_path)
        return final_path

    revision_path = operations / REVISION_SELECTION_NAME
    if revision_path.is_file():
        validate_e200_selection(revision_path)
        return revision_path

    revision_need_path = operations / REVISION_NEED_NAME
    if revision_need_path.is_file():
        need = _read_json(revision_need_path)
        if (
            need.get("schema") == CROSS_VERSION_FINAL_OUTCOME_SCHEMA
            and need.get("status") == REVISION_REQUIRED
        ):
            raise RuntimeError(
                "route-1 e200 selection is pending the authorized Generation-2 revision"
            )

    original_path = operations / ORIGINAL_SELECTION_NAME
    if not original_path.is_file():
        raise RuntimeError("route-1 e200 selection is missing")
    validate_e200_selection(original_path)
    return original_path


def materialize_final_e200_selection(
    output_root: Path, receipt_paths: Iterable[Path],
) -> dict[str, Any]:
    """Rank every authorized same-host receipt without mutating prior evidence."""
    output_root = Path(output_root).resolve()
    path = output_root / "operations" / FINAL_SELECTION_NAME
    result = adjudicate(receipt_paths, path)
    result.update({
        "selection_scope": "all_authoritative_same_host_seed2026_e200_candidates",
        "historical_adjudications_overwritten": False,
        "additional_seed_replication_deferred": [2027, 2028],
        "cross_seed_stability_claimed": False,
    })
    write_json(path, result)
    return validate_e200_selection(path)

