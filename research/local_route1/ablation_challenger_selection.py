"""Posthoc frozen-seed selection for an unexpected ablation challenger."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from operations.local_route1_winner_ablation_adjudicate import (
    SCHEMA as ABLATION_SCHEMA,
)

from .candidates import validate_candidate_id
from .protocol import file_sha256
from .runtime import write_json
from .seed_validation import MULTI_SEED_ADJUDICATION_SCHEMA


SCHEMA = "final-unsb-route1-ablation-challenger-frozen-seed-selection-v1"
WORKSPACE_SCHEMA = "final-unsb-route1-ablation-challenger-seed-workspace-v1"
CHALLENGE_STATUS = "ABLATION_CHALLENGER_REQUIRES_FROZEN_SEED_VALIDATION"
COMPLETE_SEED_STATUSES = {"ROUTE1_SUSTAINED_LOCAL", "MULTI_SEED_NOT_SUSTAINED"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _posthoc(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not keep confirmation20 closed")
    for key in (
        "paired_controller_access",
        "paired_metrics_used_for_training_or_control",
        "paired_metric_changed_algorithm",
    ):
        if key in payload and payload[key] is not False:
            raise RuntimeError(f"{label} violates posthoc-only selection: {key}")


def _validate_multi(
    path: Path, *, candidate_id: str, algorithm_fingerprint: str,
) -> dict[str, Any]:
    value = _read_json(path)
    _posthoc(value, label=f"multi-seed {candidate_id}")
    if value.get("schema") != MULTI_SEED_ADJUDICATION_SCHEMA:
        raise RuntimeError("multi-seed adjudication schema mismatch")
    if value.get("status") not in COMPLETE_SEED_STATUSES:
        raise RuntimeError("multi-seed adjudication is incomplete")
    if value.get("candidate_id") != candidate_id:
        raise RuntimeError("multi-seed candidate identity mismatch")
    if value.get("algorithm_fingerprint") != algorithm_fingerprint:
        raise RuntimeError("multi-seed algorithm identity mismatch")
    if value.get("included_seeds") not in ([2026, 2027], [2026, 2027, 2028]):
        raise RuntimeError("multi-seed set violates the registered protocol")
    if value.get("algorithm_changes_after_seed2026_freeze") is not False:
        raise RuntimeError("algorithm changed after seed2026 freeze")
    for key in (
        "combined_late_three_mean_macro_psnr_delta",
        "combined_late_average_positive_domains",
        "combined_late_average_worst_domain_delta",
    ):
        number = value.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise RuntimeError(f"multi-seed selection field is nonfinite: {key}")
    return value


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row["status"] == "ROUTE1_SUSTAINED_LOCAL" else 1,
        -float(row["combined_late_three_mean_macro_psnr_delta"]),
        -float(row["combined_late_average_positive_domains"]),
        -float(row["combined_late_average_worst_domain_delta"]),
        str(row["candidate_id"]),
    )


def adjudicate_ablation_challenger_selection(
    output_root: Path, challenger_workspace: Path,
) -> dict[str, Any]:
    """Compare full and proposal identities only after both frozen seed runs."""
    output_root = Path(output_root).resolve()
    challenger_workspace = Path(challenger_workspace).resolve()
    ablation_path = output_root / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
    ablation = _read_json(ablation_path)
    _posthoc(ablation, label="winner ablation adjudication")
    if ablation.get("schema") != ABLATION_SCHEMA or ablation.get("status") != (
        CHALLENGE_STATUS
    ):
        raise RuntimeError("frozen-seed challenger selection requires a complete challenge")
    if (
        ablation.get("proposal_only_out_ranks_full") is not True
        or ablation.get("selection_change_blocked_pending_seed_validation") is not True
        or ablation.get("selection_changed") is not False
    ):
        raise RuntimeError("winner ablation selection was not held fail-closed")
    roles = ablation.get("roles")
    if not isinstance(roles, dict):
        raise RuntimeError("winner ablation roles are missing")
    full_role = roles.get("projected_or_full")
    challenger_role = roles.get("proposal_only")
    if not isinstance(full_role, dict) or not isinstance(challenger_role, dict):
        raise RuntimeError("winner ablation full/proposal role is missing")
    full_id = validate_candidate_id(str(full_role["candidate_id"]))
    challenger_id = validate_candidate_id(str(challenger_role["candidate_id"]))
    if full_id == challenger_id:
        raise RuntimeError("ablation challenger and full winner are identical")

    workspace_record_path = challenger_workspace / "CHALLENGER_SEED_WORKSPACE.json"
    workspace = _read_json(workspace_record_path)
    _posthoc(workspace, label="challenger seed workspace")
    if (
        workspace.get("schema") != WORKSPACE_SCHEMA
        or Path(workspace.get("source_root", "")).resolve() != output_root
        or Path(workspace.get("workspace_root", "")).resolve() != challenger_workspace
        or workspace.get("candidate_id") != challenger_id
        or workspace.get("full_winner_seed_namespace_reused") is not False
    ):
        raise RuntimeError("challenger seed workspace identity mismatch")

    full_path = output_root / "candidates" / full_id / "MULTI_SEED_ADJUDICATION.json"
    challenger_path = (
        challenger_workspace / "candidates" / challenger_id
        / "MULTI_SEED_ADJUDICATION.json"
    )
    full = _validate_multi(
        full_path, candidate_id=full_id,
        algorithm_fingerprint=str(full_role["algorithm_fingerprint"]),
    )
    challenger = _validate_multi(
        challenger_path, candidate_id=challenger_id,
        algorithm_fingerprint=str(challenger_role["algorithm_fingerprint"]),
    )
    ranking = sorted((full, challenger), key=_rank_key)
    selected = ranking[0]
    result = {
        "schema": SCHEMA,
        "status": (
            "CHALLENGER_SELECTED_AFTER_FROZEN_SEEDS"
            if selected["candidate_id"] == challenger_id else
            "FULL_WINNER_RETAINED_AFTER_CHALLENGER_SEEDS"
        ),
        "selected_candidate_id": selected["candidate_id"],
        "selected_algorithm_fingerprint": selected["algorithm_fingerprint"],
        "original_full_candidate_id": full_id,
        "challenger_candidate_id": challenger_id,
        "ranking": [
            {
                "rank": index,
                "candidate_id": row["candidate_id"],
                "algorithm_fingerprint": row["algorithm_fingerprint"],
                "multi_seed_status": row["status"],
                "included_seeds": row["included_seeds"],
                "combined_late_three_mean_macro_psnr_delta": row[
                    "combined_late_three_mean_macro_psnr_delta"
                ],
                "combined_late_average_positive_domains": row[
                    "combined_late_average_positive_domains"
                ],
                "combined_late_average_worst_domain_delta": row[
                    "combined_late_average_worst_domain_delta"
                ],
            }
            for index, row in enumerate(ranking, start=1)
        ],
        "ranking_policy": (
            "sustained-local status, combined late-three macro PSNR, positive "
            "domains, worst-domain delta, stable candidate id"
        ),
        "source_winner_ablation_adjudication_sha256": file_sha256(ablation_path),
        "source_full_multi_seed_sha256": file_sha256(full_path),
        "source_challenger_multi_seed_sha256": file_sha256(challenger_path),
        "source_challenger_workspace_sha256": file_sha256(workspace_record_path),
        "selection_changed_before_both_seed_adjudications": False,
        "paired_metrics_used_only_after_complete_frozen_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(
        output_root / "operations" / "ABLATION_CHALLENGER_SELECTION.json",
        result,
    )
    return result
