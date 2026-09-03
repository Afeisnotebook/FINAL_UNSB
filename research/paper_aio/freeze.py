"""Two-stage human/Codex review boundary for the paper evidence freeze."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from research.local_route1.runtime import write_json

from .distribution import FREEZE_SCHEMA, FREEZE_STATUS
from .protocol import (
    EXPECTED_MANIFEST_SHA256,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    ROOT,
    file_sha256,
    git_commit,
    object_sha256,
)


DRAFT_SCHEMA = "final-unsb-paper-freeze-review-draft-v1"
DRAFT_STATUS = "PENDING_EXPLICIT_HUMAN_CODEX_REVIEW"
REVIEW_SCHEMA = "final-unsb-paper-freeze-review-decision-v1"
REVIEW_STATUS = "APPROVE_FULL_DATA_ALGORITHM_BASELINE_AND_CLAIM_FREEZE"
PORTFOLIO_SCHEMAS = {
    "final-unsb-paper-full-data-algorithm-portfolio-v1",
    "final-unsb-paper-full-data-algorithm-portfolio-v2",
}
PORTFOLIO_STATUSES = {
    "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_AWAITING_CONFIRMATION_DECISION",
    "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_WITH_DCLGAN_AWAITING_CONFIRMATION_DECISION",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _committed_json(path: Path) -> tuple[dict[str, Any], str, str]:
    path = Path(path).resolve()
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError("freeze review decision must be inside the repository") from error
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("freeze review decision has uncommitted changes")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT, text=True,
    ).strip()
    if len(commit) != 40:
        raise RuntimeError("freeze review decision has no committed Git identity")
    text = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT, text=True,
    )
    committed = json.loads(text)
    current = _read(path)
    if object_sha256(current) != object_sha256(committed):
        raise RuntimeError("working freeze review differs from its committed Git blob")
    return current, commit, relative


def validate_portfolio(path: Path) -> tuple[dict[str, Any], list[str]]:
    value = _read(path)
    if (
        value.get("schema") not in PORTFOLIO_SCHEMAS
        or value.get("status") not in PORTFOLIO_STATUSES
        or int(value.get("primary_epoch", -1)) != 200
        or value.get("paper_claims_frozen") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("cross_non_equivalent_runtime_delta") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("paper portfolio is not eligible for freeze review")
    lanes = {str((value.get("plain_control") or {}).get("lane_id", ""))}
    for section in ("methods", "external_baselines"):
        for row in (value.get(section) or {}).values():
            result = row.get("result") if section == "methods" else row
            lane = str((result or {}).get("lane_id", ""))
            if lane:
                lanes.add(lane)
    lanes.discard("")
    if "plain" not in lanes or "input" not in lanes or len(lanes) < 5:
        raise RuntimeError("paper portfolio has an incomplete frozen lane set")
    return value, sorted(lanes)


def create_review_draft(
    *, portfolio: Path, claims: list[str], destination: Path,
) -> dict[str, Any]:
    portfolio = Path(portfolio).resolve()
    value, lanes = validate_portfolio(portfolio)
    cleaned_claims = [str(claim).strip() for claim in claims]
    if (
        not cleaned_claims
        or any(not claim for claim in cleaned_claims)
        or len(cleaned_claims) != len(set(cleaned_claims))
    ):
        raise ValueError("freeze review requires a nonempty unique explicit claim set")
    result = {
        "schema": DRAFT_SCHEMA,
        "status": DRAFT_STATUS,
        "source_portfolio_path": str(portfolio),
        "source_portfolio_sha256": file_sha256(portfolio),
        "source_portfolio_schema": value["schema"],
        "source_portfolio_status": value["status"],
        "primary_epoch": 200,
        "distribution_lanes": lanes,
        "paper_claims": cleaned_claims,
        "paper_claims_sha256": object_sha256(cleaned_claims),
        "review_requirements": [
            "confirm every named algorithm and baseline configuration is final",
            "confirm every paper claim is supported by fixed e200 and sustained evidence",
            "confirm failures, deferrals and reproduction-incomplete baselines are labeled honestly",
            "confirm no result was selected by best checkpoint or confirmation20",
            "author an explicit committed freeze review decision; this draft cannot self-approve",
        ],
        "human_approval_recorded": False,
        "algorithm_configuration_frozen": False,
        "baseline_configuration_frozen": False,
        "paper_claims_frozen": False,
        "e200_results_frozen": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    write_json(Path(destination).resolve(), result)
    return result


def materialize_freeze_receipt(
    *, portfolio: Path, review_decision: Path, destination: Path,
) -> dict[str, Any]:
    portfolio = Path(portfolio).resolve()
    _, lanes = validate_portfolio(portfolio)
    review, review_commit, review_relative = _committed_json(review_decision)
    claims = review.get("paper_claims")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != REVIEW_STATUS
        or review.get("source_portfolio_path") != str(portfolio)
        or review.get("source_portfolio_sha256") != file_sha256(portfolio)
        or review.get("distribution_lanes") != lanes
        or not isinstance(claims, list)
        or not claims
        or len(claims) != len(set(claims))
        or review.get("paper_claims_sha256") != object_sha256(claims)
        or review.get("human_approval_recorded") is not True
        or review.get("codex_scientific_review_recorded") is not True
        or review.get("best_checkpoint_selection") is not False
        or review.get("paired_metric_control") is not False
        or review.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("committed freeze review decision is invalid")
    destination = Path(destination).resolve()
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("freeze receipt must be materialized inside the repository") from error
    result = {
        "schema": FREEZE_SCHEMA,
        "status": FREEZE_STATUS,
        "algorithm_configuration_frozen": True,
        "baseline_configuration_frozen": True,
        "paper_claims_frozen": True,
        "e200_results_frozen": True,
        "primary_epoch": 200,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "source_portfolio_path": str(portfolio),
        "source_portfolio_sha256": file_sha256(portfolio),
        "distribution_lanes": lanes,
        "paper_claims": claims,
        "paper_claims_sha256": object_sha256(claims),
        "review_decision": review_relative,
        "review_decision_sha256": file_sha256(review_decision),
        "review_decision_git_commit": review_commit,
        "materializer_git_commit": git_commit(),
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    write_json(destination, result)
    return result
