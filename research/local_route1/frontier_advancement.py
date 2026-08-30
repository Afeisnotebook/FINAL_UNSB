"""Classify complete e200 trajectories for evidence-qualified advancement.

This module does not construct or tune an algorithm.  Paired trajectory
metrics are read only after a complete terminal receipt and determine whether
a mechanism remains in the frontier.  A mathematical revision still requires
a separate target-blind defect audit and an unused revision allowance.
"""

from __future__ import annotations

from typing import Any

from research.local_route1.generation1_adjudication import POSITIVE_STATUS


STRICT = "strict_sustained"
NEAR = "causally_repairable_near_boundary_pending_target_blind_audit"
ALTERNATE = "evidence_backed_alternate"
CLOSED = "closed_current_operator"


def classify_complete_trajectory(
    receipt: dict[str, Any], trajectory: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("candidate_id") != trajectory.get("candidate_id"):
        raise RuntimeError("frontier receipt and trajectory identities differ")
    if receipt.get("trajectory_status") != trajectory.get("status"):
        raise RuntimeError("frontier receipt and trajectory statuses differ")
    fields = receipt.get("ranking_fields", {})
    late = float(fields["late_three_mean_macro_psnr_delta"])
    endpoint = float(fields["e200_macro_psnr_delta"])
    coverage = int(fields["late_points_with_four_of_six_positive_domains"])
    worst = float(fields["late_average_worst_domain_delta"])
    ssim = float(fields["late_mean_macro_ssim_delta"])
    lpips = float(fields["late_mean_macro_lpips_delta"])
    drawdown = float(fields["candidate_best_to_terminal_three_point_rolling_drawdown"])
    collapse = trajectory.get("plain_collapse_adjudication", {}).get("status")
    checks = {
        "late_three_positive": late > 0.0,
        "e200_positive": endpoint > 0.0,
        "two_late_points_four_of_six_positive": coverage >= 2,
        "average_worst_domain_above_minus_one_db": worst > -1.0,
        "late_ssim_nonnegative": ssim >= 0.0,
        "late_lpips_nonpositive": lpips <= 0.0,
        "rolling_drawdown_at_most_point_three_db": drawdown <= 0.3,
        "not_plain_collapse": collapse == "PASS_NOT_PLAIN_COLLAPSE",
    }
    strict = receipt.get("trajectory_status") == POSITIVE_STATUS and all(checks.values())
    guardrail_names = (
        "e200_positive",
        "two_late_points_four_of_six_positive",
        "average_worst_domain_above_minus_one_db",
        "late_ssim_nonnegative",
        "late_lpips_nonpositive",
        "rolling_drawdown_at_most_point_three_db",
    )
    failures = [name for name in guardrail_names if not checks[name]]
    near = bool(
        not strict
        and late > 0.0
        and endpoint >= -0.1
        and checks["not_plain_collapse"]
        and len(failures) <= 1
    )
    if strict:
        classification = STRICT
    elif near:
        classification = NEAR
    elif late > 0.0 or endpoint > 0.0:
        classification = ALTERNATE
    else:
        classification = CLOSED
    return {
        "candidate_id": receipt["candidate_id"],
        "classification": classification,
        "checks": checks,
        "failed_numeric_or_guardrail_checks": failures,
        "near_boundary_endpoint_floor_db": -0.1,
        "target_blind_revision_audit_required": classification == NEAR,
        "mathematical_revision_authorized": False,
        "paired_metrics_used_only_after_complete_e200": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "confirmation20_opened": False,
    }


def classify_frontier(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    classified = [
        classify_complete_trajectory(receipt, trajectory)
        for receipt, trajectory in rows
    ]
    ids = [row["candidate_id"] for row in classified]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frontier advancement candidate identity is duplicated")
    return {
        "schema": "final-unsb-route1-frontier-advancement-classification-v1",
        "status": "COMPLETE_E200_FRONTIER_CLASSIFIED_FOR_SECOND_WAVE",
        "candidates": classified,
        "strict_candidate_ids": [
            row["candidate_id"] for row in classified
            if row["classification"] == STRICT
        ],
        "near_boundary_pending_target_blind_audit_ids": [
            row["candidate_id"] for row in classified
            if row["classification"] == NEAR
        ],
        "second_wave_maximum_parallel_e200_trajectories": 2,
        "second_wave_formula_requires_target_blind_evidence": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "confirmation20_opened": False,
    }

