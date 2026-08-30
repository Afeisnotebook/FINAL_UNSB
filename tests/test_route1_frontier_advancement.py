from __future__ import annotations

from research.local_route1.frontier_advancement import (
    ALTERNATE,
    CLOSED,
    NEAR,
    STRICT,
    classify_complete_trajectory,
    classify_frontier,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)


def _pair(candidate_id: str, *, status: str, **overrides):
    fields = {
        "late_three_mean_macro_psnr_delta": 0.2,
        "e200_macro_psnr_delta": 0.1,
        "late_points_with_four_of_six_positive_domains": 2,
        "late_average_worst_domain_delta": -0.2,
        "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
    }
    fields.update(overrides)
    receipt = {
        "candidate_id": candidate_id,
        "trajectory_status": status,
        "ranking_fields": fields,
    }
    trajectory = {
        "candidate_id": candidate_id,
        "status": status,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
    }
    return receipt, trajectory


def test_strict_trajectory_remains_a_competing_frontier_candidate():
    receipt, trajectory = _pair("strict", status=POSITIVE_STATUS)
    result = classify_complete_trajectory(receipt, trajectory)
    assert result["classification"] == STRICT
    assert result["mathematical_revision_authorized"] is False


def test_near_boundary_requires_a_separate_target_blind_defect_audit():
    receipt, trajectory = _pair(
        "near", status=NEGATIVE_STATUS, e200_macro_psnr_delta=-0.05,
    )
    result = classify_complete_trajectory(receipt, trajectory)
    assert result["classification"] == NEAR
    assert result["failed_numeric_or_guardrail_checks"] == ["e200_positive"]
    assert result["target_blind_revision_audit_required"] is True
    assert result["mathematical_revision_authorized"] is False


def test_two_failures_do_not_consume_a_revision_just_to_fill_the_gpu():
    receipt, trajectory = _pair(
        "alternate", status=NEGATIVE_STATUS,
        e200_macro_psnr_delta=-0.05,
        late_mean_macro_lpips_delta=0.02,
    )
    result = classify_complete_trajectory(receipt, trajectory)
    assert result["classification"] == ALTERNATE
    assert result["target_blind_revision_audit_required"] is False


def test_clear_long_horizon_negative_closes_only_the_current_operator():
    receipt, trajectory = _pair(
        "closed", status=NEGATIVE_STATUS,
        late_three_mean_macro_psnr_delta=-0.3,
        e200_macro_psnr_delta=-0.5,
    )
    result = classify_complete_trajectory(receipt, trajectory)
    assert result["classification"] == CLOSED


def test_frontier_preserves_all_classifications_and_two_wave_cap():
    rows = [
        _pair("strict", status=POSITIVE_STATUS),
        _pair("near", status=NEGATIVE_STATUS, e200_macro_psnr_delta=-0.05),
    ]
    result = classify_frontier(rows)
    assert result["strict_candidate_ids"] == ["strict"]
    assert result["near_boundary_pending_target_blind_audit_ids"] == ["near"]
    assert result["second_wave_maximum_parallel_e200_trajectories"] == 2

