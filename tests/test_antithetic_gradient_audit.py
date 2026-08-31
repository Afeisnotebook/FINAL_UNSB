from __future__ import annotations

import torch

from research.local_route1.antithetic_gradient_audit import (
    _GradientTraceAccumulator,
    negated_gaussian_draws,
    summarize_gradient_pairs,
)
from research.local_route1.runtime import capture_rng, full_state_hash, restore_rng


def _rng_hash():
    return full_state_hash({"rng": capture_rng()})


def test_negated_gaussian_involution_preserves_rng_consumption():
    torch.manual_seed(2026)
    before = capture_rng()
    normal = (torch.randn(8), torch.randn_like(torch.empty(4)))
    after_normal = _rng_hash()
    restore_rng(before)
    with negated_gaussian_draws():
        opposite = (torch.randn(8), torch.randn_like(torch.empty(4)))
    assert after_normal == _rng_hash()
    assert torch.equal(opposite[0], -normal[0])
    assert torch.equal(opposite[1], -normal[1])


def test_synthetic_antithetic_pairs_report_lower_variance_than_iid():
    native = _GradientTraceAccumulator()
    anti_marginal = _GradientTraceAccumulator()
    iid_marginal = _GradientTraceAccumulator()
    anti_mean = _GradientTraceAccumulator()
    iid_mean = _GradientTraceAccumulator()
    anti_cross = iid_cross = 0.0
    values = (-3.0, -1.0, 1.0, 3.0)
    iid_values = (1.0, 3.0, -3.0, -1.0)
    for value, iid in zip(values, iid_values):
        first = (torch.tensor([2.0 + value]),)
        opposite = (torch.tensor([2.0 - value]),)
        independent = (torch.tensor([2.0 + iid]),)
        native.add(first)
        anti_marginal.add(opposite)
        iid_marginal.add(independent)
        anti_mean.add((0.5 * (first[0] + opposite[0]),))
        iid_mean.add((0.5 * (first[0] + independent[0]),))
        anti_cross += float((first[0] * opposite[0]).sum())
        iid_cross += float((first[0] * independent[0]).sum())
    summary = summarize_gradient_pairs(
        native,
        anti_marginal,
        iid_marginal,
        anti_mean,
        iid_mean,
        native_antithetic_cross_dot_sum=anti_cross,
        native_independent_cross_dot_sum=iid_cross,
    )
    assert summary["antithetic_pair_mean_trace_variance"] == 0.0
    assert summary["independent_pair_mean_trace_variance"] > 0.0
    assert summary["native_antithetic_trace_covariance"] < 0.0
    assert summary["antithetic_to_independent_variance_ratio"] == 0.0

