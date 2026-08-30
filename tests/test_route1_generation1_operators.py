from __future__ import annotations

import torch

from models.route1.bvcp import minimum_velocity_chord_endpoint
from models.route1.rsmg import average_replica_gradients


def test_bvcp_is_exact_identity_when_current_is_no_faster():
    x = torch.zeros(1, 1, 1, 2)
    current = torch.tensor([[[[1.0, 0.0]]]])
    lagged = torch.tensor([[[[2.0, 0.0]]]])
    projected, diag = minimum_velocity_chord_endpoint(x, current, lagged)
    assert torch.equal(projected, current)
    assert diag.eligible == 0
    assert diag.intervened == 0
    assert diag.mean_lambda == 0.0


def test_bvcp_returns_minimum_feasible_chord_root():
    x = torch.zeros(1, 1, 1, 2)
    current = torch.tensor([[[[2.0, 0.0]]]])
    lagged = torch.tensor([[[[1.0, 0.0]]]])
    projected, diag = minimum_velocity_chord_endpoint(x, current, lagged)
    assert torch.allclose(projected, lagged, atol=1e-7, rtol=0.0)
    assert diag.eligible == 1
    assert diag.intervened == 1
    assert abs(diag.mean_lambda - 1.0) < 1e-7
    assert diag.projected_rms <= diag.lagged_rms + 1e-7


def test_bvcp_can_use_an_interior_chord_point():
    x = torch.zeros(1, 1, 1, 2)
    current = torch.tensor([[[[2.0, 0.0]]]])
    lagged = torch.tensor([[[[0.0, 1.0]]]])
    projected, diag = minimum_velocity_chord_endpoint(x, current, lagged)
    assert 0.0 < diag.mean_lambda < 1.0
    assert diag.projected_rms <= diag.lagged_rms + 1e-7
    before = current + (diag.mean_lambda - 1e-4) * (lagged - current)
    before_rms = before.double().square().mean().sqrt().item()
    assert before_rms > diag.lagged_rms


def test_rsmg_replica_gradient_average_is_coordinatewise_mean():
    first = (torch.tensor([1.0, 3.0]), torch.tensor([-2.0]))
    second = (torch.tensor([3.0, 1.0]), torch.tensor([2.0]))
    result = average_replica_gradients([first, second])
    assert torch.equal(result[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(result[1], torch.tensor([0.0]))


def test_rsmg_iid_average_halves_empirical_variance():
    generator = torch.Generator().manual_seed(2026)
    values = torch.randn(200000, generator=generator)
    single = values[:100000]
    replicated = 0.5 * (values[:100000] + values[100000:])
    ratio = float(replicated.var(unbiased=True) / single.var(unbiased=True))
    assert 0.48 < ratio < 0.52

