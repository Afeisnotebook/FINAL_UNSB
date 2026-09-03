import math

import torch

from production.metrics import bridge_times
from research.paper_aio.terminal_audit import (
    _differentiable_rollout_from_state,
    _gram_spectrum,
    _local_jvp_gain,
    _rollout_jvp_gain,
    perturbation_gain_to_final,
)


def test_gram_spectrum_is_unbiased_sample_covariance_not_feature_normalized():
    result = _gram_spectrum([
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([4.0]),
    ])
    assert math.isclose(result["top_eigenvalue"], 4.0, rel_tol=1e-12)
    assert math.isclose(result["trace"], 4.0, rel_tol=1e-12)
    assert math.isclose(result["effective_rank"], 1.0, rel_tol=1e-12)
    assert result["normalization"] == (
        "unbiased_sample_covariance_nonzero_spectrum_n_minus_1"
    )
    assert result["effective_rank_definition"] == (
        "participation_ratio_trace_squared_over_frobenius_squared"
    )
    assert result["sample_count"] == 3
    assert result["flattened_dimension"] == 1


class _ScaledEndpoint(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, state, time_index, latent):
        del time_index, latent
        return self.scale * state


def _bundle(shape=(1, 1, 2, 2)):
    return {
        "z": [torch.zeros((1, 1)) for _ in range(5)],
        "noise": [torch.zeros(shape) for _ in range(5)],
    }


def test_local_jvp_uses_supplied_direction_and_recovers_isotropic_scale():
    model = _ScaledEndpoint(1.75)
    state = torch.ones((1, 1, 2, 2))
    direction = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    value = _local_jvp_gain(
        model,
        state,
        3,
        torch.zeros((1, 1)),
        direction,
        iterations=2,
    )
    assert math.isclose(value, 1.75, rel_tol=1e-6)


def test_rollout_jvp_includes_endpoint_mediated_future_state_transition():
    scale = 2.0
    model = _ScaledEndpoint(scale)
    state = torch.ones((1, 1, 2, 2))
    direction = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    bundle = _bundle(state.shape)
    times = bridge_times(5)
    step = 3
    alpha = float(times[step + 1] - times[step]) / float(times[-1] - times[step])
    expected = scale * ((1.0 - alpha) + alpha * scale)
    value = _rollout_jvp_gain(
        model,
        state,
        bundle,
        start_step=step,
        tau=0.01,
        initial_direction=direction,
        iterations=2,
    )
    assert math.isclose(value, expected, rel_tol=1e-6)


def test_rollout_jvp_matches_lane_blind_finite_difference_probe():
    model = _ScaledEndpoint(1.5)
    state = torch.ones((1, 1, 2, 2))
    direction = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
    bundle = _bundle(state.shape)
    analytical = _rollout_jvp_gain(
        model,
        state,
        bundle,
        start_step=2,
        tau=0.01,
        initial_direction=direction,
        iterations=2,
    )
    finite = perturbation_gain_to_final(
        model,
        state,
        bundle,
        start_step=2,
        tau=0.01,
        epsilon=1e-3,
        direction=direction,
    )
    assert math.isclose(analytical, finite, rel_tol=5e-4)


def test_differentiable_rollout_matches_numerical_values():
    model = _ScaledEndpoint(1.25)
    state = torch.ones((1, 1, 2, 2))
    bundle = _bundle(state.shape)
    differentiable = _differentiable_rollout_from_state(
        model,
        state,
        bundle,
        start_step=1,
        tau=0.01,
    )
    with torch.no_grad():
        numerical = state
        endpoint = None
        times = bridge_times(5)
        for step in range(1, 5):
            if step > 1:
                alpha = float(times[step] - times[step - 1]) / float(
                    times[-1] - times[step - 1]
                )
                numerical = (1.0 - alpha) * numerical + alpha * endpoint.detach()
            endpoint = model(numerical, torch.tensor([step]), bundle["z"][step])
    assert torch.equal(differentiable, endpoint)
