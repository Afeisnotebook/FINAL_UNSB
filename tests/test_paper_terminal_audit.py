import math
from types import SimpleNamespace

import torch

from production.metrics import bridge_times
from research.local_route1.runtime import capture_rng, full_state_hash, model_state
from research.paper_aio.terminal_audit import (
    _differentiable_rollout_from_state,
    _gram_spectrum,
    _local_jvp_gain,
    _rollout_jvp_gain,
    gradient_stratum_statistics,
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


class _SerializableScalarStream:
    def __init__(self):
        self.index = 0

    def next(self):
        self.index += 1
        return torch.tensor([[float(self.index)]])

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = int(state["index"])


class _GradientAuditModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.opt = SimpleNamespace(num_timesteps=2, lambda_SB=1.0)
        self.netG = torch.nn.Linear(1, 1, bias=False)
        self.netF = torch.nn.Linear(1, 1, bias=False)
        self.netD = torch.nn.Linear(1, 1, bias=False)
        self.netE = torch.nn.Linear(1, 1, bias=False)
        self.model_names = ["G", "F", "D", "E"]
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1e-3)
        self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=1e-3)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=1e-3)
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=1e-3)
        self.optimizers = [
            self.optimizer_G,
            self.optimizer_F,
            self.optimizer_D,
            self.optimizer_E,
        ]
        self.schedulers = []
        self.audit_log = []

    def _sample_training_time_idx(self, total):
        return torch.randint(int(total), size=[1])

    def set_input(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def forward(self):
        self.time_idx = self._sample_training_time_idx(self.opt.num_timesteps)
        self.random_value = torch.rand(())
        self.audit_log.append({
            "time": int(self.time_idx.item()),
            "sample": float(self.primary.item()),
            "random": float(self.random_value.item()),
            "training": bool(self.netG.training),
        })

    def compute_G_loss(self):
        value = self.netG(
            self.primary
            + self.secondary
            + self.random_value
            + self.time_idx.float().reshape(1, 1)
        )
        feature = self.netF(value)
        self.loss_G_GAN = feature.square().mean()
        self.loss_SB = value.square().mean()
        self.loss_NCE = feature.mean()
        return self.loss_G_GAN + self.loss_SB + self.loss_NCE

    @staticmethod
    def set_requires_grad(network, value):
        for parameter in network.parameters():
            parameter.requires_grad_(bool(value))

    def get_extra_training_state(self):
        return {}

    def load_extra_training_state(self, state):
        assert state == {}


def test_gradient_strata_use_cross_time_crn_training_mode_and_restore_state():
    torch.manual_seed(991)
    model = _GradientAuditModel()
    primary = _SerializableScalarStream()
    secondary = _SerializableScalarStream()
    model.netG.eval()
    model.netF.train()
    model.netD.eval()
    model.netE.train()
    next(model.netE.parameters()).requires_grad_(False)
    modes_before = {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    flags_before = [
        parameter.requires_grad
        for name in model.model_names
        for parameter in getattr(model, "net" + name).parameters()
    ]
    state_before = full_state_hash(model_state(model))
    rng_before = full_state_hash(capture_rng())

    result = gradient_stratum_statistics(
        model,
        primary,
        secondary,
        replicates=2,
    )

    assert result["cross_time_common_sampler_state"] is True
    assert result["cross_time_common_rng_state"] is True
    assert result["forward_mode"] == "training_for_every_replicate"
    assert all(row["training"] for row in model.audit_log)
    by_time = {
        time: [row for row in model.audit_log if row["time"] == time]
        for time in range(2)
    }
    assert [row["sample"] for row in by_time[0]] == [
        row["sample"] for row in by_time[1]
    ]
    assert [row["random"] for row in by_time[0]] == [
        row["random"] for row in by_time[1]
    ]
    assert all(
        row["gradient_variance_normalization"]
        == "unbiased_sample_covariance_trace_n_minus_1"
        for row in result["strata"]
    )
    assert primary.state_dict() == {"index": 0}
    assert secondary.state_dict() == {"index": 0}
    assert "_sample_training_time_idx" not in model.__dict__
    assert {
        name: getattr(model, "net" + name).training for name in model.model_names
    } == modes_before
    assert [
        parameter.requires_grad
        for name in model.model_names
        for parameter in getattr(model, "net" + name).parameters()
    ] == flags_before
    assert full_state_hash(model_state(model)) == state_before
    assert full_state_hash(capture_rng()) == rng_before


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
