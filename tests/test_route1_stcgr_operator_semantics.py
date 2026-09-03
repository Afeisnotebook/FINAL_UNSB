from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.route1.pcrsmg_ablation import PCRSMGAblationMixin, PROPOSAL_SCHEDULE
from models.route1.stratified_time import StratifiedTimeConditionalGFMixin


class _ScalarModule(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, lr: float = 1.0) -> None:
        super().__init__(parameters, lr=lr)
        self.committed_steps = 0

    def step(self, closure=None):
        self.committed_steps += 1
        return super().step(closure)


class _NativeScalarGame:
    """Small differentiable game exposing the real Proposal/ST-CGR hooks."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.device = torch.device("cpu")
        self.opt = SimpleNamespace(
            route1_stcgr_enable=bool(enabled),
            num_timesteps=5,
            netF="mlp_sample",
        )
        self.netG = _ScalarModule(0.0)
        self.netF = _ScalarModule(0.0)
        self.netD = _ScalarModule(2.0)
        self.netE = _ScalarModule(-3.0)
        self.optimizer_G = _CountingSGD(self.netG.parameters())
        self.optimizer_F = _CountingSGD(self.netF.parameters())
        self.optimizer_D = _CountingSGD(self.netD.parameters())
        self.optimizer_E = _CountingSGD(self.netE.parameters())
        self.forward_events: list[dict[str, float | int]] = []
        self.g_targets: list[float] = []
        self.d_time: int | None = None
        self.e_time: int | None = None
        self.native_optimize_calls = 0

    def _sample_training_time_idx(self, size: int) -> torch.Tensor:
        return torch.randint(int(size), size=[1]).long()

    def forward(self) -> None:
        self.time_idx = self._sample_training_time_idx(self.opt.num_timesteps)
        torch_noise = torch.rand(())
        numpy_noise = float(np.random.random())
        target = self.time_idx.float().reshape(()) + torch_noise + numpy_noise
        self.real_A_noisy = target.reshape(1)
        self.real_A_noisy2 = (target + 0.25).reshape(1)
        self.fake_B = self.netG.weight.reshape(1)
        self.fake_B2 = self.netG.weight.reshape(1)
        self.fake = self.fake_B
        self.fake_B = self.fake_B
        self.forward_events.append({
            "time": int(self.time_idx.item()),
            "torch_noise": float(torch_noise.item()),
            "numpy_noise": numpy_noise,
            "d": float(self.netD.weight.detach().item()),
            "e": float(self.netE.weight.detach().item()),
        })

    def compute_D_loss(self) -> torch.Tensor:
        self.d_time = int(self.time_idx.item())
        self.loss_D_real = 0.5 * self.netD.weight.square()
        self.loss_D_fake = torch.zeros_like(self.loss_D_real)
        self.loss_D = self.loss_D_real
        return self.loss_D

    def compute_E_loss(self) -> torch.Tensor:
        self.e_time = int(self.time_idx.item())
        self.loss_E = 0.5 * self.netE.weight.square()
        return self.loss_E

    def compute_G_loss(self) -> torch.Tensor:
        target = self.real_A_noisy.detach().reshape(())
        self.g_targets.append(float(target.item()))
        g_loss = 0.5 * (self.netG.weight - target).square()
        f_loss = 0.5 * (self.netF.weight - (target + 1.0)).square()
        self.loss_G_GAN = g_loss
        self.loss_SB = torch.zeros_like(g_loss)
        self.loss_NCE = f_loss
        self.loss_G = g_loss + f_loss
        return self.loss_G

    @staticmethod
    def set_requires_grad(nets, requires_grad: bool = False) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for parameter in net.parameters():
                parameter.requires_grad = bool(requires_grad)

    @staticmethod
    def _before_generator_optimizer_step() -> None:
        return None

    def get_extra_training_state(self) -> dict:
        return {"native": True}

    def load_extra_training_state(self, state: dict) -> None:
        assert state["native"] is True

    def optimize_parameters(self) -> None:
        # Only the disabled-dispatch test reaches this native marker.  The
        # production zero-intervention identity is additionally covered by
        # the 2000-update paper runtime gate.
        self.native_optimize_calls += 1
        self.forward()


class _STCGROperator(
    StratifiedTimeConditionalGFMixin,
    PCRSMGAblationMixin,
    _NativeScalarGame,
):
    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._initialize_pcrsmg_ablation_state()
        self._initialize_stcgr_state()


def test_full_stcgr_operator_respects_player_boundary_and_mean_commit() -> None:
    torch.manual_seed(2026)
    np.random.seed(2026)
    game = _STCGROperator()
    game.optimize_parameters()

    # One native D/E view is followed by two fresh post-D/E G/F views.
    assert len(game.forward_events) == 3
    native, first_gf, second_gf = game.forward_events
    assert game.d_time == native["time"]
    assert game.e_time == native["time"]
    assert first_gf["d"] == pytest.approx(0.0)
    assert first_gf["e"] == pytest.approx(0.0)
    assert second_gf["d"] == pytest.approx(0.0)
    assert second_gf["e"] == pytest.approx(0.0)

    # The time coupling alone is without replacement.  Torch and NumPy draws
    # used as stand-ins for rollout/latent and PatchNCE randomness are fresh.
    assert first_gf["time"] != second_gf["time"]
    assert first_gf["torch_noise"] != second_gf["torch_noise"]
    assert first_gf["numpy_noise"] != second_gf["numpy_noise"]

    # Both joint G/F gradients are accumulated with weight 1/2 and each Adam
    # analogue commits once.  With lr=1 and a quadratic at zero, the resulting
    # parameters equal the arithmetic mean of the two per-view targets.
    expected_g = sum(game.g_targets) / 2.0
    assert float(game.netG.weight.detach()) == pytest.approx(expected_g)
    assert float(game.netF.weight.detach()) == pytest.approx(expected_g + 1.0)
    assert game.optimizer_D.committed_steps == 1
    assert game.optimizer_E.committed_steps == 1
    assert game.optimizer_G.committed_steps == 1
    assert game.optimizer_F.committed_steps == 1

    state = game.get_extra_training_state()
    assert state["pcrsmg_proposal"]["last_schedule"] == list(PROPOSAL_SCHEDULE)
    assert state["pcrsmg_proposal"]["update_index"] == 1
    assert state["pcrsmg_proposal"]["gf_bundle_count"] == 1
    assert state["stcgr"]["bundle_count"] == 1
    assert state["stcgr"]["last_pair"] == [first_gf["time"], second_gf["time"]]


def test_stcgr_disabled_dispatches_the_native_optimizer_without_method_state() -> None:
    torch.manual_seed(101)
    np.random.seed(101)
    game = _STCGROperator(enabled=False)
    game.optimize_parameters()

    assert game.native_optimize_calls == 1
    assert len(game.forward_events) == 1
    state = game.get_extra_training_state()
    assert state == {"native": True}
    assert "pcrsmg_proposal" not in state
    assert "stcgr" not in state
