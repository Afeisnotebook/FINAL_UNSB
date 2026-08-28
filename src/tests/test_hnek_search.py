import torch
import pytest

from models.hnek.hnek_search import (
    HnekSearchConfig,
    endpoint_from_residual_gamma,
    hnek_search_installation_status,
    install_hnek_search_model,
    normalized_residual_gamma,
    set_hnek_search_active,
)


def test_gamma_roundtrip():
    x = torch.tensor([[1.0, 2.0], [0.5, -0.5]], requires_grad=True)
    r = torch.tensor([[0.2, -0.3], [0.4, 0.1]], requires_grad=True)
    h = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
    for gamma in (0.25, 0.5, 0.75, 1.0):
        y = endpoint_from_residual_gamma(x, r, h, gamma=gamma)
        r2 = normalized_residual_gamma(x, y, h, gamma=gamma, eps=1e-8)
        assert torch.allclose(r, r2, atol=1e-6, rtol=1e-6)


def test_h1_and_h0_special_cases():
    x = torch.tensor([1.0, 2.0])
    r = torch.tensor([0.5, -0.5])
    y1 = endpoint_from_residual_gamma(x, r, torch.tensor([1.0, 1.0]), gamma=0.5)
    y0 = endpoint_from_residual_gamma(x, r, torch.tensor([0.0, 0.0]), gamma=0.5)
    assert torch.equal(y1, x + r)
    assert torch.equal(y0, x)


def test_config_defaults():
    cfg = HnekSearchConfig()
    assert cfg.gamma == 0.25
    assert cfg.coord == "residual"
    assert cfg.horizon_mode == "physical"
    assert cfg.partial == "all"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0},
        {"coord": "unknown"},
        {"horizon_mode": "unknown"},
        {"partial": "unknown"},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        HnekSearchConfig(**kwargs)


class _DummyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, time_cond, z, layers=None, encode_only=False):
        del time_cond, z, encode_only
        if layers:
            return [x]
        return x + self.scale


class _DummyModel:
    def __init__(self):
        self.netG = _DummyGenerator()
        self.opt = type("Opt", (), {"num_timesteps": 5})()

    def compute_E_loss(self):
        return "plain-e"

    def compute_G_loss(self):
        return "plain-g"


def test_hnek_active_switch_is_idempotent_and_state_preserving():
    model = _DummyModel()
    keys_before = tuple(model.netG.state_dict())
    install_hnek_search_model(model, HnekSearchConfig(partial="endpoint_only"))

    x = torch.zeros(1, 1, 2, 2)
    time = torch.tensor([1])
    z = torch.zeros(1, 1)
    on = model.netG(x, time, z)
    assert hnek_search_installation_status(model)["active"] is True

    set_hnek_search_active(model, False)
    set_hnek_search_active(model, False)
    off = model.netG(x, time, z)
    assert torch.equal(off, torch.ones_like(off))
    assert not torch.equal(on, off)

    set_hnek_search_active(model, True)
    set_hnek_search_active(model, True)
    on_again = model.netG(x, time, z)
    assert torch.equal(on, on_again)
    assert tuple(model.netG.state_dict()) == keys_before
