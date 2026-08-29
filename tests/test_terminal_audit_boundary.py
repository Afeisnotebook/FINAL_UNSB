from types import SimpleNamespace

import pytest
import torch

from research.local_route1.causal_audit import (
    _audit_regimes,
    _restore_terminal_base_lrs,
)


def test_terminal_regimes_do_not_cross_the_training_boundary():
    registered = _audit_regimes((1, 8, 32, 200), start_step=29_999)
    terminal = _audit_regimes((1, 8, 32, 200), start_step=30_000)
    assert any(horizon == 200 for _regime, horizon, _pulse in registered)
    assert {horizon for _regime, horizon, _pulse in terminal} == {1, 8, 32}


def test_terminal_lr_restore_changes_only_optimizer_step_scale():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.0)
    optimizer.state[parameter]["exp_avg"] = torch.tensor([0.25])
    optimizer.state[parameter]["exp_avg_sq"] = torch.tensor([0.5])
    scheduler = SimpleNamespace(base_lrs=[0.0001], last_epoch=200)
    model = SimpleNamespace(optimizers=[optimizer], schedulers=[scheduler])
    moments = {
        key: value.clone() for key, value in optimizer.state[parameter].items()
    }
    assert _restore_terminal_base_lrs(model) == (0.0001,)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert scheduler.last_epoch == 200
    for key, value in moments.items():
        assert torch.equal(optimizer.state[parameter][key], value)
